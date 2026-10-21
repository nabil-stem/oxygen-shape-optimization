"""Controlled deletions and area-balanced twins, each evaluated by global flux."""

from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from oxygen_shape.experiments.deletion_config import DeletionConfig
from oxygen_shape.experiments.deletion_report import plot_effects, write_report
from oxygen_shape.experiments.reporting import (
    create_run,
    log_failure,
    provenance,
    save_state,
    save_summary,
    write_csv,
)
from oxygen_shape.geometry.baselines import matched_area
from oxygen_shape.geometry.constraints import (
    ShapeConstraints,
    feature_violations,
    validate_shape,
)
from oxygen_shape.geometry.disk_edits import apply_disk, propose_paired_disk
from oxygen_shape.geometry.raster import RasterGeometry, refine_geometry
from oxygen_shape.optimization.donor_sampling import eligible_wall_flux, low_flux_weights
from oxygen_shape.physics.model import DiffusionModel
from oxygen_shape.solvers.finite_volume import SolveResult, solve
from oxygen_shape.viz.fields import plot_fields


def candidate_violations(
    geometry: RasterGeometry,
    initial: RasterGeometry,
    config: DeletionConfig,
    *,
    deletion: bool,
) -> tuple[str, ...]:
    """Area loss is explicit for deletion; all other constraints are shared."""
    removed = initial.area - geometry.area
    if deletion and not 0 < removed <= config.max_removed_area_fraction * initial.area + 1e-12:
        return ("Deletion area loss outside the declared bound",)
    constraints = ShapeConstraints(
        config.target_area,
        config.perimeter_limit,
        config.minimum_feature_width,
        config.protected_length,
    )
    if deletion:
        constraints = replace(constraints, target_area=geometry.area)
    reasons = validate_shape(geometry, constraints, initial).violations
    if reasons:
        return reasons
    for factor in config.refinement_factors:
        failures = feature_violations(
            refine_geometry(geometry, factor), config.minimum_feature_width
        )
        if failures:
            return (f"Feature check at factor {factor}: " + "; ".join(failures),)
    return ()


def effect_row(
    geometry: RasterGeometry,
    result: SolveResult,
    reference: SolveResult,
) -> dict:
    m = result.metrics
    return {
        "physical_flux_Q": m.physical_flux_Q,
        "initial_Q": reference.metrics.physical_flux_Q,
        "delta_Q": m.physical_flux_Q - reference.metrics.physical_flux_Q,
        "ideal_flux_Q": m.ideal_flux_Q,
        "efficiency_eta": m.efficiency_eta,
        "initial_eta": reference.metrics.efficiency_eta,
        "area": geometry.area,
        "robin_perimeter": m.robin_perimeter,
        "conservation_error": m.conservation_error,
        "optimization_score": None,
        "constraint_violations": "",
    }


def run_deletion_study(config: DeletionConfig, progress=None) -> tuple[Path, dict]:
    directory = create_run(config, "deletion_hypothesis")
    started = perf_counter()
    rows, attempts, quotas, refinements = [], [], [], []
    (directory / "metrics.csv").write_text("case,group,mode,delta_Q\n", encoding="utf-8")
    (directory / "refinement.csv").write_text("case,sample_id,factor,delta_Q\n", encoding="utf-8")
    solves = 0
    maximum_conservation = 0.0
    try:
        initial_geometries = {}
        # Reject an inadmissible experiment matrix before spending any PDE solves.
        for shape in config.shapes:
            initial = matched_area(shape, config.n, config.target_area).geometry
            constraints = ShapeConstraints(
                config.target_area,
                config.perimeter_limit,
                config.minimum_feature_width,
                config.protected_length,
            )
            validate_shape(initial, constraints, initial).require_valid()
            for factor in config.refinement_factors:
                failures = feature_violations(
                    refine_geometry(initial, factor), config.minimum_feature_width
                )
                if failures:
                    raise ValueError(f"Initial {shape} fails feature check at {factor}: {failures}")
            initial_geometries[shape] = initial
        cases = [(shape, Bi) for shape in config.shapes for Bi in config.permeabilities]
        for case_index, (shape, Bi) in enumerate(cases):
            case_id = f"{shape}_Bi_{Bi:g}"
            path = directory / "cases" / case_id
            path.mkdir(parents=True)
            initial = initial_geometries[shape]
            model = DiffusionModel(Bi=Bi)
            reference = solve(initial, model, config.solver)
            solves += 1
            maximum_conservation = max(maximum_conservation, reference.metrics.conservation_error)
            save_state(path, initial, reference)
            if case_index == 0:
                save_state(directory, initial, reference)
                plot_fields(initial, reference, directory / "figures" / "baseline_fields.png")
            flux = eligible_wall_flux(initial, reference, config.protected_length)
            low_weights = low_flux_weights(flux, config.low_flux_fraction)
            case_rows = []
            representative_fields = {}
            for group_index, group in enumerate(("uniform", "low_flux")):
                seed = config.seed + 2 * case_index + group_index
                rng = np.random.default_rng(seed)
                seen: set[bytes] = set()
                collected = 0
                weights = low_weights if group == "low_flux" else None
                for attempt in range(1, config.max_attempts_per_group + 1):
                    proposal = propose_paired_disk(
                        initial,
                        rng,
                        float(rng.uniform(*config.radii)),
                        config.protected_length,
                        donor_weights=weights,
                    )
                    p = proposal.parameters
                    rejection = proposal.rejection
                    deletion = None
                    if proposal.geometry is not None:
                        deletion = apply_disk(
                            initial,
                            np.array([p["remove_x"], p["remove_y"]]),
                            p["remove_radius"],
                            add=False,
                        )
                        rejection = "; ".join(
                            candidate_violations(proposal.geometry, initial, config, deletion=False)
                            + candidate_violations(deletion, initial, config, deletion=True)
                        )
                        signature = deletion.inside.tobytes()
                        if not rejection and signature in seen:
                            rejection = "Duplicate donor cut"
                    attempts.append(
                        {
                            "case": case_id,
                            "group": group,
                            "seed": seed,
                            "attempt": attempt,
                            "rejection": rejection,
                            **p,
                        }
                    )
                    if rejection or proposal.geometry is None:
                        continue
                    seen.add(signature)
                    collected += 1
                    for mode, geometry in (("deletion", deletion), ("paired", proposal.geometry)):
                        sample_id = f"{group}_{collected:03d}_{mode}"
                        output = path / sample_id
                        output.mkdir()
                        solution = solve(geometry, model, config.solver)
                        solves += 1
                        maximum_conservation = max(
                            maximum_conservation, solution.metrics.conservation_error
                        )
                        save_state(output, geometry, solution)
                        np.save(output / "initial_geometry.npy", initial.inside)
                        row = {
                            "case": case_id,
                            "shape": shape,
                            "Bi": Bi,
                            "group": group,
                            "mode": mode,
                            "seed": seed,
                            "sample_id": sample_id,
                            "attempt": attempt,
                            "donor_flux_density": float(flux[p["donor_index"]]),
                            "low_flux_threshold": float(
                                np.quantile(flux, config.low_flux_fraction)
                            ),
                            "removed_area": int(p["changed_cells"]) * initial.h**2,
                            **p,
                            **effect_row(geometry, solution, reference),
                        }
                        rows.append(row)
                        case_rows.append(row)
                        if case_index == 0 and group == "low_flux" and mode == "deletion":
                            representative_fields[sample_id] = (geometry, solution)
                    if collected == config.samples_per_group:
                        break
                quotas.append(
                    {
                        "case": case_id,
                        "group": group,
                        "seed": seed,
                        "collected_pairs": collected,
                        "requested_pairs": config.samples_per_group,
                        "attempts": attempt,
                        "eligible_donor_faces": len(flux),
                        "low_flux_donor_faces": int(np.count_nonzero(low_weights)),
                    }
                )
                if progress:
                    progress(
                        f"{case_id}, {group}: {collected}/{config.samples_per_group} pairs, "
                        f"{attempt} attempts"
                    )
                if rows:
                    write_csv(directory / "metrics.csv", rows)
                write_csv(directory / "proposals.csv", attempts)
                write_csv(directory / "quotas.csv", quotas)

            # Inspect the most positive and negative coarse effects in each mode/group.
            # Selection is descriptive; these examples are not an independent test set.
            selected = {}
            for group in ("uniform", "low_flux"):
                for mode in ("deletion", "paired"):
                    subset = [r for r in case_rows if r["group"] == group and r["mode"] == mode]
                    for sign in (-1, 1):
                        signed = [
                            r for r in subset if sign * r["delta_Q"] > config.effect_tolerance
                        ]
                        if signed:
                            best = max(signed, key=lambda r: sign * r["delta_Q"])
                            selected[best["sample_id"]] = best
            for sample_id in selected:
                if sample_id in representative_fields:
                    geometry, solution = representative_fields[sample_id]
                    plot_fields(
                        geometry, solution, directory / "figures" / f"{sample_id}_fields.png"
                    )
            for factor in config.refinement_factors if selected else []:
                refined_initial = refine_geometry(initial, factor)
                fine_reference = solve(refined_initial, model, config.solver)
                solves += 1
                maximum_conservation = max(
                    maximum_conservation, fine_reference.metrics.conservation_error
                )
                for sample_id, row in selected.items():
                    mask = np.load(path / sample_id / "final_geometry.npy", allow_pickle=False)
                    candidate = refine_geometry(initial.with_mask(mask), factor)
                    fine_solution = solve(candidate, model, config.solver)
                    solves += 1
                    maximum_conservation = max(
                        maximum_conservation, fine_solution.metrics.conservation_error
                    )
                    refinements.append(
                        {
                            "case": case_id,
                            "sample_id": sample_id,
                            "group": row["group"],
                            "mode": row["mode"],
                            "factor": factor,
                            "grid_n": config.n * factor,
                            **effect_row(candidate, fine_solution, fine_reference),
                        }
                    )
            for sample_id, row in selected.items():
                fine = [
                    r for r in refinements if r["case"] == case_id and r["sample_id"] == sample_id
                ]
                last_delta = fine[-2]["delta_Q"] if len(fine) > 1 else row["delta_Q"]
                sensitivity = abs(fine[-1]["delta_Q"] - last_delta)
                row.update(
                    fine_delta_Q=fine[-1]["delta_Q"],
                    observed_grid_sensitivity=sensitivity,
                    effect_resolved=(
                        row["delta_Q"] * fine[-1]["delta_Q"] > 0
                        and abs(fine[-1]["delta_Q"]) > sensitivity + config.effect_tolerance
                    ),
                )
            if progress:
                progress(f"{case_id}: PDE solves so far={solves}; refinement finished")
            if rows:
                write_csv(directory / "metrics.csv", rows)
            if refinements:
                write_csv(directory / "refinement.csv", refinements)
        summary = {
            **provenance(),
            "seed": config.seed,
            "grid_shape": [config.n, config.n],
            "permeabilities": config.permeabilities,
            "solver": "finite_volume",
            "solver_tolerance": None,
            "diagnostic_tolerances": config.solver.model_dump(),
            "status": "complete"
            if all(q["collected_pairs"] == q["requested_pairs"] for q in quotas)
            else "incomplete_quotas",
            "number_of_pde_solves": solves,
            "proposal_attempts": len(attempts),
            "evaluated_candidates": len(rows),
            "refined_candidates": sum("fine_delta_Q" in r for r in rows),
            "max_conservation_error": maximum_conservation,
            "wall_clock_seconds": perf_counter() - started,
            "root_geometry": "Unedited first-case baseline; candidates are under cases/",
            "low_flux_deletion_positive": sum(
                r["group"] == "low_flux"
                and r["mode"] == "deletion"
                and r["delta_Q"] > config.effect_tolerance
                for r in rows
            ),
            "low_flux_deletion_negative": sum(
                r["group"] == "low_flux"
                and r["mode"] == "deletion"
                and r["delta_Q"] < -config.effect_tolerance
                for r in rows
            ),
            "resolved_low_flux_deletion_positive": sum(
                r["group"] == "low_flux"
                and r["mode"] == "deletion"
                and r.get("effect_resolved", False)
                and r["fine_delta_Q"] > 0
                for r in rows
            ),
            "resolved_low_flux_deletion_negative": sum(
                r["group"] == "low_flux"
                and r["mode"] == "deletion"
                and r.get("effect_resolved", False)
                and r["fine_delta_Q"] < 0
                for r in rows
            ),
        }
        plot_effects(rows, directory / "figures" / "deletion_effects.png")
        write_report(directory, config, rows, quotas, summary)
        save_summary(directory, summary)
        return directory, summary
    except Exception as exc:
        log_failure(directory, exc)
        raise
