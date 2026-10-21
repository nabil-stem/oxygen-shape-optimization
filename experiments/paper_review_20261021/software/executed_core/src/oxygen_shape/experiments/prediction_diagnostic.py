"""Disjoint fixed-geometry probes, held-out ranking and radius replay."""

import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np

from oxygen_shape.experiments.frozen_geometry import load_fixed_cases
from oxygen_shape.experiments.prediction_config import PredictionConfig
from oxygen_shape.experiments.prediction_report import plot_diagnostic, write_report
from oxygen_shape.experiments.reporting import (
    create_run,
    log_failure,
    provenance,
    save_state,
    save_summary,
    write_csv,
)
from oxygen_shape.experiments.study_worker import atomic_json
from oxygen_shape.geometry.constraints import raster_perimeter
from oxygen_shape.geometry.disk_edits import prepare_boundary, propose_paired_disk
from oxygen_shape.geometry.raster import refine_geometry
from oxygen_shape.optimization.donor_sampling import eligible_wall_flux
from oxygen_shape.optimization.guided_search import metrics_row, validate_candidate
from oxygen_shape.optimization.prediction import predict_scores, ranking_metrics
from oxygen_shape.solvers.finite_volume import solve
from oxygen_shape.viz.fields import plot_fields


def score_holdout(
    config: PredictionConfig, calibration: list[dict], heldout: list[dict]
) -> tuple[list[dict], list[dict]]:
    metrics, predictions = [], []
    if not heldout:
        return metrics, predictions

    def features(rows):
        return np.asarray(
            [
                [
                    r[k]
                    for k in (
                        "addition_anchor_x",
                        "addition_anchor_y",
                        "donor_anchor_x",
                        "donor_anchor_y",
                        "radius",
                    )
                ]
                for r in rows
            ]
        )

    query = features(heldout)
    actual = np.asarray([r["delta_Q"] for r in heldout])
    area = np.asarray([r["reallocated_area"] for r in heldout])
    midpoint = sum(config.radii) / 2
    for count in config.calibration_sizes:
        if len(calibration) < count:
            continue
        training = calibration[:count]
        scores = predict_scores(
            features(training),
            np.asarray([r["delta_Q"] for r in training]),
            np.asarray([r["reallocated_area"] for r in training]),
            query,
            config.spatial_bandwidth,
            config.radius_bandwidth,
        )
        scores["flux_control"] = np.asarray([r["flux_score"] for r in heldout])
        for index, row in enumerate(heldout):
            predictions.append(
                {
                    "shape": row["shape"],
                    "family": row["family"],
                    "reference_state": row["reference_state"],
                    "replicate": row["replicate"],
                    "calibration_count": count,
                    "sample_id": row["sample_id"],
                    "actual_delta_Q": row["delta_Q"],
                    "actual_sensitivity": row["sensitivity"],
                    "radius": row["radius"],
                    **{name: float(values[index]) for name, values in scores.items()},
                }
            )
        for name, values in scores.items():
            for stratum, selection in (
                ("all", np.ones(len(heldout), dtype=bool)),
                ("small", query[:, 4] < midpoint),
                ("large", query[:, 4] >= midpoint),
            ):
                if not selection.any():
                    continue
                metrics.append(
                    {
                        "shape": heldout[0]["shape"],
                        "family": heldout[0]["family"],
                        "reference_state": heldout[0]["reference_state"],
                        "replicate": heldout[0]["replicate"],
                        "calibration_seed": calibration[0]["stream_seed"],
                        "heldout_seed": heldout[0]["stream_seed"],
                        "calibration_count": count,
                        "score": name,
                        "radius_stratum": stratum,
                        **ranking_metrics(
                            values[selection],
                            actual[selection],
                            area[selection],
                            config.top_fraction,
                            config.effect_tolerance,
                        ),
                    }
                )
    return metrics, predictions


def run_prediction_diagnostic(config: PredictionConfig, progress=None) -> tuple[Path, dict]:
    root = create_run(config, "prediction_diagnostic")
    started = perf_counter()
    outcomes, attempts, quotas, scores, predictions, refinements = [], [], [], [], [], []
    solves, refinement_solves, max_error = 0, 0, 0.0
    fixed_metrics = []
    for name, header in (
        ("metrics.csv", "sample_id,physical_flux_Q,delta_Q"),
        ("scores.csv", "shape,score,calibration_count"),
        ("predictions.csv", "sample_id,actual_delta_Q"),
        ("refinement.csv", "sample_id,factor,delta_Q"),
    ):
        (root / name).write_text(header + "\n", encoding="utf-8")

    def flush():
        for name, rows in (
            ("metrics", outcomes),
            ("attempts", attempts),
            ("quotas", quotas),
            ("scores", scores),
            ("predictions", predictions),
            ("refinement", refinements),
        ):
            if rows:
                write_csv(root / f"{name}.csv", rows)

    try:
        fixed_cases = load_fixed_cases(config, root)
        baselines = {}
        for fixed in fixed_cases:
            shape, geometry = fixed.label, fixed.geometry
            baseline = solve(geometry, config.model, config.solver)
            solves += 1
            if fixed.expected_Q is not None and not np.isclose(
                baseline.metrics.physical_flux_Q,
                fixed.expected_Q,
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError(f"Recomputed frozen-source flux disagrees for {shape}")
            max_error = max(max_error, baseline.metrics.conservation_error)
            fixed_metrics.append(
                {
                    "shape": shape,
                    "family": fixed.family,
                    "reference_state": fixed.state,
                    "Q": baseline.metrics.physical_flux_Q,
                    "area": geometry.area,
                    "robin_perimeter": raster_perimeter(geometry),
                    "perimeter_fraction": raster_perimeter(geometry) / config.perimeter_limit,
                }
            )
            write_csv(root / "fixed_cases.csv", fixed_metrics)
            baseline_path = root / "cases" / shape / "baseline"
            baseline_path.mkdir(parents=True)
            save_state(baseline_path, geometry, baseline)
            if shape == config.case_labels[0]:
                save_state(root, geometry, baseline)
            baselines[shape] = baseline
        for fixed in fixed_cases:
            shape, geometry = fixed.label, fixed.geometry
            search_config = config.search_config(fixed.family)
            baseline = baselines[shape]
            prepared = prepare_boundary(geometry, config.protected_length)
            points = prepared.points
            if not len(points):
                raise ValueError("No eligible faces for diagnostic")
            flux = eligible_wall_flux(geometry, baseline, config.protected_length)
            shape_heldout = []

            for replicate in range(config.replicates):
                seen: set[str] = set()

                def evaluate(
                    role,
                    seed,
                    attempt,
                    pair,
                    proposal_seed,
                    radius,
                    family_id=None,
                    *,
                    context=(
                        shape,
                        replicate,
                        geometry,
                        search_config,
                        prepared,
                        points,
                        flux,
                        baseline,
                        seen,
                        fixed,
                    ),
                ):
                    nonlocal solves, max_error
                    (
                        shape,
                        replicate,
                        geometry,
                        search_config,
                        prepared,
                        points,
                        flux,
                        baseline,
                        seen,
                        fixed,
                    ) = context
                    proposal = propose_paired_disk(
                        geometry,
                        np.random.default_rng(proposal_seed),
                        radius,
                        config.protected_length,
                        face_pair=pair,
                        prepared=prepared,
                    )
                    rejection = proposal.rejection
                    signature = None
                    if proposal.geometry is not None:
                        rejection = validate_candidate(proposal.geometry, geometry, search_config)
                        signature = hashlib.sha256(proposal.geometry.inside.tobytes()).hexdigest()
                        if not rejection and role != "radius" and signature in seen:
                            rejection = "Duplicate mask across calibration/held-out pools"
                    attempt_row = {
                        "shape": shape,
                        "family": fixed.family,
                        "reference_state": fixed.state,
                        "replicate": replicate,
                        "role": role,
                        "stream_seed": seed,
                        "attempt": attempt,
                        "family_id": family_id,
                        "proposal_seed": proposal_seed,
                        "addition_face_index": pair[0],
                        "donor_face_index": pair[1],
                        "radius": radius,
                        "rejection": rejection,
                        "valid": not rejection and proposal.geometry is not None,
                    }
                    attempts.append(attempt_row)
                    if rejection or proposal.geometry is None:
                        return None
                    candidate = solve(proposal.geometry, config.model, config.solver)
                    solves += 1
                    max_error = max(max_error, candidate.metrics.conservation_error)
                    if role != "radius":
                        seen.add(signature)
                    sample_id = f"{shape}/replicate_{replicate:02d}/{role}_{attempt:05d}"
                    output = root / "cases" / sample_id
                    output.mkdir(parents=True)
                    save_state(output, proposal.geometry, candidate)
                    np.save(output / "initial_geometry.npy", geometry.inside)
                    area = int(proposal.parameters["changed_cells"]) * geometry.h**2
                    delta = candidate.metrics.physical_flux_Q - baseline.metrics.physical_flux_Q
                    row = {
                        **attempt_row,
                        "sample_id": sample_id,
                        "geometry_hash": signature,
                        "initial_Q": baseline.metrics.physical_flux_Q,
                        "delta_Q": delta,
                        "reallocated_area": area,
                        "sensitivity": delta / area,
                        "addition_anchor_x": float(points[pair[0], 0]),
                        "addition_anchor_y": float(points[pair[0], 1]),
                        "donor_anchor_x": float(points[pair[1], 0]),
                        "donor_anchor_y": float(points[pair[1], 1]),
                        "flux_score": float(flux[pair[0]] * (flux.max() - flux[pair[1]])),
                        **proposal.parameters,
                        **metrics_row(
                            proposal.geometry, candidate, baseline.metrics.physical_flux_Q
                        ),
                        "optimization_score": None,
                    }
                    outcomes.append(row)
                    return row

                calibration, heldout = [], []
                for role, offset, requested, pool in (
                    ("calibration", 0, config.calibration_sizes[-1], calibration),
                    ("heldout", 1000, config.heldout_samples, heldout),
                ):
                    seed = config.seed + replicate + offset
                    rng = np.random.default_rng(seed)
                    for attempt in range(1, config.max_attempts_per_pool + 1):
                        pair = divmod(int(rng.integers(len(points) ** 2)), len(points))
                        proposal_seed = int(rng.integers(2**63 - 1))
                        row = evaluate(
                            role,
                            seed,
                            attempt,
                            pair,
                            proposal_seed,
                            float(rng.uniform(*config.radii)),
                        )
                        if row is not None:
                            pool.append(row)
                        if attempt % 1000 == 0:
                            atomic_json(
                                root / "status.json",
                                {
                                    "status": "running",
                                    "shape": shape,
                                    "replicate": replicate,
                                    "role": role,
                                    "attempts_in_pool": attempt,
                                    "collected": len(pool),
                                    "requested": requested,
                                    "number_of_pde_solves": solves,
                                },
                            )
                            if progress:
                                progress(
                                    f"{shape}, replicate={replicate}, {role}: "
                                    f"{len(pool)}/{requested} after {attempt} attempts"
                                )
                        if len(pool) == requested:
                            break
                    quotas.append(
                        {
                            "shape": shape,
                            "replicate": replicate,
                            "role": role,
                            "seed": seed,
                            "requested": requested,
                            "collected": len(pool),
                            "attempts": attempt,
                        }
                    )
                    flush()
                metrics, predicted = score_holdout(config, calibration, heldout)
                scores.extend(metrics)
                predictions.extend(predicted)
                shape_heldout.extend(heldout)
                radius_seed = config.seed + replicate + 2000
                rng = np.random.default_rng(radius_seed)
                radius_attempt = 0
                for family in range(config.radius_families):
                    pair = divmod(int(rng.integers(len(points) ** 2)), len(points))
                    proposal_seed = int(rng.integers(2**63 - 1))
                    for radius in config.matched_radii:
                        radius_attempt += 1
                        evaluate(
                            "radius",
                            radius_seed,
                            radius_attempt,
                            pair,
                            proposal_seed,
                            radius,
                            family,
                        )
                flush()
                atomic_json(
                    root / "status.json",
                    {
                        "status": "running",
                        "shape": shape,
                        "completed_replicate": replicate,
                        "number_of_pde_solves": solves,
                        "valid_edits": len(outcomes),
                    },
                )
                if progress:
                    progress(
                        f"{shape}, replicate={replicate}: calibration={len(calibration)}, "
                        f"heldout={len(heldout)}, cumulative solves={solves}"
                    )

            selected = []
            for sign in (-1, 1):
                signed = [r for r in shape_heldout if sign * r["delta_Q"] > config.effect_tolerance]
                if signed:
                    selected.append(max(signed, key=lambda r: sign * r["delta_Q"]))
            for factor in config.refinement_factors if selected else []:
                fine_initial = refine_geometry(geometry, factor)
                fine_baseline = solve(fine_initial, config.model, config.solver)
                solves += 1
                refinement_solves += 1
                max_error = max(max_error, fine_baseline.metrics.conservation_error)
                for row in selected:
                    mask = np.load(
                        root / "cases" / row["sample_id"] / "final_geometry.npy", allow_pickle=False
                    )
                    fine = refine_geometry(geometry.with_mask(mask), factor)
                    solution = solve(fine, config.model, config.solver)
                    solves += 1
                    refinement_solves += 1
                    max_error = max(max_error, solution.metrics.conservation_error)
                    refinements.append(
                        {
                            "shape": shape,
                            "sample_id": row["sample_id"],
                            "factor": factor,
                            "coarse_delta_Q": row["delta_Q"],
                            "delta_Q": solution.metrics.physical_flux_Q
                            - fine_baseline.metrics.physical_flux_Q,
                            "conservation_error": max(
                                solution.metrics.conservation_error,
                                fine_baseline.metrics.conservation_error,
                            ),
                        }
                    )
                    output = root / "refinement" / f"factor_{factor}" / row["sample_id"]
                    output.mkdir(parents=True)
                    save_state(output, fine, solution)
                    np.save(output / "initial_geometry.npy", fine_initial.inside)
                    if factor == config.refinement_factors[-1]:
                        label = row["sample_id"].replace("/", "_")
                        plot_fields(fine, solution, root / "figures" / f"{label}_fields.png")
            flush()
        complete = all(q["collected"] == q["requested"] for q in quotas)
        summary = {
            **provenance(),
            "status": "complete" if complete else "incomplete_quotas",
            "seed": config.seed,
            "calibration_seeds": list(range(config.seed, config.seed + config.replicates)),
            "heldout_seeds": list(
                range(config.seed + 1000, config.seed + 1000 + config.replicates)
            ),
            "radius_seeds": list(range(config.seed + 2000, config.seed + 2000 + config.replicates)),
            "grid_shape": [config.n, config.n],
            "Bi": config.model.Bi,
            "solver": "finite_volume",
            "solver_tolerance": None,
            "diagnostic_tolerances": config.solver.model_dump(),
            "number_of_pde_solves": solves,
            "baseline_pde_solves": len(fixed_cases),
            "fixed_cases": fixed_metrics,
            "input_manifest": "input_manifest.json",
            "calibration_pde_solves": sum(r["role"] == "calibration" for r in outcomes),
            "heldout_pde_solves": sum(r["role"] == "heldout" for r in outcomes),
            "radius_pde_solves": sum(r["role"] == "radius" for r in outcomes),
            "refinement_pde_solves": refinement_solves,
            "valid_edits": len(outcomes),
            "proposal_attempts": len(attempts),
            "max_conservation_error": max_error,
            "wall_clock_seconds": perf_counter() - started,
            "root_geometry": "First fixed input case; no search or acceptance occurs",
        }
        plot_diagnostic(root, config, scores, attempts)
        write_report(root, config, scores, attempts, outcomes, quotas, summary)
        save_summary(root, summary)
        atomic_json(
            root / "status.json", {"status": summary["status"], "number_of_pde_solves": solves}
        )
        return root, summary
    except Exception as exc:
        flush()
        log_failure(root, exc)
        atomic_json(
            root / "status.json",
            {"status": "failed", "reason": str(exc), "number_of_pde_solves": solves},
        )
        raise
