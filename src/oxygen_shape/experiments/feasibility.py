"""Geometry-only audit followed by a frozen sampler's equal-PDE-budget search."""

import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

from oxygen_shape.experiments.audit_contrasts import refine_audit_contrasts
from oxygen_shape.experiments.feasibility_config import FeasibilityConfig
from oxygen_shape.experiments.feasibility_report import write_report
from oxygen_shape.experiments.frozen_geometry import load_fixed_cases
from oxygen_shape.experiments.optimization import save_search
from oxygen_shape.experiments.reporting import (
    create_run,
    log_failure,
    provenance,
    save_state,
    save_summary,
    write_csv,
)
from oxygen_shape.experiments.study_worker import atomic_json
from oxygen_shape.geometry.disk_edits import prepare_boundary
from oxygen_shape.optimization.guided_search import validate_candidate
from oxygen_shape.optimization.perimeter_pool import choose_pool, draw_pool, run_pool_search
from oxygen_shape.optimization.refinement import validate_refinement
from oxygen_shape.solvers.finite_volume import solve
from oxygen_shape.viz.fields import plot_fields


def run_feasibility_study(config: FeasibilityConfig, progress=None) -> tuple[Path, dict]:
    root = create_run(config, "proposal_feasibility")
    metadata, start = provenance(), perf_counter()
    audit, raw, evaluations, histories = [], [], [], []
    baseline_solves, search_solves, refinement_solves = 0, 0, 0
    audit_solves, audit_refinement_solves = 0, 0
    baseline_fluxes, audit_candidates = {}, []
    maximum_error = 0.0

    def update(phase, active):
        atomic_json(
            root / "status.json",
            {
                "status": "running",
                "phase": phase,
                "active": active,
                "completed_searches": len(evaluations),
                "number_of_pde_solves": baseline_solves
                + search_solves
                + refinement_solves
                + audit_solves
                + audit_refinement_solves,
            },
        )

    try:
        cases = load_fixed_cases(config, root)
        (root / "fixed_cases").mkdir()
        if config.audit_response_solves:
            (root / "audit_candidates").mkdir()
        # Verify every frozen Q before either the audit or the searches.
        for case in cases:
            solution = solve(case.geometry, config.model, config.solver)
            baseline_solves += 1
            maximum_error = max(maximum_error, solution.metrics.conservation_error)
            baseline_fluxes[case.label] = solution.metrics.physical_flux_Q
            if case.expected_Q is not None and not np.isclose(
                solution.metrics.physical_flux_Q, case.expected_Q, rtol=1e-10, atol=1e-12
            ):
                raise ValueError(f"Frozen Q mismatch: {case.label}")
            np.save(root / "fixed_cases" / f"{case.label}.npy", case.geometry.inside)
            if case is cases[0]:
                save_state(root, case.geometry, solution)
        for case in cases:
            cfg = config.search_config(case.family)
            prepared = prepare_boundary(case.geometry, config.protected_length)
            for replicate in range(config.replicates):
                seed = config.seed + replicate
                rng = np.random.default_rng(seed)
                update("audit", f"{case.label}/{replicate}")
                for pool in range(config.pools_per_replicate):
                    edits, rows = draw_pool(case.geometry, rng, cfg, prepared, config.pool_size)
                    selection_seed = None
                    if "screened_uniform" in config.methods:
                        selection_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                        choices = {"uniform": (0, "uniform")}
                        for method in config.methods:
                            if method != "uniform":
                                index, mode, screens = choose_pool(
                                    edits,
                                    case.geometry,
                                    cfg,
                                    np.random.default_rng(selection_seed),
                                    config.epsilon_uniform,
                                    selection="screened_uniform"
                                    if method == "screened_uniform"
                                    else "perimeter",
                                )
                                choices[method] = (index, mode)
                    else:
                        selected, mode, screens = choose_pool(
                            edits, case.geometry, cfg, rng, config.epsilon_uniform
                        )
                        choices = {"uniform": (0, "uniform"), "perimeter_pool": (selected, mode)}
                    outcomes = []
                    for edit, row, screen in zip(edits, rows, screens, strict=True):
                        tick = perf_counter()
                        reason = (
                            edit.rejection
                            if edit.geometry is None
                            else validate_candidate(edit.geometry, case.geometry, cfg)
                        )
                        result = {
                            "case": case.label,
                            "replicate": replicate,
                            "seed": seed,
                            "pool": pool,
                            "valid": not bool(reason),
                            "rejection": reason,
                            "validation_seconds": perf_counter() - tick,
                            "geometry_hash": None
                            if edit.geometry is None
                            else hashlib.sha256(edit.geometry.inside.tobytes()).hexdigest(),
                            **screen,
                            **row,
                        }
                        raw.append(result)
                        outcomes.append(result)
                    pool_responses = {}
                    for method in config.methods:
                        index, mode = choices[method]
                        response = {}
                        charged = False
                        outcome = outcomes[index]
                        if config.audit_response_solves and outcome["valid"]:
                            signature = outcome["geometry_hash"]
                            if signature not in pool_responses:
                                candidate = solve(
                                    edits[index].geometry, config.model, config.solver
                                )
                                audit_solves += 1
                                charged = True
                                maximum_error = max(
                                    maximum_error, candidate.metrics.conservation_error
                                )
                                response = {
                                    "audit_solution_id": audit_solves,
                                    "delta_Q": candidate.metrics.physical_flux_Q
                                    - baseline_fluxes[case.label],
                                    **candidate.metrics.__dict__,
                                }
                                pool_responses[signature] = response
                                np.savez_compressed(
                                    root / "audit_candidates" / f"solve_{audit_solves:05d}.npz",
                                    mask=edits[index].geometry.inside,
                                    concentration=candidate.concentration,
                                )
                                audit_candidates.append(
                                    {
                                        "case": case.label,
                                        "replicate": replicate,
                                        "pool": pool,
                                        "geometry_hash": signature,
                                        **response,
                                    }
                                )
                            response = pool_responses[signature]
                        audit.append(
                            {
                                "case": case.label,
                                "replicate": replicate,
                                "seed": seed,
                                "pool": pool,
                                "method": method,
                                "selected_slot": index,
                                "selection_mode": mode,
                                "selection_seed": selection_seed,
                                "raw_cost": 1 if method == "uniform" else config.pool_size,
                                "valid": outcomes[index]["valid"],
                                "geometry_hash": outcomes[index]["geometry_hash"],
                                "addition_face": outcomes[index]["addition_face"],
                                "donor_face": outcomes[index]["donor_face"],
                                "radius": outcomes[index]["radius"],
                                "selected_perimeter": outcomes[index]["perimeter"],
                                "rejection": outcomes[index]["rejection"],
                                **(
                                    {
                                        "audit_pde_solve_charged": charged,
                                        "audit_solution_id": None,
                                        "delta_Q": None,
                                        "one_step_gain_Q": response.get("delta_Q", 0.0)
                                        if response.get("delta_Q", 0.0) > cfg.min_improvement
                                        else 0.0,
                                        **response,
                                    }
                                    if config.audit_response_solves
                                    else {}
                                ),
                            }
                        )
                write_csv(root / "audit.csv", audit)
                write_csv(root / "audit_raw.csv", raw)
                if audit_candidates:
                    write_csv(root / "audit_candidate_metrics.csv", audit_candidates)
                if progress:
                    progress(
                        f"Audit {case.label}, replicate={replicate}: "
                        f"{config.pools_per_replicate} pools complete"
                    )
        if config.audit_response_solves:
            update("audit_refinement", "extreme shared-pool contrasts")
            audit_refinement_solves, error = refine_audit_contrasts(root, cases, config, audit)
            maximum_error = max(maximum_error, error)
        for case in [c for c in cases if c.state == "optimized"]:
            for seed_index, seed in enumerate(config.search_seeds):
                offset = seed_index % len(config.methods)
                order = config.methods[offset:] + config.methods[:offset]
                for method in order:
                    job = f"{case.label}_{method}_{seed}"
                    path = root / "runs" / job
                    path.mkdir(parents=True)
                    (path / "figures").mkdir()
                    (path / "candidates").mkdir()
                    cfg = config.search_config(case.family).model_copy(
                        update={"seed": seed, "solve_budget": config.search_solve_budget}
                    )
                    (path / "config.yaml").write_text(
                        yaml.safe_dump(
                            {
                                **cfg.model_dump(mode="json"),
                                "method": method,
                                "pool_size": config.pool_size,
                                "raw_draw_budget": config.raw_draw_budget,
                                "epsilon_uniform": config.epsilon_uniform,
                            }
                        ),
                        encoding="utf-8",
                    )
                    update("search", job)
                    candidate_rows = []

                    def save_candidate(
                        index,
                        geometry,
                        solution,
                        accepted,
                        *,
                        destination=path,
                        rows=candidate_rows,
                    ):
                        np.savez_compressed(
                            destination / "candidates" / f"solve_{index:03d}.npz",
                            mask=geometry.inside,
                            concentration=solution.concentration,
                        )
                        rows.append(
                            {
                                "solve": index,
                                "accepted": accepted,
                                "geometry_hash": hashlib.sha256(
                                    geometry.inside.tobytes()
                                ).hexdigest(),
                                "area": geometry.area,
                                **solution.metrics.__dict__,
                            }
                        )

                    tick = perf_counter()
                    result = run_pool_search(
                        cfg,
                        case.geometry,
                        method=method,
                        pool_size=config.pool_size,
                        epsilon_uniform=config.epsilon_uniform,
                        raw_budget=config.raw_draw_budget,
                        on_evaluation=save_candidate,
                        progress=progress,
                    )
                    search_seconds = perf_counter() - tick
                    search = result.search
                    search_solves += search.number_of_pde_solves
                    save_search(path, search)
                    if candidate_rows:
                        write_csv(path / "candidate_metrics.csv", candidate_rows)
                    refinements = validate_refinement(cfg, search, path)
                    refinement_solves += 2 * len(config.refinement_factors)
                    maximum_error = max(
                        maximum_error,
                        result.maximum_conservation_error,
                        max(r["conservation_error"] for r in refinements),
                    )
                    plot_fields(
                        search.final_geometry, search.final_solution, path / "figures/final.png"
                    )
                    row = {
                        "case": case.label,
                        "family": case.family,
                        "seed": seed,
                        "method": method,
                        "status": result.status,
                        "initial_Q": search.initial_solution.metrics.physical_flux_Q,
                        "final_Q": search.final_solution.metrics.physical_flux_Q,
                        "initial_eta": search.initial_solution.metrics.efficiency_eta,
                        "final_eta": search.final_solution.metrics.efficiency_eta,
                        "area": search.final_geometry.area,
                        "robin_perimeter": search.final_solution.metrics.robin_perimeter,
                        "raw_draws": search.attempts,
                        "selected_checks": result.selected_checks,
                        "search_pde_solves": search.number_of_pde_solves,
                        "refinement_pde_solves": 2 * len(config.refinement_factors),
                        "accepted_moves": sum(r["accepted"] for r in search.history),
                        "geometry_seconds": result.geometry_seconds,
                        "search_seconds": search_seconds,
                        "fine_gain_Q": refinements[-1]["gain_Q"],
                        "fine_final_Q": refinements[-1]["final_Q"],
                        "conservation_error": max(
                            result.maximum_conservation_error,
                            max(r["conservation_error"] for r in refinements),
                        ),
                    }
                    save_summary(path, {**metadata, **row})
                    evaluations.append(row)
                    histories.extend(
                        {"case": case.label, "method": method, "seed": seed, **r}
                        for r in search.history
                    )
                    write_csv(root / "evaluations.csv", evaluations)
                    write_csv(root / "metrics.csv", histories)
                    if progress:
                        progress(
                            f"Search {job}: {result.status}, "
                            f"Q={row['initial_Q']:.7g}->{row['final_Q']:.7g}, raw={search.attempts}"
                        )
        summary = {
            **metadata,
            "status": "complete"
            if all(r["status"] == "complete" for r in evaluations)
            else "incomplete",
            "seed": config.seed,
            "audit_seeds": list(range(config.seed, config.seed + config.replicates)),
            "search_seeds": config.search_seeds,
            "grid_shape": [config.n, config.n],
            "Bi": config.model.Bi,
            "solver": "finite_volume",
            "solver_tolerance": None,
            "diagnostic_tolerances": config.solver.model_dump(),
            "pool_size": config.pool_size,
            "epsilon_uniform": config.epsilon_uniform,
            "baseline_pde_solves": baseline_solves,
            "audit_pde_solves": audit_solves,
            "audit_refinement_pde_solves": audit_refinement_solves,
            "search_pde_solves": search_solves,
            "refinement_pde_solves": refinement_solves,
            "number_of_pde_solves": baseline_solves
            + search_solves
            + refinement_solves
            + audit_solves
            + audit_refinement_solves,
            "audit_raw_draws": len(raw),
            "completed_searches": sum(r["status"] == "complete" for r in evaluations),
            "planned_searches": len(config.methods)
            * len(config.search_seeds)
            * len(config.frozen_references),
            "methods": config.methods,
            "max_conservation_error": maximum_error,
            "wall_clock_seconds": perf_counter() - start,
        }
        save_summary(root, summary)
        write_report(root, config, summary, audit, raw, evaluations, histories)
        atomic_json(
            root / "status.json",
            {"status": summary["status"], "number_of_pde_solves": summary["number_of_pde_solves"]},
        )
        return root, summary
    except Exception as exc:
        for name, rows in (
            ("audit.csv", audit),
            ("audit_raw.csv", raw),
            ("evaluations.csv", evaluations),
            ("metrics.csv", histories),
            ("audit_candidate_metrics.csv", audit_candidates),
        ):
            if rows:
                write_csv(root / name, rows)
        log_failure(root, exc)
        atomic_json(root / "status.json", {"status": "failed", "error": str(exc)})
        raise
