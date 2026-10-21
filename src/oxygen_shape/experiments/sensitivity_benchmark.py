"""Fresh-seed equal-cost comparison of three constrained proposal samplers."""

from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

from oxygen_shape.experiments.optimization import save_search, search_summary
from oxygen_shape.experiments.reporting import (
    create_run,
    log_failure,
    provenance,
    save_summary,
    write_csv,
)
from oxygen_shape.experiments.sensitivity_report import plot_comparison, plot_pair_map, write_report
from oxygen_shape.experiments.study_worker import atomic_json
from oxygen_shape.geometry.baselines import matched_area
from oxygen_shape.optimization.guided_config import SensitivityBenchmarkConfig
from oxygen_shape.optimization.guided_search import run_guided_search, validate_candidate
from oxygen_shape.optimization.refinement import validate_refinement
from oxygen_shape.viz.fields import plot_fields
from oxygen_shape.viz.optimization import plot_search

METHODS = ("uniform", "flux", "sensitivity")


def run_sensitivity_benchmark(
    config: SensitivityBenchmarkConfig, progress=None
) -> tuple[Path, dict]:
    directory = create_run(config, "sensitivity_benchmark")
    metadata = provenance()
    started = perf_counter()
    evaluations, histories = [], []
    total_runs = len(config.shapes) * len(config.evaluation_seeds) * len(METHODS)
    try:
        # Validate every family before paying for any PDE solve.
        for shape in config.shapes:
            cfg = config.model_copy(update={"initial_shape": shape})
            initial = matched_area(shape, cfg.n, cfg.target_area).geometry
            reasons = validate_candidate(initial, initial, cfg)
            if reasons:
                raise ValueError(f"Initial {shape}: {reasons}")
        for shape in config.shapes:
            for seed_index, seed in enumerate(config.evaluation_seeds):
                # Rotate execution order to reduce systematic warmup/order effects.
                order = METHODS[seed_index % 3 :] + METHODS[: seed_index % 3]
                for method in order:
                    job_id = f"{shape}_{method}_{seed}"
                    path = directory / "runs" / job_id
                    path.mkdir(parents=True)
                    (path / "figures").mkdir()
                    cfg = config.model_copy(update={"initial_shape": shape, "seed": seed})
                    (path / "config.yaml").write_text(
                        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False),
                        encoding="utf-8",
                    )
                    atomic_json(
                        directory / "status.json",
                        {
                            "status": "running",
                            "active_run": job_id,
                            "completed_runs": len(evaluations),
                            "planned_runs": total_runs,
                        },
                    )
                    run_started = perf_counter()

                    def save_map(
                        round_index, geometry, points, probabilities, responses, *, run_path=path
                    ):
                        maps = run_path / "maps"
                        maps.mkdir(exist_ok=True)
                        np.savez_compressed(
                            maps / f"round_{round_index:03d}.npz",
                            geometry=geometry.inside,
                            points=points,
                            probabilities=probabilities,
                            addition=np.asarray([r.addition for r in responses]),
                            donor=np.asarray([r.donor for r in responses]),
                            delta_Q=np.asarray([r.delta_Q for r in responses]),
                            reallocated_area=np.asarray([r.reallocated_area for r in responses]),
                        )

                    result = run_guided_search(cfg, method, on_map=save_map)
                    search_seconds = perf_counter() - run_started
                    search = result.search
                    save_search(path, search)
                    write_csv(path / "rounds.csv", result.rounds)
                    if result.probes:
                        write_csv(path / "sensitivity_probes.csv", result.probes)
                    refinements = validate_refinement(cfg, search, path)
                    target = search.initial_solution.metrics.physical_flux_Q * (
                        1 + config.target_relative_gain
                    )
                    reached = [
                        r["number_of_pde_solves"]
                        for r in search.history
                        if r["physical_flux_Q"] >= target
                    ]
                    row = {
                        "shape": shape,
                        "method": method,
                        "status": result.status,
                        **search_summary(cfg, search, refinements),
                        "conservation_error": max(
                            result.maximum_candidate_conservation_error,
                            max(r["conservation_error"] for r in refinements),
                        ),
                        "probe_pde_solves": result.probe_pde_solves,
                        "nonprobe_candidate_pde_solves": search.number_of_pde_solves
                        - 1
                        - result.probe_pde_solves,
                        "accepted_rounds": sum(r["accepted"] for r in result.rounds),
                        "search_wall_seconds": search_seconds,
                        "wall_clock_seconds": perf_counter() - run_started,
                        "first_target_solve": min(reached) if reached else None,
                        "previous_grid_final_Q": refinements[-2]["final_Q"],
                    }
                    save_summary(path, {**metadata, **row})
                    evaluations.append(row)
                    if len(evaluations) == 1:
                        save_search(directory, search)
                    histories.extend(
                        {"shape": shape, "method": method, "seed": seed, **r}
                        for r in search.history
                    )
                    write_csv(
                        directory / "evaluations.csv",
                        [{k: v for k, v in r.items() if k != "constraints"} for r in evaluations],
                    )
                    write_csv(directory / "metrics.csv", histories)
                    write_report(directory, config, evaluations, status="running")
                    if progress:
                        progress(
                            f"{job_id}: {result.status}, Q={row['final_Q']:.7g}; "
                            f"solves={search.number_of_pde_solves}, "
                            f"probes={result.probe_pde_solves}, "
                            f"search={search_seconds:.2f}s"
                        )
                    if seed_index == 0:
                        plot_fields(
                            search.final_geometry,
                            search.final_solution,
                            path / "figures/final_fields.png",
                        )
                        plot_search(search, path / "figures/history.png")
                        maps = sorted((path / "maps").glob("*.npz"))
                        if maps:
                            plot_pair_map(maps[0], path / "figures/first_map.png")
        status = (
            "complete" if all(r["status"] == "complete" for r in evaluations) else "incomplete_runs"
        )
        summary = {
            **metadata,
            "status": status,
            "seed_schedule": config.evaluation_seeds,
            "grid_shape": [config.n, config.n],
            "Bi": config.model.Bi,
            "solver": "finite_volume",
            "solver_tolerance": None,
            "diagnostic_tolerances": config.solver.model_dump(),
            "planned_runs": total_runs,
            "recorded_runs": len(evaluations),
            "complete_runs": sum(r["status"] == "complete" for r in evaluations),
            "number_of_pde_solves": sum(r["number_of_pde_solves"] for r in evaluations),
            "probe_pde_solves_included": sum(r["probe_pde_solves"] for r in evaluations),
            "refinement_pde_solves": sum(r["refinement_pde_solves"] for r in evaluations),
            "max_conservation_error": max(r["conservation_error"] for r in evaluations),
            "wall_clock_seconds": perf_counter() - started,
            "root_geometry": "First uniform run; see runs/ for all methods/shapes/seeds",
        }
        plot_comparison(histories, evaluations, config, directory / "figures/comparison.png")
        write_report(directory, config, evaluations, status=status)
        save_summary(directory, summary)
        atomic_json(
            directory / "status.json",
            {
                "status": status,
                "active_run": None,
                "recorded_runs": len(evaluations),
                "planned_runs": total_runs,
            },
        )
        return directory, summary
    except Exception as exc:
        log_failure(directory, exc)
        atomic_json(
            directory / "status.json",
            {
                "status": "failed",
                "reason": str(exc),
                "recorded_runs": len(evaluations),
                "planned_runs": total_runs,
            },
        )
        raise
