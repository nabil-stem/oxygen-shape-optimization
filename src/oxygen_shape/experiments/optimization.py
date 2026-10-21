"""Reproducible random disk-search experiment and storage shared with RL."""

from pathlib import Path
from time import perf_counter

import numpy as np

from oxygen_shape.experiments.reporting import (
    create_run,
    log_failure,
    provenance,
    save_state,
    save_summary,
    write_csv,
)
from oxygen_shape.optimization.config import SearchConfig
from oxygen_shape.optimization.random_search import SearchResult, run_search
from oxygen_shape.optimization.refinement import validate_refinement
from oxygen_shape.viz.fields import plot_fields
from oxygen_shape.viz.optimization import plot_search


def save_search(directory: Path, result: SearchResult) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    save_state(directory, result.final_geometry, result.final_solution)
    np.save(directory / "initial_geometry.npy", result.initial_geometry.inside)
    np.save(directory / "initial_concentration.npy", result.initial_solution.concentration)
    write_csv(directory / "metrics.csv", result.history)
    write_csv(directory / "proposals.csv", result.proposals)


def search_summary(config: SearchConfig, result: SearchResult, refinements: list[dict]) -> dict:
    coarse, fine = refinements[0], refinements[-1]
    sensitivity = abs(fine["gain_Q"] - refinements[-2]["gain_Q"])
    return {
        "seed": config.seed,
        "grid_shape": list(result.final_geometry.inside.shape),
        "Bi": config.model.Bi,
        "solver": "finite_volume",
        "solver_tolerance": None,
        "diagnostic_tolerances": config.solver.model_dump(),
        "initial_Q": coarse["initial_Q"],
        "final_Q": coarse["final_Q"],
        "initial_eta": result.initial_solution.metrics.efficiency_eta,
        "final_eta": result.final_solution.metrics.efficiency_eta,
        "area": result.final_geometry.area,
        "robin_perimeter": result.final_solution.metrics.robin_perimeter,
        "conservation_error": max(row["conservation_error"] for row in result.history),
        "number_of_pde_solves": result.number_of_pde_solves,
        "refinement_pde_solves": 2 * len(config.refinement_factors),
        "proposal_attempts": result.attempts,
        "uniform_fallback_attempts": sum(row["uniform_fallback"] for row in result.proposals),
        "accepted_moves": sum(row["accepted"] for row in result.history),
        "fine_initial_Q": fine["initial_Q"],
        "fine_final_Q": fine["final_Q"],
        "fine_gain_Q": fine["gain_Q"],
        "observed_gain_grid_sensitivity": sensitivity,
        "gain_exceeds_observed_sensitivity": fine["gain_Q"] > sensitivity,
        "constraints": {
            "target_area": config.target_area,
            "area_tolerance": 1e-12,
            "perimeter_limit": config.perimeter_limit,
            "minimum_feature_width": config.minimum_feature_width,
            "protected_length": config.protected_length,
            "forbid_holes": True,
        },
    }


def run_optimization(config: SearchConfig, progress=None) -> tuple[Path, dict]:
    directory = create_run(config, "optimize_disks")
    start = perf_counter()
    (directory / "iterations").mkdir()

    def on_step(step, geometry, row):
        np.save(directory / "iterations" / f"geometry_{step:05d}.npy", geometry.inside)
        if progress and step % 100 == 0:
            progress(
                f"attempt={step}, solves={row['number_of_pde_solves']}, "
                f"Q={row['physical_flux_Q']:.6g}"
            )

    try:
        result = run_search(config, on_step=on_step)
        save_search(directory, result)
        refinements = validate_refinement(config, result, directory)
        plot_fields(
            result.final_geometry, result.final_solution, directory / "figures" / "final_fields.png"
        )
        plot_search(result, directory / "figures" / "history.png")
        summary = {
            **provenance(),
            **search_summary(config, result, refinements),
            "wall_clock_seconds": perf_counter() - start,
            "policy": "uniform_boundary",
        }
        save_summary(directory, summary)
        return directory, summary
    except Exception as exc:
        log_failure(directory, exc)
        raise
