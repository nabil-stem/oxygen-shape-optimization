"""Reproducible solve and slab validation workflows."""

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from oxygen_shape.experiments.config import ExperimentConfig, SlabConfig, SolveConfig
from oxygen_shape.experiments.reporting import (
    create_run,
    log_failure,
    provenance,
    save_state,
    save_summary,
    write_csv,
)
from oxygen_shape.geometry.boundary import BoundaryKind
from oxygen_shape.geometry.generators import rectangle
from oxygen_shape.geometry.raster import RasterGeometry
from oxygen_shape.solvers.analytical_1d import slab_flux, slab_solution
from oxygen_shape.solvers.finite_volume import SolveResult, solve
from oxygen_shape.viz.fields import plot_fields, plot_slab_errors


def base_summary(
    config: ExperimentConfig,
    geometry: RasterGeometry,
    result: SolveResult,
) -> dict[str, Any]:
    return {
        **provenance(),
        "seed": config.seed,
        "stochastic": False,
        "grid_shape": list(geometry.inside.shape),
        "h": geometry.h,
        "Bi": config.model.Bi,
        "solver": "finite_volume",
        "linear_solver": "scipy.sparse.linalg.spsolve",
        "solver_tolerance": None,
        "diagnostic_tolerances": config.solver.model_dump(),
        "initial_Q": result.metrics.physical_flux_Q,
        "final_Q": result.metrics.physical_flux_Q,
        "initial_eta": result.metrics.efficiency_eta,
        "final_eta": result.metrics.efficiency_eta,
        **asdict(result.metrics),
        "boundary_conditions": "u=1 at source; homogeneous Robin; zero Neumann",
        "optimization_performed": False,
    }


def run_solve(config: SolveConfig) -> tuple[Path, dict[str, Any]]:
    directory = create_run(config, "rectangle")
    start = perf_counter()
    try:
        g = config.geometry
        geometry = rectangle(g.nx, g.ny, g.h, slab=g.boundary_mode == "slab")
        result = solve(geometry, config.model, config.solver)
        save_state(directory, geometry, result)
        write_csv(directory / "metrics.csv", [asdict(result.metrics)])
        plot_fields(geometry, result, directory / "figures" / "fields.png")
        summary = base_summary(config, geometry, result)
        summary.update(number_of_pde_solves=1, wall_clock_seconds=perf_counter() - start)
        save_summary(directory, summary)
        return directory, summary
    except Exception as exc:
        log_failure(directory, exc)
        raise


def run_slab_validation(config: SlabConfig) -> tuple[Path, dict[str, Any]]:
    directory = create_run(config, "validate_1d")
    start = perf_counter()
    rows: list[dict[str, Any]] = []
    try:
        for nx in config.resolutions:
            h = config.length / nx
            geometry = rectangle(nx, config.transverse_cells, h, slab=True)
            result = solve(geometry, config.model, config.solver)
            x = (np.arange(nx) + 0.5) * h
            exact = slab_solution(x, config.model.Bi, config.length)
            robin = np.array([f.kind == BoundaryKind.ROBIN for f in result.faces])
            q = slab_flux(config.model.Bi, config.length)
            width = config.transverse_cells * h
            row = {
                "nx": nx,
                "ny": config.transverse_cells,
                "h": h,
                "width": width,
                **asdict(result.metrics),
                "concentration_error_inf": float(np.max(np.abs(result.concentration - exact))),
                "flux_density_error": abs(result.metrics.physical_flux_Q / width - q),
                "wall_concentration_error": float(
                    np.max(
                        np.abs(
                            result.wall_concentration[robin]
                            - slab_solution(config.length, config.model.Bi, config.length)
                        )
                    )
                ),
            }
            rows.append(row)
            if (
                max(
                    row["concentration_error_inf"],
                    row["flux_density_error"],
                    row["wall_concentration_error"],
                )
                > config.validation_tolerance
            ):
                raise ArithmeticError(f"Slab analytical validation failed at nx={nx}: {row}")
            grid_dir = directory / f"grid_{nx}"
            grid_dir.mkdir()
            save_state(grid_dir, geometry, result)
        save_state(directory, geometry, result)
        write_csv(directory / "metrics.csv", rows)
        plot_slab_errors(rows, directory / "figures" / "slab_errors.png")
        plot_fields(geometry, result, directory / "figures" / "fields.png")
        summary = base_summary(config, geometry, result)
        summary.update(
            experiment="analytical_1d_validation",
            resolutions=config.resolutions,
            number_of_pde_solves=len(rows),
            wall_clock_seconds=perf_counter() - start,
            analytical_flux_density=q,
            max_concentration_error=max(r["concentration_error_inf"] for r in rows),
            max_flux_density_error=max(r["flux_density_error"] for r in rows),
            max_wall_concentration_error=max(r["wall_concentration_error"] for r in rows),
            max_conservation_error=max(r["conservation_error"] for r in rows),
            validation_tolerance=config.validation_tolerance,
            refinement_note="Slab is linear and exact to roundoff; no fitted convergence order. "
            "Transverse width varies; compare flux per unit width.",
        )
        save_summary(directory, summary)
        return directory, summary
    except Exception as exc:
        log_failure(directory, exc)
        raise
