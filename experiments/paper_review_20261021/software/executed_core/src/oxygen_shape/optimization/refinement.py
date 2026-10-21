"""Refine the same physical raster cell union without changing its perimeter."""

from pathlib import Path

from oxygen_shape.experiments.reporting import save_state, write_csv
from oxygen_shape.geometry.constraints import validate_shape
from oxygen_shape.geometry.raster import refine_geometry
from oxygen_shape.optimization.config import SearchConfig
from oxygen_shape.optimization.random_search import SearchResult, constraints_for
from oxygen_shape.solvers.finite_volume import solve


def validate_refinement(config: SearchConfig, result: SearchResult, directory: Path) -> list[dict]:
    rows = [
        {
            "factor": 1,
            "n": config.n,
            "initial_Q": result.initial_solution.metrics.physical_flux_Q,
            "final_Q": result.final_solution.metrics.physical_flux_Q,
            "gain_Q": result.final_solution.metrics.physical_flux_Q
            - result.initial_solution.metrics.physical_flux_Q,
            "constraints_valid": True,
            "conservation_error": max(
                result.initial_solution.metrics.conservation_error,
                result.final_solution.metrics.conservation_error,
            ),
        }
    ]
    for factor in config.refinement_factors:
        initial = refine_geometry(result.initial_geometry, factor)
        final = refine_geometry(result.final_geometry, factor)
        constraints = constraints_for(config)
        validate_shape(initial, constraints, initial).require_valid()
        validate_shape(final, constraints, initial).require_valid()
        initial_solution = solve(initial, config.model, config.solver)
        final_solution = solve(final, config.model, config.solver)
        path = directory / f"refinement_{factor}"
        (path / "initial").mkdir(parents=True)
        (path / "final").mkdir()
        save_state(path / "initial", initial, initial_solution)
        save_state(path / "final", final, final_solution)
        rows.append(
            {
                "factor": factor,
                "n": config.n * factor,
                "initial_Q": initial_solution.metrics.physical_flux_Q,
                "final_Q": final_solution.metrics.physical_flux_Q,
                "gain_Q": final_solution.metrics.physical_flux_Q
                - initial_solution.metrics.physical_flux_Q,
                "constraints_valid": True,
                "conservation_error": max(
                    initial_solution.metrics.conservation_error,
                    final_solution.metrics.conservation_error,
                ),
            }
        )
    write_csv(directory / "refinement.csv", rows)
    return rows
