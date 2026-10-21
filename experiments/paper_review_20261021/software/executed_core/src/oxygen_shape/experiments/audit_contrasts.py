"""Descriptive refinement of prespecified extreme shared-pool flux contrasts."""

from pathlib import Path

import numpy as np

from oxygen_shape.experiments.feasibility_config import FeasibilityConfig
from oxygen_shape.experiments.frozen_geometry import FixedCase
from oxygen_shape.experiments.reporting import write_csv
from oxygen_shape.geometry.constraints import validate_shape
from oxygen_shape.geometry.raster import refine_geometry
from oxygen_shape.optimization.random_search import constraints_for
from oxygen_shape.solvers.finite_volume import solve


def refine_audit_contrasts(
    root: Path, cases: list[FixedCase], config: FeasibilityConfig, audit: list[dict]
) -> tuple[int, float]:
    rows, solves, maximum_error = [], 0, 0.0
    for case in cases:
        grouped = {}
        for row in audit:
            if row["case"] == case.label:
                grouped.setdefault((row["replicate"], row["pool"]), {})[row["method"]] = row
        pairs = []
        for key, methods in grouped.items():
            first, second = methods["screened_uniform"], methods["perimeter_pool"]
            if (
                first["valid"]
                and second["valid"]
                and first["geometry_hash"] != second["geometry_hash"]
            ):
                pairs.append((first["delta_Q"] - second["delta_Q"], key, first, second))
        for direction, sign in (("positive", 1), ("negative", -1)):
            eligible = [p for p in pairs if sign * p[0] > config.effect_tolerance]
            if not eligible:
                rows.append({"case": case.label, "direction": direction, "status": "unavailable"})
                continue
            contrast, (replicate, pool), first, second = max(eligible, key=lambda p: sign * p[0])
            masks = []
            for selected in (first, second):
                with np.load(
                    root / "audit_candidates" / f"solve_{selected['audit_solution_id']:05d}.npz"
                ) as data:
                    masks.append(data["mask"].copy())
            for factor in config.refinement_factors:
                solutions = []
                reference = refine_geometry(case.geometry, factor)
                for mask in masks:
                    geometry = refine_geometry(case.geometry.with_mask(mask), factor)
                    validate_shape(
                        geometry, constraints_for(config.search_config(case.family)), reference
                    ).require_valid()
                    solution = solve(geometry, config.model, config.solver)
                    solutions.append(solution)
                    solves += 1
                    maximum_error = max(maximum_error, solution.metrics.conservation_error)
                path = root / "audit_contrasts" / f"{case.label}_{direction}_{factor}.npz"
                path.parent.mkdir(exist_ok=True)
                np.savez_compressed(
                    path,
                    screened_mask=np.repeat(np.repeat(masks[0], factor, 0), factor, 1),
                    perimeter_mask=np.repeat(np.repeat(masks[1], factor, 0), factor, 1),
                    screened_concentration=solutions[0].concentration,
                    perimeter_concentration=solutions[1].concentration,
                )
                rows.append(
                    {
                        "case": case.label,
                        "direction": direction,
                        "status": "complete",
                        "replicate": replicate,
                        "pool": pool,
                        "factor": factor,
                        "screened_audit_solution_id": first["audit_solution_id"],
                        "perimeter_audit_solution_id": second["audit_solution_id"],
                        "coarse_contrast_Q": contrast,
                        "contrast_Q": solutions[0].metrics.physical_flux_Q
                        - solutions[1].metrics.physical_flux_Q,
                        "conservation_error": max(s.metrics.conservation_error for s in solutions),
                    }
                )
    write_csv(root / "audit_contrasts.csv", rows)
    return solves, maximum_error
