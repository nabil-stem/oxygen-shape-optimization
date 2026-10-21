"""Matched-resource baseline sweep, screening diagnostics, and refinement report."""

from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from oxygen_shape.experiments.baseline_report import build_report, ranking_rows
from oxygen_shape.experiments.config import BaselineConfig
from oxygen_shape.experiments.reporting import (
    create_run,
    log_failure,
    provenance,
    save_state,
    save_summary,
    write_csv,
)
from oxygen_shape.geometry.baselines import (
    SHAPE_NAMES,
    BaselineGeometry,
    matched_area,
    matched_perimeter,
)
from oxygen_shape.geometry.boundary import BoundaryKind
from oxygen_shape.geometry.metrics import geometry_metrics
from oxygen_shape.physics.model import DiffusionModel
from oxygen_shape.solvers.finite_volume import solve
from oxygen_shape.viz.comparisons import plot_overview, plot_refinement, plot_sweep


def refinement_metrics(directory: Path, rows: list[dict]) -> list[dict]:
    refinements: list[dict] = []
    for row in rows:
        preceding = [
            old
            for old in rows
            if old["comparison"] == row["comparison"]
            and old["shape"] == row["shape"]
            and old["Bi"] == row["Bi"]
            and old["n"] * 2 == row["n"]
        ]
        if not preceding:
            continue
        old = preceding[0]
        n = old["n"]
        coarse = np.load(directory / old["case_path"] / "concentration.npy")
        fine = np.load(directory / row["case_path"] / "concentration.npy")
        blocks = fine.reshape(n, 2, n, 2)
        valid = np.isfinite(coarse) & np.isfinite(blocks).all(axis=(1, 3))
        restricted = blocks.mean(axis=(1, 3))
        field_difference = (
            float(np.max(np.abs(coarse[valid] - restricted[valid]))) if valid.any() else None
        )
        refinements.append(
            {
                "comparison": row["comparison"],
                "shape": row["shape"],
                "Bi": row["Bi"],
                "coarse_n": n,
                "fine_n": row["n"],
                "relative_Q_change": abs(row["physical_flux_Q"] - old["physical_flux_Q"])
                / row["physical_flux_Q"],
                "absolute_eta_change": abs(row["efficiency_eta"] - old["efficiency_eta"]),
                "relative_area_change": abs(row["area"] - old["area"]) / row["area"],
                "relative_robin_perimeter_change": abs(
                    row["robin_perimeter"] - old["robin_perimeter"]
                )
                / row["robin_perimeter"],
                "interior_field_difference_inf": field_difference,
                "interior_comparison_coverage": float(valid.sum() / np.isfinite(coarse).sum()),
            }
        )
    return refinements


def run_baselines(
    config: BaselineConfig,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    directory = create_run(config, "baseline_shapes")
    start = perf_counter()
    rows: list[dict] = []
    try:
        for n in config.resolutions:
            generated: list[BaselineGeometry] = [
                matched_area(name, n, config.target_area, hilbert_order=config.hilbert_order)
                for name in SHAPE_NAMES
            ]
            generated.extend(
                matched_perimeter(name, n, config.target_perimeter)
                for name in ("rectangle", "square")
            )
            for baseline in generated:
                geometry = baseline.geometry
                metrics = geometry_metrics(geometry)
                if (
                    baseline.comparison == "fixed_area"
                    and metrics.robin_perimeter > config.perimeter_budget
                ):
                    raise ValueError(f"Membrane budget exceeded by {baseline.shape} at n={n}")
                if baseline.matching_error > baseline.matching_tolerance + 1e-12:
                    raise ValueError("Resource matching failed")
                finest_cases = []
                for Bi in config.permeabilities:
                    case_start = perf_counter()
                    result = solve(geometry, DiffusionModel(Bi=Bi), config.solver)
                    relative = (
                        Path("cases")
                        / baseline.comparison
                        / baseline.shape
                        / f"n_{n}"
                        / f"Bi_{Bi:g}"
                    )
                    case_dir = directory / relative
                    case_dir.mkdir(parents=True)
                    save_state(case_dir, geometry, result)
                    robin = np.array([f.kind == BoundaryKind.ROBIN for f in result.faces])
                    wall = result.wall_concentration[robin]
                    row = {
                        "comparison": baseline.comparison,
                        "shape": baseline.shape,
                        "n": n,
                        "h": geometry.h,
                        "Bi": Bi,
                        **asdict(metrics),
                        **asdict(result.metrics),
                        "matching_target": baseline.target,
                        "matching_error": baseline.matching_error,
                        "matching_tolerance": baseline.matching_tolerance,
                        "screened_wall_fraction": float(np.mean(wall < 0.1)),
                        "minimum_wall_concentration": float(wall.min()),
                        "effective_membrane_length": result.metrics.physical_flux_Q / Bi,
                        "case_path": relative.as_posix(),
                        "number_of_pde_solves": 1,
                        "wall_clock_seconds": perf_counter() - case_start,
                    }
                    rows.append(row)
                    write_csv(case_dir / "metrics.csv", [row])
                    # Incremental metrics survive an interrupted sweep.
                    write_csv(directory / "metrics.csv", rows)
                    if n == config.resolutions[-1]:
                        finest_cases.append((Bi, result))
                if progress:
                    progress(
                        f"n={n} {baseline.comparison}/{baseline.shape}: "
                        f"{len(config.permeabilities)} solves passed"
                    )
                # Store just paths for final overview loading; no large all-case cache.
                if n == config.resolutions[-1]:
                    if not (directory / "initial_geometry.npy").exists():
                        save_state(directory, geometry, finest_cases[0][1])
        refinements = refinement_metrics(directory, rows)
        write_csv(directory / "refinement.csv", refinements)
        write_csv(directory / "rankings.csv", ranking_rows(config, rows))
        for group in ("fixed_area", "fixed_perimeter"):
            plot_sweep(rows, group, directory / "figures" / f"{group}_sweep.png")
            plot_refinement(rows, group, directory / "figures" / f"{group}_refinement.png")
            for Bi in config.permeabilities:
                cases = []
                for name in SHAPE_NAMES if group == "fixed_area" else ("rectangle", "square"):
                    baseline = (
                        matched_area(
                            name,
                            config.resolutions[-1],
                            config.target_area,
                            hilbert_order=config.hilbert_order,
                        )
                        if group == "fixed_area"
                        else matched_perimeter(
                            name, config.resolutions[-1], config.target_perimeter
                        )
                    )
                    # Reconstruct plot data from saved faces/fields, without an extra PDE solve.
                    row = next(
                        r
                        for r in rows
                        if r["comparison"] == group
                        and r["shape"] == name
                        and r["n"] == config.resolutions[-1]
                        and r["Bi"] == Bi
                    )
                    cases.append(
                        (name, baseline.geometry, load_case(directory / row["case_path"], row))
                    )
                plot_overview(cases, Bi, directory / "figures" / f"{group}_Bi_{Bi:g}.png")
        report = build_report(config, rows, refinements)
        (directory / "report.md").write_text(report + "\n", encoding="utf-8")
        final_refinement = [r for r in refinements if r["fine_n"] == config.resolutions[-1]]
        summary = {
            **provenance(),
            "experiment": "baseline_shapes",
            "seed": config.seed,
            "solver": "finite_volume",
            "solver_tolerance": None,
            "diagnostic_tolerances": config.solver.model_dump(),
            "resolutions": config.resolutions,
            "Bi": config.permeabilities,
            "grid_shape": [config.resolutions[-1]] * 2,
            "number_of_pde_solves": len(rows),
            "optimization_performed": False,
            "max_conservation_error": max(r["conservation_error"] for r in rows),
            "max_final_relative_Q_change": max(r["relative_Q_change"] for r in final_refinement),
            "refinement_flagged_cases": sum(
                r["relative_Q_change"] > config.refinement_warning_rtol for r in final_refinement
            ),
            "representative_root_state": "fixed_area/rectangle, finest grid, lowest Bi",
            "wall_clock_seconds": perf_counter() - start,
            "minimum_neck": None,
        }
        save_summary(directory, summary)
        return directory, summary
    except Exception as exc:
        log_failure(directory, exc)
        raise


def load_case(path: Path, row: dict):
    """Read stored physical face data for plotting, preserving the exact saved solution."""
    import csv

    from oxygen_shape.geometry.boundary import BoundaryFace
    from oxygen_shape.geometry.raster import FaceKey, Side
    from oxygen_shape.physics.metrics import FluxMetrics
    from oxygen_shape.solvers.finite_volume import SolveResult

    with (path / "boundary_faces.csv").open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    faces = tuple(
        BoundaryFace(
            FaceKey(int(r["row"]), int(r["col"]), Side(r["side"])), BoundaryKind(r["kind"])
        )
        for r in records
    )
    metrics = FluxMetrics(**{key: row[key] for key in FluxMetrics.__dataclass_fields__})
    return SolveResult(
        np.load(path / "concentration.npy"),
        metrics,
        faces,
        np.array([float(r["outward_integrated_flux"]) for r in records]),
        np.array([float(r["wall_concentration"]) for r in records]),
        np.array([float(r["conductance"]) for r in records]),
    )
