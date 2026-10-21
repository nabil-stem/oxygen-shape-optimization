"""Cached numerical checks with separate physical-polygon and raster questions."""

import hashlib
import json
from dataclasses import asdict
from io import BytesIO
from pathlib import Path

import numpy as np

from oxygen_shape.experiments.learning_diagnostics import incumbent_solution
from oxygen_shape.experiments.reporting import write_csv
from oxygen_shape.geometry.polygon_controls import chamber_pslg, rasterize_chamber
from oxygen_shape.geometry.raster import refine_geometry
from oxygen_shape.geometry.research_families import make_research_geometry
from oxygen_shape.optimization.trajectory_search import _atomic, _json
from oxygen_shape.physics.model import DiffusionModel
from oxygen_shape.solvers.fem_reference import FEMResult, solve_pslg, solve_raster_fem
from oxygen_shape.solvers.finite_volume import solve


def cached_fvm(path: Path, geometry, Bi: float, settings):
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(
        geometry.inside.tobytes()
        + repr(
            (geometry.h, geometry.source_faces, geometry.neumann_faces, Bi, settings.model_dump())
        ).encode()
    ).hexdigest()
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if str(data["identity"]) != identity:
                raise ValueError(f"Cached FVM identity mismatch: {path}")
            return incumbent_solution(geometry, data["field"], json.loads(str(data["metrics"])), Bi)
    result = solve(geometry, DiffusionModel(Bi=Bi), settings)
    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        identity=np.array(identity),
        mask=geometry.inside,
        field=result.concentration,
        metrics=np.array(json.dumps(asdict(result.metrics))),
    )
    _atomic(path, buffer.getvalue())
    return result


def cached_fem(path: Path, geometry_or_pslg, Bi: float, max_area: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    raster = not isinstance(geometry_or_pslg, dict)
    content = (
        geometry_or_pslg.inside.tobytes()
        + repr(
            (geometry_or_pslg.h, geometry_or_pslg.source_faces, geometry_or_pslg.neumann_faces)
        ).encode()
        if raster
        else b"".join(
            np.asarray(geometry_or_pslg[k]).tobytes()
            for k in ("vertices", "segments", "segment_markers")
        )
    )
    identity = hashlib.sha256(content + repr((Bi, max_area)).encode()).hexdigest()
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if str(data["identity"]) != identity:
                raise ValueError(f"Cached FEM identity mismatch: {path}")
            return FEMResult(
                **{
                    key: data[key].copy() if data[key].ndim else float(data[key])
                    for key in FEMResult.__dataclass_fields__
                }
            )
    result = (
        solve_raster_fem(geometry_or_pslg, Bi, max_area)
        if raster
        else solve_pslg(geometry_or_pslg, Bi, max_area)
    )
    buffer = BytesIO()
    np.savez_compressed(buffer, identity=np.array(identity), **asdict(result))
    _atomic(path, buffer.getvalue())
    return result


def run_catalog(root: Path, config, geometry_seeds, progress=print):
    rows = []
    bank = root / "geometry_bank"
    bank.mkdir(exist_ok=True)
    for index in range(len(config.evaluation_seeds)):
        for family in config.families:
            case = make_research_geometry(
                family, config.n, config.target_area, seed=geometry_seeds[family][index]
            )
            path = bank / f"{family}__{index}"
            path.mkdir(exist_ok=True)
            np.save(path / "initial_geometry.npy", case.geometry.inside)
            _json(
                path / "generator.json",
                {
                    "family": family,
                    "seed": case.seed,
                    "parameters": case.parameters,
                    "mask_sha256": hashlib.sha256(case.geometry.inside.tobytes()).hexdigest(),
                },
            )
            if index:
                continue
            for Bi in config.permeabilities:
                for factor in (1, 2, 4):
                    geometry = (
                        case.geometry if factor == 1 else refine_geometry(case.geometry, factor)
                    )
                    result = cached_fvm(
                        root / "catalog" / family / f"Bi_{Bi}_f{factor}.fvm.npz",
                        geometry,
                        Bi,
                        config.solver,
                    )
                    rows.append(
                        {"family": family, "Bi": Bi, "n": config.n * factor, **vars(result.metrics)}
                    )
            progress(f"Catalog verified: {family}", flush=True)
    write_csv(root / "catalog_metrics.csv", rows)
    return rows


def run_physical_polygon_checks(root: Path, config, progress=print):
    rows = []
    for family in ("rectangle", "hexagon", "koch_1", "koch_2"):
        pslg = chamber_pslg(family, config.target_area)
        for area in (2e-4, 5e-5, 1.25e-5):
            result = cached_fem(
                root / "polygon_validation" / family / f"a{area}.fem.npz",
                pslg,
                config.model.Bi,
                area,
            )
            rows.append(
                dict(
                    family=family,
                    solver="polygon_FEM",
                    resolution=area,
                    Q=result.Q,
                    area=result.area,
                    perimeter=result.robin_perimeter,
                    conservation_error=result.conservation_error,
                )
            )
        for n in (100, 200, 400):
            geometry = rasterize_chamber(pslg, n)
            result = cached_fvm(
                root / "polygon_validation" / family / f"n{n}.fvm.npz",
                geometry,
                config.model.Bi,
                config.solver,
            )
            rows.append(
                dict(
                    family=family,
                    solver="raster_FVM",
                    resolution=n,
                    Q=result.metrics.physical_flux_Q,
                    area=geometry.area,
                    perimeter=result.metrics.robin_perimeter,
                    conservation_error=result.metrics.conservation_error,
                )
            )
        progress(f"Physical polygon comparison: {family}", flush=True)
    write_csv(root / "physical_polygon_checks.csv", rows)
    return rows


def validate_endpoint(root: Path, config, job: dict):
    directory = root / job["path"]
    initial = make_research_geometry(
        job["family"], config.n, config.target_area, seed=job["geometry_seed"]
    ).geometry
    best = initial.with_mask(np.load(directory / "final_geometry.npy", allow_pickle=False))
    from oxygen_shape.optimization.guided_search import validate_candidate

    reason = validate_candidate(best, initial, config)
    if reason:
        raise ValueError(f"Invalid saved endpoint {directory}: {reason}")
    results = {}
    for label, geometry in (("initial", initial), ("best", best)):
        for factor in (2, 4):
            field = cached_fvm(
                directory / "validation" / f"{label}_f{factor}.fvm.npz",
                refine_geometry(geometry, factor),
                config.model.Bi,
                config.solver,
            )
            results[f"{label}_Q{config.n * factor}"] = field.metrics.physical_flux_Q
        for area in (8e-5, 2e-5):
            fem = cached_fem(
                directory / "validation" / f"{label}_a{area}.fem.npz",
                geometry,
                config.model.Bi,
                area,
            )
            results[f"{label}_FEM_{area}"] = fem.Q
            results[f"{label}_FEM_cons_{area}"] = fem.conservation_error
    row = {**job, **json.loads((directory / "summary.json").read_text()), **results}
    row["fine_gain_percent"] = 100 * (row["best_Q400"] / row["initial_Q400"] - 1)
    row["FEM_gain_percent"] = 100 * (row["best_FEM_2e-05"] / row["initial_FEM_2e-05"] - 1)
    row["membrane_change_percent"] = 100 * (row["best_perimeter"] / row["initial_perimeter"] - 1)
    return row
