"""Controlled square-island maps, maze layouts, interactions and paid relocation."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from oxygen_shape.experiments.config import load_config
from oxygen_shape.experiments.holes_config import HolesConfig
from oxygen_shape.experiments.holes_evidence import HoleEvidence, require_frozen_sources
from oxygen_shape.experiments.reporting import create_run, provenance, write_csv
from oxygen_shape.geometry.boundary import BoundaryKind
from oxygen_shape.geometry.raster import refine_geometry
from oxygen_shape.geometry.research_families import make_research_geometry
from oxygen_shape.geometry.square_holes import (
    SquareHole,
    boundary_hole_ids,
    hole_boundary,
    maze_chamber,
    perforate,
)
from oxygen_shape.optimization.hole_relocation import run_hole_relocation
from oxygen_shape.optimization.trajectory_search import _json


def geometry_for(family, config):
    return (
        maze_chamber(config.n)
        if family == "maze_chamber"
        else make_research_geometry(family, config.n, config.target_area, seed=config.seed).geometry
    )


def hole_kwargs(config):
    return dict(
        gap=config.gap,
        protected_length=config.protected_length,
        minimum_feature_width=config.minimum_feature_width,
    )


def position_roster(outer, width, config):
    valid, rejected = [], []
    stride = round(config.position_stride / outer.h)
    for row in range(0, outer.inside.shape[0] - width + 1, stride):
        for col in range(0, outer.inside.shape[1] - width + 1, stride):
            hole = SquareHole(row, col, width)
            try:
                perforate(outer, (hole,), **hole_kwargs(config))
            except ValueError as error:
                rejected.append(dict(**asdict(hole), reason=str(error)))
            else:
                valid.append(hole)
    return valid, rejected


def declared_layouts(n):
    """Outcome-blind square positions; six-square layouts exactly match A and P."""

    def squares(points):
        return tuple(SquareHole(round(y * n), round(x * n), round(0.08 * n)) for x, y in points)

    spread = squares([(x, y) for y in (0.32, 0.60) for x in (0.28, 0.50, 0.72)])
    near = squares([(0.40, 0.46), (0.52, 0.46)])
    far = squares([(0.28, 0.46), (0.72, 0.46)])
    return {
        "count_1": spread[:1],
        "count_2": spread[:2],
        "count_4": spread[:4],
        "spread_6": spread,
        "proximal_6": squares([(x, y) for y in (0.36, 0.52) for x in (0.24, 0.36, 0.48)]),
        "distal_6": squares([(x, y) for y in (0.36, 0.52) for x in (0.52, 0.64, 0.76)]),
        "staggered_6": squares([(x, y) for y in (0.26, 0.46, 0.66) for x in (0.36, 0.64)]),
        "near_pair": near,
        "near_single_1": near[:1],
        "near_single_2": near[1:],
        "far_pair": far,
        "far_single_1": far[:1],
        "far_single_2": far[1:],
    }


def paired_metrics(outer_Q, reactive, reflecting, geometry):
    qr, qn = reactive.metrics.physical_flux_Q, reflecting.metrics.physical_flux_Q
    tolerance = 1e-9 * max(1.0, outer_Q, qr, qn)
    if qn > outer_Q + tolerance or qn > qr + tolerance:
        raise ArithmeticError("Hole energy monotonicity failed: check labels/geometry/solver")
    ids = boundary_hole_ids(geometry)
    kinds = np.array([f.kind == BoundaryKind.ROBIN for f in reactive.faces])
    inner = float(reactive.outward_face_flux[ids > 0].sum())
    exterior = float(reactive.outward_face_flux[(ids == 0) & kinds].sum())
    if abs(inner + exterior - qr) > tolerance:
        raise ArithmeticError("Inner/exterior Robin flux partition does not conserve Q")
    return dict(
        Q0=outer_Q,
        Q_R=qr,
        Q_N=qn,
        delta_Q=qr - outer_Q,
        obstruction_delta=qn - outer_Q,
        activation_delta=qr - qn,
        inner_Q=inner,
        hole_count=int(ids.max()),
        euler_characteristic=1 - int(ids.max()),
        outer_Q=exterior,
        outer_delta=exterior - outer_Q,
        gain_percent=100 * (qr / outer_Q - 1),
        area=geometry.area,
        robin_perimeter=reactive.metrics.robin_perimeter,
        reflecting_robin_perimeter=reflecting.metrics.robin_perimeter,
        eta=reactive.metrics.efficiency_eta,
        Q_ideal=reactive.metrics.ideal_flux_Q,
        reflecting_eta=reflecting.metrics.efficiency_eta,
        reflecting_Q_ideal=reflecting.metrics.ideal_flux_Q,
        source_influx=reactive.metrics.source_influx,
        residual_inf=max(reactive.metrics.residual_inf, reflecting.metrics.residual_inf),
        conservation_error=max(
            reactive.metrics.conservation_error, reflecting.metrics.conservation_error
        ),
        source_length=reactive.metrics.source_length,
    )


def run_holes_study(config: HolesConfig, resume: Path | None = None, progress=print):
    root = create_run(config, "square_holes") if resume is None else resume.resolve()
    identity = config.model_dump(mode="json")
    identity_path = root / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("Square-study config changed; use a new experiment directory")
    elif resume is not None:
        raise ValueError("Not a square-hole experiment directory")
    else:
        _json(identity_path, identity)
        _json(root / "provenance.json", provenance())
    require_frozen_sources(root)
    evidence = HoleEvidence(root, config)
    _json(root / "status.json", dict(status="running"))
    cases, map_rows, rosters, per_hole = {}, [], [], []
    paired_cache, validation_selection = {}, set()

    def register(key, family, holes, phase):
        outer = geometry_for(family, config)
        geometry = perforate(outer, holes, **hole_kwargs(config))
        cases[key] = (outer, geometry)
        case_dir = root / "cases" / key
        case_dir.mkdir(parents=True, exist_ok=True)
        np.save(case_dir / "outer.npy", outer.inside)
        np.save(case_dir / "geometry.npy", geometry.inside)
        _json(
            case_dir / "geometry.json",
            dict(
                family=family,
                phase=phase,
                h=geometry.h,
                holes=[asdict(h) for h in holes],
                source_faces=[
                    dict(row=f.row, col=f.col, side=f.side.value) for f in geometry.source_faces
                ],
                mask_sha256=hashlib.sha256(geometry.inside.tobytes()).hexdigest(),
            ),
        )
        return geometry

    def pair(key, Bi, factor=1):
        query = (key, Bi, factor)
        if query in paired_cache:
            return paired_cache[query]
        outer, geometry = cases[key]
        refined = geometry if factor == 1 else refine_geometry(geometry, factor)
        outer_fine = outer if factor == 1 else refine_geometry(outer, factor)
        family = json.loads((root / "cases" / key / "geometry.json").read_text())["family"]
        baseline = evidence.get(f"base/{family}_B{Bi}_f{factor}", outer_fine, Bi)
        reactive = evidence.get(f"{key}_B{Bi}_f{factor}_R", refined, Bi)
        reflecting = evidence.get(f"{key}_B{Bi}_f{factor}_N", hole_boundary(refined, "neumann"), Bi)
        row = dict(
            case=key,
            Bi=Bi,
            n=config.n * factor,
            baseline_area=baseline.metrics.area,
            baseline_robin_perimeter=baseline.metrics.robin_perimeter,
            **paired_metrics(baseline.metrics.physical_flux_Q, reactive, reflecting, refined),
        )
        paired_cache[query] = row
        if factor == 1:
            ids = boundary_hole_ids(refined)
            for component in range(1, int(ids.max()) + 1):
                per_hole.append(
                    dict(
                        case=key,
                        Bi=Bi,
                        hole=component,
                        Q=float(reactive.outward_face_flux[ids == component].sum()),
                        perimeter=float((ids == component).sum() * refined.h),
                    )
                )
        return row

    for family in config.families:
        outer = geometry_for(family, config)
        for width in config.widths:
            cells = round(width * config.n)
            positions, rejected = position_roster(outer, cells, config)
            rosters.append(
                dict(
                    family=family,
                    width=width,
                    eligible=len(positions),
                    rejected=len(rejected),
                    status="eligible" if positions else "no_fit",
                )
            )
            _json(
                root / f"roster_{family}_w{cells}.json",
                dict(eligible=[asdict(h) for h in positions], rejected=rejected),
            )
            keys = []
            for i, hole in enumerate(positions):
                key = f"map/{family}_w{cells}_{i:03d}"
                register(key, family, (hole,), "single_square_map")
                keys.append(key)
                for Bi in config.permeabilities:
                    map_rows.append(
                        dict(
                            family=family, width=width, row=hole.row, col=hole.col, **pair(key, Bi)
                        )
                    )
            for Bi in config.validation_Bi:
                subset = [r for r in map_rows if r["case"] in keys and r["Bi"] == Bi]
                if subset:
                    for chosen in (
                        min(subset, key=lambda r: r["delta_Q"]),
                        max(subset, key=lambda r: r["delta_Q"]),
                    ):
                        validation_selection.add((chosen["case"], Bi))
            progress(f"Square map: {family}, width={width:g}, {len(positions)} valid positions")
    write_csv(root / "roster_summary.csv", rosters)
    if map_rows:
        write_csv(root / "position_maps.csv", map_rows)

    layouts = declared_layouts(config.n)
    layout_rows = []
    for name, holes in layouts.items():
        key = f"layout/{name}"
        register(key, "maze_chamber", holes, "controlled_layout")
        for Bi in config.permeabilities:
            layout_rows.append(dict(layout=name, number_of_holes=len(holes), **pair(key, Bi)))
        validation_selection.update((key, Bi) for Bi in config.validation_Bi)
    write_csv(root / "layout_metrics.csv", layout_rows)
    interactions = []
    for name in ("near", "far"):
        for Bi in config.permeabilities:
            one, two, both = (
                pair(f"layout/{name}_{suffix}", Bi) for suffix in ("single_1", "single_2", "pair")
            )
            ir = both["Q_R"] - one["Q_R"] - two["Q_R"] + both["Q0"]
            inn = both["Q_N"] - one["Q_N"] - two["Q_N"] + both["Q0"]
            interactions.append(
                dict(
                    arrangement=name,
                    Bi=Bi,
                    reactive_interaction=ir,
                    reflecting_interaction=inn,
                    activation_interaction=ir - inn,
                )
            )
    write_csv(root / "interactions.csv", interactions)
    progress("Count, equal-resource layouts and pair interactions complete")

    search_rows = []
    outer = maze_chamber(config.n)
    positions, _ = position_roster(outer, round(0.08 * config.n), config)
    for seed in config.search_seeds:
        for method in ("uniform_greedy", "concentration_greedy"):
            key = f"search/{method}_{seed}"
            holes, summary = run_hole_relocation(
                config, outer, layouts["spread_6"], positions, method, seed, evidence, key
            )
            search_rows.append(summary)
            register(key, "maze_chamber", holes, "fixed_resource_relocation")
            pair(key, config.search_Bi)
            validation_selection.add((key, config.search_Bi))
            progress(f"Square relocation: {method}, seed={seed}, Q={summary['final_Q']:.7f}")
    if search_rows:
        write_csv(root / "search_summary.csv", search_rows)

    validation = []
    _json(
        root / "validation_selection.json",
        [dict(case=k, Bi=b) for k, b in sorted(validation_selection)],
    )
    for key, Bi in sorted(validation_selection):
        for factor in config.refinement_factors:
            validation.append(dict(solver="FVM", mesh_area=None, **pair(key, Bi, factor)))
        outer, geometry = cases[key]
        for area in config.fem_max_areas:
            results = {}
            for mode, g in (
                ("base", outer),
                ("R", geometry),
                ("N", hole_boundary(geometry, "neumann")),
            ):
                cache_key = (
                    f"base/maze_chamber_B{Bi}_a{area}"
                    if mode == "base" and key.startswith(("layout/", "search/"))
                    else f"{key}_B{Bi}_a{area}_{mode}"
                )
                results[mode] = evidence.get(cache_key, g, Bi, fem_area=area)
            r, nn, base = results["R"], results["N"], results["base"]
            validation.append(
                dict(
                    case=key,
                    Bi=Bi,
                    n=None,
                    solver="FEM",
                    mesh_area=area,
                    Q0=base.Q,
                    Q_R=r.Q,
                    Q_N=nn.Q,
                    delta_Q=r.Q - base.Q,
                    obstruction_delta=nn.Q - base.Q,
                    activation_delta=r.Q - nn.Q,
                    area=r.area,
                    robin_perimeter=r.robin_perimeter,
                    eta=r.Q / (Bi * r.robin_perimeter),
                    conservation_error=max(r.conservation_error, nn.conservation_error),
                )
            )
        progress(f"Independent refinement: {key}, Bi={Bi:g}")
    write_csv(root / "validation.csv", validation)
    if per_hole:
        write_csv(root / "per_hole_flux.csv", per_hole)
    require_frozen_sources(root)
    summary = dict(
        status="complete",
        seed=config.seed,
        n=config.n,
        scope="Square excluded islands; no anatomical optimality or RL claim",
        position_cases=len(map_rows),
        controlled_layouts=len(layouts),
        search_runs=len(search_rows),
        validation_rows=len(validation),
        **evidence.accounting(),
    )
    _json(root / "summary.json", summary)
    from oxygen_shape.experiments.holes_report import write_holes_report

    write_holes_report(root)
    _json(root / "status.json", dict(status="complete"))
    progress(f"Completed square-hole study: {root}")
    return root, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    run_holes_study(load_config(args.config, HolesConfig), args.resume)


if __name__ == "__main__":
    main()
