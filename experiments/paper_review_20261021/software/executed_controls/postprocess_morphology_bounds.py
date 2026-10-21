"""Postprocess completed morphology endpoints without performing PDE solves."""

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from flux_energy_bounds import lower_flux_estimate, primal_upper_estimate

from oxygen_shape.experiments.reporting import write_csv
from oxygen_shape.geometry.raster import refine_geometry
from oxygen_shape.geometry.research_families import make_research_geometry
from oxygen_shape.physics.model import SolverSettings


def run(root: Path):
    if json.loads((root / "status.json").read_text())["phase"] != "finished":
        raise RuntimeError("Finish search and endpoint validation before comparing flux intervals")
    endpoints = pd.read_csv(root / "endpoint_metrics.csv", float_precision="round_trip")
    config = yaml.safe_load((root / "config.yaml").read_text())
    if config["n"] != 100 or config["model"]["Bi"] != 1.0:
        raise ValueError("Postprocessor expects the frozen n=100, Bi=1 morphology protocol")
    settings = SolverSettings.model_validate(config["solver"])
    output = root / "postprocess/energy_bounds"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    input_hashes = {}
    for job in endpoints.to_dict("records"):
        g0 = make_research_geometry(job["family"], seed=int(job["geometry_seed"])).geometry
        directory = root / job["path"]
        best = g0.with_mask(np.load(directory / "final_geometry.npy", allow_pickle=False))
        row = {key: job[key] for key in ("phase", "family", "seed", "method", "path")}
        row["mask_sha256"] = hashlib.sha256(best.inside.tobytes()).hexdigest()
        for label, geometry in (("initial", g0), ("best", best)):
            lower_estimates, upper_estimates = [], []
            for factor in (2, 4):
                refined = refine_geometry(geometry, factor)
                file = directory / "validation" / f"{label}_f{factor}.fvm.npz"
                with np.load(file, allow_pickle=False) as data:
                    expected = hashlib.sha256(
                        refined.inside.tobytes()
                        + repr(
                            (
                                refined.h,
                                refined.source_faces,
                                refined.neumann_faces,
                                1.0,
                                settings.model_dump(),
                            )
                        ).encode()
                    ).hexdigest()
                    if str(data["identity"]) != expected:
                        raise ValueError("FVM input identity mismatch in bound postprocessing")
                    if not np.array_equal(data["mask"], refined.inside):
                        raise ValueError("Bound field mask does not match declared geometry")
                    estimate = lower_flux_estimate(refined, data["field"], 1.0)
                    lower_estimates.append((estimate.lower, factor, estimate))
                input_hashes[file.relative_to(root).as_posix()] = hashlib.sha256(
                    file.read_bytes()
                ).hexdigest()
            for area in (8e-5, 2e-5):
                file = directory / "validation" / f"{label}_a{area}.fem.npz"
                with np.load(file, allow_pickle=False) as data:
                    content = (
                        geometry.inside.tobytes()
                        + repr((geometry.h, geometry.source_faces, geometry.neumann_faces)).encode()
                    )
                    expected = hashlib.sha256(content + repr((1.0, area)).encode()).hexdigest()
                    if str(data["identity"]) != expected:
                        raise ValueError("FEM input identity mismatch in bound postprocessing")
                    upper, energy, guard = primal_upper_estimate(
                        data["vertices"],
                        data["triangles"],
                        data["concentration"],
                        data["boundary_segments"],
                        data["boundary_markers"],
                        1.0,
                    )
                    if abs(energy - float(data["energy"])) > 1e-9:
                        raise ArithmeticError(
                            "Independent primal integration disagrees with FEM matrix"
                        )
                    if abs(float(data["area"]) - geometry.area) > 1e-10:
                        raise ValueError("FEM domain area does not match raster domain")
                    upper_estimates.append((upper, area, guard))
                input_hashes[file.relative_to(root).as_posix()] = hashlib.sha256(
                    file.read_bytes()
                ).hexdigest()
            low, factor, estimate = max(lower_estimates, key=lambda x: x[0])
            high, area, guard = min(upper_estimates, key=lambda x: x[0])
            if low > high:
                raise ArithmeticError(
                    f"Inconsistent variational estimates for {job['path']}: {low=} {high=}"
                )
            row.update(
                {
                    f"{label}_lower": low,
                    f"{label}_upper": high,
                    f"{label}_relative_width_percent": 100 * (high - low) / max(abs(low), 1e-14),
                    f"{label}_lower_grid": factor * 100,
                    f"{label}_upper_mesh_area": area,
                    f"{label}_upper_guard": guard,
                    **{f"{label}_{k}": v for k, v in asdict(estimate).items()},
                }
            )
        row["resolved_improvement"] = row["best_lower"] > row["initial_upper"]
        row["minimum_relative_gain_percent"] = 100 * (row["best_lower"] / row["initial_upper"] - 1)
        rows.append(row)
    write_csv(output / "endpoint_bounds.csv", rows)
    frame = pd.DataFrame(rows)
    paired = []
    for phase in frame.phase.unique():
        group = frame[frame.phase == phase]
        control = group[group.method == "flux_greedy"].set_index(["family", "seed"])
        for method in group.method.unique():
            current = group[group.method == method].set_index(["family", "seed"])
            for key, row in current.iterrows():
                reference = control.loc[key]
                decision = (
                    "resolved_win"
                    if row.best_lower > reference.best_upper
                    else "resolved_loss"
                    if row.best_upper < reference.best_lower
                    else "overlap"
                )
                paired.append(
                    dict(
                        phase=phase,
                        family=key[0],
                        seed=int(key[1]),
                        method=method,
                        outcome=decision,
                        minimum_delta_Q=row.best_lower - reference.best_upper,
                        maximum_delta_Q=row.best_upper - reference.best_lower,
                    )
                )
    write_csv(output / "paired_bounds.csv", paired)
    selected = frame.loc[frame.best_lower.idxmax()]
    contenders = frame[frame.best_upper >= selected.best_lower]
    summary = {
        "PDE_solves": 0,
        "endpoints": len(frame),
        "maximum_relative_width_percent": float(frame.best_relative_width_percent.max()),
        "median_relative_width_percent": float(frame.best_relative_width_percent.median()),
        "resolved_improvements": int(frame.resolved_improvement.sum()),
        "selected_path": selected.path,
        "selected_lower": float(selected.best_lower),
        "selected_upper": float(selected.best_upper),
        "contender_endpoints": len(contenders),
        "contender_distinct_masks": int(contenders.mask_sha256.nunique()),
        "interpretation": (
            "Variational estimates on the same staircase domain, with practical "
            "floating-point guards; not interval-arithmetic certification"
        ),
        "paired_counts": pd.DataFrame(paired)
        .groupby(["phase", "method", "outcome"])
        .size()
        .rename("count")
        .reset_index()
        .to_dict("records"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    sources = [
        Path(__file__).resolve(),
        Path(__file__).with_name("flux_energy_bounds.py").resolve(),
    ]
    sources.append(Path(__file__).resolve().parents[1] / "docs/complementary_flux_bounds.md")
    hashes = {}
    for source in sources:
        shutil.copyfile(source, output / source.name)
        hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    (output / "source_manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")
    for name in (
        "config.yaml",
        "endpoint_metrics.csv",
        "bank_manifest.json",
        "source_manifest.json",
    ):
        input_hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    (output / "input_manifest.json").write_text(json.dumps(input_hashes, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    run(parser.parse_args().root)
