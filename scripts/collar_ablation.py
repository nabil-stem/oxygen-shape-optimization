"""Fixed-geometry sensitivity to a nonabsorbing proximal conducting collar.

Run only after the morphology study finishes. The search itself is unchanged.
The collar is a model hypothesis, not a measured anatomical length.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from oxygen_shape.experiments.morphology_validation import cached_fem, cached_fvm
from oxygen_shape.experiments.reporting import write_csv
from oxygen_shape.geometry.boundary import BoundaryKind, extract_boundary
from oxygen_shape.geometry.raster import OFFSETS, RasterGeometry, refine_geometry
from oxygen_shape.geometry.research_families import make_research_geometry
from oxygen_shape.physics.model import SolverSettings


def conducting_collar(geometry, length):
    if not np.isfinite(length) or length < 0:
        raise ValueError("Collar length must be finite and nonnegative")
    impermeable = set(geometry.neumann_faces)
    for face in extract_boundary(geometry):
        _, dx = OFFSETS[face.key.side]
        x = (face.key.col + 0.5 + dx / 2) * geometry.h
        if face.kind == BoundaryKind.ROBIN and x < length - 1e-12:
            impermeable.add(face.key)
    return RasterGeometry(
        geometry.inside, geometry.h, geometry.source_faces, tuple(sorted(impermeable))
    )


def write_report(output, data, fem_data, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finest = data[data.n == 400]
    initial = finest[finest["case"].str.startswith("initial_")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, constrained_layout=True)
    for ax, length in zip(axes, (0.0, 0.06, 0.12), strict=True):
        for label, group in initial[initial.collar == length].groupby("case", sort=False):
            group = group.sort_values("Bi")
            ax.loglog(group.Bi, group.Q, marker="o", label=label.removeprefix("initial_"))
        ax.set(title=f"Nonabsorbing collar length {length}", xlabel="Bi")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Total Q at n=400")
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.savefig(output / "initial_flux_by_Bi.png", dpi=160)
    plt.close(fig)
    comparisons = summary["initial_family_comparisons"]
    table = "\n".join(
        f"| {row['Bi']:g} | {row['collar']:g} | {row['best_case'].removeprefix('initial_')} "
        f"| {row['maximum_Q']:.6f} | {row['relative_spread_percent']:.3f}% |"
        for row in comparisons
    )
    coarse = data[data.n == 200].set_index(["case", "collar", "Bi"])
    fine = finest.set_index(["case", "collar", "Bi"])
    refinement = 100 * abs(fine.Q / coarse.Q - 1)
    fv = finest[finest.Bi == 1].set_index(["case", "collar"])
    fem = fem_data[fem_data.max_area == 2e-5].set_index(["case", "collar"])
    disagreement = 100 * abs(fv.Q / fem.Q - 1)
    text = f"""# Where absorption begins: fixed-geometry model sensitivity

The original search uses absorbing Robin walls immediately adjacent to the source.
This separate sensitivity study makes only the proximal collar nonabsorbing,
with distal Robin absorption unchanged. Source value and physical masks are fixed.
Collar lengths 0.06 and 0.12 are model hypotheses, not anatomical measurements.
The nine initial family representatives and five distinct high-FEM endpoints
are reused without reoptimization. Endpoint selection is post-hoc.

![Flux under three boundary rules](initial_flux_by_Bi.png)

| Bi | Nonabsorbing collar | Highest initial proxy | Q400 | Initial-family spread |
|---:|---:|---|---:|---:|
{table}

Spread is 100*(max Q-min Q)/max Q for the nine fixed initial representatives.
Their actual membrane lengths differ. This is neither a matched-membrane ranking
nor an uncertainty or biological variation estimate. Reactive perimeter is recorded
for every case; across collar rules its length changes. No geometry is retrained
under the new boundary conditions.

Maximum n200-to-n400 relative flux change: {refinement.max():.3f}%.
Maximum Q400-to-fine-FEM difference at Bi=1: {disagreement.max():.3f}%.
These observed differences are diagnostics, not rigorous error bars. Every
nonabsorbing-neck solve also satisfies Q<=0.1/collar_length. All concentrations
and face labels are retained in the field caches or defined by case_manifest.json.

Additional evaluations cost {summary["additional_FVM_solves"]} FVM and
{summary["additional_FEM_solves"]} FEM solves; {summary["reused_main_FVM_fields"]}
FVM and {summary["reused_main_FEM_fields"]} FEM fields are reused from the main study.
Search cost is unchanged. Input hashes and this script are archived separately.
The frozen search conclusions concern its original boundary model. This ablation
tests sensitivity of selected shapes; it cannot identify the best reoptimized
shape under a conducting-collar model or establish biological optimality.
"""
    (output / "report.md").write_text(text, encoding="utf-8")


def run(root):
    if json.loads((root / "status.json").read_text())["phase"] != "finished":
        raise RuntimeError("Complete the original search before selecting ablation cases")
    output = root / "postprocess/collar_ablation"
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((root / "config.yaml").read_text())
    if config["n"] != 100 or config["model"]["Bi"] != 1.0:
        raise ValueError("Expected the frozen n=100, Bi=1 morphology protocol")
    settings = SolverSettings.model_validate(config["solver"])
    manifest = json.loads((root / "bank_manifest.json").read_text())
    endpoints = pd.read_csv(root / "endpoint_metrics.csv", float_precision="round_trip")
    cases = []
    seen = set()
    for row in manifest["evaluation"]:
        if row["index"]:
            continue
        g = make_research_geometry(row["family"], seed=row["seed"]).geometry
        cases.append((f"initial_{row['family']}", g, {"kind": "initial", **row}))
        seen.add(hashlib.sha256(g.inside.tobytes()).hexdigest())
    for row in endpoints.sort_values("best_FEM_2e-05", ascending=False).to_dict("records"):
        g = make_research_geometry(row["family"], seed=int(row["geometry_seed"])).geometry
        g = g.with_mask(np.load(root / row["path"] / "final_geometry.npy", allow_pickle=False))
        digest = hashlib.sha256(g.inside.tobytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        cases.append(
            (
                f"endpoint_{len(cases) - len(config['families'])}",
                g,
                {
                    "kind": "selected_endpoint",
                    "mask_sha256": digest,
                    **{
                        key: row[key]
                        for key in ("path", "family", "phase", "method", "seed", "geometry_seed")
                    },
                },
            )
        )
        if len(cases) == len(config["families"]) + 5:
            break
    rows, fem_rows, sources = [], [], []
    inputs, reused_fvm, reused_fem = {}, set(), set()

    def record(path):
        relative = path.relative_to(root).as_posix()
        inputs[relative] = hashlib.sha256(path.read_bytes()).hexdigest()

    for index, (label, geometry, provenance) in enumerate(cases):
        sources.append({"case": label, **provenance})
        if provenance["kind"] == "initial":
            matching = endpoints[
                (endpoints.family == provenance["family"])
                & (endpoints.geometry_seed == provenance["seed"])
            ]
            validation = root / matching.iloc[0]["path"] / "validation"
            original_label = "initial"
        else:
            validation = root / provenance["path"] / "validation"
            original_label = "best"
        for length in (0.0, 0.06, 0.12):
            changed = conducting_collar(geometry, length)
            collar_name = f"c{length}"
            for Bi in (0.1, 1.0, 10.0, 100.0):
                for factor in (2, 4):
                    fine = refine_geometry(changed, factor)
                    path = (
                        output
                        / "fields"
                        / f"{index:02d}"
                        / collar_name
                        / f"Bi{Bi}_f{factor}.fvm.npz"
                    )
                    if length == 0:
                        if provenance["kind"] == "initial":
                            path = (
                                root
                                / "catalog"
                                / provenance["family"]
                                / f"Bi_{Bi}_f{factor}.fvm.npz"
                            )
                        elif Bi == 1:
                            path = validation / f"{original_label}_f{factor}.fvm.npz"
                        if path.is_relative_to(root) and not path.is_relative_to(output):
                            if not path.exists():
                                raise FileNotFoundError(path)
                            reused_fvm.add(path.relative_to(root).as_posix())
                    field = cached_fvm(path, fine, Bi, settings)
                    record(path)
                    q = field.metrics.physical_flux_Q
                    # Common unreactive rectangular source neck: average u is
                    # linear in x, so Q <= source_width/collar_length.
                    neck_bound = 0.1 / length if length else None
                    if neck_bound is not None and q > neck_bound + 1e-9:
                        raise ArithmeticError("Conducting-neck analytical upper bound violated")
                    rows.append(
                        dict(
                            case=label,
                            index=index,
                            collar=length,
                            Bi=Bi,
                            n=100 * factor,
                            Q=q,
                            robin_perimeter=field.metrics.robin_perimeter,
                            eta=field.metrics.efficiency_eta,
                            conservation_error=field.metrics.conservation_error,
                            neck_bound=neck_bound,
                        )
                    )
            for max_area in (8e-5, 2e-5):
                path = output / "fields" / f"{index:02d}" / collar_name / f"a{max_area}.fem.npz"
                if length == 0:
                    path = validation / f"{original_label}_a{max_area}.fem.npz"
                    if not path.exists():
                        raise FileNotFoundError(path)
                    reused_fem.add(path.relative_to(root).as_posix())
                fem = cached_fem(path, changed, 1.0, max_area)
                record(path)
                fem_rows.append(
                    dict(
                        case=label,
                        collar=length,
                        max_area=max_area,
                        Q=fem.Q,
                        conservation_error=fem.conservation_error,
                    )
                )
        print(f"Collar ablation: {index + 1}/{len(cases)} {label}", flush=True)
    write_csv(output / "metrics.csv", rows)
    write_csv(output / "fem_metrics.csv", fem_rows)
    (output / "case_manifest.json").write_text(json.dumps(sources, indent=2) + "\n")
    data = pd.DataFrame(rows)
    finest = data[data.n == 400]
    initial = finest[finest["case"].str.startswith("initial_")]
    contrasts = []
    for Bi in (0.1, 1.0, 10.0, 100.0):
        for length in (0.0, 0.06, 0.12):
            group = initial[(initial.Bi == Bi) & (initial.collar == length)].sort_values(
                "Q", ascending=False
            )
            contrasts.append(
                dict(
                    Bi=Bi,
                    collar=length,
                    best_case=group.iloc[0]["case"],
                    minimum_Q=float(group.Q.min()),
                    maximum_Q=float(group.Q.max()),
                    relative_spread_percent=float(
                        100 * (group.Q.max() - group.Q.min()) / group.Q.max()
                    ),
                )
            )
    write_csv(output / "initial_family_rankings.csv", contrasts)
    summary = {
        "cases": len(cases),
        "additional_FVM_solves": len(list(output.rglob("*.fvm.npz"))),
        "additional_FEM_solves": len(list(output.rglob("*.fem.npz"))),
        "reused_main_FVM_fields": len(reused_fvm),
        "reused_main_FEM_fields": len(reused_fem),
        "FVM_evaluations": len(rows),
        "FEM_evaluations": len(fem_rows),
        "search_PDE_solves": 0,
        "initial_family_comparisons": contrasts,
        "interpretation": (
            "Fixed-geometry boundary-condition sensitivity; "
            "no reoptimization or anatomical calibration"
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    source = Path(__file__).resolve()
    shutil.copyfile(source, output / source.name)
    (output / "source_manifest.json").write_text(
        json.dumps({source.name: hashlib.sha256(source.read_bytes()).hexdigest()}, indent=2) + "\n"
    )
    for name in (
        "config.yaml",
        "bank_manifest.json",
        "endpoint_metrics.csv",
        "source_manifest.json",
    ):
        record(root / name)
    (output / "input_manifest.json").write_text(json.dumps(inputs, indent=2) + "\n")
    write_report(output, data, pd.DataFrame(fem_rows), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    run(parser.parse_args().root)
