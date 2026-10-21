"""Produce vector review figures from completed tables; perform no PDE solves."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from oxygen_shape.geometry.boundary import BoundaryKind, extract_boundary
from oxygen_shape.geometry.raster import OFFSETS
from oxygen_shape.geometry.research_families import make_research_geometry
from oxygen_shape.viz.history import gain_by_solve
from scripts.collar_ablation import conducting_collar
from scripts.winner_mechanisms import run as plot_selected


def run(evidence, main):
    endpoints_path = evidence / "all_returned_endpoints.csv"
    collar_path = main / "postprocess/collar_ablation/metrics.csv"
    data = pd.read_csv(endpoints_path, float_precision="round_trip")
    collar = pd.read_csv(collar_path, float_precision="round_trip")
    output = evidence / "figures"
    output.mkdir(exist_ok=True)

    def save(fig, name):
        fig.savefig(output / f"{name}.png", dpi=190, bbox_inches="tight")
        fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    bank_path = main / "bank_manifest.json"
    bank = json.loads(bank_path.read_text())
    case = next(r for r in bank["evaluation"] if r["family"] == "rectangle" and r["index"] == 0)
    geometry = make_research_geometry("rectangle", seed=case["seed"]).geometry
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), constrained_layout=True)
    styles = {
        BoundaryKind.SOURCE_DIRICHLET: ("tab:red", "Source: u=1"),
        BoundaryKind.ROBIN: ("tab:blue", "Reactive wall: du/dn + Bi*u=0"),
        BoundaryKind.NEUMANN: ("black", "Conducting-collar wall: du/dn=0"),
    }
    for ax, length in zip(axes, (0.0, 0.12), strict=True):
        labeled = conducting_collar(geometry, length)
        ax.imshow(
            np.where(labeled.inside, 0.9, np.nan),
            origin="lower",
            extent=[0, 1, 0, 1],
            cmap="Greys",
            vmin=0,
            vmax=3,
        )
        by_kind = {kind: [] for kind in styles}
        for face in extract_boundary(labeled):
            dr, dc = OFFSETS[face.key.side]
            center = (
                np.array([face.key.col + 0.5 + dc / 2, face.key.row + 0.5 + dr / 2]) * labeled.h
            )
            tangent = np.array([abs(dr), abs(dc)]) * labeled.h / 2
            by_kind[face.kind].append([center - tangent, center + tangent])
        for kind, segments in by_kind.items():
            if segments:
                ax.add_collection(LineCollection(segments, colors=styles[kind][0], lw=2.4))
        ax.axvspan(0, 0.12, color="tab:green", alpha=0.07)
        ax.set(
            title="Original wall model" if length == 0 else "Collar ablation, length 0.12",
            xlim=(-0.02, 1),
            ylim=(0, 1),
            xlabel="x/L",
            ylabel="y/L",
            aspect="equal",
        )
    fig.legend(
        handles=[
            Line2D([0], [0], color=color, lw=2, label=label) for color, label in styles.values()
        ],
        loc="outside lower center",
        ncol=1,
        fontsize=8,
    )
    fig.suptitle("Same area and source; shaded source-collar geometry is protected")
    save(fig, "boundary_conditions")

    catalog_path = main / "catalog_metrics.csv"
    catalog = pd.read_csv(catalog_path, float_precision="round_trip")
    catalog = catalog[(catalog.Bi == 1) & (catalog.n == 400)].set_index("family")
    fig, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)
    for ax, case in zip(axes.flat, [r for r in bank["evaluation"] if r["index"] == 0], strict=True):
        geometry = make_research_geometry(case["family"], seed=case["seed"]).geometry
        ax.imshow(
            np.where(geometry.inside, 1, np.nan),
            origin="lower",
            extent=[0, 1, 0, 1],
            cmap="Greys",
            vmin=0,
            vmax=1,
        )
        ax.plot([0, 0], [0.45, 0.55], color="red", lw=2)
        values = catalog.loc[case["family"]]
        ax.set(
            title=f"{case['family']}; P={values.robin_perimeter:.2f}\n"
            f"Q400={values.physical_flux_Q:.6f}",
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="x/L",
            ylabel="y/L",
            aspect="equal",
        )
        ax.tick_params(labelsize=8)
    fig.suptitle("Initial finite proxies: common area 0.24 and source 0.1; membrane lengths differ")
    save(fig, "geometry_bank")

    curve_inputs = {}

    def plot_curves(rows, families, name):
        many = len(families) > 1
        fig, axes = plt.subplots(
            3 if many else 1,
            3 if many else 1,
            figsize=(14, 10) if many else (9, 5),
            squeeze=False,
            constrained_layout=True,
        )
        colors = dict(
            uniform_greedy="grey",
            pool_greedy="tab:purple",
            flux_greedy="tab:blue",
            flux_anneal="tab:orange",
            learned_anneal="tab:green",
        )
        summaries = []
        for ax, family in zip(axes.flat, families, strict=False):
            for method in rows.method.unique():
                jobs = rows[(rows.family == family) & (rows.method == method)]
                curves = []
                for job in jobs.to_dict("records"):
                    file = Path(job["resolved_path"]) / "metrics.csv"
                    history = pd.read_csv(file, float_precision="round_trip")
                    curve = gain_by_solve(history)
                    if curve.iloc[0] != 0 or curve.index[-1] != job["pde_solves"]:
                        raise ArithmeticError("Curve lost its initial or final charged solve")
                    if not np.isclose(history.best_Q.iloc[-1], job["best_Q"], atol=1e-13, rtol=0):
                        raise ArithmeticError("Curve endpoint differs from reported coarse best Q")
                    curve_inputs[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
                    curves.append(curve)
                values = pd.concat(curves, axis=1)
                if values.isna().any().any():
                    raise ValueError("Unexpected missing charged-solve curve values")
                median = values.median(axis=1)
                q25, q75 = values.quantile(0.25, axis=1), values.quantile(0.75, axis=1)
                ax.plot(median.index, median, color=colors[method], label=method)
                ax.fill_between(median.index, q25, q75, color=colors[method], alpha=0.12)
                summaries.extend(
                    dict(
                        family=family,
                        method=method,
                        pde_solves=int(x),
                        median_gain=float(median.loc[x]),
                        q25=float(q25.loc[x]),
                        q75=float(q75.loc[x]),
                        trajectories=len(curves),
                    )
                    for x in median.index
                )
            ax.set(
                title=family, xlabel="Charged coarse PDE solves", ylabel="Best coarse Q gain (%)"
            )
            ax.grid(alpha=0.2)
        for ax in list(axes.flat)[len(families) :]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=7)
        pd.DataFrame(summaries).to_csv(output / f"{name}_summary.csv", index=False)
        save(fig, name)

    comparisons = data[data.phase == "comparison"]
    plot_curves(comparisons, list(comparisons.family.unique()), "algorithm_curves")
    original = comparisons[comparisons.study == main.name]
    plot_curves(original, list(original.family.unique()), "original_comparison_curves")
    deep = data[data.phase == "deep"]
    plot_curves(deep, list(deep.family.unique()), "long_horizon_curves")

    # Selection is among returned endpoints across explicitly unequal phases.
    top = data.sort_values("best_FEM_2e-05", ascending=False).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    y = np.arange(len(top))
    ax.hlines(y, top.best_lower, top.best_upper, color="tab:blue", lw=2.5)
    ax.plot(top["best_FEM_2e-05"], y, "o", ms=4, color="tab:orange", label="Fine FEM Q")
    ax.plot(top.best_Q400, y, "x", ms=5, color="black", label="FVM Q400")
    ax.axvline(
        data.best_lower.max(), color="tab:blue", ls="--", alpha=0.65, label="Highest lower estimate"
    )
    ax.set(
        yticks=y,
        yticklabels=[
            f"{r.family}, {r.method}\n{r.phase}, seed {int(r.seed)}" for r in top.itertuples()
        ],
        xlabel="Dimensionless total flux Q on each fixed staircase",
        title="Leading returned endpoints: practical energy ranges",
    )
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", alpha=0.2)
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "leading_flux_ranges")

    fixed = data[data.phase == "matched_perimeter"]
    families = list(fixed.family.unique())
    fig, axes = plt.subplots(
        1, len(families), figsize=(13, 4.3), sharey=True, constrained_layout=True, squeeze=False
    )
    colors = {"flux_greedy": "tab:blue", "flux_anneal": "tab:orange", "learned_anneal": "tab:green"}
    for ax, family in zip(axes.flat, families, strict=True):
        group = fixed[fixed.family == family]
        for method, color in colors.items():
            rows = group[group.method == method].sort_values("seed")
            ax.scatter(
                rows.initial_perimeter, rows.FEM_gain_percent, label=method, color=color, s=28
            )
        ax.axhline(0, color="grey", lw=0.8)
        ticks = np.unique(np.round(group.initial_perimeter.to_numpy(), 8))
        ax.set_xticks(ticks, labels=[f"{value:.2f}" for value in ticks])
        ax.set(title=family, xlabel="Unchanged per-start Robin length")
        ax.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Independent FEM gain over own start (%)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Exact-perimeter control: 65 solves; five geometry/search pairs")
    save(fig, "fixed_membrane_control")

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    initial = data.drop_duplicates(["family", "geometry_seed"])
    ax.scatter(
        initial.initial_perimeter,
        initial["initial_FEM_2e-05"],
        facecolors="none",
        edgecolors="grey",
        s=26,
        label="45 initial masks",
    )
    for phase, marker, color, label in (
        ("comparison", "o", "tab:blue", "Main / classical, 129 solves"),
        ("deep", "^", "tab:orange", "Sac-family long search, 1,025 solves"),
        ("matched_perimeter", "s", "tab:green", "Fixed per-start membrane, 65 solves"),
    ):
        rows = data[data.phase == phase]
        ax.scatter(
            rows.best_perimeter,
            rows["best_FEM_2e-05"],
            marker=marker,
            color=color,
            s=27,
            alpha=0.55,
            label=label,
        )
    # This is an empirical envelope, not a proven or interpolated feasible front.
    front, maximum = [], -np.inf
    for row in data.sort_values(
        ["best_perimeter", "best_FEM_2e-05"], ascending=[True, False]
    ).to_dict("records"):
        if row["best_FEM_2e-05"] > maximum + 1e-12:
            front.append(row)
            maximum = row["best_FEM_2e-05"]
    ax.plot(
        [r["best_perimeter"] for r in front],
        [r["best_FEM_2e-05"] for r in front],
        color="black",
        ls="--",
        lw=0.9,
        label="Observed nominal endpoint envelope",
    )
    ax.set(
        xlabel="Actual Robin membrane length",
        ylabel="Independent fine-FEM total Q",
        title="Flux and membrane resources; area 0.24, Bi=1",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    save(fig, "flux_and_membrane_resources")

    initial = collar[(collar.n == 400) & collar["case"].str.startswith("initial_")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True, constrained_layout=True)
    for ax, length in zip(axes, (0.0, 0.06, 0.12), strict=True):
        for case, group in initial[initial.collar == length].groupby("case", sort=False):
            group = group.sort_values("Bi")
            ax.plot(group.Bi, group.Q, marker="o", ms=3, label=case.removeprefix("initial_"))
        ax.set(xscale="log", xlabel="Bi", title=f"Nonabsorbing collar length {length:g}")
        if length:
            ax.axhline(
                0.1 / length, color="black", ls="--", lw=0.8, label="Conducting-neck upper bound"
            )
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Initial-proxy FVM Q400")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=8)
    fig.suptitle("Boundary-model sensitivity; fixed geometries, no reoptimization")
    save(fig, "conducting_collar_sensitivity")
    plot_selected(evidence / "geometry_review")
    selected = evidence / "geometry_review/postprocess/winner_mechanisms"
    for suffix in (".png", ".pdf"):
        shutil.copyfile(selected / ("mechanisms" + suffix), output / ("selected_geometry" + suffix))
    script = Path(__file__).resolve()
    shutil.copyfile(script, output / script.name)
    inputs = {
        str(file.resolve()): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in (
            endpoints_path,
            collar_path,
            bank_path,
            catalog_path,
            script,
            Path(__file__).with_name("collar_ablation.py"),
            Path(__file__).with_name("winner_mechanisms.py"),
            Path(__file__).resolve().parents[1] / "src/oxygen_shape/viz/history.py",
        )
    }
    inputs.update(curve_inputs)
    (output / "input_manifest.json").write_text(json.dumps(inputs, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("main", type=Path)
    args = parser.parse_args()
    run(args.evidence.resolve(), args.main.resolve())
