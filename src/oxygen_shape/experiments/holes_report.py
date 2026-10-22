"""Saved-evidence figures and a qualified report; this module performs no solve."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, PowerNorm


def write_holes_report(root: Path) -> None:
    summary = json.loads((root / "summary.json").read_text())
    config = json.loads((root / "identity.json").read_text())
    layouts = pd.read_csv(root / "layout_metrics.csv", float_precision="round_trip")
    validation = pd.read_csv(root / "validation.csv", float_precision="round_trip")
    figures = root / "figures"

    def save(fig, name):
        fig.savefig(figures / f"{name}.png", dpi=160)
        fig.savefig(figures / f"{name}.pdf")
        plt.close(fig)

    bi = config["search_Bi"]
    available = layouts[layouts.Bi == bi]
    if available.empty:
        bi = float(layouts.Bi.iloc[0])
        available = layouts[layouts.Bi == bi]
    six = available[available.layout.isin(["spread_6", "proximal_6", "distal_6", "staggered_6"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    x = np.arange(len(six))
    axes[0].bar(x, six.obstruction_delta, label="Reflecting obstacle effect", color="#b64d4d")
    axes[0].bar(x, six.activation_delta, label="Activating inner Robin walls", color="#4477aa")
    axes[0].scatter(x, six.delta_Q, marker="D", color="black", label="Net change", zorder=3)
    axes[0].set_xticks(x, six.layout.str.replace("_6", ""))
    axes[0].axhline(0, color="grey", linewidth=0.7)
    axes[0].set(
        title=f"Six equal squares; fixed air area and membrane, Bi={bi:g}", ylabel="Change in Q"
    )
    axes[0].legend(fontsize=7)
    counts = layouts[layouts.layout.isin(["count_1", "count_2", "count_4", "spread_6"])]
    for permeability, group in counts.groupby("Bi", sort=True):
        group = group.sort_values("number_of_holes")
        axes[1].plot(group.number_of_holes, group.gain_percent, "o-", label=f"Bi={permeability:g}")
    axes[1].set(
        xlabel="Square count (area and membrane change)",
        ylabel="Total Q change (%)",
        title="Declared sequence; not a fixed-resource comparison",
    )
    axes[1].axhline(0, color="grey", linewidth=0.7)
    axes[1].legend(fontsize=8)
    save(fig, "hole_effects")

    labels = ["baseline", "spread_6", "proximal_6", "distal_6", "staggered_6"]
    selected_search = None
    if (root / "search_summary.csv").exists():
        searches = pd.read_csv(root / "search_summary.csv", float_precision="round_trip")
        best = searches.sort_values(["final_Q", "method", "seed"]).iloc[-1]
        selected_search = f"search/{best.method}_{best.seed}"
        labels.append("selected search")
        fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
        for method, group in searches.groupby("method"):
            curves = []
            for row in group.itertuples():
                history = json.loads(
                    (root / f"search/{method}_{row.seed}/history.json").read_text()
                )
                q = np.array([r["Q"] for r in history])
                q = np.pad(q, (0, config["search_budget"] - len(q)), mode="edge")
                curves.append(100 * (q / q[0] - 1))
            curves = np.asarray(curves)
            solves = np.arange(curves.shape[1]) + 1
            low, median, high = np.quantile(curves, [0.25, 0.5, 0.75], axis=0)
            ax.plot(solves, median, label=method)
            ax.fill_between(solves, low, high, alpha=0.18)
        ax.set(
            xlabel="Charged FVM solves, including initial state",
            ylabel="Flux gain (%)",
            title="Relocating six squares: medians and IQR across search seeds",
        )
        ax.legend()
        save(fig, "hole_relocation")

    for mode in ("linear", "power"):
        fig, axes = plt.subplots(2, 3, figsize=(11, 7.4), constrained_layout=True)
        image = None
        for ax, name in zip(axes.flat, labels, strict=False):
            key = "layout/" + name if name != "selected search" else selected_search
            if name == "baseline":
                path = root / f"cache/base/maze_chamber_B{bi}_f1.fvm.npz"
            else:
                path = root / f"cache/{key}_B{bi}_f1_R.fvm.npz"
            with np.load(path, allow_pickle=False) as data:
                field = data["field"]
                metrics = json.loads(str(data["metrics"]))
            norm = Normalize(0, 1) if mode == "linear" else PowerNorm(0.35, vmin=0, vmax=1)
            image = ax.imshow(field, origin="lower", extent=(0, 1, 0, 1), cmap="viridis", norm=norm)
            ax.plot([0, 0], [0.45, 0.55], color="red", linewidth=3)
            ax.set(
                title=f"{name}; Q={metrics['physical_flux_Q']:.5f}\n"
                f"A={metrics['area']:.4f}, P_R={metrics['robin_perimeter']:.2f}",
                xlabel="x",
                ylabel="y",
            )
        for ax in list(axes.flat)[len(labels) :]:
            ax.set_visible(False)
        if image is not None:
            fig.colorbar(image, ax=list(axes.flat), shrink=0.75, label="Concentration u")
        fig.suptitle(
            f"Bi={bi:g}; source red; empty squares excluded from air. "
            + ("Shared linear scale" if mode == "linear" else "Shared power scale, gamma=0.35")
        )
        save(fig, f"square_fields_{mode}")

    if (root / "position_maps.csv").exists():
        maps = pd.read_csv(root / "position_maps.csv", float_precision="round_trip")
        maps = maps[(maps.family == "maze_chamber") & (maps.width == max(config["widths"]))]
        values = sorted(maps.Bi.unique())
        if len(values) > 3:
            values = [values[0], values[len(values) // 2], values[-1]]
        if values:
            fig, axes = plt.subplots(
                1,
                len(values),
                figsize=(4 * len(values), 3.7),
                constrained_layout=True,
                squeeze=False,
            )
            limit = max(float(maps.gain_percent.abs().max()), 1e-12)
            for ax, permeability in zip(axes.flat, values, strict=True):
                group = maps[maps.Bi == permeability]
                image = ax.scatter(
                    (group.col / config["n"] + group.width / 2),
                    (group.row / config["n"] + group.width / 2),
                    c=group.gain_percent,
                    marker="s",
                    s=130,
                    cmap="coolwarm",
                    vmin=-limit,
                    vmax=limit,
                )
                ax.set(
                    title=f"Bi={permeability:g}",
                    xlabel="Square center x",
                    ylabel="Square center y",
                    aspect="equal",
                    xlim=(0, 1),
                    ylim=(0, 1),
                )
            fig.colorbar(
                image, ax=list(axes.flat), label="Total Q change from unperforated chamber (%)"
            )
            fig.suptitle("Same square area and perimeter at every eligible position")
            save(fig, "square_position_maps")

    fem = validation[validation.solver == "FEM"]
    lines = [
        "# Square-island study",
        "",
        "Exploratory 2D results; separate from v0.1.0.",
        "",
        "An empty square is excluded material, not a source or an inter-alveolar pore.",
        "Exterior source/wall labels are fixed. Maps compare positions at the same square size;",
        "six-square layouts and relocation hold air area and Robin length fixed.",
        "Count curves change both resources and must not be interpreted as a free topology gain.",
        "",
        f"Committed FVM solves: {summary['committed_fvm_solves']}; "
        f"FEM solves: {summary['committed_fem_solves']}.",
        "Possible uncommitted work upper bound: "
        f"{summary['possible_uncommitted_solve_upper_bound']}.",
        "",
        f"## Equal-resource six-square layouts, Bi={bi:g}",
        "",
        "| Layout | Q0 | Q_R | Q_N | Net change (%) | Inner-wall Q | Air area | Robin length |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in six.itertuples():
        lines.append(
            f"| {r.layout} | {r.Q0:.7f} | {r.Q_R:.7f} | {r.Q_N:.7f} | "
            f"{r.gain_percent:.4f} | {r.inner_Q:.7f} | {r.area:.5f} | {r.robin_perimeter:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Refinement on the identical polygons",
            "",
            "| Case | Q at finest FVM | Q at finest FEM | Finest FEM net change (%) |",
            "|---|---:|---:|---:|",
        ]
    )
    reported_cases = ["layout/" + name for name in six.layout]
    if selected_search:
        reported_cases.append(selected_search)
    for key in reported_cases:
        checked = validation[(validation["case"] == key) & (validation.Bi == bi)]
        fv = checked[checked.solver == "FVM"].sort_values("n")
        fe = checked[checked.solver == "FEM"].sort_values("mesh_area", ascending=False)
        if len(fv) and len(fe):
            last_fv, last_fe = fv.iloc[-1], fe.iloc[-1]
            gain = 100 * (last_fe.Q_R / last_fe.Q0 - 1)
            lines.append(f"| {key} | {last_fv.Q_R:.7f} | {last_fe.Q_R:.7f} | {gain:.4f} |")
    if selected_search:
        lines.extend(
            [
                "",
                "The displayed search endpoint is selected post hoc among returned",
                "coarse-grid endpoints, with additional search effort relative to static layouts.",
            ]
        )
    roster = pd.read_csv(root / "roster_summary.csv")
    lines.extend(
        [
            "",
            "## Placement feasibility",
            "",
            "| Family | Square width | Eligible positions | Rejected positions |",
            "|---|---:|---:|---:|",
        ]
    )
    for r in roster.itertuples():
        lines.append(f"| {r.family} | {r.width:g} | {r.eligible} | {r.rejected} |")
    lines.extend(
        [
            "",
            "![Maze square fields](figures/square_fields_linear.png)",
            "",
            "![Flux attribution](figures/hole_effects.png)",
            "",
            "## Numerical and interpretation limits",
            "",
            f"Refinement rows: {len(validation)}; independent FEM rows: {len(fem)}.",
            "Selected map extrema are chosen on the coarse grid; validation_selection.json records",
            "that selection. This does not refine every placement or prove global rankings.",
            "FVM subdivisions and FEM solve the same staircase exterior and exact square holes.",
            "No smooth anatomical boundary, 3D pore, convection or blood-side coupling is modeled.",
            "Search curves use one starting maze and multiple search seeds; IQR is variability,",
            "not a confidence interval. The proposal heuristic is not RL or a shape derivative.",
            "No-fit families, rejected placements and paid rejected candidates are retained.",
            "The obstruction/activation decomposition is a controlled comparison; inner uptake",
            "alone is not the net gain because concentration on the exterior wall also changes.",
            "",
            "Close prior work includes [Freus & Freus 2015](https://doi.org/10.17512/jamcm.2015.2.01)",
            "and [Grebenkov & Traytak 2019](https://doi.org/10.1016/j.jcp.2018.10.033).",
            "No first-study or biological-optimality claim follows from these computations.",
            "",
        ]
    )
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
