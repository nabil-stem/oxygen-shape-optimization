"""Plot completed initial-bank sweeps, explicitly showing unequal membrane amounts."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def run(root):
    source = root / "catalog_metrics.csv"
    data = pd.read_csv(source, float_precision="round_trip")
    fine = data[data.n == 400].set_index(["family", "Bi"])
    medium = data[data.n == 200].set_index(["family", "Bi"])
    change = 100 * abs(fine.physical_flux_Q / medium.physical_flux_Q - 1)
    output = root / "postprocess/catalog_review"
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for family in data.family.unique():
        group = fine.loc[family].sort_index()
        label = f"{family} (P={group.robin_perimeter.iloc[0]:.2f})"
        axes[0].loglog(group.index, group.physical_flux_Q, marker="o", label=label)
        axes[1].semilogx(group.index, 100 * group.efficiency_eta, marker="o")
        axes[2].loglog(group.index, change.loc[family].loc[group.index], marker="o")
    axes[0].set(ylabel="Total Q400", title="Total oxygen transfer")
    axes[1].set(ylabel="Wall utilization eta (%)", ylim=(0, 100), title="Diagnostic utilization")
    axes[2].set(ylabel="100*abs(Q400/Q200-1) (%)", title="Observed grid change")
    for ax in axes:
        ax.set_xlabel("Bi")
        ax.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle("Nine initial proxies: fixed area 0.24 and source, actual membrane lengths differ")
    fig.tight_layout(rect=(0, 0.17, 1, 0.95))
    fig.savefig(output / "permeability_sweep.png", dpi=180)
    fig.savefig(output / "permeability_sweep.pdf")
    plt.close(fig)
    comparisons = []
    for Bi, group in fine.reset_index().groupby("Bi"):
        best = group.loc[group.physical_flux_Q.idxmax()]
        comparisons.append(
            dict(
                Bi=Bi,
                highest_initial_proxy=best.family,
                highest_Q400=float(best.physical_flux_Q),
                spread_percent=100
                * float(
                    (group.physical_flux_Q.max() - group.physical_flux_Q.min())
                    / group.physical_flux_Q.max()
                ),
                maximum_grid_change_percent=float(change.xs(Bi, level="Bi").max()),
            )
        )
    table = "\n".join(
        f"| {row['Bi']:g} | {row['highest_initial_proxy']} | {row['highest_Q400']:.6f} "
        f"| {row['spread_percent']:.4f}% | {row['maximum_grid_change_percent']:.4f}% |"
        for row in comparisons
    )
    text = f"""# Initial geometry bank: permeability, membrane and grid diagnostics

![Completed initial-bank sweep](permeability_sweep.png)

All nine representatives have the same source, area 0.24, box and perimeter cap
8, but different actual membrane lengths. This figure does not establish a
matched-perimeter ranking. Only one representative per family is swept; it
does not estimate anatomical variability or the optimized-family outcome.
Total Q is the objective. Eta remains a diagnostic.

| Bi | Highest initial proxy | Q400 | Across-family spread | Maximum grid change |
|---:|---|---:|---:|---:|
{table}

Spread is 100*(max Q-min Q)/max Q. Grid change uses Q200 as reference and is an
observed discretization diagnostic, not a rigorous error bar. In particular,
small high-Bi family differences require care when grid changes are much larger.
The source-touching absorbing corners can dominate high-Bi transfer; the
independent analytical strip and conducting-collar study test that issue.
The low-Bi spread also reflects unequal membrane amounts and cannot by itself
refute a fixed-perimeter asymptotic prediction. No new PDE solve is performed.
"""
    (output / "report.md").write_text(text, encoding="utf-8")
    script = Path(__file__).resolve()
    shutil.copyfile(script, output / script.name)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "additional_PDE_solves": 0,
                "comparisons": comparisons,
                "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    print(output / "report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    run(parser.parse_args().root)
