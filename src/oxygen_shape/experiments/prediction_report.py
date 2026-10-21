"""Replicate-aware predictive diagnostics; no optimizer changes are selected."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from oxygen_shape.experiments.prediction_config import PredictionConfig
from oxygen_shape.optimization.prediction import SCORE_NAMES


def median_text(rows: list[dict], field: str) -> str:
    values = [r[field] for r in rows if r[field] is not None]
    return f"{np.median(values):.6g}" if values else "undefined"


def plot_diagnostic(
    root: Path, config: PredictionConfig, scores: list[dict], attempts: list[dict]
) -> None:
    fig, axes = plt.subplots(
        len(config.case_labels),
        2,
        figsize=(12, 4 * len(config.case_labels)),
        layout="constrained",
        squeeze=False,
    )
    for row_axes, shape in zip(axes, config.case_labels, strict=True):
        for name in (*SCORE_NAMES, "flux_control"):
            x, y = [], []
            for count in config.calibration_sizes:
                values = [
                    r["spearman_delta_Q"]
                    for r in scores
                    if r["shape"] == shape
                    and r["score"] == name
                    and r["radius_stratum"] == "all"
                    and r["calibration_count"] == count
                    and r["spearman_delta_Q"] is not None
                ]
                if values:
                    x.append(count)
                    y.append(float(np.median(values)))
            if x:
                row_axes[0].plot(x, y, "o-", label=name)
        row_axes[0].axhline(0, color="gray", linewidth=1)
        row_axes[0].set(
            title=f"{shape}: median held-out rank correlation",
            xlabel="Valid calibration PDE solves",
            ylabel="Spearman rank vs global delta-Q",
            ylim=(-1, 1),
        )
        if row_axes[0].get_legend_handles_labels()[0]:
            row_axes[0].legend(fontsize=8)
        subset = [a for a in attempts if a["shape"] == shape and a["role"] == "radius"]
        fractions = [
            np.mean([a["valid"] for a in subset if a["radius"] == radius])
            for radius in config.matched_radii
        ]
        row_axes[1].plot(config.matched_radii, fractions, "o-")
        row_axes[1].set(
            title=f"{shape}: independent matched-radius families",
            xlabel="Addition radius",
            ylabel="Geometrically feasible fraction",
            ylim=(-0.02, 1.02),
        )
        for ax in row_axes:
            ax.grid(alpha=0.2)
    fig.savefig(root / "figures" / "predictive_diagnostic.png", dpi=160)
    plt.close(fig)


def write_report(
    root: Path,
    config: PredictionConfig,
    scores: list[dict],
    attempts: list[dict],
    outcomes: list[dict],
    quotas: list[dict],
    summary: dict,
) -> None:
    maximum = config.calibration_sizes[-1]
    lines = [
        "# Fixed-geometry predictive diagnostic",
        "",
        f"Status: **{summary['status']}**.",
        "",
        f"The geometry stays fixed within every family. Bi={config.model.Bi:g}; n={config.n}; "
        f"{config.replicates} replicates per fixed case. Each replicate uses up to {maximum} valid "
        f"calibration edits and {config.heldout_samples} held-out edits from disjoint seed "
        "streams and mask sets. No candidate is accepted and no optimizer is modified. "
        "All edits use the original area/source/topology/footprint/perimeter/feature constraints.",
        "",
        f"Spatial bandwidth is {config.spatial_bandwidth:g}; radius bandwidth is "
        f"{config.radius_bandwidth:g}. Four ablations and the flux ranking control are "
        "prespecified. Raw signed kernel sums are clipped below at zero for ranking, "
        "matching proposal-probability order on a fixed geometry. They are not calibrated "
        "delta-Q forecasts. Scores are compared with actual global delta-Q and delta-Q per "
        "reallocated area. Top-quartile selection uses fractional tie weights. Constant "
        "correlations are undefined and excluded from the median, with their count shown. "
        "Sign accuracy excludes actual sub-tolerance effects and is compared with an "
        "always-positive baseline. Prefixes reuse one test pool and are not independent tests.",
        "",
        "## Full-calibration held-out ranking (median across replicates)",
        "",
        "| Fixed case | Score | Defined rho / available / planned | "
        "Rho vs delta-Q | Rho vs delta-Q/area | "
        "Top mean delta-Q | Top minus pool mean | Sign accuracy | Always-positive accuracy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for shape in config.case_labels:
        for name in (*SCORE_NAMES, "flux_control"):
            group = [
                r
                for r in scores
                if r["shape"] == shape
                and r["score"] == name
                and r["calibration_count"] == maximum
                and r["radius_stratum"] == "all"
            ]
            if not group:
                lines.append(
                    f"| {shape} | {name} | 0/0/{config.replicates} | undefined | "
                    "undefined | undefined | undefined | undefined | undefined |"
                )
                continue
            defined = sum(r["spearman_delta_Q"] is not None for r in group)
            lines.append(
                f"| {shape} | {name} | {defined}/{len(group)}/{config.replicates} | "
                f"{median_text(group, 'spearman_delta_Q')} | "
                f"{median_text(group, 'spearman_sensitivity')} | "
                f"{median_text(group, 'top_mean_delta_Q')} | "
                f"{median_text(group, 'top_gain_over_pool_mean')} | "
                f"{median_text(group, 'sign_accuracy')} | "
                f"{median_text(group, 'always_positive_accuracy')} |"
            )
    lines += [
        "",
        "Medians are computed per replicate, rather than pooling all test edits as "
        "independent replicates. Scores.csv also contains every prefix, per-replicate "
        "metric and small/large radius stratum. Sign accuracy alone can be misleading if "
        "the valid pool contains mostly improvements. A positive top-minus-pool statistic "
        "is a retrospective ranking diagnostic, not an achieved optimization speedup.",
        "",
        "![Predictive ranking and radius feasibility](figures/predictive_diagnostic.png)",
        "",
        "## Prefix-size and radius-stratified rank correlations",
        "",
        "| Shape | Score | Calibration solves | All-radius rho | "
        "Small-radius rho | Large-radius rho |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for shape in config.case_labels:
        for name in SCORE_NAMES:
            for count in config.calibration_sizes:
                group = [
                    r
                    for r in scores
                    if r["shape"] == shape
                    and r["score"] == name
                    and r["calibration_count"] == count
                ]
                if group:
                    values = [
                        median_text(
                            [r for r in group if r["radius_stratum"] == stratum], "spearman_delta_Q"
                        )
                        for stratum in ("all", "small", "large")
                    ]
                    lines.append(f"| {shape} | {name} | {count} | " + " | ".join(values) + " |")
    lines += [
        "",
        "## Independent matched-radius families",
        "",
        f"Each of {config.radius_families} uniform face-pair families per replicate reuses "
        "the same angular/jitter RNG stream at every prespecified radius. Families are "
        "drawn independently of calibration/test feasibility and Q. Every failed variant "
        "remains in the denominator; no family is chosen because a cut looks useful.",
        "",
        "| Shape | Radius | Feasible / attempted | Median delta-Q among feasible |",
        "|---|---:|---:|---:|",
    ]
    for shape in config.case_labels:
        for radius in config.matched_radii:
            draws = [
                a
                for a in attempts
                if a["shape"] == shape and a["role"] == "radius" and a["radius"] == radius
            ]
            valid = [
                r
                for r in outcomes
                if r["shape"] == shape and r["role"] == "radius" and r["radius"] == radius
            ]
            lines.append(
                f"| {shape} | {radius:g} | {sum(a['valid'] for a in draws)}/{len(draws)} | "
                f"{median_text(valid, 'delta_Q')} |"
            )
    families = {}
    for draw in attempts:
        if draw["role"] == "radius":
            families.setdefault((draw["shape"], draw["replicate"], draw["family_id"]), []).append(
                draw
            )
    mixed = sum(
        any(a["valid"] for a in group) and not all(a["valid"] for a in group)
        for group in families.values()
    )
    all_valid = sum(all(a["valid"] for a in group) for group in families.values())
    lines += [
        "",
        f"Families feasible at every radius: {all_valid}/{len(families)}. "
        f"Families with both feasible and infeasible radii: {mixed}/{len(families)}. "
        "Variant feasibility and observed gains must be interpreted separately.",
        "",
        "## Quotas, cost and numerical limits",
        "",
        "| Shape | Replicate | Pool | Collected / requested | Attempts |",
        "|---|---:|---|---:|---:|",
    ]
    for q in quotas:
        lines.append(
            f"| {q['shape']} | {q['replicate']} | {q['role']} | "
            f"{q['collected']}/{q['requested']} | {q['attempts']} |"
        )
    lines += [
        "",
        f"Total PDE solves: {summary['number_of_pde_solves']}; baseline "
        f"{summary['baseline_pde_solves']}, calibration {summary['calibration_pde_solves']}, "
        f"held-out {summary['heldout_pde_solves']}, radius {summary['radius_pde_solves']}, "
        f"refinement {summary['refinement_pde_solves']}. All calibration/test costs are "
        "reported; evaluating every test candidate is diagnostic work, not a saved solve.",
        "",
        f"Maximum conservation error: {summary['max_conservation_error']:.3e}. "
        "The largest positive and negative held-out effects per shape are refined at "
        "the configured factors as descriptive checks. Predictive metrics otherwise "
        "refer to the coarse discrete problem. Refining a staircase is not a certified "
        "smooth-boundary error bound. Feasible-pool conditioning, small samples and one "
        "Bi limit generality. No score or hyperparameter is promoted using these held-out results.",
        "",
        "All attempted pairs and proposal seeds are in attempts.csv, including invalid "
        "geometries. Metrics.csv retains every valid edit's mask hash, features, Q, ideal Q, "
        "eta, area, Robin perimeter and conservation. Predictions.csv contains signed scores "
        "and true responses; cases/ contains full masks/fields; refinement.csv contains "
        "numerical checks. Root geometry files are the unchanged first baseline. Config, "
        "solver tolerances, Git state, source snapshot and package versions are retained.",
        "",
    ]
    lines += [
        "## Fixed inputs and provenance",
        "",
        "| Fixed case | Family | State | Q | Area | Robin perimeter | Fraction of budget |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for fixed in summary["fixed_cases"]:
        lines.append(
            f"| {fixed['shape']} | {fixed['family']} | {fixed['reference_state']} | "
            f"{fixed['Q']:.8g} | {fixed['area']:.6g} | {fixed['robin_perimeter']:.6g} | "
            f"{fixed['perimeter_fraction']:.5g} |"
        )
    lines += [
        "",
        "input_manifest.json identifies each reference, its verified SHA-256 hashes "
        "and its copied input_snapshots/. Frozen inputs must match the recorded "
        "physics, grid, source, constraints and recomputed flux. Controls are regenerated "
        "from the canonical family. Every input stays fixed during sampling. Equal seed "
        "labels across cases do not imply identical physical edits. Prefix statistics "
        "are conditional on attaining enough valid calibration/test data; available and "
        "planned replicate counts and all exhausted quotas remain visible. Root geometry "
        "files belong to the first fixed case.",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
