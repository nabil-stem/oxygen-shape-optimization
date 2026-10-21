"""Scientific plots and explicit comparison limits for the deletion experiment."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from oxygen_shape.experiments.deletion_config import DeletionConfig


def plot_effects(rows: list[dict], output: Path) -> None:
    cases = list(dict.fromkeys(r["case"] for r in rows))
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), layout="constrained")
    for ax, mode in zip(axes, ("deletion", "paired"), strict=True):
        for offset, group in ((-0.15, "uniform"), (0.15, "low_flux")):
            first = True
            for index, case in enumerate(cases):
                subset = [
                    r
                    for r in rows
                    if r["case"] == case and r["mode"] == mode and r["group"] == group
                ]
                values = [100 * r["delta_Q"] / r["initial_Q"] for r in subset]
                if not values:
                    continue
                # Plot jitter is deterministic and unrelated to scientific sampling.
                ax.scatter(
                    index + offset + np.linspace(-0.07, 0.07, len(values)),
                    values,
                    s=22,
                    alpha=0.65,
                    color="C0" if group == "uniform" else "C1",
                    label=group if first else None,
                )
                ax.plot(
                    [index + offset - 0.09, index + offset + 0.09],
                    [np.median(values)] * 2,
                    color="black",
                    linewidth=2,
                )
                first = False
        ax.axhline(0, color="gray", linewidth=1)
        ax.set_xticks(range(len(cases)), [case.replace("_Bi_", "\nBi=") for case in cases])
        ax.set_ylabel("Change in total Q (%)")
        ax.set_title(
            "Deletion alone: bounded area loss"
            if mode == "deletion"
            else "Paired cut + addition: fixed area"
        )
        if cases:
            ax.legend()
        ax.grid(axis="y", alpha=0.2)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def write_report(
    directory: Path, config: DeletionConfig, rows: list[dict], quotas: list[dict], summary: dict
) -> None:
    lines = [
        "# Does removing low-flux wall improve global oxygen transfer?",
        "",
        f"Status: **{summary['status']}**. Seed: {config.seed}. "
        f"Candidate PDE evaluations: {len(rows)}; total PDE solves including baseline/refinement: "
        f"{summary['number_of_pde_solves']}.",
        "",
        "## Protocol",
        "",
        f"Eligible Robin faces exclude the source collar x<{config.protected_length} "
        "and a two-cell margin. Uniform donors are sampled by face measure; low-flux donors "
        f"are sampled uniformly from the bottom {100 * config.low_flux_fraction:g}% "
        "of initial j=Bi*u_wall, including quantile ties. "
        "Both use the same uniform addition sampler. The first valid, unique donor cuts "
        "are evaluated; candidates are never selected for their measured Q.",
        "",
        "Each pair is also evaluated as a deletion alone using the identical cut. A pair enters "
        "the sample only if both variants satisfy their declared constraints. This conditions "
        "results on joint feasibility and can change the distribution of accepted donor faces. "
        "All geometry proposals and rejection reasons are retained in proposals.csv.",
        "",
        "Shared constraints: unit bounding box, source length 0.1 and location fixed, "
        "source collar fixed, face connectivity, no holes, "
        f"Robin perimeter <= {config.perimeter_limit}, "
        f"morphology feature width {config.minimum_feature_width}, tested on n={config.n} "
        "and refinement factors "
        f"{config.refinement_factors}. Paired edits fix area {config.target_area} within 1e-12. "
        "Deletion alone permits area in "
        f"[{config.target_area * (1 - config.max_removed_area_fraction):g}, "
        f"{config.target_area:g}); perimeter may change within its budget.",
        "",
        "Donor groups do not have exactly matched removed areas or membrane lengths. The paired "
        "protocol is the fixed-area comparison. Pure deletion is a local intervention study, "
        "not a fixed-resource shape ranking. Every candidate is globally re-solved; local "
        "concentration/flux is never treated as a shape derivative.",
        "",
        "## All evaluated candidates",
        "",
        "Positive/negative counts use the configured absolute delta-Q tolerance. Medians include "
        "every evaluated candidate, including losses. No globally optimal geometry is claimed.",
        "",
        "| Case | Donor | Mode | N | Gains/losses | Median delta Q | "
        "Median eta | Median area | Median Robin P |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for quota in quotas:
        for mode in ("deletion", "paired"):
            subset = [
                r
                for r in rows
                if r["case"] == quota["case"] and r["group"] == quota["group"] and r["mode"] == mode
            ]
            if not subset:
                continue
            positive = sum(r["delta_Q"] > config.effect_tolerance for r in subset)
            negative = sum(r["delta_Q"] < -config.effect_tolerance for r in subset)
            lines.append(
                f"| {quota['case']} | {quota['group']} | {mode} | "
                f"{len(subset)} | {positive}/{negative} | "
                f"{np.median([r['delta_Q'] for r in subset]):.7g} | "
                f"{np.median([r['efficiency_eta'] for r in subset]):.6g} | "
                f"{np.median([r['area'] for r in subset]):.6g} | "
                f"{np.median([r['robin_perimeter'] for r in subset]):.6g} |"
            )
    lines += [
        "",
        "![Changes in global flux](figures/deletion_effects.png)",
        "",
        "## Quotas and sampling costs",
        "",
        "| Case | Donor | Valid pairs / requested | Attempts | Eligible / low-flux faces |",
        "|---|---|---:|---:|---:|",
    ]
    for q in quotas:
        lines.append(
            f"| {q['case']} | {q['group']} | {q['collected_pairs']}/{q['requested_pairs']} | "
            f"{q['attempts']} | {q['eligible_donor_faces']}/{q['low_flux_donor_faces']} |"
        )
    lines += [
        "",
        "## Refinement of selected examples",
        "",
        "The largest positive and negative coarse effects in each case/donor/mode are "
        "selected descriptively and solved at every refinement factor. Refinement keeps "
        "the same physical staircase mask. Observed sensitivity is the change between "
        "the last two delta-Q estimates, not a rigorous continuum error bound. An effect "
        "is labelled resolved only if its sign survives and its finest magnitude exceeds "
        "that sensitivity plus the effect tolerance. Unselected candidates have no such label.",
        "",
        "| Case | Sample | Coarse delta Q | Fine delta Q | Observed sensitivity | Resolved |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        if "fine_delta_Q" in row:
            lines.append(
                f"| {row['case']} | {row['sample_id']} | {row['delta_Q']:.7g} | "
                f"{row['fine_delta_Q']:.7g} | {row['observed_grid_sensitivity']:.4g} | "
                f"{row['effect_resolved']} |"
            )
    lines += [
        "",
        "## Interpretation",
        "",
        f"Low-flux deletion alone: {summary['low_flux_deletion_positive']} increases and "
        f"{summary['low_flux_deletion_negative']} decreases on the coarse grid; "
        f"{summary['resolved_low_flux_deletion_positive']} positive and "
        f"{summary['resolved_low_flux_deletion_negative']} negative selected examples pass the "
        "empirical refinement criterion. Missing signs are reported as missing, not inferred.",
        "",
        "Deleting air introduces a new Robin cut surface and can change access, membrane "
        "measure, and concentrations everywhere. Q and eta measure different things; neither "
        "the donor's initial j nor a change in eta determines the sign of delta Q. A positive "
        "paired move combines cutting and reallocation, so it cannot be attributed to the "
        "cut alone. These small, single-seed exploratory samples do not establish statistical "
        "superiority of a donor strategy.",
        "",
        f"Maximum conservation error across all solves: {summary['max_conservation_error']:.3e}. "
        "metrics.csv contains Q, Q_ideal, eta, area, Robin perimeter, donor features, actual "
        "disk parameters and diagnostics. refinement.csv contains every finer-grid estimate. "
        "cases/ retains every candidate mask and coarse field; root masks are the unedited "
        "first baseline, as labelled in summary.json. Config, tolerances, source snapshots, "
        "Git state and dependency versions are saved.",
        "",
    ]
    field_plots = sorted((directory / "figures").glob("*_fields.png"))
    if field_plots:
        lines += ["## Representative fields (first case)", ""]
        for path in field_plots:
            lines += [f"![{path.stem}](figures/{path.name})", ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
