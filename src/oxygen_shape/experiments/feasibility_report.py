"""Explicit raw-draw accounting and matched complete-seed comparisons."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from oxygen_shape.experiments.feasibility_config import FeasibilityConfig


def write_report(
    root: Path,
    config: FeasibilityConfig,
    summary: dict,
    audit: list[dict],
    raw: list[dict],
    evaluations: list[dict],
    histories: list[dict],
) -> None:
    a, r, e, h = map(pd.DataFrame, (audit, raw, evaluations, histories))
    lines = [
        "# Geometry-only proposal screening study",
        "",
        f"Status: **{summary['status']}**.",
        "",
        "The audit compares selectors on shared pools. All raw proposals receive full validation",
        "only for auditing; the selector sees area, collar and perimeter checks, never those",
        "full-validation outcomes or candidate PDE responses. Search validates only its selected",
        "candidate and accepts solely by global Q. Physical constraints remain unchanged.",
        "",
        "## Fixed-geometry audit",
        "",
        "| Case | Method | Selected valid / checks | Raw-draw cost | Valid / raw cost | "
        "Distinct selected addition/donor faces |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (case, method), g in a.groupby(["case", "method"], sort=False):
        valid, cost = int(g.valid.sum()), int(g.raw_cost.sum())
        lines.append(
            f"| {case} | {method} | {valid}/{len(g)} | {cost} | {valid / cost:.4f} | "
            f"{g.addition_face.nunique()}/{g.donor_face.nunique()} |"
        )
    lines += [
        "",
        "Distinct face counts are descriptive coverage, not proof of equal distributions.",
        "The 25% uniform branch retains the original proposal support. Eight-draw selection",
        "changes the candidate distribution and costs eight raw draws even when seven are",
        "not selected. audit_raw.csv records every raw outcome and generation/validation time.",
        "",
        "| Case | All raw valid / draws | Duplicate valid masks within replicate |",
        "|---|---:|---:|",
    ]
    for case, g in r.groupby("case", sort=False):
        valid = g[g.valid]
        duplicates = int(valid.duplicated(["replicate", "geometry_hash"]).sum())
        lines.append(f"| {case} | {len(valid)}/{len(g)} | {duplicates} |")
    if config.audit_response_solves:
        lines += [
            "",
            "## Shared-pool response ablation",
            "",
            "All selectors choose before full validation and response evaluation. Both pool",
            "rules use the same per-pool selection seed, coupling their uniform branch.",
            "Each unique valid selected mask is solved once per pool; shared solution IDs",
            "avoid double charging. Invalid choices have no delta-Q value. Retrospective",
            "one-step greedy gain is zero for invalid or non-improving choices, otherwise",
            "the actual global delta-Q. It is not a sequential optimization result.",
            "",
            "| Case | Method | Mean gain per choice: median [Q1, Q3] across replicates |",
            "|---|---|---:|",
        ]
        for (case, method), g in a.groupby(["case", "method"], sort=False):
            means = g.groupby("replicate").one_step_gain_Q.mean()
            values = means.quantile([0.25, 0.5, 0.75]).values
            lines.append(
                f"| {case} | {method} | {values[1]:.7g} [{values[0]:.7g}, {values[2]:.7g}] |"
            )
        lines += [
            "",
            "| Case | Screened minus minimum: median replicate mean gain | "
            "Replicate wins / ties / losses | Both-valid pools / all pools |",
            "|---|---:|---:|---:|",
        ]
        for case, g in a.groupby("case", sort=False):
            means = g.groupby(["replicate", "method"]).one_step_gain_Q.mean().unstack("method")
            delta = means.screened_uniform - means.perimeter_pool
            paired = g.pivot(index=["replicate", "pool"], columns="method", values="valid")
            valid_count = int((paired.screened_uniform & paired.perimeter_pool).sum())
            lines.append(
                f"| {case} | {delta.median():.7g} | "
                f"{int((delta > config.effect_tolerance).sum())} / "
                f"{int((delta.abs() <= config.effect_tolerance).sum())} / "
                f"{int((delta < -config.effect_tolerance).sum())} | {valid_count}/{len(paired)} |"
            )
        lines += [
            "",
            "Audit response metrics refer to the coarse discrete problem. Extreme distinct",
            "both-valid contrasts in each direction are checked at configured refinement",
            "factors in audit_contrasts.csv; unavailable directions remain explicit.",
            "Replicates share fixed inputs and are not independent optimized geometries.",
        ]
    lines += [
        "",
        "## Frozen equal-PDE-budget search",
        "",
        "Each run is capped at the configured solve count including its initial solve,",
        "and a separate strict raw-draw budget. All selected constraint checks, unselected",
        "draws and candidate solves are charged. Incomplete runs are retained.",
        "",
        "| Case | Method | Complete / planned | Median Q | Median eta | Median raw draws | "
        "Median search seconds | Median fine gain Q |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (case, method), g in e.groupby(["case", "method"], sort=False):
        c = g[g.status == "complete"]
        values = [
            f"{c[col].median():.7g}" if len(c) else "undefined"
            for col in ("final_Q", "final_eta", "raw_draws", "search_seconds", "fine_gain_Q")
        ]
        lines.append(
            f"| {case} | {method} | {len(c)}/{len(config.search_seeds)} | "
            + " | ".join(values)
            + " |"
        )
    lines += [
        "",
        "| Case | Method vs control | Complete paired seeds | Wins coarse / fine | "
        "Ties coarse / fine | Method faster |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for case, g in e.groupby("case", sort=False):
        complete = g[g.status == "complete"]
        comparisons = [(m, "uniform") for m in config.methods if m != "uniform"]
        if "screened_uniform" in config.methods:
            comparisons.append(("screened_uniform", "perimeter_pool"))
        for method, control in comparisons:
            pair = complete[complete.method.isin([method, control])]
            if not {method, control}.issubset(pair.method):
                continue
            wide = pair.pivot(
                index="seed", columns="method", values=["final_Q", "fine_final_Q", "search_seconds"]
            ).dropna()
            differences = [
                wide[(c, method)] - wide[(c, control)] for c in ("final_Q", "fine_final_Q")
            ]
            wins = [int((d > config.effect_tolerance).sum()) for d in differences]
            ties = [int((d.abs() <= config.effect_tolerance).sum()) for d in differences]
            faster = int(
                (wide[("search_seconds", method)] < wide[("search_seconds", control)]).sum()
            )
            lines.append(
                f"| {case} | {method} vs {control} | {len(wide)} | {wins[0]} / {wins[1]} | "
                f"{ties[0]} / {ties[1]} | {faster} |"
            )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    audit_means = a.groupby(["case", "method"]).valid.mean().unstack("method")
    audit_sd = (
        a.groupby(["case", "method", "replicate"])
        .valid.mean()
        .groupby(["case", "method"])
        .std()
        .unstack("method")
        .fillna(0)
    )
    audit_means.plot.bar(ax=axes[0], rot=25, yerr=audit_sd, capsize=3)
    axes[0].set_ylabel("Selected full-valid fraction")
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("")
    for (case, method), g in h.groupby(["case", "method"]):
        grouped = g.groupby("number_of_pde_solves").physical_flux_Q
        curve = grouped.median()
        (line,) = axes[1].plot(curve.index, curve.values, label=f"{case}: {method}")
        axes[1].fill_between(
            curve.index,
            grouped.quantile(0.25).values,
            grouped.quantile(0.75).values,
            color=line.get_color(),
            alpha=0.12,
        )
    axes[1].set_xlabel("PDE solves including initial")
    axes[1].set_ylabel("Median incumbent Q (available runs)")
    axes[1].legend(fontsize=7)
    fig.savefig(root / "figures/comparison.png", dpi=160)
    plt.close(fig)
    lines += [
        "",
        "![Selected feasibility and search histories](figures/comparison.png)",
        "Audit bars show mean selected yield with replicate SD; search curves show median Q "
        "with interquartile bands. These are variability summaries, not confidence intervals. "
        f"Paired Q ties use absolute tolerance {config.effect_tolerance:g}.",
        "",
        f"PDE solves: {summary['number_of_pde_solves']} = "
        f"{summary['baseline_pde_solves']} input checks + {summary['search_pde_solves']} search + "
        f"{summary['refinement_pde_solves']} search refinement + "
        f"{summary.get('audit_pde_solves', 0)} shared audit responses + "
        f"{summary.get('audit_refinement_pde_solves', 0)} audit contrast refinement.",
        f"Maximum conservation error: {summary['max_conservation_error']:.4g}.",
        "",
        "Timing is exploratory single-machine elapsed search time, including raw generation,",
        "selection, validation, PDE solves and candidate storage; execution order rotates by seed.",
        "Fine validation repeats the same staircase domain at configured factors; it is not a",
        "continuum error bound. One frozen input per family and five seeds limit generality.",
        "A selected-yield improvement alone is not evidence of saved time or improved Q.",
        "No existing sampler, predictor, RL checkpoint or physical model is promoted or changed.",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
