"""Training readiness, full costs, and paired final-replica evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def write_pioneer_report(
    root: Path, config, summary: dict, bank: dict, training: list[dict], evaluations: list[dict]
) -> None:
    t, e = pd.DataFrame(training), pd.DataFrame(evaluations)
    e["gain_Q"] = e.final_Q - e.initial_Q
    lines = [
        "# Multi-start RL training and controls",
        "",
        f"Status: **{summary['status']}**.",
        "",
        "Five control types and every final training replica are evaluated separately.",
        "No model selection or hyperparameter tuning uses validation or test results.",
        "The physical objective remains globally solved Q under the original constraints.",
        f"Refill quota={config.refill_quota}, raw cap={config.raw_draw_budget}, "
        f"per-pool cap={config.refill_draw_limit}, fixed flux prior={config.flux_prior_scale}.",
        "The raw uniform control draws one proposal per decision; all pooled controls and "
        "learners share the same refill rule. Untrained is the zero-score uniform pool "
        "control; flux_pool is the frozen flux-prior control. Compare workflow time as well "
        "as PDE budgets; refilling spends additional geometry work.",
        "",
        "## Starting bank",
        "",
        "| Split | Number of distinct masks |",
        "|---|---:|",
    ]
    for split in ("train", "validation", "test"):
        lines.append(f"| {split} | {sum(r['split'] == split for r in bank['starts'])} |")
    lines += [
        "",
        "Train masks include canonical curriculum starts and pinned earlier improvements.",
        "Held-out starts are selected before training by refined Q from fixed controls.",
        "All bank mask hashes are unique across all splits. Cases remain from two geometry",
        "families, not a representative sample of real pulmonary anatomy.",
        "",
        "## All evaluation outcomes",
        "",
        "| Split | Method | Complete / runs | Median Q | Median gain Q | "
        "Median eta | Median seconds |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for (split, method), g in e.groupby(["split", "method"], sort=False):
        complete = g[g.status == "complete"]
        values = [
            f"{complete[key].median():.7g}" if len(complete) else "undefined"
            for key in ("final_Q", "gain_Q", "eta", "wall_seconds")
        ]
        lines.append(
            f"| {split} | {method} | {len(complete)}/{len(g)} | " + " | ".join(values) + " |"
        )
    lines += [
        "",
        "## Test pairs: learned replica versus each control",
        "",
        "Pairs are identical start/rollout-seed combinations, conditional on both runs completing.",
        "Controls are reused across replicas; these rows are not independent experiments.",
        "",
        "| Replica | Control | Complete pairs | Coarse wins / ties / losses | "
        "Fine wins / ties / losses |",
        "|---|---|---:|---:|---:|",
    ]
    test = e[(e.split == "test") & (e.status == "complete")]
    methods = [m for m in test.method.unique() if m.startswith("rl_")]
    for method in methods:
        learned = test[test.method == method].set_index(["start", "seed"])
        for control in ("uniform", "screened_uniform", "perimeter_pool", "flux_pool", "untrained"):
            base = test[test.method == control].set_index(["start", "seed"])
            pairs = learned.join(base, lsuffix="_rl", rsuffix="_base", how="inner")
            counts = []
            for key in ("final_Q", "fine_final_Q"):
                delta = pairs[f"{key}_rl"] - pairs[f"{key}_base"]
                counts.append(
                    f"{int((delta > config.min_improvement).sum())} / "
                    f"{int((abs(delta) <= config.min_improvement).sum())} / "
                    f"{int((delta < -config.min_improvement).sum())}"
                )
            lines.append(f"| {method} | {control} | {len(pairs)} | " + " | ".join(counts) + " |")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for replica, g in t.groupby("replica"):
        axes[0].plot(g.episode + 1, g.final_Q / g.initial_Q - 1, label=str(replica), alpha=0.7)
    axes[0].set_xlabel("Training episode")
    axes[0].set_ylabel("Relative Q gain on cycling curriculum")
    axes[0].legend(title="Replica", fontsize=7)
    labels, values = [], []
    for method, g in test.groupby("method", sort=False):
        labels.append(method)
        values.append(g.gain_Q.values)
    if values:
        axes[1].boxplot(values, tick_labels=labels, showmeans=True)
    axes[1].tick_params(axis="x", rotation=55)
    axes[1].set_ylabel("Test gain Q across available start/seed pairs")
    fig.savefig(root / "figures/training_and_test.png", dpi=160)
    plt.close(fig)
    lines += [
        "",
        "![Curriculum learning and held-out gains](figures/training_and_test.png)",
        "",
        "Training curves vary with the starting curriculum; they are not monotone learning proofs.",
        "Boxplots show variability across start/seed pairs. Different replicas share those starts.",
        "",
        "## Readiness, costs and limitations",
        "",
        f"Total PDE solves: {summary['number_of_pde_solves']} = "
        f"{summary['preparation_pde_solves']} pioneer preparation + "
        f"{summary['preflight_pde_solves']} preflight + "
        f"{summary['training_pde_solves']} training + "
        f"{summary['evaluation_pde_solves']} evaluation + "
        f"{summary['refinement_pde_solves']} evaluation refinement.",
        "Historical cost of reused pioneers: "
        f"{summary.get('historical_reused_pioneer_pde_solves', 0)} "
        "recorded generating-run solves, separate from this run's new solves.",
        f"Training episodes: {summary['training_episodes']}; "
        f"incomplete: {summary['incomplete_training_episodes']}.",
        f"Maximum conservation error: {summary['max_conservation_error']:.5g}.",
        "",
        "The learner is an on-policy linear Expected Sarsa research prototype with budget-aware",
        "incumbent/candidate features. It is a partially observed approximation, not a convergence",
        "or optimality result. Masks are screened geometrically; no candidate PDE response enters",
        "action selection. Raw draws for bootstrap pools count toward the same cap.",
        "Five final replicas, failed budgets and every control remain visible. Training cost is",
        "not hidden in the evaluation comparison. Final shapes use 200/400 staircase refinement,",
        "not a certified continuum error bound. No policy is promoted automatically.",
        "",
        "policies/ contains atomic resumable checkpoints, each including episode history and",
        "a bank/config/source identity. status.json records the active job. evaluation_index.json",
        "allows completed evaluations to be skipped on resume; frozen model digests are checked.",
        "bank/manifest.json pins source and split hashes; readiness.json records preflight.",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
