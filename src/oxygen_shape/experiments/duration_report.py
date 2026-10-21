"""Development-only checkpoint curves; shared pairs are kept explicit."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def write_duration_report(root: Path, summary: dict, evaluations: list[dict]):
    data = pd.DataFrame(evaluations)
    base = data[data.method == "flux_pool"]
    learned = data[data.method == "learned"]
    pairs = learned.merge(base, on=["start", "seed"], suffixes=("_rl", "_flux"))
    pairs["fine_delta_Q"] = pairs.fine_final_Q_rl - pairs.fine_final_Q_flux
    pairs["workflow_ratio"] = pairs.workflow_seconds_total_rl / pairs.workflow_seconds_total_flux
    pairs["complete_pair"] = (pairs.status_rl == "complete") & (pairs.status_flux == "complete")
    pairs.to_csv(root / "paired_checkpoints.csv", index=False)
    threshold = summary.get("min_improvement", 1e-7)
    rows = []
    for episode, group in pairs.groupby("checkpoint_episode_rl", sort=True):
        complete = group[group.complete_pair]
        delta = complete.fine_delta_Q
        rows.append(
            {
                "episode": episode,
                "shared_pairs": len(group),
                "complete_pairs": len(complete),
                "wins": int((delta > threshold).sum()),
                "ties": int((delta.abs() <= threshold).sum()),
                "losses": int((delta < -threshold).sum()),
                "mean_delta_Q": delta.mean(),
                "median_delta_Q": delta.median(),
                "median_workflow_ratio": complete.workflow_ratio.median(),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(root / "checkpoint_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for replica, group in pairs[pairs.complete_pair].groupby("replica_rl"):
        means = group.groupby("checkpoint_episode_rl").fine_delta_Q.mean()
        ax.plot(means.index, means.values, marker="o", label=str(int(replica)), alpha=0.8)
    ax.axhline(0, color="black", linewidth=1)
    ax.set(
        xlabel="Completed training episodes",
        ylabel="Mean refined Q difference versus flux",
        title="Development curves: identical starts and seeds shared across replicas",
    )
    ax.legend(title="Replica")
    fig.savefig(root / "figures/development_learning_curve.png", dpi=160)
    plt.close(fig)
    lines = [
        "# Controlled training-duration study",
        "",
        f"Status: {summary['status']}.",
        "",
        "All previous validation/test geometries are development data here. No fresh final",
        "test is used or claimed, and no model is promoted. The first 24 episodes and their",
        "exploration schedule are reused exactly; exploration stays at its final floor later.",
        "The table/curve use pairs with both PDE budgets complete. All incomplete outcomes",
        "remain in paired_checkpoints.csv, evaluations.csv and the summary counts and costs.",
        "",
        "| Episodes | Complete / shared pairs | Wins / ties / losses vs flux | "
        "Mean delta Q | Median delta Q |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['episode']} | {row['complete_pairs']} / {row['shared_pairs']} | "
            f"{row['wins']} / {row['ties']} / {row['losses']} | "
            f"{row['mean_delta_Q']:.7g} | {row['median_delta_Q']:.7g} |"
        )
    lines += [
        "",
        "![Development learning curves](figures/development_learning_curve.png)",
        "",
        f"New experimental PDE solves: {summary['number_of_new_pde_solves']}. "
        f"Reused prefix solves: {summary['reused_training_pde_solves']} (historical, separate).",
        "",
        "These paired comparisons share geometries and controls. They do not constitute",
        "independent samples across replicas. Episode reward on a cycling curriculum is",
        "not a learning curve; the plot evaluates frozen checkpoints on fixed development cases.",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
