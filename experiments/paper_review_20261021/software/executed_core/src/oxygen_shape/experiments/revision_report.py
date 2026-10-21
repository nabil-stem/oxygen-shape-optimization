"""All-setting validation, paired final tests and fully exposed workflow costs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oxygen_shape.experiments.revision_preflight import REVISION_CONTROLS


def write_revision_report(
    root: Path,
    config,
    summary: dict,
    training: list[dict],
    evaluations: list[dict],
    selection: dict,
) -> None:
    t, e = pd.DataFrame(training), pd.DataFrame(evaluations)
    e["fine_normalized_gain"] = e.fine_gain_Q / (e.fine_final_Q - e.fine_gain_Q)
    val = e[e.split == "validation"]
    test = e[(e.split == "test") & (e.status == "complete")]
    rows = []
    for score in selection["validation_scores"]:
        group = val[val.arm == score["arm"]]
        train = t[t.arm == score["arm"]]
        rows.append(
            {
                **score,
                "training_PDE_solves": int(train.number_of_pde_solves.sum()),
                "training_workflow_seconds": float(train.workflow_seconds.sum()),
                "training_invalid_fraction": float(
                    train.invalid_decisions.sum() / train.decisions.sum()
                ),
                "validation_PDE_solves": int(group.number_of_pde_solves.sum()),
                "validation_refinement_solves": int(group.refinement_pde_solves.sum()),
                "validation_workflow_seconds": float(group.workflow_seconds.sum()),
                "training_geometry_checks": int(train.full_geometry_checks.sum()),
                "training_geometry_seconds": float(train.geometry_validation_seconds.sum()),
            }
        )
    pd.DataFrame(rows).to_csv(root / "setting_validation_and_costs.csv", index=False)
    methods = e.groupby(["split", "method"], dropna=False).agg(
        runs=("method", "size"),
        complete=("status", lambda x: int((x == "complete").sum())),
        median_fine_Q=("fine_final_Q", "median"),
        median_fine_gain=("fine_gain_Q", "median"),
        median_eta=("eta", "median"),
        median_environment_seconds=("wall_seconds", "median"),
        median_workflow_seconds=("workflow_seconds", "median"),
        total_geometry_checks=("full_geometry_checks", "sum"),
        geometry_validation_seconds=("geometry_validation_seconds", "sum"),
    )
    methods.to_csv(root / "method_summary.csv")
    pairs = []
    for name in [n for n in test.method.unique() if n not in REVISION_CONTROLS]:
        learned = test[test.method == name].set_index(["start", "seed"])
        for control in REVISION_CONTROLS:
            base = test[test.method == control].set_index(["start", "seed"])
            p = learned.join(
                base, lsuffix="_rl", rsuffix="_base", how="inner", validate="one_to_one"
            )
            fine = p.fine_final_Q_rl - p.fine_final_Q_base
            coarse = p.final_Q_rl - p.final_Q_base

            def sign(a):
                return np.where(
                    a > config.min_improvement, 1, np.where(a < -config.min_improvement, -1, 0)
                )

            f, c = sign(fine), sign(coarse)
            pairs.append(
                {
                    "model": name,
                    "arm": str(learned.arm.iloc[0]),
                    "control": control,
                    "complete_pairs": len(p),
                    "wins": int((f == 1).sum()),
                    "ties": int((f == 0).sum()),
                    "losses": int((f == -1).sum()),
                    "coarse_fine_changes": int((f != c).sum()),
                    "median_fine_delta_Q": float(fine.median()) if len(p) else None,
                    "median_workflow_ratio_RL_over_control": float(
                        (p.workflow_seconds_rl / p.workflow_seconds_base).median()
                    )
                    if len(p)
                    else None,
                }
            )
    pair_table = pd.DataFrame(pairs)
    pair_table.to_csv(root / "paired_test.csv", index=False)
    lines = [
        "# Reward, flux-prior and feasibility revision",
        "",
        f"Status: **{summary['status']}**; "
        f"validation-selected setting: **{selection['selected_arm']}**.",
        "",
        "All validation settings and final replicas are reported. "
        "No individual replica is selected.",
        "Fixed core settings also enter the final test. No policy is promoted automatically.",
        "",
        "## Validation and development costs",
        "",
        "| Setting | Complete validation quota | Mean normalized fine gain | "
        "Training PDE solves | Training invalid % | Training workflow seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        score = row["mean_normalized_fine_gain"]
        lines.append(
            f"| {row['arm']} | {row['eligible']} ({row['runs']}/{row['expected_runs']}) | "
            f"{score if score is None else format(score, '.7g')} | "
            f"{row['training_PDE_solves']} | {100 * row['training_invalid_fraction']:.3g} | "
            f"{row['training_workflow_seconds']:.3f} |"
        )
    lines += [
        "",
        "Complete costs/check counts are in setting_validation_and_costs.csv "
        "and method_summary.csv.",
        "Validation uses all replicas and fresh rollout streams; "
        "no final-test response selects the setting.",
        "",
        "## Every final-test model versus controls",
        "",
        "Pairs match start and rollout seed and require both methods to complete. "
        "Counts use n=400 Q.",
        "Controls are reused across replicas; summed pairs are not independent trials.",
        "",
        "| Model | Control | Pairs | Wins / ties / losses | Median fine delta Q | "
        "Workflow time ratio RL/control |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in pairs:
        lines.append(
            f"| {row['model']} | {row['control']} | {row['complete_pairs']} | "
            f"{row['wins']} / {row['ties']} / {row['losses']} | "
            f"{row['median_fine_delta_Q']} | {row['median_workflow_ratio_RL_over_control']} |"
        )
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    labels = [a.name for a in config.arms]
    values = [
        val[(val.arm == name) & (val.status == "complete")].fine_normalized_gain.to_numpy()
        for name in labels
    ]
    if all(len(v) for v in values):
        axes[0].boxplot(values, tick_labels=labels, showmeans=True)
    axes[0].set_ylabel("Validation normalized Q gain at n=400")
    axes[0].tick_params(axis="x", rotation=65, labelsize=8)
    fractions = [100 * r["training_invalid_fraction"] for r in rows]
    axes[1].bar(labels, fractions)
    axes[1].set_ylabel("Training invalid selected decisions (%)")
    axes[1].tick_params(axis="x", rotation=65, labelsize=8)
    fig.savefig(root / "figures/validation_and_feasibility.png", dpi=160)
    plt.close(fig)
    grouped = test.copy()
    grouped["group"] = grouped.arm.fillna(grouped.method)
    # Average replicas within each shared physical case/seed; show no confidence interval.
    average = grouped.groupby(["group", "start", "seed"]).fine_gain_Q.mean().reset_index()
    groups = sorted(average.group.unique())
    fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True)
    if groups:
        ax.boxplot(
            [average[average.group == g].fine_gain_Q for g in groups],
            tick_labels=groups,
            showmeans=True,
        )
    ax.tick_params(axis="x", rotation=65, labelsize=8)
    ax.set_ylabel("Test Q gain at n=400 (replicas averaged per start/seed)")
    ax.set_title(
        "Descriptive variability; shared controls, no independent-replica confidence claim"
    )
    fig.savefig(root / "figures/test_flux.png", dpi=160)
    plt.close(fig)
    lines += [
        "",
        "![Validation and feasibility](figures/validation_and_feasibility.png)",
        "",
        "![Final-test refined flux](figures/test_flux.png)",
        "",
        "## Provenance and limitations",
        "",
        f"New experimental PDE solves: {summary['number_of_pde_solves']} = "
        f"{summary['preparation_pde_solves']} preparation + "
        f"{summary['preflight_pde_solves']} preflight + "
        f"{summary['training_pde_solves']} training + "
        f"{summary['validation_pde_solves']} validation + "
        f"{summary['test_pde_solves']} test + "
        f"{summary['refinement_pde_solves']} refinement.",
        "Historical reused-pioneer generating cost: "
        f"{summary['historical_reused_pioneer_pde_solves']} additional solves, "
        "separate from new work.",
        f"Training episodes: {summary['training_episodes']}; "
        f"incomplete: {summary['incomplete_training_episodes']}. "
        f"Validation/test runs: {summary['validation_runs']}/{summary['test_runs']}; "
        f"incomplete evaluations: {summary['incomplete_evaluations']}.",
        f"Maximum conservation error: {summary['max_conservation_error']:.7g}.",
        "",
        "Training workflow times include initial validation, setup and final state/trace "
        "serialization, but exclude subsequent atomic checkpoint writes. Evaluation workflow "
        "times also include refinement. Root invocation times include these overheads and "
        "preparation. An interrupted invocation is not included in successful-invocation "
        "time totals. Environment time and geometry-only validation time are separate. "
        "No equal-wall-time or amortized speedup is inferred from equal PDE budgets.",
        "",
        "The physical objective remains global Robin flux Q. Area, source/collar, topology, "
        "unit footprint, perimeter and feature constraints are unchanged. Full screening "
        "has extra geometric cost, uses no candidate PDE response, and preserves the "
        "unconditional exploration branch. Prior-based action values can degrade the "
        "initial heuristic; no nondegradation or convergence guarantee is asserted.",
        "",
        "The bank and every evaluation use 100/200/400 staircase grids; refinement is not "
        "a continuum error certificate. These are two related families at Bi=1. "
        "Scientific advantage, MC and 3D gates are unchanged. See docs/rl_revision_method.md, "
        "config.yaml, source_snapshot/, source_manifest.json, selection.json, readiness.json, "
        "evaluation_index.json and policies/.",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
