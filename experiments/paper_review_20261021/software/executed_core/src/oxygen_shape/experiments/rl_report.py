"""Report the physical and computational outcomes of a frozen RL benchmark."""

import numpy as np

from oxygen_shape.optimization.config import RLConfig


def benchmark_report(config: RLConfig, evaluations: list[dict], training_cost: int) -> str:
    lines = [
        "# Reinforcement learning disk-proposal experiment",
        "",
        "Tabular Q-learning chooses a boundary region and radius bin. The deterministic "
        "PDE solver evaluates every valid candidate, and greedy acceptance uses global Q.",
        "",
        "## Fixed problem and controls",
        "",
        f"Grid {config.n}x{config.n}, Bi={config.model.Bi:g}, initial body {config.initial_shape}, "
        f"area {config.target_area:g}, Robin budget {config.perimeter_limit:g}, fixed unit box, "
        f"source length 0.1, protected collar x<{config.protected_length:g}. "
        f"Feature-opening width {config.minimum_feature_width:g}; no holes or disconnected air.",
        "",
        "Area is preserved by paired disk addition/removal, with radii at least 3h. "
        "Coarse and refined morphology checks precede candidate PDE solves. "
        "The feature criterion is an operational raster opening test, not a proven "
        "continuum minimum-neck bound.",
        "",
        f"Training: {config.training_episodes} episodes, at most {config.training_solve_budget} "
        f"solves and {config.training_solve_budget * config.training_attempts_per_solve} attempts "
        f"per episode. Actual cost is recorded. Evaluation seeds: {config.evaluation_seeds}, "
        "disjoint from training. "
        f"Each evaluation uses {config.solve_budget} PDE solves, including its initial solve. "
        "Invalid proposals consume attempts and wall time but no PDE solve. All three "
        "policies use identical feasibility, reward accounting, and acceptance rules. "
        f"During evaluation, after {config.policy_stall_limit} consecutive invalid proposals, "
        "the learned and untrained policies switch to uniform boundary sampling until a valid "
        "candidate resets the counter. This is a hybrid policy with a feasibility safeguard.",
        "",
        "Controls: uniform boundary-measure sampler; zero-table untrained policy with the "
        "same region/radius action space; frozen learned policy. Evaluation explores with "
        f"epsilon={config.evaluation_epsilon:g}. Training seeds and every proposal are saved.",
        "",
        "## Held-out evaluation",
        "",
        "| Policy | Median final Q | Q interquartile range | Median gain (%) | "
        "Evaluation PDE solves | Refinement solves |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("uniform", "untrained", "trained"):
        rows = [r for r in evaluations if r["policy"] == name]
        q = [r["final_Q"] for r in rows]
        gains = [100 * (r["final_Q"] / r["initial_Q"] - 1) for r in rows]
        lines.append(
            f"| {name} | {np.median(q):.8g} | "
            f"{np.quantile(q, 0.25):.8g}–{np.quantile(q, 0.75):.8g} | "
            f"{np.median(gains):.5g} | {sum(r['number_of_pde_solves'] for r in rows)} | "
            f"{sum(r['refinement_pde_solves'] for r in rows)} |"
        )
    learned = {r["seed"]: r for r in evaluations if r["policy"] == "trained"}
    random = {r["seed"]: r for r in evaluations if r["policy"] == "uniform"}
    wins = sum(learned[seed]["final_Q"] > random[seed]["final_Q"] for seed in learned)
    fine_wins = sum(
        learned[seed]["fine_final_Q"] > random[seed]["fine_final_Q"] for seed in learned
    )
    lines += [
        "",
        "### Proposal and wall-time costs",
        "",
        "| Policy | Median proposal attempts | Median run seconds, including refinement | "
        "Total fallback attempts |",
        "|---|---:|---:|---:|",
    ]
    for name in ("uniform", "untrained", "trained"):
        group = [row for row in evaluations if row["policy"] == name]
        lines.append(
            f"| {name} | {np.median([r['proposal_attempts'] for r in group]):.6g} | "
            f"{np.median([r['wall_clock_seconds'] for r in group]):.6g} | "
            f"{sum(r['uniform_fallback_attempts'] for r in group)} |"
        )
    lines += [
        "",
        f"Learned policy has higher Q than uniform in **{wins}/{len(learned)}** coarse "
        f"and **{fine_wins}/{len(learned)}** finest-grid matched-seed runs. "
        "These are descriptive counts from a small sample, not a statistical superiority claim.",
        "",
        "![Evaluation curves](figures/benchmark.png)",
        "",
        "## Cost and refinement",
        "",
        f"Training cost: **{training_cost} PDE solves**, reported separately from the "
        "evaluation-only x-axis. Refinement cost is additional. Training is not free, "
        "and this experiment does not establish an end-to-end speedup.",
        "",
        "Initial and final domains are subdivided into finer cells with identical physical "
        "staircase boundaries, area, source and membrane length. This checks the gain "
        "against PDE discretization sensitivity without rerasterizing a smooth shape. "
        "All final shapes must retain the same geometric constraints on those grids.",
        "",
        "| Policy | Seed | Coarse Q gain | Finest Q gain | "
        "Change in gain on last refinement | Gain exceeds that change |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in evaluations:
        lines.append(
            f"| {row['policy']} | {row['seed']} | "
            f"{row['final_Q'] - row['initial_Q']:.6g} | {row['fine_gain_Q']:.6g} | "
            f"{row['observed_gain_grid_sensitivity']:.4g} | "
            f"{row['gain_exceeds_observed_sensitivity']} |"
        )
    lines += [
        "",
        "The last-grid gain change is an observed sensitivity, not a rigorous error bound.",
        "",
        "## What is learned",
        "",
        "There are eight observed states: flux gain below/above 10%, membrane usage "
        "below/above 85% of budget, and distal mean concentration below/above 0.1. "
        "There are twelve actions: six spatial regions (three x thirds times two y halves) "
        "and two radius bins. The selected boundary face and removal donor are random. "
        "Uniform proposals sample equal-length eligible faces with equal probability.",
        "",
        f"Reward = {config.reward_scale:g} * accepted_delta_Q / initial_Q; "
        f"invalid attempts receive {config.invalid_reward:g}, valid nonimprovements receive zero. "
        "Geometry changes only after a feasible positive global-flux gain.",
        "",
        "The update is Q(s,a) += alpha * (reward + gamma*max Q(s_next,.) - Q(s,a)), "
        "with no bootstrap at terminal steps. This is the tabular Q-learning update "
        "of [Watkins and Dayan (1992)](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf). "
        "The coarse observation aliases many different geometries, so this prototype "
        "does not meet the full-state Markov assumptions of its convergence theorem. "
        "Finite training with a constant learning rate supplies no convergence guarantee.",
        "",
        "![Learning diagnostics](figures/learning.png)",
        "",
        "## Artifacts and limits",
        "",
        "`policy.npz` contains learned values and visit counts; `policy_metadata.json` "
        "defines their interpretation. `training.csv`, `evaluations.csv`, and per-run "
        "`metrics.csv`, `proposals.csv`, masks, fields and refinement data retain the evidence. "
        "The checkpoint is frozen during evaluation. Source snapshots, environment, "
        "seeds, tolerances and Git state are stored at the root.",
        "",
        "Held-out seeds test search randomness on the same initial geometry and Bi. "
        "They do not test generalization to unseen shapes or physical regimes. "
        "This is a compact RL proof of concept; neural policies and learned PDE surrogates "
        "are not implemented. No globally optimal geometry is claimed. "
        "The low-concentration deletion hypothesis and sensitivity-guided sampling "
        "remain separate research milestones.",
        "",
    ]
    return "\n".join(lines)
