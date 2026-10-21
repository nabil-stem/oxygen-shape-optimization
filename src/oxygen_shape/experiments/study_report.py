"""Incremental, failure-aware reporting for a bounded multi-run study."""

import json
from pathlib import Path

import numpy as np

from oxygen_shape.experiments.reporting import write_csv


def write_study_report(root: Path, state: dict, jobs: list[dict]) -> None:
    evaluations, training = [], []
    for job in jobs:
        path = root / "jobs" / job["id"] / "result.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") == "complete":
            if job["kind"] == "evaluate":
                evaluations.append(
                    {
                        "job_id": job["id"],
                        "condition": job["condition"],
                        "policy_label": job["policy_label"],
                        **result,
                    }
                )
            elif job["kind"] == "train":
                training.append({"job_id": job["id"], **result})
    if evaluations:
        write_csv(
            root / "evaluations.csv",
            [{k: v for k, v in row.items() if k != "constraints"} for row in evaluations],
        )
    if training:
        write_csv(root / "training_summary.csv", training)
    statuses = state.get("jobs", {})
    completed = sum(value == "complete" for value in statuses.values())
    failed = sum(value == "failed" for value in statuses.values())
    interrupted = sum(value == "interrupted" for value in statuses.values())
    lines = [
        "# Two-hour oxygen-shape RL study",
        "",
        f"Status: **{state['status']}**. Updated: {state['updated_utc']}.",
        "",
        f"Deadline: **{state['deadline_utc']}**. Controller PID: {state.get('controller_pid')}. "
        f"Active job: {state.get('active_job') or 'none'}.",
        "",
        f"Planned jobs: {len(jobs)}; completed: {completed}; failed: {failed}; "
        f"interrupted: {interrupted}; other jobs remain pending or were skipped "
        "for budget/dependencies.",
        "",
        "## Protocol",
        "",
        "The existing checkpoint and five independently seeded 40-episode training replicas "
        "are assessed against uniform boundary sampling and the zero-table action-space control. "
        "Fresh search seeds are prespecified in plan.json. Shared controls run once per "
        "condition/seed, rather than being duplicated for every training replica.",
        "",
        "Every completed evaluation uses 41 search PDE solves, including the initial solve, "
        "and 200/400-grid validation of the same physical staircase geometry. "
        "Area, source collar, membrane budget, topology and morphology constraints stay fixed. "
        "A 50-invalid-attempt uniform fallback applies to frozen learned and untrained policies. "
        "Training costs, feasibility attempts and wall time are additional.",
        "",
        "Training and search seeds are disjoint. All final training checkpoints are evaluated "
        "without choosing one using held-out rewards. Transfer conditions are labelled separately; "
        "they change the initial geometry or Bi while freezing the original policy.",
        "",
        "The controller admits jobs using timing estimates and enforces the UTC deadline. "
        "A job that misses its solve budget is a failure, not a completed lower-budget comparison. "
        "Interrupted jobs retain the latest checkpoint/progress and are excluded from medians.",
        "",
        "## Completed evaluations",
        "",
        "| Condition | Policy | Completed / planned | Median Q | Q interquartile range | "
        "Median gain (%) | Median attempts | Median seconds incl. refinement |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    pairs = list(
        dict.fromkeys(
            (job.get("condition"), job.get("policy_label"))
            for job in jobs
            if job["kind"] == "evaluate"
        )
    )
    for condition, policy in pairs:
        rows = [
            row
            for row in evaluations
            if row["condition"] == condition and row["policy_label"] == policy
        ]
        planned = sum(
            job["kind"] == "evaluate"
            and job.get("condition") == condition
            and job.get("policy_label") == policy
            for job in jobs
        )
        if not rows:
            continue
        q = [row["final_Q"] for row in rows]
        gains = [100 * (row["final_Q"] / row["initial_Q"] - 1) for row in rows]
        lines.append(
            f"| {condition} | {policy} | {len(rows)}/{planned} | {np.median(q):.8g} | "
            f"{np.quantile(q, 0.25):.8g}–{np.quantile(q, 0.75):.8g} | "
            f"{np.median(gains):.5g} | {np.median([r['proposal_attempts'] for r in rows]):.6g} | "
            f"{np.median([r['wall_clock_seconds'] for r in rows]):.6g} |"
        )
    lines += [
        "",
        "## Matched-seed comparisons against uniform",
        "",
        "Only complete pairs are compared; failure and missing counts remain visible above. "
        "Rows with fewer than five complete pairs are preliminary. Training replicas are "
        "reported separately, rather than pooling model/seed combinations as independent samples.",
        "",
        "| Condition | Policy | Complete pairs | Median paired Q difference | "
        "Coarse wins | Finest-grid wins |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for condition, policy in pairs:
        if policy == "uniform":
            continue
        baseline = {
            row["seed"]: row
            for row in evaluations
            if row["condition"] == condition and row["policy_label"] == "uniform"
        }
        matched = [
            row
            for row in evaluations
            if row["condition"] == condition
            and row["policy_label"] == policy
            and row["seed"] in baseline
        ]
        if not matched:
            continue
        differences = [r["final_Q"] - baseline[r["seed"]]["final_Q"] for r in matched]
        fine_wins = sum(r["fine_final_Q"] > baseline[r["seed"]]["fine_final_Q"] for r in matched)
        lines.append(
            f"| {condition} | {policy} | {len(matched)} | {np.median(differences):.7g} | "
            f"{sum(d > 0 for d in differences)}/{len(matched)} | {fine_wins}/{len(matched)} |"
        )
    lines += [
        "",
        "## Cost, refinement and limitations",
        "",
        f"Completed new training PDE solves: {sum(r['training_pde_solves'] for r in training)}. "
        f"Completed evaluation PDE solves: {sum(r['number_of_pde_solves'] for r in evaluations)}. "
        f"Completed refinement PDE solves: {sum(r['refinement_pde_solves'] for r in evaluations)}.",
        "The reference checkpoint previously cost 809 training solves; those are not counted "
        "as new work. Failed/interrupted jobs may consume additional compute; their partial "
        "counters are retained in job result/progress files and excluded from complete-run costs.",
        "",
        f"Completed final gains exceeding observed last-grid sensitivity: "
        f"{sum(r['gain_exceeds_observed_sensitivity'] for r in evaluations)}/{len(evaluations)}. "
        "That sensitivity is an empirical grid check, not a rigorous error bound. "
        "The opening constraint is a raster proxy; smooth-boundary accuracy is not established.",
        "",
        "No model is promoted automatically and the original models/ checkpoint is unchanged. "
        "No globally optimal shape, universal RL advantage, or end-to-end speedup is claimed.",
        "",
        "## Files",
        "",
        "`status.json` is the live state; `plan.json` fixes the job order and seeds; "
        "`jobs/` contains configs, results, logs, masks, fields and partial progress. "
        "Training jobs retain per-episode checkpoints. `evaluations.csv` and "
        "`training_summary.csv` aggregate completed runs. Code and test snapshots, "
        "environment versions, Git state, and reference-checkpoint hashes are preserved.",
        "",
        "To stop this study early, create a file named `STOP` in this experiment directory. "
        "The controller stops its own worker and writes a partial report.",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
