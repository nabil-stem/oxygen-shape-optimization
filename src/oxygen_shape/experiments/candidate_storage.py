"""Complete candidate rollout artifacts shared by preparation and training."""

import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import yaml

from oxygen_shape.experiments.optimization import save_search
from oxygen_shape.experiments.reporting import write_csv
from oxygen_shape.optimization.candidate_rollout import run_candidate_rollout


def archive_uncommitted_rollout(path: Path) -> None:
    """Preserve interrupted work before replay from the last atomic checkpoint."""
    if not path.exists() or not any(path.iterdir()):
        return
    workspace = path.parent.resolve()
    target = path.resolve()
    if not target.is_relative_to(workspace) or target == workspace:
        raise ValueError("Interrupted rollout is outside its expected parent")
    archive = workspace / "interrupted_attempts" / f"{path.name}_{uuid4().hex[:8]}"
    archive.parent.mkdir(exist_ok=True)
    if not archive.resolve().is_relative_to(workspace):
        raise ValueError("Archive is outside the rollout workspace")
    candidates = len(list(path.glob("candidates/solve_*.npz")))
    summary = path / "rollout_summary.json"
    completed = json.loads(summary.read_text()) if summary.exists() else None
    path.rename(archive)
    (archive / "interruption_cost.json").write_text(
        json.dumps(
            {
                "replayed_from_checkpoint": True,
                "persisted_candidate_solves": candidates,
                "pde_solves_lower_bound": completed["number_of_pde_solves"]
                if completed
                else candidates,
                "pde_solves_upper_bound": completed["number_of_pde_solves"]
                if completed
                else candidates + 2,
                "endpoint_refinement_additional_upper_bound": 4
                if (archive / "final_geometry.npy").exists()
                else 0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def run_stored_rollout(
    path: Path, config, initial, *, method, policy=None, training=False, epsilon=0
):
    started = perf_counter()
    archive_uncommitted_rollout(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "candidates").mkdir(exist_ok=True)
    (path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "configuration": config.model_dump(mode="json"),
                "method": method,
                "training": training,
                "epsilon": epsilon,
                "policy_digest_before": None if policy is None else policy.digest(),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    def save_candidate(index, geometry, solution, accepted):
        destination = path / "candidates" / f"solve_{index:03d}.npz"
        temporary = destination.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, mask=geometry.inside, concentration=solution.concentration)
        temporary.replace(destination)

    result = run_candidate_rollout(
        config,
        initial,
        method=method,
        policy=policy,
        training=training,
        epsilon=epsilon,
        on_evaluation=save_candidate,
    )
    save_search(path, result.search)
    write_csv(path / "transitions.csv", result.transitions)
    if result.candidate_metrics:
        write_csv(path / "candidate_metrics.csv", result.candidate_metrics)
    np.save(path / "selected_features.npy", result.selected_features)
    summary = {
        "status": result.status,
        "number_of_pde_solves": result.search.number_of_pde_solves,
        "raw_draws": result.search.attempts,
        "decisions": len(result.transitions),
        "max_conservation_error": result.maximum_conservation_error,
        "initial_Q": result.search.initial_solution.metrics.physical_flux_Q,
        "final_Q": result.search.final_solution.metrics.physical_flux_Q,
        "wall_seconds": result.seconds,
        "training": training,
        "method": method,
        "candidate_screening": config.candidate_screening,
        "refill_quota": config.refill_quota if method != "uniform" else 0,
        "multi_choice_decisions": sum(t["eligible_candidates"] >= 2 for t in result.transitions),
        "quota_reached_decisions": sum(t["quota_reached"] for t in result.transitions),
        "full_geometry_checks": result.full_geometry_checks,
        "geometry_validation_seconds": result.geometry_validation_seconds,
        "invalid_decisions": sum(t["candidate_Q"] is None for t in result.transitions),
        "reward_sum": sum(t["reward"] for t in result.transitions),
        "positive_reward_sum": sum(max(0, t["reward"]) for t in result.transitions),
        "workflow_seconds": perf_counter() - started,
    }
    (path / "rollout_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result, summary
