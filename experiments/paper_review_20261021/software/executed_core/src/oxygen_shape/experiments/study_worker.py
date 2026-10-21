"""One isolated, checkpointed job in a fixed-code long-running study."""

import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter, sleep
from uuid import uuid4

import numpy as np
import yaml

from oxygen_shape.experiments.optimization import save_search, search_summary
from oxygen_shape.experiments.reporting import write_csv
from oxygen_shape.optimization.config import RLConfig
from oxygen_shape.optimization.q_learning import QPolicy
from oxygen_shape.optimization.random_search import run_search
from oxygen_shape.optimization.refinement import validate_refinement


def replace_with_retry(source: Path, destination: Path) -> None:
    """Allow short Windows reader/sync locks; preserve a permanent failure."""
    for attempt in range(7):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 6:
                raise
            sleep(min(0.02 * 2**attempt, 0.25))


def atomic_json(path: Path, value: dict) -> None:
    content = json.dumps(value, indent=2, allow_nan=False)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        replace_with_retry(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except PermissionError:
            # Keep an inspectable temporary if the sync service still holds it.
            pass


def _train(root: Path, path: Path, job: dict, config: RLConfig) -> dict:
    policy, rows = QPolicy(), []
    start = perf_counter()
    for episode in range(config.training_episodes):
        seed = config.training_seed_offset + config.seed + episode
        fraction = episode / max(1, config.training_episodes - 1)
        epsilon = config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)
        cfg = config.model_copy(
            update={
                "seed": seed,
                "solve_budget": config.training_solve_budget,
                "max_attempts_per_solve": config.training_attempts_per_solve,
            }
        )
        result = run_search(cfg, policy=policy, epsilon=epsilon, training=True)
        episode_path = path / "episodes" / f"{episode:04d}"
        save_search(episode_path, result)
        policy.save(episode_path / "policy.npz")
        policy.save(path / "policy.tmp.npz")
        replace_with_retry(path / "policy.tmp.npz", path / "policy.npz")
        rows.append(
            {
                "episode": episode,
                "seed": seed,
                "epsilon": epsilon,
                "initial_Q": result.initial_solution.metrics.physical_flux_Q,
                "final_Q": result.final_solution.metrics.physical_flux_Q,
                "number_of_pde_solves": result.number_of_pde_solves,
                "attempts": result.attempts,
            }
        )
        write_csv(path / "training.csv", rows)
        atomic_json(
            path / "progress.json",
            {
                "completed_episodes": episode + 1,
                "training_pde_solves": sum(row["number_of_pde_solves"] for row in rows),
                "last_seed": seed,
                "last_Q": result.final_solution.metrics.physical_flux_Q,
            },
        )
        print(
            f"episode {episode + 1}/{config.training_episodes}: "
            f"Q={result.final_solution.metrics.physical_flux_Q:.7g}",
            flush=True,
        )
    save_search(path, result)
    metadata = {
        "algorithm": "tabular_q_learning",
        "states": 8,
        "actions": 12,
        "training_configuration": config.model_dump(mode="json"),
        "completed_episodes": len(rows),
        "evaluation_stall_limit": config.policy_stall_limit,
    }
    atomic_json(path / "policy_metadata.json", metadata)
    return {
        "status": "complete",
        "training_seed": config.seed,
        "completed_episodes": len(rows),
        "training_pde_solves": sum(row["number_of_pde_solves"] for row in rows),
        "transitions": int(policy.visits.sum()),
        "Bi": config.model.Bi,
        "n": config.n,
        "wall_clock_seconds": perf_counter() - start,
    }


def _evaluate(root: Path, path: Path, job: dict, config: RLConfig) -> dict:
    policy = (
        None
        if job["policy_label"] == "uniform"
        else QPolicy()
        if job["policy_label"] == "untrained"
        else QPolicy.load(root / job["checkpoint"])
    )
    values = None if policy is None else policy.values.copy()
    visits = None if policy is None else policy.visits.copy()
    start = perf_counter()
    progress_rows = []

    def on_step(step, geometry, row):
        progress_rows.append(row)
        if step % 25 == 0 or row["accepted"] or row["number_of_pde_solves"] == config.solve_budget:
            atomic_json(path / "progress.json", row)
            np.save(path / "latest_geometry.npy", geometry.inside)

    try:
        result = run_search(
            config, policy=policy, epsilon=config.evaluation_epsilon, on_step=on_step
        )
    except Exception:
        if progress_rows:
            write_csv(path / "partial_metrics.csv", progress_rows)
            atomic_json(path / "progress.json", progress_rows[-1])
        raise
    if policy is not None and (
        not np.array_equal(policy.values, values) or not np.array_equal(policy.visits, visits)
    ):
        raise ArithmeticError("Frozen study policy changed during evaluation")
    save_search(path, result)
    refinements = validate_refinement(config, result, path)
    return {
        "status": "complete",
        **search_summary(config, result, refinements),
        "transfer": job.get("transfer", False),
        "wall_clock_seconds": perf_counter() - start,
    }


def execute_job(root: Path, job: dict) -> int:
    path = root / "jobs" / job["id"]
    path.mkdir(parents=True, exist_ok=True)
    atomic_json(path / "job.json", job)
    try:
        if job["kind"] == "tests":
            snapshot = root / "source_snapshot"
            commands = [
                [sys.executable, "-m", "pytest", "-q", str(snapshot / "tests")],
                [
                    sys.executable,
                    "-m",
                    "ruff",
                    "check",
                    str(snapshot / "src"),
                    str(snapshot / "tests"),
                ],
            ]
            for command in commands:
                result = subprocess.run(command, cwd=snapshot, check=False)
                if result.returncode:
                    raise ArithmeticError("Scientific regression or lint gate failed")
            result = {"status": "complete", "regression_gate": "passed"}
        else:
            config = RLConfig.model_validate(job["config"])
            (path / "config.yaml").write_text(
                yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8"
            )
            result = (
                _train(root, path, job, config)
                if job["kind"] == "train"
                else _evaluate(root, path, job, config)
            )
        result["source_manifest"] = str(root / "source_manifest.json")
        atomic_json(path / "result.json", result)
        return 0
    except Exception as exc:
        fatal = isinstance(exc, ArithmeticError)
        atomic_json(
            path / "result.json",
            {
                "status": "failed",
                "exception": type(exc).__name__,
                "reason": str(exc),
                "fatal_numerical": fatal,
            },
        )
        print(f"FAILED {type(exc).__name__}: {exc}", flush=True)
        return 2 if fatal else 1
