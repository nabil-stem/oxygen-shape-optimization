"""Load a learned checkpoint for a new search without retraining."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from oxygen_shape.experiments.optimization import save_search, search_summary
from oxygen_shape.experiments.reporting import create_run, log_failure, provenance, save_summary
from oxygen_shape.optimization.config import RLConfig
from oxygen_shape.optimization.q_learning import QPolicy
from oxygen_shape.optimization.random_search import run_search
from oxygen_shape.optimization.refinement import validate_refinement
from oxygen_shape.viz.fields import plot_fields
from oxygen_shape.viz.optimization import plot_search


def evaluate_policy(config: RLConfig, checkpoint: Path) -> tuple[Path, dict]:
    checkpoint = checkpoint.resolve()
    metadata_path = checkpoint.with_name("policy_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    original = metadata["training_configuration"]
    current = config.model_dump(mode="json")
    environment_fields = (
        "n",
        "model",
        "initial_shape",
        "target_area",
        "perimeter_limit",
        "minimum_feature_width",
        "protected_length",
        "radii",
        "min_improvement",
        "refinement_factors",
    )
    if any(original[key] != current[key] for key in environment_fields):
        raise ValueError("Checkpoint environment differs from this configuration")
    training_seeds = {
        original["training_seed_offset"] + original["seed"] + i
        for i in range(original["training_episodes"])
    }
    if config.seed in training_seeds:
        raise ValueError("Choose an inference seed outside the checkpoint's training seeds")
    policy = QPolicy.load(checkpoint)
    values, visits = policy.values.copy(), policy.visits.copy()
    directory = create_run(config, "evaluate_rl")
    start = perf_counter()
    try:
        result = run_search(config, policy=policy, epsilon=config.evaluation_epsilon)
        if not (np.array_equal(policy.values, values) and np.array_equal(policy.visits, visits)):
            raise ArithmeticError("Frozen evaluation mutated checkpoint weights")
        save_search(directory, result)
        policy.save(directory / "policy.npz")
        (directory / "policy_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        refinements = validate_refinement(config, result, directory)
        plot_fields(
            result.final_geometry, result.final_solution, directory / "figures/final_fields.png"
        )
        plot_search(result, directory / "figures/history.png")
        summary = {
            **provenance(),
            **search_summary(config, result, refinements),
            "policy": "frozen_q_learning_with_uniform_stall_fallback",
            "policy_stall_limit": config.policy_stall_limit,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "evaluation_epsilon": config.evaluation_epsilon,
            "wall_clock_seconds": perf_counter() - start,
        }
        save_summary(directory, summary)
        return directory, summary
    except Exception as exc:
        log_failure(directory, exc)
        raise
