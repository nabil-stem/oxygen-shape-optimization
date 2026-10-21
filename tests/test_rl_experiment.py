import csv
import json

from oxygen_shape.experiments.policy_evaluation import evaluate_policy
from oxygen_shape.experiments.rl import run_rl
from oxygen_shape.optimization.config import RLConfig
from oxygen_shape.optimization.q_learning import QPolicy


def test_training_and_held_out_benchmark_artifacts(tmp_path):
    config = RLConfig(
        output_root=tmp_path,
        training_episodes=1,
        training_solve_budget=4,
        solve_budget=4,
        evaluation_seeds=[0],
        refinement_factors=[2],
    )
    directory, summary = run_rl(config)
    assert summary["training_pde_solves"] == 4
    assert summary["evaluation_pde_solves"] == 12
    assert summary["refinement_pde_solves"] == 6
    assert summary["evaluation_frozen"]
    assert {r["policy"] for r in summary["evaluations"]} == {"uniform", "untrained", "trained"}
    assert all(r["number_of_pde_solves"] == 4 for r in summary["evaluations"])
    assert QPolicy.load(directory / "policy.npz").visits.sum() > 0
    for name in (
        "config.yaml",
        "metrics.csv",
        "training.csv",
        "evaluations.csv",
        "summary.json",
        "policy.npz",
        "policy_metadata.json",
        "report.md",
        "figures/benchmark.png",
        "figures/learning.png",
        "initial_geometry.npy",
        "final_geometry.npy",
    ):
        assert (directory / name).is_file()
    metadata = json.loads((directory / "policy_metadata.json").read_text())
    assert metadata["evaluation_frozen"]
    with (directory / "training.csv").open() as stream:
        training = list(csv.DictReader(stream))
    assert int(training[0]["seed"]) not in config.evaluation_seeds
    inference, result = evaluate_policy(config, directory / "policy.npz")
    assert result["number_of_pde_solves"] == 4
    assert result["policy"] == "frozen_q_learning_with_uniform_stall_fallback"
    assert (inference / "policy.npz").is_file()
