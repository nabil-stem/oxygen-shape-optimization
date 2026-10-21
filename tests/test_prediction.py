import csv
import json

import numpy as np
import pytest

from oxygen_shape.experiments.prediction_config import PredictionConfig
from oxygen_shape.experiments.prediction_diagnostic import run_prediction_diagnostic
from oxygen_shape.geometry.disk_edits import DiskEdit
from oxygen_shape.optimization.prediction import predict_scores, ranking_metrics, top_weights
from oxygen_shape.optimization.sensitivity import (
    PairResponse,
    mixture_probabilities,
    sensitivity_probabilities,
)
from oxygen_shape.solvers.finite_volume import solve


def test_scores_reproduce_the_current_joint_map_and_radius_factor():
    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    training = np.array([[0, 0, 1, 0, 0.05], [1, 0, 0, 0, 0.08]])
    delta, area = np.array([2.0, -4.0]), np.array([2.0, 2.0])
    queries = np.array([[*a, *d, 0.05] for a in points for d in points])
    scores = predict_scores(training, delta, area, queries, 0.1, 0.005)
    probabilities = mixture_probabilities(
        np.maximum(scores["spatial_sensitivity"].reshape(2, 2), 0), 0.25
    )
    expected = sensitivity_probabilities(
        points, [PairResponse((0, 0), (1, 0), 2, 2), PairResponse((1, 0), (0, 0), -4, 2)], 0.1, 0.25
    )
    np.testing.assert_allclose(probabilities, expected)
    assert scores["radius_delta"][1] == pytest.approx(2)
    assert abs(scores["radius_delta"][2]) < 1e-6
    np.testing.assert_allclose(scores["spatial_delta"], 2 * scores["spatial_sensitivity"])


def test_top_selection_handles_ties_without_looking_at_responses():
    np.testing.assert_array_equal(top_weights(np.zeros(4), 0.25), np.full(4, 0.25))
    np.testing.assert_allclose(top_weights(np.array([2.0, 2.0, 1.0, 0.0]), 0.25), [0.5, 0.5, 0, 0])
    metrics = ranking_metrics(np.arange(1.0, 5.0), np.arange(1.0, 5.0), np.ones(4), 0.25, 1e-7)
    assert metrics["spearman_delta_Q"] == 1
    assert metrics["top_mean_delta_Q"] == 4
    assert metrics["top_gain_over_pool_mean"] == 1.5
    tied = ranking_metrics(
        np.zeros(4), np.array([100.0, -100.0, 1.0, -1.0]), np.ones(4), 0.25, 1e-7
    )
    assert tied["spearman_delta_Q"] is None
    assert tied["top_gain_over_pool_mean"] == 0


def test_clipped_negatives_and_unresolved_signs_do_not_receive_false_credit():
    metrics = ranking_metrics(-np.arange(1.0, 5.0), -np.arange(1.0, 5.0), np.ones(4), 0.25, 1e-7)
    assert metrics["spearman_delta_Q"] is None
    assert metrics["sign_accuracy"] == 1
    assert metrics["always_positive_accuracy"] == 0
    unresolved = ranking_metrics(np.ones(4), np.zeros(4), np.ones(4), 0.25, 1e-7)
    assert unresolved["sign_accuracy"] is None
    assert unresolved["always_positive_accuracy"] is None


def test_fixed_geometry_diagnostic_has_disjoint_pools_and_real_solve_accounting(tmp_path):
    config = PredictionConfig(
        output_root=tmp_path,
        seed=120000,
        replicates=1,
        shapes=["deep_cavity"],
        calibration_sizes=[2, 4],
        heldout_samples=8,
        radius_families=2,
        max_attempts_per_pool=300,
        refinement_factors=[2],
    )
    first, summary = run_prediction_diagnostic(config)
    second, repeat = run_prediction_diagnostic(config)
    assert summary["status"] == "complete"
    assert summary["calibration_pde_solves"] == 4
    assert summary["heldout_pde_solves"] == 8
    assert summary["number_of_pde_solves"] == sum(
        summary[k]
        for k in (
            "baseline_pde_solves",
            "calibration_pde_solves",
            "heldout_pde_solves",
            "radius_pde_solves",
            "refinement_pde_solves",
        )
    )
    assert summary["max_conservation_error"] < 1e-9
    assert summary["number_of_pde_solves"] == repeat["number_of_pde_solves"]
    assert (first / "metrics.csv").read_text() == (second / "metrics.csv").read_text()
    assert (first / "scores.csv").read_text() == (second / "scores.csv").read_text()
    assert (first / "source_manifest.json").is_file()
    assert (first / "figures/predictive_diagnostic.png").is_file()
    assert (first / "report.md").is_file()
    with (first / "metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    training = [r for r in rows if r["role"] == "calibration"]
    heldout = [r for r in rows if r["role"] == "heldout"]
    assert {r["geometry_hash"] for r in training}.isdisjoint(r["geometry_hash"] for r in heldout)
    assert {r["stream_seed"] for r in training} == {"120000"}
    assert {r["stream_seed"] for r in heldout} == {"121000"}
    assert len({r["initial_Q"] for r in rows}) == 1
    for row in rows:
        assert abs(float(row["area"]) - config.target_area) < 1e-12
        assert float(row["robin_perimeter"]) <= config.perimeter_limit
        output = first / "cases" / row["sample_id"]
        assert (output / "final_geometry.npy").is_file()
        np.testing.assert_array_equal(
            np.load(output / "initial_geometry.npy"), np.load(first / "initial_geometry.npy")
        )


def test_infeasible_pools_keep_quotas_and_do_not_invent_refinement_costs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "oxygen_shape.experiments.prediction_diagnostic.propose_paired_disk",
        lambda *args, **kwargs: DiskEdit(None, {}, "Deliberate infeasibility"),
    )
    directory, summary = run_prediction_diagnostic(
        PredictionConfig(
            output_root=tmp_path,
            replicates=1,
            shapes=["rectangle"],
            calibration_sizes=[2],
            heldout_samples=4,
            max_attempts_per_pool=4,
            radius_families=1,
            refinement_factors=[2],
        )
    )
    assert summary["status"] == "incomplete_quotas"
    assert summary["number_of_pde_solves"] == summary["baseline_pde_solves"] == 1
    assert summary["refinement_pde_solves"] == 0
    assert summary["proposal_attempts"] == 11
    assert summary["valid_edits"] == 0
    assert json.loads((directory / "status.json").read_text())["status"] == "incomplete_quotas"


def test_numerical_failure_keeps_attempt_log_and_propagates(tmp_path, monkeypatch):
    calls = []

    def failed_candidate(*args):
        calls.append(1)
        if len(calls) > 1:
            raise ArithmeticError("Injected conservation failure")
        return solve(*args)

    monkeypatch.setattr("oxygen_shape.experiments.prediction_diagnostic.solve", failed_candidate)
    config = PredictionConfig(
        output_root=tmp_path,
        replicates=1,
        shapes=["deep_cavity"],
        calibration_sizes=[2],
        heldout_samples=4,
        max_attempts_per_pool=100,
        refinement_factors=[2],
    )
    with pytest.raises(ArithmeticError, match="conservation"):
        run_prediction_diagnostic(config)
    output = next(tmp_path.iterdir())
    assert (output / "attempts.csv").is_file()
    assert "Injected conservation failure" in (output / "run.log").read_text()
    assert json.loads((output / "status.json").read_text())["status"] == "failed"


@pytest.mark.parametrize(
    "update",
    [
        {"calibration_sizes": []},
        {"calibration_sizes": [4, 2]},
        {"calibration_sizes": [2, 2]},
        {"heldout_samples": 0},
        {"matched_radii": [0.01, 0.08]},
        {"matched_radii": [0.08, 0.05]},
        {"model": {"Bi": 0}},
        {"replicates": 101},
        {"radius_bandwidth": 0},
        {"shapes": []},
    ],
)
def test_invalid_prediction_configs(update):
    with pytest.raises(ValueError):
        PredictionConfig(**update)
