import json

import numpy as np
import pandas as pd
import pytest

from oxygen_shape.geometry.research_families import make_research_geometry
from oxygen_shape.optimization.trajectory_ranker import (
    RANK_FEATURE_NAMES,
    ResponseRanker,
    source_path_distances,
)
from oxygen_shape.optimization.trajectory_search import (
    TrajectoryConfig,
    annealing_probability,
    run_trajectory,
)


def test_access_feature_detects_long_path_in_maze():
    rectangle = make_research_geometry("rectangle").geometry
    maze = make_research_geometry("maze").geometry
    assert (
        source_path_distances(maze)[maze.inside].max()
        > 2 * source_path_distances(rectangle)[rectangle.inside].max()
    )


def test_ranker_learns_signed_global_responses():
    ranker = ResponseRanker(0.01)
    features = np.zeros((2, len(RANK_FEATURE_NAMES)))
    features[:, 0] = 1
    features[:, -1] = [-1, 1]
    for _ in range(20):
        ranker.update(features[0], -0.5)
        ranker.update(features[1], 0.5)
    np.testing.assert_allclose(ranker.scores(features), [-0.5, 0.5], atol=0.001)
    assert ranker.updates == 40


def test_annealing_can_cross_a_flux_decrease():
    assert annealing_probability(0.01, 0.5, 0.001) == 1
    assert annealing_probability(-0.001, 0.5, 0.01) > annealing_probability(-0.001, 0.5, 0.001) > 0


def test_atomic_resume_reproduces_long_trajectory_without_extra_solves(tmp_path, monkeypatch):
    import oxygen_shape.optimization.trajectory_search as module

    config = TrajectoryConfig(solve_budget=8, raw_draw_budget=10000, restart_interval=4)
    geometry = make_research_geometry("rectangle").geometry
    actual = module.solve
    calls = []

    def counted(*args):
        calls.append(1)
        return actual(*args)

    monkeypatch.setattr(module, "solve", counted)
    full = run_trajectory(config, geometry, "learned_anneal", tmp_path / "full")
    assert len(calls) == 8

    def interrupt(row):
        if row["pde_solves"] == 3:
            raise InterruptedError("Simulated shutdown after a committed step")

    with pytest.raises(InterruptedError):
        run_trajectory(
            config, geometry, "learned_anneal", tmp_path / "resumed", on_commit=interrupt
        )
    assert len(calls) == 11
    resumed = run_trajectory(config, geometry, "learned_anneal", tmp_path / "resumed")
    assert len(calls) == 16
    assert full["best_Q"] == resumed["best_Q"]
    assert resumed["model_updates"] == 7
    a = pd.read_csv(tmp_path / "full/metrics.csv")
    b = pd.read_csv(tmp_path / "resumed/metrics.csv")
    for key in ("current_Q", "best_Q", "candidate_Q", "raw_draws", "delta_Q", "prediction"):
        np.testing.assert_array_equal(a[key], b[key])
    assert np.all(np.diff(a.best_Q) >= 0)
    before = (tmp_path / "resumed/summary.json").read_bytes()
    run_trajectory(config, geometry, "learned_anneal", tmp_path / "resumed")
    assert len(calls) == 16
    assert before == (tmp_path / "resumed/summary.json").read_bytes()
    with pytest.raises(ValueError, match="identity"):
        run_trajectory(
            config.model_copy(update={"seed": 99}), geometry, "learned_anneal", tmp_path / "resumed"
        )
    assert json.loads(before)["status"] == "complete"
