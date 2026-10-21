import json

import numpy as np
import pandas as pd
import pytest

from oxygen_shape.geometry.research_families import make_research_geometry
from oxygen_shape.optimization.trajectory_search import TrajectoryConfig
from scripts.audit_conventional import audit_job
from scripts.conventional_search import run_classical


@pytest.mark.parametrize("method", ["uniform_greedy", "pool_greedy"])
def test_charged_candidates_and_exact_mid_batch_resume(tmp_path, monkeypatch, method):
    import scripts.conventional_search as module

    config = TrajectoryConfig(solve_budget=9, raw_draw_budget=10000)
    initial = make_research_geometry("hexagon").geometry
    actual = module.solve
    calls = []

    def counted(*args):
        calls.append(1)
        return actual(*args)

    monkeypatch.setattr(module, "solve", counted)
    full = run_classical(config, initial, method, tmp_path / "full")
    assert len(calls) == 9
    audit = audit_job(tmp_path / "full", initial)
    assert audit["candidate_fields"] == 9
    assert audit["additional_PDE_solves"] == 0
    assert len(calls) == 9

    def interrupt(row):
        if row["pde_solves"] == 2:
            raise InterruptedError("Stop after committed candidate, possibly within a batch")

    with pytest.raises(InterruptedError):
        run_classical(config, initial, method, tmp_path / "resumed", on_commit=interrupt)
    assert len(calls) == 11
    if method == "pool_greedy":
        with np.load(tmp_path / "resumed/states/state_000002.npz", allow_pickle=False) as state:
            assert len(state["pending_masks"]) > 0
    resumed = run_classical(config, initial, method, tmp_path / "resumed")
    assert len(calls) == 18
    assert full["best_Q"] == resumed["best_Q"]
    for name in ("final_geometry.npy", "concentration.npy"):
        np.testing.assert_array_equal(
            np.load(tmp_path / "full" / name), np.load(tmp_path / "resumed" / name)
        )
    a = pd.read_csv(tmp_path / "full/metrics.csv", float_precision="round_trip")
    b = pd.read_csv(tmp_path / "resumed/metrics.csv", float_precision="round_trip")
    for key in ("current_Q", "candidate_Q", "best_Q", "batch_id", "raw_draws", "accepted"):
        np.testing.assert_array_equal(a[key], b[key])
    assert np.all(np.diff(a.best_Q) >= 0)
    if method == "pool_greedy":
        assert a.batch_id.nunique() < len(a)
        for _, group in a.iloc[1:].groupby("batch_id"):
            assert group.current_Q.iloc[-1] >= group.candidate_Q.max() - config.min_improvement
    else:
        assert a.batch_id.nunique() == len(a)
    before = (tmp_path / "resumed/summary.json").read_bytes()
    run_classical(config, initial, method, tmp_path / "resumed")
    assert len(calls) == 18
    assert before == (tmp_path / "resumed/summary.json").read_bytes()
    assert json.loads(before)["status"] == "complete"


def test_last_pool_is_explicitly_truncated_at_remaining_charged_budget(tmp_path, monkeypatch):
    import scripts.conventional_search as module

    actual = module.solve
    calls = []

    def counted(*args):
        calls.append(1)
        return actual(*args)

    monkeypatch.setattr(module, "solve", counted)
    initial = make_research_geometry("hexagon").geometry
    result = run_classical(TrajectoryConfig(solve_budget=2), initial, "pool_greedy", tmp_path)
    assert result["pde_solves"] == len(calls) == 2
    history = pd.read_csv(tmp_path / "metrics.csv")
    last = history.iloc[-1]
    assert last.batch_budget_truncated
    assert last.batch_available_choices >= 2
    assert last.batch_scheduled_evaluations == 1


def test_classical_audit_rejects_falsified_acceptance(tmp_path):
    initial = make_research_geometry("hexagon").geometry
    run_classical(TrajectoryConfig(solve_budget=3), initial, "uniform_greedy", tmp_path)
    file = tmp_path / "states/state_000002.npz"
    with np.load(file, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    metadata = json.loads(str(arrays["metadata"]))
    metadata["row"]["accepted"] = not metadata["row"]["accepted"]
    arrays["metadata"] = np.array(json.dumps(metadata))
    np.savez_compressed(file, **arrays)
    with pytest.raises(ValueError, match="acceptance mismatch"):
        audit_job(tmp_path, initial)


def test_uncommitted_mid_pool_solve_is_bounded_and_committed_work_is_reused(tmp_path, monkeypatch):
    import scripts.conventional_search as module

    config = TrajectoryConfig(solve_budget=9)
    initial = make_research_geometry("hexagon").geometry
    actual_solve, actual_atomic = module.solve, module._atomic
    calls = []

    def counted(*args):
        calls.append(1)
        return actual_solve(*args)

    monkeypatch.setattr(module, "solve", counted)
    run_classical(config, initial, "pool_greedy", tmp_path / "full")

    def fail_commit(path, content):
        if path.name == "state_000003.npz":
            raise OSError("Simulated loss after solve but before atomic commit")
        return actual_atomic(path, content)

    monkeypatch.setattr(module, "_atomic", fail_commit)
    with pytest.raises(OSError, match="Simulated loss"):
        run_classical(config, initial, "pool_greedy", tmp_path / "resumed")
    assert len(calls) == 12
    monkeypatch.setattr(module, "_atomic", actual_atomic)
    result = run_classical(config, initial, "pool_greedy", tmp_path / "resumed")
    assert result["pde_solves"] == 9
    assert len(calls) == 19  # Two complete runs plus one genuinely lost solve.
    interruptions = list((tmp_path / "resumed").glob("interrupted_*.json"))
    assert len(interruptions) == 1
    loss = json.loads(interruptions[0].read_text())
    assert loss["extra_pde_lower_bound"] == 0
    assert loss["extra_pde_upper_bound"] == 1
    for name in ("final_geometry.npy", "concentration.npy"):
        np.testing.assert_array_equal(
            np.load(tmp_path / "full" / name), np.load(tmp_path / "resumed" / name)
        )
