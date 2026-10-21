import json

import numpy as np
import pytest

from oxygen_shape.geometry.research_families import make_research_geometry
from oxygen_shape.optimization.trajectory_search import TrajectoryConfig, run_trajectory
from scripts.audit_morphology_journals import audit_job


@pytest.mark.parametrize("method", ["flux_greedy", "learned_anneal"])
def test_saved_flux_decisions_and_exact_regression_updates_replay(tmp_path, method):
    initial = make_research_geometry("rectangle").geometry
    config = TrajectoryConfig(solve_budget=7, restart_interval=3)
    run_trajectory(config, initial, "learned_anneal", tmp_path / "warm")
    warm = tmp_path / "warm/response_model.npz" if method.startswith("learned") else None
    result = run_trajectory(config, initial, method, tmp_path / "evaluation", warm_start=warm)
    audit = audit_job(tmp_path / "evaluation", initial, warm)
    assert audit["PDE_solves_reconstructed"] == 7
    assert audit["new_parameter_updates_replayed"] == result["new_model_updates"]
    assert audit["maximum_conservation_error"] < 1e-9


def test_audit_rejects_a_falsified_learning_label(tmp_path):
    initial = make_research_geometry("rectangle").geometry
    run_trajectory(TrajectoryConfig(solve_budget=3), initial, "learned_anneal", tmp_path)
    path = tmp_path / "states/state_000002.npz"
    with np.load(path, allow_pickle=False) as state:
        arrays = {key: state[key].copy() for key in state.files}
    meta = json.loads(str(arrays["metadata"]))
    meta["row"]["target"] += 1
    arrays["metadata"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **arrays)
    with pytest.raises(ArithmeticError, match="signed learning label"):
        audit_job(tmp_path, initial)
