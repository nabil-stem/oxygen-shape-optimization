import json

import numpy as np
import pytest
from test_candidate_rl import fake_bank

from oxygen_shape.experiments.learning_diagnostics import incumbent_solution, replay_prefix
from oxygen_shape.experiments.learning_duration import run_duration
from oxygen_shape.experiments.pioneer_training import run_pioneer_training
from oxygen_shape.geometry.generators import rectangle
from oxygen_shape.optimization.candidate_policy import CandidatePolicy
from oxygen_shape.optimization.duration_config import DurationConfig
from oxygen_shape.optimization.pioneer_config import PioneerRLConfig
from oxygen_shape.solvers.finite_volume import solve


def test_exploration_prefix_is_unchanged_and_then_clamped():
    config = DurationConfig(reference_run=".", reference_identity="a" * 64)
    old = PioneerRLConfig()
    for episode in range(24):
        expected = old.epsilon_start + (old.epsilon_end - old.epsilon_start) * episode / 23
        assert config.exploration(episode) == expected
    assert config.exploration(24) == config.exploration(95) == pytest.approx(0.15)
    with pytest.raises(ValueError, match="sorted checkpoints"):
        DurationConfig(
            reference_run=".", reference_identity="a" * 64, checkpoint_episodes=[24, 12, 96]
        )
    with pytest.raises(ValueError, match="Development seeds"):
        DurationConfig(
            reference_run=".", reference_identity="a" * 64, development_rollout_seeds=[50000]
        )


def test_saved_incumbent_flux_reconstruction_matches_solver():
    geometry = rectangle(12, 6, 1 / 12)
    cfg = PioneerRLConfig()
    result = solve(geometry, cfg.model, cfg.solver)
    rebuilt = incumbent_solution(geometry, result.concentration, result.metrics.__dict__, 1)
    assert rebuilt.faces == result.faces
    np.testing.assert_array_equal(rebuilt.outward_face_flux, result.outward_face_flux)


def test_continuation_reuses_exact_prefix_checkpoints_and_resumes_interruption(
    tmp_path, monkeypatch
):
    import oxygen_shape.experiments.learning_duration as duration
    import oxygen_shape.experiments.pioneer_training as pioneer

    def ready(root, *args):
        row = {"status": "ready", "number_of_pde_solves": 0, "max_conservation_error": 0}
        (root / "readiness.json").write_text(json.dumps(row))
        return row

    cfg = PioneerRLConfig(
        families=["deep_cavity"],
        replica_seeds=[50000],
        episodes=1,
        test_generation_seeds=[62000],
        test_rollout_seeds=[72000],
        solve_budget=2,
        raw_draw_budget=1000,
        output_root=tmp_path,
    )
    monkeypatch.setattr(pioneer, "prepare_bank", fake_bank)
    monkeypatch.setattr(pioneer, "readiness", ready)
    reference, _ = run_pioneer_training(cfg, progress=None)
    identity = json.loads((reference / "run_identity.json").read_text())["identity"]
    (reference / "artifact_audit.json").write_text(json.dumps({"passed": True}))
    replay, _, _, _ = replay_prefix(reference, 50000, 1)
    original, _ = CandidatePolicy.load(reference / "policies/replica_50000.npz", identity)
    assert replay.digest() == original.digest()
    config = DurationConfig(
        **{
            **cfg.model_dump(),
            "episodes": 2,
            "reference_run": reference,
            "reference_identity": identity,
            "reused_episodes": 1,
            "exploration_episodes": 1,
            "checkpoint_episodes": [1, 2],
            "development_rollout_seeds": [73000],
        }
    )
    monkeypatch.setattr(duration, "readiness", ready)
    real = duration.run_stored_rollout

    def interrupt(*args, **kwargs):
        if kwargs.get("training"):
            raise RuntimeError("simulated interruption after prefix evaluation")
        return real(*args, **kwargs)

    monkeypatch.setattr(duration, "run_stored_rollout", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_duration(config, progress=None)
    root = next(tmp_path.glob("*_learning_duration_*"))
    assert not (root / "evaluation_index.json").exists()

    def interrupt_evaluation(*args, **kwargs):
        if args[0].name.endswith("_e2"):
            raise RuntimeError("simulated evaluation interruption")
        return real(*args, **kwargs)

    monkeypatch.setattr(duration, "run_stored_rollout", interrupt_evaluation)
    with pytest.raises(RuntimeError, match="evaluation interruption"):
        run_duration(config, resume=root, progress=None)
    completed = json.loads((root / "evaluation_index.json").read_text())
    assert len(completed) == 4
    monkeypatch.setattr(duration, "run_stored_rollout", real)
    _, summary = run_duration(config, resume=root, progress=None)
    assert summary["training_episodes"] == 2 and summary["new_training_episodes"] == 1
    assert summary["evaluation_runs"] == 6 and summary["fresh_final_test_runs"] == 0
    assert summary["number_of_new_pde_solves"] == 38
    new_identity = summary["identity"]
    prefix, _ = CandidatePolicy.load(
        root / "checkpoints/episode_0001/replica_50000.npz", new_identity
    )
    assert prefix.digest() == original.digest()

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed resume must not perform new PDE work")

    monkeypatch.setattr(duration, "run_stored_rollout", forbidden)
    monkeypatch.setattr(duration, "readiness", forbidden)
    monkeypatch.setattr(duration, "validate_refinement", forbidden)
    _, again = run_duration(config, resume=root, progress=None)
    assert again["number_of_new_pde_solves"] == 38
    with pytest.raises(ValueError, match="configuration"):
        run_duration(config.model_copy(update={"learning_rate": 0.01}), resume=root, progress=None)
