import json

import numpy as np
import pytest

from oxygen_shape.experiments.candidate_storage import archive_uncommitted_rollout
from oxygen_shape.geometry.baselines import matched_area
from oxygen_shape.optimization.candidate_policy import CandidatePolicy
from oxygen_shape.optimization.candidate_rollout import run_candidate_rollout
from oxygen_shape.optimization.feasible_pool import run_feasible_search
from oxygen_shape.optimization.pioneer_config import PioneerRLConfig


def refill_config(**updates):
    return PioneerRLConfig(
        seed=230001,
        solve_budget=3,
        raw_draw_budget=768,
        candidate_screening="full",
        refill_quota=4,
        discount=1,
        invalid_reward=0,
        **updates,
    )


def test_refill_rl_preserves_validated_flux_control_and_charges_pool_work():
    cfg = refill_config()
    geometry = matched_area("deep_cavity", 100, 0.24).geometry
    control = run_feasible_search(cfg, geometry, quota=4, ranking="flux")
    policy = CandidatePolicy.with_flux_prior(1)
    before = policy.digest()
    result = run_candidate_rollout(cfg, geometry, method="learned", policy=policy)
    assert result.status == control.status == "complete"
    assert policy.digest() == before
    assert result.search.attempts == control.search.attempts
    assert result.full_geometry_checks == control.full_checks
    np.testing.assert_array_equal(
        result.search.final_geometry.inside,
        control.search.final_geometry.inside,
    )
    assert result.search.attempts == sum(t["pool_draws"] for t in result.transitions)
    assert result.search.attempts == len(result.search.proposals)
    assert any(t["eligible_candidates"] >= 4 for t in result.transitions)


def test_real_refill_learning_replays_updates_and_reward_is_global_flux_gain(tmp_path):
    cfg = refill_config()
    geometry = matched_area("deep_cavity", 100, 0.24).geometry
    first, second = CandidatePolicy.with_flux_prior(1), CandidatePolicy.with_flux_prior(1)
    a = run_candidate_rollout(
        cfg, geometry, method="learned", policy=first, training=True, epsilon=1
    )
    b = run_candidate_rollout(
        cfg, geometry, method="learned", policy=second, training=True, epsilon=1
    )
    assert a.transitions == b.transitions
    assert first.digest() == second.digest()
    assert first.updates == len(a.transitions)
    assert np.linalg.norm(first.weights) > 0
    assert a.transitions[-1]["terminal"] and a.transitions[-1]["bootstrap"] == 0
    assert sum(t["reward"] for t in a.transitions) == pytest.approx(
        100
        * (
            a.search.final_solution.metrics.physical_flux_Q
            / a.search.initial_solution.metrics.physical_flux_Q
            - 1
        ),
        abs=1e-12,
    )
    assert a.search.initial_geometry.source_faces == a.search.final_geometry.source_faces
    assert a.maximum_conservation_error < 1e-9
    checkpoint = tmp_path / "policy.npz"
    first.save(checkpoint, "refill", [{"episode": 0}])
    restored, records = CandidatePolicy.load(checkpoint, "refill")
    assert restored.digest() == first.digest() and records == [{"episode": 0}]


def test_refill_truncation_has_no_free_bootstrap_draws(monkeypatch):
    import oxygen_shape.optimization.feasible_pool as module
    from oxygen_shape.geometry.disk_edits import DiskEdit

    cfg = refill_config().model_copy(update={"raw_draw_budget": 10})
    geometry = matched_area("deep_cavity", 100, 0.24).geometry
    draws = []

    def invalid(*args):
        draws.append(1)
        return [DiskEdit(None, {}, "invalid")], [
            {"addition_face": 0, "donor_face": 0, "radius": 0.05}
        ]

    monkeypatch.setattr(module, "draw_pool", invalid)
    policy = CandidatePolicy()
    result = run_candidate_rollout(cfg, geometry, method="learned", policy=policy, training=True)
    assert len(draws) == result.search.attempts == 10
    assert result.status == "incomplete" and result.search.number_of_pde_solves == 1
    assert policy.updates == 1 and result.transitions[0]["bootstrap"] == 0


def test_resume_preserves_uncommitted_artifacts_and_counts_discarded_work(tmp_path):
    path = tmp_path / "episode_0000"
    (path / "candidates").mkdir(parents=True)
    payload = b"a completed atomic candidate file"
    (path / "candidates/solve_002.npz").write_bytes(payload)
    archive_uncommitted_rollout(path)
    assert not path.exists()
    archive = next((tmp_path / "interrupted_attempts").iterdir())
    assert (archive / "candidates/solve_002.npz").read_bytes() == payload
    cost = json.loads((archive / "interruption_cost.json").read_text())
    assert cost["pde_solves_lower_bound"] == 1 and cost["pde_solves_upper_bound"] == 3


def test_refill_protocol_requires_full_screening_and_finite_cap():
    with pytest.raises(ValueError, match="Refill requires"):
        PioneerRLConfig(refill_quota=4)
    with pytest.raises(ValueError, match="Refill requires"):
        PioneerRLConfig(refill_quota=4, candidate_screening="full", refill_draw_limit=4)
