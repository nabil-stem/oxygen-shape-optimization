import hashlib
import json

import numpy as np
import pytest

from oxygen_shape.experiments.revision_state import validation_selection
from oxygen_shape.experiments.revision_training import run_revision_study
from oxygen_shape.geometry.baselines import matched_area
from oxygen_shape.geometry.disk_edits import DiskEdit
from oxygen_shape.optimization.candidate_policy import FEATURE_COUNT, CandidatePolicy
from oxygen_shape.optimization.candidate_rollout import run_candidate_rollout
from oxygen_shape.optimization.pioneer_config import PioneerRLConfig
from oxygen_shape.optimization.revision_config import RevisionArm, RevisionConfig


def test_undiscounted_return_telescopes_in_bounded_real_episode():
    config = PioneerRLConfig(
        seed=12, solve_budget=6, raw_draw_budget=512, discount=1, invalid_reward=0
    )
    initial = matched_area("deep_cavity", 100, 0.24).geometry
    result = run_candidate_rollout(
        config, initial, method="learned", policy=CandidatePolicy(), training=True, epsilon=0.5
    )
    reward = sum(t["reward"] for t in result.transitions)
    physical_gain = config.reward_scale * (
        result.search.final_solution.metrics.physical_flux_Q
        / result.search.initial_solution.metrics.physical_flux_Q
        - 1
    )
    assert reward == pytest.approx(physical_gain, abs=1e-12)
    assert result.transitions[-1]["terminal"]
    assert result.transitions[-1]["bootstrap"] == 0
    assert len(result.transitions) <= config.raw_draw_budget
    q = np.array([2, 2, 2.1, 2.1, 2.6])
    assert (100 * np.diff(q) / q[0]).sum() == pytest.approx(30)
    with pytest.raises(ValueError):
        PioneerRLConfig(discount=1.01)


def test_prior_initial_policy_matches_flux_and_update_includes_prior(tmp_path):
    policy = CandidatePolicy.with_flux_prior()
    features = np.random.default_rng(5).normal(size=(8, FEATURE_COUNT))
    np.testing.assert_array_equal(policy.scores(features), features[:, 16])
    phi = np.eye(FEATURE_COUNT)[16]
    delta = policy.update(phi, 3, 999, alpha=0.4, gamma=1, terminal=True)
    assert delta == 2 and policy.weights[16] == pytest.approx(0.4)
    assert policy.prior_weights[16] == 1
    assert policy.scores(phi[None])[0] == pytest.approx(1.4)
    file = tmp_path / "prior.npz"
    policy.save(file, "identity", [{"episode": 0}])
    restored, records = CandidatePolicy.load(file, "identity")
    assert restored.digest() == policy.digest() and records == [{"episode": 0}]
    np.testing.assert_array_equal(restored.prior_weights, policy.prior_weights)


def test_original_checkpoint_without_prior_still_loads_with_same_digest(tmp_path):
    weights = np.arange(FEATURE_COUNT, dtype=float) / 10
    file = tmp_path / "old.npz"
    from oxygen_shape.optimization.candidate_policy import FEATURE_NAMES

    np.savez(
        file,
        schema="candidate_expected_sarsa_v1",
        feature_names=FEATURE_NAMES,
        weights=weights,
        updates=3,
        identity="old",
        records="[]",
    )
    policy, _ = CandidatePolicy.load(file, "old")
    assert not policy.prior_weights.any()
    assert policy.digest() == hashlib.sha256(weights.tobytes() + b"3").hexdigest()


def test_full_screening_checks_geometry_once_masks_bad_edits_and_charges_pool(monkeypatch):
    import oxygen_shape.optimization.candidate_rollout as module

    config = PioneerRLConfig(seed=2, solve_budget=2, raw_draw_budget=8, candidate_screening="full")
    good = matched_area("deep_cavity", 100, 0.24).geometry
    bad = good.with_mask(good.inside.copy())
    calls = []

    def validate(geometry, initial, cfg):
        calls.append(geometry)
        return "Invalid feature" if geometry is bad else ""

    def draw(geometry, rng, cfg, prepared, size):
        return [DiskEdit(good if i % 2 == 0 else bad, {"radius": 0.06}) for i in range(size)], [
            {"slot": i, "addition_face": 0, "donor_face": 0, "radius": 0.06} for i in range(size)
        ]

    # Different bytes are needed to test independent cache entries.
    changed = good.inside.copy()
    changed[0, 0] = True
    bad = good.with_mask(changed)
    monkeypatch.setattr(module, "validate_candidate", validate)
    monkeypatch.setattr(module, "cheap_screen", lambda *a: ("", 3.5))
    monkeypatch.setattr(module, "draw_pool", draw)
    result = run_candidate_rollout(config, good, method="untrained")
    assert result.search.number_of_pde_solves == 2 and result.search.attempts == 8
    assert result.full_geometry_checks == 2 and len(calls) == 3  # includes initial validation
    assert result.transitions[0]["action"] % 2 == 0
    assert sum(r["validation_cache_hit"] for r in result.search.proposals) == 6
    assert all(r["selection_probability"] == 0 for r in result.search.proposals if r["slot"] % 2)


def test_full_screened_prior_matches_frozen_flux_and_zero_control_matches_uniform():
    initial = matched_area("deep_cavity", 100, 0.24).geometry
    config = PioneerRLConfig(
        seed=7, solve_budget=3, raw_draw_budget=256, candidate_screening="full"
    )
    a = run_candidate_rollout(config, initial, method="flux_pool")
    policy = CandidatePolicy.with_flux_prior()
    before = policy.digest()
    b = run_candidate_rollout(config, initial, method="learned", policy=policy)
    assert before == policy.digest()
    assert a.transitions == b.transitions
    np.testing.assert_array_equal(a.search.final_geometry.inside, b.search.final_geometry.inside)
    c = run_candidate_rollout(config, initial, method="screened_uniform")
    d = run_candidate_rollout(config, initial, method="untrained")
    assert c.transitions == d.transitions


@pytest.mark.parametrize("changed_label", ["source", "neumann"])
def test_full_screen_cache_includes_boundary_labels(monkeypatch, changed_label):
    import oxygen_shape.optimization.candidate_rollout as module
    from oxygen_shape.geometry.raster import FaceKey, RasterGeometry, Side

    good = matched_area("deep_cavity", 100, 0.24).geometry
    bad = RasterGeometry(
        good.inside,
        good.h,
        () if changed_label == "source" else good.source_faces,
        () if changed_label == "source" else (FaceKey(0, 0, Side.WEST),),
    )

    def validate(geometry, initial, cfg):
        return "Boundary labels changed" if geometry is bad else ""

    def draw(geometry, rng, cfg, prepared, size):
        return [DiskEdit(good if i % 2 == 0 else bad, {"radius": 0.06}) for i in range(size)], [
            {"slot": i, "addition_face": 0, "donor_face": 0, "radius": 0.06} for i in range(size)
        ]

    monkeypatch.setattr(module, "validate_candidate", validate)
    monkeypatch.setattr(module, "cheap_screen", lambda *args: ("", 3.5))
    monkeypatch.setattr(module, "draw_pool", draw)
    config = PioneerRLConfig(solve_budget=2, raw_draw_budget=8, candidate_screening="full")
    result = run_candidate_rollout(config, good, method="untrained")
    assert result.full_geometry_checks == 2
    assert all(r["selection_probability"] == 0 for r in result.search.proposals if r["slot"] % 2)


def selection_config():
    return RevisionConfig(
        families=["deep_cavity"],
        replica_seeds=[150000],
        arms=[
            RevisionArm(name="a", discount=1, invalid_reward=0),
            RevisionArm(name="b", discount=1, invalid_reward=0),
        ],
        test_core_arms=["a"],
        validation_rollout_seeds=[171000],
        test_generation_seeds=[162000],
        test_rollout_seeds=[172000],
    )


def test_selection_uses_only_validation_and_all_replica_quota():
    config = selection_config()
    rows = [
        {
            "split": "validation",
            "arm": a,
            "status": "complete",
            "fine_gain_Q": gain,
            "fine_final_Q": 1 + gain,
        }
        for a, gain in [("a", 0.1), ("b", 0.2)]
    ]
    rows.append(
        {
            "split": "test",
            "arm": "a",
            "status": "complete",
            "fine_gain_Q": 999,
            "fine_final_Q": 1000,
        }
    )
    selected = validation_selection(config, rows, {})
    assert selected["selected_arm"] == "b" and selected["test_arms"] == ["a", "b"]
    rows[1]["status"] = "incomplete"
    assert validation_selection(config, rows, {})["selected_arm"] == "a"
    rows[0]["status"] = "incomplete"
    assert validation_selection(config, rows, {})["selected_arm"] is None


def test_revision_protocol_rejects_invalid_arms_and_seed_streams():
    with pytest.raises(ValueError):
        RevisionConfig(replica_seeds=[-1])
    with pytest.raises(ValueError, match="distinct"):
        RevisionConfig(arms=[RevisionArm(name="x", discount=1, invalid_reward=0)] * 2)
    with pytest.raises(ValueError):
        RevisionConfig(test_core_arms=["unknown"])


def test_revision_end_to_end_selection_freeze_and_zero_solve_resume(tmp_path, monkeypatch):
    from test_candidate_rl import fake_bank

    import oxygen_shape.experiments.revision_training as module

    config = selection_config().model_copy(
        update={"output_root": tmp_path, "episodes": 1, "solve_budget": 2, "raw_draw_budget": 256}
    )
    monkeypatch.setattr(module, "prepare_bank", fake_bank)

    def ready(root, config, starts):
        value = {"status": "ready", "number_of_pde_solves": 0, "max_conservation_error": 0}
        (root / "readiness.json").write_text(json.dumps(value))
        return value

    monkeypatch.setattr(module, "revision_preflight", ready)
    monkeypatch.setattr(
        module,
        "REVISION_CONTROLS",
        {"flux_pool": ("flux_pool", "cheap"), "flux_pool_full": ("flux_pool", "full")},
    )
    root, prepared = run_revision_study(config, prepare_only=True)
    assert prepared["status"] == "ready"
    root, summary = run_revision_study(config, resume=root)
    assert summary["models"] == 2 and summary["training_episodes"] == 2
    records = json.loads((root / "evaluation_index.json").read_text())
    assert len({r["job"] for r in records}) == len(records)
    assert {r["method"] for r in records if r["arm"] is not None} == {"a__150000", "b__150000"}
    selection = json.loads((root / "selection.json").read_text())
    assert selection["individual_replica_selection"] is False
    assert "a" in selection["test_arms"]
    assert summary["number_of_pde_solves"] == sum(
        summary[k]
        for k in [
            "preparation_pde_solves",
            "preflight_pde_solves",
            "training_pde_solves",
            "validation_pde_solves",
            "test_pde_solves",
            "refinement_pde_solves",
        ]
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed jobs must not be recomputed")

    monkeypatch.setattr(module, "run_stored_rollout", forbidden)
    _, again = run_revision_study(config, resume=root)
    assert again["number_of_pde_solves"] == summary["number_of_pde_solves"]
    with pytest.raises(ValueError, match="identity"):
        run_revision_study(config.model_copy(update={"learning_rate": 0.1}), resume=root)
    assert json.loads((root / "status.json").read_text())["status"] == summary["status"]
