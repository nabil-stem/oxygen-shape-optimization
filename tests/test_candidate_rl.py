import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from oxygen_shape.experiments.pioneer_bank import PioneerStart, geometry_hash, load_bank
from oxygen_shape.experiments.pioneer_training import run_pioneer_training
from oxygen_shape.geometry.baselines import matched_area
from oxygen_shape.geometry.disk_edits import DiskEdit, prepare_boundary
from oxygen_shape.optimization.candidate_policy import (
    FEATURE_COUNT,
    CandidatePolicy,
    mixed_probabilities,
)
from oxygen_shape.optimization.candidate_rollout import run_candidate_rollout
from oxygen_shape.optimization.guided_search import validate_candidate
from oxygen_shape.optimization.perimeter_pool import draw_pool
from oxygen_shape.optimization.pioneer_config import PioneerRLConfig


def test_expected_sarsa_delayed_reward_terminal_and_nonmax_bootstrap():
    policy = CandidatePolicy()
    first, second, other = np.eye(FEATURE_COUNT)[:3]
    for _ in range(250):
        policy.update(second, 2, 999, alpha=0.2, gamma=0.9, terminal=True)
        policy.update(other, 1, 999, alpha=0.2, gamma=0.9, terminal=True)
        expected = 0.25 * policy.weights[1] + 0.75 * policy.weights[2]
        policy.update(first, 0, expected, alpha=0.2, gamma=0.9, terminal=False)
    assert policy.weights[1] == pytest.approx(2, abs=1e-8)
    assert policy.weights[0] == pytest.approx(0.9 * 1.25, abs=1e-8)
    assert policy.weights[0] < 0.9 * max(policy.weights[1], policy.weights[2])


def test_mixed_policy_support_ties_and_screening():
    p = mixed_probabilities(np.array([100.0, 2, 2]), np.array([False, True, True]), 0, 0.25)
    np.testing.assert_allclose(p, [0.25, 0.375, 0.375])
    p = mixed_probabilities(np.array([10.0, 2, 1]), np.array([False, True, True]), 0, 0.25)
    np.testing.assert_allclose(p, [0.25, 0.75, 0])
    np.testing.assert_allclose(mixed_probabilities(np.zeros(2), np.zeros(2, bool), 0, 0.25), [1, 0])
    with pytest.raises(ValueError):
        mixed_probabilities(np.array([np.nan]), np.array([True]), 0.5, 0.25)


def test_atomic_policy_checkpoint_schema_identity_and_records(tmp_path):
    p = CandidatePolicy(np.arange(FEATURE_COUNT, dtype=float) / 100, 9)
    path = tmp_path / "policy.npz"
    p.save(path, "frozen", [{"episode": 0, "seed": 12}])
    q, records = CandidatePolicy.load(path, "frozen")
    assert p.digest() == q.digest() and records == [{"episode": 0, "seed": 12}]
    with pytest.raises(ValueError, match="Incompatible"):
        CandidatePolicy.load(path, "other")
    assert not list(tmp_path.glob("*.tmp"))


def test_untrained_and_screened_uniform_have_exact_same_seeded_transition_law():
    cfg = PioneerRLConfig(seed=12, solve_budget=3, raw_draw_budget=320)
    g = matched_area("deep_cavity", 100, 0.24).geometry
    a = run_candidate_rollout(cfg, g, method="untrained")
    b = run_candidate_rollout(cfg, g, method="screened_uniform")
    np.testing.assert_array_equal(a.search.final_geometry.inside, b.search.final_geometry.inside)
    assert [(x["action"], x["candidate_Q"]) for x in a.transitions] == [
        (x["action"], x["candidate_Q"]) for x in b.transitions
    ]
    assert a.search.number_of_pde_solves == b.search.number_of_pde_solves


def test_bootstrap_pool_is_charged_once_and_last_truncated_pool_is_processed(monkeypatch):
    import oxygen_shape.optimization.candidate_rollout as module

    cfg = PioneerRLConfig(solve_budget=9, raw_draw_budget=10)
    g = matched_area("deep_cavity", 100, 0.24).geometry
    sizes = []

    def invalid(geometry, rng, config, prepared, size):
        sizes.append(size)
        return [DiskEdit(None, {"radius": 0.06}, "Invalid") for _ in range(size)], [
            {"slot": i, "addition_face": 0, "donor_face": 0, "radius": 0.06} for i in range(size)
        ]

    monkeypatch.setattr(module, "draw_pool", invalid)
    policy = CandidatePolicy()
    r = run_candidate_rollout(cfg, g, method="learned", policy=policy, training=True, epsilon=0.5)
    assert sizes == [8, 2] and r.search.attempts == 10
    assert r.search.number_of_pde_solves == 1 and policy.updates == 2
    assert [t["terminal"] for t in r.transitions] == [False, True]
    assert r.transitions[-1]["bootstrap"] == 0
    assert len(r.search.proposals) == 10


def test_global_flux_rejects_lower_candidate_and_frozen_weights_never_update(monkeypatch):
    import oxygen_shape.optimization.candidate_rollout as module
    from oxygen_shape.solvers.finite_volume import solve

    cfg = PioneerRLConfig(solve_budget=2, raw_draw_budget=8)
    g = matched_area("deep_cavity", 100, 0.24).geometry
    solution = solve(g, cfg.model, cfg.solver)
    calls = []

    def lower(*args):
        calls.append(1)
        return (
            solution
            if len(calls) == 1
            else replace(solution, metrics=replace(solution.metrics, physical_flux_Q=0.1))
        )

    def same(geometry, rng, config, prepared, size):
        return [DiskEdit(g, {"radius": 0.06}) for _ in range(size)], [
            {"slot": i, "addition_face": 0, "donor_face": 0, "radius": 0.06} for i in range(size)
        ]

    monkeypatch.setattr(module, "solve", lower)
    monkeypatch.setattr(module, "draw_pool", same)
    policy = CandidatePolicy()
    before = policy.digest()
    r = run_candidate_rollout(cfg, g, method="learned", policy=policy)
    assert not r.transitions[0]["accepted"] and r.transitions[0]["reward"] == 0
    assert r.search.final_geometry is g and policy.digest() == before


def test_learning_replays_weights_and_features():
    cfg = PioneerRLConfig(seed=4, solve_budget=3, raw_draw_budget=160)
    g = matched_area("deep_cavity", 100, 0.24).geometry
    a, b = CandidatePolicy(), CandidatePolicy()
    first = run_candidate_rollout(cfg, g, method="learned", policy=a, training=True, epsilon=0.5)
    second = run_candidate_rollout(cfg, g, method="learned", policy=b, training=True, epsilon=0.5)
    assert a.digest() == b.digest()
    assert first.transitions == second.transitions
    np.testing.assert_array_equal(first.selected_features, second.selected_features)


def test_protocol_rejects_seed_overlap():
    with pytest.raises(ValueError, match="disjoint"):
        PioneerRLConfig(test_rollout_seeds=[50000])


def fake_bank(root, config, progress=None):
    g = matched_area("deep_cavity", 100, 0.24).geometry
    edits, _ = draw_pool(g, np.random.default_rng(3), config, prepare_boundary(g, 0.12), 64)
    geometries = [g] + [
        e.geometry
        for e in edits
        if e.geometry is not None and not validate_candidate(e.geometry, g, config)
    ][:2]
    starts, rows = [], []
    path = root / "bank"
    path.mkdir()
    for split, geometry in zip(["train", "validation", "test"], geometries, strict=True):
        name = f"{split}_deep_cavity_" + ("canonical" if split == "train" else "heldout")
        filename = f"{name}.npy"
        np.save(path / filename, geometry.inside)
        rows.append(
            {
                "name": name,
                "family": "deep_cavity",
                "split": split,
                "mask_file": filename,
                "mask_sha256": hashlib.sha256((path / filename).read_bytes()).hexdigest(),
                "geometry_hash": geometry_hash(geometry),
            }
        )
        starts.append(PioneerStart(name, "deep_cavity", split, geometry))
    manifest = {
        "schema": 1,
        "n": 100,
        "Bi": 1.0,
        "starts": rows,
        "preparation_pde_solves": 0,
        "max_conservation_error": 0,
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return starts, manifest


def test_end_to_end_training_freeze_and_resume_skips_completed_solves(tmp_path, monkeypatch):
    import oxygen_shape.experiments.pioneer_training as module

    cfg = PioneerRLConfig(
        families=["deep_cavity"],
        replica_seeds=[50000],
        episodes=1,
        test_generation_seeds=[62000],
        test_rollout_seeds=[72000],
        solve_budget=2,
        raw_draw_budget=256,
        output_root=tmp_path,
    )
    monkeypatch.setattr(module, "prepare_bank", fake_bank)

    def ready(root, config, starts):
        result = {"status": "ready", "number_of_pde_solves": 0, "max_conservation_error": 0}
        (root / "readiness.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(module, "readiness", ready)
    root, prepared = run_pioneer_training(cfg, prepare_only=True)
    assert prepared["status"] == "ready" and prepared["number_of_pde_solves"] == 0
    root, summary = run_pioneer_training(cfg, resume=root)
    assert summary["replicas"] == 1 and summary["evaluation_runs"] == 12
    assert summary["number_of_pde_solves"] == sum(
        summary[k]
        for k in [
            "preparation_pde_solves",
            "preflight_pde_solves",
            "training_pde_solves",
            "evaluation_pde_solves",
            "refinement_pde_solves",
        ]
    )
    identity = json.loads((root / "run_identity.json").read_text())["identity"]
    policy, records = CandidatePolicy.load(root / "policies/replica_50000.npz", identity)
    assert len(records) == 1 and policy.updates > 0
    before = policy.digest()
    evaluations = json.loads((root / "evaluation_index.json").read_text())
    assert {r["method"] for r in evaluations} == {
        "uniform",
        "screened_uniform",
        "perimeter_pool",
        "flux_pool",
        "untrained",
        "rl_50000",
    }
    assert len({(r["start"], r["seed"], r["method"]) for r in evaluations}) == 12
    learned_rows = [r for r in evaluations if r["method"] == "rl_50000"]
    assert len(learned_rows) == 2
    assert all(
        r["rollout_method"] == "learned" and r["policy_digest"] == before for r in learned_rows
    )
    assert "| rl_50000 | uniform | 1 |" in (root / "report.md").read_text()

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed resume must not solve again")

    monkeypatch.setattr(module, "run_stored_rollout", forbidden)
    _, again = run_pioneer_training(cfg, resume=root)
    restored, _ = CandidatePolicy.load(root / "policies/replica_50000.npz", identity)
    assert (
        restored.digest() == before
        and again["number_of_pde_solves"] == summary["number_of_pde_solves"]
    )
    index_path = root / "evaluation_index.json"
    original_index = index_path.read_text()
    learned_rows[0]["policy_digest"] = "corrupted"
    index_path.write_text(json.dumps(evaluations))
    with pytest.raises(ValueError, match="different model weights"):
        run_pioneer_training(cfg, resume=root)
    index_path.write_text(original_index)
    with pytest.raises(ValueError, match="configuration"):
        run_pioneer_training(cfg.model_copy(update={"learning_rate": 0.1}), resume=root)
    assert json.loads((root / "status.json").read_text())["status"] == summary["status"]
    starts, manifest = load_bank(root, cfg)
    assert len(starts) == 3
    mask = root / "bank" / manifest["starts"][1]["mask_file"]
    mask.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        load_bank(root, cfg)
