"""Audit saved RL updates, exact selected edits and FV balances without PDE solves."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from audit_feasible_pools import check_field

from oxygen_shape.experiments.config import load_config
from oxygen_shape.experiments.pioneer_bank import load_bank
from oxygen_shape.experiments.pioneer_training import assert_source_unchanged
from oxygen_shape.geometry.baselines import matched_area
from oxygen_shape.geometry.disk_edits import prepare_boundary, propose_paired_disk
from oxygen_shape.geometry.raster import refine_geometry
from oxygen_shape.optimization.candidate_policy import CandidatePolicy
from oxygen_shape.optimization.guided_search import validate_candidate
from oxygen_shape.optimization.pioneer_config import PioneerRLConfig


def audit(root):
    config = load_config(root / "config.yaml", PioneerRLConfig)
    assert_source_unchanged(root)
    starts, bank = load_bank(root, config)
    for row in bank["starts"]:
        if row["split"] != "train":
            assert row["geometry_hash"] not in config.excluded_geometry_hashes
    identity = json.loads((root / "run_identity.json").read_text())["identity"]
    summary = json.loads((root / "summary.json").read_text())
    training = pd.read_csv(root / "training.csv")
    evaluations = pd.read_csv(root / "evaluations.csv")
    assert len(training) == len(config.replica_seeds) * config.episodes
    expected = {
        (s.name, seed, method)
        for s in starts
        if s.split != "train"
        for seed in (
            config.validation_rollout_seeds
            if s.split == "validation"
            else config.test_rollout_seeds
        )
        for method in (
            "uniform",
            "screened_uniform",
            "perimeter_pool",
            "flux_pool",
            "untrained",
            *(f"rl_{seed}" for seed in config.replica_seeds),
        )
    }
    assert (
        set(zip(evaluations.start, evaluations.seed, evaluations.method, strict=True)) == expected
    )
    assert len(evaluations) == len(expected) and (evaluations.status == "complete").all()
    assert (training.status == "complete").all()
    policies, updates = {}, 0
    for seed in config.replica_seeds:
        saved, records = CandidatePolicy.load(root / f"policies/replica_{seed}.npz", identity)
        assert len(records) == config.episodes
        replay = CandidatePolicy.with_flux_prior(config.flux_prior_scale)
        for episode, record in enumerate(records):
            path = root / f"training/replica_{seed}/episode_{episode:04d}"
            transitions = pd.read_csv(path / "transitions.csv")
            features = np.load(path / "selected_features.npy")
            assert len(features) == len(transitions) == record["decisions"]
            for phi, row in zip(features, transitions.to_dict("records"), strict=True):
                delta = replay.update(
                    phi,
                    row["reward"],
                    row["bootstrap"],
                    alpha=config.learning_rate,
                    gamma=config.discount,
                    terminal=row["terminal"],
                )
                assert abs(delta - row["td_error"]) < 1e-11
            assert replay.updates == record["updates"]
        np.testing.assert_allclose(replay.weights, saved.weights, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(replay.prior_weights, saved.prior_weights)
        assert np.linalg.norm(saved.weights) > 0
        policies[f"rl_{seed}"] = saved
        updates += saved.updates
    for row in evaluations.to_dict("records"):
        if row["method"] in policies:
            assert row["policy_digest"] == policies[row["method"]].digest()
    fields = replays = solves = accepted_count = raw_count = full_checks = 0
    max_error = 0.0
    paths = sorted(root.glob("training/replica_*/episode_*"))
    paths += sorted(root.glob("evaluation/*"))
    paths += sorted(root.glob("pioneer_generation/*"))
    paths += [p for p in sorted(root.glob("preflight/*")) if p.is_dir()]
    assert len(paths) == 271
    for index, path in enumerate(paths, 1):
        stored = yaml.safe_load((path / "config.yaml").read_text())
        cfg = PioneerRLConfig(**stored["configuration"])
        row = json.loads((path / "rollout_summary.json").read_text())
        canonical = matched_area(cfg.initial_shape, cfg.n, cfg.target_area).geometry
        initial = canonical.with_mask(np.load(path / "initial_geometry.npy"))
        assert not validate_candidate(initial, canonical, cfg)
        field = np.load(path / "initial_concentration.npy")
        Q, error = check_field(initial, field, cfg.model.Bi)
        assert abs(Q - row["initial_Q"]) < 1e-12
        fields += 1
        max_error = max(max_error, error)
        transitions = pd.read_csv(path / "transitions.csv")
        proposals = pd.read_csv(path / "proposals.csv", keep_default_na=False)
        assert len(transitions) == row["decisions"]
        assert proposals.raw_draw.tolist() == list(range(1, len(proposals) + 1))
        assert len(proposals) == row["raw_draws"] <= cfg.raw_draw_budget
        assert int(proposals.full_checked.sum()) == row["full_geometry_checks"]
        assert int(transitions.pool_draws.sum()) == len(proposals)
        assert transitions.iloc[-1].terminal and transitions.iloc[-1].bootstrap == 0
        assert not transitions.iloc[:-1].terminal.any()
        grouped = proposals.groupby("decision", sort=True)
        assert np.allclose(grouped.selection_probability.sum(), 1, atol=1e-13)
        assert (grouped.selected.sum() == 1).all()
        assert (proposals.selection_probability >= 0).all()
        assert (proposals.loc[proposals.slot == 0, "selection_probability"] >= 0.25 - 1e-14).all()
        current, current_Q, solve_index = initial, Q, 1
        prepared = prepare_boundary(current, cfg.protected_length)
        for transition in transitions.to_dict("records"):
            choices = grouped.get_group(transition["decision"])
            eligible = (choices.screen_rejection == "") & (choices.full_rejection == "")
            assert int(eligible.sum()) == transition["eligible_candidates"]
            assert not (
                (~eligible) & (choices.slot != 0) & (choices.selection_probability > 0)
            ).any()
            selected = choices[choices.selected].iloc[0]
            if pd.isna(transition["candidate_Q"]):
                continue
            local = np.random.default_rng(int(selected.proposal_seed))
            count = len(prepared.points)
            pair = divmod(int(local.integers(count**2)), count)
            radius = float(local.uniform(*cfg.radii))
            edit = propose_paired_disk(
                current, local, radius, cfg.protected_length, face_pair=pair, prepared=prepared
            )
            assert edit.geometry is not None
            assert not validate_candidate(edit.geometry, initial, cfg)
            solve_index += 1
            with np.load(path / f"candidates/solve_{solve_index:03d}.npz") as saved:
                np.testing.assert_array_equal(saved["mask"], edit.geometry.inside)
                candidate_field = saved["concentration"]
                Q, error = check_field(edit.geometry, candidate_field, cfg.model.Bi)
            fields += 1
            replays += 1
            max_error = max(max_error, error)
            assert abs(Q - transition["candidate_Q"]) < 1e-12
            accepted = Q > current_Q + cfg.min_improvement
            assert accepted == transition["accepted"]
            if accepted:
                current, current_Q, field = edit.geometry, Q, candidate_field
                prepared = prepare_boundary(current, cfg.protected_length)
                accepted_count += 1
        assert solve_index == row["number_of_pde_solves"] == cfg.solve_budget
        assert len(list((path / "candidates").glob("*.npz"))) == solve_index - 1
        np.testing.assert_array_equal(np.load(path / "final_geometry.npy"), current.inside)
        np.testing.assert_array_equal(np.load(path / "concentration.npy"), field)
        assert abs(current_Q - row["final_Q"]) < 1e-12
        if cfg.invalid_reward == 0:
            assert (
                abs(
                    transitions.reward.sum() - cfg.reward_scale * (current_Q / row["initial_Q"] - 1)
                )
                < 1e-11
            )
        if (path / "refinement.csv").exists():
            refinement = pd.read_csv(path / "refinement.csv")
            for factor in cfg.refinement_factors:
                for label, geometry in (("initial", initial), ("final", current)):
                    Q, error = check_field(
                        refine_geometry(geometry, factor),
                        np.load(path / f"refinement_{factor}/{label}/concentration.npy"),
                        cfg.model.Bi,
                    )
                    expected_Q = refinement.loc[refinement.factor == factor, f"{label}_Q"].iloc[0]
                    assert abs(Q - expected_Q) < 1e-12
                    fields += 1
                    max_error = max(max_error, error)
                    solves += 1
        solves += solve_index
        raw_count += len(proposals)
        full_checks += row["full_geometry_checks"]
        if index % 30 == 0:
            print(f"Audited {index}/{len(paths)} rollouts", flush=True)
    assert solves == fields == summary["number_of_pde_solves"] - 37
    result = {
        "passed": True,
        "rollouts": len(paths),
        "saved_fields_checked": fields,
        "exact_selected_candidate_replays": replays,
        "policy_updates_replayed": updates,
        "accepted_moves": accepted_count,
        "raw_draws": raw_count,
        "full_geometry_checks": full_checks,
        "max_independent_conservation_error": max_error,
        "new_pde_solves": 0,
        "not_saved_as_fields": {"training_bank_checks": 36, "preflight_slab": 1},
        "source_hashes_unchanged": True,
        "fresh_heldout_exclusions_passed": True,
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (root / "artifact_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    audit(parser.parse_args().run.resolve())
