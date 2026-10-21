"""Read-only policy reconstruction and ranking diagnosis from persisted fields."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from oxygen_shape.experiments.config import load_config
from oxygen_shape.experiments.pioneer_bank import load_bank
from oxygen_shape.experiments.reporting import write_csv
from oxygen_shape.geometry.boundary import BoundaryKind, extract_boundary
from oxygen_shape.geometry.disk_edits import prepare_boundary
from oxygen_shape.optimization.candidate_policy import CandidatePolicy, mixed_probabilities
from oxygen_shape.optimization.candidate_rollout import candidate_features
from oxygen_shape.optimization.donor_sampling import eligible_wall_flux
from oxygen_shape.optimization.pioneer_config import PioneerRLConfig


def replay_prefix(reference: Path, seed: int, episodes: int):
    """Round-trip CSV parsing preserves the exact floats used in online updates."""
    cfg = load_config(reference / "config.yaml", PioneerRLConfig)
    identity = json.loads((reference / "run_identity.json").read_text())["identity"]
    final, records = CandidatePolicy.load(reference / f"policies/replica_{seed}.npz", identity)
    if not 0 < episodes <= len(records):
        raise ValueError("Requested prefix is outside the recorded training")
    policy = CandidatePolicy.with_flux_prior(cfg.flux_prior_scale)
    updates, weights = [], []
    for episode in range(episodes):
        path = reference / f"training/replica_{seed}/episode_{episode:04d}"
        transitions = pd.read_csv(path / "transitions.csv", float_precision="round_trip")
        features = np.load(path / "selected_features.npy", allow_pickle=False)
        if len(transitions) != len(features):
            raise ValueError("Incomplete recorded transitions")
        for phi, row in zip(features, transitions.to_dict("records"), strict=True):
            prediction = float(policy.scores(phi[None])[0])
            delta = policy.update(
                phi,
                row["reward"],
                row["bootstrap"],
                alpha=cfg.learning_rate,
                gamma=cfg.discount,
                terminal=row["terminal"],
            )
            if delta != row["td_error"]:
                raise ArithmeticError("Recorded TD update does not reproduce exactly")
            updates.append(
                {
                    "replica": seed,
                    "episode": episode + 1,
                    "prediction": prediction,
                    "reward": row["reward"],
                    "td_error": delta,
                    "terminal": row["terminal"],
                    "evaluated": not pd.isna(row["candidate_Q"]),
                }
            )
        if policy.updates != records[episode]["updates"]:
            raise ValueError("Prefix update count disagrees with checkpoint")
        weights.append(
            {
                "replica": seed,
                "episode": episode + 1,
                **{f"w_{i}": float(w) for i, w in enumerate(policy.weights)},
            }
        )
    if episodes == len(records) and policy.digest() != final.digest():
        raise ArithmeticError("Full prefix does not reproduce the original policy byte for byte")
    return policy, records[:episodes], updates, weights


def incumbent_solution(geometry, field, metrics, Bi):
    """Rebuild face fluxes from the saved incumbent, without a new PDE solve."""
    faces = extract_boundary(geometry)
    robin = Bi * geometry.h / (1 + Bi * geometry.h / 2)
    flux = np.array(
        [
            robin * field[f.key.row, f.key.col]
            if f.kind == BoundaryKind.ROBIN
            else 2 * (field[f.key.row, f.key.col] - 1)
            if f.kind == BoundaryKind.SOURCE_DIRICHLET
            else 0
            for f in faces
        ]
    )
    return SimpleNamespace(
        concentration=field, metrics=SimpleNamespace(**metrics), faces=faces, outward_face_flux=flux
    )


def diagnose(reference: Path, output: Path, progress=None):
    output.mkdir(parents=True, exist_ok=False)
    cfg = load_config(reference / "config.yaml", PioneerRLConfig)
    policies, update_rows, weight_rows = {}, [], []
    for seed in cfg.replica_seeds:
        policy, _, updates, weights = replay_prefix(reference, seed, cfg.episodes)
        policies[f"rl_{seed}"] = policy
        update_rows.extend(updates)
        weight_rows.extend(weights)
        if progress:
            progress(f"Exact replay: replica {seed}, {policy.updates} updates")
    starts, _ = load_bank(reference, cfg)
    bank = {s.name: s.geometry for s in starts}
    evaluations = json.loads((reference / "evaluation_index.json").read_text())
    decisions = []
    for job in (r for r in evaluations if r["method"] in policies):
        path = reference / "evaluation" / job["job"]
        policy = policies[job["method"]]
        if job["policy_digest"] != policy.digest():
            raise ValueError("Evaluation does not belong to this frozen policy")
        geometry = bank[job["start"]]
        field = np.load(path / "initial_concentration.npy")
        metrics = pd.read_csv(path / "metrics.csv", float_precision="round_trip")
        transitions = pd.read_csv(path / "transitions.csv", float_precision="round_trip")
        proposals = pd.read_csv(
            path / "proposals.csv", keep_default_na=False, float_precision="round_trip"
        )
        selected = np.load(path / "selected_features.npy")
        solves = 1
        for index, transition in enumerate(transitions.to_dict("records")):
            rows = proposals[proposals.decision == transition["decision"]].to_dict("records")
            prepared = prepare_boundary(geometry, cfg.protected_length)
            solution = incumbent_solution(
                geometry, field, metrics.iloc[index].to_dict(), cfg.model.Bi
            )
            flux = eligible_wall_flux(geometry, solution, cfg.protected_length)
            edits = [
                SimpleNamespace(
                    parameters={
                        "changed_cells": float(r["changed_cells"])
                        if r["changed_cells"] != ""
                        else 0
                    }
                )
                for r in rows
            ]
            screens = [
                {
                    "screen_rejection": r["screen_rejection"],
                    "perimeter": float(r["perimeter"]) if r["perimeter"] != "" else None,
                }
                for r in rows
            ]
            features = candidate_features(
                geometry,
                solution,
                cfg,
                job["initial_Q"],
                solves,
                transition["raw_draws"],
                prepared,
                edits,
                rows,
                screens,
                flux,
            )
            action = transition["action"]
            np.testing.assert_allclose(features[action], selected[index], rtol=0, atol=2e-14)
            eligible = np.array(
                [not r["screen_rejection"] and not r["full_rejection"] for r in rows]
            )
            learned = policy.scores(features)
            prior = features @ policy.prior_weights
            p = mixed_probabilities(learned, eligible, 0, cfg.epsilon_uniform)
            q = mixed_probabilities(prior, eligible, 0, cfg.epsilon_uniform)
            np.testing.assert_allclose(
                p, [r["selection_probability"] for r in rows], rtol=0, atol=2e-14
            )
            prior_best = prior[eligible].max() if eligible.any() else 0
            decisions.append(
                {
                    "job": job["job"],
                    "method": job["method"],
                    "start": job["start"],
                    "seed": job["seed"],
                    "decision": index + 1,
                    "eligible": int(eligible.sum()),
                    "ranking_changed": bool(np.max(np.abs(p - q)) > 1e-12),
                    "probability_total_variation": float(np.abs(p - q).sum() / 2),
                    "selected_valid": not pd.isna(transition["candidate_Q"]),
                    "selected_outside_prior_best": bool(
                        eligible[action] and prior[action] < prior_best - 1e-12
                    ),
                    "reward": transition["reward"],
                    "accepted": transition["accepted"],
                }
            )
            solves = transition["number_of_pde_solves"]
            if transition["accepted"]:
                with np.load(path / f"candidates/solve_{solves:03d}.npz") as saved:
                    geometry = geometry.with_mask(saved["mask"])
                    field = saved["concentration"]
    write_csv(output / "updates.csv", update_rows)
    write_csv(output / "weights.csv", weight_rows)
    write_csv(output / "ranking_decisions.csv", decisions)
    updates, ranks = pd.DataFrame(update_rows), pd.DataFrame(decisions)
    result = {
        "status": "complete",
        "exact_final_policy_replays": len(policies),
        "updates": len(updates),
        "evaluated_policy_decisions": len(ranks),
        "ranking_changed_fraction": float(ranks.ranking_changed.mean()),
        "mean_absolute_td_error": float(updates.td_error.abs().mean()),
        "new_pde_solves": 0,
        "counterfactual_limit": (
            "Only selected candidates have evaluated outcomes; ranking changes do not "
            "establish the unselected candidate's true flux."
        ),
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
