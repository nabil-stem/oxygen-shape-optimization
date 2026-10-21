"""Verify committed policies/fields and preserve crash bytes before a resume."""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from audit_feasible_pools import check_field

from oxygen_shape.experiments.config import load_config
from oxygen_shape.experiments.learning_diagnostics import replay_prefix
from oxygen_shape.experiments.learning_duration import validate_records, verify_reference
from oxygen_shape.experiments.pioneer_bank import load_bank
from oxygen_shape.experiments.pioneer_training import assert_source_unchanged
from oxygen_shape.optimization.candidate_policy import CandidatePolicy
from oxygen_shape.optimization.duration_config import DurationConfig
from oxygen_shape.optimization.guided_search import validate_candidate


def check(root):
    config = load_config(root / "config.yaml", DurationConfig)
    assert_source_unchanged(root)
    identity = json.loads((root / "run_identity.json").read_text())
    assert verify_reference(config) == identity["reference_files"]
    starts, _ = load_bank(root, config)
    training = [s for s in starts if s.split == "train"]
    bank = {s.name: s.geometry for s in starts}
    counts, bad, fields, maximum_error = [], [], 0, 0.0
    for seed in config.replica_seeds:
        latest = root / f"policies/replica_{seed}.npz"
        policy, records = CandidatePolicy.load(latest, identity["identity"])
        validate_records(policy, records, config, training, seed)
        replay, _, _, _ = replay_prefix(config.reference_run, seed, config.reused_episodes)
        new = [r for r in records if not r["is_reused"]]
        for record in new:
            episode = record["episode"]
            path = root / f"training/replica_{seed}/episode_{episode:04d}"
            try:
                summary = json.loads((path / "rollout_summary.json").read_text())
                for key in ("status", "initial_Q", "final_Q", "number_of_pde_solves", "raw_draws"):
                    assert summary[key] == record[key], key
                transitions = pd.read_csv(path / "transitions.csv", float_precision="round_trip")
                proposals = pd.read_csv(path / "proposals.csv", float_precision="round_trip")
                metrics = pd.read_csv(path / "candidate_metrics.csv", float_precision="round_trip")
                history = pd.read_csv(path / "metrics.csv", float_precision="round_trip")
                features = np.load(path / "selected_features.npy", allow_pickle=False)
                assert len(transitions) == len(features) == record["decisions"]
                assert len(history) == len(transitions) + 1
                assert len(proposals) == record["raw_draws"]
                assert np.allclose(proposals.groupby("decision").selection_probability.sum(), 1)
                for phi, row in zip(features, transitions.to_dict("records"), strict=True):
                    delta = replay.update(
                        phi,
                        row["reward"],
                        row["bootstrap"],
                        alpha=config.learning_rate,
                        gamma=config.discount,
                        terminal=row["terminal"],
                    )
                    assert delta == row["td_error"], "TD replay mismatch"
                initial = bank[record["start"]]
                np.testing.assert_array_equal(
                    np.load(path / "initial_geometry.npy"), initial.inside
                )
                Q, error = check_field(
                    initial, np.load(path / "initial_concentration.npy"), config.model.Bi
                )
                assert abs(Q - record["initial_Q"]) < 1e-12
                fields += 1
                maximum_error = max(maximum_error, error)
                candidates = list((path / "candidates").glob("*.npz"))
                assert len(candidates) + 1 == record["number_of_pde_solves"]
                for file in candidates:
                    index = int(file.stem.split("_")[1])
                    with np.load(file, allow_pickle=False) as saved:
                        geometry = initial.with_mask(saved["mask"])
                        assert not validate_candidate(geometry, initial, config)
                        Q, error = check_field(geometry, saved["concentration"], config.model.Bi)
                    expected = metrics.loc[metrics.solve == index, "physical_flux_Q"].iloc[0]
                    assert abs(Q - expected) < 1e-12
                    fields += 1
                    maximum_error = max(maximum_error, error)
                final = initial.with_mask(np.load(path / "final_geometry.npy"))
                assert not validate_candidate(final, initial, config)
                Q, error = check_field(final, np.load(path / "concentration.npy"), config.model.Bi)
                assert abs(Q - record["final_Q"]) < 1e-12
                maximum_error = max(maximum_error, error)
            except Exception as exc:
                bad.append(
                    {
                        "replica": seed,
                        "episode": episode + 1,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        if not bad:
            assert replay.digest() == policy.digest(), "Final replayed policy differs"
        counts.append(
            {
                "replica": seed,
                "committed_episodes": len(records),
                "new_episodes": len(new),
                "updates": policy.updates,
                "sha256": hashlib.sha256(latest.read_bytes()).hexdigest(),
            }
        )
        print(f"Checked replica {seed}: {len(records)} committed episodes", flush=True)
    if bad:
        print(json.dumps({"damaged_committed_episodes": bad}, indent=2))
        raise ValueError("Committed episode artifacts need recovery before resume")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    journal = root / f"crash_recovery_{stamp}"
    (journal / "committed_checkpoints").mkdir(parents=True)
    shutil.copy2(root / "status.json", journal / "status.before_recovery.json")
    for row in counts:
        name = f"replica_{row['replica']}.npz"
        shutil.copy2(root / "policies" / name, journal / "committed_checkpoints" / name)
    result = {
        "passed": True,
        "preserved_checkpoints": counts,
        "committed_new_fields_checked": fields,
        "maximum_independent_conservation_error": maximum_error,
        "new_pde_solves": 0,
        "source_hashes_and_historical_inputs_match": True,
        "status_file_was_all_zero": not any((root / "status.json").read_bytes()),
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (journal / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    shutil.copy2(Path(__file__), journal / Path(__file__).name)
    shutil.copy2(
        Path(__file__).with_name("audit_feasible_pools.py"), journal / "audit_feasible_pools.py"
    )
    print(json.dumps(result, indent=2))
    print(f"Recovery journal: {journal}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    check(parser.parse_args().run.resolve())
