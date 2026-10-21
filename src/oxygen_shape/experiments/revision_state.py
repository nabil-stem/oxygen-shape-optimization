"""Frozen study identity and validation-only setting selection."""

import hashlib
import json
from pathlib import Path

import numpy as np

from oxygen_shape.experiments.pioneer_training import assert_source_unchanged, protocol_digest
from oxygen_shape.experiments.reporting import provenance


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def numerical_environment() -> dict:
    data = provenance()
    return {
        "python": data["python_version"],
        "packages": {k: data["package_versions"].get(k) for k in ("numpy", "scipy", "pydantic")},
    }


def identity_record(root: Path, config) -> dict:
    assert_source_unchanged(root)
    record = {
        "protocol_digest": protocol_digest(config),
        "bank_sha256": hashlib.sha256((root / "bank/manifest.json").read_bytes()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(
            (root / "source_manifest.json").read_bytes()
        ).hexdigest(),
        "environment": numerical_environment(),
        "algorithm": "expected_sarsa_reward_prior_screening_v2",
    }
    return {**record, "identity": digest(record)}


def validation_selection(config, rows: list[dict], models: dict) -> dict:
    validation = [r for r in rows if r["split"] == "validation"]
    expected = (
        len(config.families)
        * len(config.validation_generation_seeds)
        * len(config.validation_rollout_seeds)
        * len(config.replica_seeds)
    )
    scores = []
    for arm in config.arms:
        records = [r for r in validation if r.get("arm") == arm.name]
        complete = len(records) == expected and all(r["status"] == "complete" for r in records)
        score = (
            float(
                np.mean(
                    [r["fine_gain_Q"] / (r["fine_final_Q"] - r["fine_gain_Q"]) for r in records]
                )
            )
            if records
            else None
        )
        scores.append(
            {
                "arm": arm.name,
                "runs": len(records),
                "expected_runs": expected,
                "eligible": complete,
                "mean_normalized_fine_gain": score,
            }
        )
    eligible = [r for r in scores if r["eligible"]]
    winner = None
    if eligible:
        best = max(r["mean_normalized_fine_gain"] for r in eligible)
        winner = min(
            r["arm"]
            for r in eligible
            if r["mean_normalized_fine_gain"] >= best - config.min_improvement
        )
    test_arms = sorted(set(config.test_core_arms) | ({winner} if winner is not None else set()))
    return {
        "selected_arm": winner,
        "test_arms": test_arms,
        "validation_scores": scores,
        "rule": "Highest complete-setting mean normalized n=400 validation gain; "
        "lexical tie within min_improvement",
        "validation_records_digest": digest(validation),
        "all_final_model_digests": {name: policy.digest() for name, policy in models.items()},
        "individual_replica_selection": False,
    }
