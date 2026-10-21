"""Continue exact historical prefixes and evaluate frozen development checkpoints."""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from oxygen_shape.experiments.candidate_storage import run_stored_rollout
from oxygen_shape.experiments.config import load_config
from oxygen_shape.experiments.duration_report import write_duration_report
from oxygen_shape.experiments.learning_diagnostics import replay_prefix
from oxygen_shape.experiments.pioneer_bank import load_bank
from oxygen_shape.experiments.pioneer_training import (
    assert_source_unchanged,
    protocol_digest,
    readiness,
)
from oxygen_shape.experiments.reporting import create_run, log_failure, provenance, write_csv
from oxygen_shape.experiments.study_worker import atomic_json
from oxygen_shape.optimization.candidate_policy import CandidatePolicy
from oxygen_shape.optimization.duration_config import DurationConfig
from oxygen_shape.optimization.pioneer_config import PioneerRLConfig
from oxygen_shape.optimization.refinement import validate_refinement


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_files(reference):
    files = [
        reference / n
        for n in (
            "config.yaml",
            "summary.json",
            "run_identity.json",
            "bank/manifest.json",
            "artifact_audit.json",
        )
    ]
    files += list((reference / "bank").glob("*.npy"))
    files += list((reference / "policies").glob("*.npz"))
    files += list(reference.glob("training/replica_*/episode_*/transitions.csv"))
    files += list(reference.glob("training/replica_*/episode_*/selected_features.npy"))
    return {p.relative_to(reference).as_posix(): sha(p) for p in sorted(files)}


def verify_reference(config):
    reference = config.reference_run
    assert_source_unchanged(reference)
    previous = load_config(reference / "config.yaml", PioneerRLConfig)
    identity = json.loads((reference / "run_identity.json").read_text())["identity"]
    summary = json.loads((reference / "summary.json").read_text())
    if identity != config.reference_identity or summary["status"] != "complete":
        raise ValueError("Reference identity or completion does not match")
    if not json.loads((reference / "artifact_audit.json").read_text())["passed"]:
        raise ValueError("Only an audited prefix may be reused")
    if previous.episodes != config.reused_episodes:
        raise ValueError("Reference episode count differs from the reused prefix")
    for key in PioneerRLConfig.model_fields.keys() - {"episodes", "seed", "output_root"}:
        if getattr(previous, key) != getattr(config, key):
            raise ValueError(f"Changing {key} would confound training duration")
    for key in ("numpy", "scipy", "pydantic"):
        if summary["package_versions"][key] != provenance()["package_versions"][key]:
            raise ValueError("Numerical environment differs from the reference prefix")
    return reference_files(reference)


def validate_records(policy, records, config, starts, replica):
    if len(records) > config.episodes or policy.updates != sum(r["decisions"] for r in records):
        raise ValueError("Invalid episode/update counts")
    expected = CandidatePolicy.with_flux_prior(config.flux_prior_scale).prior_weights
    if not np.array_equal(policy.prior_weights, expected):
        raise ValueError("Checkpoint prior changed")
    for episode, record in enumerate(records):
        cycle, position = divmod(episode, len(starts))
        order = np.random.default_rng(replica + cycle).permutation(len(starts))
        if (
            record["episode"] != episode
            or record["seed"] != replica * 1000 + episode
            or record["start"] != starts[int(order[position])].name
            or record["epsilon"] != config.exploration(episode)
        ):
            raise ValueError("Checkpoint curriculum or exploration differs from frozen protocol")


def run_duration(config: DurationConfig, *, resume: Path | None = None, progress=print):
    started = perf_counter()
    root = create_run(config, "learning_duration") if resume is None else resume.resolve()
    metadata = provenance()
    changed = False
    if progress:
        progress(f"Output: {root}")

    def update(phase, active):
        nonlocal changed
        assert_source_unchanged(root)
        changed = True
        atomic_json(root / "status.json", {"status": "running", "phase": phase, "active": active})
        with (root / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(f"{datetime.now(timezone.utc).isoformat()} {phase}: {active}\n")

    try:
        if resume is None:
            update("initialization", "verify audited reference")
            pinned = verify_reference(config)
            (root / "bank").mkdir()
            for name in [
                "manifest.json",
                *[p.name for p in (config.reference_run / "bank").glob("*.npy")],
            ]:
                shutil.copy2(config.reference_run / "bank" / name, root / "bank" / name)
            identity_record = {
                "protocol_digest": protocol_digest(config),
                "reference_files": pinned,
                "source_manifest_sha256": sha(root / "source_manifest.json"),
                "bank_sha256": sha(root / "bank/manifest.json"),
                "environment": {k: metadata[k] for k in ("python_version", "package_versions")},
            }
            identity_record["identity"] = hashlib.sha256(
                json.dumps(identity_record, sort_keys=True).encode()
            ).hexdigest()
            atomic_json(root / "run_identity.json", identity_record)
        else:
            identity_record = json.loads((root / "run_identity.json").read_text())
            if identity_record["protocol_digest"] != protocol_digest(config):
                raise ValueError("Resume configuration differs from the frozen duration protocol")
            assert_source_unchanged(root)
            if verify_reference(config) != identity_record["reference_files"]:
                raise ValueError("Pinned historical input files changed")
        fingerprint = {k: v for k, v in identity_record.items() if k != "identity"}
        if (
            hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
            != identity_record["identity"]
        ):
            raise ValueError("Duration identity is corrupted")
        if (
            sha(root / "source_manifest.json") != identity_record["source_manifest_sha256"]
            or sha(root / "bank/manifest.json") != identity_record["bank_sha256"]
        ):
            raise ValueError("Frozen source or bank identity changed")
        if identity_record["environment"] != {
            k: metadata[k] for k in ("python_version", "package_versions")
        }:
            raise ValueError("Numerical environment changed")
        identity = identity_record["identity"]
        starts, _ = load_bank(root, config)
        training_cases = [s for s in starts if s.split == "train"]
        development_cases = [s for s in starts if s.split != "train"]
        (root / "policies").mkdir(exist_ok=True)
        reference_identity = config.reference_identity
        for replica in config.replica_seeds:
            latest = root / f"policies/replica_{replica}.npz"
            if not latest.exists():
                policy, records = CandidatePolicy.load(
                    config.reference_run / f"policies/replica_{replica}.npz", reference_identity
                )
                records = [{**r, "is_reused": True} for r in records]
                validate_records(policy, records, config, training_cases, replica)
                policy.save(latest, identity, records)
            for target in (e for e in config.checkpoint_episodes if e <= config.reused_episodes):
                snapshot = root / f"checkpoints/episode_{target:04d}/replica_{replica}.npz"
                if not snapshot.exists():
                    policy, records, _, _ = replay_prefix(config.reference_run, replica, target)
                    snapshot.parent.mkdir(parents=True, exist_ok=True)
                    records = [{**r, "is_reused": True} for r in records]
                    policy.save(snapshot, identity, records)
        if not (root / "readiness.json").exists():
            update("preflight", "solver and frozen/reloaded environments")
            readiness(root, config, starts)
        ready = json.loads((root / "readiness.json").read_text())
        if ready["status"] != "ready":
            raise ValueError("Preflight did not pass")
        evaluations = (
            json.loads((root / "evaluation_index.json").read_text())
            if (root / "evaluation_index.json").exists()
            else []
        )
        done = {r["job"]: r for r in evaluations}
        if len(done) != len(evaluations):
            raise ValueError("Duplicate evaluation jobs")
        for target in config.checkpoint_episodes:
            for replica in config.replica_seeds:
                latest = root / f"policies/replica_{replica}.npz"
                policy, records = CandidatePolicy.load(latest, identity)
                validate_records(policy, records, config, training_cases, replica)
                for episode in range(len(records), target):
                    cycle, position = divmod(episode, len(training_cases))
                    order = np.random.default_rng(replica + cycle).permutation(len(training_cases))
                    case = training_cases[int(order[position])]
                    cfg = config.model_copy(
                        update={"seed": replica * 1000 + episode, "initial_shape": case.family}
                    )
                    update(
                        "training", f"replica={replica}, episode={episode + 1}/{config.episodes}"
                    )
                    _, row = run_stored_rollout(
                        root / f"training/replica_{replica}/episode_{episode:04d}",
                        cfg,
                        case.geometry,
                        method="learned",
                        policy=policy,
                        training=True,
                        epsilon=config.exploration(episode),
                    )
                    records.append(
                        {
                            "replica": replica,
                            "episode": episode,
                            "seed": cfg.seed,
                            "start": case.name,
                            "epsilon": config.exploration(episode),
                            "updates": policy.updates,
                            "is_reused": False,
                            **row,
                        }
                    )
                    policy.save(latest, identity, records)
                    write_csv(root / f"policies/replica_{replica}_episodes.csv", records)
                    if progress:
                        progress(
                            f"replica={replica}, episode={episode + 1}: {row['status']}, "
                            f"Q={row['final_Q']:.7g}"
                        )
                snapshot = root / f"checkpoints/episode_{target:04d}/replica_{replica}.npz"
                if not snapshot.exists():
                    if len(records) != target:
                        raise ValueError(
                            "Missing historical checkpoint cannot be recovered from later weights"
                        )
                    snapshot.parent.mkdir(parents=True, exist_ok=True)
                    policy.save(snapshot, identity, records)
                frozen, history = CandidatePolicy.load(snapshot, identity)
                validate_records(frozen, history, config, training_cases, replica)
                if len(history) != target:
                    raise ValueError("Snapshot has the wrong episode count")
        # Evaluation cannot steer this fixed training protocol. Persist every
        # requested snapshot first, then evaluate them all without mutation.
        for target in config.checkpoint_episodes:
            models = {
                replica: CandidatePolicy.load(
                    root / f"checkpoints/episode_{target:04d}/replica_{replica}.npz", identity
                )[0]
                for replica in config.replica_seeds
            }
            for case in development_cases:
                for seed in config.development_rollout_seeds:
                    for replica, model in [(None, None), *models.items()]:
                        label = "flux_pool" if model is None else f"rl_{replica}_e{target}"
                        job = f"{case.name}_{seed}_{label}"
                        before = None if model is None else model.digest()
                        if job in done:
                            if done[job]["policy_digest"] != before:
                                raise ValueError(
                                    "Evaluation belongs to different checkpoint weights"
                                )
                            continue
                        cfg = config.model_copy(update={"seed": seed, "initial_shape": case.family})
                        update("development_evaluation", job)
                        path = root / "evaluation" / job
                        evaluation_started = perf_counter()
                        result, row = run_stored_rollout(
                            path,
                            cfg,
                            case.geometry,
                            method="flux_pool" if model is None else "learned",
                            policy=model,
                        )
                        refined = validate_refinement(cfg, result.search, path)
                        if model is not None and model.digest() != before:
                            raise ArithmeticError("Evaluation changed frozen policy weights")
                        evaluation = {
                            **row,
                            "job": job,
                            "split": "development",
                            "original_split": case.split,
                            "start": case.name,
                            "seed": seed,
                            "replica": replica,
                            "checkpoint_episode": 0 if model is None else target,
                            "policy_digest": before,
                            "eta": result.search.final_solution.metrics.efficiency_eta,
                            "area": result.search.final_geometry.area,
                            "robin_perimeter": result.search.final_solution.metrics.robin_perimeter,
                            "fine_final_Q": refined[-1]["final_Q"],
                            "fine_gain_Q": refined[-1]["gain_Q"],
                            "refinement_pde_solves": 2 * len(config.refinement_factors),
                            "max_conservation_error": max(
                                row["max_conservation_error"],
                                max(r["conservation_error"] for r in refined),
                            ),
                            "workflow_seconds_total": perf_counter() - evaluation_started,
                        }
                        evaluations.append(evaluation)
                        done[job] = evaluation
                        atomic_json(root / "evaluation_index.json", evaluations)
                        write_csv(root / "evaluations.csv", evaluations)
                        if progress:
                            progress(
                                f"development {job}: {row['status']}, "
                                f"fine Q={evaluation['fine_final_Q']:.7g}"
                            )
        training = []
        for replica in config.replica_seeds:
            _, records = CandidatePolicy.load(root / f"policies/replica_{replica}.npz", identity)
            training.extend(records)
        write_csv(root / "training.csv", training)
        expected = (
            len(development_cases)
            * len(config.development_rollout_seeds)
            * (1 + len(config.replica_seeds) * len(config.checkpoint_episodes))
        )
        if len(evaluations) != expected:
            raise ValueError("Evaluation matrix is incomplete")
        if reference_files(config.reference_run) != identity_record["reference_files"]:
            raise ValueError("Historical reference files changed during training")
        new_cost = sum(r["number_of_pde_solves"] for r in training if not r["is_reused"])
        eval_cost = sum(r["number_of_pde_solves"] + r["refinement_pde_solves"] for r in evaluations)
        interrupted = [json.loads(p.read_text()) for p in root.rglob("interruption_cost.json")]
        summary = {
            **metadata,
            "status": "complete"
            if all(r["status"] == "complete" for r in [*training, *evaluations])
            else "incomplete",
            "identity": identity,
            "training_episodes": len(training),
            "new_training_episodes": sum(not r["is_reused"] for r in training),
            "training_new_pde_solves": new_cost,
            "evaluation_new_pde_solves": eval_cost,
            "preflight_new_pde_solves": ready["number_of_pde_solves"],
            "number_of_new_pde_solves": new_cost + eval_cost + ready["number_of_pde_solves"],
            "reused_training_pde_solves": sum(
                r["number_of_pde_solves"] for r in training if r["is_reused"]
            ),
            "evaluation_runs": len(evaluations),
            "min_improvement": config.min_improvement,
            "incomplete_training_episodes": sum(r["status"] != "complete" for r in training),
            "incomplete_evaluation_runs": sum(r["status"] != "complete" for r in evaluations),
            "fresh_final_test_runs": 0,
            "interrupted_attempts": len(interrupted),
            "additional_interrupted_pde_lower_bound": sum(
                r["pde_solves_lower_bound"] for r in interrupted
            ),
            "additional_interrupted_pde_upper_bound": sum(
                r["pde_solves_upper_bound"] + r["endpoint_refinement_additional_upper_bound"]
                for r in interrupted
            ),
            "maximum_conservation_error": max(
                ready["max_conservation_error"],
                *(r["max_conservation_error"] for r in [*training, *evaluations]),
            ),
            "wall_seconds_this_invocation": perf_counter() - started,
            "selection": "Development duration comparison; no model promotion or final-test claim",
        }
        atomic_json(root / "summary.json", summary)
        write_duration_report(root, summary, evaluations)
        update("finished", summary["status"])
        atomic_json(root / "status.json", {"status": summary["status"], "phase": "finished"})
        return root, summary
    except Exception as exc:
        log_failure(root, exc)
        atomic_json(
            root / ("status.json" if changed or resume is None else "last_resume_rejection.json"),
            {"status": "failed", "error": str(exc)},
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    root, summary = run_duration(load_config(args.config, DurationConfig), resume=args.resume)
    print(f"Status={summary['status']}; output={root}", flush=True)


if __name__ == "__main__":
    main()
