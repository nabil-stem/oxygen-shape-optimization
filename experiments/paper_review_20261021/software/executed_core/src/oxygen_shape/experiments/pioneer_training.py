"""Resumable multi-start Expected Sarsa with frozen final-model evaluation."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from oxygen_shape.experiments.candidate_storage import run_stored_rollout
from oxygen_shape.experiments.pioneer_bank import load_bank, prepare_bank
from oxygen_shape.experiments.pioneer_report import write_pioneer_report
from oxygen_shape.experiments.reporting import create_run, log_failure, provenance, write_csv
from oxygen_shape.experiments.study_worker import atomic_json
from oxygen_shape.geometry.generators import rectangle
from oxygen_shape.optimization.candidate_policy import FEATURE_NAMES, CandidatePolicy
from oxygen_shape.optimization.candidate_rollout import CONTROLS
from oxygen_shape.optimization.pioneer_config import PioneerRLConfig
from oxygen_shape.optimization.refinement import validate_refinement
from oxygen_shape.solvers.analytical_1d import slab_flux
from oxygen_shape.solvers.finite_volume import solve


def protocol_digest(config: PioneerRLConfig) -> str:
    value = config.model_dump(mode="json")
    value.pop("output_root")
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def assert_source_unchanged(root: Path) -> None:
    repository = Path(__file__).resolve().parents[3]
    for name, expected in json.loads((root / "source_manifest.json").read_text()).items():
        if hashlib.sha256((repository / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Executed source changed; cannot resume: {name}")


def readiness(root, config, starts) -> dict:
    geometry = rectangle(16, 3, 1 / 16, slab=True)
    solution = solve(geometry, config.model, config.solver)
    if abs(solution.metrics.physical_flux_Q / (3 / 16) - slab_flux(config.model.Bi)) > 1e-9:
        raise ArithmeticError("RL preflight slab check failed")
    rows, solves, error = [], 1, solution.metrics.conservation_error
    for family in config.families:
        case = next(c for c in starts if c.family == family and c.name.endswith("canonical"))
        for method in (*CONTROLS, "learned"):
            cfg = config.model_copy(
                update={"seed": 990000 + len(rows), "solve_budget": 3, "initial_shape": family}
            )
            policy = (
                CandidatePolicy.with_flux_prior(config.flux_prior_scale)
                if method == "learned"
                else None
            )
            result, row = run_stored_rollout(
                root / "preflight" / f"{family}_{method}",
                cfg,
                case.geometry,
                method=method,
                policy=policy,
                training=method == "learned",
                epsilon=1,
            )
            if result.status != "complete" or (
                policy is not None and policy.updates != len(result.transitions)
            ):
                raise ArithmeticError("RL environment preflight did not complete")
            solves += result.search.number_of_pde_solves
            error = max(error, result.maximum_conservation_error)
            rows.append({"family": family, **row})
    checkpoint = root / "preflight/smoke_policy.npz"
    policy.save(checkpoint, "preflight", [])
    restored, _ = CandidatePolicy.load(checkpoint, "preflight")
    if restored.digest() != policy.digest():
        raise ArithmeticError("Preflight checkpoint round-trip failed")
    frozen, row = run_stored_rollout(
        root / "preflight/frozen_reload",
        cfg.model_copy(update={"seed": 990100}),
        case.geometry,
        method="learned",
        policy=restored,
    )
    if frozen.status != "complete":
        raise ArithmeticError("Frozen policy preflight did not complete")
    solves += frozen.search.number_of_pde_solves
    error = max(error, frozen.maximum_conservation_error)
    rows.append({"family": case.family, **row})
    # Discard the smoke learner; each replica gets the configured fixed prior.
    row = {
        "status": "ready",
        "number_of_pde_solves": solves,
        "max_conservation_error": error,
        "controls": list(CONTROLS),
        "all_environment_checks_passed": True,
    }
    write_csv(root / "preflight.csv", rows)
    atomic_json(root / "readiness.json", row)
    return row


def run_pioneer_training(
    config: PioneerRLConfig,
    *,
    resume: Path | None = None,
    prepare_only: bool = False,
    progress=None,
) -> tuple[Path, dict]:
    root = create_run(config, "pioneer_rl") if resume is None else resume.resolve()
    invocation_started = perf_counter()
    run_active = False
    identity_file = root / "run_identity.json"
    metadata = provenance()
    environment = {
        "python": metadata["python_version"],
        "packages": {
            key: metadata["package_versions"].get(key) for key in ("numpy", "scipy", "pydantic")
        },
    }

    def update(phase, active):
        nonlocal run_active
        assert_source_unchanged(root)
        run_active = True
        atomic_json(root / "status.json", {"status": "running", "phase": phase, "active": active})
        with (root / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(f"{datetime.now(timezone.utc).isoformat()} {phase}: {active}\n")

    try:
        if resume is not None:
            identity_record = json.loads(identity_file.read_text())
            if identity_record["protocol_digest"] != protocol_digest(config):
                raise ValueError("Resume configuration differs from frozen protocol")
            assert_source_unchanged(root)
            starts, bank = load_bank(root, config)
        else:
            update("preparation", "pioneer bank")
            starts, bank = prepare_bank(root, config, progress)
            identity_record = {
                "protocol_digest": protocol_digest(config),
                "bank_sha256": hashlib.sha256(
                    (root / "bank/manifest.json").read_bytes()
                ).hexdigest(),
                "source_manifest_sha256": hashlib.sha256(
                    (root / "source_manifest.json").read_bytes()
                ).hexdigest(),
                "feature_names": FEATURE_NAMES,
                "algorithm": "linear_expected_sarsa_v1",
                "environment": environment,
            }
            identity_record["identity"] = hashlib.sha256(
                json.dumps(identity_record, sort_keys=True).encode()
            ).hexdigest()
            atomic_json(identity_file, identity_record)
        if (
            hashlib.sha256((root / "bank/manifest.json").read_bytes()).hexdigest()
            != identity_record["bank_sha256"]
        ):
            raise ValueError("Frozen bank manifest changed")
        if identity_record["environment"] != environment:
            raise ValueError("Numerical environment changed; cannot resume this checkpoint")
        if (
            hashlib.sha256((root / "source_manifest.json").read_bytes()).hexdigest()
            != identity_record["source_manifest_sha256"]
        ):
            raise ValueError("Source manifest identity changed")
        fingerprint = {key: value for key, value in identity_record.items() if key != "identity"}
        if (
            hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
            != identity_record["identity"]
        ):
            raise ValueError("Run identity is corrupted")
        if not (root / "readiness.json").exists():
            update("preflight", "solver and all control environments")
            ready = readiness(root, config, starts)
        else:
            ready = json.loads((root / "readiness.json").read_text())
            if ready.get("status") != "ready":
                raise ValueError("Preflight is not ready")
        if prepare_only:
            summary = {
                **metadata,
                "status": "ready",
                "preparation_pde_solves": bank["preparation_pde_solves"],
                "preflight_pde_solves": ready["number_of_pde_solves"],
                "identity": identity_record["identity"],
                "number_of_pde_solves": bank["preparation_pde_solves"]
                + ready["number_of_pde_solves"],
                "historical_reused_pioneer_pde_solves": bank.get(
                    "historical_pioneer_pde_solves", 0
                ),
                "seed": config.seed,
                "grid_shape": [config.n, config.n],
                "Bi": config.model.Bi,
                "solver": "finite_volume",
                "solver_tolerance": None,
                "diagnostic_tolerances": config.solver.model_dump(),
                "wall_clock_seconds_this_invocation": perf_counter() - invocation_started,
            }
            atomic_json(root / "summary.json", summary)
            atomic_json(root / "status.json", {"status": "ready", "phase": "prepared"})
            return root, summary
        identity = identity_record["identity"]
        (root / "policies").mkdir(exist_ok=True)
        training_cases = [c for c in starts if c.split == "train"]
        models, training = {}, []
        for replica in config.replica_seeds:
            checkpoint = root / "policies" / f"replica_{replica}.npz"
            if checkpoint.exists():
                policy, records = CandidatePolicy.load(checkpoint, identity)
            else:
                policy, records = CandidatePolicy.with_flux_prior(config.flux_prior_scale), []
                policy.save(checkpoint, identity, records)
            if len(records) > config.episodes or any(
                r["episode"] != i for i, r in enumerate(records)
            ):
                raise ValueError("Checkpoint episode sequence is invalid")
            expected_prior = CandidatePolicy.with_flux_prior(config.flux_prior_scale).prior_weights
            if not np.array_equal(policy.prior_weights, expected_prior) or policy.updates != sum(
                r["decisions"] for r in records
            ):
                raise ValueError("Checkpoint prior or update count differs from episode records")
            for episode in range(len(records), config.episodes):
                cycle, position = divmod(episode, len(training_cases))
                order = np.random.default_rng(replica + cycle).permutation(len(training_cases))
                case = training_cases[int(order[position])]
                epsilon = config.epsilon_start + (
                    config.epsilon_end - config.epsilon_start
                ) * episode / max(config.episodes - 1, 1)
                cfg = config.model_copy(
                    update={"seed": replica * 1000 + episode, "initial_shape": case.family}
                )
                update("training", f"replica={replica}, episode={episode + 1}/{config.episodes}")
                result, row = run_stored_rollout(
                    root / "training" / f"replica_{replica}" / f"episode_{episode:04d}",
                    cfg,
                    case.geometry,
                    method="learned",
                    policy=policy,
                    training=True,
                    epsilon=epsilon,
                )
                records.append(
                    {
                        "replica": replica,
                        "episode": episode,
                        "start": case.name,
                        "seed": cfg.seed,
                        "epsilon": epsilon,
                        "updates": policy.updates,
                        **row,
                    }
                )
                policy.save(checkpoint, identity, records)
                write_csv(root / "policies" / f"replica_{replica}_episodes.csv", records)
                if progress:
                    progress(
                        f"RL replica={replica} episode={episode + 1}/{config.episodes}: "
                        f"{result.status}, Q={row['initial_Q']:.6g}->{row['final_Q']:.6g}, "
                        f"raw={row['raw_draws']}"
                    )
            models[f"rl_{replica}"] = policy
            training.extend(records)
        write_csv(root / "training.csv", training)
        evaluations = (
            json.loads((root / "evaluation_index.json").read_text())
            if (root / "evaluation_index.json").exists()
            else []
        )
        done = {r["job"] for r in evaluations}
        if len(done) != len(evaluations):
            raise ValueError("Duplicate completed evaluation records")
        for row in evaluations:
            model = models.get(row["method"])
            if model is not None and row["policy_digest"] != model.digest():
                raise ValueError("Completed evaluation belongs to different model weights")
        methods = [*CONTROLS, *models]
        job_index = 0
        for split, seeds in (
            ("validation", config.validation_rollout_seeds),
            ("test", config.test_rollout_seeds),
        ):
            for case in [c for c in starts if c.split == split]:
                for seed in seeds:
                    offset = job_index % len(methods)
                    job_index += 1
                    for name in methods[offset:] + methods[:offset]:
                        job = f"{case.name}_{seed}_{name}"
                        if job in done:
                            continue
                        path = root / "evaluation" / job
                        cfg = config.model_copy(update={"seed": seed, "initial_shape": case.family})
                        model = models.get(name)
                        before = None if model is None else model.digest()
                        update("evaluation", job)
                        result, row = run_stored_rollout(
                            path,
                            cfg,
                            case.geometry,
                            method="learned" if model else name,
                            policy=model,
                        )
                        refinements = validate_refinement(cfg, result.search, path)
                        if model is not None and model.digest() != before:
                            raise ArithmeticError("Evaluation mutated a frozen model")
                        evaluation = {
                            **row,
                            "job": job,
                            "split": split,
                            "start": case.name,
                            "family": case.family,
                            "seed": seed,
                            "method": name,
                            "rollout_method": row["method"],
                            "policy_digest": before,
                            "eta": result.search.final_solution.metrics.efficiency_eta,
                            "area": result.search.final_geometry.area,
                            "robin_perimeter": result.search.final_solution.metrics.robin_perimeter,
                            "fine_final_Q": refinements[-1]["final_Q"],
                            "fine_gain_Q": refinements[-1]["gain_Q"],
                            "refinement_pde_solves": 2 * len(config.refinement_factors),
                            "max_conservation_error": max(
                                result.maximum_conservation_error,
                                max(r["conservation_error"] for r in refinements),
                            ),
                        }
                        evaluations.append(evaluation)
                        atomic_json(root / "evaluation_index.json", evaluations)
                        write_csv(root / "evaluations.csv", evaluations)
                        if progress:
                            progress(f"held-out {job}: {result.status}, Q={row['final_Q']:.7g}")
        preparation = bank["preparation_pde_solves"]
        preflight = ready["number_of_pde_solves"]
        train_cost = sum(r["number_of_pde_solves"] for r in training)
        eval_cost = sum(r["number_of_pde_solves"] for r in evaluations)
        refinement_cost = sum(r["refinement_pde_solves"] for r in evaluations)
        interruptions = [json.loads(p.read_text()) for p in root.rglob("interruption_cost.json")]
        summary = {
            **metadata,
            "status": "complete"
            if all(r["status"] == "complete" for r in [*training, *evaluations])
            else "incomplete",
            "identity": identity,
            "seed": config.seed,
            "grid_shape": [config.n, config.n],
            "Bi": config.model.Bi,
            "solver": "finite_volume",
            "solver_tolerance": None,
            "diagnostic_tolerances": config.solver.model_dump(),
            "preparation_pde_solves": preparation,
            "preflight_pde_solves": preflight,
            "training_pde_solves": train_cost,
            "evaluation_pde_solves": eval_cost,
            "refinement_pde_solves": refinement_cost,
            "number_of_pde_solves": preparation
            + preflight
            + train_cost
            + eval_cost
            + refinement_cost,
            "interrupted_attempts": len(interruptions),
            "additional_interrupted_pde_solves_lower_bound": sum(
                r["pde_solves_lower_bound"] for r in interruptions
            ),
            "additional_interrupted_pde_solves_upper_bound": sum(
                r["pde_solves_upper_bound"] + r["endpoint_refinement_additional_upper_bound"]
                for r in interruptions
            ),
            "training_episodes": len(training),
            "incomplete_training_episodes": sum(r["status"] != "complete" for r in training),
            "evaluation_runs": len(evaluations),
            "replicas": len(models),
            "refill_quota": config.refill_quota,
            "flux_prior_scale": config.flux_prior_scale,
            "training_workflow_seconds": sum(
                r.get("workflow_seconds", r["wall_seconds"]) for r in training
            ),
            "evaluation_workflow_seconds_excluding_refinement": sum(
                r.get("workflow_seconds", r["wall_seconds"]) for r in evaluations
            ),
            "max_conservation_error": max(
                bank["max_conservation_error"],
                ready["max_conservation_error"],
                max(r["max_conservation_error"] for r in training),
                max(r["max_conservation_error"] for r in evaluations),
            ),
            "model_selection": "All final replicas; no validation/test selection or promotion",
            "historical_reused_pioneer_pde_solves": bank.get("historical_pioneer_pde_solves", 0),
            "wall_clock_seconds_this_invocation": perf_counter() - invocation_started,
        }
        atomic_json(root / "summary.json", summary)
        write_pioneer_report(root, config, summary, bank, training, evaluations)
        atomic_json(
            root / "status.json",
            {
                "status": summary["status"],
                "phase": "finished",
                "number_of_pde_solves": summary["number_of_pde_solves"],
            },
        )
        return root, summary
    except Exception as exc:
        log_failure(root, exc)
        atomic_json(
            root
            / ("status.json" if resume is None or run_active else "last_resume_rejection.json"),
            {"status": "failed", "error": str(exc), "resume_directory": str(root)},
        )
        raise
