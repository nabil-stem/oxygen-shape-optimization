"""Budgeted reward ablations, flux priors and geometry screening with fresh tests."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from oxygen_shape.experiments.candidate_storage import run_stored_rollout
from oxygen_shape.experiments.pioneer_bank import load_bank, prepare_bank
from oxygen_shape.experiments.pioneer_training import assert_source_unchanged
from oxygen_shape.experiments.reporting import create_run, log_failure, provenance, write_csv
from oxygen_shape.experiments.revision_preflight import REVISION_CONTROLS, revision_preflight
from oxygen_shape.experiments.revision_report import write_revision_report
from oxygen_shape.experiments.revision_state import identity_record, validation_selection
from oxygen_shape.experiments.study_worker import atomic_json
from oxygen_shape.optimization.candidate_policy import CandidatePolicy
from oxygen_shape.optimization.refinement import validate_refinement
from oxygen_shape.optimization.revision_config import RevisionConfig


def run_revision_study(
    config: RevisionConfig, *, resume: Path | None = None, prepare_only: bool = False, progress=None
) -> tuple[Path, dict]:
    root = create_run(config, "rl_revision") if resume is None else resume.resolve()
    started, active = perf_counter(), False

    def update(phase, job):
        nonlocal active
        assert_source_unchanged(root)
        active = True
        atomic_json(root / "status.json", {"status": "running", "phase": phase, "active": job})

    def announce(message):
        with (root / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")
        if progress:
            progress(message)

    def finish(summary):
        elapsed = perf_counter() - started
        timing_path = root / "successful_invocation_seconds.json"
        times = json.loads(timing_path.read_text()) if timing_path.exists() else []
        times.append(elapsed)
        atomic_json(timing_path, times)
        summary.update(
            wall_clock_seconds_this_invocation=elapsed,
            recorded_successful_invocation_seconds=sum(times),
        )
        atomic_json(root / "summary.json", summary)
        atomic_json(
            root / "status.json",
            {"status": summary["status"], "phase": "prepared" if prepare_only else "finished"},
        )
        return root, summary

    try:
        identity_path = root / "run_identity.json"
        if resume is None:
            update("preparation", "fresh held-out bank")
            starts, bank = prepare_bank(root, config, announce)
            record = identity_record(root, config)
            atomic_json(identity_path, record)
        else:
            record = json.loads(identity_path.read_text())
            if identity_record(root, config) != record:
                raise ValueError("Revision resume identity/configuration/environment differs")
            starts, bank = load_bank(root, config)
        fingerprint = record["identity"]
        if not (root / "readiness.json").exists():
            update("preflight", "reward identity, controls, prior and full screening")
            ready = revision_preflight(root, config, starts)
        else:
            ready = json.loads((root / "readiness.json").read_text())
            if ready.get("status") != "ready":
                raise ValueError("Revision preflight not ready")
        base = {
            **provenance(),
            "identity": fingerprint,
            "seed": config.seed,
            "grid_shape": [config.n, config.n],
            "Bi": config.model.Bi,
            "solver": "finite_volume",
            "solver_tolerance": None,
            "diagnostic_tolerances": config.solver.model_dump(),
            "preparation_pde_solves": bank["preparation_pde_solves"],
            "preflight_pde_solves": ready["number_of_pde_solves"],
            "historical_reused_pioneer_pde_solves": bank.get("historical_pioneer_pde_solves", 0),
        }
        if prepare_only:
            return finish(
                {
                    **base,
                    "status": "ready",
                    "number_of_pde_solves": bank["preparation_pde_solves"]
                    + ready["number_of_pde_solves"],
                }
            )
        cases = [s for s in starts if s.split == "train"]
        models, model_arms, training = {}, {}, []
        for arm in config.arms:
            for replica in config.replica_seeds:
                name = f"{arm.name}__{replica}"
                checkpoint = root / "policies" / f"{name}.npz"
                checkpoint.parent.mkdir(exist_ok=True)
                policy_identity = f"{fingerprint}:{name}"
                prior = CandidatePolicy.with_flux_prior(arm.flux_prior_scale)
                if checkpoint.exists():
                    policy, records = CandidatePolicy.load(checkpoint, policy_identity)
                else:
                    policy, records = prior, []
                    policy.save(checkpoint, policy_identity, records)
                if (
                    len(records) > config.episodes
                    or any(
                        r["episode"] != i or r["arm"] != arm.name or r["replica"] != replica
                        for i, r in enumerate(records)
                    )
                    or policy.updates != sum(r["decisions"] for r in records)
                    or not np.array_equal(policy.prior_weights, prior.prior_weights)
                ):
                    raise ValueError("Revision checkpoint history or fixed prior mismatch")
                for episode in range(len(records), config.episodes):
                    cycle, position = divmod(episode, len(cases))
                    case = cases[
                        int(
                            np.random.default_rng(replica + cycle).permutation(len(cases))[position]
                        )
                    ]
                    epsilon = config.epsilon_start + (
                        config.epsilon_end - config.epsilon_start
                    ) * episode / max(config.episodes - 1, 1)
                    cfg = config.for_arm(arm).model_copy(
                        update={"seed": replica * 1000 + episode, "initial_shape": case.family}
                    )
                    update("training", f"{name}: episode {episode + 1}/{config.episodes}")
                    job_started = perf_counter()
                    result, row = run_stored_rollout(
                        root / "training" / name / f"episode_{episode:04d}",
                        cfg,
                        case.geometry,
                        method="learned",
                        policy=policy,
                        training=True,
                        epsilon=epsilon,
                    )
                    if arm.discount == 1 and arm.invalid_reward == 0:
                        if (
                            abs(
                                row["reward_sum"]
                                - cfg.reward_scale * (row["final_Q"] / row["initial_Q"] - 1)
                            )
                            > 1e-9
                        ):
                            raise ArithmeticError("Training reward failed global flux identity")
                    records.append(
                        {
                            **row,
                            "arm": arm.name,
                            "replica": replica,
                            "episode": episode,
                            "seed": cfg.seed,
                            "start": case.name,
                            "epsilon": epsilon,
                            "discount": cfg.discount,
                            "invalid_reward": cfg.invalid_reward,
                            "prior_scale": arm.flux_prior_scale,
                            "updates": policy.updates,
                            "workflow_seconds": perf_counter() - job_started,
                        }
                    )
                    policy.save(checkpoint, policy_identity, records)
                    write_csv(checkpoint.with_suffix(".csv"), records)
                    announce(
                        f"{name} episode={episode + 1}/{config.episodes}: {result.status}, "
                        f"Q={row['initial_Q']:.6g}->{row['final_Q']:.6g}, raw={row['raw_draws']}"
                    )
                models[name], model_arms[name] = policy, arm.name
                training.extend(records)
        write_csv(root / "training.csv", training)
        index = root / "evaluation_index.json"
        evaluations = json.loads(index.read_text()) if index.exists() else []
        done = {r["job"] for r in evaluations}
        if len(done) != len(evaluations):
            raise ValueError("Duplicate revision evaluation jobs")
        for row in evaluations:
            model = models.get(row["method"])
            if model is not None and row["policy_digest"] != model.digest():
                raise ValueError("Stored revision evaluation has a different frozen model")
            if model is None and row["method"] not in REVISION_CONTROLS:
                raise ValueError("Unknown stored revision evaluation method")
        arm_lookup = {a.name: a for a in config.arms}

        def evaluate(split, seeds, methods):
            for case_index, case in enumerate(s for s in starts if s.split == split):
                for seed_index, seed in enumerate(seeds):
                    offset = (case_index * len(seeds) + seed_index) % len(methods)
                    for name in methods[offset:] + methods[:offset]:
                        job = f"{case.name}_{seed}_{name}"
                        if job in done:
                            continue
                        model = models.get(name)
                        if model is not None:
                            arm_name = model_arms[name]
                            cfg = config.for_arm(arm_lookup[arm_name])
                            method = "learned"
                        else:
                            arm_name = None
                            method, screen = REVISION_CONTROLS[name]
                            cfg = config.model_copy(update={"candidate_screening": screen})
                        cfg = cfg.model_copy(update={"seed": seed, "initial_shape": case.family})
                        before = None if model is None else model.digest()
                        update(split, job)
                        job_started = perf_counter()
                        path = root / "evaluation" / job
                        result, row = run_stored_rollout(
                            path, cfg, case.geometry, method=method, policy=model
                        )
                        refinements = validate_refinement(cfg, result.search, path)
                        if model is not None and model.digest() != before:
                            raise ArithmeticError("Frozen revision model mutated")
                        evaluation = {
                            **row,
                            "job": job,
                            "split": split,
                            "start": case.name,
                            "family": case.family,
                            "seed": seed,
                            "method": name,
                            "rollout_method": method,
                            "arm": arm_name,
                            "policy_digest": before,
                            "eta": result.search.final_solution.metrics.efficiency_eta,
                            "area": result.search.final_geometry.area,
                            "robin_perimeter": result.search.final_solution.metrics.robin_perimeter,
                            "fine_final_Q": refinements[-1]["final_Q"],
                            "fine_gain_Q": refinements[-1]["gain_Q"],
                            "refinement_pde_solves": 2 * len(cfg.refinement_factors),
                            "workflow_seconds": perf_counter() - job_started,
                            "max_conservation_error": max(
                                result.maximum_conservation_error,
                                max(r["conservation_error"] for r in refinements),
                            ),
                        }
                        evaluations.append(evaluation)
                        done.add(job)
                        atomic_json(index, evaluations)
                        write_csv(root / "evaluations.csv", evaluations)
                        announce(
                            f"{split} {job}: {result.status}, "
                            f"fine Q={evaluation['fine_final_Q']:.7g}"
                        )

        evaluate("validation", config.validation_rollout_seeds, [*REVISION_CONTROLS, *models])
        selection = validation_selection(config, evaluations, models)
        selection_path = root / "selection.json"
        if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
            raise ValueError("Frozen validation selection changed on resume")
        atomic_json(selection_path, selection)
        announce(
            f"Validation selected setting: {selection['selected_arm']}; "
            f"test settings={selection['test_arms']}"
        )
        test_models = [name for name in models if model_arms[name] in selection["test_arms"]]
        evaluate("test", config.test_rollout_seeds, [*REVISION_CONTROLS, *test_models])
        costs = {
            "training_pde_solves": sum(r["number_of_pde_solves"] for r in training),
            "validation_pde_solves": sum(
                r["number_of_pde_solves"] for r in evaluations if r["split"] == "validation"
            ),
            "test_pde_solves": sum(
                r["number_of_pde_solves"] for r in evaluations if r["split"] == "test"
            ),
            "refinement_pde_solves": sum(r["refinement_pde_solves"] for r in evaluations),
        }
        summary = {
            **base,
            **costs,
            "status": "complete"
            if all(r["status"] == "complete" for r in training + evaluations)
            else "incomplete",
            "training_episodes": len(training),
            "incomplete_training_episodes": sum(r["status"] != "complete" for r in training),
            "settings": len(config.arms),
            "models": len(models),
            "selected_arm": selection["selected_arm"],
            "validation_runs": sum(r["split"] == "validation" for r in evaluations),
            "test_runs": sum(r["split"] == "test" for r in evaluations),
            "incomplete_evaluations": sum(r["status"] != "complete" for r in evaluations),
            "number_of_pde_solves": base["preparation_pde_solves"]
            + base["preflight_pde_solves"]
            + sum(costs.values()),
            "max_conservation_error": max(
                bank["max_conservation_error"],
                ready["max_conservation_error"],
                max(r["max_conservation_error"] for r in training + evaluations),
            ),
            "training_workflow_seconds": sum(r["workflow_seconds"] for r in training),
            "evaluation_workflow_seconds": sum(r["workflow_seconds"] for r in evaluations),
            "selection": "Validation setting only; all replicas; fixed core settings; no promotion",
        }
        write_revision_report(root, config, summary, training, evaluations, selection)
        announce(
            f"Finished revision study: {summary['status']}; "
            f"new PDE solves={summary['number_of_pde_solves']}"
        )
        return finish(summary)
    except Exception as exc:
        log_failure(root, exc)
        atomic_json(
            root / ("status.json" if resume is None or active else "last_resume_rejection.json"),
            {"status": "failed", "error": str(exc)},
        )
        raise
