"""Resume a frozen checkpoint benchmark without repeating training."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from oxygen_shape.experiments.optimization import save_search, search_summary
from oxygen_shape.experiments.reporting import (
    create_run,
    log_failure,
    provenance,
    save_summary,
    write_csv,
)
from oxygen_shape.experiments.rl_report import benchmark_report
from oxygen_shape.optimization.config import RLConfig
from oxygen_shape.optimization.q_learning import QPolicy
from oxygen_shape.optimization.random_search import run_search
from oxygen_shape.optimization.refinement import validate_refinement
from oxygen_shape.viz.fields import plot_fields
from oxygen_shape.viz.optimization import plot_benchmark, plot_learning, plot_search


def benchmark_checkpoint(
    config: RLConfig, training_directory: Path, progress=None
) -> tuple[Path, dict]:
    import csv

    checkpoint = training_directory / "policy.npz"
    metadata = json.loads((training_directory / "policy_metadata.json").read_text(encoding="utf-8"))
    original = metadata["training_configuration"]
    current = config.model_dump(mode="json")
    environment_keys = (
        "n",
        "model",
        "initial_shape",
        "target_area",
        "perimeter_limit",
        "minimum_feature_width",
        "protected_length",
        "radii",
        "min_improvement",
        "refinement_factors",
        "training_episodes",
        "training_solve_budget",
    )
    if any(original[key] != current[key] for key in environment_keys):
        raise ValueError("Benchmark configuration differs from the checkpoint's training problem")
    with (training_directory / "training.csv").open(encoding="utf-8", newline="") as stream:
        training = [
            {
                key: (
                    value == "True"
                    if key == "solve_budget_completed"
                    else int(value)
                    if key in ("episode", "seed", "number_of_pde_solves", "attempts")
                    else float(value)
                )
                for key, value in row.items()
            }
            for row in csv.DictReader(stream)
        ]
    policy = QPolicy.load(checkpoint)
    if {row["seed"] for row in training} & set(config.evaluation_seeds):
        raise ValueError("Benchmark seeds overlap checkpoint training seeds")
    values, visits = policy.values.copy(), policy.visits.copy()
    directory = create_run(config, "rl_frozen_benchmark")
    start = perf_counter()
    evaluations, runs = [], []
    try:
        policy.save(directory / "policy.npz")
        metadata["evaluation_stall_limit"] = config.policy_stall_limit
        (directory / "policy_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        write_csv(directory / "training.csv", training)
        for seed in config.evaluation_seeds:
            for name in ("uniform", "untrained", "trained"):
                eval_config = config.model_copy(update={"seed": seed})
                selected = (
                    None if name == "uniform" else QPolicy() if name == "untrained" else policy
                )
                run_start = perf_counter()
                result = run_search(eval_config, policy=selected, epsilon=config.evaluation_epsilon)
                path = directory / "evaluation" / name / f"seed_{seed}"
                save_search(path, result)
                refinements = validate_refinement(eval_config, result, path)
                row = {
                    "policy": name,
                    **search_summary(eval_config, result, refinements),
                    "wall_clock_seconds": perf_counter() - run_start,
                }
                evaluations.append(row)
                runs.append((name, seed, result))
                write_csv(
                    directory / "evaluations.csv",
                    [{k: v for k, v in r.items() if k != "constraints"} for r in evaluations],
                )
                if progress:
                    progress(
                        f"evaluation {name}, seed={seed}: Q={row['final_Q']:.6g}; "
                        f"fallback attempts={row['uniform_fallback_attempts']}"
                    )
        if not (np.array_equal(values, policy.values) and np.array_equal(visits, policy.visits)):
            raise ArithmeticError("Evaluation mutated the checkpoint")
        representative = next(r for name, _, r in runs if name == "trained")
        save_search(directory, representative)
        write_csv(
            directory / "metrics.csv",
            [{"policy": name, "seed": seed, **row} for name, seed, r in runs for row in r.history],
        )
        plot_benchmark(runs, config.solve_budget, directory / "figures/benchmark.png")
        plot_learning(training, policy, directory / "figures/learning.png")
        plot_search(representative, directory / "figures/representative_history.png")
        plot_fields(
            representative.final_geometry,
            representative.final_solution,
            directory / "figures/representative_fields.png",
        )
        cost = int(sum(r["number_of_pde_solves"] for r in training))
        report = benchmark_report(config, evaluations, cost)
        pilot = (
            "## Feasibility pilot and frozen-checkpoint recovery\n\n"
            "The original pure-policy pilot (seed 0) reached only 24/41 solves after 4100 "
            "attempts and failed its budget check. Training weights were preserved. "
            f"The evaluation protocol now switches to uniform boundary sampling after "
            f"{config.policy_stall_limit} consecutive invalid proposals, until one valid solve "
            "resets the counter. The untrained control uses the same fallback. "
            "Training used no fallback. This is a hybrid learned proposal policy with a "
            "uniform feasibility safeguard. Fresh seeds 100–104 assess that revised protocol; "
            "the pilot is not pooled into the benchmark. No reward-based hyperparameter "
            "tuning or checkpoint selection was performed.\n\n"
        )
        log = (training_directory / "run.log").read_text(encoding="utf-8")
        if "Only 24/41 solves after 4100 attempts" in log:
            report = report.replace("## Held-out evaluation", pilot + "## Held-out evaluation", 1)
        (directory / "report.md").write_text(report, encoding="utf-8")
        summary = {
            **provenance(),
            "algorithm": "tabular_q_learning_with_uniform_stall_fallback",
            "seed": config.seed,
            "Bi": config.model.Bi,
            "grid_shape": [config.n] * 2,
            "solver": "finite_volume",
            "solver_tolerance": None,
            "diagnostic_tolerances": config.solver.model_dump(),
            "training_pde_solves": cost,
            "training_wall_clock_seconds": sum(row["wall_clock_seconds"] for row in training),
            "evaluation_pde_solves": sum(r["number_of_pde_solves"] for r in evaluations),
            "refinement_pde_solves": sum(r["refinement_pde_solves"] for r in evaluations),
            "training_directory": str(training_directory.resolve()),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "evaluation_frozen": True,
            "evaluation_seeds": config.evaluation_seeds,
            "evaluations": evaluations,
            "wall_clock_seconds": perf_counter() - start,
        }
        save_summary(directory, summary)
        return directory, summary
    except Exception as exc:
        log_failure(directory, exc)
        raise
