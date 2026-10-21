"""Train a Q policy, freeze it, and compare held-out runs at equal PDE budgets."""

import json
from pathlib import Path
from time import perf_counter

from oxygen_shape.experiments.optimization import save_search
from oxygen_shape.experiments.reporting import (
    create_run,
    log_failure,
    provenance,
    save_summary,
    write_csv,
)
from oxygen_shape.optimization.config import RLConfig
from oxygen_shape.optimization.q_learning import QPolicy
from oxygen_shape.optimization.random_search import run_search


def run_rl(config: RLConfig, progress=None) -> tuple[Path, dict]:
    directory = create_run(config, "rl_disk_policy")
    start = perf_counter()
    policy = QPolicy()
    training = []
    try:
        for episode in range(config.training_episodes):
            fraction = episode / max(config.training_episodes - 1, 1)
            epsilon = config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)
            seed = config.training_seed_offset + config.seed + episode
            episode_config = config.model_copy(
                update={
                    "seed": seed,
                    "solve_budget": config.training_solve_budget,
                    "max_attempts_per_solve": config.training_attempts_per_solve,
                }
            )
            episode_start = perf_counter()
            result = run_search(episode_config, policy=policy, epsilon=epsilon, training=True)
            path = directory / "training" / f"episode_{episode:04d}"
            save_search(path, result)
            policy.save(path / "policy.npz")
            training.append(
                {
                    "episode": episode,
                    "seed": seed,
                    "epsilon": epsilon,
                    "initial_Q": result.initial_solution.metrics.physical_flux_Q,
                    "final_Q": result.final_solution.metrics.physical_flux_Q,
                    "relative_gain": result.final_solution.metrics.physical_flux_Q
                    / result.initial_solution.metrics.physical_flux_Q
                    - 1,
                    "number_of_pde_solves": result.number_of_pde_solves,
                    "attempts": result.attempts,
                    "solve_budget_completed": result.number_of_pde_solves
                    == config.training_solve_budget,
                    "wall_clock_seconds": perf_counter() - episode_start,
                }
            )
            write_csv(directory / "training.csv", training)
            policy.save(directory / "policy.npz")
            if progress:
                progress(
                    f"training {episode + 1}/{config.training_episodes}: Q="
                    f"{result.final_solution.metrics.physical_flux_Q:.6g}, epsilon={epsilon:.3f}"
                )
        metadata = {
            "algorithm": "tabular_q_learning",
            "states": 8,
            "actions": 12,
            "state_encoding": "4*(gain>=10%) + 2*(P/Pmax>=85%) + (mean distal u>=0.1)",
            "action_encoding": "region*2 + radius_bin; region=2*x_third + upper_half",
            "radii": list(config.radii),
            "training_configuration": config.model_dump(mode="json"),
            "evaluation_frozen": True,
        }
        (directory / "policy_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        save_search(directory, result)
        save_summary(
            directory,
            {
                **provenance(),
                "phase": "rl_training",
                "seed": config.seed,
                "training_pde_solves": sum(r["number_of_pde_solves"] for r in training),
                "wall_clock_seconds": perf_counter() - start,
                "root_state": "last training episode",
            },
        )
        from oxygen_shape.experiments.rl_benchmark import benchmark_checkpoint

        return benchmark_checkpoint(config, directory, progress=progress)
    except Exception as exc:
        log_failure(directory, exc)
        raise
