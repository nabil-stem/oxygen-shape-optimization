"""Discarded solver, full-screening, reward and prior checkpoint checks."""

from pathlib import Path

from oxygen_shape.experiments.candidate_storage import run_stored_rollout
from oxygen_shape.experiments.reporting import write_csv
from oxygen_shape.experiments.study_worker import atomic_json
from oxygen_shape.geometry.generators import rectangle
from oxygen_shape.optimization.candidate_policy import CandidatePolicy
from oxygen_shape.solvers.analytical_1d import slab_flux
from oxygen_shape.solvers.finite_volume import solve

REVISION_CONTROLS = {
    "uniform": ("uniform", "cheap"),
    "screened_uniform": ("screened_uniform", "cheap"),
    "perimeter_pool": ("perimeter_pool", "cheap"),
    "flux_pool": ("flux_pool", "cheap"),
    "untrained": ("untrained", "cheap"),
    "screened_uniform_full": ("screened_uniform", "full"),
    "perimeter_pool_full": ("perimeter_pool", "full"),
    "flux_pool_full": ("flux_pool", "full"),
    "untrained_full": ("untrained", "full"),
}


def revision_preflight(root: Path, config, starts) -> dict:
    slab = solve(rectangle(16, 3, 1 / 16, slab=True), config.model, config.solver)
    if abs(slab.metrics.physical_flux_Q / (3 / 16) - slab_flux(config.model.Bi)) > 1e-9:
        raise ArithmeticError("Revision preflight analytical slab failed")
    rows, cost, error = [], 1, slab.metrics.conservation_error
    for family in config.families:
        case = next(s for s in starts if s.family == family and s.name.endswith("canonical"))
        jobs = [
            (name, method, screen, None) for name, (method, screen) in REVISION_CONTROLS.items()
        ]
        jobs += [(a.name, "learned", a.candidate_screening, a) for a in config.arms]
        for name, method, screen, arm in jobs:
            cfg = config if arm is None else config.for_arm(arm)
            cfg = cfg.model_copy(
                update={
                    "initial_shape": family,
                    "seed": 1990000 + len(rows),
                    "solve_budget": 2,
                    "candidate_screening": screen,
                }
            )
            policy = CandidatePolicy.with_flux_prior(arm.flux_prior_scale) if arm else None
            result, row = run_stored_rollout(
                root / "preflight" / f"{family}_{name}",
                cfg,
                case.geometry,
                method=method,
                policy=policy,
                training=arm is not None,
                epsilon=0.5,
            )
            if result.status != "complete":
                raise ArithmeticError(f"Preflight incomplete: {family}/{name}")
            if arm is not None and arm.discount == 1 and arm.invalid_reward == 0:
                target = config.reward_scale * (row["final_Q"] / row["initial_Q"] - 1)
                if abs(row["reward_sum"] - target) > 1e-10:
                    raise ArithmeticError("Undiscounted physical reward identity failed")
            rows.append({"name": name, "family": family, **row})
            cost += row["number_of_pde_solves"]
            error = max(error, row["max_conservation_error"])
    checkpoint = root / "preflight/prior_reload.npz"
    prior = CandidatePolicy.with_flux_prior(1)
    prior.save(checkpoint, "preflight", [])
    restored, _ = CandidatePolicy.load(checkpoint, "preflight")
    if prior.digest() != restored.digest():
        raise ArithmeticError("Flux-prior checkpoint failed round-trip")
    result, row = run_stored_rollout(
        root / "preflight/frozen_prior", cfg, case.geometry, method="learned", policy=restored
    )
    if result.status != "complete" or prior.digest() != restored.digest():
        raise ArithmeticError("Frozen prior reload evaluation failed")
    rows.append({"name": "frozen_prior", "family": family, **row})
    cost += row["number_of_pde_solves"]
    error = max(error, row["max_conservation_error"])
    ready = {
        "status": "ready",
        "number_of_pde_solves": cost,
        "max_conservation_error": error,
        "controls": list(REVISION_CONTROLS),
        "arms": [a.name for a in config.arms],
    }
    write_csv(root / "preflight.csv", rows)
    atomic_json(root / "readiness.json", ready)
    return ready
