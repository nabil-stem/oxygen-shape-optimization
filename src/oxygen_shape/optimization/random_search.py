"""PDE-scored greedy disk search shared by random and learned proposal policies."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from oxygen_shape.geometry.baselines import matched_area
from oxygen_shape.geometry.constraints import (
    ShapeConstraints,
    feature_violations,
    raster_perimeter,
    validate_shape,
)
from oxygen_shape.geometry.disk_edits import propose_paired_disk
from oxygen_shape.geometry.raster import RasterGeometry, refine_geometry
from oxygen_shape.optimization.config import RLConfig, SearchConfig
from oxygen_shape.optimization.q_learning import QPolicy, observe
from oxygen_shape.solvers.finite_volume import SolveResult, solve


@dataclass(frozen=True)
class SearchResult:
    initial_geometry: RasterGeometry
    final_geometry: RasterGeometry
    initial_solution: SolveResult
    final_solution: SolveResult
    history: list[dict]
    proposals: list[dict]
    number_of_pde_solves: int
    attempts: int


def constraints_for(config: SearchConfig) -> ShapeConstraints:
    return ShapeConstraints(
        config.target_area,
        config.perimeter_limit,
        config.minimum_feature_width,
        config.protected_length,
    )


def run_search(
    config: SearchConfig,
    *,
    policy: QPolicy | None = None,
    epsilon: float = 0.0,
    training: bool = False,
    on_step: Callable[[int, RasterGeometry, dict], None] | None = None,
) -> SearchResult:
    if training and (policy is None or not isinstance(config, RLConfig)):
        raise ValueError("Training requires a Q policy and RLConfig")
    rng = np.random.default_rng(config.seed)
    initial = matched_area(config.initial_shape, config.n, config.target_area).geometry
    constraints = constraints_for(config)
    validate_shape(initial, constraints, initial).require_valid()
    geometry = initial
    solution = solve(geometry, config.model, config.solver)
    initial_solution = solution
    reference_Q = solution.metrics.physical_flux_Q
    if reference_Q <= 0:
        raise ValueError("Search requires positive initial Q (use Bi>0)")
    history, proposals = [], []
    solves, attempts = 1, 0
    consecutive_invalid = 0
    max_attempts = config.solve_budget * config.max_attempts_per_solve

    def record(accepted: bool, reward: float) -> dict:
        m = solution.metrics
        row = {
            "attempt": attempts,
            "number_of_pde_solves": solves,
            "physical_flux_Q": m.physical_flux_Q,
            "ideal_flux_Q": m.ideal_flux_Q,
            "efficiency_eta": m.efficiency_eta,
            "area": geometry.area,
            "robin_perimeter": raster_perimeter(geometry),
            "roughness": None,
            "constraint_violations": "",
            "optimization_score": m.physical_flux_Q / reference_Q,
            "conservation_error": m.conservation_error,
            "accepted": accepted,
            "reward": reward,
        }
        history.append(row)
        if on_step:
            on_step(attempts, geometry, row)
        return row

    record(False, 0.0)
    while solves < config.solve_budget and attempts < max_attempts:
        state = observe(geometry, solution, reference_Q, config.perimeter_limit)
        fallback = (
            policy is not None and not training and consecutive_invalid >= config.policy_stall_limit
        )
        action = -1 if policy is None or fallback else policy.choose(state, rng, epsilon)
        if action < 0:
            region = None
            radius = float(rng.uniform(*config.radii))
        else:
            region, radius_bin = divmod(action, 2)
            lower, upper = config.radii
            midpoint = (lower + upper) / 2
            radius = float(
                rng.uniform(
                    lower if radius_bin == 0 else midpoint, midpoint if radius_bin == 0 else upper
                )
            )
        proposal = propose_paired_disk(
            geometry, rng, radius, config.protected_length, region=region
        )
        attempts += 1
        reasons = proposal.rejection
        candidate_Q = None
        accepted = False
        reward = config.invalid_reward if isinstance(config, RLConfig) else 0.0
        if proposal.geometry is not None:
            report = validate_shape(proposal.geometry, constraints, initial)
            reasons = "; ".join(report.violations)
            if report.valid:
                for factor in config.refinement_factors:
                    failures = feature_violations(
                        refine_geometry(proposal.geometry, factor), config.minimum_feature_width
                    )
                    if failures:
                        reasons = f"feature check at factor {factor}: " + "; ".join(failures)
                        break
            if report.valid and not reasons:
                candidate = solve(proposal.geometry, config.model, config.solver)
                solves += 1
                candidate_Q = candidate.metrics.physical_flux_Q
                delta = candidate_Q - solution.metrics.physical_flux_Q
                accepted = delta > config.min_improvement
                reward = (
                    (config.reward_scale if isinstance(config, RLConfig) else 1.0)
                    * max(delta, 0.0)
                    / reference_Q
                    if accepted
                    else 0.0
                )
                if accepted:
                    geometry, solution = proposal.geometry, candidate
        terminal = solves >= config.solve_budget or attempts >= max_attempts
        next_state = observe(geometry, solution, reference_Q, config.perimeter_limit)
        consecutive_invalid = consecutive_invalid + 1 if candidate_Q is None else 0
        td_error = None
        if training:
            td_error = policy.update(
                state,
                action,
                reward,
                next_state,
                alpha=config.learning_rate,
                gamma=config.discount,
                terminal=terminal,
            )
        proposals.append(
            {
                "attempt": attempts,
                "action": action,
                "state": state,
                "next_state": next_state,
                "candidate_Q": candidate_Q,
                "accepted": accepted,
                "reward": reward,
                "td_error": td_error,
                "rejection": reasons,
                "uniform_fallback": fallback,
                **proposal.parameters,
            }
        )
        record(accepted, reward)
    if solves != config.solve_budget and not training:
        raise RuntimeError(
            f"Only {solves}/{config.solve_budget} solves after {attempts} attempts; "
            "proposal feasibility exhausted before the requested budget"
        )
    return SearchResult(
        initial, geometry, initial_solution, solution, history, proposals, solves, attempts
    )
