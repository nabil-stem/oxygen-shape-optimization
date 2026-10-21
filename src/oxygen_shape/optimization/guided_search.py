"""Batch-greedy search with fully charged empirical paired perturbations."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from oxygen_shape.geometry.baselines import matched_area
from oxygen_shape.geometry.constraints import feature_violations, validate_shape
from oxygen_shape.geometry.disk_edits import prepare_boundary, propose_paired_disk
from oxygen_shape.geometry.raster import RasterGeometry, refine_geometry
from oxygen_shape.optimization.donor_sampling import eligible_wall_flux
from oxygen_shape.optimization.guided_config import GuidedConfig
from oxygen_shape.optimization.random_search import SearchResult, constraints_for
from oxygen_shape.optimization.sensitivity import (
    PairResponse,
    flux_probabilities,
    sensitivity_probabilities,
)
from oxygen_shape.solvers.finite_volume import SolveResult, solve

Method = Literal["uniform", "flux", "sensitivity"]


@dataclass(frozen=True)
class GuidedResult:
    search: SearchResult
    status: str
    rounds: list[dict]
    probes: list[dict]
    probe_pde_solves: int
    maximum_candidate_conservation_error: float


def validate_candidate(
    geometry: RasterGeometry,
    initial: RasterGeometry,
    config: GuidedConfig,
) -> str:
    report = validate_shape(geometry, constraints_for(config), initial)
    if not report.valid:
        return "; ".join(report.violations)
    for factor in config.refinement_factors:
        reasons = feature_violations(
            refine_geometry(geometry, factor), config.minimum_feature_width
        )
        if reasons:
            return f"Feature check at factor {factor}: " + "; ".join(reasons)
    return ""


def metrics_row(geometry: RasterGeometry, solution: SolveResult, reference_Q: float) -> dict:
    metrics = solution.metrics
    return {
        "physical_flux_Q": metrics.physical_flux_Q,
        "ideal_flux_Q": metrics.ideal_flux_Q,
        "efficiency_eta": metrics.efficiency_eta,
        "area": geometry.area,
        "robin_perimeter": metrics.robin_perimeter,
        "roughness": None,
        "constraint_violations": "",
        "optimization_score": metrics.physical_flux_Q / reference_Q,
        "conservation_error": metrics.conservation_error,
    }


def run_guided_search(
    config: GuidedConfig,
    method: Method,
    *,
    on_map: Callable | None = None,
) -> GuidedResult:
    if method not in ("uniform", "flux", "sensitivity"):
        raise ValueError("Unknown sampling method")
    rng = np.random.default_rng(config.seed)
    initial = matched_area(config.initial_shape, config.n, config.target_area).geometry
    reasons = validate_candidate(initial, initial, config)
    if reasons:
        raise ValueError(f"Initial geometry: {reasons}")
    geometry = initial
    solution = solve(initial, config.model, config.solver)
    initial_solution = solution
    reference_Q = solution.metrics.physical_flux_Q
    maximum_error = solution.metrics.conservation_error
    history, proposals, probes, rounds = [], [], [], []
    solves, attempts, probe_solves = 1, 0, 0
    max_attempts = config.solve_budget * config.max_attempts_per_solve

    def record(best_geometry, best_solution, round_index, phase):
        history.append(
            {
                "attempt": attempts,
                "number_of_pde_solves": solves,
                "round": round_index,
                "phase": phase,
                **metrics_row(best_geometry, best_solution, reference_Q),
                "accepted": False,
                "reward": 0.0,
            }
        )

    record(geometry, solution, 0, "initial")
    while solves < config.solve_budget and attempts < max_attempts:
        round_index = len(rounds) + 1
        round_start = solves
        base_Q = solution.metrics.physical_flux_Q
        best_geometry, best_solution = geometry, solution
        prepared = prepare_boundary(geometry, config.protected_length)
        points = prepared.points
        if not len(points):
            raise ValueError("No eligible boundary measure for search")
        probabilities = (
            flux_probabilities(
                eligible_wall_flux(geometry, solution, config.protected_length),
                config.epsilon_uniform,
            )
            if method == "flux"
            else None
        )
        responses: list[PairResponse] = []
        seen: set[bytes] = set()
        evaluated, consecutive_invalid, map_ready = 0, 0, False
        while (
            evaluated < config.batch_size
            and solves < config.solve_budget
            and attempts < max_attempts
        ):
            is_probe = method == "sensitivity" and evaluated < config.probe_count
            if method == "sensitivity" and not is_probe and not map_ready:
                probabilities = sensitivity_probabilities(
                    points,
                    responses,
                    config.kernel_bandwidth,
                    config.epsilon_uniform,
                )
                map_ready = True
                if on_map:
                    on_map(round_index, geometry, points, probabilities, responses)
            fallback = method != "uniform" and consecutive_invalid >= config.policy_stall_limit
            guided = not is_probe and probabilities is not None and not fallback
            pair_index = int(
                rng.choice(probabilities.size, p=probabilities.ravel())
                if guided
                else rng.integers(len(points) ** 2)
            )
            pair = divmod(pair_index, len(points))
            proposal = propose_paired_disk(
                geometry,
                rng,
                float(rng.uniform(*config.radii)),
                config.protected_length,
                face_pair=pair,
                prepared=prepared,
            )
            attempts += 1
            rejection = proposal.rejection
            candidate_Q = None
            sensitivity = None
            if proposal.geometry is not None:
                rejection = validate_candidate(proposal.geometry, initial, config)
                signature = proposal.geometry.inside.tobytes()
                if not rejection and signature in seen:
                    rejection = "Duplicate candidate within round"
                if not rejection:
                    seen.add(signature)
                    candidate = solve(proposal.geometry, config.model, config.solver)
                    solves += 1
                    evaluated += 1
                    maximum_error = max(maximum_error, candidate.metrics.conservation_error)
                    candidate_Q = candidate.metrics.physical_flux_Q
                    p = proposal.parameters
                    response = PairResponse(
                        tuple(points[pair[0]]),
                        tuple(points[pair[1]]),
                        candidate_Q - base_Q,
                        int(p["changed_cells"]) * geometry.h**2,
                    )
                    sensitivity = response.sensitivity
                    if is_probe:
                        responses.append(response)
                        probe_solves += 1
                        probes.append(
                            {
                                "round": round_index,
                                "attempt": attempts,
                                "base_Q": base_Q,
                                "candidate_Q": candidate_Q,
                                "delta_Q": response.delta_Q,
                                "reallocated_area": response.reallocated_area,
                                "empirical_sensitivity": response.sensitivity,
                                "addition_x": response.addition[0],
                                "addition_y": response.addition[1],
                                "donor_x": response.donor[0],
                                "donor_y": response.donor[1],
                            }
                        )
                    if candidate_Q > best_solution.metrics.physical_flux_Q + config.min_improvement:
                        best_geometry, best_solution = proposal.geometry, candidate
            consecutive_invalid = consecutive_invalid + 1 if candidate_Q is None else 0
            phase = "probe" if is_probe else "guided" if guided else "uniform"
            proposals.append(
                {
                    "round": round_index,
                    "attempt": attempts,
                    "phase": phase,
                    "candidate_Q": candidate_Q,
                    "round_base_Q": base_Q,
                    "empirical_sensitivity": sensitivity,
                    "rejection": rejection,
                    "uniform_fallback": fallback,
                    **proposal.parameters,
                }
            )
            if candidate_Q is not None:
                record(best_geometry, best_solution, round_index, phase)
        improvement = best_solution.metrics.physical_flux_Q - base_Q
        accepted = improvement > config.min_improvement
        if accepted:
            geometry, solution = best_geometry, best_solution
            history[-1].update(accepted=True, reward=improvement / reference_Q)
        rounds.append(
            {
                "round": round_index,
                "base_Q": base_Q,
                "final_Q": solution.metrics.physical_flux_Q,
                "accepted": accepted,
                "candidate_pde_solves": solves - round_start,
                "probe_pde_solves": len(responses),
                "map_generated": map_ready,
                "number_of_pde_solves": solves,
                "attempts_so_far": attempts,
            }
        )
    search = SearchResult(
        initial,
        geometry,
        initial_solution,
        solution,
        history,
        proposals,
        solves,
        attempts,
    )
    return GuidedResult(
        search,
        "complete" if solves == config.solve_budget else "incomplete_attempt_budget",
        rounds,
        probes,
        probe_solves,
        maximum_error,
    )
