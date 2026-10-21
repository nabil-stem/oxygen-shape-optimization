"""A common budgeted environment for five controls and a learned selector."""

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np

from oxygen_shape.geometry.disk_edits import prepare_boundary
from oxygen_shape.geometry.raster import RasterGeometry
from oxygen_shape.optimization.candidate_policy import CandidatePolicy, mixed_probabilities
from oxygen_shape.optimization.donor_sampling import eligible_wall_flux
from oxygen_shape.optimization.feasible_pool import build_pool
from oxygen_shape.optimization.guided_search import metrics_row, validate_candidate
from oxygen_shape.optimization.perimeter_pool import cheap_screen, draw_pool
from oxygen_shape.optimization.pioneer_config import PioneerRLConfig
from oxygen_shape.optimization.random_search import SearchResult
from oxygen_shape.solvers.finite_volume import solve

Control = Literal[
    "uniform", "screened_uniform", "perimeter_pool", "flux_pool", "untrained", "learned"
]
CONTROLS = ("uniform", "screened_uniform", "perimeter_pool", "flux_pool", "untrained")


@dataclass(frozen=True)
class CandidateRollout:
    search: SearchResult
    status: str
    transitions: list[dict]
    candidate_metrics: list[dict]
    selected_features: np.ndarray
    maximum_conservation_error: float
    seconds: float
    full_geometry_checks: int = 0
    geometry_validation_seconds: float = 0


def candidate_features(
    geometry, solution, config, reference_Q, solves, raw_count, prepared, edits, rows, screens, flux
) -> np.ndarray:
    remaining = (config.solve_budget - solves) / config.solve_budget
    perimeter = solution.metrics.robin_perimeter
    eta = solution.metrics.efficiency_eta
    state = [
        1,
        remaining,
        (config.raw_draw_budget - raw_count) / config.raw_draw_budget,
        np.clip(solution.metrics.physical_flux_Q / reference_Q - 1, -1, 1),
        perimeter / config.perimeter_limit,
        eta,
        float(np.nanmean(solution.concentration)),
    ]
    features = []
    for edit, row, screen in zip(edits, rows, screens, strict=True):
        add, donor = int(row["addition_face"]), int(row["donor_face"])
        ax, ay = prepared.points[add]
        dx, dy = prepared.points[donor]
        radius = float(row["radius"]) / 0.25
        area = float(edit.parameters.get("changed_cells", 0)) * geometry.h**2 / config.target_area
        delta = (
            0
            if screen["perimeter"] is None
            else (screen["perimeter"] - perimeter) / config.perimeter_limit
        )
        ja, jd = flux[add] / config.model.Bi, flux[donor] / config.model.Bi
        product = ja * (flux.max() / config.model.Bi - jd)
        candidate = [
            radius,
            area,
            delta,
            ax,
            ay,
            dx,
            dy,
            ja,
            jd,
            product,
            float(not screen["screen_rejection"]),
        ]
        interactions = [
            remaining * delta,
            remaining * product,
            area * state[4],
            ax * eta,
            dx * eta,
            radius * state[4],
        ]
        features.append(state + candidate + interactions)
    result = np.asarray(features, float)
    if not np.isfinite(result).all():
        raise ArithmeticError("Non-finite incumbent-only candidate features")
    return result


def run_candidate_rollout(
    config: PioneerRLConfig,
    initial: RasterGeometry,
    *,
    method: Control,
    policy: CandidatePolicy | None = None,
    training: bool = False,
    epsilon: float = 0,
    on_evaluation: Callable | None = None,
) -> CandidateRollout:
    if method not in (*CONTROLS, "learned") or (method == "learned" and policy is None):
        raise ValueError("Unknown method or missing learned policy")
    if training and method != "learned":
        raise ValueError("Only the learned selector may update")
    if not 0 <= epsilon <= 1:
        raise ValueError("Invalid exploration probability")
    reason = validate_candidate(initial, initial, config)
    if reason:
        raise ValueError(f"Invalid starting geometry: {reason}")
    started = perf_counter()
    rng = np.random.default_rng(config.seed)
    if config.refill_quota:
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, 0]))
        selection_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 1]))
    else:
        selection_rng = rng
    geometry = initial
    solution = solve(geometry, config.model, config.solver)
    initial_solution = solution
    reference_Q = solution.metrics.physical_flux_Q
    if reference_Q <= 0:
        raise ValueError("Positive source flux is required")
    solves, raw_count, checked = 1, 0, 0
    full_checks, validation_seconds = 0, 0.0
    maximum_error = solution.metrics.conservation_error
    history, proposals, transitions, candidate_rows, selected_features, seen = (
        [],
        [],
        [],
        [],
        [],
        set(),
    )
    frozen_digest = policy.digest() if policy else None
    prepared = prepare_boundary(geometry, config.protected_length)
    validation_cache: dict[tuple, str] = {}
    needs_features = method in ("learned", "untrained", "flux_pool")
    flux = (
        eligible_wall_flux(geometry, solution, config.protected_length) if needs_features else None
    )

    def make_pool():
        nonlocal raw_count, full_checks, validation_seconds
        if config.refill_quota and method != "uniform":
            before = raw_count
            built = build_pool(
                config,
                geometry,
                initial,
                prepared,
                rng,
                config.raw_draw_budget - raw_count,
                seen,
                quota=config.refill_quota,
            )
            edits = built.edits
            raw_count += len(edits)
            full_checks += built.full_checks
            validation_seconds += built.validation_seconds
            # Keep the historical cheap-pass feature; full eligibility is separate.
            screens, rows = [], []
            for edit, record in zip(edits, built.rows, strict=True):
                cheap_reason, perimeter = cheap_screen(edit, initial, config)
                screens.append(
                    {
                        "screen_rejection": cheap_reason,
                        "perimeter": perimeter,
                        "full_checked": record["full_checked"],
                        "full_rejection": record["screen_rejection"],
                        "validation_cache_hit": False,
                    }
                )
                rows.append({k: v for k, v in record.items() if k not in screens[-1]})
            features = (
                candidate_features(
                    geometry,
                    solution,
                    config,
                    reference_Q,
                    solves,
                    raw_count,
                    prepared,
                    edits,
                    rows,
                    screens,
                    flux,
                )
                if needs_features
                else None
            )
            return edits, rows, screens, features, before, raw_count
        size = min(
            1 if method == "uniform" else config.pool_size, config.raw_draw_budget - raw_count
        )
        before = raw_count
        edits, rows = draw_pool(geometry, rng, config, prepared, size)
        raw_count += size
        screens = [
            dict(
                zip(
                    ("screen_rejection", "perimeter"), cheap_screen(e, initial, config), strict=True
                )
            )
            for e in edits
        ]
        for edit, screen in zip(edits, screens, strict=True):
            screen.update(full_checked=False, full_rejection=None, validation_cache_hit=False)
            if config.candidate_screening == "full" and not screen["screen_rejection"]:
                signature = edit.geometry.inside.tobytes()
                cache_key = (
                    edit.geometry.h,
                    edit.geometry.inside.shape,
                    edit.geometry.source_faces,
                    edit.geometry.neumann_faces,
                    signature,
                )
                screen["full_checked"] = True
                if cache_key in validation_cache:
                    reason = validation_cache[cache_key]
                    screen["validation_cache_hit"] = True
                else:
                    started_validation = perf_counter()
                    reason = validate_candidate(edit.geometry, initial, config)
                    validation_seconds += perf_counter() - started_validation
                    full_checks += 1
                    validation_cache[cache_key] = reason
                if not reason and signature in seen:
                    reason = "Duplicate candidate at fixed incumbent"
                screen["full_rejection"] = reason
        features = (
            candidate_features(
                geometry,
                solution,
                config,
                reference_Q,
                solves,
                raw_count,
                prepared,
                edits,
                rows,
                screens,
                flux,
            )
            if method in ("learned", "untrained", "flux_pool")
            else None
        )
        return edits, rows, screens, features, before, raw_count

    def probabilities(pool):
        edits, rows, screens, features, _, _ = pool
        eligible = np.array(
            [not s["screen_rejection"] and not s["full_rejection"] for s in screens]
        )
        if method == "uniform":
            return np.ones(1)
        if method == "perimeter_pool":
            scores = np.array(
                [-s["perimeter"] if s["perimeter"] is not None else 0 for s in screens]
            )
        elif method == "flux_pool":
            scores = features[:, 16]
        elif method == "learned":
            scores = policy.scores(features)
        else:
            scores = np.zeros(len(edits))
        return mixed_probabilities(
            scores, eligible, epsilon if method == "learned" else 0, config.epsilon_uniform
        )

    history.append(
        {
            "number_of_pde_solves": 1,
            "raw_draws": 0,
            "accepted": False,
            "reward": 0,
            **metrics_row(geometry, solution, reference_Q),
        }
    )
    pool = make_pool()
    while True:
        edits, rows, screens, features, before, after = pool
        p = probabilities(pool)
        action = int(selection_rng.choice(len(edits), p=p))
        eligible_count = sum(not s["screen_rejection"] and not s["full_rejection"] for s in screens)
        edit = edits[action]
        selected_phi = features[action].copy() if features is not None else None
        reason = edit.rejection
        candidate = None
        accepted = False
        reward = config.invalid_reward
        if edit.geometry is not None:
            screen = screens[action]
            if config.candidate_screening == "full":
                reason = screen["screen_rejection"] or screen["full_rejection"] or ""
            else:
                started_validation = perf_counter()
                reason = validate_candidate(edit.geometry, initial, config)
                validation_seconds += perf_counter() - started_validation
                full_checks += 1
            signature = edit.geometry.inside.tobytes()
            if not reason and signature in seen:
                reason = "Duplicate candidate at fixed incumbent"
            if not reason:
                seen.add(signature)
                candidate = solve(edit.geometry, config.model, config.solver)
                solves += 1
                maximum_error = max(maximum_error, candidate.metrics.conservation_error)
                delta = candidate.metrics.physical_flux_Q - solution.metrics.physical_flux_Q
                accepted = delta > config.min_improvement
                reward = config.reward_scale * delta / reference_Q if accepted else 0.0
                candidate_rows.append(
                    {
                        "solve": solves,
                        "accepted": accepted,
                        "delta_Q": delta,
                        **candidate.metrics.__dict__,
                    }
                )
                if on_evaluation:
                    on_evaluation(solves, edit.geometry, candidate, accepted)
                if accepted:
                    geometry, solution = edit.geometry, candidate
                    prepared = prepare_boundary(geometry, config.protected_length)
                    flux = (
                        eligible_wall_flux(geometry, solution, config.protected_length)
                        if needs_features
                        else None
                    )
                    seen.clear()
                    validation_cache.clear()
        checked += 1
        for i, row in enumerate(rows):
            proposals.append(
                {
                    "raw_draw": before + i + 1,
                    "decision": checked,
                    "selected": i == action,
                    "selection_probability": float(p[i]),
                    "rejection": reason if i == action else "Not selected",
                    "uniform_fallback": False,
                    **screens[i],
                    **row,
                }
            )
        terminal = solves >= config.solve_budget or after >= config.raw_draw_budget
        # The next pool is a charged environment transition, not a free bootstrap sample.
        next_pool = None if terminal else make_pool()
        bootstrap, td_error = 0.0, None
        if training:
            if not terminal:
                bootstrap = float(probabilities(next_pool) @ policy.scores(next_pool[3]))
            td_error = policy.update(
                selected_phi,
                reward,
                bootstrap,
                alpha=config.learning_rate,
                gamma=config.discount,
                terminal=terminal,
            )
        transitions.append(
            {
                "decision": checked,
                "number_of_pde_solves": solves,
                "raw_draws": after,
                "action": action,
                "pool_draws": len(edits),
                "eligible_candidates": eligible_count,
                "quota_reached": bool(
                    config.refill_quota and eligible_count >= config.refill_quota
                ),
                "reward": reward,
                "accepted": accepted,
                "candidate_Q": None if candidate is None else candidate.metrics.physical_flux_Q,
                "rejection": reason,
                "terminal": terminal,
                "bootstrap": bootstrap,
                "td_error": td_error,
                "pde_remaining": config.solve_budget - solves,
            }
        )
        if selected_phi is not None:
            selected_features.append(selected_phi)
        history.append(
            {
                "number_of_pde_solves": solves,
                "raw_draws": after,
                "accepted": accepted,
                "reward": reward,
                **metrics_row(geometry, solution, reference_Q),
            }
        )
        if terminal:
            break
        pool = next_pool
    if not training and policy is not None and policy.digest() != frozen_digest:
        raise ArithmeticError("Frozen evaluation changed policy weights")
    result = SearchResult(
        initial, geometry, initial_solution, solution, history, proposals, solves, raw_count
    )
    return CandidateRollout(
        result,
        "complete" if solves == config.solve_budget else "incomplete",
        transitions,
        candidate_rows,
        np.asarray(selected_features),
        maximum_error,
        perf_counter() - started,
        full_checks,
        validation_seconds,
    )
