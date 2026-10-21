"""Bounded rejection sampling supplies distinct admissible choices before ranking.

No candidate concentration is used to form or rank a pool. Refill changes the
proposal distribution and spends geometry work; it does not increase raw yield.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, model_validator

from oxygen_shape.geometry.disk_edits import DiskEdit, PreparedBoundary, prepare_boundary
from oxygen_shape.geometry.raster import RasterGeometry
from oxygen_shape.optimization.donor_sampling import eligible_wall_flux
from oxygen_shape.optimization.guided_config import GuidedConfig
from oxygen_shape.optimization.guided_search import metrics_row, validate_candidate
from oxygen_shape.optimization.perimeter_pool import cheap_screen, draw_pool
from oxygen_shape.optimization.random_search import SearchResult
from oxygen_shape.solvers.finite_volume import solve


class FeasiblePoolConfig(GuidedConfig):
    solve_budget: int = Field(default=17, ge=2, strict=True)
    raw_draw_budget: int = Field(default=6000, ge=1, strict=True)
    pool_size: int = Field(default=8, ge=2, strict=True)
    refill_draw_limit: int = Field(default=256, ge=2, strict=True)
    evaluation_seeds: list[Annotated[int, Field(strict=True, ge=0)]] = Field(
        default_factory=lambda: [181000, 181001, 181002, 181003, 181004]
    )

    @model_validator(mode="after")
    def check_pool(self):
        if self.refill_draw_limit < max(4, self.pool_size):
            raise ValueError("Refill limit must permit the base pool and four alternatives")
        if not self.evaluation_seeds or len(set(self.evaluation_seeds)) != len(
            self.evaluation_seeds
        ):
            raise ValueError("Evaluation seeds must be distinct and nonempty")
        return self


@dataclass(frozen=True)
class FeasiblePool:
    edits: list[DiskEdit]
    rows: list[dict]
    eligible: np.ndarray
    full_checks: int
    validation_seconds: float
    quota_reached: bool


def build_pool(
    config: FeasiblePoolConfig,
    geometry: RasterGeometry,
    initial: RasterGeometry,
    prepared: PreparedBoundary,
    rng: np.random.Generator,
    remaining: int,
    seen: set[bytes],
    *,
    quota: int,
) -> FeasiblePool:
    """At least base-size draws, then refill to quota or the charged finite cap.

    Repeated masks and the incumbent are excluded before ranking. All checks
    include the unchanged source/Neumann faces and the 100/200/400 feature tests.
    No cache or repair can silently bypass validation.
    """
    if quota not in (0, 2, 4) or remaining < 1:
        raise ValueError("Quota must be 0 (fixed), 2 or 4; draws must remain")
    limit = min(remaining, config.pool_size if quota == 0 else config.refill_draw_limit)
    edits, rows, eligible = [], [], []
    signatures = {geometry.inside.tobytes(), *seen}
    full_checks, validation_seconds = 0, 0.0
    for slot in range(limit):
        batch, metadata = draw_pool(geometry, rng, config, prepared, 1)
        edit, row = batch[0], metadata[0]
        reason, perimeter = cheap_screen(edit, initial, config)
        checked = False
        if not reason:
            signature = edit.geometry.inside.tobytes()
            if signature in signatures:
                reason = "Duplicate or incumbent mask"
            else:
                start = perf_counter()
                reason = validate_candidate(edit.geometry, initial, config)
                validation_seconds += perf_counter() - start
                full_checks += 1
                checked = True
                # Even invalid identical masks need no repeated full check here.
                signatures.add(signature)
        edits.append(edit)
        eligible.append(not reason)
        rows.append(
            {
                **row,
                "slot": slot,
                "screen_rejection": reason,
                "perimeter": perimeter,
                "full_checked": checked,
            }
        )
        if quota and len(edits) >= config.pool_size and sum(eligible) >= quota:
            break
    return FeasiblePool(
        edits,
        rows,
        np.asarray(eligible, bool),
        full_checks,
        validation_seconds,
        bool(quota and sum(eligible) >= quota),
    )


def selection_probabilities(pool: FeasiblePool, scores, epsilon_uniform: float) -> np.ndarray:
    """Keep the unconditional first raw draw; otherwise rank eligible choices."""
    if not 0 < epsilon_uniform <= 1:
        raise ValueError("An unconditional raw-proposal exploration floor is required")
    scores = np.asarray(scores, float)
    if scores.shape != pool.eligible.shape or not np.isfinite(scores).all():
        raise ValueError("Finite scores must match pool slots")
    probabilities = np.zeros(len(scores))
    probabilities[0] = epsilon_uniform
    if pool.eligible.any():
        best = np.max(scores[pool.eligible])
        choices = pool.eligible & np.isclose(scores, best, rtol=0, atol=1e-12)
        probabilities[choices] += (1 - epsilon_uniform) / choices.sum()
    else:
        probabilities[0] = 1
    return probabilities


@dataclass(frozen=True)
class FeasibleSearch:
    search: SearchResult
    pools: list[dict]
    full_checks: int
    validation_seconds: float
    seconds: float
    maximum_conservation_error: float
    status: str


def run_feasible_search(
    config: FeasiblePoolConfig,
    initial: RasterGeometry,
    *,
    quota: int,
    ranking: Literal["uniform", "flux"],
    on_evaluation=None,
) -> FeasibleSearch:
    if quota not in (0, 2, 4) or ranking not in ("uniform", "flux"):
        raise ValueError("Invalid pool quota or ranking")
    started = perf_counter()
    reason = validate_candidate(initial, initial, config)
    if reason:
        raise ValueError(f"Invalid starting geometry: {reason}")
    # Raw proposal stream is independent of selection RNG and ranking method.
    proposal_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 0]))
    selection_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 1]))
    geometry = initial
    solution = initial_solution = solve(initial, config.model, config.solver)
    initial_Q = solution.metrics.physical_flux_Q
    maximum_error = solution.metrics.conservation_error
    solves, raw, checks, validation_seconds = 1, 0, 0, 0.0
    history, proposals, pools, seen = [], [], [], set()
    prepared = prepare_boundary(geometry, config.protected_length)

    def record(accepted=False, candidate_Q=None):
        history.append(
            {
                "number_of_pde_solves": solves,
                "raw_draws": raw,
                "accepted": accepted,
                "candidate_Q": candidate_Q,
                **metrics_row(geometry, solution, initial_Q),
            }
        )

    record()
    while solves < config.solve_budget and raw < config.raw_draw_budget:
        pool = build_pool(
            config,
            geometry,
            initial,
            prepared,
            proposal_rng,
            config.raw_draw_budget - raw,
            seen,
            quota=quota,
        )
        checks += pool.full_checks
        validation_seconds += pool.validation_seconds
        scores = np.zeros(len(pool.edits))
        if ranking == "flux":
            flux = eligible_wall_flux(geometry, solution, config.protected_length)
            scores = np.array(
                [
                    flux[r["addition_face"]]
                    * (flux.max() - flux[r["donor_face"]])
                    / config.model.Bi**2
                    for r in pool.rows
                ]
            )
        probabilities = selection_probabilities(pool, scores, config.epsilon_uniform)
        selected = int(selection_rng.choice(len(pool.edits), p=probabilities))
        valid = bool(pool.eligible[selected])
        pool_index = len(pools)
        for i, row in enumerate(pool.rows):
            proposals.append(
                {
                    **row,
                    "raw_draw": raw + i + 1,
                    "pool": pool_index,
                    "selected": i == selected,
                    "eligible": bool(pool.eligible[i]),
                    "selection_probability": float(probabilities[i]),
                    "incumbent_flux_score": float(scores[i]),
                    "uniform_fallback": not pool.eligible.any(),
                }
            )
        raw += len(pool.edits)
        pools.append(
            {
                "pool": pool_index,
                "raw_draws": len(pool.edits),
                "raw_total": raw,
                "eligible": int(pool.eligible.sum()),
                "quota_reached": pool.quota_reached,
                "selected": selected,
                "selected_valid": valid,
                "candidate_Q": None,
                "accepted": False,
            }
        )
        if not valid:
            continue
        edit = pool.edits[selected]
        seen.add(edit.geometry.inside.tobytes())
        candidate = solve(edit.geometry, config.model, config.solver)
        solves += 1
        maximum_error = max(maximum_error, candidate.metrics.conservation_error)
        accepted = candidate.metrics.physical_flux_Q > (
            solution.metrics.physical_flux_Q + config.min_improvement
        )
        if on_evaluation:
            on_evaluation(solves, edit.geometry, candidate, accepted)
        if accepted:
            geometry, solution = edit.geometry, candidate
            prepared = prepare_boundary(geometry, config.protected_length)
            seen.clear()
        pools[-1].update(candidate_Q=candidate.metrics.physical_flux_Q, accepted=accepted)
        record(accepted, candidate.metrics.physical_flux_Q)
    return FeasibleSearch(
        SearchResult(
            initial, geometry, initial_solution, solution, history, proposals, solves, raw
        ),
        pools,
        checks,
        validation_seconds,
        perf_counter() - started,
        maximum_error,
        "complete" if solves == config.solve_budget else "raw_budget_exhausted",
    )
