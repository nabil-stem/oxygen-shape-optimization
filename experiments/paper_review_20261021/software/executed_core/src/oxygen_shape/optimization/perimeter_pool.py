"""Geometry-only proposal ranking; global PDE flux remains the acceptance rule."""

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np

from oxygen_shape.geometry.constraints import raster_perimeter
from oxygen_shape.geometry.disk_edits import (
    DiskEdit,
    PreparedBoundary,
    prepare_boundary,
    propose_paired_disk,
)
from oxygen_shape.geometry.raster import RasterGeometry
from oxygen_shape.optimization.guided_config import GuidedConfig
from oxygen_shape.optimization.guided_search import metrics_row, validate_candidate
from oxygen_shape.optimization.random_search import SearchResult
from oxygen_shape.solvers.finite_volume import solve

PoolMethod = Literal["uniform", "perimeter_pool", "screened_uniform"]


def cheap_screen(
    edit: DiskEdit, reference: RasterGeometry, config: GuidedConfig
) -> tuple[str, float | None]:
    """Necessary checks only: passing never certifies full geometric validity."""
    if edit.geometry is None:
        return edit.rejection or "No candidate geometry", None
    g = edit.geometry
    perimeter = raster_perimeter(g)
    if g.h != reference.h or g.inside.shape != reference.inside.shape:
        return "Grid changed", perimeter
    protected = (np.arange(g.inside.shape[1]) + 0.5) * g.h < config.protected_length
    if not np.array_equal(g.inside[:, protected], reference.inside[:, protected]):
        return "Protected source collar changed", perimeter
    if abs(g.area - config.target_area) > 1e-12:
        return "Area tolerance exceeded", perimeter
    if perimeter <= 0 or perimeter > config.perimeter_limit + 1e-12:
        return "Robin perimeter budget violated", perimeter
    return "", perimeter


def choose_pool(
    edits: list[DiskEdit],
    reference: RasterGeometry,
    config: GuidedConfig,
    rng: np.random.Generator,
    epsilon_uniform: float,
    *,
    selection: Literal["perimeter", "screened_uniform"] = "perimeter",
) -> tuple[int, str, list[dict]]:
    if (
        not edits
        or not 0 < epsilon_uniform <= 1
        or selection not in ("perimeter", "screened_uniform")
    ):
        raise ValueError("Use a nonempty pool and 0 < epsilon_uniform <= 1")
    screens = [
        dict(
            zip(("screen_rejection", "perimeter"), cheap_screen(e, reference, config), strict=True)
        )
        for e in edits
    ]
    if rng.random() < epsilon_uniform:
        return 0, "uniform", screens
    eligible = [i for i, s in enumerate(screens) if not s["screen_rejection"]]
    if not eligible:
        return 0, "fallback", screens
    if selection == "screened_uniform":
        return int(rng.choice(eligible)), "screened_uniform", screens
    minimum = min(screens[i]["perimeter"] for i in eligible)
    tied = [i for i in eligible if abs(screens[i]["perimeter"] - minimum) <= 1e-12]
    return int(rng.choice(tied)), "perimeter", screens


def draw_pool(
    geometry: RasterGeometry,
    rng: np.random.Generator,
    config: GuidedConfig,
    prepared: PreparedBoundary,
    size: int,
) -> tuple[list[DiskEdit], list[dict]]:
    if size < 1:
        raise ValueError("Pool size must be positive")
    edits, rows = [], []
    for slot in range(size):
        seed = int(rng.integers(0, np.iinfo(np.int64).max))
        local = np.random.default_rng(seed)
        started = perf_counter()
        pair = divmod(int(local.integers(len(prepared.points) ** 2)), len(prepared.points))
        edit = propose_paired_disk(
            geometry,
            local,
            float(local.uniform(*config.radii)),
            config.protected_length,
            face_pair=pair,
            prepared=prepared,
        )
        rows.append(
            {
                "slot": slot,
                "proposal_seed": seed,
                "addition_face": pair[0],
                "donor_face": pair[1],
                "generation_seconds": perf_counter() - started,
                **edit.parameters,
            }
        )
        edits.append(edit)
    return edits, rows


@dataclass(frozen=True)
class PoolSearchResult:
    search: SearchResult
    status: str
    selected_checks: int
    maximum_conservation_error: float
    geometry_seconds: float


def run_pool_search(
    config: GuidedConfig,
    initial: RasterGeometry,
    *,
    method: PoolMethod,
    pool_size: int,
    epsilon_uniform: float,
    raw_budget: int,
    on_evaluation: Callable | None = None,
    progress=None,
) -> PoolSearchResult:
    if (
        method not in ("uniform", "perimeter_pool", "screened_uniform")
        or pool_size < 1
        or raw_budget < 1
    ):
        raise ValueError("Invalid proposal method or budget")
    reason = validate_candidate(initial, initial, config)
    if reason:
        raise ValueError(f"Initial geometry: {reason}")
    rng = np.random.default_rng(config.seed)
    geometry = initial
    solution = solve(initial, config.model, config.solver)
    initial_solution = solution
    reference_Q = solution.metrics.physical_flux_Q
    maximum_error = solution.metrics.conservation_error
    history, proposals, seen = [], [], set()
    solves, raw_count, selected_checks, geometry_seconds = 1, 0, 0, 0.0
    start = perf_counter()
    prepared = prepare_boundary(geometry, config.protected_length)
    geometry_seconds += perf_counter() - start

    def record(accepted: bool, candidate_Q=None):
        history.append(
            {
                "number_of_pde_solves": solves,
                "raw_draws": raw_count,
                "selected_checks": selected_checks,
                "accepted": accepted,
                "candidate_Q": candidate_Q,
                **metrics_row(geometry, solution, reference_Q),
            }
        )

    record(False)
    while solves < config.solve_budget and raw_count < raw_budget:
        start = perf_counter()
        size = min(1 if method == "uniform" else pool_size, raw_budget - raw_count)
        edits, rows = draw_pool(geometry, rng, config, prepared, size)
        selected, mode, screens = choose_pool(
            edits,
            initial,
            config,
            rng,
            1.0 if method == "uniform" else epsilon_uniform,
            selection="screened_uniform" if method == "screened_uniform" else "perimeter",
        )
        proposal = edits[selected]
        reason = proposal.rejection
        signature = None
        if proposal.geometry is not None:
            reason = validate_candidate(proposal.geometry, initial, config)
            signature = proposal.geometry.inside.tobytes()
            if not reason and signature in seen:
                reason = "Duplicate candidate at fixed incumbent"
        selected_checks += 1
        for i, row in enumerate(rows):
            proposals.append(
                {
                    "raw_draw": raw_count + i + 1,
                    "selected_check": selected_checks,
                    "selected": i == selected,
                    "selection_mode": mode,
                    "rejection": reason if i == selected else "Not selected",
                    "uniform_fallback": mode == "fallback",
                    **screens[i],
                    **row,
                }
            )
        raw_count += size
        geometry_seconds += perf_counter() - start
        if reason:
            if progress and (raw_count // 1000 != (raw_count - size) // 1000):
                progress(f"{method}: raw={raw_count}, solves={solves}/{config.solve_budget}")
            continue
        seen.add(signature)
        candidate = solve(proposal.geometry, config.model, config.solver)
        solves += 1
        maximum_error = max(maximum_error, candidate.metrics.conservation_error)
        accepted = (
            candidate.metrics.physical_flux_Q
            > solution.metrics.physical_flux_Q + config.min_improvement
        )
        if on_evaluation:
            on_evaluation(solves, proposal.geometry, candidate, accepted)
        if accepted:
            geometry, solution = proposal.geometry, candidate
            start = perf_counter()
            prepared = prepare_boundary(geometry, config.protected_length)
            geometry_seconds += perf_counter() - start
            seen.clear()
        record(accepted, candidate.metrics.physical_flux_Q)
    return PoolSearchResult(
        SearchResult(
            initial, geometry, initial_solution, solution, history, proposals, solves, raw_count
        ),
        "complete" if solves == config.solve_budget else "incomplete",
        selected_checks,
        maximum_error,
        geometry_seconds,
    )
