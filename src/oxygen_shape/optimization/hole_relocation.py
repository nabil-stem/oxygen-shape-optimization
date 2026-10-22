"""Paid greedy relocation of a fixed number of identical square islands."""

import hashlib
import json
from dataclasses import asdict

import numpy as np

from oxygen_shape.geometry.constraints import raster_perimeter
from oxygen_shape.geometry.square_holes import perforate
from oxygen_shape.optimization.trajectory_search import _json


def run_hole_relocation(config, outer, initial_holes, positions, method, seed, evidence, key):
    """Replay a frozen seed from checked fields on resume; every solve is charged.

    The flux heuristic uses current concentration squared only for proposals.
    It is not a topological derivative. Acceptance always uses solved global Q.
    """
    if (
        method not in ("uniform_greedy", "concentration_greedy")
        or not positions
        or not initial_holes
    ):
        raise ValueError("Declare a supported method and nonempty square positions")
    kwargs = dict(
        gap=config.gap,
        protected_length=config.protected_length,
        minimum_feature_width=config.minimum_feature_width,
    )
    rng = np.random.default_rng(seed)
    holes = tuple(initial_holes)
    geometry = perforate(outer, holes, **kwargs)
    reference_area, reference_perimeter = geometry.area, raster_perimeter(geometry)
    result = evidence.get(f"{key}/s0000", geometry, config.search_Bi)
    initial_Q = result.metrics.physical_flux_Q
    history_path = evidence.root / f"{key}/history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(history_path.read_text()) if history_path.exists() else []
    history = []
    seen = {hashlib.sha256(geometry.inside.tobytes()).hexdigest()}
    raw, rejections = 0, []

    def commit(row):
        if len(history) < len(previous) and row != previous[len(history)]:
            raise ValueError("Relocation journal does not reproduce its committed prefix")
        history.append(row)
        _json(history_path, history)

    commit(
        dict(
            step=0,
            Q=initial_Q,
            candidate_Q=initial_Q,
            accepted=True,
            raw_draws=0,
            holes=[asdict(h) for h in holes],
        )
    )
    while len(history) < config.search_budget and raw < config.search_raw_budget:
        weights = np.ones(len(positions))
        if method == "concentration_greedy":
            weights = np.array(
                [
                    np.nan_to_num(
                        result.concentration[h.row : h.row + h.width, h.col : h.col + h.width],
                        nan=0.0,
                    ).mean()
                    ** 2
                    for h in positions
                ]
            )
            if weights.sum() > 0:
                weights /= weights.sum()
                weights = config.exploration / len(positions) + (1 - config.exploration) * weights
            else:
                weights.fill(1 / len(positions))
        else:
            weights /= weights.sum()
        index = int(rng.integers(len(holes)))
        target = positions[int(rng.choice(len(positions), p=weights))]
        candidate_holes = (*holes[:index], target, *holes[index + 1 :])
        raw += 1
        try:
            candidate = perforate(outer, candidate_holes, **kwargs)
            digest = hashlib.sha256(candidate.inside.tobytes()).hexdigest()
            if digest in seen:
                raise ValueError("Previously evaluated mask")
        except ValueError as exc:
            rejections.append(
                dict(raw_draw=raw, square=index, target=asdict(target), reason=str(exc))
            )
            continue
        if (
            abs(candidate.area - reference_area) > 1e-12
            or abs(raster_perimeter(candidate) - reference_perimeter) > 1e-12
        ):
            raise ArithmeticError("Relocation changed the fixed area or membrane length")
        evaluated = evidence.get(f"{key}/s{len(history):04d}", candidate, config.search_Bi)
        seen.add(digest)
        accepted = evaluated.metrics.physical_flux_Q > (
            result.metrics.physical_flux_Q + config.min_improvement
        )
        if accepted:
            holes, geometry, result = candidate_holes, candidate, evaluated
        commit(
            dict(
                step=len(history),
                Q=result.metrics.physical_flux_Q,
                candidate_Q=evaluated.metrics.physical_flux_Q,
                accepted=accepted,
                raw_draws=raw,
                holes=[asdict(h) for h in holes],
                candidate_holes=[asdict(h) for h in candidate_holes],
                mask_sha256=digest,
            )
        )
    if len(history) < len(previous):
        raise ValueError("Relocation replay lost committed evaluations")
    _json(evidence.root / f"{key}/rejections.json", rejections)
    summary = dict(
        method=method,
        seed=seed,
        initial_Q=initial_Q,
        final_Q=result.metrics.physical_flux_Q,
        area=geometry.area,
        robin_perimeter=raster_perimeter(geometry),
        raw_draws=raw,
        number_of_pde_solves=len(history),
        status="complete" if len(history) == config.search_budget else "raw_budget_exhausted",
    )
    _json(evidence.root / f"{key}/summary.json", summary)
    return holes, summary
