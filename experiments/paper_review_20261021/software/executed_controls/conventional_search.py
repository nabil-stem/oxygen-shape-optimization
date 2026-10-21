"""Charged uniform and fully evaluated-pool controls for the frozen raster problem."""

import hashlib
import json
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Literal
from uuid import uuid4

import numpy as np

from oxygen_shape.experiments.learning_diagnostics import incumbent_solution
from oxygen_shape.experiments.reporting import write_csv
from oxygen_shape.geometry.disk_edits import prepare_boundary
from oxygen_shape.optimization.guided_search import validate_candidate
from oxygen_shape.optimization.trajectory_pool import build_trajectory_pool
from oxygen_shape.optimization.trajectory_search import TrajectoryConfig, _atomic, _json
from oxygen_shape.solvers.finite_volume import solve

ClassicalMethod = Literal["uniform_greedy", "pool_greedy"]


def run_classical(
    config: TrajectoryConfig,
    initial,
    method: ClassicalMethod,
    directory: Path,
    *,
    on_commit=None,
):
    """Every candidate is charged; pending batch masks survive exact resumption.

    Pool greedy draws one fixed admissible pool from the current geometry and
    evaluates its eligible members in randomized order. A newly observed best
    may be accepted immediately, while the remaining proposals still belong to
    that original batch. The final current geometry is its best improving member.
    This is conventional best-of-pool greedy search, not an uncharged oracle.
    """
    if method not in ("uniform_greedy", "pool_greedy"):
        raise ValueError("Unknown conventional control")
    reason = validate_candidate(initial, initial, config)
    if reason:
        raise ValueError(reason)
    directory.mkdir(parents=True, exist_ok=True)
    states = directory / "states"
    states.mkdir(exist_ok=True)
    identity_data = dict(
        config=config.model_dump(mode="json"),
        method=method,
        geometry=hashlib.sha256(
            initial.inside.tobytes()
            + repr((initial.h, initial.source_faces, initial.neumann_faces)).encode()
        ).hexdigest(),
        implementation=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    identity = hashlib.sha256(json.dumps(identity_data, sort_keys=True).encode()).hexdigest()
    proposal_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 0]))
    selection_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 1]))
    started = perf_counter()
    history, pending_masks, pending_rows = [], [], []
    recorded = sorted(states.glob("state_*.npz"))
    previous_seconds, batch_id, batch_origin_solve, batch_base_Q = 0.0, 0, 1, None
    batch_available, batch_scheduled = 0, 0
    pending_marker = directory / "pending.json"
    if recorded:
        for file in recorded:
            with np.load(file, allow_pickle=False) as state:
                meta = json.loads(str(state["metadata"]))
                if meta["identity"] != identity or meta["row"]["pde_solves"] != len(history) + 1:
                    raise ValueError("Conventional resume identity or sequence mismatch")
                history.append(meta["row"])
        summary_file = directory / "summary.json"
        if summary_file.exists():
            return json.loads(summary_file.read_text())
        with np.load(recorded[-1], allow_pickle=False) as state:
            meta = json.loads(str(state["metadata"]))
            geometry = initial.with_mask(state["current_mask"])
            best = initial.with_mask(state["best_mask"])
            solution = incumbent_solution(
                geometry, state["current_field"], meta["metrics"], config.model.Bi
            )
            best_solution = incumbent_solution(
                best, state["best_field"], meta["best_metrics"], config.model.Bi
            )
            reference_Q, raw, checks = meta["reference_Q"], meta["raw"], meta["checks"]
            pending_masks = [mask.copy() for mask in state["pending_masks"]]
            pending_rows = meta["pending_rows"]
            proposal_rng.bit_generator.state = meta["proposal_rng"]
            selection_rng.bit_generator.state = meta["selection_rng"]
            previous_seconds = meta["seconds"]
            batch_id, batch_origin_solve = meta["batch_id"], meta["batch_origin_solve"]
            batch_base_Q = meta["batch_base_Q"]
            batch_available, batch_scheduled = meta["batch_available"], meta["batch_scheduled"]
    if pending_marker.exists():
        attempt = json.loads(pending_marker.read_text())
        if attempt["pde_solves"] > len(history):
            _json(
                directory / f"interrupted_{uuid4().hex[:8]}.json",
                {
                    **attempt,
                    "extra_pde_lower_bound": 0,
                    "extra_pde_upper_bound": 1,
                },
            )
        pending_marker.unlink()
    if not recorded:
        _json(directory / "identity.json", {"identity": identity, **identity_data})
        np.save(directory / "initial_geometry.npy", initial.inside)
        _json(pending_marker, {"pde_solves": 1, "phase": "initial"})
        geometry = best = initial
        solution = best_solution = solve(initial, config.model, config.solver)
        reference_Q = solution.metrics.physical_flux_Q
        batch_base_Q = reference_Q
        raw = checks = 0

    def metric_dict(result):
        return (
            asdict(result.metrics)
            if hasattr(result.metrics, "__dataclass_fields__")
            else vars(result.metrics)
        )

    def commit(row, candidate_geometry, candidate_solution):
        history.append(row)
        metadata = dict(
            identity=identity,
            row=row,
            reference_Q=reference_Q,
            raw=raw,
            checks=checks,
            metrics=metric_dict(solution),
            best_metrics=metric_dict(best_solution),
            pending_rows=pending_rows,
            batch_id=batch_id,
            batch_origin_solve=batch_origin_solve,
            batch_base_Q=batch_base_Q,
            batch_available=batch_available,
            batch_scheduled=batch_scheduled,
            proposal_rng=proposal_rng.bit_generator.state,
            selection_rng=selection_rng.bit_generator.state,
            seconds=previous_seconds + perf_counter() - started,
        )
        buffer = BytesIO()
        np.savez_compressed(
            buffer,
            metadata=np.array(json.dumps(metadata, allow_nan=False)),
            current_mask=geometry.inside,
            current_field=solution.concentration,
            best_mask=best.inside,
            best_field=best_solution.concentration,
            candidate_mask=candidate_geometry.inside,
            candidate_field=candidate_solution.concentration,
            pending_masks=np.stack(pending_masks)
            if pending_masks
            else np.empty((0, *initial.inside.shape), dtype=bool),
        )
        _atomic(states / f"state_{len(history):06d}.npz", buffer.getvalue())
        pending_marker.unlink(missing_ok=True)
        if on_commit:
            on_commit(row)

    if not history:
        commit(
            dict(
                pde_solves=1,
                raw_draws=0,
                checks=0,
                current_Q=reference_Q,
                best_Q=reference_Q,
                candidate_Q=reference_Q,
                accepted=False,
                batch_id=0,
                batch_origin_solve=1,
                best_perimeter=solution.metrics.robin_perimeter,
                conservation_error=solution.metrics.conservation_error,
                seconds=0.0,
            ),
            initial,
            solution,
        )
    seen = set()
    for file in reversed(recorded):
        with np.load(file, allow_pickle=False) as state:
            row = json.loads(str(state["metadata"]))["row"]
            if row["accepted"] or row["pde_solves"] == 1:
                break
            seen.add(state["candidate_mask"].tobytes())
    while len(history) < config.solve_budget:
        if not pending_masks:
            if raw >= config.raw_draw_budget:
                break
            prepared = prepare_boundary(geometry, config.protected_length)
            built = build_trajectory_pool(
                config,
                geometry,
                initial,
                prepared,
                proposal_rng,
                config.raw_draw_budget - raw,
                seen,
                quota=2,
                initial_perimeter=history[0]["best_perimeter"],
            )
            raw += len(built.edits)
            checks += built.full_checks
            eligible = np.flatnonzero(built.eligible)
            if not len(eligible):
                continue
            selected = (
                [int(selection_rng.choice(eligible))]
                if method == "uniform_greedy"
                else selection_rng.permutation(eligible).tolist()
            )[: config.solve_budget - len(history)]
            pending_masks = [built.edits[slot].geometry.inside.copy() for slot in selected]
            pending_rows = [built.rows[slot] for slot in selected]
            batch_available, batch_scheduled = len(eligible), len(selected)
            batch_id += 1
            batch_origin_solve = len(history)
            batch_base_Q = solution.metrics.physical_flux_Q
        candidate_geometry = initial.with_mask(pending_masks.pop(0))
        proposal = pending_rows.pop(0)
        _json(pending_marker, {"pde_solves": len(history) + 1, "proposal": proposal})
        candidate = solve(candidate_geometry, config.model, config.solver)
        before = solution.metrics.physical_flux_Q
        q = candidate.metrics.physical_flux_Q
        accepted = bool(q > before + config.min_improvement)
        seen.add(candidate_geometry.inside.tobytes())
        if accepted:
            geometry, solution = candidate_geometry, candidate
            seen.clear()
        if q > best_solution.metrics.physical_flux_Q:
            best, best_solution = candidate_geometry, candidate
        commit(
            dict(
                pde_solves=len(history) + 1,
                raw_draws=raw,
                checks=checks,
                current_Q=solution.metrics.physical_flux_Q,
                best_Q=best_solution.metrics.physical_flux_Q,
                candidate_Q=q,
                delta_Q=q - before,
                delta_Q_from_batch=q - batch_base_Q,
                accepted=accepted,
                batch_id=batch_id,
                batch_origin_solve=batch_origin_solve,
                pending_candidates=len(pending_masks),
                batch_available_choices=batch_available,
                batch_scheduled_evaluations=batch_scheduled,
                batch_budget_truncated=method == "pool_greedy"
                and batch_scheduled < batch_available,
                candidate_perimeter=candidate.metrics.robin_perimeter,
                best_perimeter=best_solution.metrics.robin_perimeter,
                conservation_error=candidate.metrics.conservation_error,
                seconds=previous_seconds + perf_counter() - started,
                **proposal,
            ),
            candidate_geometry,
            candidate,
        )
    summary = dict(
        status="complete" if len(history) == config.solve_budget else "raw_budget_exhausted",
        method=method,
        seed=config.seed,
        pde_solves=len(history),
        raw_draws=raw,
        checks=checks,
        initial_Q=reference_Q,
        current_Q=solution.metrics.physical_flux_Q,
        best_Q=best_solution.metrics.physical_flux_Q,
        area=best.area,
        initial_perimeter=history[0]["best_perimeter"],
        best_perimeter=best_solution.metrics.robin_perimeter,
        best_eta=best_solution.metrics.efficiency_eta,
        batches=batch_id,
        accepted_moves=sum(row["accepted"] for row in history),
        wall_seconds=previous_seconds + perf_counter() - started,
    )
    np.save(directory / "final_geometry.npy", best.inside)
    np.save(directory / "concentration.npy", best_solution.concentration)
    write_csv(directory / "metrics.csv", history)
    _json(directory / "summary.json", summary)
    return summary
