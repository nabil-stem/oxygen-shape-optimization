"""Long-horizon, restartable PDE search with separate current and best states."""

import hashlib
import json
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Literal
from uuid import uuid4

import numpy as np
from pydantic import Field

from oxygen_shape.experiments.learning_diagnostics import incumbent_solution
from oxygen_shape.experiments.reporting import write_csv
from oxygen_shape.experiments.study_worker import replace_with_retry
from oxygen_shape.geometry.disk_edits import prepare_boundary
from oxygen_shape.optimization.candidate_policy import FEATURE_NAMES
from oxygen_shape.optimization.candidate_rollout import candidate_features
from oxygen_shape.optimization.donor_sampling import eligible_wall_flux
from oxygen_shape.optimization.feasible_pool import FeasiblePoolConfig
from oxygen_shape.optimization.guided_search import validate_candidate
from oxygen_shape.optimization.trajectory_pool import build_trajectory_pool
from oxygen_shape.optimization.trajectory_ranker import (
    RANK_FEATURE_NAMES,
    ResponseRanker,
    append_access_features,
)
from oxygen_shape.solvers.finite_volume import solve

METHODS = ("flux_greedy", "flux_anneal", "learned_anneal", "learned_frozen_anneal")


class TrajectoryConfig(FeasiblePoolConfig):
    solve_budget: int = Field(default=1025, ge=2, strict=True)
    raw_draw_budget: int = Field(default=500000, ge=1, strict=True)
    pool_size: int = Field(default=4, ge=2, strict=True)
    refill_draw_limit: int = Field(default=96, ge=4, strict=True)
    perimeter_limit: float = Field(default=8.0, gt=0)
    radii: tuple[float, float] = (0.03, 0.075)
    exploration: float = Field(default=0.20, ge=0, le=1)
    uncertainty: float = Field(default=0.03, ge=0)
    temperature_start: float = Field(default=0.003, gt=0)
    temperature_end: float = Field(default=0.00002, gt=0)
    restart_interval: int = Field(default=128, ge=0, strict=True)
    # Optional hard membrane matching, relative to each run's starting geometry.
    perimeter_band: float | None = Field(default=None, ge=0)
    cell_move_probability: float = Field(default=0.25, ge=0, le=1)


def _atomic(path: Path, content: bytes):
    # Short sibling names avoid MAX_PATH failures in long Windows workspaces.
    temporary = path.with_name(f".{uuid4().hex[:12]}.tmp")
    temporary.write_bytes(content)
    replace_with_retry(temporary, path)


def _json(path: Path, value):
    _atomic(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def annealing_probability(delta: float, reference_Q: float, temperature: float) -> float:
    if reference_Q <= 0 or temperature <= 0:
        raise ValueError("Positive reference flux and temperature are required")
    return 1.0 if delta >= 0 else float(np.exp(max(-745.0, delta / reference_Q / temperature)))


def run_trajectory(
    config: TrajectoryConfig,
    initial,
    method: Literal["flux_greedy", "flux_anneal", "learned_anneal", "learned_frozen_anneal"],
    directory: Path,
    *,
    on_commit=None,
    warm_start: Path | None = None,
):
    """Every committed solve stores full resume state and the evaluated candidate.

    Selection explores only admissible candidates. Annealing is heuristic (the
    proposal is asymmetric), not a claim of equilibrium Boltzmann sampling.
    """
    if method not in METHODS:
        raise ValueError("Unknown trajectory method")
    started = perf_counter()
    reason = validate_candidate(initial, initial, config)
    if reason:
        raise ValueError(f"Invalid trajectory start: {reason}")
    identity_data = {
        "config": config.model_dump(mode="json"),
        "method": method,
        "mask": hashlib.sha256(initial.inside.tobytes()).hexdigest(),
        "warm_start": hashlib.sha256(warm_start.read_bytes()).hexdigest() if warm_start else None,
    }
    identity = hashlib.sha256(json.dumps(identity_data, sort_keys=True).encode()).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    states = directory / "states"
    states.mkdir(exist_ok=True)
    recorded = sorted(states.glob("state_*.npz"))
    proposal_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 0]))
    selection_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 1]))
    ranker = ResponseRanker()
    if warm_start:
        with np.load(warm_start, allow_pickle=False) as warm:
            if tuple(warm["feature_names"]) != RANK_FEATURE_NAMES:
                raise ValueError("Warm-start feature schema mismatch")
            ranker.weights = warm["weights"].copy()
            ranker.inverse = warm["inverse"].copy()
            ranker.updates = int(warm["updates"])
            if (
                ranker.weights.shape != (len(RANK_FEATURE_NAMES),)
                or ranker.inverse.shape != (len(RANK_FEATURE_NAMES),) * 2
                or not np.isfinite(ranker.weights).all()
                or not np.isfinite(ranker.inverse).all()
            ):
                raise ValueError("Invalid warm-start parameter arrays")
    warm_updates = ranker.updates
    history = []
    if recorded:
        for file in recorded:
            with np.load(file, allow_pickle=False) as state:
                meta = json.loads(str(state["metadata"]))
                if meta["identity"] != identity or meta["row"]["pde_solves"] != len(history) + 1:
                    raise ValueError("Resume identity or journal sequence mismatch")
                history.append(meta["row"])
        completed = directory / "summary.json"
        if completed.exists() and json.loads(completed.read_text())["status"] == "complete":
            return json.loads(completed.read_text())
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
            ranker.weights = state["weights"].copy()
            ranker.inverse = state["inverse"].copy()
            ranker.updates = meta["updates"]
            proposal_rng.bit_generator.state = meta["proposal_rng"]
            selection_rng.bit_generator.state = meta["selection_rng"]
            reference_Q = meta["reference_Q"]
            raw, checks = meta["raw"], meta["checks"]
            previous_seconds = meta["seconds"]
        # A pending marker at/before the latest commit is already accounted for.
        # Later markers preserve an upper bound on uncommitted PDE work.
        pending = directory / "pending.json"
        if pending.exists():
            attempt = json.loads(pending.read_text())
            if attempt["pde_solves"] > len(history):
                _json(
                    directory / f"interrupted_{uuid4().hex[:8]}.json",
                    {**attempt, "extra_pde_lower_bound": 0, "extra_pde_upper_bound": 1},
                )
            pending.unlink()
    else:
        geometry = best = initial
        if (directory / "pending.json").exists():
            pending = json.loads((directory / "pending.json").read_text())
            _json(
                directory / f"interrupted_{uuid4().hex[:8]}.json",
                {**pending, "extra_pde_lower_bound": 0, "extra_pde_upper_bound": 1},
            )
        _json(directory / "pending.json", {"pde_solves": 1, "phase": "initial"})
        solution = best_solution = solve(initial, config.model, config.solver)
        reference_Q = solution.metrics.physical_flux_Q
        raw = checks = 0
        previous_seconds = 0.0
        _json(directory / "identity.json", {"identity": identity, **identity_data})
        np.save(directory / "initial_geometry.npy", initial.inside)

    def metrics_dict(result):
        return (
            asdict(result.metrics)
            if hasattr(result.metrics, "__dataclass_fields__")
            else vars(result.metrics)
        )

    def commit(row, candidate_geometry, candidate_solution, features):
        history.append(row)
        meta = {
            "identity": identity,
            "row": row,
            "raw": raw,
            "checks": checks,
            "metrics": metrics_dict(solution),
            "best_metrics": metrics_dict(best_solution),
            "reference_Q": reference_Q,
            "updates": ranker.updates,
            "proposal_rng": proposal_rng.bit_generator.state,
            "selection_rng": selection_rng.bit_generator.state,
            "seconds": previous_seconds + perf_counter() - started,
        }
        buffer = BytesIO()
        np.savez_compressed(
            buffer,
            metadata=np.array(json.dumps(meta, allow_nan=False)),
            current_mask=geometry.inside,
            current_field=solution.concentration,
            best_mask=best.inside,
            best_field=best_solution.concentration,
            candidate_mask=candidate_geometry.inside,
            candidate_field=candidate_solution.concentration,
            selected_features=features,
            weights=ranker.weights,
            inverse=ranker.inverse,
        )
        _atomic(states / f"state_{len(history):06d}.npz", buffer.getvalue())
        (directory / "pending.json").unlink(missing_ok=True)
        if on_commit:
            on_commit(row)

    if not history:
        commit(
            {
                "pde_solves": 1,
                "raw_draws": 0,
                "checks": 0,
                "current_Q": reference_Q,
                "best_Q": reference_Q,
                "candidate_Q": reference_Q,
                "accepted": False,
                "downhill": False,
                "best_perimeter": solution.metrics.robin_perimeter,
                "seconds": 0.0,
            },
            initial,
            solution,
            np.empty(0),
        )
    initial_perimeter = history[0]["best_perimeter"]
    seen = set()
    if recorded:
        for file in reversed(recorded):
            with np.load(file, allow_pickle=False) as state:
                row = json.loads(str(state["metadata"]))["row"]
                if row["accepted"] or row["pde_solves"] == 1:
                    break
                seen.add(state["candidate_mask"].tobytes())
                if row["restart"]:
                    break
    while len(history) < config.solve_budget and raw < config.raw_draw_budget:
        # Restarts use a saved solution and spend no hidden PDE evaluation.
        restart = bool(config.restart_interval and len(history) % config.restart_interval == 0)
        if restart:
            geometry, solution = best, best_solution
            seen.clear()
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
            initial_perimeter=initial_perimeter,
        )
        raw += len(built.edits)
        checks += built.full_checks
        eligible = built.eligible.copy()
        if config.perimeter_band is not None:
            eligible &= np.array(
                [
                    r["perimeter"] is not None
                    and abs(r["perimeter"] - initial_perimeter) <= config.perimeter_band + 1e-12
                    for r in built.rows
                ]
            )
        if not eligible.any():
            continue
        flux = eligible_wall_flux(geometry, solution, config.protected_length)
        base = candidate_features(
            geometry,
            solution,
            config,
            reference_Q,
            len(history),
            raw,
            prepared,
            built.edits,
            built.rows,
            built.rows,
            flux,
        )
        features = append_access_features(
            base, geometry, prepared, built.rows, config.protected_length
        )
        indices = np.flatnonzero(eligible)
        scores = (
            ranker.scores(features, config.uncertainty)
            if method in ("learned_anneal", "learned_frozen_anneal")
            else base[:, FEATURE_NAMES.index("flux_product")]
        )
        exploring = bool(selection_rng.random() < config.exploration)
        tied = indices[np.isclose(scores[indices], scores[indices].max(), rtol=0, atol=1e-12)]
        slot = int(selection_rng.choice(indices if exploring else tied))
        edit = built.edits[slot]
        before_Q = solution.metrics.physical_flux_Q
        pending = {
            "pde_solves": len(history) + 1,
            "raw_draws": raw,
            "checks": checks,
            "proposal": built.rows[slot],
        }
        _json(directory / "pending.json", pending)
        candidate = solve(edit.geometry, config.model, config.solver)
        delta = candidate.metrics.physical_flux_Q - before_Q
        fraction = (len(history) - 1) / max(config.solve_budget - 2, 1)
        temperature = (
            config.temperature_start
            * (config.temperature_end / config.temperature_start) ** fraction
        )
        probability = (
            float(delta > config.min_improvement)
            if method == "flux_greedy"
            else annealing_probability(delta, reference_Q, temperature)
        )
        accepted = bool(selection_rng.random() < probability)
        target = 100 * delta / reference_Q
        prediction = float(ranker.scores(features[slot : slot + 1])[0])
        if method == "learned_anneal":
            ranker.update(features[slot], target)
        seen.add(edit.geometry.inside.tobytes())
        if accepted:
            geometry, solution = edit.geometry, candidate
            seen.clear()
        if candidate.metrics.physical_flux_Q > best_solution.metrics.physical_flux_Q:
            best, best_solution = edit.geometry, candidate
        row = {
            "pde_solves": len(history) + 1,
            "raw_draws": raw,
            "checks": checks,
            "current_Q": solution.metrics.physical_flux_Q,
            "best_Q": best_solution.metrics.physical_flux_Q,
            "candidate_Q": candidate.metrics.physical_flux_Q,
            "delta_Q": delta,
            "accepted": accepted,
            "downhill": bool(accepted and delta < 0),
            "best_perimeter": best_solution.metrics.robin_perimeter,
            "candidate_perimeter": candidate.metrics.robin_perimeter,
            "conservation_error": candidate.metrics.conservation_error,
            "temperature": temperature,
            "acceptance_probability": probability,
            "exploring": exploring,
            "prediction": prediction,
            "target": target,
            "restart": restart,
            "eligible": int(eligible.sum()),
            "seconds": previous_seconds + perf_counter() - started,
            **built.rows[slot],
        }
        commit(row, edit.geometry, candidate, features[slot])
    summary = {
        "status": "complete" if len(history) == config.solve_budget else "raw_budget_exhausted",
        "method": method,
        "seed": config.seed,
        "pde_solves": len(history),
        "raw_draws": raw,
        "checks": checks,
        "initial_Q": reference_Q,
        "best_Q": best_solution.metrics.physical_flux_Q,
        "current_Q": solution.metrics.physical_flux_Q,
        "initial_perimeter": initial_perimeter,
        "best_perimeter": best_solution.metrics.robin_perimeter,
        "best_eta": best_solution.metrics.efficiency_eta,
        "area": best.area,
        "model_updates": ranker.updates,
        "new_model_updates": ranker.updates - warm_updates,
        "downhill_moves": sum(r["downhill"] for r in history),
        "wall_seconds": previous_seconds + perf_counter() - started,
    }
    np.save(directory / "final_geometry.npy", best.inside)
    np.save(directory / "concentration.npy", best_solution.concentration)
    model = BytesIO()
    np.savez_compressed(
        model,
        weights=ranker.weights,
        inverse=ranker.inverse,
        updates=np.array(ranker.updates),
        feature_names=np.array(RANK_FEATURE_NAMES),
    )
    _atomic(directory / "response_model.npz", model.getvalue())
    write_csv(directory / "metrics.csv", history)
    _json(directory / "summary.json", summary)
    return summary
