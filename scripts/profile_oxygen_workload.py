"""Profile actual CPU phases and record GPU availability; no GPU dependency needed."""

import argparse
import cProfile
import importlib.util
import json
import pstats
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np

from oxygen_shape.experiments.config import load_config
from oxygen_shape.experiments.feasible_pool_study import PoolStudyConfig
from oxygen_shape.experiments.reporting import (
    create_run,
    provenance,
    save_state,
    save_summary,
    write_csv,
)
from oxygen_shape.geometry.baselines import matched_area
from oxygen_shape.geometry.disk_edits import prepare_boundary
from oxygen_shape.geometry.raster import refine_geometry
from oxygen_shape.optimization.candidate_policy import FEATURE_COUNT, CandidatePolicy
from oxygen_shape.optimization.guided_search import validate_candidate
from oxygen_shape.optimization.perimeter_pool import cheap_screen, draw_pool
from oxygen_shape.solvers.finite_volume import solve


def profile(root):
    config = load_config(root / "config.yaml", PoolStudyConfig).model_copy(update={"seed": 190000})
    output = create_run(config, "workload_profile")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    hardware = {
        "nvidia_smi_exit_code": gpu.returncode,
        "gpu": gpu.stdout.strip(),
        "libraries": {
            name: importlib.util.find_spec(name) is not None for name in ("torch", "cupy", "jax")
        },
        "gpu_used": False,
    }
    (output / "hardware.json").write_text(json.dumps(hardware, indent=2) + "\n", encoding="utf-8")
    initial = matched_area("deep_cavity", config.n, config.target_area).geometry.with_mask(
        np.load(root / "bank/improved_deep_0.npy", allow_pickle=False)
    )
    assert not validate_candidate(initial, initial, config)
    prepared = prepare_boundary(initial, config.protected_length)
    started = perf_counter()
    edits, rows = draw_pool(initial, np.random.default_rng(config.seed), config, prepared, 256)
    generation_seconds = perf_counter() - started
    started = perf_counter()
    screens = [cheap_screen(edit, initial, config) for edit in edits]
    cheap_seconds = perf_counter() - started
    started = perf_counter()
    full_checks = 0
    for edit, row, (reason, perimeter) in zip(edits, rows, screens, strict=True):
        if not reason:
            reason = validate_candidate(edit.geometry, initial, config)
            full_checks += 1
        row.update(rejection=reason, robin_perimeter=perimeter)
    full_seconds = perf_counter() - started
    write_csv(output / "geometry_proposals.csv", rows)
    features = np.random.default_rng(config.seed).normal(size=(256, FEATURE_COUNT))
    policy = CandidatePolicy()
    policy.scores(features)
    started = perf_counter()
    for _ in range(2000):
        policy.scores(features)
    policy_seconds = (perf_counter() - started) / 2000
    numerical = []
    for factor in (1, 2, 4):
        geometry = initial if factor == 1 else refine_geometry(initial, factor)
        profiler = cProfile.Profile()
        started = perf_counter()
        result = profiler.runcall(solve, geometry, config.model, config.solver)
        elapsed = perf_counter() - started
        profiler.dump_stats(str(output / f"solver_n{config.n * factor}.prof"))
        stats = pstats.Stats(profiler).stats
        phases = {
            name: sum(value[3] for key, value in stats.items() if key[2] == name)
            for name in ("assemble", "spsolve", "extract_boundary", "validate_geometry")
        }
        path = output / f"n{config.n * factor}"
        path.mkdir()
        save_state(path, geometry, result)
        numerical.append(
            {
                "n": config.n * factor,
                "cells": int(geometry.inside.sum()),
                "solve_seconds_with_profiler": elapsed,
                **{f"{name}_cumulative_seconds": value for name, value in phases.items()},
                "Q": result.metrics.physical_flux_Q,
                "eta": result.metrics.efficiency_eta,
                "conservation_error": result.metrics.conservation_error,
            }
        )
    write_csv(output / "metrics.csv", numerical)
    summary = {
        **provenance(),
        "seed": config.seed,
        "grid_shape": [config.n, config.n],
        "Bi": config.model.Bi,
        "solver": "finite_volume",
        "solver_tolerance": None,
        "diagnostic_tolerances": config.solver.model_dump(),
        "status": "complete",
        "number_of_pde_solves": 3,
        "max_conservation_error": max(row["conservation_error"] for row in numerical),
        "hardware": hardware,
        "raw_geometry_proposals": len(edits),
        "proposal_generation_seconds": generation_seconds,
        "cheap_screen_seconds": cheap_seconds,
        "full_geometry_checks": full_checks,
        "full_geometry_validation_seconds": full_seconds,
        "fully_valid_proposals": sum(not row["rejection"] for row in rows),
        "linear_policy_seconds_per_256_candidate_call": policy_seconds,
        "policy_features": FEATURE_COUNT,
        "profiling_limitations": (
            "One representative state; profiled solves include cProfile overhead; "
            "nested cumulative times overlap; no measured GPU speed comparison"
        ),
        "training_pde_solves": 0,
    }
    save_summary(output, summary)
    print(
        json.dumps(
            {
                "output": str(output),
                "hardware": hardware,
                "geometry_generation_seconds": generation_seconds,
                "geometry_validation_seconds": full_seconds,
                "policy_microseconds": policy_seconds * 1e6,
                "solver_phases": numerical,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix_directory", type=Path)
    profile(parser.parse_args().matrix_directory.resolve())
