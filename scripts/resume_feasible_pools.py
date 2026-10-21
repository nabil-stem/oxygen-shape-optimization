"""Recover an interrupted matrix using its unchanged, snapshotted algorithms.

Completed job artifacts are immutable. Incomplete attempts are retained under
interrupted_attempts, including conservative accounting for an in-flight solve.
This script does not change the frozen sampling or comparison protocol.
"""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from oxygen_shape.experiments.config import load_config
from oxygen_shape.experiments.feasible_pool_study import PoolStudyConfig, write_pool_report
from oxygen_shape.experiments.optimization import save_search, search_summary
from oxygen_shape.experiments.pioneer_training import assert_source_unchanged
from oxygen_shape.experiments.reporting import save_summary, write_csv
from oxygen_shape.experiments.study_worker import atomic_json
from oxygen_shape.geometry.baselines import matched_area
from oxygen_shape.optimization.feasible_pool import run_feasible_search
from oxygen_shape.optimization.guided_search import validate_candidate
from oxygen_shape.optimization.refinement import validate_refinement


def resume(root):
    root = root.resolve()
    assert_source_unchanged(root)
    config = load_config(root / "config.yaml", PoolStudyConfig)
    protocol = json.loads((root / "protocol.json").read_text())
    metadata = json.loads((root / "provenance.json").read_text())
    manifest = json.loads((root / "bank/manifest.json").read_text())
    starts = []
    for item in manifest:
        path = root / "bank" / item["mask_file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["mask_sha256"]
        canonical = matched_area(item["family"], config.n, config.target_area).geometry
        geometry = canonical.with_mask(np.load(path, allow_pickle=False))
        assert not validate_candidate(geometry, canonical, config)
        starts.append((item["name"], item["family"], geometry))
    rows = pd.read_csv(root / "evaluations.csv").to_dict("records")
    known = {(r["start"], int(r["seed"]), r["method"]) for r in rows}
    assert len(known) == len(rows)
    recovery_file = root / "recovery.json"
    if recovery_file.exists():
        recovery = json.loads(recovery_file.read_text())
    else:
        preserved = {}
        for name, seed, method in sorted(known):
            for file in ("summary.json", "final_geometry.npy"):
                path = root / "runs" / f"{name}__{seed}__{method}" / file
                preserved[path.relative_to(root).as_posix()] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
        recovery = {
            "completed_before_recovery": len(rows),
            "preserved_completed_files": preserved,
            "interrupted_attempts": [],
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "status": "running",
        }
    atomic_json(recovery_file, recovery)
    new_started = perf_counter()
    methods = [(q, r) for q in (0, 2, 4) for r in ("uniform", "flux")]
    for name, family, geometry in starts:
        for seed in config.evaluation_seeds:
            ordered = methods if seed % 2 == 0 else list(reversed(methods))
            for quota, ranking in ordered:
                method = f"{'fixed8' if quota == 0 else f'refill{quota}'}_{ranking}"
                key = (name, seed, method)
                if key in known:
                    continue
                path = root / "runs" / f"{name}__{seed}__{method}"
                if (path / "summary.json").exists():
                    row = json.loads((path / "summary.json").read_text())
                    assert (row["start"], row["seed"], row["method"]) == key
                else:
                    if path.exists():
                        destination = (
                            root
                            / "interrupted_attempts"
                            / f"{path.name}__{len(recovery['interrupted_attempts']):03d}"
                        )
                        # Verify both resolved absolute paths remain within this run.
                        assert path.resolve().is_relative_to(root) and path.resolve() != root
                        assert (
                            destination.resolve().is_relative_to(root) and not destination.exists()
                        )
                        candidates = list((path / "candidates").glob("*.npz"))
                        fields = list(path.glob("refinement_*/**/concentration.npy"))
                        initial_completed = int(
                            bool(candidates) or (path / "initial_concentration.npy").exists()
                        )
                        lower = initial_completed + len(candidates) + len(fields)
                        recovery["interrupted_attempts"].append(
                            {
                                "job": path.name,
                                "archive": destination.relative_to(root).as_posix(),
                                "pde_solves_lower_bound": lower,
                                "pde_solves_upper_bound": lower + 1,
                                "reason": (
                                    "Execution interrupted; at most one completed/in-flight "
                                    "solve lacked a stored field"
                                ),
                            }
                        )
                        destination.parent.mkdir(exist_ok=True)
                        path.rename(destination)
                        atomic_json(recovery_file, recovery)
                    cfg = config.model_copy(update={"seed": seed, "initial_shape": family})
                    (path / "candidates").mkdir(parents=True)
                    (path / "config.yaml").write_text(
                        json.dumps(
                            {
                                "configuration": cfg.model_dump(mode="json"),
                                "quota": quota,
                                "ranking": ranking,
                            },
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )

                    def save_candidate(index, g, solution, accepted, directory=path):
                        np.savez_compressed(
                            directory / "candidates" / f"solve_{index:03d}.npz",
                            mask=g.inside,
                            concentration=solution.concentration,
                            accepted=accepted,
                        )

                    started = perf_counter()
                    result = run_feasible_search(
                        cfg, geometry, quota=quota, ranking=ranking, on_evaluation=save_candidate
                    )
                    save_search(path, result.search)
                    write_csv(path / "pools.csv", result.pools)
                    refinements = validate_refinement(cfg, result.search, path)
                    row = {
                        **search_summary(cfg, result.search, refinements),
                        "git_commit": metadata["git_commit"],
                        "source_manifest": "../../source_manifest.json",
                        "start": name,
                        "family": family,
                        "method": method,
                        "status": result.status,
                        "pools": len(result.pools),
                        "multi_choice_fraction": float(
                            np.mean([p["eligible"] >= 2 for p in result.pools])
                        ),
                        "raw_valid_fraction": sum(p["eligible"] for p in result.pools)
                        / result.search.attempts,
                        "selected_invalid_fraction": float(
                            np.mean([not p["selected_valid"] for p in result.pools])
                        ),
                        "quota_reached_fraction": float(
                            np.mean([p["quota_reached"] for p in result.pools])
                        ),
                        "full_geometry_checks": result.full_checks,
                        "geometry_validation_seconds": result.validation_seconds,
                        "environment_seconds": result.seconds,
                        "workflow_seconds": perf_counter() - started,
                    }
                    save_summary(path, row)
                rows.append(row)
                known.add(key)
                write_csv(root / "evaluations.csv", rows)
                message = (
                    f"{len(rows)}/{protocol['expected_runs']} {name} {seed} {method}: "
                    f"{row['status']}, fine gain={row['fine_gain_Q']:.6g}"
                )
                print(message, flush=True)
                with (root / "run.log").open("a", encoding="utf-8") as stream:
                    stream.write("RECOVERED " + message + "\n")
    assert len(rows) == protocol["expected_runs"]
    assert_source_unchanged(root)
    for file, expected in recovery["preserved_completed_files"].items():
        assert hashlib.sha256((root / file).read_bytes()).hexdigest() == expected
    recovery["status"] = "complete"
    recovery["completed_files_unchanged"] = True
    recovery["successful_recovery_invocation_seconds"] = perf_counter() - new_started
    atomic_json(recovery_file, recovery)
    write_csv(root / "metrics.csv", rows)
    write_pool_report(root, rows)
    summary = {
        **metadata,
        "seed": config.seed,
        "grid_shape": [config.n, config.n],
        "Bi": config.model.Bi,
        "solver": "finite_volume",
        "solver_tolerance": None,
        "diagnostic_tolerances": config.solver.model_dump(),
        "status": "complete"
        if all(r["status"] == "complete" for r in rows)
        else "complete_with_budget_exhaustions",
        "runs": len(rows),
        "completed_budgets": sum(r["status"] == "complete" for r in rows),
        "number_of_pde_solves": sum(
            r["number_of_pde_solves"] + r["refinement_pde_solves"] for r in rows
        ),
        "max_conservation_error": max(
            float(
                pd.read_csv(
                    root / "runs" / f"{r['start']}__{r['seed']}__{r['method']}" / "refinement.csv"
                ).conservation_error.max()
            )
            for r in rows
        ),
        "wall_clock_seconds": sum(r["workflow_seconds"] for r in rows),
        "wall_clock_definition": (
            "Sum of completed workflow durations; interrupted work/setup excluded "
            "and separately documented"
        ),
        "max_conservation_error_scope": (
            "Saved refinement table entries; all candidate fields separately checked by audit"
        ),
        "interrupted_pde_solves_lower_bound": sum(
            r["pde_solves_lower_bound"] for r in recovery["interrupted_attempts"]
        ),
        "interrupted_pde_solves_upper_bound": sum(
            r["pde_solves_upper_bound"] for r in recovery["interrupted_attempts"]
        ),
        "training_pde_solves": 0,
        "policy_promoted": False,
    }
    save_summary(root, summary)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "status",
                    "runs",
                    "number_of_pde_solves",
                    "interrupted_pde_solves_lower_bound",
                    "interrupted_pde_solves_upper_bound",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    resume(parser.parse_args().run_directory)
