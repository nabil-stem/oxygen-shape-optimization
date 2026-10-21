"""Frozen development comparison of fixed and refilled fully screened pools."""

import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from oxygen_shape.experiments.config import StrictConfig
from oxygen_shape.experiments.optimization import save_search, search_summary
from oxygen_shape.experiments.pioneer_training import assert_source_unchanged
from oxygen_shape.experiments.reporting import (
    create_run,
    log_failure,
    provenance,
    save_summary,
    write_csv,
)
from oxygen_shape.geometry.baselines import matched_area
from oxygen_shape.optimization.feasible_pool import FeasiblePoolConfig, run_feasible_search
from oxygen_shape.optimization.guided_search import validate_candidate
from oxygen_shape.optimization.refinement import validate_refinement


class PoolStart(StrictConfig):
    name: str = Field(pattern=r"^[a-z0-9_]+$")
    family: Literal["deep_cavity", "rectangle"]
    mask_file: Path
    mask_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class PoolStudyConfig(FeasiblePoolConfig):
    inputs: list[PoolStart] = Field(default_factory=list)
    include_canonical: bool = True

    @model_validator(mode="after")
    def distinct_inputs(self):
        if self.n != 100 or self.pool_size != 8 or self.refinement_factors != [2, 4]:
            raise ValueError("This study freezes n=100, base pool=8 and refinement=[2,4]")
        if not self.include_canonical and not self.inputs:
            raise ValueError("At least one starting geometry is required")
        names = [s.name for s in self.inputs]
        if len(set(names)) != len(names) or any(n.endswith("_canonical") for n in names):
            raise ValueError("Starting names must be unique and cannot shadow canonical names")
        return self

    def resolve_paths(self, base):
        resolved = super().resolve_paths(base)
        return resolved.model_copy(
            update={
                "inputs": [
                    item.model_copy(
                        update={
                            "mask_file": item.mask_file
                            if item.mask_file.is_absolute()
                            else (base / item.mask_file).resolve()
                        }
                    )
                    for item in self.inputs
                ]
            }
        )


def _starts(config, root):
    starts, manifest, seen = [], [], set()
    bank = root / "bank"
    bank.mkdir()

    def add(name, family, geometry, original=None):
        reason = validate_candidate(
            geometry, matched_area(family, config.n, config.target_area).geometry, config
        )
        signature = hashlib.sha256(geometry.inside.tobytes()).hexdigest()
        if reason or signature in seen:
            raise ValueError(f"Invalid or duplicate input {name}: {reason}")
        seen.add(signature)
        path = bank / f"{name}.npy"
        np.save(path, geometry.inside)
        starts.append((name, family, geometry))
        manifest.append(
            {
                "name": name,
                "family": family,
                "mask_file": path.name,
                "mask_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "original": original,
                "use": "development; previously observed geometries, fresh rollout seeds",
            }
        )

    if config.include_canonical:
        for family in ("deep_cavity", "rectangle"):
            add(
                f"{family}_canonical",
                family,
                matched_area(family, config.n, config.target_area).geometry,
            )
    for item in config.inputs:
        data = item.mask_file.read_bytes()
        if hashlib.sha256(data).hexdigest() != item.mask_sha256:
            raise ValueError(f"Input checksum mismatch: {item.name}")
        mask = np.load(item.mask_file, allow_pickle=False)
        if mask.dtype != bool or mask.shape != (config.n, config.n):
            raise ValueError("Invalid input raster dtype or grid")
        canonical = matched_area(item.family, config.n, config.target_area).geometry
        add(item.name, item.family, canonical.with_mask(mask), str(item.mask_file))
    (bank / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return starts


def write_pool_report(root, rows):
    frame = pd.DataFrame(rows)
    records = []
    for method, group in frame.groupby("method", sort=False):
        records.append(
            {
                "method": method,
                "runs": len(group),
                "completed": int((group.status == "complete").sum()),
                "mean_fine_gain_Q": float(group.fine_gain_Q.mean()),
                "median_workflow_seconds": float(group.workflow_seconds.median()),
                "mean_multi_choice_fraction": float(group.multi_choice_fraction.mean()),
                "mean_raw_valid_fraction": float(group.raw_valid_fraction.mean()),
                "mean_selected_invalid_fraction": float(group.selected_invalid_fraction.mean()),
                "raw_draws": int(group.proposal_attempts.sum()),
                "full_checks": int(group.full_geometry_checks.sum()),
                "pde_solves": int((group.number_of_pde_solves + group.refinement_pde_solves).sum()),
            }
        )
    write_csv(root / "method_summary.csv", records)
    pairs = []
    for ranking in ("uniform", "flux"):
        control = frame[frame.method == f"fixed8_{ranking}"]
        for quota in (2, 4):
            treatment = frame[frame.method == f"refill{quota}_{ranking}"]
            for _, row in treatment.iterrows():
                ref = control[(control.start == row.start) & (control.seed == row.seed)].iloc[0]
                delta = float(row.fine_final_Q - ref.fine_final_Q)
                pairs.append(
                    {
                        "method": row.method,
                        "control": ref.method,
                        "start": row.start,
                        "family": row.family,
                        "seed": int(row.seed),
                        "both_complete": row.status == ref.status == "complete",
                        "fine_delta_Q": delta,
                        "outcome": "win" if delta > 1e-7 else "loss" if delta < -1e-7 else "tie",
                        "seconds_ratio": float(row.workflow_seconds / ref.workflow_seconds),
                        "raw_draws_ratio": float(row.proposal_attempts / ref.proposal_attempts),
                    }
                )
    write_csv(root / "paired_comparisons.csv", pairs)
    summary = pd.DataFrame(records).set_index("method")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    columns = [
        "mean_fine_gain_Q",
        "mean_multi_choice_fraction",
        "mean_raw_valid_fraction",
        "median_workflow_seconds",
    ]
    labels = [
        "Mean refined flux gain (n=400)",
        "Pools with at least two distinct valid choices",
        "Eligible raw draws / all raw draws",
        "Median workflow seconds, including refinement",
    ]
    for ax, column, label in zip(axes.flat, columns, labels, strict=True):
        summary[column].plot.bar(ax=ax, rot=30)
        ax.set_title(label)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(root / "figures" / "pool_comparison.png", dpi=160)
    plt.close(fig)


def run_feasible_pool_study(config: PoolStudyConfig, progress=None):
    root = create_run(config, "feasible_pool")
    started = perf_counter()
    try:
        starts = _starts(config, root)
        metadata = provenance()
        (root / "provenance.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        methods = [(quota, ranking) for quota in (0, 2, 4) for ranking in ("uniform", "flux")]
        # Saved before the first outcome: no winner selection or automatic promotion.
        protocol = {
            "methods": [f"{'fixed8' if q == 0 else f'refill{q}'}_{r}" for q, r in methods],
            "expected_runs": len(starts) * len(config.evaluation_seeds) * len(methods),
            "raw_budget": config.raw_draw_budget,
            "pde_budget_including_initial": config.solve_budget,
            "draw_limit_per_refilled_pool": config.refill_draw_limit,
            "refined_solves_per_run": 2 * len(config.refinement_factors),
            "unconditional_first_draw_probability": config.epsilon_uniform,
            "study_role": "development mechanism study; no held-out generalization claim",
            "source_manifest_sha256": hashlib.sha256(
                (root / "source_manifest.json").read_bytes()
            ).hexdigest(),
        }
        (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
        rows, maximum_error = [], 0.0
        for start_name, family, geometry in starts:
            for seed in config.evaluation_seeds:
                # Alternate execution order to reduce systematic method timing order effects.
                ordered = methods if seed % 2 == 0 else list(reversed(methods))
                for quota, ranking in ordered:
                    method = f"{'fixed8' if quota == 0 else f'refill{quota}'}_{ranking}"
                    cfg = config.model_copy(update={"seed": seed, "initial_shape": family})
                    path = root / "runs" / f"{start_name}__{seed}__{method}"
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

                    job_started = perf_counter()
                    result = run_feasible_search(
                        cfg, geometry, quota=quota, ranking=ranking, on_evaluation=save_candidate
                    )
                    save_search(path, result.search)
                    write_csv(path / "pools.csv", result.pools)
                    refinements = validate_refinement(cfg, result.search, path)
                    maximum_error = max(
                        maximum_error,
                        result.maximum_conservation_error,
                        *(r["conservation_error"] for r in refinements),
                    )
                    row = {
                        **search_summary(cfg, result.search, refinements),
                        "git_commit": metadata["git_commit"],
                        "source_manifest": "../../source_manifest.json",
                        "start": start_name,
                        "family": family,
                        "method": method,
                        "status": result.status,
                        "pools": len(result.pools),
                        "multi_choice_fraction": float(
                            np.mean([r["eligible"] >= 2 for r in result.pools])
                        ),
                        "raw_valid_fraction": sum(r["eligible"] for r in result.pools)
                        / result.search.attempts,
                        "selected_invalid_fraction": float(
                            np.mean([not r["selected_valid"] for r in result.pools])
                        ),
                        "quota_reached_fraction": float(
                            np.mean([r["quota_reached"] for r in result.pools])
                        ),
                        "full_geometry_checks": result.full_checks,
                        "geometry_validation_seconds": result.validation_seconds,
                        "environment_seconds": result.seconds,
                        "workflow_seconds": perf_counter() - job_started,
                    }
                    save_summary(path, row)
                    rows.append(row)
                    write_csv(root / "evaluations.csv", rows)
                    message = (
                        f"{len(rows)}/{protocol['expected_runs']} {start_name} {seed} {method}: "
                        f"{result.status}, fine gain={row['fine_gain_Q']:.6g}, "
                        f"raw={result.search.attempts}"
                    )
                    with (root / "run.log").open("a", encoding="utf-8") as stream:
                        stream.write(message + "\n")
                    if progress:
                        progress(message)
        assert_source_unchanged(root)
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
            "max_conservation_error": maximum_error,
            "wall_clock_seconds": perf_counter() - started,
            "training_pde_solves": 0,
            "policy_promoted": False,
        }
        save_summary(root, summary)
        return root, summary
    except Exception as exc:
        log_failure(root, exc)
        raise
