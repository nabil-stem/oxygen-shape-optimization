"""Literature-inspired geometry bank, measured-response training and long searches.

Run: python -m oxygen_shape.experiments.morphology_study --config PATH
Resume: same command with --resume EXISTING_RUN. No source edits are allowed
between launch and resume; all previous experiments remain unchanged.
"""

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np
from pydantic import Field, model_validator

from oxygen_shape.experiments.config import load_config
from oxygen_shape.experiments.morphology_bank import prepare_bank
from oxygen_shape.experiments.morphology_validation import (
    cached_fvm,
    run_catalog,
    run_physical_polygon_checks,
    validate_endpoint,
)
from oxygen_shape.experiments.reporting import create_run, write_csv
from oxygen_shape.geometry.disk_edits import prepare_boundary
from oxygen_shape.geometry.research_families import FAMILIES, make_research_geometry
from oxygen_shape.optimization.candidate_policy import FEATURE_NAMES
from oxygen_shape.optimization.candidate_rollout import candidate_features
from oxygen_shape.optimization.donor_sampling import eligible_wall_flux
from oxygen_shape.optimization.trajectory_pool import build_trajectory_pool
from oxygen_shape.optimization.trajectory_ranker import ResponseRanker, append_access_features
from oxygen_shape.optimization.trajectory_search import (
    METHODS,
    TrajectoryConfig,
    _json,
    run_trajectory,
)


class MorphologyStudyConfig(TrajectoryConfig):
    families: list[str] = Field(default_factory=lambda: list(FAMILIES))
    training_families: list[str] = Field(
        default_factory=lambda: [f for f in FAMILIES if f not in ("maze", "hilbert_2")]
    )
    methods: list[str] = Field(
        default_factory=lambda: ["flux_greedy", "flux_anneal", "learned_anneal"]
    )
    evaluation_seeds: list[int] = Field(default_factory=lambda: list(range(320000, 320005)))
    training_bank_seed: int = Field(default=310001, ge=0)
    evaluation_bank_seed: int = Field(default=310101, ge=0)
    training_solve_budget: int = Field(default=65, ge=2)
    solve_budget: int = Field(default=129, ge=2)
    deep_family: str = "acinar_sacs"
    deep_solve_budget: int = Field(default=1025, ge=2)
    perimeter_control_families: list[str] = Field(
        default_factory=lambda: ["rectangle", "hexagon", "acinar_sacs"]
    )
    perimeter_control_budget: int = Field(default=65, ge=2)
    permeabilities: list[float] = Field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0])
    counterfactual_pools: int = Field(default=5, ge=1)
    workers: int = Field(default=2, ge=1, le=2)

    @model_validator(mode="after")
    def valid_study(self):
        if self.n != 100 or self.model.Bi != 1.0:
            raise ValueError("This frozen morphology protocol uses n=100 and search Bi=1")
        for names in (self.families, self.training_families, self.perimeter_control_families):
            if not names or len(set(names)) != len(names) or not set(names) <= set(FAMILIES):
                raise ValueError("Use distinct supported morphology families")
        if self.deep_family not in self.families or not set(self.methods) <= set(METHODS):
            raise ValueError("Unknown deep family or method")
        if len(self.evaluation_seeds) < 5 or len(set(self.evaluation_seeds)) != len(
            self.evaluation_seeds
        ):
            raise ValueError(
                "The morphology experiment requires at least five distinct search seeds"
            )
        if set(
            range(self.evaluation_bank_seed, self.evaluation_bank_seed + len(self.evaluation_seeds))
        ) & {self.training_bank_seed}:
            raise ValueError("Training and evaluation geometry seeds overlap")
        return self

    def trajectory(self, **updates):
        values = {k: v for k, v in self.model_dump().items() if k in TrajectoryConfig.model_fields}
        return TrajectoryConfig.model_validate({**values, **updates})


def _verify_sources(root):
    repository = Path(__file__).resolve().parents[3]
    manifest = json.loads((root / "source_manifest.json").read_text())
    for name, expected in manifest.items():
        if hashlib.sha256((repository / name).read_bytes()).hexdigest() != expected:
            raise ValueError(
                f"Frozen source changed: {name}. Restore the snapshot before resuming."
            )


def _worker(payload):
    config_data, job, root_string = payload
    config = TrajectoryConfig.model_validate(config_data)
    root = Path(root_string)
    case = make_research_geometry(
        job["family"], config.n, config.target_area, seed=job["geometry_seed"]
    )
    warm = root / "trained_response_model.npz" if job["method"].startswith("learned") else None

    def progress(row):
        if row["pde_solves"] % 64 == 0:
            print(
                f"{job['path']}: {row['pde_solves']}/{config.solve_budget}, "
                f"best Q={row['best_Q']:.7f}",
                flush=True,
            )

    result = run_trajectory(
        config,
        case.geometry,
        job["method"],
        root / job["path"],
        warm_start=warm,
        on_commit=progress,
    )
    return job, result


def train_response_model(root: Path, config):
    warm = None
    rows = []
    for index, family in enumerate(config.training_families):
        case = make_research_geometry(
            family, config.n, config.target_area, seed=config.training_bank_seed
        )
        directory = root / "training" / f"{index:02d}_{family}"
        cfg = config.trajectory(
            seed=config.seed + 1000 + index, solve_budget=config.training_solve_budget
        )
        result = run_trajectory(cfg, case.geometry, "learned_anneal", directory, warm_start=warm)
        rows.append({"family": family, **result})
        warm = directory / "response_model.npz"
        print(
            f"Training: {family}, {result['new_model_updates']} measured response updates",
            flush=True,
        )
    destination = root / "trained_response_model.npz"
    if destination.exists() and destination.read_bytes() != warm.read_bytes():
        raise ValueError("Trained model changed on resume")
    if not destination.exists():
        shutil.copyfile(warm, destination)
    write_csv(root / "training_metrics.csv", rows)
    _json(
        root / "trained_model_card.json",
        {
            "algorithm": "recursive ridge regression of 100*delta_Q/Q_initial",
            "role": "warm-start for online candidate ranking, not an RL value function",
            "training_families": config.training_families,
            "unseen_training_families": sorted(
                set(config.families) - set(config.training_families)
            ),
            "pde_solves": sum(r["pde_solves"] for r in rows),
            "source_path_features": True,
            "model_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        },
    )


def counterfactual_diagnostic(root: Path, config, geometry_seeds):
    ranker = ResponseRanker()
    with np.load(root / "trained_response_model.npz", allow_pickle=False) as data:
        ranker.weights, ranker.inverse = data["weights"].copy(), data["inverse"].copy()
    rows, candidates = [], []
    for family_index, family in enumerate(config.families):
        g = make_research_geometry(
            family, config.n, config.target_area, seed=geometry_seeds[family][0]
        ).geometry
        solution = cached_fvm(
            root / "catalog" / family / "Bi_1.0_f1.fvm.npz", g, 1.0, config.solver
        )
        cfg = config.trajectory(seed=config.seed + 2000 + family_index)
        prepared = prepare_boundary(g, cfg.protected_length)
        flux = eligible_wall_flux(g, solution, cfg.protected_length)
        rng, raw, seen = np.random.default_rng(cfg.seed), 0, set()
        for pool_index in range(config.counterfactual_pools):
            pool = build_trajectory_pool(
                cfg,
                g,
                g,
                prepared,
                rng,
                cfg.raw_draw_budget - raw,
                seen,
                quota=4,
                initial_perimeter=solution.metrics.robin_perimeter,
            )
            raw += len(pool.edits)
            base = candidate_features(
                g,
                solution,
                cfg,
                solution.metrics.physical_flux_Q,
                1,
                raw,
                prepared,
                pool.edits,
                pool.rows,
                pool.rows,
                flux,
            )
            features = append_access_features(base, g, prepared, pool.rows, cfg.protected_length)
            eligible = np.flatnonzero(pool.eligible)[:4]
            predictions = ranker.scores(features[eligible])
            flux_scores = base[eligible, FEATURE_NAMES.index("flux_product")]
            changes = []
            for slot in eligible:
                edit = pool.edits[int(slot)]
                seen.add(edit.geometry.inside.tobytes())
                result = cached_fvm(
                    root / "counterfactual" / family / f"pool{pool_index}_slot{slot}.fvm.npz",
                    edit.geometry,
                    cfg.model.Bi,
                    cfg.solver,
                )
                change = result.metrics.physical_flux_Q - solution.metrics.physical_flux_Q
                changes.append(change)
                candidates.append(
                    dict(
                        family=family,
                        pool=pool_index,
                        slot=int(slot),
                        delta_Q=change,
                        **{f"feature_{i}": float(x) for i, x in enumerate(features[slot])},
                    )
                )
            if not len(eligible):
                continue
            changes = np.asarray(changes)
            model_slot, flux_slot = int(np.argmax(predictions)), int(np.argmax(flux_scores))
            rows.append(
                dict(
                    family=family,
                    pool=pool_index,
                    choices=len(eligible),
                    oracle_delta=float(changes.max()),
                    learned_delta=float(changes[model_slot]),
                    flux_delta=float(changes[flux_slot]),
                    learned_regret=float(changes.max() - changes[model_slot]),
                    flux_regret=float(changes.max() - changes[flux_slot]),
                    raw_draws=raw,
                    full_checks=pool.full_checks,
                )
            )
    write_csv(root / "counterfactual_pools.csv", rows)
    write_csv(root / "counterfactual_candidates.csv", candidates)


def make_jobs(config, geometry_seeds=None):
    jobs = []
    phases = [
        ("comparison", config.families, config.solve_budget, None),
        ("deep", [config.deep_family], config.deep_solve_budget, None),
        (
            "matched_perimeter",
            config.perimeter_control_families,
            config.perimeter_control_budget,
            0.0,
        ),
    ]
    for phase, families, budget, band in phases:
        for index, seed in enumerate(config.evaluation_seeds):
            for family in families:
                for method in config.methods:
                    phase_code = {"comparison": "m", "deep": "d", "matched_perimeter": "p"}[phase]
                    path = (
                        f"search/{phase_code}/{config.families.index(family):02d}_"
                        f"{index}_{config.methods.index(method)}"
                    )
                    jobs.append(
                        dict(
                            phase=phase,
                            family=family,
                            geometry_seed=geometry_seeds[family][index]
                            if geometry_seeds
                            else config.evaluation_bank_seed + index,
                            method=method,
                            seed=seed,
                            solve_budget=budget,
                            perimeter_band=band,
                            path=path,
                        )
                    )
    return jobs


def run_study(config, root: Path | None = None):
    started = perf_counter()
    if root is None:
        root = create_run(config, "morphology")
        repository = Path(__file__).resolve().parents[3]
        shutil.copyfile(repository / "docs/morphology_protocol.md", root / "protocol.md")
        shutil.copyfile(
            repository / "requirements-validation.txt", root / "requirements-validation.txt"
        )
    else:
        recorded = load_config(root / "config.yaml", MorphologyStudyConfig)
        if recorded.model_dump(mode="json") != config.model_dump(mode="json"):
            raise ValueError("Study config changed on resume")
        _verify_sources(root)
    print(f"RUN_DIRECTORY={root.resolve()}", flush=True)
    geometry_seeds = prepare_bank(root, config)
    _json(root / "status.json", {"phase": "numerical_checks", "status": "running"})
    run_physical_polygon_checks(root, config)
    run_catalog(root, config, geometry_seeds)
    train_response_model(root, config)
    counterfactual_diagnostic(root, config, geometry_seeds)
    jobs = make_jobs(config, geometry_seeds)
    _json(root / "jobs.json", jobs)
    payloads = [
        (
            config.trajectory(
                seed=j["seed"], solve_budget=j["solve_budget"], perimeter_band=j["perimeter_band"]
            ).model_dump(mode="json"),
            j,
            str(root),
        )
        for j in jobs
    ]
    complete = []
    with ProcessPoolExecutor(max_workers=config.workers) as pool:
        futures = [pool.submit(_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            job, result = future.result()
            complete.append({**job, **result})
            _json(
                root / "status.json",
                {
                    "phase": "search",
                    "status": "running",
                    "completed": len(complete),
                    "total": len(jobs),
                    "last_job": job["path"],
                },
            )
            print(
                f"Finished {len(complete)}/{len(jobs)}: {job['path']}, {result['status']}",
                flush=True,
            )
    write_csv(root / "search_metrics.csv", complete)
    endpoints = []
    for index, job in enumerate(jobs):
        endpoints.append(validate_endpoint(root, config, job))
        if (index + 1) % 10 == 0:
            _json(
                root / "status.json",
                {
                    "phase": "endpoint_validation",
                    "status": "running",
                    "completed": index + 1,
                    "total": len(jobs),
                },
            )
            print(
                f"Refined and independently checked {index + 1}/{len(jobs)} endpoints", flush=True
            )
    write_csv(root / "endpoint_metrics.csv", endpoints)
    from oxygen_shape.experiments.morphology_report import make_report

    make_report(root, config)
    _json(
        root / "status.json",
        {
            "phase": "finished",
            "status": "complete"
            if all(r["status"] == "complete" for r in complete)
            else "finished_with_budget_truncations",
            "jobs": len(jobs),
            "this_invocation_seconds": perf_counter() - started,
        },
    )
    print(f"Completed report: {root / 'report.md'}", flush=True)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    run_study(load_config(args.config, MorphologyStudyConfig), args.resume)


if __name__ == "__main__":
    main()
