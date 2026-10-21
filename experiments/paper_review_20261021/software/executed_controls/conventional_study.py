"""Add paid classical controls to a completed frozen morphology development study."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oxygen_shape.experiments.config import load_config
from oxygen_shape.experiments.morphology_study import MorphologyStudyConfig, _verify_sources
from oxygen_shape.experiments.morphology_validation import validate_endpoint
from oxygen_shape.experiments.reporting import create_run, provenance, write_csv
from oxygen_shape.geometry.research_families import make_research_geometry
from oxygen_shape.optimization.trajectory_search import TrajectoryConfig, _json
from scripts.audit_conventional import audit_job
from scripts.conventional_search import run_classical

METHODS = ("uniform_greedy", "pool_greedy")
EXTRA_SOURCES = (
    "scripts/conventional_search.py",
    "scripts/conventional_study.py",
    "scripts/audit_conventional.py",
    "scripts/audit_morphology_journals.py",
    "scripts/postprocess_morphology_bounds.py",
    "scripts/flux_energy_bounds.py",
    "docs/classical_control_protocol.md",
)


def _worker(payload):
    config, job, root = payload
    config = TrajectoryConfig.model_validate(config)
    initial = make_research_geometry(job["family"], seed=job["geometry_seed"]).geometry

    def progress(row):
        if row["pde_solves"] % 64 == 0:
            print(f"{job['path']}: {row['pde_solves']}/{config.solve_budget}", flush=True)

    return {
        **job,
        **run_classical(
            config, initial, job["method"], Path(root) / job["path"], on_commit=progress
        ),
    }


def _pin(root, source, repository):
    shutil.copyfile(source / "bank_manifest.json", root / "bank_manifest.json")
    files = [
        source / name
        for name in (
            "config.yaml",
            "bank_manifest.json",
            "endpoint_metrics.csv",
            "source_manifest.json",
        )
    ]
    rows = pd.read_csv(source / "endpoint_metrics.csv")
    for row in rows[rows.phase == "comparison"].to_dict("records"):
        files.extend((source / row["path"] / "validation").glob("*.npz"))
    _json(
        root / "reference_inputs.json",
        {
            os.path.relpath(file, root): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in files
        },
    )
    hashes = {}
    for relative in EXTRA_SOURCES:
        file = repository / relative
        destination = root / "control_sources" / file.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(file, destination)
        hashes[relative] = hashlib.sha256(file.read_bytes()).hexdigest()
    _json(root / "control_source_manifest.json", hashes)


def _verify(root, repository):
    _verify_sources(root)
    for name, expected in json.loads((root / "control_source_manifest.json").read_text()).items():
        if hashlib.sha256((repository / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Classical source changed: {name}")
    for name, expected in json.loads((root / "reference_inputs.json").read_text()).items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Imported reference changed: {name}")


def report(root, endpoints, source):
    data = pd.DataFrame(endpoints)
    controls = data[data.method == "flux_greedy"].set_index(["family", "seed"])
    comparisons = []
    training_cost = json.loads((source / "trained_model_card.json").read_text())["pde_solves"]
    for method in data.method.unique():
        group = data[data.method == method].set_index(["family", "seed"])
        paired = group.join(controls[["best_Q400", "best_FEM_2e-05"]], rsuffix="_control")
        delta = paired.best_Q400 - paired.best_Q400_control
        fem_delta = paired["best_FEM_2e-05"] - paired["best_FEM_2e-05_control"]
        comparisons.append(
            dict(
                method=method,
                pairs=len(paired),
                wins=int((delta > 1e-7).sum()),
                losses=int((delta < -1e-7).sum()),
                ties=int((abs(delta) <= 1e-7).sum()),
                mean_delta_Q400=float(delta.mean()),
                median_delta_Q400=float(delta.median()),
                mean_delta_FEM=float(fem_delta.mean()),
                complete=int((group.status == "complete").sum()),
            )
        )
    write_csv(root / "paired_method_comparisons.csv", comparisons)
    families = data.family.unique()
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), squeeze=False)
    colors = dict(
        uniform_greedy="grey",
        pool_greedy="tab:purple",
        flux_greedy="tab:blue",
        flux_anneal="tab:orange",
        learned_anneal="tab:green",
    )
    for ax, family in zip(axes.flat, families, strict=False):
        for method in data.method.unique():
            group = data[(data.family == family) & (data.method == method)]
            curves = []
            for job in group.to_dict("records"):
                history = pd.read_csv(root / job["path"] / "metrics.csv")
                curves.append(
                    pd.Series(
                        100 * (history.best_Q / history.best_Q.iloc[0] - 1),
                        index=history.pde_solves,
                    )
                )
            values = pd.concat(curves, axis=1)
            ax.plot(values.index, values.median(axis=1), label=method, color=colors[method])
            ax.fill_between(
                values.index,
                values.quantile(0.25, axis=1),
                values.quantile(0.75, axis=1),
                alpha=0.12,
                color=colors[method],
            )
        ax.set(title=family, xlabel="Charged coarse PDE solves", ylabel="Best coarse Q gain (%)")
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(families) :]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root / "figures/algorithm_curves.png", dpi=170)
    fig.savefig(root / "figures/algorithm_curves.pdf")
    plt.close(fig)
    table = "\n".join(
        f"| {r['method']} | {r['wins']}/{r['ties']}/{r['losses']} | "
        f"{r['mean_delta_Q400']:+.6f} | {r['mean_delta_FEM']:+.6f} |"
        for r in comparisons
    )
    (root / "report.md").write_text(
        f"""# Paid classical controls and five-method development comparison

The original methods and their independent fields are imported from {source.name}.
The new methods use the same starts, paired seeds, constraints, candidate primitives,
refill rules and maximum PDE budgets. Each evaluated pool candidate is charged.
The last pool may be truncated at the remaining budget, which is recorded explicitly.
All endpoints receive the same refinement/FEM checks. Source and input hashes are pinned.

| Method | Q400 wins/ties/losses vs greedy flux | Mean delta Q400 | Mean delta FEM Q |
|---|---:|---:|---:|
{table}

![Paid algorithm comparison](figures/algorithm_curves.png)

Bands are interquartile variation over five geometry/search pairs per family,
not confidence intervals or independent anatomical samples. These are known
development masks; controls were added after the main comparison was observed.
No confirmatory superiority or crossed-variance study is claimed. Win thresholds
1e-7 are descriptive; use the energy-bound overlap review for numerical resolution.
Membrane lengths differ across starts and can grow within the common cap 8.
Total Q is the objective and eta diagnostic. Concurrent timings are descriptive.

The original training costs {training_cost} solves and is
additional to the learned evaluations. The two new methods use no training.
Imported studies are not rerun or counted as newly executed work. New solver
costs and the full classical journal audit are retained in summary.json.
This is a benchmark of the tested finite raster problem, not a global optimum,
a new classical optimizer or proof of natural-acinar optimality.
""",
        encoding="utf-8",
    )


def run(source, *, workers=2, resume=None):
    source = source.resolve()
    if workers not in (1, 2, 3, 4):
        raise ValueError("Use one to four single-thread workers")
    if json.loads((source / "status.json").read_text())["phase"] != "finished":
        raise RuntimeError("Complete the original study before extending its reported comparison")
    repository = Path(__file__).resolve().parents[1]
    _verify_sources(source)
    config = load_config(source / "config.yaml", MorphologyStudyConfig)
    if config.n != 100 or config.model.Bi != 1 or config.target_area != 0.24:
        raise ValueError("The extension targets the frozen n100, Bi1, area0.24 protocol")
    original = pd.read_csv(source / "endpoint_metrics.csv", float_precision="round_trip")
    original = original[original.phase == "comparison"]
    if resume:
        root = resume.resolve()
        _verify(root, repository)
        record = json.loads((root / "extension.json").read_text())
        if Path(record["source"]).resolve() != source or record["workers"] != workers:
            raise ValueError("Changed classical extension identity")
        if (root / "status.json").exists():
            state = json.loads((root / "status.json").read_text())
            if state.get("status") in ("complete", "finished_with_budget_truncations"):
                return root
    else:
        root = create_run(config, "classical")
        _json(
            root / "extension.json",
            dict(
                source=str(source), workers=workers, scope="known development comparison extension"
            ),
        )
        _pin(root, source, repository)
        _json(root / "environment.json", provenance())
    print(f"RUN_DIRECTORY={root}", flush=True)
    starts = original[original.method == "flux_greedy"].to_dict("records")
    if not starts:
        raise ValueError("No paired greedy-flux controls in the reference study")
    jobs = []
    for index, row in enumerate(starts):
        initial = make_research_geometry(row["family"], seed=int(row["geometry_seed"])).geometry
        np.testing.assert_array_equal(
            initial.inside, np.load(source / row["path"] / "initial_geometry.npy")
        )
        for method in METHODS:
            jobs.append(
                dict(
                    phase="comparison",
                    family=row["family"],
                    geometry_seed=int(row["geometry_seed"]),
                    seed=int(row["seed"]),
                    method=method,
                    solve_budget=config.solve_budget,
                    perimeter_band=None,
                    path=f"search/{'u' if method == METHODS[0] else 'b'}/{index:02d}",
                )
            )
    _json(root / "jobs.json", jobs)
    payloads = [
        (
            config.trajectory(seed=j["seed"], solve_budget=j["solve_budget"]).model_dump(
                mode="json"
            ),
            j,
            str(root),
        )
        for j in jobs
    ]
    completed, endpoints, audits = [], [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(_worker, payload) for payload in payloads]):
            result = future.result()
            completed.append(result)
            job = {key: result[key] for key in jobs[0]}
            endpoints.append(validate_endpoint(root, config, job))
            initial = make_research_geometry(job["family"], seed=job["geometry_seed"]).geometry
            audits.append(dict(path=job["path"], **audit_job(root / job["path"], initial)))
            _json(
                root / "status.json",
                dict(
                    phase="search_with_validation",
                    status="running",
                    completed=len(completed),
                    total=len(jobs),
                ),
            )
            print(
                f"Classical search, validation and audit complete {len(completed)}/{len(jobs)}",
                flush=True,
            )
    write_csv(root / "search_metrics.csv", completed)
    write_csv(root / "new_endpoint_metrics.csv", endpoints)
    for row in original.to_dict("records"):
        row["path"] = Path(os.path.relpath(source / row["path"], root)).as_posix()
        row["reused_source_study"] = source.name
        endpoints.append(row)
    write_csv(root / "endpoint_metrics.csv", endpoints)
    report(root, endpoints, source)
    interruptions = [json.loads(file.read_text()) for file in root.rglob("interrupted_*.json")]
    _json(
        root / "summary.json",
        dict(
            new_search_PDE=sum(r["pde_solves"] for r in completed),
            new_FVM_validation=len(list(root.rglob("*.fvm.npz"))),
            new_FEM_validation=len(list(root.rglob("*.fem.npz"))),
            new_training_PDE=0,
            interrupted_attempts=len(interruptions),
            uncommitted_extra_PDE_lower=sum(r["extra_pde_lower_bound"] for r in interruptions),
            uncommitted_extra_PDE_upper=sum(r["extra_pde_upper_bound"] for r in interruptions),
            validation_scheduling="Parent validates/audits completed jobs while workers continue",
            imported_comparison_endpoints=len(original),
            added_endpoints=len(jobs),
            incomplete_searches=sum(r["status"] != "complete" for r in completed),
            audit=dict(
                candidate_fields=sum(r["candidate_fields"] for r in audits),
                maximum_conservation_error=max(r["maximum_conservation_error"] for r in audits),
                maximum_cell_residual=max(r["maximum_cell_residual"] for r in audits),
                additional_PDE_solves=0,
            ),
            source_study=source.name,
        ),
    )
    write_csv(root / "journal_audit.csv", audits)
    _json(root / "status.json", dict(phase="finished", status="energy_review_running"))
    subprocess.run(
        [sys.executable, str(repository / "scripts/postprocess_morphology_bounds.py"), str(root)],
        check=True,
    )
    _verify(root, repository)
    _json(
        root / "status.json",
        dict(
            phase="finished",
            status="complete"
            if all(r["status"] == "complete" for r in completed)
            else "finished_with_budget_truncations",
        ),
    )
    print(f"Completed classical comparison: {root / 'report.md'}", flush=True)
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    run(args.source, workers=args.workers, resume=args.resume)
