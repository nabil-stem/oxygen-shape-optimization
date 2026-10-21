"""Physical flux, computational cost and qualified conclusions from saved data."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm

from oxygen_shape.experiments.morphology_validation import cached_fvm
from oxygen_shape.experiments.reporting import write_csv
from oxygen_shape.geometry.boundary import BoundaryKind
from oxygen_shape.geometry.raster import OFFSETS, refine_geometry
from oxygen_shape.geometry.research_families import make_research_geometry
from oxygen_shape.optimization.trajectory_search import _json
from oxygen_shape.viz.history import gain_by_solve

COLORS = {
    "flux_greedy": "tab:blue",
    "flux_anneal": "tab:orange",
    "learned_anneal": "tab:green",
    "learned_frozen_anneal": "tab:purple",
}


def _save(fig, root, name):
    fig.savefig(root / "figures" / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_report(root: Path, config):
    endpoints = pd.read_csv(root / "endpoint_metrics.csv", float_precision="round_trip")
    physical = pd.read_csv(root / "physical_polygon_checks.csv", float_precision="round_trip")
    pools = pd.read_csv(root / "counterfactual_pools.csv", float_precision="round_trip")
    model = json.loads((root / "trained_model_card.json").read_text())
    bank = json.loads((root / "bank_manifest.json").read_text())
    figs = root / "figures"
    figs.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)
    for ax, family in zip(axes.flat, config.families, strict=False):
        with np.load(root / "catalog" / family / "Bi_1.0_f4.fvm.npz", allow_pickle=False) as data:
            field = data["field"]
            metrics = json.loads(str(data["metrics"]))
        image = ax.imshow(
            field,
            origin="lower",
            extent=[0, 1, 0, 1],
            cmap="viridis",
            norm=PowerNorm(0.35, vmin=0, vmax=1),
        )
        ax.plot([0, 0], [0.45, 0.55], color="red", linewidth=2)
        ax.set_title(
            f"{family}\nQ400={metrics['physical_flux_Q']:.5f}; P={metrics['robin_perimeter']:.2f}"
        )
        ax.set(xlim=(0, 1), ylim=(0, 1))
    fig.colorbar(
        image, ax=axes.ravel().tolist(), shrink=0.55, label="u, common power display gamma=0.35"
    )
    for ax in list(axes.flat)[len(config.families) :]:
        ax.set_visible(False)
    _save(fig, root, "geometry_catalog.png")

    def learning_plot(phase, name):
        families = config.families if phase == "comparison" else [config.deep_family]
        fig, axes = plt.subplots(
            3 if len(families) > 1 else 1,
            3 if len(families) > 1 else 1,
            figsize=(13, 10) if len(families) > 1 else (9, 5),
            squeeze=False,
        )
        for ax, family in zip(axes.flat, families, strict=False):
            for method in config.methods:
                jobs = endpoints[
                    (endpoints.phase == phase)
                    & (endpoints.family == family)
                    & (endpoints.method == method)
                ]
                curves = []
                for job in jobs.to_dict("records"):
                    history = pd.read_csv(root / job["path"] / "metrics.csv")
                    curves.append(gain_by_solve(history))
                if not curves:
                    continue
                values = pd.concat(curves, axis=1)
                median = values.median(axis=1)
                ax.plot(median.index, median, color=COLORS[method], label=method)
                ax.fill_between(
                    median.index,
                    values.quantile(0.25, axis=1),
                    values.quantile(0.75, axis=1),
                    color=COLORS[method],
                    alpha=0.15,
                )
            ax.set(
                title=family, xlabel="Charged coarse PDE solves", ylabel="Best coarse Q gain (%)"
            )
            ax.grid(alpha=0.2)
        for ax in list(axes.flat)[len(families) :]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=8)
        fig.tight_layout()
        _save(fig, root, name)

    learning_plot("comparison", "comparison_curves.png")
    learning_plot("deep", "long_horizon_curves.png")

    fig, ax = plt.subplots(figsize=(9, 6))
    for method in config.methods:
        group = endpoints[(endpoints.phase == "comparison") & (endpoints.method == method)]
        ax.scatter(
            group.best_perimeter, group.best_Q400, label=method, color=COLORS[method], alpha=0.6
        )
    start = endpoints[endpoints.method == "flux_greedy"].drop_duplicates(
        ["family", "geometry_seed"]
    )
    ax.scatter(
        start.initial_perimeter, start.initial_Q400, label="Starts", color="black", marker="x"
    )
    ax.set(
        xlabel="Actual Robin perimeter",
        ylabel="Q400",
        title="Fixed area 0.24, common perimeter cap 8",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    _save(fig, root, "flux_vs_membrane.png")

    comparison_rows = []
    for phase in endpoints.phase.unique():
        subset = endpoints[endpoints.phase == phase]
        baseline = subset[subset.method == "flux_greedy"].set_index(["family", "seed"])
        for method in config.methods:
            current = subset[subset.method == method].set_index(["family", "seed"])
            pairs = current.join(baseline[["best_Q400", "best_FEM_2e-05"]], rsuffix="_control")
            delta = pairs.best_Q400 - pairs.best_Q400_control
            fem_delta = pairs["best_FEM_2e-05"] - pairs["best_FEM_2e-05_control"]
            comparison_rows.append(
                dict(
                    phase=phase,
                    method=method,
                    pairs=len(pairs),
                    wins=int((delta > 1e-7).sum()),
                    losses=int((delta < -1e-7).sum()),
                    ties=int((abs(delta) <= 1e-7).sum()),
                    mean_delta_Q400=float(delta.mean()),
                    median_delta_Q400=float(delta.median()),
                    mean_delta_FEM=float(fem_delta.mean()),
                    complete=int((current.status == "complete").sum()),
                )
            )
    write_csv(root / "paired_method_comparisons.csv", comparison_rows)

    # Endpoint selection is explicitly post-hoc. Rank with the independent FEM
    # result as well as Q400; neither proves an optimum among all internal edits.
    winner = endpoints.sort_values("best_FEM_2e-05", ascending=False).iloc[0].to_dict()
    winner = {
        key: None if isinstance(value, float) and np.isnan(value) else value
        for key, value in winner.items()
    }
    initial = make_research_geometry(
        winner["family"], config.n, config.target_area, seed=int(winner["geometry_seed"])
    ).geometry
    best = initial.with_mask(np.load(root / winner["path"] / "final_geometry.npy"))
    np.save(root / "initial_geometry.npy", initial.inside)
    np.save(root / "final_geometry.npy", best.inside)
    diagnostics = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for index, (label, geometry) in enumerate(
        (("Initial", initial), ("Best returned endpoint", best))
    ):
        geometry4 = refine_geometry(geometry, 4)
        field = cached_fvm(
            root
            / winner["path"]
            / "validation"
            / f"{'initial' if index == 0 else 'best'}_f4.fvm.npz",
            geometry4,
            config.model.Bi,
            config.solver,
        )
        axes[index].imshow(
            field.concentration,
            origin="lower",
            extent=[0, 1, 0, 1],
            cmap="viridis",
            norm=PowerNorm(0.35, vmin=0, vmax=1),
        )
        axes[index].plot([0, 0], [0.45, 0.55], color="red", linewidth=2)
        axes[index].set_title(
            f"{label}\nQ400={field.metrics.physical_flux_Q:.6f}; "
            f"P={field.metrics.robin_perimeter:.2f}"
        )
        binned = np.zeros(5)
        boundaries = np.array([0, 0.12, 0.25, 0.5, 0.75, 1.00001])
        for face, flux in zip(field.faces, field.outward_face_flux, strict=True):
            if face.kind != BoundaryKind.ROBIN:
                continue
            _, dx = OFFSETS[face.key.side]
            x = (face.key.col + 0.5 + dx / 2) * geometry4.h
            bin_index = min(4, max(0, int(np.searchsorted(boundaries, x, side="right") - 1)))
            binned[bin_index] += flux
        for i, value in enumerate(binned):
            diagnostics.append(
                dict(
                    shape=label,
                    x_min=float(boundaries[i]),
                    x_max=float(boundaries[i + 1]),
                    Q=float(value),
                    fraction=float(value / field.metrics.physical_flux_Q),
                )
            )
    axes[2].imshow(best.inside, origin="lower", extent=[0, 1, 0, 1], cmap="Greys", vmin=0, vmax=1)
    axes[2].plot([0, 0], [0.45, 0.55], color="red", linewidth=2)
    axes[2].set_title(f"Physical raster mask\n{winner['family']}, {winner['method']}")
    _save(fig, root, "best_geometry.png")
    write_csv(root / "winner_flux_by_position.csv", diagnostics)
    _json(root / "best_endpoint.json", winner)

    training = pd.read_csv(root / "training_metrics.csv")
    costs = {
        "training_PDE": int(training.pde_solves.sum()),
        "search_PDE": int(endpoints.pde_solves.sum()),
        "cached_validation_and_counterfactual_FVM_PDE": len(list(root.rglob("*.fvm.npz"))),
        "independent_FEM_solves": len(list(root.rglob("*.fem.npz"))),
        "search_raw_draws": int(endpoints.raw_draws.sum()),
        "search_full_geometry_checks": int(endpoints.checks.sum()),
        "search_downhill_moves": int(endpoints.downhill_moves.sum()),
        "incomplete_searches": int((endpoints.status != "complete").sum()),
        "interrupted_search_attempts": len(list(root.rglob("interrupted_*.json"))),
    }
    physical_rows = []
    for family in physical.family.unique():
        data = physical[physical.family == family]
        fem = data[data.solver == "polygon_FEM"].sort_values("resolution").iloc[0]
        fv = data[(data.solver == "raster_FVM") & (data.resolution == 400)].iloc[0]
        physical_rows.append(
            f"| {family} | {fem.Q:.6f} | {fv.Q:.6f} | {100 * (fv.Q / fem.Q - 1):+.2f}% |"
        )
    method_rows = [
        f"| {r['phase']} | {r['method']} | {r['wins']}/{r['ties']}/{r['losses']} | "
        f"{r['mean_delta_Q400']:+.6f} | {r['mean_delta_FEM']:+.6f} |"
        for r in comparison_rows
    ]
    zero_band = endpoints[endpoints.phase == "matched_perimeter"]
    band_drift = float(abs(zero_band.best_perimeter - zero_band.initial_perimeter).max())
    known = set(config.training_families)
    holdout = endpoints[(endpoints.phase == "comparison") & ~endpoints.family.isin(known)]
    text = f"""# Literature-inspired geometry and long-horizon search study

The physical objective is total Q, with fixed area {config.target_area}, source length 0.1,
unit bounding box, connected no-hole domains,
operational feature width {config.minimum_feature_width}
and a shared Robin-perimeter cap {config.perimeter_limit}. Eta is reported as a diagnostic only.
The cap differs from the earlier cap-4 studies; their winners are not directly comparable.

Nine finite proxies cover simple chambers, regularized square-Koch profiles, a serpentine maze,
binary branching, hexagonal terminal sacs and a Hilbert-like path. These are literature-inspired
controls, not reproductions of measured acini. Training excludes maze and Hilbert families.
The training model uses {model["pde_solves"]} PDE solves. Online search variants continue updating
from actual signed flux changes. This is response regression and heuristic annealing, not a new
deep-RL or PINN result. Every accepted and rejected evaluated move is retained.
All methods use the same raw mixture of paired disks and area/perimeter-preserving
cell exchanges. The latter are declared raster edits, with full feature checks;
they are not represented as disk operations. This supplies small resource-preserving
moves when large disk edits would stall the exact-perimeter control.

![Geometry bank](figures/geometry_catalog.png)

## Search comparisons

Five search seeds are paired within each family and geometry variant. These pairs share their
training model and are grouped by morphology; they are not independent evidence for each
possible natural acinus. All truncated searches remain included. Wins use a reporting threshold
1e-7, which is not a continuum significance threshold.
The evaluation bank contains {bank["evaluation_distinct_masks"]} distinct masks and zero
training-mask overlap. Any seed retries were based only on geometry validity or identity;
bank_manifest.json preserves them. Geometric distinction is not statistical independence
or evidence that the variants reproduce natural acinar variation.

| Phase | Method | Wins/ties/losses vs greedy flux | Mean delta Q400 | Mean delta FEM Q |
|---|---|---:|---:|---:|
{chr(10).join(method_rows)}

![Best flux trajectories](figures/comparison_curves.png)

Deep trajectories start from the same predeclared sac-like geometry family and spend up to
{config.deep_solve_budget} PDE solves each ({config.deep_solve_budget - 1} evaluated edits).
Annealing can temporarily lose flux; the saved best is never discarded. Bands show interquartile
variation, not confidence intervals. Progress is indexed by actual solves, not raw proposal count.

![Long-horizon search](figures/long_horizon_curves.png)

The exact-perimeter control keeps each start's actual membrane length unchanged. Maximum measured
drift is {band_drift:.3g}. This controls within-start membrane growth; different geometry families
still have different initial perimeters. The Q/perimeter plot makes that distinction explicit.

![Flux and membrane amount](figures/flux_vs_membrane.png)

## Measured candidate-response diagnosis

All candidates in {len(pools)} fixed pools were evaluated and charged separately. Learned mean
regret relative to the best evaluated candidate is {pools.learned_regret.mean():.6g} Q; the flux
heuristic's is {pools.flux_regret.mean():.6g}. These oracle labels are diagnostic and were not
used to retrain the warm-start model. Unseen-family comparison rows: {len(holdout)}.

## Independent solver and boundary representation

Every returned endpoint and its start is refined at n=200/400 and checked with independently
assembled triangular FEM on the identical staircase boundary, at two mesh area limits. This
checks a discrete geometry, not an anatomical continuum surface.

Separately, declared physical polygons have exact area 0.24 and the same source. Their raster
versions are regenerated at increasing resolution, without correcting membrane length or Bi.

| Physical polygon | Finest polygon FEM Q | Raster FVM Q400 | Relative difference |
|---|---:|---:|---:|
{chr(10).join(physical_rows)}

Any persistent difference is a boundary/modeling issue as well as a possible PDE error. Agreement
on a staircase must not be presented as proof of smooth-boundary accuracy. Source-corner effects
and Cartesian perimeter bias remain limitations. The complete Bi sweep is in catalog_metrics.csv.

## Best returned endpoint

Post-hoc highest independently checked FEM Q: family {winner["family"]}, method {winner["method"]},
seed {int(winner["seed"])}, phase {winner["phase"]}. Q400={winner["best_Q400"]:.9f};
FEM Q={winner["best_FEM_2e-05"]:.9f}. Fine-grid gain over its own start:
{winner["fine_gain_percent"]:.3f}%;
membrane-length change: {winner["membrane_change_percent"]:+.3f}%.
The independent FEM gain is {winner["FEM_gain_percent"]:.3f}%.

![Selected endpoint](figures/best_geometry.png)

The reason to call this endpoint better is increased global absorption under its declared
constraints, supported by the two discretizations. A color pattern or resemblance to anatomy
is not evidence. winner_flux_by_position.csv attributes absorption to wall-position bins;
best_endpoint.json records the full selection provenance. Selection is among returned coarse-best
endpoints, not every internal candidate on the fine grid, and is not a global-optimality claim.

## Interpretation and reproducibility

This experiment tests whether nature-inspired proxies perform well under a simplified steady
diffusion model. It does not assume they win. A proxy that loses on Q is a result to report,
not a reason to change the objective after seeing outcomes. Biological optimality would require
anatomical data and justified convection, blood-side and mechanical assumptions.

Source snapshots, exact configurations, geometry seeds, full solve journals and model parameters
are retained. Training cost is additional to evaluation; equal PDE budgets do not imply equal
runtime. Two workers may execute concurrently, so per-job wall time is descriptive. No journal
submission or publication is performed. Costs and incomplete runs are retained in summary.json.
"""
    (root / "report.md").write_text(text, encoding="utf-8")
    _json(
        root / "summary.json",
        {
            "costs": costs,
            "paired_comparisons": comparison_rows,
            "best_endpoint": winner,
            "model": model,
            "bank": {
                "evaluation_distinct_masks": bank["evaluation_distinct_masks"],
                "training_evaluation_overlap": bank["training_evaluation_mask_overlap"],
                "geometry_only_rejections": len(bank["rejections"]),
            },
            "conclusions_scope": "finite 2D geometry proxies, exploratory study",
        },
    )
