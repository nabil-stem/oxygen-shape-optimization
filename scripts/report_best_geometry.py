"""Identify and display the best saved refined comparison endpoint at fixed constraints."""

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from audit_feasible_pools import check_field
from pydantic import Field

from oxygen_shape.experiments.config import ExperimentConfig, load_config
from oxygen_shape.experiments.pioneer_bank import load_bank
from oxygen_shape.experiments.reporting import create_run, provenance
from oxygen_shape.geometry.raster import refine_geometry
from oxygen_shape.optimization.guided_search import validate_candidate
from oxygen_shape.optimization.pioneer_config import PioneerRLConfig
from oxygen_shape.viz.concentration import ConcentrationPanel, _draw, concentration_norm


class BestGeometryConfig(ExperimentConfig):
    seed: int = 270000
    n: int = 100
    target_area: float = 0.24
    perimeter_limit: float = 4
    minimum_feature_width: float = 0.04
    protected_length: float = 0.12
    comparison_files: list[Path]
    excluded_files: list[Path] = Field(default_factory=list)


def main():
    repository = Path(__file__).resolve().parents[1]
    files, excluded, frames = [], [], []
    for file in sorted((repository / "experiments").glob("*/evaluations.csv")):
        config = yaml.safe_load((file.parent / "config.yaml").read_text())
        matched = (
            config.get("model", {}).get("Bi") == 1
            and config.get("n") == 100
            and config.get("target_area") == 0.24
            and config.get("perimeter_limit") == 4
            and config.get("minimum_feature_width") == 0.04
            and config.get("protected_length") == 0.12
        )
        data = pd.read_csv(file, float_precision="round_trip")
        if not matched or "fine_final_Q" not in data:
            excluded.append(file)
            continue
        if "status" in data:
            data = data[data.status == "complete"]
        data = data[np.isfinite(data.fine_final_Q)].copy()
        data["comparison_file"] = str(file)
        data["run_root"] = str(file.parent)
        frames.append(data)
        files.append(file)
    all_rows = pd.concat(frames, ignore_index=True).sort_values("fine_final_Q", ascending=False)
    best = all_rows.iloc[0]
    source = Path(best.run_root)
    cfg = load_config(source / "config.yaml", PioneerRLConfig)
    starts, _ = load_bank(source, cfg)
    initial = next(s.geometry for s in starts if s.name == best.start)
    job = source / "evaluation" / best.job
    winner = initial.with_mask(np.load(job / "final_geometry.npy", allow_pickle=False))
    assert not validate_candidate(winner, initial, cfg)
    geometry = refine_geometry(winner, 4)
    field = np.load(job / "refinement_4/final/concentration.npy", allow_pickle=False)
    Q, error = check_field(geometry, field, cfg.model.Bi)
    assert abs(Q - best.fine_final_Q) < 1e-12
    refinements = pd.read_csv(job / "refinement.csv", float_precision="round_trip")
    initial_fine = refine_geometry(initial, 4)
    initial_field = np.load(job / "refinement_4/initial/concentration.npy", allow_pickle=False)
    initial_Q, _ = check_field(initial_fine, initial_field, 1)
    perimeter = float(best.robin_perimeter)
    efficiency = Q / perimeter
    config = BestGeometryConfig(
        output_root=repository / "experiments", comparison_files=files, excluded_files=excluded
    )
    root = create_run(config, "best_geometry_review")
    all_rows.to_csv(root / "comparison_ranking.csv", index=False)
    np.save(root / "best_geometry_n100.npy", winner.inside)
    np.save(root / "best_geometry_n400.npy", geometry.inside)
    np.save(root / "concentration_n400.npy", field)
    input_paths = [
        *files,
        job / "final_geometry.npy",
        job / "refinement.csv",
        job / "refinement_4/final/concentration.npy",
        job / "refinement_4/initial/concentration.npy",
        source / "config.yaml",
    ]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths}
    (root / "input_manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")
    summary = {
        **provenance(),
        "status": "complete",
        "Bi": 1,
        "grid_shape": [400, 400],
        "solver": "finite_volume",
        "solver_tolerance": None,
        "diagnostic_tolerances": cfg.solver.model_dump(),
        "seed": config.seed,
        "selection": "Post-hoc maximum Q400 over saved complete matched comparison endpoints",
        "rows_screened": len(all_rows),
        "comparison_files": len(files),
        "source_run": str(source),
        "job": best.job,
        "policy": best.method,
        "rollout_seed": int(best.seed),
        "training_episodes": 24,
        "Q400": Q,
        "Q200": float(refinements.loc[refinements.factor == 2, "final_Q"].iloc[0]),
        "Q100": float(best.final_Q),
        "area": winner.area,
        "robin_perimeter": perimeter,
        "eta400": efficiency,
        "initial_Q400": initial_Q,
        "gain_percent_over_start": 100 * (Q / initial_Q - 1),
        "independent_conservation_error": error,
        "source_length": len(winner.source_faces) * winner.h,
        "new_pde_solves": 0,
        "global_optimality_claim": False,
        "scope": (
            "Comparison endpoints only; unrefined training/internal candidates "
            "and unmatched physics are excluded."
        ),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
    norm = concentration_norm("power", 0.35)
    _draw(
        axes[0],
        ConcentrationPanel(initial_fine, initial_field, f"Starting geometry\nQ400={initial_Q:.5f}"),
        norm,
    )
    image = _draw(
        axes[1],
        ConcentrationPanel(
            geometry,
            field,
            f"Highest saved refined endpoint\nQ400={Q:.5f}; "
            f"gain={summary['gain_percent_over_start']:.2f}%",
        ),
        norm,
    )
    mask = np.where(geometry.inside, 1.0, np.nan)
    axes[2].imshow(mask, origin="lower", extent=(0, 1, 0, 1), cmap="Blues", vmin=0, vmax=1)
    axes[2].plot([0, 0], [0.45, 0.55], color="red", linewidth=3)
    axes[2].set(
        title=f"Selected physical geometry\nArea={winner.area:.2f}; "
        f"Robin perimeter={perimeter:.2f}",
        xlim=(0, 1),
        ylim=(0, 1),
        aspect="equal",
    )
    axes[2].set_xticks([0, 0.5, 1])
    axes[2].set_yticks([0, 0.5, 1])
    fig.colorbar(
        image,
        ax=axes[:2],
        shrink=0.75,
        ticks=[0, 0.01, 0.03, 0.1, 0.3, 0.6, 1],
        label="Concentration u; power scale gamma=0.35",
    )
    fig.suptitle(
        "Best observed refined comparison geometry; Bi=1, fixed source u=1 (red)\n"
        "Fixed area, bounded membrane length; no global-optimality claim"
    )
    figure = root / "figures/best_geometry.png"
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    shutil.copy2(figure, repository / "docs/figures/best_geometry.png")
    lines = [
        "# Best observed refined geometry",
        "",
        f"Among {len(all_rows)} saved complete endpoints from {len(files)} comparisons with "
        "the same recorded Bi=1, n=100 search grid, area 0.24, perimeter cap 4, feature "
        "width 0.04 and protected collar 0.12, the highest Q400 is **0.567037493**.",
        "",
        f"Source job: `{best.job}`, policy `{best.method}`, rollout seed {int(best.seed)}.",
        "This is one selected geometry from the earlier 24-episode study, not a claim "
        "that its policy performs best on average. Selection is post-hoc development analysis.",
        "",
        f"Q100={best.final_Q:.9f}; Q200={summary['Q200']:.9f}; Q400={Q:.9f}. "
        f"Area={winner.area:.2f}, Robin perimeter={perimeter:.2f}, source length=0.10, "
        f"wall utilization eta400={100 * efficiency:.2f}%.",
        f"Its own starting geometry has Q400={initial_Q:.9f}; "
        f"gain={summary['gain_percent_over_start']:.2f}%.",
        "",
        "![Starting shape, winning concentration and physical geometry](figures/best_geometry.png)",
        "",
        "Fields are saved n=400 solutions with a shared color scale and "
        "unchanged normalized source one.",
        "The selected mask passes the original geometry constraints; independently recomputed "
        f"conservation error is {error:.3g}. This report performs no new PDE solve.",
        "",
        "The ranking covers saved grid-refined comparison endpoints. It does not certify "
        "a global or continuum optimum, and excludes unrefined training/internal candidates. "
        "Maximizing total Q differs from maximizing membrane utilization eta.",
        "",
        f"Full ranking, copied masks, fields, provenance and input hashes: `{root.name}`.",
    ]
    (repository / "docs/best_geometry.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy2(Path(__file__), root / Path(__file__).name)
    print(json.dumps(summary, indent=2))
    print(f"Output: {root}")


if __name__ == "__main__":
    main()
