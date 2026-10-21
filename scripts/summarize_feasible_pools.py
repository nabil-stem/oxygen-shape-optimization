"""Descriptive strata and ranking comparisons; no model selection or new solves."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def summarize(root):
    data = pd.read_csv(root / "evaluations.csv")
    data["stratum"] = np.where(data.start.str.endswith("_canonical"), "canonical", "improved")
    data["fine_relative_gain"] = data.fine_gain_Q / data.fine_initial_Q
    data["fine_eta"] = data.fine_final_Q / (data.Bi * data.robin_perimeter)
    strata = []
    for (stratum, method), group in data.groupby(["stratum", "method"]):
        strata.append(
            {
                "stratum": stratum,
                "method": method,
                "runs": len(group),
                "completed": int((group.status == "complete").sum()),
                "mean_fine_gain_Q": float(group.fine_gain_Q.mean()),
                "mean_fine_final_Q": float(group.fine_final_Q.mean()),
                "mean_fine_eta": float(group.fine_eta.mean()),
                "mean_ideal_Q": float((group.Bi * group.robin_perimeter).mean()),
                "mean_robin_perimeter": float(group.robin_perimeter.mean()),
                "mean_fine_relative_gain": float(group.fine_relative_gain.mean()),
                "mean_multi_choice_fraction": float(group.multi_choice_fraction.mean()),
                "mean_raw_valid_fraction": float(group.raw_valid_fraction.mean()),
                "mean_selected_invalid_fraction": float(group.selected_invalid_fraction.mean()),
                "median_workflow_seconds": float(group.workflow_seconds.median()),
                "median_environment_seconds": float(group.environment_seconds.median()),
                "raw_draws": int(group.proposal_attempts.sum()),
            }
        )
    pd.DataFrame(strata).to_csv(root / "stratum_summary.csv", index=False)
    ranking_pairs = []
    for pool in ("fixed8", "refill2", "refill4"):
        for (start, seed), group in data.groupby(["start", "seed"]):
            flux = group[group.method == f"{pool}_flux"].iloc[0]
            uniform = group[group.method == f"{pool}_uniform"].iloc[0]
            fine_delta = float(flux.fine_final_Q - uniform.fine_final_Q)
            coarse_delta = float(flux.final_Q - uniform.final_Q)
            masks = [
                np.load(root / "runs" / f"{start}__{seed}__{method}" / "final_geometry.npy")
                for method in (flux.method, uniform.method)
            ]
            ranking_pairs.append(
                {
                    "pool_method": pool,
                    "start": start,
                    "seed": int(seed),
                    "stratum": flux.stratum,
                    "both_complete": flux.status == uniform.status == "complete",
                    "fine_delta_Q": fine_delta,
                    "coarse_delta_Q": coarse_delta,
                    "fine_outcome": "win"
                    if fine_delta > 1e-7
                    else "loss"
                    if fine_delta < -1e-7
                    else "tie",
                    "same_final_mask": bool(np.array_equal(*masks)),
                    "workflow_ratio": float(flux.workflow_seconds / uniform.workflow_seconds),
                }
            )
    pd.DataFrame(ranking_pairs).to_csv(root / "ranking_comparisons.csv", index=False)
    quota_pairs = pd.read_csv(root / "paired_comparisons.csv")
    quota_pairs["stratum"] = np.where(
        quota_pairs.start.str.endswith("_canonical"), "canonical", "improved"
    )
    pair_summary = []
    for (stratum, method), group in quota_pairs.groupby(["stratum", "method"]):
        pair_summary.append(
            {
                "stratum": stratum,
                "method": method,
                "pairs": len(group),
                "both_complete": int(group.both_complete.sum()),
                "wins": int((group.outcome == "win").sum()),
                "ties": int((group.outcome == "tie").sum()),
                "losses": int((group.outcome == "loss").sum()),
                "mean_fine_delta_Q": float(group.fine_delta_Q.mean()),
                "median_seconds_ratio": float(group.seconds_ratio.median()),
                "median_raw_draws_ratio": float(group.raw_draws_ratio.median()),
            }
        )
    pd.DataFrame(pair_summary).to_csv(root / "stratified_pairs.csv", index=False)
    result = {
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "role": "posthoc descriptive analysis; no selection or generalization claim",
        "runs": len(data),
        "budget_exhaustions": int((data.status != "complete").sum()),
        "all_positive_fine_gains": bool((data.fine_gain_Q > 0).all()),
        "minimum_fine_gain_Q": float(data.fine_gain_Q.min()),
        "gains_exceed_observed_grid_sensitivity": int(data.gain_exceeds_observed_sensitivity.sum()),
        "raw_draws": int(data.proposal_attempts.sum()),
        "full_geometry_checks": int(data.full_geometry_checks.sum()),
        "workflow_seconds": float(data.workflow_seconds.sum()),
        "environment_seconds": float(data.environment_seconds.sum()),
        "geometry_validation_seconds": float(data.geometry_validation_seconds.sum()),
        "pde_solves": int((data.number_of_pde_solves + data.refinement_pde_solves).sum()),
        "ranking": {
            pool: {
                "wins": sum(
                    r["fine_outcome"] == "win" for r in ranking_pairs if r["pool_method"] == pool
                ),
                "ties": sum(
                    r["fine_outcome"] == "tie" for r in ranking_pairs if r["pool_method"] == pool
                ),
                "losses": sum(
                    r["fine_outcome"] == "loss" for r in ranking_pairs if r["pool_method"] == pool
                ),
                "identical_masks": sum(
                    r["same_final_mask"] for r in ranking_pairs if r["pool_method"] == pool
                ),
            }
            for pool in ("fixed8", "refill2", "refill4")
        },
        "new_pde_solves": 0,
    }
    (root / "descriptive_analysis.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    plot_examples(root, data)
    print(json.dumps(result, indent=2))


def plot_examples(root, data):
    """Descriptive middle seed and first improved start; no best-outcome choice."""
    starts = [
        "deep_cavity_canonical",
        "rectangle_canonical",
        "improved_deep_0",
        "improved_rectangle_0",
    ]
    if not set(starts).issubset(set(data.start)) or 181002 not in set(data.seed):
        return
    methods = ["fixed8_flux", "refill2_flux", "refill4_flux"]
    fig, axes = plt.subplots(4, 3, figsize=(11, 13), constrained_layout=True)
    for i, start in enumerate(starts):
        for j, method in enumerate(methods):
            path = root / "runs" / f"{start}__181002__{method}"
            field = np.load(path / "refinement_4/final/concentration.npy")
            row = data[
                (data.start == start) & (data.seed == 181002) & (data.method == method)
            ].iloc[0]
            ax = axes[i, j]
            image = ax.imshow(field, origin="lower", extent=(0, 1, 0, 1), vmin=0, vmax=1)
            ax.plot([0, 0], [0.45, 0.55], color="red", linewidth=2)
            ax.set_title(f"{start}\n{method}: Q={row.fine_final_Q:.5f}", fontsize=9)
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
    fig.colorbar(image, ax=axes, shrink=0.6, label="Dimensionless concentration u")
    fig.suptitle("Refined endpoint fields, fixed seed 181002; red = protected source")
    fig.savefig(root / "figures/refined_examples.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    summarize(parser.parse_args().run_directory.resolve())
