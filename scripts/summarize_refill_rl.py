"""Publish paired outcomes and all test starts at a prespecified replica/seed."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oxygen_shape.experiments.config import load_config
from oxygen_shape.experiments.pioneer_bank import load_bank
from oxygen_shape.optimization.pioneer_config import PioneerRLConfig
from oxygen_shape.viz.concentration import ConcentrationPanel, plot_concentration_panels


def summarize(root):
    config = load_config(root / "config.yaml", PioneerRLConfig)
    starts, _ = load_bank(root, config)
    e = pd.read_csv(root / "evaluations.csv")
    t = pd.read_csv(root / "training.csv")
    rows = []
    for split in ("validation", "test"):
        learned = e[(e.split == split) & e.method.str.startswith("rl_")]
        for control in ("uniform", "screened_uniform", "perimeter_pool", "flux_pool", "untrained"):
            base = e[(e.split == split) & (e.method == control)]
            pairs = learned.merge(base, on=["start", "seed"], suffixes=("_rl", "_base"))
            delta = pairs.fine_final_Q_rl - pairs.fine_final_Q_base
            rows.append(
                {
                    "split": split,
                    "control": control,
                    "shared_pairs": len(pairs),
                    "wins": int((delta > config.min_improvement).sum()),
                    "ties": int((delta.abs() <= config.min_improvement).sum()),
                    "losses": int((delta < -config.min_improvement).sum()),
                    "mean_fine_delta_Q": float(delta.mean()),
                    "median_fine_delta_Q": float(delta.median()),
                    "median_workflow_ratio": float(
                        (pairs.workflow_seconds_rl / pairs.workflow_seconds_base).median()
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(root / "paired_controls.csv", index=False)
    test = e[(e.split == "test") & e.method.str.startswith("rl_")]
    fine_initial = test.fine_final_Q - test.fine_gain_Q
    result = {
        "training_decisions": int(t.decisions.sum()),
        "training_updates_by_replica": {
            str(k): int(v) for k, v in t.groupby("replica").decisions.sum().items()
        },
        "training_multi_choice_fraction": float(t.multi_choice_decisions.sum() / t.decisions.sum()),
        "training_invalid_fraction": float(t.invalid_decisions.sum() / t.decisions.sum()),
        "test_multi_choice_fraction": float(
            test.multi_choice_decisions.sum() / test.decisions.sum()
        ),
        "test_refined_gain_percent_min": float((100 * test.fine_gain_Q / fine_initial).min()),
        "test_refined_gain_percent_median": float((100 * test.fine_gain_Q / fine_initial).median()),
        "test_refined_gain_percent_max": float((100 * test.fine_gain_Q / fine_initial).max()),
        "distinct_test_learned_final_masks": len(
            {np.load(root / "evaluation" / j / "final_geometry.npy").tobytes() for j in test.job}
        ),
        "paired_outcomes": rows,
        "warning": (
            "Shared starts/seeds/controls across replicas are not independent samples; "
            "no model selection or promotion."
        ),
    }
    (root / "learning_diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    seed = config.test_rollout_seeds[0]
    method = f"rl_{config.replica_seeds[0]}"
    panels = []
    for case in (s for s in starts if s.split == "test"):
        selected = e[(e.start == case.name) & (e.seed == seed)]
        first = root / "evaluation" / selected.iloc[0].job
        initial_Q = selected.iloc[0].fine_final_Q - selected.iloc[0].fine_gain_Q
        panels.append(
            ConcentrationPanel(
                case.geometry,
                np.load(first / "initial_concentration.npy"),
                f"{case.name}\nInitial; Q400={initial_Q:.5f}",
            )
        )
        for label in ("uniform", "flux_pool", method):
            row = selected[selected.method == label].iloc[0]
            path = root / "evaluation" / row.job
            geometry = case.geometry.with_mask(np.load(path / "final_geometry.npy"))
            panels.append(
                ConcentrationPanel(
                    geometry,
                    np.load(path / "concentration.npy"),
                    f"{label}; seed={seed}\nQ400={row.fine_final_Q:.5f}",
                )
            )
    plot_concentration_panels(panels, root / "figures/test_geometries.png", columns=4)
    base = e[(e.split == "test") & (e.method == "flux_pool")]
    pairs = test.merge(base, on=["start", "seed"], suffixes=("_rl", "_base"))
    keys = list(dict.fromkeys(zip(pairs.start, pairs.seed, strict=True)))
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for replica, group in pairs.groupby("method_rl", sort=True):
        x = [keys.index((r.start, r.seed)) for r in group.itertuples()]
        ax.scatter(x, group.fine_final_Q_rl - group.fine_final_Q_base, label=replica, alpha=0.7)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(
        range(len(keys)),
        [f"{name.removeprefix('test_')}\nseed={seed}" for name, seed in keys],
        rotation=35,
        ha="right",
    )
    ax.set_ylabel("Refined Q difference: learned minus frozen flux control")
    ax.set_title("All 40 test comparisons; starts and controls are shared across replicas")
    ax.legend(fontsize=8)
    fig.savefig(root / "figures/paired_flux_comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    summarize(parser.parse_args().run.resolve())
