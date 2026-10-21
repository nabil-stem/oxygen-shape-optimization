"""Describe prediction errors on observed selected responses, without new labels."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def run(root):
    if json.loads((root / "status.json").read_text())["phase"] != "finished":
        raise RuntimeError("Finish the frozen study before its post-hoc diagnosis")
    output = root / "postprocess/response_diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    jobs = json.loads((root / "jobs.json").read_text())
    observed, hashes = [], {}
    for job in jobs:
        if job["method"] != "learned_anneal":
            continue
        file = root / job["path"] / "metrics.csv"
        data = pd.read_csv(file, float_precision="round_trip")
        data = data[data.pde_solves > 1].copy()
        hashes[file.relative_to(root).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
        for name in ("phase", "family", "seed", "path"):
            data[name] = job[name]
        data["absolute_error"] = abs(data.prediction - data.target)
        data["zero_absolute_error"] = abs(data.target)
        data["squared_error"] = (data.prediction - data.target) ** 2
        data["zero_squared_error"] = data.target**2
        observed.append(data)
    combined = pd.concat(observed, ignore_index=True)
    if combined[["prediction", "target"]].isna().any().any():
        raise ValueError("Observed prediction/target missing")
    rows = []
    for phase, group in combined.groupby("phase"):
        rows.append(
            dict(
                phase=phase,
                trajectories=group.path.nunique(),
                selected_responses=len(group),
                prediction_MAE=float(group.absolute_error.mean()),
                prediction_RMSE=float(np.sqrt(group.squared_error.mean())),
                zero_prediction_MAE=float(group.zero_absolute_error.mean()),
                zero_prediction_RMSE=float(np.sqrt(group.zero_squared_error.mean())),
                positive_prediction_fraction=float((group.prediction > 0).mean()),
                positive_observed_fraction=float((group.target > 0).mean()),
                mean_bias=float((group.prediction - group.target).mean()),
            )
        )
    combined.to_csv(output / "observed_selected_responses.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "phase_errors.csv", index=False)
    per_job = (
        combined.groupby(["phase", "family", "seed", "path"])[
            ["absolute_error", "zero_absolute_error", "squared_error", "zero_squared_error"]
        ]
        .mean()
        .reset_index()
    )
    per_job.to_csv(output / "per_trajectory_errors.csv", index=False)
    summary = dict(
        analysis="Post-hoc diagnosis of observed selected labels only",
        units="100 times delta Q divided by the trajectory initial Q",
        selected_responses=len(combined),
        trajectories=combined.path.nunique(),
        additional_PDE_solves=0,
        phase_errors=rows,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    table = "\n".join(
        f"| {r['phase']} | {r['selected_responses']} | {r['prediction_MAE']:.6f} "
        f"| {r['zero_prediction_MAE']:.6f} | {r['mean_bias']:+.6f} |"
        for r in rows
    )
    (output / "report.md").write_text(
        "# Selected-response prediction diagnosis\n\n"
        "Post-hoc descriptive analysis. Predictions precede the corresponding update "
        "and exclude the leverage exploration bonus. Targets are 100*deltaQ/Q_initial, "
        "so errors are in percent-of-initial-flux units. All selected evaluated moves, "
        "including rejected moves, are retained.\n\n"
        "| Phase | Observed labels | Prediction MAE | Constant-zero MAE | Mean bias |\n"
        "|---|---:|---:|---:|---:|\n" + table + "\n\n"
        "The zero predictor is a calibration reference, not a competing search policy. "
        "Small changes can dominate absolute prediction error while ranking still matters. "
        "These selected-label distributions depend on the evolving policy and geometry; "
        "they do not give regret on unselected candidates, independent generalization error "
        "or a causal diagnosis of the method's endpoint performance. Their pooled statistics "
        "are descriptive; per-trajectory values are supplied separately. No new labels, "
        "training or PDE solves are introduced.\n",
        encoding="utf-8",
    )
    script = Path(__file__).resolve()
    shutil.copyfile(script, output / script.name)
    hashes[script.name] = hashlib.sha256(script.read_bytes()).hexdigest()
    (output / "input_manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    run(parser.parse_args().root.resolve())
