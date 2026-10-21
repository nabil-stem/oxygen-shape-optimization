"""Preserve original plots and apply the verified post-run curve correction."""

import argparse
import difflib
import hashlib
import json
import shutil
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(main, classical, paper):
    main, classical, paper = main.resolve(), classical.resolve(), paper.resolve()
    archive = paper / "post_run_curve_fix"
    before = json.loads((archive / "frozen_source_verification.json").read_text())
    for root in (main, classical):
        if json.loads((root / "status.json").read_text())["status"] != "complete":
            raise RuntimeError("Finish all numerical studies before a presentation correction")
    for name in ("input_manifest.json", "figures/input_manifest.json"):
        for filename, expected in json.loads((paper / name).read_text()).items():
            file = Path(filename)
            if file.suffix == ".csv" and sha(file) != expected:
                raise ValueError(f"Numerical table or history changed: {file}")
    original = archive / "original_figures"
    original.mkdir(exist_ok=True)
    existing_file = archive / "correction.json"
    previous_targets = (
        {row["target"] for row in json.loads(existing_file.read_text())["plot_changes"]}
        if existing_file.exists()
        else set()
    )
    records = []
    for label, root, old_name, corrected_name in (
        ("main_comparison", main, "comparison_curves", "original_comparison_curves"),
        ("main_deep", main, "long_horizon_curves", "long_horizon_curves"),
        ("classical_comparison", classical, "algorithm_curves", "algorithm_curves"),
    ):
        for suffix in (".png", ".pdf"):
            target = root / "figures" / (old_name + suffix)
            source = paper / "figures" / (corrected_name + suffix)
            backup = original / (label + suffix)
            if target.exists() and not backup.exists() and str(target) not in previous_targets:
                shutil.copyfile(target, backup)
            shutil.copyfile(source, target)
            records.append(
                dict(
                    target=str(target),
                    corrected_sha256=sha(target),
                    preserved_original=str(backup) if backup.exists() else None,
                    original_sha256=sha(backup) if backup.exists() else None,
                )
            )
    repository = Path(__file__).resolve().parents[1]
    source_changes = []
    for name in (
        "src/oxygen_shape/experiments/morphology_report.py",
        "scripts/conventional_study.py",
        "src/oxygen_shape/viz/history.py",
    ):
        file = repository / name
        old = archive / (file.name + ".before")
        previous = old.read_text().splitlines(keepends=True) if old.exists() else []
        patch = "".join(
            difflib.unified_diff(
                previous,
                file.read_text().splitlines(keepends=True),
                fromfile=name + " (executed)",
                tofile=name + " (fixed)",
            )
        )
        (archive / (file.name + ".diff")).write_text(patch)
        shutil.copyfile(file, archive / file.name)
        source_changes.append(
            dict(path=name, old_sha256=sha(old) if old.exists() else None, current_sha256=sha(file))
        )
    metadata = dict(
        scope="Post-run presentation correction only; zero new PDE solves",
        reason=before["reason"],
        numerical_tables_and_histories="hash-verified unchanged",
        source_changes=source_changes,
        plot_changes=records,
        historical_replay=(
            "Use each experiment's archived executed sources; current plotting code is corrected"
        ),
    )
    (archive / "correction.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (archive / "README.md").write_text(
        "# Post-run curve-index correction\n\n"
        "All 195 original and 90 classical searches and independent reviews finished before "
        "this correction. The executed reports constructed a pandas Series from another "
        "Series with a new one-based index, shifting the plotted history by one label and "
        "leaving its last point missing. A shared positional converter now keeps both the "
        "initial zero gain and final charged-solve gain. Regression cases include a "
        "non-default DataFrame index.\n\n"
        "Original figures and source diffs are retained here. Source hashes matched the "
        "executed manifests before the correction. Histories, fields, endpoint tables, "
        "model parameters, summary costs and numerical rankings are unchanged. Corrected "
        "curves have been copied into the original reports and the paper figures. No PDE "
        "was solved for this rendering change.\n\n"
        "The original full source snapshots and control_sources remain the exact executed "
        "versions. Identity-guarded historical replay uses those archived sources, while "
        "fresh studies use the corrected plotter. Do not rewrite the old source manifests "
        "to pretend that the new plotter generated the production searches.\n",
        encoding="utf-8",
    )
    print(archive / "correction.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("main", type=Path)
    parser.add_argument("classical", type=Path)
    parser.add_argument("paper", type=Path)
    args = parser.parse_args()
    run(args.main, args.classical, args.paper)
