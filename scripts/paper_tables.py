"""Collect completed evidence for a human paper review, without any PDE solve."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from oxygen_shape.experiments.evidence_paths import EvidencePaths


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def markdown_table(frame, columns):
    header = "| " + " | ".join(columns) + " |\n"
    header += "|" + "---|" * len(columns) + "\n"
    rows = []
    for row in frame[columns].to_dict("records"):
        rows.append("| " + " | ".join(str(row[key]) for key in columns) + " |")
    return header + "\n".join(rows) + "\n"


def footprint(mask):
    """Exact occupied box and diameter of the union of frozen n100 cells."""
    if mask.shape != (100, 100) or mask.dtype != np.bool_ or not mask.any():
        raise ValueError("Expected a nonempty frozen n100 Boolean geometry")
    rows, cols = np.nonzero(mask)
    origins = np.c_[cols, rows]
    corners = (origins[:, None, :] + np.array([[0, 0], [1, 0], [1, 1], [0, 1]])).reshape(-1, 2)
    points = corners[ConvexHull(corners).vertices] / 100
    diameter = np.sqrt(np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1)).max()
    return dict(
        centroid_x=float(np.mean(cols + 0.5) / 100),
        centroid_y=float(np.mean(rows + 0.5) / 100),
        occupied_width=float((cols.max() - cols.min() + 1) / 100),
        occupied_height=float((rows.max() - rows.min() + 1) / 100),
        bounding_diameter=float(diameter),
    )


def run(main, classical, output):
    main, classical, output = main.resolve(), classical.resolve(), output.resolve()
    if output in (main, classical):
        raise ValueError("Use a separate paper evidence directory")
    for root in (main, classical):
        state = read_json(root / "status.json")
        if state.get("phase") != "finished" or state.get("status") not in (
            "complete",
            "finished_with_budget_truncations",
        ):
            raise RuntimeError(f"Study is unfinished: {root}")
    if read_json(main / "postprocess/review/status.json").get("status") != "complete":
        raise RuntimeError("Complete the independent supplemental review first")
    paths = EvidencePaths(main, classical)
    if paths.resolve(read_json(classical / "extension.json")["source"], classical) != main:
        raise ValueError("Classical controls refer to a different original study")
    inputs = {}

    def record(file):
        file = file.resolve()
        inputs[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()

    def csv(root, name):
        file = root / name
        record(file)
        return pd.read_csv(file, float_precision="round_trip")

    def js(root, name):
        file = root / name
        record(file)
        return read_json(file)

    endpoints = []
    ranges = []
    for root, selector in ((main, None), (classical, "new")):
        data = csv(root, "new_endpoint_metrics.csv" if selector else "endpoint_metrics.csv")
        data["study"] = root.name
        data["resolved_path"] = [str((root / path).resolve()) for path in data.path]
        endpoints.append(data)
        bounds = csv(root, "postprocess/energy_bounds/endpoint_bounds.csv")
        bounds["resolved_path"] = [str((root / path).resolve()) for path in bounds.path]
        ranges.append(bounds)
    endpoints = pd.concat(endpoints, ignore_index=True)
    if len(endpoints) != 285 or endpoints.resolved_path.duplicated().any():
        raise ValueError("Expected 195 original and 90 new unique returned endpoints")
    ranges = pd.concat(ranges, ignore_index=True).drop_duplicates("resolved_path")
    if set(endpoints.resolved_path) != set(ranges.resolved_path):
        raise ValueError("Endpoint and energy-range identities differ")
    endpoints = endpoints.merge(
        ranges[
            [
                "resolved_path",
                "best_lower",
                "best_upper",
                "resolved_improvement",
                "minimum_relative_gain_percent",
                "mask_sha256",
            ]
        ],
        on="resolved_path",
        validate="one_to_one",
    )
    resources = []
    for row in endpoints.to_dict("records"):
        directory = Path(row["resolved_path"])
        change = {"resolved_path": row["resolved_path"]}
        for label, name in (("initial", "initial_geometry.npy"), ("best", "final_geometry.npy")):
            file = directory / name
            record(file)
            change.update(
                {f"{label}_{k}": v for k, v in footprint(np.load(file, allow_pickle=False)).items()}
            )
        resources.append(change)
    endpoints = endpoints.merge(pd.DataFrame(resources), on="resolved_path", validate="one_to_one")

    def clean_row(row):
        return {key: None if pd.isna(value) else value for key, value in row.items()}

    winner = clean_row(endpoints.loc[endpoints["best_FEM_2e-05"].idxmax()].to_dict())
    lower_winner = clean_row(endpoints.loc[endpoints.best_lower.idxmax()].to_dict())
    contenders = endpoints[endpoints.best_upper >= lower_winner["best_lower"]]
    comparison = csv(classical, "paired_method_comparisons.csv")
    paired = csv(classical, "postprocess/energy_bounds/paired_bounds.csv")
    counts = paired.groupby(["method", "outcome"]).size().unstack(fill_value=0)
    comparison = comparison.merge(counts, on="method", validate="one_to_one")
    for name in ("resolved_win", "resolved_loss", "overlap"):
        if name not in comparison:
            comparison[name] = 0
    matched = endpoints[endpoints.phase == "matched_perimeter"]
    if (abs(matched.best_perimeter - matched.initial_perimeter) > 1e-10).any():
        raise ArithmeticError("A fixed-perimeter endpoint changed membrane length")
    phase_rows = []
    for (phase, method), group in endpoints.groupby(["phase", "method"]):
        phase_rows.append(
            dict(
                phase=phase,
                method=method,
                endpoints=len(group),
                resolved_improvements=int(group.resolved_improvement.sum()),
                median_FEM_gain_percent=float(group.FEM_gain_percent.median()),
                median_membrane_change_percent=float(group.membrane_change_percent.median()),
                search_PDE=int(group.pde_solves.sum()),
                median_raw_draws=float(group.raw_draws.median()),
                median_full_checks=float(group.checks.median()),
                median_wall_seconds=float(group.wall_seconds.median()),
                incomplete=int((group.status != "complete").sum()),
            )
        )
    main_summary = js(main, "summary.json")
    classical_summary = js(classical, "summary.json")
    audit = js(main, "postprocess/journal_audit/summary.json")
    collar = js(main, "postprocess/collar_ablation/summary.json")
    corner = js(main, "postprocess/source_corner/summary.json")
    permeability = js(main, "postprocess/permeability_review/summary.json")
    collar_rankings = csv(main, "postprocess/collar_ablation/initial_family_rankings.csv")
    for root in (main, classical):
        js(root, "postprocess/energy_bounds/summary.json")
        for name in ("source_manifest.json", "config.yaml", "environment.json"):
            record(root / name)
    # Check the evidence consumed by the completed numerical reviews. Shared
    # imported fields are read only once after resolving their physical paths.
    expected_inputs = {}
    for root, relative in (
        (main, "postprocess/energy_bounds/input_manifest.json"),
        (classical, "postprocess/energy_bounds/input_manifest.json"),
        (main, "postprocess/collar_ablation/input_manifest.json"),
        (classical, "reference_inputs.json"),
    ):
        for name, digest in js(root, relative).items():
            path = paths.resolve(name, root)
            if path in expected_inputs and expected_inputs[path] != digest:
                raise ValueError(f"Conflicting input identities: {path}")
            expected_inputs[path] = digest
    for path, digest in expected_inputs.items():
        record(path)
        if inputs[str(path)] != digest:
            raise ValueError(f"Reviewed numerical input changed: {path}")
    costs = main_summary["costs"]
    original_total = sum(
        costs[key]
        for key in (
            "training_PDE",
            "search_PDE",
            "cached_validation_and_counterfactual_FVM_PDE",
            "independent_FEM_solves",
        )
    )
    new_total = sum(
        classical_summary[key]
        for key in (
            "new_search_PDE",
            "new_FVM_validation",
            "new_FEM_validation",
            "new_training_PDE",
        )
    )
    supplemental = (
        collar["additional_FVM_solves"]
        + collar["additional_FEM_solves"]
        + corner["additional_FVM_solves"]
    )
    evidence = dict(
        scope="Finite raster development benchmark; prepared for human review, not submitted",
        unique_returned_endpoints=len(endpoints),
        highest_FEM_endpoint=winner,
        highest_lower_endpoint=lower_winner,
        contender_endpoints=len(contenders),
        contender_distinct_masks=contenders.mask_sha256.nunique(),
        resolved_improvements=int(endpoints.resolved_improvement.sum()),
        original_solver_calls=original_total,
        classical_new_solver_calls=new_total,
        supplemental_solver_calls=supplemental,
        total_committed_solver_calls=original_total + new_total + supplemental,
        uncommitted_classical_extra_lower=classical_summary["uncommitted_extra_PDE_lower"],
        uncommitted_classical_extra_upper=classical_summary["uncommitted_extra_PDE_upper"],
        original_audited_fields=audit["recorded_PDE_solves_reconstructed"],
        new_audited_fields=classical_summary["audit"]["candidate_fields"],
        additional_PDE_for_this_collection=0,
        permeability_checked_curves=permeability["fixed_geometry_grid_curves"],
    )
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "catalog_metrics.csv",
        "physical_polygon_checks.csv",
        "counterfactual_pools.csv",
        "counterfactual_candidates.csv",
        "training_metrics.csv",
    ):
        csv(main, name).to_csv(output / name, index=False)
    for source, destination in (
        (main / "postprocess/source_corner/refinement.csv", "source_corner_refinement.csv"),
        (main / "postprocess/source_corner/analytical_flux.csv", "source_corner_series.csv"),
        (main / "postprocess/collar_ablation/metrics.csv", "collar_metrics.csv"),
        (main / "postprocess/collar_ablation/fem_metrics.csv", "collar_fem_metrics.csv"),
        (main / "postprocess/response_diagnostics/phase_errors.csv", "response_phase_errors.csv"),
    ):
        record(source)
        shutil.copyfile(source, output / destination)
    for filename, frame in (
        ("all_returned_endpoints.csv", endpoints),
        ("five_method_comparisons.csv", comparison),
        ("phase_summary.csv", pd.DataFrame(phase_rows)),
        ("contenders.csv", contenders),
        ("collar_initial_rankings.csv", collar_rankings),
    ):
        frame.to_csv(output / filename, index=False)
    (output / "summary.json").write_text(
        json.dumps(evidence, indent=2, default=int, allow_nan=False) + "\n"
    )
    body = (
        "# Paper evidence tables\n\n"
        f"{len(endpoints)} unique returned endpoints; {len(contenders)} endpoints / "
        f"{contenders.mask_sha256.nunique()} distinct masks overlap the highest practical "
        "lower flux estimate. The highest fine-FEM value does not by itself prove a unique "
        "winner. All ranges concern the same staircase, with practical floating guards.\n\n"
        "## Equal-budget five-method comparison\n\n"
        "Known development starts, 45 pairs, 129 coarse solves per trajectory. Added controls "
        "were specified after the original results. Win counts use the descriptive 1e-7 "
        "threshold; resolved outcomes compare complementary energy ranges.\n\n"
        + markdown_table(
            comparison,
            [
                "method",
                "wins",
                "ties",
                "losses",
                "mean_delta_Q400",
                "mean_delta_FEM",
                "resolved_win",
                "overlap",
                "resolved_loss",
            ],
        )
        + "\n## Separate phases\n\n"
        + markdown_table(
            pd.DataFrame(phase_rows),
            [
                "phase",
                "method",
                "endpoints",
                "resolved_improvements",
                "median_FEM_gain_percent",
                "median_membrane_change_percent",
                "incomplete",
            ],
        )
        + "\nDifferent phase budgets and actual membrane lengths preclude attributing "
        "cross-phase winners to morphology or method alone. Fixed perimeter is within "
        "each start, not a shared length across families.\n\n"
        f"Committed costs: original study {original_total:,}, additional classical controls "
        f"{new_total:,}, collar/strip checks {supplemental:,}; total "
        f"{original_total + new_total + supplemental:,} solver calls. Warm training is included "
        "once; imported endpoints are not charged twice. Journal replay and this collection "
        "add no PDE solves.\n"
    )
    (output / "report.md").write_text(body, encoding="utf-8")
    source = Path(__file__).resolve()
    record(source)
    shutil.copyfile(source, output / source.name)
    (output / "input_manifest.json").write_text(json.dumps(inputs, indent=2) + "\n")
    # A separate zero-solve geometry-review view can consume the existing plotter.
    view = output / "geometry_review"
    view.mkdir(exist_ok=True)
    selected = dict(winner, path=Path(os.path.relpath(winner["resolved_path"], view)).as_posix())
    (view / "best_endpoint.json").write_text(json.dumps(selected, indent=2, allow_nan=False) + "\n")
    (view / "status.json").write_text(
        json.dumps(
            dict(phase="finished", status="complete", role="View of checked existing endpoints")
        )
        + "\n"
    )
    for name in ("config.yaml", "source_manifest.json"):
        shutil.copyfile(main / name, view / name)
    print(output / "report.md")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("main", type=Path)
    parser.add_argument("classical", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    run(arguments.main, arguments.classical, arguments.output)
