"""Verified immutable geometry inputs, with portable copies of their provenance."""

import hashlib
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import yaml

from oxygen_shape.experiments.prediction_config import FrozenReference, PredictionConfig
from oxygen_shape.geometry.baselines import matched_area
from oxygen_shape.geometry.constraints import raster_perimeter
from oxygen_shape.geometry.raster import RasterGeometry
from oxygen_shape.optimization.guided_search import validate_candidate


@dataclass(frozen=True)
class FixedCase:
    label: str
    family: str
    state: str
    geometry: RasterGeometry
    expected_Q: float | None


def load_reference(
    config: PredictionConfig,
    family: str,
    reference: FrozenReference,
    destination: Path,
) -> tuple[RasterGeometry, dict]:
    source = reference.run_directory.resolve()
    names = ("final_geometry.npy", "initial_geometry.npy", "config.yaml", "summary.json")
    data = {name: (source / name).read_bytes() for name in names}
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in data.items()}
    if hashes["final_geometry.npy"] != reference.mask_sha256:
        raise ValueError(f"Frozen mask checksum mismatch for {family}")
    original = yaml.safe_load(data["config.yaml"])
    summary = json.loads(data["summary.json"])
    if not isinstance(original, dict) or not isinstance(summary, dict):
        raise ValueError("Frozen source config and summary must be mappings")
    expected = config.search_config(family).model_dump(mode="json")
    keys = (
        "n",
        "model",
        "initial_shape",
        "target_area",
        "perimeter_limit",
        "minimum_feature_width",
        "protected_length",
        "radii",
        "refinement_factors",
        "solver",
    )
    if any(original.get(key) != expected[key] for key in keys):
        raise ValueError(f"Frozen source physics/grid/constraints mismatch for {family}")
    if (
        summary.get("status") != "complete"
        or summary.get("shape") != family
        or summary.get("method") != reference.expected_method
        or summary.get("seed") != reference.expected_seed
        or original.get("seed") != reference.expected_seed
        or summary.get("grid_shape") != [config.n, config.n]
        or summary.get("Bi") != config.model.Bi
    ):
        raise ValueError(f"Frozen source identity/status mismatch for {family}")
    canonical = matched_area(family, config.n, config.target_area).geometry
    initial = np.load(BytesIO(data["initial_geometry.npy"]), allow_pickle=False)
    final = np.load(BytesIO(data["final_geometry.npy"]), allow_pickle=False)
    if initial.dtype != np.bool_ or not np.array_equal(initial, canonical.inside):
        raise ValueError(f"Frozen source initial geometry mismatch for {family}")
    if final.dtype != np.bool_ or final.shape != canonical.inside.shape:
        raise ValueError(f"Frozen source mask must be boolean on the configured grid for {family}")
    geometry = canonical.with_mask(final)
    reason = validate_candidate(geometry, canonical, config.search_config(family))
    if reason:
        raise ValueError(f"Frozen {family}: {reason}")
    perimeter = raster_perimeter(geometry)
    if perimeter + 1e-12 < reference.minimum_perimeter_fraction * config.perimeter_limit:
        raise ValueError(f"Frozen {family} is below the requested perimeter fraction")
    for key, actual in (("area", geometry.area), ("robin_perimeter", perimeter)):
        recorded = summary.get(key)
        if (
            not isinstance(recorded, (float, int))
            or not np.isfinite(recorded)
            or abs(recorded - actual) > 1e-12
        ):
            raise ValueError(f"Frozen source {key} mismatch for {family}")
    if (
        not isinstance(summary.get("final_Q"), (float, int))
        or not np.isfinite(summary["final_Q"])
        or summary["final_Q"] <= 0
    ):
        raise ValueError(f"Frozen source final flux must be positive and finite for {family}")
    destination.mkdir(parents=True)
    for name, content in data.items():
        (destination / name).write_bytes(content)
    return geometry, {
        "source_run": str(source),
        "expected_seed": reference.expected_seed,
        "expected_method": reference.expected_method,
        "file_sha256": hashes,
        "snapshot_directory": destination.as_posix(),
        "recorded_Q": summary["final_Q"],
        "area": geometry.area,
        "robin_perimeter": perimeter,
        "perimeter_fraction": perimeter / config.perimeter_limit,
    }


def load_fixed_cases(config: PredictionConfig, root: Path) -> list[FixedCase]:
    cases, manifest = [], {}
    for label, (family, state) in config.case_specs.items():
        expected_Q = None
        if state == "initial":
            geometry = matched_area(family, config.n, config.target_area).geometry
            metadata = {
                "generator": "matched_area",
                "family": family,
                "mask_sha256": hashlib.sha256(geometry.inside.tobytes()).hexdigest(),
            }
        else:
            geometry, metadata = load_reference(
                config,
                family,
                config.frozen_references[family],
                root / "input_snapshots" / family,
            )
            expected_Q = metadata["recorded_Q"]
            metadata["snapshot_directory"] = f"input_snapshots/{family}"
        reason = validate_candidate(geometry, geometry, config.search_config(family))
        if reason:
            raise ValueError(f"Fixed case {label}: {reason}")
        cases.append(FixedCase(label, family, state, geometry, expected_Q))
        manifest[label] = {"family": family, "state": state, **metadata}
    (root / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return cases
