import hashlib
import json

import numpy as np
import pytest
import yaml

from oxygen_shape.experiments.config import load_config
from oxygen_shape.experiments.frozen_geometry import load_fixed_cases
from oxygen_shape.experiments.prediction_config import FrozenReference, PredictionConfig
from oxygen_shape.experiments.prediction_diagnostic import run_prediction_diagnostic
from oxygen_shape.geometry.baselines import matched_area
from oxygen_shape.geometry.constraints import raster_perimeter
from oxygen_shape.solvers.finite_volume import solve


def make_reference(path, config):
    path.mkdir()
    geometry = matched_area("deep_cavity", config.n, config.target_area).geometry
    np.save(path / "initial_geometry.npy", geometry.inside)
    np.save(path / "final_geometry.npy", geometry.inside)
    original = config.search_config().model_copy(update={"seed": 77}).model_dump(mode="json")
    (path / "config.yaml").write_text(yaml.safe_dump(original))
    summary = {
        "seed": 77,
        "method": "uniform",
        "shape": "deep_cavity",
        "status": "complete",
        "grid_shape": [config.n, config.n],
        "Bi": config.model.Bi,
        "area": geometry.area,
        "robin_perimeter": raster_perimeter(geometry),
        "final_Q": solve(geometry, config.model, config.solver).metrics.physical_flux_Q,
    }
    (path / "summary.json").write_text(json.dumps(summary))
    return FrozenReference(
        run_directory=path,
        expected_seed=77,
        minimum_perimeter_fraction=0.5,
        mask_sha256=hashlib.sha256((path / "final_geometry.npy").read_bytes()).hexdigest(),
    )


def test_verified_inputs_are_copied_and_config_paths_are_relative_to_yaml(tmp_path):
    config = PredictionConfig(shapes=["deep_cavity"], replicates=1)
    reference = make_reference(tmp_path / "saved", config)
    payload = config.model_dump(mode="json")
    payload.update(
        output_root="runs",
        frozen_references={
            "deep_cavity": {**reference.model_dump(mode="json"), "run_directory": "saved"}
        },
        include_initial_controls=True,
    )
    config_file = tmp_path / "diagnostic.yaml"
    config_file.write_text(yaml.safe_dump(payload))
    resolved = load_config(config_file, PredictionConfig)
    assert resolved.output_root == tmp_path / "runs"
    assert resolved.frozen_references["deep_cavity"].run_directory == tmp_path / "saved"
    root = tmp_path / "output"
    root.mkdir()
    cases = load_fixed_cases(resolved, root)
    assert [c.label for c in cases] == ["deep_cavity__initial", "deep_cavity__optimized"]
    assert cases[0].expected_Q is None and cases[1].expected_Q > 0
    assert not cases[1].geometry.inside.flags.writeable
    manifest = json.loads((root / "input_manifest.json").read_text())
    assert manifest[cases[1].label]["file_sha256"]["final_geometry.npy"] == reference.mask_sha256
    for name in ("final_geometry.npy", "initial_geometry.npy", "config.yaml", "summary.json"):
        assert (root / "input_snapshots/deep_cavity" / name).read_bytes() == (
            tmp_path / "saved" / name
        ).read_bytes()


@pytest.mark.parametrize(
    "problem,pattern",
    [
        ("checksum", "checksum"),
        ("physics", "physics"),
        ("incomplete", "identity/status"),
        ("seed", "identity/status"),
        ("collar", "Protected source collar"),
        ("perimeter_fraction", "perimeter fraction"),
        ("initial_mask", "initial geometry"),
        ("dtype", "boolean"),
    ],
)
def test_frozen_source_rejects_unverified_or_inadmissible_inputs(tmp_path, problem, pattern):
    config = PredictionConfig(shapes=["deep_cavity"])
    reference = make_reference(tmp_path / "saved", config)
    if problem == "checksum":
        reference = reference.model_copy(update={"mask_sha256": "0" * 64})
    elif problem == "physics":
        path = reference.run_directory / "config.yaml"
        data = yaml.safe_load(path.read_text())
        data["model"]["Bi"] = 2
        path.write_text(yaml.safe_dump(data))
    elif problem in ("incomplete", "seed"):
        path = reference.run_directory / "summary.json"
        data = json.loads(path.read_text())
        data["status" if problem == "incomplete" else "seed"] = (
            "partial" if problem == "incomplete" else 78
        )
        path.write_text(json.dumps(data))
    elif problem == "perimeter_fraction":
        reference = reference.model_copy(update={"minimum_perimeter_fraction": 0.99})
    elif problem == "initial_mask":
        path = reference.run_directory / "initial_geometry.npy"
        mask = np.load(path)
        mask[45, 5] = False
        np.save(path, mask)
    else:
        path = reference.run_directory / "final_geometry.npy"
        mask = np.load(path)
        if problem == "collar":
            mask[45, 5] = False
            mask[44, 30] = True
        else:
            mask = mask.astype(float)
        np.save(path, mask)
        reference = reference.model_copy(
            update={"mask_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    root = tmp_path / "output"
    root.mkdir()
    config = config.model_copy(update={"frozen_references": {"deep_cavity": reference}})
    with pytest.raises(ValueError, match=pattern):
        load_fixed_cases(config, root)


def test_all_reference_fluxes_checked_before_sampling_any_edits(tmp_path, monkeypatch):
    config = PredictionConfig(
        output_root=tmp_path / "outputs",
        shapes=["deep_cavity"],
        replicates=1,
        calibration_sizes=[2],
        heldout_samples=4,
        max_attempts_per_pool=4,
    )
    reference = make_reference(tmp_path / "saved", config)
    path = reference.run_directory / "summary.json"
    data = json.loads(path.read_text())
    data["final_Q"] += 0.1
    path.write_text(json.dumps(data))
    config = config.model_copy(
        update={"frozen_references": {"deep_cavity": reference}, "include_initial_controls": True}
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("No edit may be sampled before every reference is verified")

    monkeypatch.setattr(
        "oxygen_shape.experiments.prediction_diagnostic.propose_paired_disk", forbidden
    )
    with pytest.raises(ValueError, match="Recomputed frozen-source flux"):
        run_prediction_diagnostic(config)


def test_frozen_case_diagnostic_preserves_reference_inputs_and_labels(tmp_path):
    config = PredictionConfig(
        output_root=tmp_path / "outputs",
        seed=130000,
        shapes=["deep_cavity"],
        replicates=1,
        calibration_sizes=[2],
        heldout_samples=4,
        max_attempts_per_pool=200,
        radius_families=1,
        refinement_factors=[2],
    )
    reference = make_reference(tmp_path / "saved", config)
    original = (reference.run_directory / "final_geometry.npy").read_bytes()
    config = config.model_copy(update={"frozen_references": {"deep_cavity": reference}})
    directory, summary = run_prediction_diagnostic(config)
    assert summary["status"] == "complete"
    assert summary["baseline_pde_solves"] == 1
    assert summary["fixed_cases"][0]["reference_state"] == "optimized"
    assert (reference.run_directory / "final_geometry.npy").read_bytes() == original
    assert "deep_cavity__optimized" in (directory / "metrics.csv").read_text()
    assert (directory / "input_manifest.json").is_file()


@pytest.mark.parametrize(
    "values",
    [
        {"include_initial_controls": True},
        {
            "shapes": ["rectangle"],
            "frozen_references": {
                "deep_cavity": {
                    "run_directory": "dummy",
                    "mask_sha256": "0" * 64,
                    "expected_seed": 8000,
                }
            },
        },
    ],
)
def test_invalid_reference_configuration(values):
    with pytest.raises(ValueError):
        PredictionConfig(**values)
