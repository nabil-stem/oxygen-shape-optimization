import csv
import hashlib
import json

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from oxygen_shape.cli import app
from oxygen_shape.experiments.config import SlabConfig, SolveConfig, load_config
from oxygen_shape.experiments.runner import run_slab_validation, run_solve


def test_experiment_files_and_repeatability(tmp_path):
    config = SolveConfig.model_validate(
        {"output_root": tmp_path, "geometry": {"nx": 8, "ny": 6, "h": 0.125}}
    )
    first, a = run_solve(config)
    second, b = run_solve(config)
    assert first != second
    assert a["final_Q"] == b["final_Q"]
    assert "git_commit" in a and "git_dirty" in a and "package_versions" in a
    for name in [
        "config.yaml",
        "metrics.csv",
        "initial_geometry.npy",
        "final_geometry.npy",
        "summary.json",
        "run.log",
        "concentration.npy",
        "boundary_faces.csv",
        "figures/fields.png",
    ]:
        assert (first / name).is_file()
    assert json.loads((first / "summary.json").read_text())["Bi"] == 1
    manifest = json.loads((first / "source_manifest.json").read_text())
    for relative, digest in manifest.items():
        assert (
            hashlib.sha256((first / "source_snapshot" / relative).read_bytes()).hexdigest()
            == digest
        )
    np.testing.assert_array_equal(
        np.load(first / "initial_geometry.npy"), np.load(first / "final_geometry.npy")
    )


def test_slab_output_and_error_rows(tmp_path):
    directory, summary = run_slab_validation(SlabConfig(output_root=tmp_path, resolutions=[4, 8]))
    assert summary["max_concentration_error"] < 1e-12
    assert (directory / "figures/slab_errors.png").is_file()
    with (directory / "metrics.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 2


def test_cli_success_and_config_errors(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"output_root": "runs", "geometry": {"nx": 4, "ny": 3}}))
    assert load_config(path, SolveConfig).output_root == tmp_path / "runs"
    runner = CliRunner()
    result = runner.invoke(app, ["solve", "--config", str(path)])
    assert result.exit_code == 0, result.output
    assert "Q=" in result.output
    path.write_text("model:\n  Bi: -1\n")
    result = runner.invoke(app, ["solve", "--config", str(path)])
    assert result.exit_code != 0
    assert "failed" in result.output


@pytest.mark.parametrize("resolutions", [[], [1], [8, 4], [4, 4], [0, 4], [True, 4], [1.5, 4]])
def test_invalid_slab_resolution_config(resolutions):
    with pytest.raises(ValueError):
        SlabConfig(resolutions=resolutions)
