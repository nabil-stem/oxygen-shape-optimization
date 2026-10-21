import csv
import json

import numpy as np
import pytest

from oxygen_shape.experiments.baselines import load_case, run_baselines
from oxygen_shape.experiments.config import BaselineConfig


@pytest.mark.parametrize(
    "data",
    [
        {"resolutions": [100]},
        {"resolutions": [100, 300]},
        {"resolutions": [30, 60]},
        {"resolutions": [200, 100]},
        {"permeabilities": []},
        {"permeabilities": [1, 1]},
        {"permeabilities": [1, 0.1]},
        {"permeabilities": [0, 1]},
        {"permeabilities": [float("nan")]},
        {"model": {"Bi": 2}},
    ],
)
def test_invalid_comparison_config(data):
    with pytest.raises(ValueError):
        BaselineConfig(**data)


def test_complete_small_sweep_outputs_and_physics(tmp_path):
    config = BaselineConfig(output_root=tmp_path, resolutions=[40, 80], permeabilities=[0.01, 10])
    messages = []
    directory, summary = run_baselines(config, progress=messages.append)
    assert summary["number_of_pde_solves"] == 36
    assert len(messages) == 18
    assert summary["max_conservation_error"] < 1e-9
    with (directory / "metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 36
    for group, count in [("fixed_area", 7), ("fixed_perimeter", 2)]:
        subset = [r for r in rows if r["comparison"] == group and r["n"] == "80"]
        assert len(subset) == 2 * count
        for name in {r["shape"] for r in subset}:
            series = [r for r in subset if r["shape"] == name]
            low, high = series
            assert float(high["physical_flux_Q"]) > float(low["physical_flux_Q"])
            assert float(high["efficiency_eta"]) < float(low["efficiency_eta"])
            assert float(high["ideal_flux_Q"]) == pytest.approx(10 * float(high["robin_perimeter"]))
    for file in [
        "summary.json",
        "report.md",
        "config.yaml",
        "refinement.csv",
        "rankings.csv",
        "source_manifest.json",
        "initial_geometry.npy",
        "final_geometry.npy",
        "run.log",
        "figures/fixed_area_sweep.png",
        "figures/fixed_area_Bi_10.png",
    ]:
        assert (directory / file).is_file()
    assert json.loads((directory / "summary.json").read_text())["minimum_neck"] is None
    assert "orientation bias" in (directory / "report.md").read_text()
    assert "Area varies" in (directory / "report.md").read_text()
    with (directory / "refinement.csv").open() as stream:
        refinement = list(csv.DictReader(stream))
    assert len(refinement) == 18
    assert all(0 < float(row["interior_comparison_coverage"]) <= 1 for row in refinement)
    # Loading plot data must preserve the numerical output without another solve.
    first = rows[0]
    numeric = {
        **first,
        **{
            key: float(first[key])
            for key in (
                "physical_flux_Q",
                "source_influx",
                "ideal_flux_Q",
                "efficiency_eta",
                "area",
                "robin_perimeter",
                "source_length",
                "conservation_error",
                "residual_inf",
            )
        },
    }
    saved = load_case(directory / first["case_path"], numeric)
    assert saved.metrics.physical_flux_Q == float(first["physical_flux_Q"])
    np.testing.assert_array_equal(
        saved.concentration, np.load(directory / first["case_path"] / "concentration.npy")
    )


def test_membrane_budget_failure_is_logged(tmp_path):
    config = BaselineConfig(
        output_root=tmp_path, resolutions=[40, 80], permeabilities=[1], perimeter_budget=1
    )
    with pytest.raises(ValueError, match="budget"):
        run_baselines(config)
    run = next(tmp_path.iterdir())
    assert "FAILED" in (run / "run.log").read_text()
    assert not (run / "summary.json").exists()
