import csv
import json

import numpy as np

from oxygen_shape.experiments.sensitivity_benchmark import run_sensitivity_benchmark
from oxygen_shape.optimization.guided_config import SensitivityBenchmarkConfig


def test_benchmark_includes_probing_costs_maps_and_all_three_controls(tmp_path):
    config = SensitivityBenchmarkConfig(
        output_root=tmp_path,
        evaluation_seeds=[9002],
        shapes=["deep_cavity"],
        solve_budget=9,
        batch_size=4,
        probe_count=2,
        max_attempts_per_solve=100,
        refinement_factors=[2],
    )
    directory, summary = run_sensitivity_benchmark(config)
    assert summary["status"] == "complete"
    assert summary["complete_runs"] == 3
    assert summary["number_of_pde_solves"] == 27
    assert summary["probe_pde_solves_included"] == 4
    assert summary["refinement_pde_solves"] == 6
    assert summary["max_conservation_error"] < 1e-9
    with (directory / "evaluations.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert {r["method"] for r in rows} == {"uniform", "flux", "sensitivity"}
    assert all(int(r["number_of_pde_solves"]) == 9 for r in rows)
    assert (directory / "figures/comparison.png").is_file()
    assert (directory / "source_manifest.json").is_file()
    run = directory / "runs/deep_cavity_sensitivity_9002"
    assert (run / "sensitivity_probes.csv").is_file()
    assert (run / "figures/first_map.png").is_file()
    with np.load(run / "maps/round_001.npz", allow_pickle=False) as data:
        assert len(data["delta_Q"]) == 2
        assert (data["reallocated_area"] > 0).all()
        probabilities = data["probabilities"]
        np.testing.assert_allclose(probabilities.sum(), 1)
        assert probabilities.min() >= config.epsilon_uniform / probabilities.size
    status = json.loads((directory / "status.json").read_text())
    assert status["active_run"] is None
    assert status["status"] == "complete"
    assert "probe" in (directory / "report.md").read_text()
