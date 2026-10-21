import json

import pandas as pd
import pytest

from oxygen_shape.experiments.morphology_study import MorphologyStudyConfig, run_study
from scripts.conventional_study import run


def test_classical_extension_imports_frozen_controls_charges_all_solves_and_resumes(tmp_path):
    pytest.importorskip("triangle")
    config = MorphologyStudyConfig(
        output_root=tmp_path,
        families=["rectangle"],
        training_families=["rectangle"],
        methods=["flux_greedy"],
        perimeter_control_families=["rectangle"],
        deep_family="rectangle",
        solve_budget=2,
        deep_solve_budget=2,
        training_solve_budget=2,
        perimeter_control_budget=2,
        counterfactual_pools=1,
        workers=1,
        raw_draw_budget=2000,
    )
    reference = run_study(config)
    root = run(reference, workers=1)
    summary = json.loads((root / "summary.json").read_text())
    assert summary["new_search_PDE"] == 20
    assert summary["new_FVM_validation"] == 40
    assert summary["new_FEM_validation"] == 40
    assert summary["imported_comparison_endpoints"] == 5
    assert summary["added_endpoints"] == 10
    assert summary["audit"]["candidate_fields"] == 20
    merged = pd.read_csv(root / "endpoint_metrics.csv")
    assert len(merged) == 15
    assert set(merged.method) == {"flux_greedy", "uniform_greedy", "pool_greedy"}
    assert (root / "postprocess/energy_bounds/summary.json").exists()
    assert (root / "figures/algorithm_curves.pdf").exists()
    before = (root / "summary.json").read_bytes()
    assert run(reference, workers=1, resume=root) == root
    assert (root / "summary.json").read_bytes() == before
