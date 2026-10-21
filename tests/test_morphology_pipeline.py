import json

import numpy as np
import pandas as pd
import pytest

from oxygen_shape.experiments.morphology_study import MorphologyStudyConfig, make_jobs, run_study
from oxygen_shape.experiments.morphology_validation import cached_fvm
from oxygen_shape.geometry.polygon_controls import chamber_pslg, polygon_area, rasterize_chamber
from oxygen_shape.geometry.research_families import make_research_geometry
from oxygen_shape.optimization.trajectory_search import TrajectoryConfig, run_trajectory


@pytest.mark.parametrize("family", ["rectangle", "hexagon", "koch_1", "koch_2"])
def test_declared_physical_polygon_has_exact_area_and_same_source(family):
    polygon = chamber_pslg(family)
    assert polygon_area(polygon["vertices"]) == pytest.approx(0.24, abs=1e-12)
    source = polygon["segments"][polygon["segment_markers"].ravel() == 1]
    assert np.linalg.norm(
        np.diff(polygon["vertices"][source], axis=1), axis=2
    ).sum() == pytest.approx(0.1)
    assert len(rasterize_chamber(polygon, 100).source_faces) == 10


def test_fixed_study_has_five_seeds_and_charged_long_trajectories():
    config = MorphologyStudyConfig()
    jobs = make_jobs(config)
    assert sum(j["solve_budget"] for j in jobs if j["phase"] == "comparison") == 17415
    assert sum(j["solve_budget"] for j in jobs if j["phase"] == "deep") == 15375
    assert sum(j["solve_budget"] for j in jobs if j["phase"] == "matched_perimeter") == 2925
    assert {"maze", "hilbert_2"}.isdisjoint(config.training_families)
    with pytest.raises(ValueError):
        MorphologyStudyConfig(evaluation_seeds=[1, 2])


def test_field_cache_performs_no_hidden_resolve_and_rejects_changed_physics(tmp_path, monkeypatch):
    import oxygen_shape.experiments.morphology_validation as module

    geometry = make_research_geometry("rectangle").geometry
    config = TrajectoryConfig()
    original = module.solve
    calls = []

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(module, "solve", counted)
    first = cached_fvm(tmp_path / "field.fvm.npz", geometry, 1.0, config.solver)
    other = cached_fvm(tmp_path / "field.fvm.npz", geometry, 1.0, config.solver)
    assert len(calls) == 1
    np.testing.assert_array_equal(first.concentration, other.concentration)
    assert first.metrics.physical_flux_Q == other.metrics.physical_flux_Q
    with pytest.raises(ValueError, match="identity"):
        cached_fvm(tmp_path / "field.fvm.npz", geometry, 2.0, config.solver)


def test_warm_model_is_frozen_when_requested(tmp_path):
    geometry = make_research_geometry("rectangle").geometry
    config = TrajectoryConfig(solve_budget=4)
    trained = run_trajectory(config, geometry, "learned_anneal", tmp_path / "training")
    warm = tmp_path / "training/response_model.npz"
    first = warm.read_bytes()
    result = run_trajectory(
        config, geometry, "learned_frozen_anneal", tmp_path / "evaluation", warm_start=warm
    )
    assert result["new_model_updates"] == 0
    assert result["model_updates"] == trained["model_updates"]
    assert warm.read_bytes() == first


def test_minimal_pipeline_produces_finite_independent_results_and_report(tmp_path):
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
    root = run_study(config)
    endpoints = pd.read_csv(root / "endpoint_metrics.csv")
    assert len(endpoints) == 15
    assert np.isfinite(endpoints["best_FEM_2e-05"]).all()
    assert (root / "figures/best_geometry.png").is_file()
    summary = json.loads((root / "summary.json").read_text())
    assert summary["costs"]["training_PDE"] == 2
    assert "Bi" in (root / "config.yaml").read_text()
    assert (root / "source_manifest.json").is_file()
    assert "not a global" in (root / "report.md").read_text()
