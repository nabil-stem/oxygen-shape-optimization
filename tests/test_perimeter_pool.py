import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from oxygen_shape.experiments.feasibility import run_feasibility_study
from oxygen_shape.experiments.feasibility_config import FeasibilityConfig
from oxygen_shape.experiments.prediction_config import FrozenReference
from oxygen_shape.geometry.baselines import matched_area
from oxygen_shape.geometry.constraints import raster_perimeter
from oxygen_shape.geometry.disk_edits import DiskEdit, prepare_boundary
from oxygen_shape.optimization.guided_config import GuidedConfig
from oxygen_shape.optimization.guided_search import validate_candidate
from oxygen_shape.optimization.perimeter_pool import (
    cheap_screen,
    choose_pool,
    draw_pool,
    run_pool_search,
)
from oxygen_shape.solvers.finite_volume import solve


def test_cheap_screen_does_not_replace_topology_or_feature_validation():
    cfg = GuidedConfig(initial_shape="rectangle")
    g = matched_area("rectangle", 100, 0.24).geometry
    mask = g.inside.copy()
    mask[90, 90], mask[50, 40] = True, False
    edit = DiskEdit(g.with_mask(mask), {})
    assert cheap_screen(edit, g, cfg)[0] == ""
    assert validate_candidate(edit.geometry, g, cfg)
    mask[:, 10] = ~mask[:, 10]
    assert "collar" in cheap_screen(DiskEdit(g.with_mask(mask), {}), g, cfg)[0]


def test_uniform_branch_keeps_invalid_support_and_rank_uses_only_geometry():
    cfg = GuidedConfig()
    g = matched_area("deep_cavity", 100, 0.24).geometry
    prepared = prepare_boundary(g, 0.12)
    edits, _ = draw_pool(g, np.random.default_rng(3), cfg, prepared, 32)
    edits.insert(0, DiskEdit(None, {}, "Invalid draw"))
    index, mode, _ = choose_pool(edits, g, cfg, np.random.default_rng(9), 1)
    assert index == 0 and mode == "uniform"
    index, mode, screens = choose_pool(edits, g, cfg, np.random.default_rng(9), 0.0001)
    assert mode == "perimeter" and not screens[index]["screen_rejection"]
    assert screens[index]["perimeter"] == min(
        s["perimeter"] for s in screens if not s["screen_rejection"]
    )
    index, mode, _ = choose_pool([edits[0]], g, cfg, np.random.default_rng(9), 0.0001)
    assert index == 0 and mode == "fallback"
    with pytest.raises(ValueError):
        choose_pool(edits, g, cfg, np.random.default_rng(1), 0)


def test_raw_draw_seed_replays_independent_of_elapsed_time():
    cfg = GuidedConfig()
    g = matched_area("deep_cavity", 100, 0.24).geometry
    prepared = prepare_boundary(g, 0.12)
    first, rows = draw_pool(g, np.random.default_rng(8600), cfg, prepared, 8)
    second, again = draw_pool(g, np.random.default_rng(8600), cfg, prepared, 8)
    assert [r["proposal_seed"] for r in rows] == [r["proposal_seed"] for r in again]
    for a, b in zip(first, second, strict=True):
        assert a.parameters == b.parameters and a.rejection == b.rejection
        if a.geometry is not None:
            np.testing.assert_array_equal(a.geometry.inside, b.geometry.inside)


@pytest.mark.parametrize("method", ["perimeter_pool", "screened_uniform"])
def test_invalid_pools_charge_every_draw_and_stop_without_candidate_solve(monkeypatch, method):
    import oxygen_shape.optimization.perimeter_pool as module

    cfg = GuidedConfig(solve_budget=5)
    g = matched_area("deep_cavity", 100, 0.24).geometry
    actual_solve, calls = module.solve, []

    def count_solve(*args):
        calls.append(1)
        return actual_solve(*args)

    def invalid_pool(geometry, rng, config, prepared, size):
        return [DiskEdit(None, {}, "Invalid") for _ in range(size)], [
            {"slot": i} for i in range(size)
        ]

    monkeypatch.setattr(module, "solve", count_solve)
    monkeypatch.setattr(module, "draw_pool", invalid_pool)
    result = run_pool_search(
        cfg, g, method=method, pool_size=8, epsilon_uniform=0.25, raw_budget=10
    )
    assert result.status == "incomplete"
    assert len(calls) == result.search.number_of_pde_solves == 1
    assert result.search.attempts == len(result.search.proposals) == 10
    assert result.selected_checks == 2
    assert sum(r["selected"] for r in result.search.proposals) == 2


def test_global_flux_rejection_and_duplicate_cache_are_independent_of_perimeter(monkeypatch):
    import oxygen_shape.optimization.perimeter_pool as module

    cfg = GuidedConfig(solve_budget=4)
    g = matched_area("deep_cavity", 100, 0.24).geometry
    baseline = solve(g, cfg.model, cfg.solver)
    calls = []

    def lower_flux(*args):
        calls.append(1)
        return (
            baseline
            if len(calls) == 1
            else replace(
                baseline,
                metrics=replace(
                    baseline.metrics, physical_flux_Q=baseline.metrics.physical_flux_Q / 2
                ),
            )
        )

    def same_pool(geometry, rng, config, prepared, size):
        return [DiskEdit(g, {}) for _ in range(size)], [{"slot": i} for i in range(size)]

    monkeypatch.setattr(module, "solve", lower_flux)
    monkeypatch.setattr(module, "draw_pool", same_pool)
    result = run_pool_search(
        cfg, g, method="uniform", pool_size=8, epsilon_uniform=0.25, raw_budget=3
    )
    assert len(calls) == result.search.number_of_pde_solves == 2
    assert not any(r["accepted"] for r in result.search.history)
    assert result.search.final_geometry is g
    assert sum("Duplicate" in r["rejection"] for r in result.search.proposals) == 2


@pytest.mark.parametrize("responses", [False, True])
def test_small_study_keeps_frozen_inputs_and_exact_solve_accounting(
    tmp_path, monkeypatch, responses
):
    shape_cfg = GuidedConfig(refinement_factors=[2], seed=8000)
    g = matched_area("deep_cavity", 100, 0.24).geometry
    source = tmp_path / "reference"
    source.mkdir()
    np.save(source / "initial_geometry.npy", g.inside)
    np.save(source / "final_geometry.npy", g.inside)
    (source / "config.yaml").write_text(yaml.safe_dump(shape_cfg.model_dump(mode="json")))
    baseline = solve(g, shape_cfg.model, shape_cfg.solver)
    (source / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "seed": 8000,
                "shape": "deep_cavity",
                "method": "uniform",
                "grid_shape": [100, 100],
                "Bi": 1,
                "area": g.area,
                "robin_perimeter": raster_perimeter(g),
                "final_Q": baseline.metrics.physical_flux_Q,
            }
        )
    )
    sha = hashlib.sha256((source / "final_geometry.npy").read_bytes()).hexdigest()
    methods = (
        ["uniform", "perimeter_pool", "screened_uniform"]
        if responses
        else ["uniform", "perimeter_pool"]
    )
    if responses:
        import oxygen_shape.experiments.feasibility as module

        actual_choose, actual_validate = module.choose_pool, module.validate_candidate
        order = {"validated": False}

        def matched_draw(geometry, rng, config, prepared, size):
            order["validated"] = False
            # A shared valid selection tests cost sharing and unavailable contrasts.
            return [DiskEdit(geometry, {"radius": 0.06}) for _ in range(size)], [
                {"slot": i, "addition_face": i, "donor_face": i, "radius": 0.06}
                for i in range(size)
            ]

        def check_order(*args, **kwargs):
            assert not order["validated"], "Selection must precede full validation"
            return actual_choose(*args, **kwargs)

        def validate(*args):
            order["validated"] = True
            return actual_validate(*args)

        monkeypatch.setattr(module, "draw_pool", matched_draw)
        monkeypatch.setattr(module, "choose_pool", check_order)
        monkeypatch.setattr(module, "validate_candidate", validate)
    cfg = FeasibilityConfig(
        shapes=["deep_cavity"],
        replicates=1,
        pools_per_replicate=2,
        pool_size=2,
        refinement_factors=[2],
        search_seeds=[8800],
        search_solve_budget=2,
        raw_draw_budget=40,
        output_root=tmp_path / "runs",
        methods=methods,
        audit_response_solves=responses,
        frozen_references={
            "deep_cavity": FrozenReference(
                run_directory=source,
                mask_sha256=sha,
                expected_seed=8000,
                minimum_perimeter_fraction=0.5,
            )
        },
    )
    root, summary = run_feasibility_study(cfg)
    e = pd.read_csv(root / "evaluations.csv")
    assert summary["audit_raw_draws"] == 4
    assert (
        summary["number_of_pde_solves"]
        == 1
        + summary["audit_pde_solves"]
        + summary["audit_refinement_pde_solves"]
        + e.search_pde_solves.sum()
        + e.refinement_pde_solves.sum()
    )
    assert (root / "report.md").exists() and (root / "input_manifest.json").exists()
    assert hashlib.sha256((source / "final_geometry.npy").read_bytes()).hexdigest() == sha
    assert set(e.method) == set(methods)
    assert e.raw_draws.max() <= 40
    if responses:
        a = pd.read_csv(root / "audit.csv")
        assert a.audit_pde_solve_charged.sum() == a.audit_solution_id.nunique() == 2
        assert summary["audit_pde_solves"] == 2
        assert a.one_step_gain_Q.eq(0).all()
        assert a.groupby(["replicate", "pool"]).audit_solution_id.nunique().eq(1).all()
        contrasts = pd.read_csv(root / "audit_contrasts.csv")
        assert contrasts.status.eq("unavailable").all()
        assert summary["audit_refinement_pde_solves"] == 0


def test_audit_and_search_seed_overlap_is_rejected():
    reference = FrozenReference(run_directory=Path("."), mask_sha256="0" * 64, expected_seed=8000)
    with pytest.raises(ValueError, match="disjoint"):
        FeasibilityConfig(
            shapes=["deep_cavity"],
            frozen_references={"deep_cavity": reference},
            search_seeds=[8600],
        )


def test_screened_uniform_covers_nonminimum_candidates_without_perimeter_ranking():
    cfg = GuidedConfig()
    g = matched_area("deep_cavity", 100, 0.24).geometry
    edits, _ = draw_pool(g, np.random.default_rng(3), cfg, prepare_boundary(g, 0.12), 32)
    eligible = [(cheap_screen(e, g, cfg)[1], e) for e in edits if not cheap_screen(e, g, cfg)[0]]
    low, high = min(eligible, key=lambda x: x[0]), max(eligible, key=lambda x: x[0])
    assert low[0] < high[0]
    pool = [low[1], high[1]]
    rng = np.random.default_rng(26000)
    indices = [
        choose_pool(pool, g, cfg, rng, 1e-12, selection="screened_uniform")[0] for _ in range(500)
    ]
    assert 0.4 < np.mean(indices) < 0.6
    # Identical selection seeds couple the uniform exploration branch.
    for seed in range(50):
        first = choose_pool(pool, g, cfg, np.random.default_rng(seed), 0.25)
        second = choose_pool(
            pool, g, cfg, np.random.default_rng(seed), 0.25, selection="screened_uniform"
        )
        assert (first[1] == "uniform") == (second[1] == "uniform")
        if first[1] == "uniform":
            assert first[0] == second[0] == 0


def test_contrast_refinement_uses_shared_baseline_cancellation_and_counts_both_directions(tmp_path):
    from oxygen_shape.experiments.audit_contrasts import refine_audit_contrasts
    from oxygen_shape.experiments.frozen_geometry import FixedCase

    reference = FrozenReference(run_directory=tmp_path, mask_sha256="0" * 64, expected_seed=8000)
    cfg = FeasibilityConfig(
        shapes=["deep_cavity"],
        refinement_factors=[2],
        frozen_references={"deep_cavity": reference},
        methods=["uniform", "perimeter_pool", "screened_uniform"],
        audit_response_solves=True,
    )
    g = matched_area("deep_cavity", 100, 0.24).geometry
    search = cfg.search_config()
    edits, _ = draw_pool(g, np.random.default_rng(3), search, prepare_boundary(g, 0.12), 64)
    valid = [
        e.geometry
        for e in edits
        if e.geometry is not None and not validate_candidate(e.geometry, g, search)
    ]
    assert len(valid) >= 2
    (tmp_path / "audit_candidates").mkdir()
    solutions = [solve(geometry, cfg.model, cfg.solver) for geometry in valid[:2]]
    for i, geometry in enumerate(valid[:2], 1):
        np.savez_compressed(
            tmp_path / "audit_candidates" / f"solve_{i:05d}.npz", mask=geometry.inside
        )
    audit = []
    for pool, ids in enumerate([(1, 2), (2, 1)]):
        for method, i in zip(["screened_uniform", "perimeter_pool"], ids, strict=True):
            audit.append(
                {
                    "case": "deep_cavity",
                    "replicate": 0,
                    "pool": pool,
                    "method": method,
                    "valid": True,
                    "geometry_hash": str(i),
                    "audit_solution_id": i,
                    "delta_Q": solutions[i - 1].metrics.physical_flux_Q - 0.4,
                }
            )
    solves, error = refine_audit_contrasts(
        tmp_path, [FixedCase("deep_cavity", "deep_cavity", "initial", g, None)], cfg, audit
    )
    assert solves == 4 and error < 1e-9
    r = pd.read_csv(tmp_path / "audit_contrasts.csv")
    assert set(r.direction) == {"positive", "negative"}
    assert r.contrast_Q.sum() == pytest.approx(0, abs=1e-12)
