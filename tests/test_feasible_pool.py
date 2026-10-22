import json
from pathlib import Path

import numpy as np
import pytest

from oxygen_shape.experiments.config import load_config
from oxygen_shape.experiments.feasible_pool_study import PoolStudyConfig, run_feasible_pool_study
from oxygen_shape.geometry.baselines import matched_area
from oxygen_shape.geometry.boundary import BoundaryKind, extract_boundary
from oxygen_shape.geometry.disk_edits import DiskEdit
from oxygen_shape.geometry.raster import RasterGeometry
from oxygen_shape.optimization.feasible_pool import (
    FeasiblePool,
    FeasiblePoolConfig,
    build_pool,
    run_feasible_search,
    selection_probabilities,
)
from oxygen_shape.optimization.guided_search import validate_candidate


def test_hard_constraints_reject_collapse_islands_evasion_and_boundary_relabeling():
    cfg = FeasiblePoolConfig()
    initial = matched_area("rectangle", 100, 0.24).geometry
    with pytest.raises(ValueError, match="nonempty"):
        initial.with_mask(np.zeros_like(initial.inside))
    tiny = np.zeros_like(initial.inside)
    for face in initial.source_faces:
        tiny[face.row, face.col] = True
    assert "Area tolerance" in validate_candidate(initial.with_mask(tiny), initial, cfg)
    # Transfer one interior cell into a remote corner, keeping area exactly fixed.
    island = initial.inside.copy()
    row, col = np.argwhere(island & (np.indices(island.shape)[1] > 20))[0]
    island[row, col] = False
    assert not island[0, -1]
    island[0, -1] = True
    assert initial.with_mask(island).area == initial.area
    assert "face-connected" in validate_candidate(initial.with_mask(island), initial, cfg)
    assert validate_candidate(initial.with_mask(np.roll(initial.inside, 30, axis=1)), initial, cfg)
    changed_source = RasterGeometry(initial.inside, initial.h, initial.source_faces[:-1])
    assert "source location or length" in validate_candidate(changed_source, initial, cfg)
    wall = next(f.key for f in extract_boundary(initial) if f.kind == BoundaryKind.ROBIN)
    changed_wall = RasterGeometry(initial.inside, initial.h, initial.source_faces, (wall,))
    assert "impermeable boundary classification" in validate_candidate(changed_wall, initial, cfg)


def test_refill_stops_at_quota_charges_rejections_and_excludes_duplicates(monkeypatch):
    import oxygen_shape.optimization.feasible_pool as module

    cfg = FeasiblePoolConfig(pool_size=2, refill_draw_limit=12)
    initial = matched_area("rectangle", 100, 0.24).geometry
    masks = []
    for i in range(3):
        mask = initial.inside.copy()
        mask[0, i] = True
        masks.append(initial.with_mask(mask))
    # Isolate stopping/duplicates using supplied validator verdicts. Physical
    # constraints are exercised separately above and by the real rollout below.
    edits = [
        DiskEdit(None, {}, "bad draw"),
        DiskEdit(initial, {}),
        DiskEdit(masks[0], {}),
        DiskEdit(masks[0], {}),
        DiskEdit(masks[1], {}),
    ]
    calls = []

    def draw(*args):
        index = len(calls)
        calls.append(index)
        return [edits[index]], [{"proposal_seed": index}]

    monkeypatch.setattr(module, "draw_pool", draw)
    monkeypatch.setattr(module, "cheap_screen", lambda e, *args: (e.rejection, 3.0))
    validated = []
    monkeypatch.setattr(module, "validate_candidate", lambda g, *args: validated.append(g) or "")
    pool = build_pool(cfg, initial, initial, None, np.random.default_rng(1), 10, set(), quota=2)
    assert len(pool.rows) == 5 and pool.quota_reached
    assert pool.eligible.tolist() == [False, False, True, False, True]
    assert pool.full_checks == len(validated) == 2
    assert [r["slot"] for r in pool.rows] == list(range(5))
    calls.clear()
    limited = build_pool(cfg, initial, initial, None, np.random.default_rng(1), 3, set(), quota=4)
    assert len(limited.rows) == 3 and not limited.quota_reached


def test_pool_exploration_and_no_eligible_fallback():
    pool = FeasiblePool([], [], np.array([False, True, True]), 0, 0, True)
    np.testing.assert_allclose(selection_probabilities(pool, [100, 0, 1], 0.25), [0.25, 0, 0.75])
    np.testing.assert_allclose(selection_probabilities(pool, [0, 0, 0], 0.25), [0.25, 0.375, 0.375])
    empty = FeasiblePool([], [], np.array([False, False]), 0, 0, False)
    np.testing.assert_array_equal(selection_probabilities(empty, [1, 5], 0.25), [1, 0])
    with pytest.raises(ValueError):
        selection_probabilities(pool, [0, np.nan, 0], 0.25)
    with pytest.raises(ValueError):
        selection_probabilities(pool, [0, 0, 0], 0)


def test_real_search_is_reproducible_conservative_bounded_and_records_global_acceptance():
    cfg = FeasiblePoolConfig(seed=181000, solve_budget=3, raw_draw_budget=400)
    initial = matched_area("deep_cavity", 100, 0.24).geometry
    first = run_feasible_search(cfg, initial, quota=2, ranking="flux")
    second = run_feasible_search(cfg, initial, quota=2, ranking="flux")
    assert first.search.attempts <= 400 and first.search.number_of_pde_solves <= 3
    assert len(first.search.proposals) == first.search.attempts
    assert first.maximum_conservation_error < 1e-9
    np.testing.assert_array_equal(
        first.search.final_geometry.inside, second.search.final_geometry.inside
    )
    assert first.pools == second.pools
    assert (
        first.search.final_solution.metrics.physical_flux_Q
        >= first.search.initial_solution.metrics.physical_flux_Q
    )
    for row in first.search.history[1:]:
        if row["accepted"]:
            assert row["physical_flux_Q"] == row["candidate_Q"]
    assert not validate_candidate(first.search.final_geometry, initial, cfg)
    for pool_index in range(len(first.pools)):
        rows = [r for r in first.search.proposals if r["pool"] == pool_index]
        assert sum(r["selection_probability"] for r in rows) == pytest.approx(1)
        assert sum(r["selected"] for r in rows) == 1
    with pytest.raises(ValueError):
        run_feasible_search(cfg, initial, quota=3, ranking="flux")


def test_exhausted_raw_budget_is_retained_as_an_outcome(monkeypatch):
    import oxygen_shape.optimization.feasible_pool as module

    cfg = FeasiblePoolConfig(seed=2, solve_budget=3, raw_draw_budget=9)
    initial = matched_area("rectangle", 100, 0.24).geometry

    def draw(g, rng, config, prepared, size):
        return [DiskEdit(None, {}, "Impossible")], [{"addition_face": 0, "donor_face": 0}]

    monkeypatch.setattr(module, "draw_pool", draw)
    result = run_feasible_search(cfg, initial, quota=4, ranking="uniform")
    assert result.status == "raw_budget_exhausted"
    assert result.search.attempts == 9 and result.search.number_of_pde_solves == 1


def test_small_study_writes_full_roster_provenance_and_refinements(tmp_path):
    cfg = PoolStudyConfig(
        output_root=tmp_path, solve_budget=2, raw_draw_budget=200, evaluation_seeds=[181000]
    )
    root, summary = run_feasible_pool_study(cfg)
    assert summary["runs"] == 12
    assert summary["number_of_pde_solves"] <= 12 * 6
    assert (root / "source_snapshot/src/oxygen_shape/optimization/feasible_pool.py").is_file()
    assert json.loads((root / "protocol.json").read_text())["expected_runs"] == 12
    assert (root / "paired_comparisons.csv").is_file()
    assert len(list(root.glob("runs/*/refinement.csv"))) == 12
    assert summary["training_pde_solves"] == 0 and not summary["policy_promoted"]
    # The shipped example generates its own starts without private run files.
    config = load_config(Path("configs/feasible_pools.yaml"), PoolStudyConfig)
    assert config.inputs == [] and config.include_canonical
