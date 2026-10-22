import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("triangle")

from oxygen_shape.experiments.holes_config import HolesConfig
from oxygen_shape.experiments.holes_evidence import HoleEvidence
from oxygen_shape.experiments.holes_study import (
    declared_layouts,
    position_roster,
    run_holes_study,
)
from oxygen_shape.experiments.reporting import create_run
from oxygen_shape.geometry.square_holes import maze_chamber
from oxygen_shape.optimization.hole_relocation import run_hole_relocation


def test_paid_relocation_replays_interrupted_prefix_and_preserves_resources(tmp_path):
    cfg = HolesConfig(output_root=tmp_path, search_budget=5)
    root = create_run(cfg, "hole_test")
    evidence = HoleEvidence(root, cfg)
    outer = maze_chamber()
    positions, _ = position_roster(outer, 8, cfg)
    original = evidence.get

    def interrupted(key, *args, **kwargs):
        if key.endswith("s0003"):
            raise RuntimeError("power loss")
        return original(key, *args, **kwargs)

    evidence.get = interrupted
    with pytest.raises(RuntimeError, match="power loss"):
        run_hole_relocation(
            cfg,
            outer,
            declared_layouts(100)["spread_6"],
            positions,
            "concentration_greedy",
            420000,
            evidence,
            "search/test",
        )
    prefix = json.loads((root / "search/test/history.json").read_text())
    recovered = HoleEvidence(root, cfg)
    holes, summary = run_hole_relocation(
        cfg,
        outer,
        declared_layouts(100)["spread_6"],
        positions,
        "concentration_greedy",
        420000,
        recovered,
        "search/test",
    )
    final = json.loads((root / "search/test/history.json").read_text())
    assert final[: len(prefix)] == prefix
    assert summary["number_of_pde_solves"] == 5
    assert recovered.new_solves == 2
    assert summary["final_Q"] >= summary["initial_Q"]
    assert len(holes) == 6
    again = HoleEvidence(root, cfg)
    _, same = run_hole_relocation(
        cfg,
        outer,
        declared_layouts(100)["spread_6"],
        positions,
        "concentration_greedy",
        420000,
        again,
        "search/test",
    )
    assert again.new_solves == 0 and same == summary
    assert all(b["Q"] >= a["Q"] for a, b in zip(final[:-1], final[1:], strict=True))


def test_evidence_rejects_changed_cached_fields_and_counts_pending_loss(tmp_path):
    cfg = HolesConfig(output_root=tmp_path)
    root = create_run(cfg, "hole_cache")
    evidence = HoleEvidence(root, cfg)
    evidence.get("one", maze_chamber(), 1)
    (root / "pending_solve.json").write_text(
        json.dumps(dict(path="cache/two.fvm.npz", solver="fvm"))
    )
    recovered = HoleEvidence(root, cfg)
    assert recovered.accounting()["possible_uncommitted_solve_upper_bound"] == 1
    path = root / "cache/one.fvm.npz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Changed or missing"):
        HoleEvidence(root, cfg)


def test_small_square_study_produces_checked_results_and_zero_solve_resume(tmp_path, monkeypatch):
    cfg = HolesConfig(
        output_root=tmp_path,
        families=["maze_chamber"],
        widths=[0.08],
        position_stride=0.16,
        permeabilities=[1.0],
        validation_Bi=[1.0],
        refinement_factors=[2],
        fem_max_areas=[0.001],
        search_seeds=[420000],
        search_budget=2,
    )
    root, summary = run_holes_study(cfg, progress=lambda _: None)
    assert summary["status"] == "complete"
    assert summary["search_runs"] == 2
    rows = pd.read_csv(root / "layout_metrics.csv")
    six = rows[rows.number_of_holes == 6]
    assert np.ptp(six.area) < 1e-12 and np.ptp(six.robin_perimeter) < 1e-12
    assert (root / "figures/square_fields_linear.png").is_file()
    original = (root / "position_maps.csv").read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("Finished resume performed a new PDE solve")

    monkeypatch.setattr("oxygen_shape.experiments.morphology_validation.solve", forbidden)
    monkeypatch.setattr(
        "oxygen_shape.experiments.morphology_validation.solve_raster_fem", forbidden
    )
    _, resumed = run_holes_study(cfg, resume=root, progress=lambda _: None)
    assert resumed["this_invocation_new_solves"] == 0
    assert (root / "position_maps.csv").read_bytes() == original
    with pytest.raises(ValueError, match="config changed"):
        run_holes_study(cfg.model_copy(update={"gap": 0.05}), resume=root)
