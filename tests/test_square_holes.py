import numpy as np
import pytest

from oxygen_shape.experiments.holes_study import declared_layouts, paired_metrics
from oxygen_shape.geometry.boundary import BoundaryKind, extract_boundary
from oxygen_shape.geometry.constraints import ShapeConstraints, raster_perimeter, validate_shape
from oxygen_shape.geometry.raster import refine_geometry
from oxygen_shape.geometry.square_holes import (
    SquareHole,
    boundary_hole_ids,
    hole_boundary,
    hole_labels,
    maze_chamber,
    perforate,
)
from oxygen_shape.physics.model import DiffusionModel
from oxygen_shape.solvers.finite_volume import solve


def test_square_islands_preserve_outer_labels_and_partition_both_boundary_kinds():
    outer = maze_chamber()
    holes = declared_layouts(100)["spread_6"]
    g = perforate(outer, holes)
    assert hole_labels(g)[1] == 6
    assert g.area == pytest.approx(outer.area - 6 * 0.08**2)
    assert raster_perimeter(g) == pytest.approx(raster_perimeter(outer) + 6 * 4 * 0.08)
    ids = boundary_hole_ids(g)
    faces = extract_boundary(g)
    assert sum(ids > 0) == 6 * 4 * 8
    outer_faces = {(f.key, f.kind) for f, i in zip(faces, ids, strict=True) if not i}
    assert outer_faces == {(f.key, f.kind) for f in extract_boundary(outer)}
    reflected = hole_boundary(g, "neumann")
    for face, component in zip(extract_boundary(reflected), ids, strict=True):
        if component:
            assert face.kind == BoundaryKind.NEUMANN
    assert raster_perimeter(reflected) == pytest.approx(raster_perimeter(outer))
    assert hole_boundary(reflected, "robin").neumann_faces == g.neumann_faces
    fine = refine_geometry(reflected, 2)
    assert hole_labels(fine)[1] == 6
    assert fine.area == pytest.approx(reflected.area)
    assert raster_perimeter(fine) == pytest.approx(raster_perimeter(reflected))
    # Earlier optimizers retain their explicit no-hole policy.
    assert not validate_shape(g, ShapeConstraints(g.area, 10), outer).valid


@pytest.mark.parametrize(
    "holes,match",
    [
        ((SquareHole(40, 8, 8),), "collar"),
        ((SquareHole(16, 40, 8),), "clearance"),
        ((SquareHole(40, 40, 8), SquareHole(40, 48, 8)), "clearance"),
        ((SquareHole(40, 40, 2),), "cutoff"),
    ],
)
def test_bad_islands_are_rejected_before_solving(holes, match):
    with pytest.raises(ValueError, match=match):
        perforate(maze_chamber(), holes)


@pytest.mark.parametrize("Bi", [0.01, 1.0, 100.0])
def test_energy_monotonicity_and_exact_global_flux_attribution(Bi):
    outer = maze_chamber()
    geometry = perforate(outer, declared_layouts(100)["near_pair"])
    model = DiffusionModel(Bi=Bi)
    base = solve(outer, model)
    r = solve(geometry, model)
    n = solve(hole_boundary(geometry, "neumann"), model)
    metrics = paired_metrics(base.metrics.physical_flux_Q, r, n, geometry)
    assert metrics["obstruction_delta"] <= 1e-12
    assert metrics["activation_delta"] >= -1e-12
    assert metrics["delta_Q"] == pytest.approx(
        metrics["activation_delta"] + metrics["obstruction_delta"], abs=1e-14
    )
    assert metrics["inner_Q"] + metrics["outer_Q"] == pytest.approx(metrics["Q_R"])
    assert metrics["inner_Q"] + metrics["outer_delta"] == pytest.approx(metrics["delta_Q"])
    assert metrics["conservation_error"] < 1e-9
    assert r.metrics.source_length == pytest.approx(base.metrics.source_length)
    assert np.all(n.outward_face_flux[boundary_hole_ids(geometry) > 0] == 0)


def test_zero_permeability_with_multiple_holes_is_exactly_constant():
    g = perforate(maze_chamber(), declared_layouts(100)["spread_6"])
    result = solve(g, DiffusionModel(Bi=0))
    assert result.metrics.physical_flux_Q == 0
    np.testing.assert_array_equal(result.concentration[g.inside], 1)
