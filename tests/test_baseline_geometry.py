import numpy as np
import pytest

from oxygen_shape.geometry.baselines import (
    SHAPE_NAMES,
    hilbert_vertices,
    matched_area,
    matched_perimeter,
)
from oxygen_shape.geometry.boundary import extract_boundary
from oxygen_shape.geometry.constraints import validate_geometry
from oxygen_shape.geometry.generators import rectangle
from oxygen_shape.geometry.metrics import geometry_metrics


@pytest.mark.parametrize("shape", SHAPE_NAMES)
@pytest.mark.parametrize("n", [40, 80])
def test_area_source_connectivity_and_determinism(shape, n):
    baseline = matched_area(shape, n)
    geometry = baseline.geometry
    reference = matched_area("rectangle", n).geometry
    assert validate_geometry(geometry, reference=reference).valid
    assert geometry.area == pytest.approx(0.24, abs=1e-14)
    assert geometry.inside.sum() == round(0.24 * n * n)
    assert len(geometry.source_faces) / n == pytest.approx(0.1)
    assert geometry.source_faces == reference.source_faces
    assert geometry.inside.shape == (n, n)
    np.testing.assert_array_equal(geometry.inside, matched_area(shape, n).geometry.inside)
    assert geometry_metrics(geometry).robin_perimeter < 18


@pytest.mark.parametrize("shape", ["rectangle", "square"])
@pytest.mark.parametrize("n", [40, 100, 200])
def test_perimeter_matching_has_reported_tolerance(shape, n):
    baseline = matched_perimeter(shape, n)
    metrics = geometry_metrics(baseline.geometry)
    assert baseline.matching_error == pytest.approx(abs(metrics.robin_perimeter - 3))
    assert baseline.matching_error <= baseline.matching_tolerance + 1e-12
    assert baseline.matching_tolerance == 4 / n
    assert metrics.source_length == pytest.approx(0.1)


def test_geometry_metrics_against_rectangle_values():
    geometry = rectangle(5, 4, 0.1)
    m = geometry_metrics(geometry)
    assert m.area == pytest.approx(0.2)
    assert m.source_length == pytest.approx(0.4)
    assert m.robin_perimeter == pytest.approx(1.4)
    assert m.total_perimeter == pytest.approx(1.8)
    assert m.neumann_perimeter == 0
    assert m.bounding_diameter == pytest.approx(np.hypot(0.5, 0.4))
    assert m.occupied_width == pytest.approx(0.5)
    assert m.occupied_height == pytest.approx(0.4)
    assert m.opening_removed_fraction == pytest.approx(4 / 20)
    assert m.opening_radius == pytest.approx(0.1)
    assert m.minimum_neck is None
    slab = geometry_metrics(rectangle(5, 4, 0.1, slab=True))
    assert slab.neumann_perimeter == pytest.approx(1)
    assert slab.robin_perimeter == pytest.approx(0.4)
    assert len(extract_boundary(geometry)) * geometry.h == pytest.approx(m.total_perimeter)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_hilbert_is_a_contiguous_lattice_path(order):
    points = hilbert_vertices(order)
    size = 2**order
    assert points.shape == (size * size, 2)
    assert len(np.unique(points, axis=0)) == size * size
    np.testing.assert_allclose(np.abs(np.diff(points, axis=0)).sum(axis=1), 1 / (size - 1))
    assert points.min() == 0 and points.max() == 1


def test_invalid_generators_and_metrics():
    for n in [0, 30, True]:
        with pytest.raises(ValueError):
            matched_area("disk", n)
    for area in [-1, 0.240001, 0.01, 2]:
        with pytest.raises(ValueError):
            matched_area("disk", 100, area)
    with pytest.raises(ValueError):
        matched_perimeter("disk", 100)
    with pytest.raises(ValueError):
        matched_perimeter("rectangle", 100, 100)
    with pytest.raises(ValueError):
        matched_area("unknown", 100)
    with pytest.raises(ValueError):
        hilbert_vertices(0)
    with pytest.raises(ValueError):
        geometry_metrics(rectangle(4, 4, 0.1), 0)
