import numpy as np
import pytest

from oxygen_shape.geometry.boundary import BoundaryKind, extract_boundary
from oxygen_shape.geometry.generators import rectangle
from oxygen_shape.geometry.raster import FaceKey, RasterGeometry, Side
from oxygen_shape.physics.model import DiffusionModel, SolverSettings
from oxygen_shape.solvers.diagnostics import require_conservation
from oxygen_shape.solvers.finite_volume import assemble, solve


@pytest.mark.parametrize("Bi", [0.01, 0.1, 1, 10, 100])
def test_rectangle_conservation_and_maximum_principle(Bi):
    geometry = rectangle(24, 18, 1 / 24)
    result = solve(geometry, DiffusionModel(Bi=Bi))
    m = result.metrics
    assert m.conservation_error < 1e-10
    assert m.residual_inf < 1e-12
    assert 0 < result.concentration.min() <= result.concentration.max() < 1
    assert 0 < m.efficiency_eta < 1
    assert m.area == pytest.approx(0.75)
    assert m.robin_perimeter == pytest.approx(2.75)
    assert m.source_length == pytest.approx(0.75)
    robin = np.array([f.kind == BoundaryKind.ROBIN for f in result.faces])
    np.testing.assert_allclose(
        result.outward_face_flux[robin],
        Bi * geometry.h * result.wall_concentration[robin],
        atol=1e-14,
    )
    assert m.efficiency_eta == pytest.approx(result.wall_concentration[robin].mean())


def test_constant_dirichlet_and_spd_matrix():
    geometry = rectangle(5, 4, 0.1)
    sources = tuple(face.key for face in extract_boundary(geometry))
    geometry = RasterGeometry(geometry.inside, geometry.h, sources)
    system = assemble(geometry, DiffusionModel())
    dense = system.matrix.toarray()
    np.testing.assert_array_equal(dense, dense.T)
    assert np.linalg.eigvalsh(dense).min() > 0
    result = solve(geometry, DiffusionModel())
    np.testing.assert_allclose(result.concentration, 1, atol=1e-14)


def test_single_cell_has_hand_derived_balance():
    geometry = rectangle(1, 1, 0.2)
    result = solve(geometry, DiffusionModel(Bi=5))
    # Three Robin faces, each G=2/3; one Dirichlet face G=2.
    assert result.concentration[0, 0] == pytest.approx(0.5)
    assert result.metrics.physical_flux_Q == pytest.approx(1.0)


def test_conservation_check_rejects_imbalance():
    with pytest.raises(ArithmeticError, match="conservation"):
        require_conservation(1, 0.9, SolverSettings())


def test_nonrectangular_mask_conserves_on_exposed_faces():
    # Algebraic conservation regression only; no scientific shape comparison.
    mask = np.ones((8, 10), dtype=bool)
    mask[3:5, 6:] = False
    source = tuple(FaceKey(row, 0, Side.WEST) for row in range(8))
    result = solve(RasterGeometry(mask, 0.1, source), DiffusionModel(Bi=2))
    assert result.metrics.conservation_error < 1e-11
    assert np.isnan(result.concentration[~mask]).all()
