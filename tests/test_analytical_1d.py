import numpy as np
import pytest

from oxygen_shape.geometry.boundary import BoundaryKind
from oxygen_shape.geometry.generators import rectangle
from oxygen_shape.physics.model import DiffusionModel
from oxygen_shape.solvers.analytical_1d import slab_flux, slab_solution
from oxygen_shape.solvers.finite_volume import solve


@pytest.mark.parametrize("Bi", [0, 0.01, 0.1, 1, 10, 100])
@pytest.mark.parametrize("length", [0.5, 1.0, 2.0])
def test_slab_multigrid_is_exact_for_linear_solution(Bi, length):
    for nx in [1, 8, 24, 64]:
        h = length / nx
        geometry = rectangle(nx, 3, h, slab=True)
        result = solve(geometry, DiffusionModel(Bi=Bi))
        exact = slab_solution((np.arange(nx) + 0.5) * h, Bi, length)
        np.testing.assert_allclose(
            result.concentration, np.broadcast_to(exact, (3, nx)), atol=2e-11
        )
        q = slab_flux(Bi, length)
        assert result.metrics.physical_flux_Q / (3 * h) == pytest.approx(q, abs=2e-11)
        robin = np.array([f.kind == BoundaryKind.ROBIN for f in result.faces])
        np.testing.assert_allclose(
            result.wall_concentration[robin], slab_solution(length, Bi, length), atol=2e-11
        )


def test_known_analytical_values():
    assert slab_flux(1) == 0.5
    np.testing.assert_allclose(slab_solution([0, 0.5, 1], 1), [1, 0.75, 0.5])


@pytest.mark.parametrize("Bi,length", [(-1, 1), (np.nan, 1), (np.inf, 1), (1, 0)])
def test_invalid_slab_parameters(Bi, length):
    with pytest.raises(ValueError):
        slab_flux(Bi, length)


def test_invalid_slab_coordinate():
    with pytest.raises(ValueError):
        slab_solution([1.1], 1)
