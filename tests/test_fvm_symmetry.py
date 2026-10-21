import numpy as np

from oxygen_shape.geometry.generators import rectangle
from oxygen_shape.physics.model import DiffusionModel
from oxygen_shape.solvers.finite_volume import solve


def test_symmetric_source_and_rectangle():
    result = solve(rectangle(31, 24, 1 / 31), DiffusionModel(Bi=10))
    np.testing.assert_allclose(result.concentration, result.concentration[::-1], atol=1e-13)


def test_true_2d_refinement_at_fixed_physical_geometry():
    results = [solve(rectangle(n, n, 1 / n), DiffusionModel(Bi=1)) for n in [16, 32, 64]]
    q = [r.metrics.physical_flux_Q for r in results]
    assert abs(q[2] - q[1]) < abs(q[1] - q[0])
    assert abs(q[2] - q[1]) / q[2] < 0.002
    errors = []
    for coarse, fine in zip(results[:-1], results[1:], strict=True):
        n = coarse.concentration.shape[0]
        restricted = fine.concentration.reshape(n, 2, n, 2).mean(axis=(1, 3))
        errors.append(np.max(np.abs(coarse.concentration - restricted)))
        assert coarse.metrics.area == fine.metrics.area == 1
        assert coarse.metrics.robin_perimeter == fine.metrics.robin_perimeter == 3
    assert errors[1] < errors[0]
