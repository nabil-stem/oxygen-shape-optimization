import numpy as np
import pytest

from oxygen_shape.geometry.constraints import raster_perimeter
from oxygen_shape.geometry.research_families import make_research_geometry
from oxygen_shape.physics.model import DiffusionModel
from oxygen_shape.solvers.finite_volume import solve
from scripts.collar_ablation import conducting_collar


@pytest.mark.parametrize("length", [0.06, 0.12])
def test_collar_changes_only_declared_wall_labels_and_respects_neck_bound(length):
    original = make_research_geometry("hexagon").geometry
    changed = conducting_collar(original, length)
    np.testing.assert_array_equal(changed.inside, original.inside)
    assert changed.source_faces == original.source_faces
    assert raster_perimeter(original) - raster_perimeter(changed) == pytest.approx(2 * length)
    result = solve(changed, DiffusionModel(Bi=100.0))
    assert result.metrics.physical_flux_Q <= 0.1 / length
    assert result.metrics.conservation_error < 1e-9
