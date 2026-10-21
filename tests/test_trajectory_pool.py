import numpy as np

from oxygen_shape.geometry.constraints import raster_perimeter
from oxygen_shape.geometry.research_families import make_research_geometry
from oxygen_shape.optimization.trajectory_pool import cell_frontier, propose_cell_pair


def test_cell_exchange_preserves_area_membrane_and_source_exactly():
    geometry = make_research_geometry("hexagon").geometry
    frontier = cell_frontier(geometry, 0.12)
    rng = np.random.default_rng(10)
    tested = 0
    for _ in range(100):
        edit = propose_cell_pair(geometry, frontier, rng)
        if edit.geometry is None:
            continue
        tested += 1
        assert edit.parameters["operation"] == "cell_pair"
        assert edit.geometry.area == geometry.area
        assert raster_perimeter(edit.geometry) == raster_perimeter(geometry)
        np.testing.assert_array_equal(edit.geometry.inside[:, :12], geometry.inside[:, :12])
        assert np.count_nonzero(edit.geometry.inside != geometry.inside) == 2
    assert tested > 20
