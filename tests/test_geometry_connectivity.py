import numpy as np
import pytest

from oxygen_shape.geometry.boundary import extract_boundary
from oxygen_shape.geometry.constraints import validate_geometry
from oxygen_shape.geometry.generators import rectangle
from oxygen_shape.geometry.raster import FaceKey, RasterGeometry, Side
from oxygen_shape.physics.model import DiffusionModel
from oxygen_shape.solvers.finite_volume import solve


@pytest.mark.parametrize(
    "mask", [np.eye(3, dtype=bool), np.array([[1, 0, 1], [1, 0, 1]], dtype=bool)]
)
def test_disconnected_air_rejected(mask):
    geometry = RasterGeometry(mask, 0.1, (FaceKey(0, 0, Side.WEST),))
    assert not validate_geometry(geometry).valid
    with pytest.raises(ValueError, match="connected"):
        solve(geometry, DiffusionModel())


def test_missing_source_rejected_even_with_robin_sink():
    geometry = RasterGeometry(np.ones((3, 3), dtype=bool), 0.1, ())
    with pytest.raises(ValueError, match="source"):
        solve(geometry, DiffusionModel())


def test_boundary_count_and_unique_faces():
    geometry = rectangle(7, 5, 0.1)
    faces = extract_boundary(geometry)
    assert len(faces) == 24
    assert len({f.key for f in faces}) == len(faces)


def test_conflicting_or_duplicate_face_labels_rejected():
    geometry = rectangle(3, 3, 0.1)
    source = geometry.source_faces
    for bad in [
        RasterGeometry(geometry.inside, 0.1, source + source),
        RasterGeometry(geometry.inside, 0.1, source, source),
    ]:
        assert not validate_geometry(bad).valid
