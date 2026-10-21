import numpy as np

from oxygen_shape.geometry.constraints import validate_geometry
from oxygen_shape.geometry.generators import rectangle
from oxygen_shape.geometry.raster import FaceKey, RasterGeometry, Side


def test_deleting_source_cell_rejected():
    geometry = rectangle(8, 6, 0.1)
    mask = geometry.inside.copy()
    mask[2, 0] = False
    assert not validate_geometry(geometry.with_mask(mask), reference=geometry).valid


def test_burying_source_face_rejected():
    mask = np.zeros((6, 8), dtype=bool)
    mask[:, 1:6] = True
    source = tuple(FaceKey(row, 1, Side.WEST) for row in range(6))
    geometry = RasterGeometry(mask, 0.1, source)
    mask[:, 0] = True
    assert not validate_geometry(geometry.with_mask(mask), reference=geometry).valid


def test_changing_source_length_or_spacing_rejected():
    geometry = rectangle(8, 6, 0.1)
    shorter = RasterGeometry(geometry.inside, 0.1, geometry.source_faces[:-1])
    scaled = RasterGeometry(geometry.inside, 0.2, geometry.source_faces)
    assert not validate_geometry(shorter, reference=geometry).valid
    assert not validate_geometry(scaled, reference=geometry).valid


def test_edit_away_from_source_retains_source():
    geometry = rectangle(8, 6, 0.1)
    mask = geometry.inside.copy()
    mask[0, -1] = False
    assert validate_geometry(geometry.with_mask(mask), reference=geometry).valid
    assert geometry.inside[0, -1]
