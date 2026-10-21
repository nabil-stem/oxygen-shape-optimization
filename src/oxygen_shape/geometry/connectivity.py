"""Connectivity uses shared faces, never diagonal contact."""

from scipy.ndimage import generate_binary_structure, label

from oxygen_shape.geometry.raster import RasterGeometry


def component_count(geometry: RasterGeometry) -> int:
    _, count = label(geometry.inside, structure=generate_binary_structure(2, 1))
    return int(count)
