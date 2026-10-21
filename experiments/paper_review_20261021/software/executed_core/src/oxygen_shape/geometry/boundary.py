"""Extract and classify each exposed face exactly once."""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from oxygen_shape.geometry.raster import OFFSETS, FaceKey, RasterGeometry


class BoundaryKind(str, Enum):
    SOURCE_DIRICHLET = "source_dirichlet"
    ROBIN = "robin"
    NEUMANN = "neumann"


@dataclass(frozen=True)
class BoundaryFace:
    key: FaceKey
    kind: BoundaryKind


def extract_boundary(geometry: RasterGeometry) -> tuple[BoundaryFace, ...]:
    mask = geometry.inside
    ny, nx = mask.shape
    exposed: list[FaceKey] = []
    for row, col in np.argwhere(mask):
        for side, (dy, dx) in OFFSETS.items():
            y, x = row + dy, col + dx
            if not (0 <= y < ny and 0 <= x < nx) or not mask[y, x]:
                exposed.append(FaceKey(int(row), int(col), side))
    source, neumann = set(geometry.source_faces), set(geometry.neumann_faces)
    if len(source) != len(geometry.source_faces) or len(neumann) != len(geometry.neumann_faces):
        raise ValueError("Duplicate boundary labels")
    if source & neumann:
        raise ValueError("Source and Neumann labels overlap")
    if not (source | neumann).issubset(set(exposed)):
        raise ValueError("Every labelled face must be exposed on an interior cell")
    return tuple(
        BoundaryFace(
            key,
            BoundaryKind.SOURCE_DIRICHLET
            if key in source
            else BoundaryKind.NEUMANN
            if key in neumann
            else BoundaryKind.ROBIN,
        )
        for key in exposed
    )
