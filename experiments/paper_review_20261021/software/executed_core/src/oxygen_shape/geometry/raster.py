"""Square-cell masks. Row zero is at the bottom of the physical domain."""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class Side(str, Enum):
    WEST = "west"
    EAST = "east"
    SOUTH = "south"
    NORTH = "north"


OFFSETS: dict[Side, tuple[int, int]] = {
    Side.WEST: (0, -1),
    Side.EAST: (0, 1),
    Side.SOUTH: (-1, 0),
    Side.NORTH: (1, 0),
}


@dataclass(frozen=True, order=True)
class FaceKey:
    row: int
    col: int
    side: Side

    def __post_init__(self) -> None:
        if (
            isinstance(self.row, bool)
            or isinstance(self.col, bool)
            or not isinstance(self.row, int)
            or not isinstance(self.col, int)
            or not isinstance(self.side, Side)
        ):
            raise ValueError("Face keys require integer row/col and a Side enum")


@dataclass(frozen=True, eq=False)
class RasterGeometry:
    inside: NDArray[np.bool_]
    h: float
    source_faces: tuple[FaceKey, ...]
    neumann_faces: tuple[FaceKey, ...] = ()

    def __post_init__(self) -> None:
        mask = np.asarray(self.inside)
        if mask.ndim != 2 or mask.dtype != np.bool_ or not mask.size or not mask.any():
            raise ValueError("inside must be a nonempty 2D boolean mask with interior cells")
        if not np.isfinite(self.h) or self.h <= 0:
            raise ValueError("h must be positive and finite")
        mask = mask.copy()
        mask.flags.writeable = False
        object.__setattr__(self, "inside", mask)
        object.__setattr__(self, "source_faces", tuple(self.source_faces))
        object.__setattr__(self, "neumann_faces", tuple(self.neumann_faces))

    @property
    def area(self) -> float:
        return float(self.inside.sum() * self.h**2)

    def with_mask(self, inside: NDArray[np.bool_]) -> "RasterGeometry":
        """Retain boundary labels; validation rejects any buried/deleted source."""
        return RasterGeometry(inside, self.h, self.source_faces, self.neumann_faces)


def refine_geometry(geometry: RasterGeometry, factor: int) -> RasterGeometry:
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 2:
        raise ValueError("Refinement factor must be an integer >=2")

    def subdivide(faces: tuple[FaceKey, ...]) -> tuple[FaceKey, ...]:
        result = []
        for face in faces:
            row, col = face.row * factor, face.col * factor
            for offset in range(factor):
                if face.side in (Side.WEST, Side.EAST):
                    result.append(
                        FaceKey(
                            row + offset,
                            col + (factor - 1 if face.side == Side.EAST else 0),
                            face.side,
                        )
                    )
                else:
                    result.append(
                        FaceKey(
                            row + (factor - 1 if face.side == Side.NORTH else 0),
                            col + offset,
                            face.side,
                        )
                    )
        return tuple(result)

    mask = np.repeat(np.repeat(geometry.inside, factor, axis=0), factor, axis=1)
    return RasterGeometry(
        mask,
        geometry.h / factor,
        subdivide(geometry.source_faces),
        subdivide(geometry.neumann_faces),
    )
