"""Boundary-attached disk operations with optional external donor weights."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from oxygen_shape.geometry.boundary import BoundaryKind, extract_boundary
from oxygen_shape.geometry.raster import OFFSETS, RasterGeometry


@dataclass(frozen=True)
class DiskEdit:
    geometry: RasterGeometry | None
    parameters: dict[str, float | int | str]
    rejection: str = ""


@dataclass(frozen=True)
class PreparedBoundary:
    geometry: RasterGeometry
    protected_length: float
    points: NDArray
    normals: NDArray


def prepare_boundary(geometry: RasterGeometry, protected_length: float) -> PreparedBoundary:
    points, normals = boundary_samples(geometry, protected_length)
    points.flags.writeable = normals.flags.writeable = False
    return PreparedBoundary(geometry, protected_length, points, normals)


def boundary_samples(
    geometry: RasterGeometry,
    protected_length: float,
) -> tuple[NDArray, NDArray]:
    faces = [
        f
        for f in extract_boundary(geometry)
        if f.kind == BoundaryKind.ROBIN
        and (f.key.col + 0.5) * geometry.h >= protected_length + 2 * geometry.h
    ]
    if not faces:
        return np.empty((0, 2)), np.empty((0, 2))
    smooth = gaussian_filter(geometry.inside.astype(float), sigma=1)
    gy, gx = np.gradient(smooth)
    points, normals = [], []
    for face in faces:
        key = face.key
        dy, dx = OFFSETS[key.side]
        points.append(
            ((key.col + 0.5 + dx / 2) * geometry.h, (key.row + 0.5 + dy / 2) * geometry.h)
        )
        normal = np.array([-gx[key.row, key.col], -gy[key.row, key.col]])
        if np.linalg.norm(normal) < 1e-12:
            normal = np.array([dx, dy], dtype=float)
        normal /= np.linalg.norm(normal)
        normals.append(normal)
    return np.asarray(points), np.asarray(normals)


def disk_mask(geometry: RasterGeometry, center: NDArray, radius: float) -> NDArray[np.bool_]:
    if not np.isfinite(radius) or radius < 3 * geometry.h - 1e-12:
        raise ValueError("Disk radii must be finite and at least 3h")
    if np.asarray(center).shape != (2,) or not np.isfinite(center).all():
        raise ValueError("Disk center must have two finite coordinates")
    y, x = np.indices(geometry.inside.shape)
    return ((x + 0.5) * geometry.h - center[0]) ** 2 + (
        (y + 0.5) * geometry.h - center[1]
    ) ** 2 <= radius**2


def apply_disk(
    geometry: RasterGeometry, center: NDArray, radius: float, *, add: bool
) -> RasterGeometry:
    disk = disk_mask(geometry, center, radius)
    if add:
        height, width = np.asarray(geometry.inside.shape) * geometry.h
        if np.any(center - radius < 0) or np.any(center + radius > [width, height]):
            raise ValueError("Addition disk exceeds the fixed box")
    inside = geometry.inside | disk if add else geometry.inside & ~disk
    return geometry.with_mask(inside)


def propose_paired_disk(
    geometry: RasterGeometry,
    rng: np.random.Generator,
    radius: float,
    protected_length: float,
    *,
    region: int | None = None,
    theta_max: float = np.pi / 3,
    donor_weights: NDArray | None = None,
    face_pair: tuple[int, int] | None = None,
    prepared: PreparedBoundary | None = None,
) -> DiskEdit:
    """Uniform boundary-measure sampling, with optional learned addition region.

    Region = 2*x_third + upper_half. Donor faces are uniform unless weights
    aligned with boundary_samples are supplied. No local field accepts a move.
    The removal radius is an exact order statistic of distances to its center.
    A subcell random center offset almost surely avoids equal-distance ties.
    """
    if not 0 <= theta_max <= np.pi:
        raise ValueError("theta_max must lie in [0, pi]")
    if radius < 3 * geometry.h - 1e-12:
        raise ValueError("Proposal radius must be >=3h")
    if prepared is None:
        points, normals = boundary_samples(geometry, protected_length)
    else:
        if prepared.geometry is not geometry or prepared.protected_length != protected_length:
            raise ValueError("Prepared boundary belongs to a different geometry or source collar")
        points, normals = prepared.points, prepared.normals
    if face_pair is not None:
        if region is not None or donor_weights is not None:
            raise ValueError("A fixed face pair cannot be combined with region/donor weights")
        if len(face_pair) != 2 or any(
            isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 0 <= i < len(points)
            for i in face_pair
        ):
            raise ValueError("Face pair indices must identify eligible boundary faces")
    if donor_weights is not None:
        donor_weights = np.asarray(donor_weights, dtype=float)
        if (
            donor_weights.shape != (len(points),)
            or not np.isfinite(donor_weights).all()
            or np.any(donor_weights < 0)
            or donor_weights.sum() <= 0
        ):
            raise ValueError("Donor weights must match boundary samples and have positive mass")
        donor_weights = donor_weights / donor_weights.sum()
    params: dict[str, float | int | str] = {
        "operation": "paired",
        "radius": radius,
        "region": -1 if region is None else region,
    }
    eligible = np.arange(len(points))
    if region is not None:
        if not 0 <= region < 6:
            raise ValueError("Region must be 0..5")
        regions = 2 * np.minimum((points[:, 0] * 3).astype(int), 2) + (points[:, 1] >= 0.5)
        eligible = eligible[regions == region]
    if not len(eligible):
        return DiskEdit(None, params, "No eligible boundary face in selected region")
    index = int(rng.choice(eligible) if face_pair is None else face_pair[0])
    angle = rng.uniform(-theta_max, theta_max)
    normal = normals[index]
    direction = np.cos(angle) * normal + np.sin(angle) * np.array([-normal[1], normal[0]])
    center = points[index] + 0.25 * radius * direction
    center += rng.uniform(-0.2, 0.2, 2) * geometry.h
    params.update(
        addition_index=index, add_x=float(center[0]), add_y=float(center[1]), angle=float(angle)
    )
    try:
        addition = apply_disk(geometry, center, radius, add=True)
    except ValueError as exc:
        return DiskEdit(None, params, str(exc))
    added = addition.inside & ~geometry.inside
    count = int(added.sum())
    if count == 0:
        return DiskEdit(None, params, "Empty addition")
    if face_pair is not None:
        donor = int(face_pair[1])
    else:
        donor = int(
            rng.integers(len(points))
            if donor_weights is None
            else rng.choice(len(points), p=donor_weights)
        )
    removal_center = points[donor] - 0.25 * radius * normals[donor]
    removal_center += rng.uniform(-0.2, 0.2, 2) * geometry.h
    y, x = np.indices(geometry.inside.shape)
    distances = ((x + 0.5) * geometry.h - removal_center[0]) ** 2 + (
        (y + 0.5) * geometry.h - removal_center[1]
    ) ** 2
    # Choose a radius based on all original interior cells; protected cells are
    # not silently omitted from a disk. The constraint checker rejects such cuts.
    original = np.flatnonzero(geometry.inside)
    if count >= len(original):
        return DiskEdit(None, params, "Removal consumes the domain")
    ordered = np.sort(distances.ravel()[original])
    threshold = (ordered[count - 1] + ordered[count]) / 2
    removal_radius = float(np.sqrt(threshold))
    if removal_radius < 3 * geometry.h - 1e-12 or removal_radius > 2 * radius:
        return DiskEdit(None, params, "Balanced removal radius outside [3h, 2r]")
    removed = (distances <= threshold) & geometry.inside
    # Ensure the operation is exactly two Boolean disks, including overlap checks.
    if np.count_nonzero(removed) != count or np.any(added & (distances <= threshold)):
        return DiskEdit(None, params, "Paired disks overlap or tie prevents exact area balance")
    candidate = geometry.with_mask(addition.inside & ~(distances <= threshold))
    params.update(
        donor_index=donor,
        remove_x=float(removal_center[0]),
        remove_y=float(removal_center[1]),
        remove_radius=removal_radius,
        changed_cells=count,
    )
    return DiskEdit(candidate, params)
