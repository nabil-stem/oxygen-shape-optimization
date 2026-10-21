"""Solver admissibility and explicit raster constraints for shape search."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_fill_holes, binary_opening, distance_transform_edt, label

from oxygen_shape.geometry.boundary import extract_boundary
from oxygen_shape.geometry.connectivity import component_count
from oxygen_shape.geometry.raster import RasterGeometry


@dataclass(frozen=True)
class ValidationReport:
    violations: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.violations

    def require_valid(self) -> None:
        if self.violations:
            raise ValueError("Invalid geometry: " + "; ".join(self.violations))


def validate_geometry(
    geometry: RasterGeometry,
    *,
    reference: RasterGeometry | None = None,
) -> ValidationReport:
    violations: list[str] = []
    try:
        extract_boundary(geometry)
    except ValueError as exc:
        violations.append(str(exc))
    if not geometry.source_faces:
        violations.append("At least one exposed Dirichlet source face is required")
    if component_count(geometry) != 1:
        violations.append("Air domain must be one face-connected component reachable from source")
    if reference is not None:
        if geometry.h != reference.h or geometry.inside.shape != reference.inside.shape:
            violations.append("Protected source grid and spacing changed")
        if set(geometry.source_faces) != set(reference.source_faces):
            violations.append("Protected source location or length changed")
        if set(geometry.neumann_faces) != set(reference.neumann_faces):
            violations.append("Protected impermeable boundary classification changed")
    return ValidationReport(tuple(violations))


@dataclass(frozen=True)
class ShapeConstraints:
    target_area: float
    perimeter_limit: float
    minimum_feature_width: float = 0.04
    protected_length: float = 0.12
    area_tolerance: float = 1e-12
    forbid_holes: bool = True

    def __post_init__(self) -> None:
        values = (
            self.target_area,
            self.perimeter_limit,
            self.minimum_feature_width,
            self.protected_length,
            self.area_tolerance,
        )
        if any(not np.isfinite(value) or value <= 0 for value in values):
            raise ValueError(
                "Constraint lengths, budgets, and tolerances must be positive and finite"
            )


def raster_perimeter(geometry: RasterGeometry) -> float:
    padded = np.pad(geometry.inside, 1)
    total = np.count_nonzero(padded[1:] != padded[:-1])
    total += np.count_nonzero(padded[:, 1:] != padded[:, :-1])
    return float((total - len(geometry.source_faces) - len(geometry.neumann_faces)) * geometry.h)


def validate_shape(
    geometry: RasterGeometry,
    constraints: ShapeConstraints,
    reference: RasterGeometry,
) -> ValidationReport:
    violations = list(validate_geometry(geometry, reference=reference).violations)
    if geometry.inside.shape != reference.inside.shape or geometry.h != reference.h:
        return ValidationReport(tuple(violations))
    x = (np.arange(geometry.inside.shape[1]) + 0.5) * geometry.h
    protected = x < constraints.protected_length
    if not np.array_equal(geometry.inside[:, protected], reference.inside[:, protected]):
        violations.append("Protected source collar changed")
    if abs(geometry.area - constraints.target_area) > constraints.area_tolerance:
        violations.append("Area tolerance exceeded")
    perimeter = raster_perimeter(geometry)
    if perimeter <= 0 or perimeter > constraints.perimeter_limit + 1e-12:
        violations.append("Robin perimeter budget violated")
    if constraints.forbid_holes and not np.array_equal(
        binary_fill_holes(geometry.inside), geometry.inside
    ):
        violations.append("Internal hole is forbidden")
    violations.extend(feature_violations(geometry, constraints.minimum_feature_width))
    return ValidationReport(tuple(violations))


def feature_violations(geometry: RasterGeometry, minimum_feature_width: float) -> list[str]:
    violations: list[str] = []
    radius = minimum_feature_width / 2
    cells = int(np.ceil(radius / geometry.h))
    yy, xx = np.ogrid[-cells : cells + 1, -cells : cells + 1]
    disk = (xx * geometry.h) ** 2 + (yy * geometry.h) ** 2 <= radius**2 + 1e-15
    core = binary_opening(geometry.inside, structure=disk)
    # Operational feature criterion: opening must retain a connected source core,
    # and no long narrow appendage may survive only as an opening residual.
    _, components = label(core)
    if components != 1:
        violations.append("Minimum-feature opening core is empty or disconnected")
    elif not any(core[f.row, f.col] for f in geometry.source_faces):
        violations.append("Minimum-feature opening disconnects the source")
    else:
        distance = distance_transform_edt(~core, sampling=geometry.h)
        allowance = radius / 2 + np.sqrt(2) * geometry.h
        if float(distance[geometry.inside].max()) > allowance + 1e-12:
            violations.append("Thin appendage exceeds minimum-feature residual allowance")
    return violations
