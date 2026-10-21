from oxygen_shape.geometry.baselines import matched_area
from oxygen_shape.geometry.constraints import ShapeConstraints, validate_shape


def test_initial_geometry_valid_and_protected_collar_frozen():
    g = matched_area("deep_cavity", 100).geometry
    constraints = ShapeConstraints(0.24, 4)
    assert validate_shape(g, constraints, g).valid
    mask = g.inside.copy()
    mask[44, 8] = True
    mask[22, 80] = False
    report = validate_shape(g.with_mask(mask), constraints, g)
    assert "Protected source collar changed" in report.violations


def test_disconnected_minimum_width_core_and_holes_rejected():
    g = matched_area("deep_cavity", 100).geometry
    mask = g.inside.copy()
    mask[45:55, 35:45] = False
    mask[49:51, 35:45] = True
    thin = g.with_mask(mask)
    constraints = ShapeConstraints(thin.area, 10)
    report = validate_shape(thin, constraints, g)
    assert any("opening" in reason for reason in report.violations)
    hole = g.inside.copy()
    hole[49:51, 80:82] = False
    candidate = g.with_mask(hole)
    report = validate_shape(candidate, ShapeConstraints(candidate.area, 10), g)
    assert "Internal hole is forbidden" in report.violations


def test_area_and_perimeter_limits_are_hard():
    g = matched_area("deep_cavity", 100).geometry
    assert not validate_shape(g, ShapeConstraints(0.25, 4), g).valid
    assert not validate_shape(g, ShapeConstraints(0.24, 2), g).valid
