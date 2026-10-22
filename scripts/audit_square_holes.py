"""Independent saved-field replay for this declared unit-box square study.

No project solver or matrix assembly is imported. Uses NumPy/SciPy and the
stored fields, meshes, source geometry, config and hashed ledger only.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_fill_holes


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, message="Saved-field audit failed"):
    if not condition:
        raise ValueError(message)


def verify_fem_boundary(vertices, edges, markers, mask, reflecting):
    """Classify each stored boundary edge independently from the coarse mask."""
    n = len(mask)
    points = vertices[edges]
    tangent = points[:, 1] - points[:, 0]
    lengths = np.linalg.norm(tangent, axis=1)
    check(np.all(lengths > 0), "Zero-length FEM boundary edge")
    check(np.all(np.min(abs(tangent), axis=1) < 1e-12), "Non-raster boundary edge")
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]]) / lengths[:, None]
    middle = points.mean(axis=1)
    (plus, minus) = (middle + normal * 1e-07 / n, middle - normal * 1e-07 / n)

    def sample(array, xy):
        indices = np.floor(xy * n).astype(int)
        valid = ((indices >= 0) & (indices < n)).all(axis=1)
        answer = np.zeros(len(xy), dtype=bool)
        answer[valid] = array[indices[valid, 1], indices[valid, 0]]
        return answer

    (positive_air, negative_air) = (sample(mask, plus), sample(mask, minus))
    check(np.all(positive_air != negative_air), "Mesh segment is not an air boundary")
    excluded = np.where(positive_air[:, None], minus, plus)
    inner = sample(binary_fill_holes(mask) & ~mask, excluded)
    source = np.max(abs(points[:, :, 0]), axis=1) < 1e-12
    expected = np.where(source, 1, np.where(inner & reflecting, 3, 2))
    np.testing.assert_array_equal(markers, expected)
    check(abs(lengths[source].sum() - mask[:, 0].sum() / n) < 1e-10, "FEM source length changed")
    return float(lengths[expected == 2].sum())


def face_replay(mask, field, Bi, reflecting):
    n = mask.shape[0]
    check(mask.shape == (n, n), "Saved-field audit failed")
    h = 1 / n
    holes = binary_fill_holes(mask) & ~mask
    padmask = np.pad(mask, 1)
    padfield = np.pad(np.nan_to_num(field, nan=0), 1)
    padholes = np.pad(holes, 1)
    balance = np.zeros_like(field)
    (absorption, influx, perimeter) = (0.0, 0.0, 0.0)
    for dy, dx in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        neighbor = padmask[1 + dy : 1 + dy + n, 1 + dx : 1 + dx + n]
        neighbor_u = padfield[1 + dy : 1 + dy + n, 1 + dx : 1 + dx + n]
        internal = mask & neighbor
        balance[internal] += field[internal] - neighbor_u[internal]
        exposed = mask & ~neighbor
        source = np.zeros_like(mask)
        if dx == -1:
            source[:, 0] = mask[:, 0]
        neumann = (
            exposed & padholes[1 + dy : 1 + dy + n, 1 + dx : 1 + dx + n]
            if reflecting
            else np.zeros_like(mask)
        )
        reactive = exposed & ~source & ~neumann
        flux = Bi * h / (1 + Bi * h / 2) * field[reactive]
        balance[reactive] += flux
        source_flux = 2 * (field[source] - 1)
        balance[source] += source_flux
        absorption += float(flux.sum())
        influx -= float(source_flux.sum())
        perimeter += reactive.sum() * h
    return (absorption, influx, float(perimeter), float(abs(balance[mask]).max()))


def expected_mask(root, key, families):
    case = key.split("_B")[0]
    if case.startswith("base/"):
        return families[case[5:]]
    if re.search("/s\\d+$", case):
        (search, step) = case.rsplit("/", 1)
        history = json.loads((root / search / "history.json").read_text())
        item = history[int(step[1:])]
        holes = item["holes"] if int(step[1:]) == 0 else item["candidate_holes"]
        mask = np.load(root / "cases" / search / "outer.npy", allow_pickle=False).copy()
        for hole in holes:
            (r, c, w) = (hole[k] for k in ["row", "col", "width"])
            mask[r : r + w, c : c + w] = False
        return mask
    path = root / "cases" / case
    return np.load(
        path / ("outer.npy" if key.endswith("_base") else "geometry.npy"), allow_pickle=False
    )


def audit(root):
    check(
        json.loads((root / "status.json").read_text())["status"] == "complete",
        "Saved-field audit failed",
    )
    config = json.loads((root / "identity.json").read_text())
    ledger = json.loads((root / "solve_ledger.json").read_text())
    for relative, expected_hash in json.loads((root / "source_manifest.json").read_text()).items():
        check(
            not Path(relative).is_absolute() and ".." not in Path(relative).parts,
            "Unsafe source path",
        )
        check(
            digest(root / "source_snapshot" / relative) == expected_hash,
            f"Changed executed source snapshot: {relative}",
        )
    actual_caches = {p.relative_to(root).as_posix() for p in (root / "cache").rglob("*.npz")}
    check(actual_caches == set(ledger), "Unaccounted or missing cache files")
    families = {}
    for path in (root / "cases").rglob("geometry.json"):
        case = json.loads(path.read_text())
        source = case["source_faces"]
        check(
            all(s["col"] == 0 and s["side"] == "west" for s in source), "Saved-field audit failed"
        )
        outer = np.load(path.parent / "outer.npy", allow_pickle=False)
        reconstructed = outer.copy()
        for hole in case["holes"]:
            row, col, width = (hole[k] for k in ("row", "col", "width"))
            reconstructed[row : row + width, col : col + width] = False
        stored = np.load(path.parent / "geometry.npy", allow_pickle=False)
        np.testing.assert_array_equal(stored, reconstructed)
        check(
            hashlib.sha256(stored.tobytes()).hexdigest() == case["mask_sha256"],
            f"Changed geometry: {path}",
        )
        check(
            sorted(s["row"] for s in source) == list(np.flatnonzero(outer[:, 0])),
            "Source faces do not match the protected inlet",
        )
        if case["family"] in families:
            np.testing.assert_array_equal(families[case["family"]], outer)
        families[case["family"]] = outer
    result = dict(
        fvm_fields=0,
        fem_fields=0,
        max_fvm_residual=0.0,
        max_fem_residual=0.0,
        max_flux_conservation_error=0.0,
        max_fem_energy_error=0.0,
        new_pde_solves=0,
    )
    for relative, record in ledger.items():
        check(
            relative.startswith("cache/") and ".." not in Path(relative).parts,
            "Saved-field audit failed",
        )
        path = root / relative
        check(digest(path) == record["sha256"], relative)
        suffix = ".fvm.npz" if record["solver"] == "fvm" else ".fem.npz"
        key = relative[6 : -len(suffix)]
        found = re.search("_B([^_]+)", key)
        Bi = float(found.group(1)) if found else config["search_Bi"]
        expected = expected_mask(root, key, families)
        with np.load(path, allow_pickle=False) as data:
            if record["solver"] == "fvm":
                (mask, field) = (data["mask"], data["field"])
                factor = mask.shape[0] // config["n"]
                check(factor in [1, *config["refinement_factors"]], "Saved-field audit failed")
                expected = np.repeat(np.repeat(expected, factor, axis=0), factor, axis=1)
                np.testing.assert_array_equal(mask, expected)
                check(np.isfinite(field[mask]).all(), "Saved-field audit failed")
                check(
                    field[mask].min() >= -1e-10 and field[mask].max() <= 1 + 1e-10,
                    "Saved-field audit failed",
                )
                (q, source, perimeter, residual) = face_replay(mask, field, Bi, key.endswith("_N"))
                metrics = json.loads(str(data["metrics"]))
                check(abs(q - metrics["physical_flux_Q"]) < 1e-11, relative)
                check(abs(source - metrics["source_influx"]) < 1e-11, relative)
                check(abs(perimeter - metrics["robin_perimeter"]) < 1e-10, relative)
                check(
                    abs(mask.sum() / mask.shape[0] ** 2 - metrics["area"]) < 1e-12,
                    "Saved-field audit failed",
                )
                check(residual < config["solver"]["residual_tolerance"], (relative, residual))
                result["fvm_fields"] += 1
                result["max_fvm_residual"] = max(result["max_fvm_residual"], residual)
            else:
                (vertices, triangles, u) = (
                    data["vertices"],
                    data["triangles"],
                    data["concentration"],
                )
                (edges, markers) = (data["boundary_segments"], data["boundary_markers"])
                check(
                    np.isfinite(u).all() and u.min() >= -1e-10 and (u.max() <= 1 + 1e-10),
                    f"FEM concentration bounds: {relative}",
                )
                perimeter = verify_fem_boundary(
                    vertices, edges, markers, expected, key.endswith("_N")
                )
                check(
                    abs(perimeter - float(data["robin_perimeter"])) < 1e-10,
                    f"FEM boundary perimeter: {relative}",
                )
                p = vertices[triangles]
                det = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (
                    p[:, 2, 0] - p[:, 0, 0]
                ) * (p[:, 1, 1] - p[:, 0, 1])
                area = abs(det) / 2
                grads = (
                    np.stack(
                        [
                            p[:, [1, 2, 0], 1] - p[:, [2, 0, 1], 1],
                            p[:, [2, 0, 1], 0] - p[:, [1, 2, 0], 0],
                        ],
                        axis=2,
                    )
                    / det[:, None, None]
                )
                gradient = np.einsum("ti,tij->tj", u[triangles], grads)
                nodal = area[:, None] * np.einsum("tij,tj->ti", grads, gradient)
                reaction = np.zeros(len(u))
                np.add.at(reaction, triangles.ravel(), nodal.ravel())
                robin = edges[markers == 2]
                lengths = np.linalg.norm(vertices[robin[:, 0]] - vertices[robin[:, 1]], axis=1)
                (a, b) = (u[robin[:, 0]], u[robin[:, 1]])
                np.add.at(reaction, robin[:, 0], Bi * lengths * (2 * a + b) / 6)
                np.add.at(reaction, robin[:, 1], Bi * lengths * (a + 2 * b) / 6)
                fixed = np.unique(edges[markers == 1])
                np.testing.assert_allclose(vertices[fixed, 0], 0, atol=1e-12)
                np.testing.assert_array_equal(u[fixed], 1)
                free = np.ones(len(u), dtype=bool)
                free[fixed] = False
                residual = float(abs(reaction[free]).max())
                q = float((Bi * lengths * (a + b) / 2).sum())
                source = float(reaction[fixed].sum())
                energy = float(
                    (area * (gradient**2).sum(axis=1)).sum()
                    + (Bi * lengths * (a * a + a * b + b * b) / 3).sum()
                )
                check(abs(q - float(data["Q"])) < 1e-10, relative)
                check(abs(energy - q) < 1e-09, relative)
                check(residual < 1e-09, (relative, residual))
                check(
                    abs(area.sum() - expected.sum() / config["n"] ** 2) < 1e-10,
                    "Saved-field audit failed",
                )
                centers = np.floor(p.mean(axis=1) * config["n"]).astype(int)
                check(expected[centers[:, 1], centers[:, 0]].all(), relative)
                result["fem_fields"] += 1
                result["max_fem_residual"] = max(result["max_fem_residual"], residual)
                result["max_fem_energy_error"] = max(
                    result["max_fem_energy_error"], abs(energy - q)
                )
            error = abs(q - source) / max(abs(q), abs(source), 1e-14)
            check(error < 1e-09, (relative, error))
            result["max_flux_conservation_error"] = max(
                result["max_flux_conservation_error"], error
            )
        if (result["fvm_fields"] + result["fem_fields"]) % 500 == 0:
            print(
                "Replayed",
                result["fvm_fields"],
                "FVM and",
                result["fem_fields"],
                "FEM fields",
                flush=True,
            )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.root.resolve())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
