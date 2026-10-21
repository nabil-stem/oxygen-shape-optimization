"""Field and boundary plots; plotting never participates in scientific logic."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from oxygen_shape.geometry.boundary import BoundaryKind
from oxygen_shape.geometry.raster import FaceKey, RasterGeometry, Side
from oxygen_shape.solvers.finite_volume import SolveResult


def face_segment(key: FaceKey, h: float) -> list[tuple[float, float]]:
    x, y = key.col * h, key.row * h
    if key.side == Side.WEST:
        return [(x, y), (x, y + h)]
    if key.side == Side.EAST:
        return [(x + h, y), (x + h, y + h)]
    if key.side == Side.SOUTH:
        return [(x, y), (x + h, y)]
    return [(x, y + h), (x + h, y + h)]


def plot_fields(geometry: RasterGeometry, result: SolveResult, path: Path) -> None:
    ny, nx = geometry.inside.shape
    extent = (0, nx * geometry.h, 0, ny * geometry.h)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    axes[0].imshow(geometry.inside, origin="lower", extent=extent, cmap="Greys", vmin=0, vmax=1)
    axes[0].set_title("Geometry / source (red)")
    image = axes[1].imshow(
        result.concentration, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=1
    )
    if (
        min(nx, ny) > 1
        and np.nanmax(result.concentration) - np.nanmin(result.concentration) > 1e-12
    ):
        axes[1].contour(
            (np.arange(nx) + 0.5) * geometry.h,
            (np.arange(ny) + 0.5) * geometry.h,
            result.concentration,
            levels=7,
            colors="white",
            linewidths=0.5,
        )
    fig.colorbar(image, ax=axes[1], label="u (cell center)")
    axes[1].set_title("Concentration and contours")
    robin = [i for i, f in enumerate(result.faces) if f.kind == BoundaryKind.ROBIN]
    if robin:
        collection = LineCollection(
            [face_segment(result.faces[i].key, geometry.h) for i in robin],
            cmap="magma",
            linewidths=3,
        )
        collection.set_array(result.outward_face_flux[robin] / geometry.h)
        axes[2].add_collection(collection)
        fig.colorbar(collection, ax=axes[2], label="Robin flux density")
    eta = result.metrics.efficiency_eta
    eta_text = "undefined" if eta is None else f"{eta:.4f}"
    axes[2].set_title(f"Q={result.metrics.physical_flux_Q:.6g}; eta={eta_text}")
    source = [
        face_segment(f.key, geometry.h)
        for f in result.faces
        if f.kind == BoundaryKind.SOURCE_DIRICHLET
    ]
    for ax in axes:
        ax.add_collection(LineCollection(source, colors="red", linewidths=3))
        ax.set(xlim=extent[:2], ylim=extent[2:], xlabel="x", ylabel="y", aspect="equal")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_slab_errors(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    h = [row["h"] for row in rows]
    for key, label in (
        ("concentration_error_inf", "cell concentration"),
        ("flux_density_error", "flux density"),
        ("wall_concentration_error", "wall concentration"),
    ):
        ax.loglog(h, np.maximum([row[key] for row in rows], 1e-17), "o-", label=label)
    ax.set(
        xlabel="h",
        ylabel="absolute error (plot floor 1e-17)",
        title="Linear slab: exact to roundoff on all grids",
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(path, dpi=160)
    plt.close(fig)
