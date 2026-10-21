"""Shared-scale field comparisons and parameter/refinement curves."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

from oxygen_shape.geometry.boundary import BoundaryKind
from oxygen_shape.geometry.raster import RasterGeometry
from oxygen_shape.solvers.finite_volume import SolveResult
from oxygen_shape.viz.fields import face_segment


def plot_overview(
    cases: list[tuple[str, RasterGeometry, SolveResult]],
    Bi: float,
    path: Path,
) -> None:
    fig, axes = plt.subplots(
        len(cases), 3, figsize=(11, 2.8 * len(cases)), squeeze=False, constrained_layout=True
    )
    highest_flux = max(
        float(np.max(result.outward_face_flux)) / geometry.h for _, geometry, result in cases
    )
    norm = LogNorm(vmin=max(highest_flux * 1e-6, 1e-15), vmax=max(highest_flux, 1e-14))
    image = boundary = None
    for row, (name, geometry, result) in enumerate(cases):
        axes[row, 0].imshow(
            geometry.inside, origin="lower", extent=(0, 1, 0, 1), cmap="Greys", vmin=0, vmax=1
        )
        axes[row, 0].set_title(name.replace("_", " "))
        image = axes[row, 1].imshow(
            result.concentration,
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        n = geometry.inside.shape[0]
        axis = (np.arange(n) + 0.5) * geometry.h
        axes[row, 1].contour(
            axis,
            axis,
            result.concentration,
            levels=[0.1, 0.3, 0.5, 0.7, 0.9],
            colors="white",
            linewidths=0.4,
        )
        robin = [i for i, face in enumerate(result.faces) if face.kind == BoundaryKind.ROBIN]
        boundary = LineCollection(
            [face_segment(result.faces[i].key, geometry.h) for i in robin],
            cmap="magma",
            norm=norm,
            linewidths=1.7,
        )
        boundary.set_array(np.maximum(result.outward_face_flux[robin] / geometry.h, norm.vmin))
        axes[row, 2].add_collection(boundary)
        axes[row, 2].set_title(
            f"Q={result.metrics.physical_flux_Q:.4g}; eta={result.metrics.efficiency_eta:.4g}"
        )
        source = [
            face_segment(face.key, geometry.h)
            for face in result.faces
            if face.kind == BoundaryKind.SOURCE_DIRICHLET
        ]
        for ax in axes[row]:
            ax.add_collection(LineCollection(source, colors="red", linewidths=2))
            ax.set(xlim=(-0.015, 1.015), ylim=(-0.015, 1.015), aspect="equal")
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
    fig.colorbar(image, ax=axes[:, 1].tolist(), label="cell-center u")
    fig.colorbar(
        boundary, ax=axes[:, 2].tolist(), label="Robin j (log; display floor = max(j)*1e-6)"
    )
    fig.suptitle(f"Bi={Bi:g}; red = fixed source; n={n}")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_sweep(rows: list[dict], group: str, path: Path) -> None:
    n = max(row["n"] for row in rows)
    subset = [row for row in rows if row["comparison"] == group and row["n"] == n]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for name in dict.fromkeys(row["shape"] for row in subset):
        series = [row for row in subset if row["shape"] == name]
        axes[0].loglog(
            [r["Bi"] for r in series], [r["physical_flux_Q"] for r in series], "o-", label=name
        )
        axes[1].semilogx([r["Bi"] for r in series], [r["efficiency_eta"] for r in series], "o-")
    axes[0].set(xlabel="Bi", ylabel="Q", title=f"{group}: total flux (n={n})")
    axes[1].set(xlabel="Bi", ylabel="eta", title="Membrane utilization")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_refinement(rows: list[dict], group: str, path: Path) -> None:
    values = sorted({row["Bi"] for row in rows})
    fig, axes = plt.subplots(
        1, len(values), figsize=(3.2 * len(values), 3.4), squeeze=False, constrained_layout=True
    )
    for ax, Bi in zip(axes[0], values, strict=True):
        subset = [row for row in rows if row["comparison"] == group and row["Bi"] == Bi]
        for name in dict.fromkeys(row["shape"] for row in subset):
            series = [row for row in subset if row["shape"] == name]
            ax.plot(
                [r["n"] for r in series],
                [r["physical_flux_Q"] for r in series],
                "o-",
                label=name,
                markersize=3,
            )
        ax.set(xlabel="cells per side", ylabel="Q", title=f"Bi={Bi:g}")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=6)
    fig.savefig(path, dpi=160)
    plt.close(fig)
