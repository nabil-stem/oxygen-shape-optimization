"""Comparable linear/power views of unchanged normalized concentration fields."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, PowerNorm

from oxygen_shape.geometry.raster import RasterGeometry
from oxygen_shape.viz.fields import face_segment


@dataclass(frozen=True)
class ConcentrationPanel:
    geometry: RasterGeometry
    field: np.ndarray
    title: str


def concentration_norm(mode: Literal["linear", "power"], gamma: float = 0.35) -> Normalize:
    if mode not in ("linear", "power") or not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("Use linear/power normalization and positive finite gamma")
    return Normalize(0, 1) if mode == "linear" else PowerNorm(gamma, vmin=0, vmax=1)


def _draw(ax, panel: ConcentrationPanel, norm: Normalize):
    g, field = panel.geometry, panel.field
    if field.shape != g.inside.shape or not np.array_equal(np.isfinite(field), g.inside):
        raise ValueError("A display field must be finite exactly inside its geometry")
    if field[g.inside].min() < -1e-10 or field[g.inside].max() > 1 + 1e-10:
        raise ValueError("Expected normalized concentration with source u=1")
    ny, nx = field.shape
    extent = (0, nx * g.h, 0, ny * g.h)
    image = ax.imshow(field, origin="lower", extent=extent, norm=norm, cmap="viridis")
    ax.add_collection(
        LineCollection([face_segment(f, g.h) for f in g.source_faces], colors="red", linewidths=2)
    )
    ax.set(title=panel.title, xlim=extent[:2], ylim=extent[2:], aspect="equal")
    ax.title.set_fontsize(9)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    return image


def plot_concentration_panels(
    panels: list[ConcentrationPanel],
    path: Path,
    *,
    mode: Literal["linear", "power"] = "power",
    gamma: float = 0.35,
    columns: int = 3,
) -> None:
    if not panels or columns < 1:
        raise ValueError("Supply panels and a positive column count")
    norm = concentration_norm(mode, gamma)
    rows = (len(panels) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.6 * columns + 1, 3.25 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, panel in zip(axes.flat, panels, strict=False):
        image = _draw(ax, panel, norm)
    for ax in list(axes.flat)[len(panels) :]:
        ax.set_visible(False)
    ticks = [0, 0.2, 0.4, 0.6, 0.8, 1] if mode == "linear" else [0, 0.01, 0.03, 0.1, 0.3, 0.6, 1]
    fig.colorbar(image, ax=axes, shrink=0.7, ticks=ticks, label="Actual normalized concentration u")
    mapping = "linear 0--1" if mode == "linear" else f"power colour scale, gamma={gamma:g}, 0--1"
    fig.suptitle(f"Identical saved fields; shared {mapping}\nSource u=1; red = protected source")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_contrast_comparison(panels: list[ConcentrationPanel], path: Path, *, gamma=0.35) -> None:
    if not panels:
        raise ValueError("Supply at least one panel")
    fig, axes = plt.subplots(
        len(panels), 2, figsize=(10, 4 * len(panels)), squeeze=False, constrained_layout=True
    )
    for column, mode in enumerate(("linear", "power")):
        norm = concentration_norm(mode, gamma)
        for row, panel in enumerate(panels):
            image = _draw(axes[row, column], panel, norm)
        ticks = (
            [0, 0.2, 0.4, 0.6, 0.8, 1] if mode == "linear" else [0, 0.01, 0.03, 0.1, 0.3, 0.6, 1]
        )
        fig.colorbar(
            image, ax=axes[:, column].tolist(), shrink=0.7, ticks=ticks, label="u (same data)"
        )
        axes[0, column].set_xlabel(
            "Linear 0--1" if mode == "linear" else f"Power gamma={gamma:g}, 0--1"
        )
        axes[0, column].xaxis.set_label_position("top")
    fig.suptitle("Source u=1 and fields unchanged; only the colour mapping differs")
    fig.savefig(path, dpi=150)
    plt.close(fig)
