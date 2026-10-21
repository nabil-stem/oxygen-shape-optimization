"""Geometry-only preview. No PDE solve or optimization outcome is implied."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from oxygen_shape.geometry.constraints import raster_perimeter
from oxygen_shape.geometry.research_families import FAMILIES, make_research_geometry

root = Path(__file__).resolve().parents[1]
destination = root / "docs/figures/research_geometries.png"
destination.parent.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True)
for ax, family in zip(axes.flat, FAMILIES, strict=True):
    geometry = make_research_geometry(family).geometry
    ax.imshow(geometry.inside, origin="lower", extent=[0, 1, 0, 1], cmap="Greys", vmin=0, vmax=1)
    ax.plot([0, 0], [.45, .55], color="red", linewidth=3)
    ax.set(title=f"{family}\nA={geometry.area:.2f}; Robin P={raster_perimeter(geometry):.2f}",
           xlim=(0, 1), ylim=(0, 1), xticks=[0, .5, 1], yticks=[0, .5, 1])
fig.suptitle("Finite 2D geometric controls; red = fixed source, length 0.1", fontsize=13)
fig.savefig(destination, dpi=160, bbox_inches="tight")
plt.close(fig)
print(destination)
