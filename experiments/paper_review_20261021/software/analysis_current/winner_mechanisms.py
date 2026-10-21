"""Plot the selected endpoint and attribute measured wall absorption spatially."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.collections import LineCollection
from matplotlib.colors import PowerNorm
from matplotlib.patches import Patch

from oxygen_shape.experiments.morphology_validation import cached_fvm
from oxygen_shape.experiments.reporting import write_csv
from oxygen_shape.geometry.boundary import BoundaryKind
from oxygen_shape.geometry.raster import OFFSETS, refine_geometry
from oxygen_shape.geometry.research_families import make_research_geometry
from oxygen_shape.physics.model import SolverSettings


def run(root):
    if json.loads((root / "status.json").read_text())["phase"] != "finished":
        raise RuntimeError("Wait for final endpoint selection")
    config = yaml.safe_load((root / "config.yaml").read_text())
    winner = json.loads((root / "best_endpoint.json").read_text())
    directory = root / winner["path"]
    initial = make_research_geometry(winner["family"], seed=int(winner["geometry_seed"])).geometry
    best = initial.with_mask(np.load(directory / "final_geometry.npy", allow_pickle=False))
    output = root / "postprocess/winner_mechanisms"
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    bins = np.array([0, 0.12, 0.25, 0.5, 0.75, 1.000001])
    rows, metrics, inputs = [], {}, {}
    for index, (label, geometry) in enumerate((("initial", initial), ("best", best))):
        fine = refine_geometry(geometry, 4)
        file = directory / "validation" / f"{label}_f4.fvm.npz"
        if not file.exists():
            raise FileNotFoundError("Use existing checked fields; do not perform new solves")
        field = cached_fvm(file, fine, config["model"]["Bi"], SolverSettings(**config["solver"]))
        inputs[file.relative_to(root).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
        metrics[label] = vars(field.metrics)
        image = axes[0, index].imshow(
            field.concentration,
            origin="lower",
            extent=[0, 1, 0, 1],
            cmap="viridis",
            norm=PowerNorm(0.35, vmin=0, vmax=1),
        )
        axes[0, index].set_title(
            f"{label.capitalize()}: Q400={field.metrics.physical_flux_Q:.6f}\n"
            f"Robin P={field.metrics.robin_perimeter:.2f}, "
            f"eta={100 * field.metrics.efficiency_eta:.2f}%"
        )
        segments, density = [], []
        flux_bins, perimeter_bins = np.zeros(5), np.zeros(5)
        for face, flux in zip(field.faces, field.outward_face_flux, strict=True):
            if face.kind != BoundaryKind.ROBIN:
                continue
            dr, dc = OFFSETS[face.key.side]
            x = (face.key.col + 0.5 + dc / 2) * fine.h
            y = (face.key.row + 0.5 + dr / 2) * fine.h
            tangent = np.array([abs(dr), abs(dc)]) * fine.h / 2
            segments.append([np.array([x, y]) - tangent, np.array([x, y]) + tangent])
            density.append(flux / fine.h)
            slot = min(4, max(0, np.searchsorted(bins, x, side="right") - 1))
            flux_bins[slot] += flux
            perimeter_bins[slot] += fine.h
        if not np.isclose(flux_bins.sum(), field.metrics.physical_flux_Q, atol=1e-12):
            raise ArithmeticError("Spatial absorption bins do not sum to total flux")
        line = LineCollection(
            segments,
            array=np.array(density),
            cmap="viridis",
            norm=PowerNorm(0.35, vmin=0, vmax=config["model"]["Bi"]),
            linewidth=1.8,
        )
        axes[1, index].add_collection(line)
        axes[1, index].set_title(f"{label.capitalize()} Robin wall flux density")
        for slot in range(5):
            columns = ((np.arange(fine.inside.shape[1]) + 0.5) * fine.h >= bins[slot]) & (
                (np.arange(fine.inside.shape[1]) + 0.5) * fine.h < bins[slot + 1]
            )
            rows.append(
                dict(
                    shape=label,
                    x_min=float(bins[slot]),
                    x_max=float(bins[slot + 1]),
                    Q=float(flux_bins[slot]),
                    fraction_Q=float(flux_bins[slot] / flux_bins.sum()),
                    robin_perimeter=float(perimeter_bins[slot]),
                    area=float(fine.inside[:, columns].sum() * fine.h**2),
                )
            )
    for ax in axes[:, :2].flat:
        ax.plot([0, 0], [0.45, 0.55], color="red", lw=2.5)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x/L", ylabel="y/L", aspect="equal")
    fig.colorbar(
        image, ax=list(axes[0, :2]), shrink=0.75, label="u; common power display gamma=0.35"
    )
    fig.colorbar(line, ax=list(axes[1, :2]), shrink=0.75, label="j=Bi*u_wall; gamma=0.35")
    overlap = initial.inside & best.inside
    added, removed = best.inside & ~initial.inside, initial.inside & ~best.inside
    if added.sum() != removed.sum():
        raise ArithmeticError("Endpoint area redistribution is unbalanced")
    rgb = np.ones((*initial.inside.shape, 3))
    rgb[overlap], rgb[added], rgb[removed] = [0.75, 0.75, 0.75], [0.2, 0.7, 0.3], [0.8, 0.2, 0.65]
    axes[0, 2].imshow(rgb, origin="lower", extent=[0, 1, 0, 1])
    axes[0, 2].plot([0, 0], [0.45, 0.55], color="red", lw=2.5)
    axes[0, 2].set_title("Net equal-area redistribution")
    axes[0, 2].legend(
        handles=[
            Patch(color=[0.75, 0.75, 0.75], label="Shared air"),
            Patch(color=[0.2, 0.7, 0.3], label="Added air"),
            Patch(color=[0.8, 0.2, 0.65], label="Removed air"),
        ],
        fontsize=8,
        loc="upper right",
    )
    x = np.arange(5)
    for index, label in enumerate(("initial", "best")):
        values = [row["Q"] for row in rows if row["shape"] == label]
        axes[1, 2].bar(x + (index - 0.5) * 0.35, values, width=0.35, label=label.capitalize())
    axes[1, 2].set(
        xticks=x,
        xticklabels=["0-.12", ".12-.25", ".25-.5", ".5-.75", ".75-1"],
        xlabel="Wall x/L interval",
        ylabel="Absorbed Q in interval",
        title="Where measured absorption changes",
    )
    axes[1, 2].legend()
    axes[1, 2].grid(axis="y", alpha=0.2)
    fig.savefig(output / "mechanisms.png", dpi=180)
    fig.savefig(output / "mechanisms.pdf")
    plt.close(fig)
    write_csv(output / "spatial_flux_resources.csv", rows)
    initial_rows, best_rows = rows[:5], rows[5:]
    changes = [b["Q"] - a["Q"] for a, b in zip(initial_rows, best_rows, strict=True)]
    q0, q1 = metrics["initial"]["physical_flux_Q"], metrics["best"]["physical_flux_Q"]
    table = "\n".join(
        f"| {a['x_min']:.2f}-{min(1, a['x_max']):.2f} | {a['Q']:.6f} | {b['Q']:.6f} "
        f"| {delta:+.6f} | {a['robin_perimeter']:.3f} | {b['robin_perimeter']:.3f} |"
        for a, b, delta in zip(initial_rows, best_rows, changes, strict=True)
    )
    text = f"""# Selected geometry: measured absorption and resource redistribution

![Concentration, wall flux and net changes](mechanisms.png)

Selected returned endpoint: {winner["family"]}, {winner["method"]},
phase {winner["phase"]}, search seed {winner["seed"]}; charged search solves
{winner["pde_solves"]}. Q400 rises from {q0:.9f} to {q1:.9f}
({100 * (q1 / q0 - 1):.3f}%). Area remains 0.24. Actual perimeter changes
from {metrics["initial"]["robin_perimeter"]:.3f} to {metrics["best"]["robin_perimeter"]:.3f}.
Eta is diagnostic, not the optimization objective. Red marks the unchanged source.
Fields and boundary colors share explicitly labeled power normalizations.

| Wall x range | Initial Q | Best Q | Change Q | Initial membrane length | Best membrane length |
|---|---:|---:|---:|---:|---:|
{table}

The absorption bins sum to the global objective and locate its measured change.
The x<0.12 collar geometry is protected, but its absorption can change because
the PDE field is global. Net additions/removals have exactly equal area. Changes
in wall amount and accessibility occur together; these bins are not a shape
derivative or an isolated causal decomposition. The separate exact-perimeter
study supplies a within-start control for membrane growth.

Selection uses the completed study's independent FEM rule. Numerical bounds
must still be consulted for unresolved close rankings. This is a post-hoc
illustration, with unequal effort across study phases, not a globally optimal
shape or proof of superior natural acini. This figure reuses saved fields and
performs zero new PDE solves. A vector PDF is supplied alongside the PNG.
"""
    (output / "report.md").write_text(text, encoding="utf-8")
    for name in ("best_endpoint.json", "config.yaml", "source_manifest.json"):
        inputs[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    source = Path(__file__).resolve()
    shutil.copyfile(source, output / source.name)
    (output / "input_manifest.json").write_text(json.dumps(inputs, indent=2) + "\n")
    print(output / "report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    run(parser.parse_args().root)
