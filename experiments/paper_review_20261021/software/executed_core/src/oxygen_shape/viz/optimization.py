"""Physical search histories, disk outcomes, and learned-policy diagnostics."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from oxygen_shape.optimization.q_learning import QPolicy
from oxygen_shape.optimization.random_search import SearchResult


def plot_search(result: SearchResult, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
    count = [r["number_of_pde_solves"] for r in result.history]
    fields = [
        ("physical_flux_Q", "Q"),
        ("efficiency_eta", "eta"),
        ("area", "area"),
        ("robin_perimeter", "Robin perimeter"),
        ("conservation_error", "conservation error"),
    ]
    for ax, (key, title) in zip(axes.flat, fields, strict=False):
        ax.plot(count, [r[key] for r in result.history])
        ax.set(xlabel="PDE solves (including initial)", ylabel=title)
        ax.grid(True, alpha=0.3)
    labels = np.zeros(result.final_geometry.inside.shape, dtype=int)
    labels[result.initial_geometry.inside & result.final_geometry.inside] = 1
    labels[result.final_geometry.inside & ~result.initial_geometry.inside] = 2
    labels[result.initial_geometry.inside & ~result.final_geometry.inside] = 3
    axes[1, 2].imshow(
        labels,
        origin="lower",
        extent=(0, 1, 0, 1),
        vmin=0,
        vmax=3,
        cmap=ListedColormap(["white", "lightgray", "seagreen", "tomato"]),
    )
    axes[1, 2].set(title="Final edits: added green / removed red", xlabel="x", ylabel="y")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_learning(training: list[dict], policy: QPolicy, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    axes[0].plot(
        [r["episode"] for r in training], [100 * r["relative_gain"] for r in training], "o-"
    )
    axes[0].set(
        xlabel="training episode",
        ylabel="final flux gain (%)",
        title="Training flux gains (variable cost)",
    )
    for ax, values, title in zip(
        axes[1:],
        (policy.values, policy.visits),
        ("Learned action values", "Training visits"),
        strict=True,
    ):
        image = ax.imshow(values, aspect="auto", cmap="viridis")
        fig.colorbar(image, ax=ax)
        ax.set(title=title, xlabel="action: region*2 + radius bin", ylabel="observed state")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_benchmark(runs: list[tuple[str, int, SearchResult]], budget: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    x = np.arange(1, budget + 1)
    for name in dict.fromkeys(name for name, _, _ in runs):
        curves = []
        for policy, _, result in runs:
            if policy != name:
                continue
            values = {}
            for row in result.history:
                values[row["number_of_pde_solves"]] = row["physical_flux_Q"]
            curves.append([values[i] for i in x])
        curves = np.asarray(curves)
        median = np.median(curves, axis=0)
        (line,) = ax.plot(x, median, label=name)
        ax.fill_between(
            x,
            np.quantile(curves, 0.25, axis=0),
            np.quantile(curves, 0.75, axis=0),
            alpha=0.2,
            color=line.get_color(),
        )
    ax.set(
        xlabel="evaluation PDE solves (training excluded)",
        ylabel="best accepted Q",
        title="Held-out seeds: median and interquartile range",
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)
