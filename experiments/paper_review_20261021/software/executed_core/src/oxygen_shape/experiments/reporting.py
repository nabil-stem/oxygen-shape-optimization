"""Write complete experiments without overwriting previous results."""

import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import yaml

from oxygen_shape.experiments.config import RunConfig
from oxygen_shape.geometry.boundary import BoundaryKind
from oxygen_shape.geometry.raster import RasterGeometry
from oxygen_shape.solvers.finite_volume import SolveResult


def provenance() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[3]

    def git(*args: str) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(repository), *args], capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = git("status", "--porcelain")
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "git_status": status,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": {
            distribution.metadata["Name"]: distribution.version
            for distribution in importlib.metadata.distributions()
        },
        "source_manifest": "source_manifest.json",
        "source_snapshot": "source_snapshot/",
    }


def create_run(config: RunConfig, name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = config.output_root / f"{timestamp}_{name}_{uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "figures").mkdir()
    (directory / "config.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    (directory / "run.log").write_text(f"Started {name} at {timestamp} (UTC)\n", encoding="utf-8")
    save_source_snapshot(directory)
    return directory


def save_source_snapshot(directory: Path) -> None:
    """Retain exact solver sources even before the repository has its first commit."""
    package_root = Path(__file__).resolve().parents[1]
    repository = package_root.parent.parent
    files = list(package_root.rglob("*.py"))
    files.extend(
        repository / name
        for name in ("pyproject.toml", "AGENTS.md")
        if (repository / name).is_file()
    )
    manifest: dict[str, str] = {}
    for source in sorted(files):
        relative = source.relative_to(repository)
        content = source.read_bytes()
        destination = directory / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        manifest[relative.as_posix()] = hashlib.sha256(content).hexdigest()
    (directory / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write empty metrics")
    with path.open("w", newline="", encoding="utf-8") as stream:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_state(directory: Path, geometry: RasterGeometry, result: SolveResult) -> None:
    np.save(directory / "initial_geometry.npy", geometry.inside)
    np.save(directory / "final_geometry.npy", geometry.inside)
    np.save(directory / "concentration.npy", result.concentration)
    write_csv(
        directory / "boundary_faces.csv",
        [
            {
                "row": f.key.row,
                "col": f.key.col,
                "side": f.key.side.value,
                "kind": f.kind.value,
                "conductance": float(result.conductances[i]),
                "outward_integrated_flux": float(result.outward_face_flux[i]),
                "wall_concentration": float(result.wall_concentration[i]),
                "robin_flux_density": float(result.outward_face_flux[i] / geometry.h)
                if f.kind == BoundaryKind.ROBIN
                else None,
            }
            for i, f in enumerate(result.faces)
        ],
    )


def save_summary(directory: Path, summary: dict[str, Any]) -> None:
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (directory / "run.log").open("a", encoding="utf-8") as stream:
        stream.write("Completed successfully; diagnostics passed.\n")


def log_failure(directory: Path, error: Exception) -> None:
    with (directory / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(f"FAILED: {type(error).__name__}: {error}\n")
