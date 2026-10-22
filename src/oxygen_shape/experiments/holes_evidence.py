"""Hash-checked solve cache and explicit accounting for interrupted square studies."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

from oxygen_shape.experiments.morphology_validation import cached_fem, cached_fvm
from oxygen_shape.optimization.trajectory_search import _json


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_frozen_sources(root: Path) -> None:
    repository = Path(__file__).resolve().parents[3]
    expected = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
    actual = {
        str(p.relative_to(repository)).replace("\\", "/")
        for p in (repository / "src/oxygen_shape").rglob("*.py")
    }
    actual.update(name for name in ("AGENTS.md", "pyproject.toml") if (repository / name).is_file())
    if actual != set(expected) or any(
        not (repository / name).is_file() or file_hash(repository / name) != digest
        for name, digest in expected.items()
    ):
        raise ValueError("Executed sources changed; use the recorded source snapshot")


class HoleEvidence:
    """One sequential solver at a time; pending work is never silently uncharged."""

    def __init__(self, root, config):
        self.root, self.config = root, config
        self.path = root / "solve_ledger.json"
        self.pending = root / "pending_solve.json"
        self.records = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.recovery = root / "interruption_accounting.json"
        self.losses = json.loads(self.recovery.read_text()) if self.recovery.exists() else []
        for name, row in self.records.items():
            if not (root / name).is_file() or file_hash(root / name) != row["sha256"]:
                raise ValueError(f"Changed or missing committed numerical evidence: {name}")
        if self.pending.exists():
            old = json.loads(self.pending.read_text())
            if old["path"] not in self.records:
                # A completed atomic cache can be adopted when its deterministic
                # request is replayed. An absent cache bounds lost work by 0..1.
                if not (root / old["path"]).exists():
                    self.losses.append({**old, "possible_uncommitted_solves": [0, 1]})
                    _json(self.recovery, self.losses)
                else:
                    self.records[old["path"]] = {
                        "sha256": file_hash(root / old["path"]),
                        "solver": old["solver"],
                        "seconds": None,
                        "recovered_atomic_cache": True,
                    }
                    _json(self.path, self.records)
            self.pending.unlink()
        self.new_solves = 0

    def get(self, key, geometry, Bi, *, fem_area=None):
        solver = "fvm" if fem_area is None else "fem"
        relative = f"cache/{key}.{solver}.npz"
        path = self.root / relative
        committed = relative in self.records
        if path.exists() and not committed:
            raise ValueError("Uncommitted cache has no pending-solve record")
        if committed and file_hash(path) != self.records[relative]["sha256"]:
            raise ValueError("Numerical cache changed since commitment")
        if not committed:
            _json(self.pending, dict(path=relative, solver=solver))
        started = perf_counter()
        result = (
            cached_fvm(path, geometry, Bi, self.config.solver)
            if fem_area is None
            else cached_fem(path, geometry, Bi, fem_area)
        )
        if not committed:
            self.records[relative] = dict(
                sha256=file_hash(path),
                solver=solver,
                seconds=perf_counter() - started,
                recovered_atomic_cache=False,
            )
            _json(self.path, self.records)
            self.pending.unlink()
            self.new_solves += 1
        return result

    def accounting(self):
        return dict(
            committed_fvm_solves=sum(r["solver"] == "fvm" for r in self.records.values()),
            committed_fem_solves=sum(r["solver"] == "fem" for r in self.records.values()),
            this_invocation_new_solves=self.new_solves,
            possible_uncommitted_solve_upper_bound=len(self.losses),
        )
