# Contributing

Create a Python environment, install `.[dev]` and requirements-validation.txt,
then run `python -m ruff check src tests` and `python -m pytest -q` from the
repository root. The numerical workflow runs on Windows and Ubuntu.

Keep solver, geometry and optimizer changes separate. State which physical
objective and constraints a change affects, and use an analytical or independent
numerical check when changing a formula. Preserve conservation diagnostics and
report actual membrane length, not only a cap or penalized score.

Use a new experiment directory for changed protocols. Retain configs, seeds,
source hashes, failures, rejected evaluated edits and solver counts. Do not
edit executed sources during a run or weaken identity checks to resume an old
study. The [reproduction guide](docs/reproducibility.md) distinguishes current
development from the frozen release.

Public configs should generate their starting geometry or document an explicit
downloaded input. Do not commit absolute home-directory paths, process IDs,
scratch logs, local GPU inventories or experiment caches. Put useful figures
and tables in paper/ and larger reproducibility evidence in release assets.

Describe numerical limitations and negative outcomes plainly. A new high-flux
endpoint is not evidence of a global optimum or a superior policy. Consult
[AGENTS.md](AGENTS.md) and the [scientific scope](docs/morphology_limits.md).
