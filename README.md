# Oxygen diffusion shape optimization

A research codebase for a controlled question: **under fixed source and geometric
constraints, which two-dimensional shapes transfer more oxygen through a
permeable wall?** The model solves steady Laplace diffusion with a fixed
Dirichlet source and reactive Robin walls. Total absorbed flux Q is the objective;
membrane utilization eta is reported separately.

The package provides a conservative finite-volume reference, independent
triangular FEM checks, deterministic geometry generators and constrained search
baselines. The learning experiments include exploratory RL implementations and
an online response ranker. The current benchmark does not establish an overall
advantage for the tested ranker, a global optimum or natural-acinar optimality.

## Paper and data

- [Manuscript](docs/manuscript_draft.md) and [technical supplement](docs/manuscript_supplement.md).
- [Figures and numerical tables](paper/README.md), including all 285 returned endpoints.
- [v0.1.0 release](https://github.com/nabil-stem/oxygen-shape-optimization/releases/tag/v0.1.0): complete primary data, executed source versions, validation records and per-file hashes.
- [Square-island results](paper/square_holes/README.md): a separate controlled study of empty squares, wall activation, position maps and resource-matched relocation.
- [Reproduction instructions](docs/reproducibility.md) and [scientific limitations](docs/morphology_limits.md).

The released study accounts for 50,721 production solver calls. Its 324-test
regression passes on Windows and Ubuntu. Separate-folder regeneration reproduces
the checked tables and all ten PNG figures without new PDE solves. The manuscript
is an unreviewed working draft; no archival DOI has been assigned.

## Install

Use Python 3.10 or newer. Create a virtual environment:

```text
python -m venv .venv
```

Activate it using your shell (`source .venv/bin/activate` on POSIX shells or
`.venv\Scripts\Activate.ps1` in PowerShell), then install:

```text
python -m pip install -e ".[dev]" -r requirements-validation.txt
```

The optional validation requirements install Triangle for independent FEM checks.
The finite-volume solver itself needs only the core package dependencies.
Triangle retains [separate terms](THIRD_PARTY.md).

## Run

```text
python -m oxygen_shape.cli validate-1d --config configs/slab_1d.yaml
python -m oxygen_shape.cli solve --config configs/rectangle_2d.yaml
python -m oxygen_shape.cli optimize --config configs/optimize_disks.yaml
python -m oxygen_shape.cli study-holes --config configs/square_holes.yaml
```

Commands write reproducible outputs under `experiments/`, including configuration,
seeds, masks, fields, flux diagnostics and source provenance. Shipped example
configs generate their own starting geometries. The complete paper protocol is
in [configs/morphology_trajectories.yaml](configs/morphology_trajectories.yaml);
see [reproducibility](docs/reproducibility.md) for the full study commands.

The [square-island extension](docs/square_holes_protocol.md) compares maze-like
domains with empty squares, paired reflecting/reactive walls and fixed-resource
square relocation. This is a separate experiment from the hole-free v0.1.0 data.
Its [report](paper/square_holes/README.md) distinguishes added membrane from
improvements obtained by moving the same six squares, and includes independent
FEM checks and saved-field audits. Neither relocation heuristic is an RL policy.

## Validate and contribute

```text
python -m ruff check src tests
python -m pytest -q
```

The tests cover analytical diffusion, conservation, geometry constraints,
independent FEM, search accounting, interruption/resume and release verification.
[Contributor instructions](CONTRIBUTING.md) explain how to keep numerical changes
and experiment evidence reviewable.

| Directory | Contents |
|---|---|
| `src/oxygen_shape/` | Physics, geometry, solvers, search and experiment interfaces |
| `configs/` | Portable example configurations and the paper protocol |
| `tests/` | Numerical and reproducibility regression |
| `scripts/` | Current study analysis, validation and archive tools |
| `docs/` | Equations, protocols, limitations and research references |
| `paper/` | Curated public figures and tables |

[Documentation index](docs/README.md) · [Historical experiments](docs/history.md)

## Attribution

**Nabil ES-SAADI — independent researcher.** Cite the version using
[CITATION.cff](CITATION.cff). Project code is [MIT](LICENSE); original data,
figures and research text are [CC BY 4.0](LICENSE-DATA.md). Read the
[data availability statement](DATA_AVAILABILITY.md) and
[AI-assistance disclosure](docs/ai_assistance_disclosure.md).
