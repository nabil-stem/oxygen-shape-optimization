# Reproduce and validate

## Install and test the current checkout

In an activated Python >=3.10 environment, from the repository root:

```text
python -m pip install -e ".[dev]" -r requirements-validation.txt
python -m ruff check src tests
python -m pytest -q
```

Triangle is optional for the finite-volume package and required by the
independent FEM validation; see [dependency terms](../THIRD_PARTY.md).
The [CI workflow](../.github/workflows/numerics.yml) runs on Windows and Ubuntu.
The released 324-test suite passed on both platforms in
[this recorded run](https://github.com/nabil-stem/oxygen-shape-optimization/actions/runs/39670928717).
These tests run small numerical checks, not the full production matrix.

## Download the frozen paper evidence

From [v0.1.0](https://github.com/nabil-stem/oxygen-shape-optimization/releases/tag/v0.1.0),
download all numbered code, main, classical and review ZIPs plus
release_manifest.json. The standalone verify_release.py asset needs only the
Python standard library. For example:

```text
python scripts/verify_release.py downloads/v0.1.0 --extract reproduced
```

The target must be a new directory. All parts expand under
`reproduced/oxygen_shape_release/`; every archive and payload hash is checked.
This is a self-contained historical checkout, separate from current development.
It contains the original raw journals, masks, model parameters, field caches,
configs, environment inventories, executed sources and full review packet.

The source release and its SHA256SUMS.txt identify 53,934 payload files,
285 returned endpoints and 50,721 committed production solver calls. v0.1.0
also includes release_validation.json and release-validation.zip with the
recorded local/hosted tests, archive checks and relocation test.

## Regenerate paper tables and figures

From the extracted `oxygen_shape_release/` folder in a compatible environment:

```text
python scripts/paper_tables.py experiments/20261021T122626095890Z_morphology_98e4aafa experiments/20261021T141950978652Z_classical_72101312 experiments/regenerated_paper
python -m scripts.paper_figures experiments/regenerated_paper experiments/20261021T122626095890Z_morphology_98e4aafa
```

The collector verifies original field hashes. Historical Windows paths are
resolved only within the two explicitly supplied study directories. The figure
view requires existing checked fields; it does not replace missing data with
new PDE solves. The release relocation check disabled solver entry points and
reproduced all 15 checked CSV tables and ten PNG figures exactly, excluding only
relocated path columns. Ten vector PDFs were also regenerated.

The [curated public view](../paper/README.md) copies these figures and numerical
tables. It removes redundant absolute-path metadata from display copies; its
manifest records source/output hashes and the transformations. Raw release
evidence is unchanged.

## Run a new experiment

The examples generate their own starting shapes:

```text
python -m oxygen_shape.cli solve --config configs/rectangle_2d.yaml
python -m oxygen_shape.cli optimize --config configs/optimize_disks.yaml
python -m oxygen_shape.cli benchmark-valid-pools --config configs/feasible_pools.yaml
```

To run the full primary protocol rather than a small example:

```text
python -m oxygen_shape.experiments.morphology_study --config configs/morphology_trajectories.yaml
```

The [protocol](morphology_protocol.md) specifies the matrix and constraints.
Output directories preserve configuration, source hashes, seeds, solver settings
and costs. Do not change sources/configs while a run is active. Resume only
under the same recorded identity; historical production predates the first Git
commit and must retain its original source snapshots. Current development
and later documentation changes are not retroactive production provenance.

The exact installed production environment is recorded with the release.
Cross-platform bitwise rerunning of the full search matrix is not promised.

## Square-island extension

This later experiment has a [separate protocol](square_holes_protocol.md),
[result report](../paper/square_holes/README.md) and
[dated evidence release](https://github.com/nabil-stem/oxygen-shape-optimization/releases/tag/square-holes-20261022).
Download every numbered extension ZIP and its release_manifest.json. Verify
and extract to a new directory using the standalone verifier, as above.
The completed study is under `oxygen_shape_release/experiments/`; the failed
attempt is separately labeled `interrupted_attempt/`. Byte verification of the
failed attempt does not make its damaged numerical ledger usable.

From the extracted checkout, audit all successful stored fields without a PDE
solve and recreate the portable result tables in a new output directory:

```text
python scripts/audit_square_holes.py PATH_TO_COMPLETED_RUN audit.json
python scripts/curate_square_holes.py PATH_TO_COMPLETED_RUN NEW_PUBLIC_FOLDER
```

The auditor independently recomputes FVM face balances and FEM element/edge
balances, boundary classifications, energy, source uptake, geometry and source
hashes. It requires NumPy and SciPy but neither imports nor invokes a PDE solver.
The curator preserves original figure/table bytes and derives endpoint resource
comparisons and refined pair interactions. Its scope is the declared 100/200/400
study. It does not create missing numerical evidence.

Regenerate plots on an extracted copy, from saved results only:

```text
python -c "from pathlib import Path; from oxygen_shape.experiments.holes_report import write_holes_report; write_holes_report(Path('PATH_TO_COMPLETED_RUN'))"
```

Do not point regeneration at the sole original evidence copy. Full `study-holes
--resume` also replays trial histories and checks configuration/source/cache
identities. Its frozen serialized configuration includes the original output
path; cross-platform path serialization can therefore prevent a full historical
resume. The saved-field auditor, report renderer and curator use explicit input
paths and avoid that historical path dependency. Use the archived numerical
source snapshot for any historical continuation, without weakening guards.

To execute the full protocol afresh:

```text
python -m oxygen_shape.cli study-holes --config configs/square_holes.yaml
```

The completed production sources are frozen at commit `d830114`; subsequent
audit/reporting scripts are separately identified by the release reporting
commit. The earlier interrupted attempt was preserved rather than repaired;
the fresh execution changed only its output folder to local storage outside
the synchronized workspace. Ledger corruption prevents recovering its solve
cost, which remains additional unknown work outside the fresh-run totals.
