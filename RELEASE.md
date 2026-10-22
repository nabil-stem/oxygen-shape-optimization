# Research release v0.1.0

The [published release](https://github.com/nabil-stem/oxygen-shape-optimization/releases/tag/v0.1.0)
contains the complete primary data, frozen code and an unreviewed manuscript.
Its 285 returned endpoints account for 50,721 committed production solver calls.
All 53,934 payload files have verification hashes.

Download every numbered code, main, classical and review ZIP plus
release_manifest.json and verify_release.py. With Python >=3.10:

```text
python verify_release.py DOWNLOAD_FOLDER --extract NEW_PARENT
```

All parts expand under `NEW_PARENT/oxygen_shape_release/`. The target must not
already exist. See [reproducibility](docs/reproducibility.md) for installation,
numerical regression and zero-PDE table/figure regeneration.

The current default branch is a curated development checkout. The published
tag and assets preserve the full executed evidence and historical records;
they are not rewritten by later repository cleanup. Old machine-specific
instructions in that frozen snapshot are historical provenance, not prerequisites
for running the current portable examples.

Project code is MIT; original data/figures/research text are CC BY 4.0.
Optional Triangle has [separate terms](THIRD_PARTY.md). Author: Nabil ES-SAADI,
independent researcher. The manuscript remains unreviewed and no archival DOI
has been assigned. See [data availability](DATA_AVAILABILITY.md).
