# Data availability

The primary evidence is public in
[v0.1.0](https://github.com/nabil-stem/oxygen-shape-optimization/releases/tag/v0.1.0):
complete morphology and classical-control studies, evaluated rejected proposals,
atomic search/training journals, model parameters, initial/final masks,
configuration, seeds, numerical fields, validation caches, costs and executed
source snapshots. The scope is 285 returned endpoints and 50,721 committed
production solver calls. No unfavorable primary outcome is excluded.

Numbered ZIPs share one extraction root. release_manifest.json, SHA256SUMS.txt
and embedded per-file manifests identify their contents. The standalone verifier
checks 53,934 payload files; [instructions](docs/reproducibility.md) also explain
how to regenerate tables and figures without new PDE solves.

The repository's [paper view](paper/README.md) contains portable copies of the
main figures and numerical tables. Redundant absolute machine paths are omitted
from display tables. A curation manifest links each copy to the unchanged
release evidence. Full journals, historical execution details and original
provenance remain in the versioned assets instead of the development tree.

Earlier exploratory studies outside the manuscript's primary production scope
are documented in the [historical source tag](https://github.com/nabil-stem/oxygen-shape-optimization/tree/v0.1.0/docs);
their complete raw data are not all included in these archives. Original
user-supplied paper PDFs and third-party package binaries are not redistributed.
The generated benchmark contains no patient or participant data.

The later [square-island extension](paper/square_holes/README.md) is distributed
separately in the [dated square-hole release](https://github.com/nabil-stem/oxygen-shape-optimization/releases/tag/square-holes-20261022).
It retains the complete successful run, every paid trial, reflecting controls,
independent FEM fields, source snapshots, numerical audits and an earlier
interrupted attempt. The interrupted ledger is unusable; its incurred solver
cost is unknown and excluded from the completed-run totals. No caches from it
were adopted. This extension does not rewrite v0.1.0 or share its resource budget.

Author: Nabil ES-SAADI, independent researcher. Project code is [MIT](LICENSE);
original data, figures and research text are [CC BY 4.0](LICENSE-DATA.md).
Dependencies retain their [own terms](THIRD_PARTY.md).

No archival DOI has been assigned. A tagged GitHub release is the current
distribution mechanism; a later persistent-archive deposition should identify
these same verified data. The manuscript is an unreviewed working draft.
