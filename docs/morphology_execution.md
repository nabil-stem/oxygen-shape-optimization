# Execution and review record

Main study: `20261021T122626095890Z_morphology_98e4aafa`, launched 21 October
2026 at 12:26 UTC. The frozen [protocol](morphology_protocol.md) specifies
the algorithms, masks, budgets, selection rule and numerical checks.

The complete package regression passed 278 tests before launch. Additional
postprocessing tests subsequently passed 22 targeted checks: complementary
energy expressions, conducting-collar labels and bounds, exact journal/update
replay, tampered-label rejection and an analytical two-dimensional Robin strip.
The combined regression subsequently passed **306 tests in 640.20 seconds**,
finishing at 14:07 UTC. It includes six additional classical-control tests,
covering exact paid-solve accounting, interruption within a pool, uncommitted
work and the full search/refinement/FEM/audit/energy pipeline. An earlier
pipeline test caught an omitted provenance copy; this was fixed in the separate
extension before the passing gate. No frozen package source was modified.
The test XML is retained in `tmp/paper_gate_20261021/tests.xml` and copied into
the production extension's launch record with its SHA-256 hash.

The main search uses two below-normal-priority processes with one BLAS/OpenMP
thread each. At 12:54 UTC a separate single-thread process began the exact
predeclared endpoint checks for the first 90 completed jobs. Its input prefix
does not select geometries by flux or method performance. It fills the normal
identity-verified validation caches; the main driver later reuses those fields.
This changes scheduling only, not the prescribed checks or their total solver
cost. Its execution/source snapshot is in launch
`20261021T1255_prefill_validation`. Peak planned numerical concurrency during
this overlap is three single-thread processes on the eight-core CPU.

Further completed prefixes were checked in launches
`20261021T1304_prefill_135` (all 135 main endpoints),
range records `range_135_139` and `range_139_141` for six deep endpoints,
and `20261021T1404_prefill_deep` (the next eight deep endpoints).
The range-specific execution records identify the exact jobs and script hashes.
The regression suite could add one transient numerical test process during
this scheduling overlap; no extra production worker was added.

A separate review driver waits for the main report before running energy-bound
postprocessing, a full saved-trajectory audit, the fixed-geometry conducting-collar
ablation and analytical strip validation. Its launch is
`20261021T125150Z_morphology_review`. These post-hoc checks cannot alter training
or search decisions. Their additional costs are reported separately from the
frozen study; the first two use saved fields and perform zero new PDE solves.

The partial journal audit reproduces 8,195 saved PDE solutions and 3,008 model
updates across 67 completed training/search jobs. It is a partial audit and does
not imply the remaining study is finished. The final audit will cover every
committed training/search solution. Source hashes, state sequences, source
protection, exact area, membrane constraints, independent face balances,
accepted/current/best states and recursive regression updates are checked.

The exact running environment is retained in environment.json and
requirements-environment.txt inside the main run. Python is 3.10.0; the Git
commit is null because this repository still has no HEAD. Source snapshots and
hash manifests preserve the executed code without inventing commit history.
Public release, authorship and submission remain separate unfinished activities.

## Additional classical comparison, queued after the passing gate

The [extension protocol](classical_control_protocol.md) adds uniform-greedy and
paid-pool-greedy controls on the 45 known development starts. It was specified
after observing the original comparison and is not a fresh confirmatory study.
Ninety new searches use 129 solves each and receive the same endpoint checks.
Original fields are hashed and imported without being counted twice.

The first launcher, `20261021T140729Z_classical_61a372`, waited for the main study
and then failed because Get-FileHash was unavailable in its hidden PowerShell
session. It performed no numerical work and created no classical experiment.
The launcher-only hash check was replaced with .NET SHA-256, preserving the
failed launch record and all numerical sources. The new launch
`20261021T141948Z_classical_b9e775`, PID 32796, started run
`20261021T141950978652Z_classical_72101312` at 14:19 UTC.
It pins its queued sources and the passing 306-test XML. The separate
control implementation, full-batch resume protocol and production identities
are frozen before its production outcomes. Two single-thread workers execute
searches; the parent validates and audits completed jobs while they continue.
The source study's separate supplemental review may also run concurrently.
Neither scheduling nor post-hoc extension replaces the original results.

The original study finished all 195 jobs and endpoint checks at approximately
14:18 UTC, after 6,734.18 seconds. All budgets completed without interruptions.
It charged 455 warm-training, 35,715 search, 1,107 cached validation/catalog/
counterfactual FVM and 792 independent FEM solves: 38,069 in total. This frozen
summary precedes the separately charged collar/strip checks and the new controls.

## Completed review and extension

The extension finished successfully at 14:56 UTC: 11,610 search and 720
independent validation solves, all 90 budgets complete, no interruptions.
It audits every committed candidate field. The original full audit verifies
36,170 fields and exactly replays 12,288 model updates. Collar/strip checks add
322 calls, bringing combined production cost to 50,721.

After all numerical runs and reviews finished, figure QA identified pandas
label alignment shifting learning curves by one solve and omitting the final
value. A shared positional converter and four regression cases correct the
presentation. Before changing reporting sources, all 92 original and seven
extension source hashes were verified. Original plots, before/after source
diffs and unchanged numerical CSV/history hashes are retained under
paper_review_20261021/post_run_curve_fix. Only the reporting code changed after
the studies; fields, search decisions, weights, endpoint tables and costs did not.
Historical replay uses archived executed sources, and fresh reports use the
corrected converter. The corrected full regression record is retained separately.

Final corrected full regression: **310 passed in 511.75 seconds** on 21 October 2026.
The XML is retained in tmp/paper_gate_corrected_20261021/tests.xml and the review packet.
