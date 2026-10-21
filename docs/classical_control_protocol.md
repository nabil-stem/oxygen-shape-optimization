# Additional classical controls: development comparison extension

Written before any production outcome of these two controls, 21 October 2026.
The completed main comparison already informed the decision to add missing
controls. This extension uses the same known development bank and is not a
fresh confirmatory test. No settings are chosen to force a learning advantage.

Reuse all 45 comparison starts, original geometry and paired search seeds,
Bi=1, area=0.24, source/collar, unit box, Robin-perimeter cap 8 and the original
operational feature rule. Use the same candidate primitive mixture, radius
range, validity/refill checks, 500,000 raw-draw limit and 129-PDE solve budget.
The original three methods and their results remain unchanged and are imported
with explicit input hashes. Their independent fields are reused, not resolved.

Two added methods:

- uniform_greedy: uniformly choose one eligible candidate and accept only
  Q_candidate>Q_current+1e-7. No flux or learned ranking is used.
- pool_greedy: generate the same pool, randomize its eligible evaluation order
  and evaluate every member within the remaining PDE budget. Accept improving
  observed members as they arrive; the remaining candidates still come from
  that fixed original pool. At pool completion the current geometry is its best
  improving member. The last pool is explicitly truncated if the remaining
  budget cannot pay for all its eligible candidates. Unobserved candidates
  receive no labels. This is a paid conventional search, not a free oracle.

Both methods preserve best and current separately and journal each candidate
solve atomically. Pool metadata retains remaining candidate masks, rows and RNG
states, so interruption within a batch neither loses nor repeats committed
evaluations. Uncommitted interrupted work is bounded separately by 0-1 solve.
Completed or raw-budget-exhausted jobs return without further work on resume.

Ninety new searches cost at most 11,610 PDE solves. Each returned endpoint and
its start receive the same n200/400 FVM and two-density FEM checks: 720 additional
solves if all cases complete. New work is separate from imported historical
costs. All five methods are compared by paired Q400 and FEM differences against
the original greedy-flux control at equal maximum budgets. Preserve truncated
runs. Report raw draws, full checks, membrane changes, training cost and timing.
Concurrent runtimes are descriptive; do not claim causal equal-time speedups.

The source/identity and journal audit precede interpretation. Complementary
energy estimates identify resolved wins/losses or overlapping numerical ranges
on each fixed staircase. Old and new starts remain five geometry/search pairs
per family with a single shared training sequence. No crossed variance or
anatomical-population inference is added by calling this a larger comparison.

Two single-thread search workers are planned. The parent process performs the
prescribed endpoint validation and journal audit as each search finishes, while
workers continue queued jobs. Peak numerical concurrency is therefore three
single-thread processes for this extension. This changes scheduling, not
candidate selection or solver budgets. Timings remain descriptive.

The implementation and this protocol are frozen in a separate experiment before
workers start. The original package sources remain unchanged. Paper claims must
allow simple uniform or fully evaluated-pool search to win; these are standard
controls, not proposed new algorithms.
