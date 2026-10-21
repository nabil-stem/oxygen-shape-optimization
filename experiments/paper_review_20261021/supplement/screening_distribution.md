# Why screened-uniform preserves the valid proposal distribution

Derived during the 2026-10-19 ablation, without changing its frozen sampler,
settings or protocol. This statement concerns a proposal law, not oxygen-flux
optimality or a guaranteed finite-budget speedup.

## Assumptions

At a fixed incumbent and history, raw disk edits G_1,...,G_m are independent
draws from the same distribution p. Let S mean passing the cheap screens and
F mean full admissibility, including any deterministic duplicate exclusion.
Require F to be a subset of S. The selector does not see full-validation
outcomes, future candidate responses or Q. Its random choices are independent
of the pool. These match the intended implementation; the audit checks the
necessary-screen relation on observed raw edits, not every possible geometry.

Let a=P(S), b=P(F), and epsilon be the probability of choosing slot zero
unconditionally. On the other branch, choose uniformly among S-passing slots,
falling back to slot zero only when all m slots fail S.

## Derivation

Conditional on a nonempty screened set, a uniformly chosen screened member
has law p(G | S), by exchangeability. The event of a nonempty set has probability
A_m=1-(1-a)^m. On an empty set no member can satisfy F. Thus for any measurable
B contained in F, with a>0,

\[
P(G_{selected}\in B)
=\epsilon P(B)+(1-\epsilon) A_m P(B)/a
=c_S P(B),
\quad c_S=\epsilon+(1-\epsilon)\frac{1-(1-a)^m}{a}.
\]

Consequently, for b>0,

\[
P(G_{selected}\in B\mid G_{selected}\in F)=P(B)/b.
\]

This is exactly the ordinary rejection sampler's conditional valid-edit law.
The full-valid selection probability is b*c_S. For a=1, c_S=1. If a=0 or b=0,
there are no valid edits and the conditional statement is undefined.

All m raw draws are generated even on the uniform branch in the current code.
The charged valid yield per raw draw is therefore b*c_S/m. Since
1-(1-a)^m <= m*a,

\[
\frac{c_S}{m}\le (1-\epsilon)+\epsilon/m < 1
\]

when m>1 and epsilon>0. This rule cannot improve valid yield per raw draw over
ordinary rejection sampling under these assumptions. It can save wall time by
replacing expensive full checks with cheap screening on unselected edits.
Actual generation, screening, storage and solve costs still need measurement.

Minimum-perimeter selection is not uniform among screened slots. Its weight
depends on candidate perimeter, so it generally changes the valid-edit law.
That is the bias separated by this ablation.

## Implications and limits

If every full-valid edit passes S, then at each fixed history screened-uniform
preserves the next valid proposal kernel, including deterministic duplicate
exclusion. With no binding raw/time cap and the same greedy global-Q acceptance,
its search transition law matches ordinary rejection sampling. It does not
produce an identical seeded trajectory, and it does not imply better flux at
equal PDE solves. Different five-seed sample medians are compatible with equal
underlying transition laws.

Finite raw-draw caps can bind differently, and conditioning on completion can
change comparisons. Equal-wall-time stopping is also a different experiment.
Non-iid pools, response-informed screening, deterministic nonuniform selection,
or cheap screens that reject a fully valid edit invalidate this argument.

## Exact finite example

The companion check enumerates all ordered two-draw pools from four proposal
types with probabilities (0.1,0.2,0.3,0.4), cheap-pass set {A,B,C}, full-valid
set {A,B}, and epsilon=0.25. Candidate perimeters order A<C<B; D fails screens.
All arithmetic uses exact rational numbers.

Ordinary rejection gives valid-type probabilities A=1/3, B=2/3.
Screened-uniform gives exactly the same conditional law, with full-valid
selection probability 39/100. Minimum-perimeter gives A=67/147, B=80/147,
with full-valid selection probability 147/400. These probabilities check the
derivation; they are not measurements of the pulmonary geometry sampler.

Run `python scripts/check_screening_distribution.py` to regenerate
`docs/data/screening_distribution_check.json`. No PDE solves are used.
