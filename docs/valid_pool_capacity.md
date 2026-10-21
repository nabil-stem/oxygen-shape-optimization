# Why refilling creates choices but cannot cure poor raw proposal yield

This is a simple probability model, not a fit to experimental results and not
an implementation of a physics solver. Hold the incumbent fixed and suppose
independent raw proposals pass the full geometry checks with probability p.
Ignore repeated masks, already-solved candidates and incumbent copies for this
calculation. In an m-draw pool, N is binomial and

\[
\Pr(N\ge2)=1-(1-p)^m-mp(1-p)^{m-1}.
\]

This follows by subtracting the disjoint zero-pass and one-pass events from
one. The expected uncapped number of raw draws to collect k passes is k/p:
write the waiting time as the sum of k independent geometric waiting times,
each of mean 1/p. These are geometry passes, not distinct useful improvements.

With a c-draw cap, the probability of reaching k passes is

\[
\Pr(N_c\ge k)=1-\sum_{j=0}^{k-1}\binom cj p^j(1-p)^{c-j}.
\]

Requiring distinct, previously unsolved masks can only decrease the success
probability relative to counting all geometry passes. Thus the capped binomial
formula is an upper benchmark for the distinct-choice quota at a fixed
incumbent, under the independent-raw-draw assumption. The empirical per-run
fractions also mix changing incumbents, so they need not follow a single p.

| Assumed p | At least 2 passes in 8 | At least 2 in 256 | At least 4 in 256 | Uncapped mean draws for 2 / 4 |
|---|---:|---:|---:|---:|
| 0.01 | 0.00269 | 0.72635 | 0.25484 | 200 / 400 |
| 0.03 | 0.02234 | 0.99634 | 0.94993 | 66.7 / 133.3 |
| 0.05 | 0.05724 | 0.99997 | 0.99899 | 40 / 80 |

The table is evaluated directly from the displayed binomial expressions; it
does not estimate p from an optimizer run. The minimum eight-draw floor and
last-pool truncation are additional details of the actual bounded algorithm.

This explains two separate next-step questions. Refilling can expose choices
to an RL ranker, but must pay the extra generation/validation cost. Improving
the primitive or learning a feasibility-biased location/radius distribution
would instead try to increase useful alternatives per raw draw. That requires
its own comparison, a retained exploration component, unchanged hard checks and
incumbent-level data splits. Neither high classifier accuracy on mostly invalid
examples nor fewer invalid selected actions establishes a better optimizer.
