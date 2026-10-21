# Focused novelty review, 21 October 2026

This is an AI-assisted targeted primary-source search and critical synthesis,
not a systematic review, human peer review, or evidence that no closer work exists.
The existing related-work matrix remains part of this assessment.

Queries used in the extension included `"shape optimization" "Robin" "flux"
"fin" diffusion 2024 2025`, `"shape optimization" "Robin" "heat flux" fin`,
`"oxygen" "shape optimization" "placental" 2016`, `"reinforcement learning"
"Robin" "shape optimization" diffusion`, and `"reinforcement learning"
"diffusion" "shape optimization" heat transfer`. Search results were followed
to publisher or author-preprint pages. Search silence was not treated as novelty.

| Closest relevant primary source | What is already established | Consequence here |
|---|---|---|
| [Marck, Nadin and Privat, 2014](https://doi.org/10.1137/130941377), [author preprint](https://arxiv.org/abs/1310.2214) | Inlet-flux fin optimization with resource constraints; a multidimensional Laplace/Robin formulation is discussed | Neither the PDE/objective nor maximizing useful exchange is new. Reduced-model nonexistence is not a theorem about this raster class. |
| [Bucur and Giacomini, 2016](https://doi.org/10.1016/j.anihpc.2015.07.001) | Robin free-boundary variational shape optimization | Energy and boundary regularity ideas have substantial prior theory. Our practical estimates are validation tools, not a new general existence theorem. |
| [Pearce et al., 2016](https://doi.org/10.1371/journal.pone.0165369), [primary full text](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0165369) | Oxygen-transfer geometry optimization using FEM and sequential quadratic programming for an axisymmetric capillary dilation | Evaluating a geometry edit through global oxygen transfer is established. Their pressure-driven blood flow and prescribed maternal concentration differ from our air-domain Robin sink. |
| [Keramati, Hamdullahpur and Barzegari, 2022](https://doi.org/10.1016/j.ijheatmasstransfer.2022.123112) | PPO and body-fitted CFD for a parameterized fin, with heat-transfer and pressure-drop objectives | RL-based thermal shape optimization is not new. Our online linear response regressor is not equivalent to their PPO, surrogate network or convection model. Their reported percentage gain is not a comparable baseline for our Q. |
| [Keramati and Hamdullahpur, 2022 author preprint](https://arxiv.org/abs/2208.07952) | Cooperative deep RL with Bezier geometry and a pretrained thermal surrogate | Generative geometric exploration and learned proposal selection are established. We cannot claim first use of learning to design exchange surfaces. |
| [Sapoval, Filoche and Weibel, 2002](https://doi.org/10.1073/pnas.122352499), [Filoche et al., 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2409382/) | Pulmonary screening and finite prefractal exchange surfaces | Screening, an unscreened length and prefractal comparisons are background, not newly discovered phenomena. |

## Narrow contribution and decision

The defensible candidate contribution is a reusable **finite-design benchmark**
that joins explicit admissibility, actual membrane accounting, signed globally
evaluated edits, paid search controls, full solve journals, independent same-domain
FEM, practical complementary ranking ranges, and inlet-boundary sensitivity.
Its empirical findings concern the tested designs and budgets: the response
ranker has no aggregate advantage over flux guidance; higher flux can accompany
lower wall utilization; and touching source/sink assumptions materially affect
interpretation. These observations are useful only when their exact scope and
comparison costs are exposed.

This is **not a demonstrated new optimization method**, a first Robin-flux
optimization, or a general negative result about RL. No search establishes a
global optimum. No direct comparison against an adjoint optimizer in a shared
smooth design space has been made; do not describe the five controls as an
exhaustive state-of-the-art algorithm benchmark. The single training sequence
and known development starts preclude confirmatory learning-generalization claims.

The working paper can be released for feedback as a computational benchmark.
An original-research submission still needs a subject expert to judge whether
the benchmark and its findings add enough beyond the closest studies. If that
gap is judged weak, sharpen one prespecified modeling question with a new
separately identified experiment; do not search until a preferred learner wins.
An honest technical report is an acceptable outcome if journal-level novelty
is not established.

## Changes made after this review

The introduction acknowledges oxygen-transfer SQP and deep-RL heat-exchanger
design directly. Release documentation distinguishes a public research artifact
from a peer-reviewed article. The reviewer brief asks explicitly about novelty,
the operational feature constraint, source-corner mathematics and what the
negative learning result actually supports. No new numerical claim was created
by this literature extension.
