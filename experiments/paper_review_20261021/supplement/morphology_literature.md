# Geometry families and how the literature informs this implementation

Reviewed 20 October 2026. The implemented shapes are controlled proxies, not
exact replicas of published meshes, anatomically measured acini, or claims of
fractal limits. The finite feature cutoff and actual membrane length are explicit.

| Source | Geometry / question | Transfer to this project | Important distinction |
|---|---|---|---|
| Filoche, Grebenkov, Andrade and Sapoval (2008), *Passivation of irregular surfaces accessed by diffusion* | Finite square-Koch curves and cubic-Koch surfaces; screening and progressive passivation | Finite square-Koch profile controls at levels 1/2 | Their passivation changes boundary activity. Our primary search holds uniform Robin permeability fixed and changes the air geometry. |
| Sapoval, Filoche and Weibel (2002), *Smaller is better—but not too small* | Pulmonary screening and diffusivity/permeability scale | Sweep Bi and compare accessible membrane with added geometric complexity | Does not establish that any arbitrary branching raster shape is an optimal acinus. |
| Hou and colleagues (2010), *Reverse Engineering of Oxygen Transport in the Lung: Adaptation to Changing Demands and Resources through Space-Filling Networks* | Space-filling network models linking structure and oxygen transport | Binary branching and finite Hilbert-like controls | A thickened planar path is a benchmark, not a three-dimensional alveolar membrane. |
| Fujita and Koshiyama (2025), supplied acinus paper | Polyhedral acinus generation; its 2D demonstration uses seeds on a hexagonal lattice | Hexagonal chamber and branching terminal hexagonal-sac proxies | We do not reproduce their graph optimization, length weighting, anatomical statistics or objective. A single hexagon is not their acinus model. |
| Nicolaou and Mulder (2023), *A probabilistic algorithm for optimising the steady-state diffusional flux into a partially absorbing body* | Convex exterior bodies, including sphere/spheroid; optimize reactivity distribution under a total-reactivity constraint | Use as related work for controlled Robin-flux optimization and independent validation | They optimize spatial reactivity on fixed shapes. We optimize interior geometry with fixed uniform Bi. |

Primary sources:

- [Filoche et al., PNAS 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2409382/).
- [Sapoval et al., PNAS 2002](https://doi.org/10.1073/pnas.122352499).
- [Hou et al., 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2928740/).
- [Fujita and Koshiyama, 2025](https://doi.org/10.1080/15502287.2025.2463673).
- [Nicolaou and Mulder, Scientific Reports 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10739912/).

The serpentine maze is an intentionally tortuous access control, not an alleged
published optimal acinus. The rectangular chamber controls complexity; the
asymmetric sac family varies branching while retaining fixed source/area rules.
Square-Koch level 2 uses declared signed-distance smoothing length 0.012 to
respect the finite raster feature cutoff. All families are checked for holes,
connectivity, area, source protection and operational feature width before use.

Our question is whether these proxies or their edited descendants produce high
total Q under declared constraints. We must retain unfavorable outcomes. Real
acinar design has additional constraints and transport mechanisms, including
blood-side oxygenation. [Swan and Tawhai's coupled model](https://pmc.ncbi.nlm.nih.gov/articles/PMC3043789/)
is a necessary counterpoint before making biological interpretations. The current
study optimizes Q; utilization eta remains diagnostic and is not the reward.
