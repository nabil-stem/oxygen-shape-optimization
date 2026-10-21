# Permeability response on a fixed domain

This is a standard consequence of the Robin variational formulation, not a new
shape-optimization theorem. The domain, source trace and boundary partition are
fixed while beta=Bi>=0 changes. The source has positive measure, so the mixed
problem is coercive even at beta=0. The finite polygon/raster domains in this
study satisfy the required bounded-domain assumptions.

Write

    E_beta(v) = integral_Omega |grad v|^2 + beta*integral_R v^2,
    Q(beta) = min_{v|D=1} E_beta(v).

Green's identity and the boundary conditions give E_beta(u_beta)=Q(beta),
the source influx and Robin absorption. For each admissible v, E_beta(v) is
affine and nondecreasing in beta. Its infimum Q is therefore concave and
nondecreasing. The admissible constant trial function v=1 gives Q<=beta*P_R.
At beta=0, u=1 and Q=0.

Differentiating the minimized energy for finite beta gives the envelope identity

    Q'(beta) = integral_R u_beta^2.

The derivative of u contributes zero because it has zero source trace and
the state satisfies the weak equation. In particular Q'(0)=P_R. For each fixed
domain, Q/(beta*P_R) tends to one as beta tends to zero; this does not specify
a uniform small-beta threshold across different shapes and source access paths.

Concavity and Q(0)=0 imply that Q(beta)/beta is nonincreasing. Hence

    eta(beta) = Q(beta)/(beta*P_R)

is nonincreasing for a fixed domain. Its decrease is compatible with rising
total flux. Concavity does not imply a finite beta-to-infinity limit: the
touching-source-corner strip provides a counterexample to that additional claim.
Changing membrane measure, geometry or the boundary partition changes this
comparison problem and must be reported separately.

For a length-one slab with unit reactive end measure,

    Q(beta)=beta/(1+beta),
    Q'(beta)=1/(1+beta)^2=u_wall^2,
    eta(beta)=1/(1+beta).

This analytically checks the envelope and utilization identities. The separate
saved-catalog check also verifies nonnegative flux secants, decreasing secant
slopes and decreasing eta for every fixed geometry/grid curve, allowing only
a stated floating tolerance. These checks add no PDE solves and are not new
statistical observations. See the established Robin-energy literature cited
in related_work_assessment.md for the variational context.
