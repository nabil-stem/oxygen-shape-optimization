# Complementary energy checks on a fixed raster domain

This derivation concerns the physical polygon formed by the union of raster
cells. It does not remove Cartesian boundary or perimeter bias.

With u=1 on the source, zero Neumann flux and Robin coefficient Bi>0, define

\[
E(v)=\int_\Omega |\nabla v|^2+Bi\int_{\Gamma_R}v^2,
\qquad v|_{\Gamma_D}=1.
\]

The weak solution minimizes E, and integration by parts gives E(u)=Q. A
conforming piecewise-linear FEM solution therefore supplies a primal upper
estimate E(u_h)>=Q in exact arithmetic on the same polygon.

For a vector field q with continuous normal flux across interior cell faces and
q.n=0 on Neumann boundaries, define

\[
J(q)=-2\int_{\Gamma_D}q.n
-\int_\Omega |q|^2
-Bi^{-1}\int_{\Gamma_R}(q.n)^2.
\]

Expanding squares and integrating by parts gives

\[
E(u)=\int_\Omega |\nabla u+q|^2
+Bi\int_{\Gamma_R}(u-q.n/Bi)^2+J(q)
+2\int_\Omega u\,\nabla\cdot q.
\]

If q is divergence-free, J(q)<=Q. More generally, the maximum principle gives
0<=u<=1. For a cellwise constant divergence and cell residual
r_K=integral_K div(q), this gives the conservative bound

\[
J(q)-2\sum_K|r_K|\le Q.
\]

The residual term avoids assuming an exactly equilibrated floating-point field.
It also makes the expression applicable to arbitrary stored candidate fields,
although a poor field will give a loose or negative lower bound.

Construct q as a rectangular Raviart--Thomas field: interpolate q_x linearly
between each cell's west/east normal components and q_y between south/north.
Shared finite-volume face fluxes provide normal continuity. Let w,e,s,n denote
the cell's integrated outward face fluxes; then

\[
\int_K|q|^2=(w^2-we+e^2+s^2-sn+n^2)/3,
\qquad r_K=w+e+s+n.
\]

The Robin complementary term is sum_faces(q_face^2/(Bi*h)). Source influx is
-sum_source(q_face). No image-gradient estimate is used. For an exactly balanced
finite-volume solution, algebra also predicts

\[
J(q)=Q_{FV}+\frac16\sum_K[(w+e)^2+(s+n)^2].
\]

The linear slab has equal opposite face fluxes and zero correction, recovering
the analytical Q=H*Bi/(1+Bi*L). Checks must cover several Bi, arbitrary slab
fields with nonzero residuals, and comparison with independently assembled FEM
on non-affine two-dimensional solutions before interpreting experiment bounds.

The implementation reports the mathematical residual correction plus a separate
floating-point guard. It is not an interval-arithmetic certificate. Bounds that
overlap within that guard do not resolve a ranking. These checks are additional
postprocessing, with their own source hash and validation record, and do not
change the frozen search, training labels, acceptance decisions or its budget.

Functional a posteriori estimation from reconstructed finite-volume solution and
flux fields is established methodology, not a new general theorem claimed here.
See [Cochez-Dhondt, Nicaise and Repin (2009)](https://www.cambridge.org/core/journals/mathematical-modelling-of-natural-phenomena/article/abs/posteriori-error-estimates-for-finite-volume-approximations/3C56999939188C3E2A3A71DF6814AE7B).
This note derives the particular Robin total-flux expression used by this project.
