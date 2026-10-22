# Source amplitude, screening and comparable concentration figures

A linear 0--1 color map can make strongly screened regions look similarly dark.
This is a display limitation. Increasing source concentration does not change
relative screening in the present homogeneous linear model.

## Derivation before implementation

Let DeltaC = C0-Cb > 0. For fixed geometry, D, W and a prescribed uniform Cb,
the normalized solution satisfies Laplace's equation, u=1 on the source,
du/dn+Bi*u=0 on the membrane and zero Neumann conditions elsewhere.
If the driving difference becomes a*DeltaC, a>0, then

\[
C_a-C_b=a(C_1-C_b),\qquad
u_a=\frac{C_a-C_b}{a\Delta C}=u_1.
\]

Indeed, derivatives and the homogeneous Robin/Neumann conditions are linear.
Multiplying a solution relative to Cb by a therefore satisfies exactly the new
Dirichlet data and the same homogeneous wall conditions. Uniqueness follows
by applying the energy identity to a difference with zero source data:
integral(|grad w|^2)+Bi*integral_R(w^2)=0. A connected source-pinned domain
therefore has w=0, also when Bi=0.

Dimensional membrane flux is W*DeltaC*u_wall. Thus dimensional total transfer
is multiplied by a, while dimensionless Q and eta remain unchanged. Rankings
and relative flux gains across geometries remain unchanged when the same a is
applied to each. For Cb != 0 it is the difference C0-Cb that must be scaled,
not C0 alone. These conclusions assume the present linear constant-coefficient
model, not nonlinear uptake or concentration-dependent permeability.

The independent numerical check uses v=(C-Cb)/DeltaC_reference, so v_D=a.
The finite-volume matrix is independent of a: A*v_a=a*b. Its field v_a/a
and flux Q_v/a must equal the source-one reference. The ordinary solver and
optimizer retain u_D=1. For the analytical slab of length L,

\[
v_a(x)=a\left(1-\frac{Bi}{1+BiL}x\right),\qquad
q_a=a\frac{Bi}{1+BiL}.
\]

The matrix check independently solves scaled right-hand sides; it does not
merely multiply an already computed field. A linearity reference is
[Moore's UCSB lecture notes, Laplace equation and superposition](https://web.math.ucsb.edu/~moore/pde/pde.pdf).

## Adopt the valid part of the suggestion

Preserve the linear shared 0--1 view and add a power-normalized color view with
the same gamma, lower bound 0 and upper bound 1 for every geometry. The actual
field, Q, eta and source are unchanged. Tick labels show actual u, not u^gamma.
The nonlinear color mapping is explicitly labeled, as recommended by
[Matplotlib's normalization documentation](https://matplotlib.org/stable/tutorials/colors/colormapnorms.html).
No per-panel autoscaling is allowed, because that would make identical colors
represent different concentrations across geometries.

Here gamma=0.35 maps u=0.01 to color coordinate approximately 0.20 instead of
0.01, revealing the low-concentration structure. This is contrast enhancement,
not additional oxygen penetration. In the supplied deep-cavity example the
minimum saved concentration is about 0.0101 and the median about 0.0253:
dark purple does not mean zero concentration.

Changing source length or location would change the admissible physical problem
and requires a separate comparison with the same source for every geometry.
Changing Bi would change the transport/membrane balance and can change rankings.
Neither physical parameter is changed solely to make a figure brighter.
