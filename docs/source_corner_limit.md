# An analytical 2D check of the absorbing inlet corners

The large-Bi description in the original project plan needs a qualification.
The usual finite transport-limited flux presumes compatible separated source
and sink boundary data. Our original raster inlet has Robin walls touching
both endpoints of a source held at u=1. In the Bi-to-infinity limit these walls
become u=0 directly beside u=1, creating discontinuous Dirichlet data.

For a wedge with angle alpha, the local angular solution between a source ray
and a zero-concentration ray is u=1-theta/alpha. It is harmonic away from the
vertex and has |grad u|=1/(alpha*r). The wall flux integrated from radius epsilon
to R is log(R/epsilon)/alpha. Its energy diverges as epsilon tends to zero.
Thus a finite transport plateau cannot be assumed for touching ideal source and
perfect sink boundaries. Finite positive Bi remains a well-posed Robin problem.

This is a mathematical qualification, not a reason to discard completed finite-Bi
experiments. Source corners are identical across the controlled geometry bank;
their common contribution can nevertheless conceal distal geometry differences.
The separate conducting-collar ablation tests that modeling choice. No frozen
training, geometry constraints or search objective is changed retroactively.

## A convergent finite-Bi strip solution

Consider 0<x<L, -a<y<a, with u(0,y)=1, u_x(L,y)=0 and
u_y(x,a)+Bi*u(x,a)=0; the lower wall has the corresponding outward Robin rule.
The source-to-wall junctions are right angles, as in the protected inlet.
Even separated modes have transverse shape cos(k_n*y), where

    t_n=k_n*a,   t_n*tan(t_n)=b=Bi*a,
    n*pi < t_n < n*pi+pi/2, n=0,1,... .

Orthogonality gives source coefficient

    A_n = (2*sin(k_n*a)/k_n) / (a+sin(2*k_n*a)/(2*k_n)).

Then

    u = sum A_n*cos(k_n*y)*cosh(k_n*(L-x))/cosh(k_n*L),
    Q = sum 4*sin(t_n)^2*tanh(t_n*L/a)/(t_n+sin(t_n)*cos(t_n)).

Every Q term is positive. For n>=1, sin(t_n)^2=b^2/(t_n^2+b^2),
sin(t_n)*cos(t_n)>0 and t_n>n*pi. The tail after N>=2 terms satisfies

    0 <= Q-Q_N <= 4*b^2/pi^3 * sum_{n=N}^infinity n^(-3)
                  <= 2*b^2/(pi^3*(N-1)^2).

The implementation solves for delta_n=t_n-n*pi in (0,pi/2), using
(n*pi+delta_n)*sin(delta_n)-b*cos(delta_n)=0 to avoid tangent poles.
Reported series intervals add a practical floating-point guard, not a
directed-rounding certificate. Finite-volume refinement is compared against
this independent analytical 2D flux at several finite permeabilities.

For each fixed n, Bi-to-infinity gives t_n->(n+1/2)*pi. The limiting Q terms
decay as 4/((n+1/2)*pi), whose sum diverges. Positivity therefore establishes
that Q has no finite large-Bi plateau in this strip. The local wedge suggests
the leading logarithmic behavior; no fitted logarithmic law is used in search.

These expressions follow separation of variables for this specific strip.
Mixed Robin corner singularities are established analysis, not a novel general
claim here: see [Jimenez-Alfaro, Villalba and Mantic (2020)](https://doi.org/10.1007/s10704-020-00443-5)
and [Pina-Leon, Mantic and Jimenez-Alfaro (2025)](https://arxiv.org/abs/2505.22585).
Those papers analyze corner expansions; the strip calculation above is derived
independently and does not claim their mechanical boundary data match our oxygen
problem. Unlike increasing source amplitude, changing where absorption begins
changes the normalized physical problem and needs its own stated purpose.
