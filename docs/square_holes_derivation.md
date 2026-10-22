# Square islands, boundary conditions and flux attribution

An empty square in a plot means material excluded from the air domain. Its
boundary is either reactive tissue (Robin) or an impermeable obstacle (Neumann).
It is not an extra oxygen source. A pore opening an air passage between chambers
is a different geometric edit; these islands are not models of pores of Kohn.

Let H be one or several strictly interior, separated squares. Keep the exterior
boundary and source unchanged and write Omega_H = Omega_0 minus H. Define Q0
on the original domain, QN on Omega_H with reflecting square walls, and QR on
the identical Omega_H with reactive square walls, using the original Bi.

## Energy argument and a useful falsification check

For source value one, diffusion coefficient one, and homogeneous Robin sinks,

\[
Q=\min_{v|_{\Gamma_D}=1}
\left(\int_\Omega |\nabla v|^2\,dA
+Bi\int_{\Gamma_R}v^2\,ds\right).
\]

The Euler equation is the weak Laplace/Robin problem. Test it against u-1 to
obtain energy(u)=Bi*integral_R(u)=Q. Restrict the original minimizer to Omega_H:
its energy loses integral_H(|grad u0|^2), while its exterior Robin term is
unchanged. Thus **QN <= Q0**. Adding a nonnegative inner Robin term on the same
air domain instead gives **QR >= QN**. Neither argument orders QR against Q0.
The assumptions exclude removing old reactive walls or moving the source.

The conservative FVM has the same minimum-energy property: each undirected
interior edge contributes (vi-vj)^2, each source face 2*(vi-1)^2, and each Robin
face GR*vi^2. Removing strictly interior cells/edges and reflecting the new faces
removes nonnegative terms; activating Robin walls adds nonnegative terms.
Its minimum is the computed source influx/total absorption. The inequalities
therefore also hold for these paired discrete problems, to solver roundoff.

The exact controlled decomposition is

\[
Q_R-Q_0=(Q_N-Q_0)+(Q_R-Q_N).
\]

The first term measures loss from the excluded air region with reflecting walls;
the second measures the effect of activating the new membrane, including the
global redistribution of concentration. It is not just the flux on the squares.
Independently store QR = Q_outer + sum_k Q_hole_k and Q_source. A square can
absorb oxygen while lowering uptake at other walls. All terms come from the
same conservative face fluxes as the solve.

For two squares define the finite interaction
I_R = Q(H1 union H2)-Q(H1)-Q(H2)+Q0, and similarly I_N. Report I_R-I_N as the
interaction of the activation increments. These are finite geometry responses,
not a formal topological derivative or proof of a unique screening mechanism.

## Independent analytical hole test: concentric annulus

For a<r<R, impose u(R)=1 and a reactive inner circle. The outward normal of the
air on r=a points toward the center, so u'(a)=Bi*u(a). Integrating radial Laplace
gives u=A*log(r/a)+B, with A=Bi*a*B and B=1/(1+Bi*a*log(R/a)). Consequently

\[
u(r)=\frac{1+Bi\,a\log(r/a)}{1+Bi\,a\log(R/a)},\qquad
Q=\frac{2\pi a Bi}{1+Bi\,a\log(R/a)}.
\]

Bi=0 gives u=1, Q=0. Polygonal annulus FEM converges to this smooth result as
the circle segmentation and triangle mesh are refined. Raster-square FVM/FEM
comparisons instead solve the same exact square polygons. Triangle receives
an explicit interior seed for each excluded component; otherwise it can silently
mesh the hole as air. Its [PSLG instructions](https://www.cs.cmu.edu/~quake/triangle.poly.html)
describe these seeds. The meshed air area is checked against the raster area.
