# Independent finite-element reference

The bulk equation is -Laplacian(u)=0. On the source u=1; on the reactive wall
du/dn+Bi*u=0; on optional impermeable edges du/dn=0. Integration by parts against
a test function v that vanishes at the source gives

\[
\int_\Omega \nabla u\cdot\nabla v\,dA+
Bi\int_{\Gamma_R}uv\,ds=0.
\]

The independent solver uses continuous piecewise-linear triangular basis functions.
For triangle area A and the standard coordinate-difference vectors b,c, its
diffusion element matrix is K_ij=(b_i*b_j+c_i*c_j)/(4A). A straight Robin edge of
length l contributes Bi*l/6 times [[2,1],[1,2]]. This consistent boundary mass
matrix differs from the cell-center resistance used by the finite-volume code.
Dirichlet values are imposed by eliminating source degrees of freedom.

The reported absorption is Q=Bi*sum_edges(l*(u_a+u_b)/2). The summed reactions at
source nodes provide a separate source influx. Summing the free-node equations
and using the partition of unity gives conservation. Testing with the discrete
solution also yields u^T*K*u=Q because source values are one. This energy identity
is an additional verification, not a proof of continuum accuracy.

The slab has the exact affine solution u=1-Bi*x/(1+Bi) for unit length; linear
elements represent it exactly. Tests cover zero, low, intermediate and high Bi,
concentration, flux, energy and a rigidly rotated nonrectangular polygon. Mesh
refinement on non-affine solutions is still needed to measure numerical error.

Triangle preserves explicitly supplied polygon edges and subdivides them during
quality meshing. Its API documents constrained segments and maximum triangle area:
[Triangle API](https://rufat.be/triangle/API.html). The optional environment pin is
`triangle==20250106`; install with `python -m pip install -r requirements-validation.txt`.

Two comparisons must remain distinct. Meshing a raster mask checks the identical
staircase boundary with independent FEM assembly. Meshing a declared physical
polygon and separately rasterizing it measures both boundary representation and
PDE error. No automatic smoothing, Euclidean-perimeter correction or change of
Bi is hidden in either calculation. Agreement on a staircase alone does not
validate an anatomical or smooth-domain interpretation.

The raw five-segment square-Koch generator has point contacts at its second
iteration. The physical level-2 polygon therefore explicitly replaces each
internal curve vertex p_i by 0.8*p_i+0.1*(p_(i-1)+p_(i+1)) before matching area.
This is a finite regularized control, not an exact unregularized fractal. The
raster-bank construction separately uses its declared signed-distance smoothing.
The FEM entry point rejects coincident or unused vertices instead of concealing
singular boundary construction with a finite-looking numerical result.
