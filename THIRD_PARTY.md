# Dependencies and research attribution

Project code is MIT; original results and figures are CC BY 4.0. These grants
do not relicense dependencies or cited research. Dependencies are installed
separately and no virtual environment, package wheel or third-party paper PDF
is bundled in the release.

NumPy, SciPy, Matplotlib, scikit-image, PyYAML, Pydantic, pandas, Typer and the
development tools retain their upstream licenses. The recorded environment is
an inventory of what was installed during production, not a distribution of
those packages. See pyproject.toml and the archived environment records.

The optional independent FEM validation uses the Python triangle package
(version 20250106), which wraps Jonathan Richard Shewchuk's Triangle. **Triangle
has separate commercial-use/redistribution restrictions; the project MIT license
does not grant rights to Triangle.** Check the
[author's terms](https://www.cs.cmu.edu/~quake/triangle.html) and the installed
package license before use or redistribution. Triangle is not required by the
finite-volume solver or the shape optimizer. Meshes/results and our independently
assembled FEM implementation are retained; Triangle's source/binary is not.

Triangle is acknowledged for quality triangulation. The author's
[research-credit page](https://www.cs.cmu.edu/~quake/triangle.research.html)
lists the original numerical work. The scientific prior-work comparison and
primary references are in docs/related_work_assessment.md and the manuscript.
