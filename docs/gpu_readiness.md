# RTX 4060 availability and workload profiling

Local `nvidia-smi` confirms an NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB
VRAM (8 GB class), driver 616.87. This project's virtual environment currently
has no PyTorch, CuPy or JAX installation. The GPU was not used by these experiments.

The current learned selector has 24 linear features. A 256-candidate scoring call
takes about 5.68 microseconds, averaged over 2000 calls after a warm-up. This is
not the expensive part of the current pipeline. A larger neural proposer/policy
could use CUDA in a separately validated training stage; adding a GPU framework
does not accelerate the existing NumPy/SciPy algorithms automatically.

The completed 180-run study records 1089.15 seconds of full candidate geometry
validation: 78.3% of environment time and 61.0% of completed workflow time.
The latter includes endpoint refinement and artifact storage. Validation at
100/200/400 is therefore a concrete cost to address while preserving its checks.

A separate profile uses one actual improved deep-cavity input, seed 190000,
Bi=1 and the same solver tolerances. Generating 256 raw edits takes 0.0678 s;
the 68 necessary full geometry validations take 1.1733 s; 15 pass all checks.
This is a sample state, not a general feasibility estimate.

| Grid | Active cells | Total profiled solve (ms) | Assembly (ms) | Sparse linear solve (ms) |
|---|---:|---:|---:|---:|
| 100 | 2400 | 36.65 | 33.09 | 2.97 |
| 200 | 9600 | 127.03 | 111.69 | 14.13 |
| 400 | 38400 | 527.48 | 444.90 | 78.19 |

Assembly includes Python cell/face handling and geometry validation. The measured
n=400 sparse solve accounts for about 78 ms of a 527 ms profiled solve; accelerating
that phase alone leaves most of the measured work. These are one-pass cProfile
measurements with instrumentation overhead, not GPU benchmarks. Nested cumulative
function times in the raw profile overlap and must not be added indiscriminately.

Keep the validated finite-volume reference and present linear selector on CPU.
Useful next performance work is profiling and improving geometry/assembly work
with numerical equivalence checks. GPU use becomes a separate measured question
when a sufficiently large neural model or validated batched solver is introduced.
No GPU dependency was installed merely to move this small selector to CUDA.

The profiling experiment adds exactly three PDE solves, separately from the
3780-solve comparison matrix and its documented interruption overhead. Maximum
profile conservation error is 3.63e-13. Config, software versions, exact source
snapshot, all three fields and `.prof` files are retained in
[the profiling run](../experiments/20261019T130428281999Z_workload_profile_bcdf4692/summary.json).
Rerun the measurement with:

```powershell
.\.venv\Scripts\python.exe scripts/profile_oxygen_workload.py experiments/20261019T122613613156Z_feasible_pool_b8b49d2b
```
