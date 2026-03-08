# Hiruma Eddy Current Analysis Results

## Problem

Frequency-domain eddy current analysis (A-Phi formulation, f = 50 Hz) of a
copper coil around a ferromagnetic iron core.

| Region | mu_r | sigma [S/m] | Description |
|--------|------|-------------|-------------|
| core | 1000 | 0 | Ferromagnetic iron core |
| cond | 1 | 5.96e7 | Copper conductor (coil) |
| air | 1 | 0 | Surrounding air |

## Formulation

**Two-step A formulation** (HCurl only):

1. DC current distribution: solve Laplace phi in conductor (H1, sparsecholesky)
2. HCurl solve: `curl(nu * curl A) + eps*nu*A + jw*sigma*A = -jw*sigma*grad(phi)` in cond

- `HCurl(order=1, nograds=True, complex=True)` for vector potential A
- Stabilization `1e-6 * nu * (u,v) * dx` for regularity
- CG solver (`conjugate=False` for complex-symmetric)

## Solver Comparison

### HYPRE AMS (`bench_hypre_ams.py`)

HYPRE AMS preconditioner with Re/Im splitting:
- Real SPD auxiliary: `A_real = K + eps*M + |omega|*sigma*M_cond`
- ComplexHypreAMSPreconditioner: TaskManager-parallel Re/Im
- GMRES outer solver (HYPRE AMS is non-symmetric)

#### cycle_type=0 (G-only V-cycle) + direct H1 solver

| Mesh | HCurl DOFs | H1 DOFs | ns=2 iters | ns=2 time | ns=3 iters | ns=3 time | Memory |
|------|----------:|--------:|----------:|---------:|----------:|---------:|-------:|
| 2.5T | 155,527 | 23,731 | 54 | 8.6s | 46 | 9.0s | 307 MB |
| 3.5T | 197,395 | 29,901 | 59 | 13.7s | 49 | 15.4s | 414 MB |
| 4.5T | 241,129 | 36,473 | 56 | 8.8s | 51 | 10.1s | 524 MB |
| 5.5T | 331,595 | 49,643 | 68 | 16.6s | 58 | 17.7s | 684 MB |

#### cycle_type=7 (full AMS "0201020") + direct H1 solver

| Mesh | HCurl DOFs | ns=2 iters | ns=2 time | ns=3 iters | ns=3 time |
|------|----------:|----------:|---------:|----------:|---------:|
| 2.5T | 155,527 | 41 | 25.9s | 36 | 18.0s |
| 3.5T | 197,395 | 42 | 34.7s | 35 | 28.7s |
| 4.5T | 241,129 | 43 | 24.5s | 36 | 23.6s |
| 5.5T | 331,595 | 51 | 49.0s | 44 | 47.4s |

#### cycle_type=0 (G-only V-cycle) + h1amg H1 solver

| Mesh | HCurl DOFs | ns=2 iters | ns=2 time | ns=3 iters | ns=3 time |
|------|----------:|----------:|---------:|----------:|---------:|
| 2.5T | 155,527 | 138 | 27.4s | 120 | 27.9s |
| 3.5T | 197,395 | 142 | 26.1s | 121 | 20.0s |
| 4.5T | 241,129 | 154 | 22.8s | 133 | 25.3s |
| 5.5T | 331,595 | 180 | 37.3s | 161 | 42.7s |

#### cycle_type=7 (full AMS "0201020") + h1amg H1 solver

| Mesh | HCurl DOFs | ns=2 iters | ns=2 time | ns=3 iters | ns=3 time |
|------|----------:|----------:|---------:|----------:|---------:|
| 2.5T | 155,527 | 107 | 72.9s | 94 | 65.9s |
| 3.5T | 197,395 | 117 | 68.3s | 102 | 46.6s |
| 4.5T | 241,129 | 127 | 63.3s | 103 | 59.9s |
| 5.5T | 331,595 | 143 | 100.5s | 134 | 109.5s |

**Key observations**:
- cycle_type=0 + direct: fewest iterations and fastest wall time for meshes up to 5.5T
- cycle_type=7 reduces iterations ~25% vs cycle_type=0, but per-iteration cost is ~3x higher
- h1amg: ~2.5-3x more iterations than direct, but scalable to larger problems
- Memory scales linearly with DOFs (~2 MB/kDOF)

### Unified Solver Comparison (`bench_comparison.py`)

All methods benchmarked on the same meshes with identical tolerance (1e-8), CG outer solver.

#### Solve time comparison (seconds)

| Mesh | DOFs | BDDC | pointGS+d | Jacobi+d | **Cheby(3)+d** | Cheby(4)+d | pGS+amg | Ch3+amg | Ch4+amg |
|------|-----:|-----:|----------:|---------:|---------------:|-----------:|--------:|--------:|--------:|
| 2.5T | 155k | 0.31 | 4.26 | 3.35 | **2.39** | 2.43 | 11.23 | 7.40 | 6.71 |
| 3.5T | 197k | 0.40 | 6.18 | 4.39 | **3.45** | 3.27 | 15.20 | 8.92 | 9.37 |
| 4.5T | 241k | 0.53 | 7.24 | FAIL | **5.09** | 4.70 | 21.53 | 15.37 | 11.01 |
| 5.5T | 331k | 0.98 | 12.65 | 16.98 | **6.97** | 7.85 | 30.46 | 18.45 | 17.40 |

#### Total time (setup + solve)

| Mesh | DOFs | BDDC total | pointGS+d | **Cheby(3)+d** | Speedup vs BDDC | Speedup vs pGS |
|------|-----:|-----------:|----------:|---------------:|----------------:|---------------:|
| 2.5T | 155k | 12.94 | 4.26 | **2.46** | **5.26x** | **1.73x** |
| 3.5T | 197k | 17.33 | 6.18 | **3.54** | **4.89x** | **1.75x** |
| 4.5T | 241k | 24.61 | 7.24 | **5.21** | **4.72x** | **1.39x** |
| 5.5T | 331k | 49.12 | 12.65 | **7.13** | **6.89x** | **1.77x** |

#### Iteration counts

| Mesh | DOFs | BDDC | pGS+d | Jac+d | Ch3+d | Ch4+d | pGS+amg | Ch3+amg | Ch4+amg |
|------|-----:|-----:|------:|------:|------:|------:|--------:|--------:|--------:|
| 2.5T | 155k | 3 | 54 | 90 | 65 | 57 | 138 | 153 | 130 |
| 3.5T | 197k | 3 | 59 | 91 | 67 | 57 | 142 | 157 | 145 |
| 4.5T | 241k | 3 | 56 | 500* | 73 | 60 | 154 | 190 | 156 |
| 5.5T | 331k | 3 | 68 | 184 | 78 | 71 | 180 | 213 | 182 |

\* Jacobi(0.6) ns=3 did not converge at 500 iterations on mesh1_4.5T.

#### Per-iteration cost (ms/iter)

| Mesh | DOFs | pGS | Jacobi | Cheby(3) | Cheby(4) |
|------|-----:|----:|-------:|---------:|---------:|
| 2.5T | 155k | 78.9 | 37.2 | **36.7** | 42.6 |
| 3.5T | 197k | 104.7 | 48.2 | **51.5** | 57.4 |
| 4.5T | 241k | 129.3 | 72.1 | **69.7** | 78.3 |
| 5.5T | 331k | 186.0 | 92.3 | **89.4** | 110.5 |

Chebyshev(3) per-iteration cost is comparable to Jacobi (~2x cheaper than pointGS).

#### Key findings

1. **Best method: Chebyshev(3) + direct H1** -- consistently fastest total time across all meshes
2. **vs BDDC**: 4.7-6.9x faster (BDDC setup dominates; also OOMs at >1.6M DOFs)
3. **vs pointGS**: 1.4-1.8x faster (Chebyshev polynomial avoids serial GS dependency)
4. **Jacobi(0.6)**: UNSTABLE on mesh1_4.5T (diverges at 500 iters). Not recommended.
5. **H1 direct vs h1amg**: Direct gives 2-2.5x fewer CG iterations (auto threshold: 100k H1 DOFs)
6. **Chebyshev(4) vs (3)**: Degree 4 slightly fewer iterations but higher per-iter cost; degree 3 wins on wall time

### HYPRE AMS + TaskManager (`bench_hypre_ams.py`)

ComplexHypreAMSPreconditioner: Two independent HYPRE AMS instances with
TaskManager-parallel Re/Im splitting. GMRES solver (HYPRE AMS is non-symmetric).

| Mesh | DOFs | Python (sequential) | C++ TaskManager | Speedup |
|------|-----:|---:|---:|---:|
| 2.5T | 155k | 5.40s, 50 it | **3.43s, 50 it** | **1.57x** |
| 5.5T | 331k | 14.98s, 59 it | **10.19s, 59 it** | **1.47x** |
| 20.5T | 1.44M | 103.09s, 75 it | **69.16s, 75 it** | **1.49x** |

- Iteration count identical (same math, only parallelism differs)
- Consistent **~1.5x speedup** across mesh sizes

### BDDC (`eddy_current.py`)

| Mesh | DOFs | BDDC iters | setup [s] | solve [s] |
|------|-----:|:---:|-----:|-----:|
| 2.5T | 179k | **3** | 10.1 | 1.3 |
| 3.5T | 227k | **3** | 15.1 | 1.9 |
| 4.5T | 278k | **3** | 21.4 | 2.4 |
| 5.5T | 381k | **3** | 47.1 | 4.2 |
| 20.5T | 1.65M | **OOM** | - | - |
| 21.5T_HF | 7.76M | **OOM** | - | - |

BDDC (MKL PARDISO coarse): mesh-independent iterations (3) but OOM at >1.6M DOFs.

## ||B||^2 by Region

| Mesh | ||B||^2 total | core (%) | cond (%) | air (%) |
|------|-------------|----------|----------|---------|
| 2.5T | 1.634e-01 | 5.50e-02 (33.7%) | 4.97e-03 (3.0%) | 1.03e-01 (63.3%) |
| 3.5T | 1.533e-01 | 5.63e-02 (36.7%) | 4.32e-03 (2.8%) | 9.27e-02 (60.4%) |
| 4.5T | 1.378e-01 | 5.23e-02 (37.9%) | 3.80e-03 (2.8%) | 8.17e-02 (59.3%) |
| 5.5T | 1.229e-01 | 4.73e-02 (38.5%) | 3.33e-03 (2.7%) | 7.23e-02 (58.8%) |

||B||^2 decreases ~25% from coarsest to finest mesh. The meshes differ in
geometry discretization (conductor volume nearly doubles from 2.5T to 5.5T),
not purely h-refinement.

## VTK Output

VTK files (`eddy_current_mesh1_*.vtu`) contain:

| Field | Description |
|-------|-------------|
| B_re, B_im | Magnetic flux density (real/imaginary) |
| B_abs | |B| magnitude |
| J_re, J_im | Current density in conductor |
| Q_joule | Joule heat density [W/m^3] |
| MaterialID | 1=core, 2=cond, 3=air |

Values are unnormalized (phi=1 on gamma_in). Scale by (I_target/I_computed)^2
for physical quantities.

## Comparison to Hiruma EMD (IC+DDM)

Target: IC+DDM 24-partition, 3.67M DOFs, 1,004 iterations, 551 seconds.

Scaling estimate for AMS (Chebyshev(3) + h1amg, 8 cores):

| DOFs | Method | Iters | Time | ms/iter |
|-----:|--------|------:|-----:|--------:|
| 155k | Ch3+amg | 153 | 7.4s | 48.4 |
| 197k | Ch3+amg | 157 | 8.9s | 56.8 |
| 331k | Ch3+amg | 213 | 18.4s | 86.6 |
| 3.67M | Ch3+amg (est.) | ~250 | **~240s** | ~960 |

Per-iter cost scales ~O(N) (87ms at 331k -> ~960ms at 3.67M).
Estimated 250 iters x 960ms = **~240 seconds**, which would be **2.3x faster than EMD**.

With direct H1 solver (if ndof_h1 < 100k), Cheby(3)+direct at 331k DOFs
achieves 78 iters in 7.0s (**89ms/iter**), giving further 2.6x speedup over h1amg.

With Fused complex V-cycle (+AVX2), Cheby(3)+direct at 331k DOFs
achieves 78 iters in 6.0s (**77ms/iter**), an additional **1.16x** over standard.
Estimated for 3.67M DOFs: 250 iters x (77/89 * 960) = 250 * 830 = **~208s** (**2.65x faster than EMD**).

## Benchmark Environment

- Date: 2026-03-07
- Host: lab (Windows Server 2022)
- CPU: 8 cores
- JSON results: `results_hypre_ams.json`
