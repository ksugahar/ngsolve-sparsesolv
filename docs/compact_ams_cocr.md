# Compact AMS + COCR Solver

Header-only C++ preconditioner + Krylov solver for HCurl problems.
No external dependency. All parallelism via NGSolve TaskManager.

Supports both **real** (magnetostatics) and **complex** (eddy current) systems.
`Update()` method enables Newton iteration for nonlinear problems.

## Problem Setting

Eddy current finite element discretization yields:

```
A x = b,   A = K + jw * sigma * M
```

where K is the curl-curl stiffness matrix, M is the mass matrix (conductor region),
and sigma is the electrical conductivity. The system matrix A is **complex symmetric**
(A^T = A, NOT Hermitian), which enables COCR's short-recurrence Krylov method.

## Architecture

```
                        COCR Solver
                  (complex symmetric Krylov)
                           |
                  ComplexCompactAMS
              (fused Re/Im preconditioner)
                           |
           +---------+---------+---------+
           |         |         |         |
      FineSmooth  GradCorr  NodalCorr  CompactAMG
      (l1-Jacobi)  (G^T AG)  (Pi^T A Pi)  (V-cycle)
```

Three layers work together:

1. **COCR** (outer Krylov) - exploits A^T = A for O(n) memory, no restart
2. **ComplexCompactAMS** (preconditioner) - fused Re/Im at all levels
3. **CompactAMG** (coarse solver) - classical AMG with DualMult fusion

## Quick Start

### Real (magnetostatics)

```python
import sparsesolv_ngsolve as ssn
from ngsolve import *
from ngsolve.krylovspace import CGSolver

# Real HCurl system (static magnetic field)
fes = HCurl(mesh, order=1, nograds=True, dirichlet="outer")
u, v = fes.TnT()

a = BilinearForm(fes)
a += curl(u) * curl(v) * dx + 1e-6 * u * v * dx
a.Assemble()

f = LinearForm(fes)
# ... assemble right-hand side ...
f.Assemble()

gfu = GridFunction(fes)

# Gradient and coordinates
G_mat, h1_fes = fes.CreateGradient()
coord_x = [mesh.ngmesh.Points()[i+1][0] for i in range(mesh.nv)]
coord_y = [mesh.ngmesh.Points()[i+1][1] for i in range(mesh.nv)]
coord_z = [mesh.ngmesh.Points()[i+1][2] for i in range(mesh.nv)]

pre = ssn.CompactAMSPreconditioner(
    mat=a.mat, grad_mat=G_mat,
    freedofs=fes.FreeDofs(),
    coord_x=coord_x, coord_y=coord_y, coord_z=coord_z)

with TaskManager():
    inv = CGSolver(a.mat, pre, tol=1e-10, maxiter=500)
    gfu.vec.data = inv * f.vec
```

### Complex (eddy current)

```python
import sparsesolv_ngsolve as ssn
from ngsolve import *

omega = 2 * pi * 30000          # 30 kHz
sigma_cu = 5.96e7               # copper conductivity [S/m]
nu = 1.0                        # reluctivity 1/mu_0 (air/copper region)

# Complex HCurl space
fes = HCurl(mesh, order=1, nograds=True, dirichlet="outer", complex=True)
u, v = fes.TnT()

# Complex system matrix: K + jw*sigma*M
a = BilinearForm(fes)
a += nu * curl(u) * curl(v) * dx
a += 1j * omega * sigma_cu * u * v * dx("conductor")
a += 1e-6 * u * v * dx                       # regularization
a.Assemble()

f = LinearForm(fes)
# ... assemble right-hand side (e.g., source current) ...
f.Assemble()

gfu = GridFunction(fes)

# Real auxiliary space (non-complex)
fes_real = HCurl(mesh, order=1, nograds=True, dirichlet="outer")
u_r, v_r = fes_real.TnT()

# Real SPD auxiliary matrix: nu*curl-curl + |omega|*sigma*M + eps*M
a_real = BilinearForm(fes_real)
a_real += nu * curl(u_r) * curl(v_r) * dx
a_real += omega * sigma_cu * u_r * v_r * dx("conductor")
a_real += 1e-6 * u_r * v_r * dx
a_real.Assemble()

# Gradient and coordinates
G_mat, h1_fes = fes_real.CreateGradient()
coord_x = [mesh.ngmesh.Points()[i+1][0] for i in range(mesh.nv)]
coord_y = [mesh.ngmesh.Points()[i+1][1] for i in range(mesh.nv)]
coord_z = [mesh.ngmesh.Points()[i+1][2] for i in range(mesh.nv)]

# Preconditioner (uses real auxiliary matrix, applied to complex system)
pre = ssn.ComplexCompactAMSPreconditioner(
    a_real_mat=a_real.mat,
    grad_mat=G_mat,
    freedofs=fes_real.FreeDofs(),
    coord_x=coord_x, coord_y=coord_y, coord_z=coord_z,
    ndof_complex=fes.ndof,
    cycle_type=1,        # 1=01210 (default), 7=0201020
    print_level=0)

# Solve with COCR (complex symmetric Krylov)
with TaskManager():
    inv = ssn.COCRSolver(
        a.mat, pre,
        freedofs=fes.FreeDofs(),
        maxiter=500, tol=1e-10)
    gfu.vec.data = inv * f.vec

print(f"Converged in {inv.iterations} iterations")
```

### Why a_real?

The AMS preconditioner operates on a **real SPD (symmetric positive definite)** matrix,
even when the actual system is complex. This is because:

1. **AMG coarsening requires real matrices.** PMIS coarsening and classical interpolation
   use strength-of-connection based on real matrix entries. Complex entries have no
   natural ordering for strength thresholds.

2. **The real auxiliary matrix captures the spectral behavior.** For the complex system
   `A = K + jw*sigma*M`, the real auxiliary `A_real = K + |omega|*sigma*M + eps*M`
   has the same sparsity pattern and similar spectral properties. The `|omega|*sigma`
   term (without the `j`) ensures the mass matrix contribution from the conductor
   region is represented in the preconditioner.

3. **Fused Re/Im application.** The preconditioner applies the same real AMS V-cycle
   to both the real and imaginary parts of the residual simultaneously (DualMult),
   halving memory bandwidth cost.

The complex system matrix `a.mat` is only used by the outer COCR solver for matrix-vector
products. The preconditioner never sees the complex matrix directly.

### Common Mistakes

**Wrong freedofs for complex preconditioner:**
```python
# WRONG: passing complex freedofs to preconditioner
pre = ssn.ComplexCompactAMSPreconditioner(
    a_real_mat=a_real.mat, ...,
    freedofs=fes.FreeDofs())        # fes is complex -> wrong size

# CORRECT: use real freedofs
pre = ssn.ComplexCompactAMSPreconditioner(
    a_real_mat=a_real.mat, ...,
    freedofs=fes_real.FreeDofs())    # fes_real is non-complex
```

**Missing TaskManager:**
```python
# WRONG: no TaskManager -> single-threaded, 5-10x slower
inv = ssn.COCRSolver(a.mat, pre, ...)
gfu.vec.data = inv * f.vec

# CORRECT: wrap solve in TaskManager
with TaskManager():
    inv = ssn.COCRSolver(a.mat, pre, ...)
    gfu.vec.data = inv * f.vec
```

**Missing nograds=True:**
```python
# WRONG: without nograds, HCurl includes gradient DOFs that
# make the system much larger and harder to precondition
fes = HCurl(mesh, order=1, dirichlet="outer")

# CORRECT: nograds=True removes gradient DOFs (kernel of curl)
fes = HCurl(mesh, order=1, nograds=True, dirichlet="outer")
```

**Wrong matrix type for ComplexCompactAMSPreconditioner:**
```python
# WRONG: passing complex matrix
pre = ssn.ComplexCompactAMSPreconditioner(
    a_real_mat=a.mat, ...)          # a is complex BilinearForm

# CORRECT: pass real auxiliary matrix
pre = ssn.ComplexCompactAMSPreconditioner(
    a_real_mat=a_real.mat, ...)     # a_real is real BilinearForm
```

**Order > 1 not supported:**
```python
# WRONG: order=2 not supported by CompactAMS
fes = HCurl(mesh, order=2, nograds=True, dirichlet="outer")

# CORRECT: use order=1
fes = HCurl(mesh, order=1, nograds=True, dirichlet="outer")
```

## Parameters

**ComplexCompactAMSPreconditioner**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `a_real_mat` | (required) | Real SPD auxiliary matrix (K + eps*M + \|omega\|*sigma*M) |
| `grad_mat` | (required) | Discrete gradient G (HCurl -> H1) |
| `freedofs` | None | Free DOFs mask for HCurl space |
| `coord_x/y/z` | (required) | Vertex coordinates (length = ndof_h1) |
| `ndof_complex` | 0 | Number of complex DOFs (0 = auto-derive from matrix) |
| `cycle_type` | 1 | AMS cycle: 1 = "01210", 7 = "0201020" |
| `print_level` | 0 | Verbosity (0 = silent) |
| `correction_weight` | 1.0 | Subspace correction weight |
| `subspace_solver` | 0 | 0 = CompactAMG, 1 = SparseCholesky (diagnostic) |
| `num_smooth` | 1 | l1-Jacobi sweeps per smooth step |

**COCRSolver**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mat` | (required) | System matrix (real or complex BaseMatrix) |
| `pre` | (required) | Preconditioner (BaseMatrix) |
| `freedofs` | None | Free DOFs mask |
| `maxiter` | 500 | Maximum iterations |
| `tol` | 1e-8 | Relative convergence tolerance |
| `printrates` | False | Print convergence info |

## Source Files

| File | Lines | Description |
|------|-------|-------------|
| `compact_amg.hpp` | ~900 | Classical AMG: PMIS coarsening, l1-Jacobi, V-cycle, DualMult |
| `compact_ams.hpp` | ~750 | AMS preconditioner: Hiptmair-Xu auxiliary space method |
| `complex_compact_ams.hpp` | ~340 | Fused Re/Im ComplexCompactAMS for complex systems |
| `sparsesolv_solvers.hpp` | ~300 | COCRSolverNGS with fused vector operations |

All files are in `include/sparsesolv/preconditioners/` (AMG/AMS) and
`include/sparsesolv/ngsolve/` (COCR solver, Python exports).

## Algorithm Details

### CompactAMG (compact_amg.hpp)

Classical algebraic multigrid with the following components:

- **Coarsening**: PMIS (Parallel Modified Independent Set) based on De Sterck et al. (2006)
- **Interpolation**: Classical direct interpolation (Ruge-Stueben 1987)
- **Smoother**: l1-Jacobi (truncated l1 norm: `diag + 0.5*|off_diag|`, capped at `4/3*diag`)
- **Coarsest level**: NGSolve SparseCholesky direct solver
- **Strength threshold**: 0.25 (configurable via `amg_theta`)

**DualMult**: Processes two right-hand sides simultaneously at every AMG level.
SpMV is memory-bandwidth-bound; loading matrix rows once for two vectors
halves the bandwidth cost:

```cpp
// DualResidual: res1 = b1 - A*x1, res2 = b2 - A*x2  (one matrix pass)
ParallelFor(n, [&](size_t i) {
    auto cols = A.GetRowIndices(i);
    auto vals = A.GetRowValues(i);
    double d1 = 0, d2 = 0;
    for (int j = 0; j < cols.Size(); j++) {
        int c = cols[j]; double v = vals[j];
        d1 += v * x1[c]; d2 += v * x2[c];
    }
    res1[i] = b1[i] - d1;
    res2[i] = b2[i] - d2;
});
```

DualMult includes fused versions of: V-cycle, l1-Jacobi smooth, residual, SpMV, MultAdd.
Each AMG Level stores dual work vectors (`residual2`, `tmp2`, `correction2`).

### CompactAMS (compact_ams.hpp)

Implements the Hiptmair-Xu (2007) auxiliary space preconditioner for HCurl systems.

**AMS V-cycle** (cycle_type=1, pattern "01210"):

```
1. FineSmooth     : l1-Jacobi on A_bc (fine HCurl grid)
2. GradCorrect    : G^T * A_bc * G  solved by CompactAMG (H1 gradient space)
3. NodalCorrect   : Pi_d^T * A_bc * Pi_d  solved by CompactAMG x3 (H1 nodal, d=x,y,z)
4. GradCorrect    : repeat
5. FineSmooth     : repeat
```

**Pi matrix construction** (Kolev-Vassilevski formula):

```
Gd[e] = (G * coord_d)[e]           // edge vector in d-direction
Pi_d[e,v] = |G[e,v]| * 0.5 * Gd[e] // nodal interpolation operator
```

**Nodal correction** is additive: Pix, Piy, Piz corrections computed from the same
residual and added together. This saves 2 fine-level SpMVs per cycle compared to
multiplicative approach.

**Setup** builds 4 CompactAMG instances (G, Pix, Piy, Piz) in parallel via
`ParallelFor(4, ...)`.

### ComplexCompactAMS (complex_compact_ams.hpp)

Wraps CompactAMS for complex eddy current systems. The key optimization is
**fused Re/Im processing**: fine-level SpMV operations load matrix data once
and process both real and imaginary parts simultaneously.

**Fused operations**:

| Operation | Matrix | Conventional | Fused |
|-----------|--------|-------------|-------|
| FineSmooth | A_bc (HCurl) | 2 passes | 1 pass |
| Residual | A_bc (HCurl) | 2 passes | 1 pass |
| Restrict | G^T, Pi_d^T | 2 passes each | 1 pass each |
| Prolongate | G, Pi_d | 2 passes each | 1 pass each |
| AMG V-cycle | coarse levels | 2 calls | DualMult (fused) |

**FusedFineSmooth** must be two-phase to avoid race conditions:

```
Phase 1: res[i] = b[i] - (A*x)[i]   for ALL i   (reads OLD x)
Phase 2: x[i] += res[i] / l1[i]     for ALL i   (no data dependency)
```

A single-phase approach (compute residual and update in one pass) creates
chaotic Jacobi: ParallelFor may read x[j] before or after another thread
updates it, leading to divergence.

**AMG DualMult** is used for all coarse-level solves (gradient + 3 nodal).
If the subspace solver is not CompactAMG (e.g., SparseCholesky for debugging),
falls back to sequential Re/Im.

### COCR Solver (sparsesolv_solvers.hpp)

Conjugate Orthogonal Conjugate Residual method for complex symmetric systems.

COCR exploits the complex symmetry (A^T = A) with short recurrences (3-term),
requiring only 6 work vectors regardless of iteration count.

**Fused vector operations** reduce memory traffic:

```cpp
// 3 updates -> 1 ParallelFor
ParallelFor(n, [=](size_t i) {
    sol_d[i] += alpha * p_d[i];    // solution update
    r_d[i]   -= alpha * q_d[i];    // residual update
    rt_d[i]  -= alpha * qt_d[i];   // preconditioned residual update
});

// 2 updates -> 1 ParallelFor
ParallelFor(n, [=](size_t i) {
    p_d[i] = rt_d[i] + beta * p_d[i];   // direction update
    q_d[i] = t_d[i]  + beta * q_d[i];   // A*direction update
});
```

Raw pointer access via `GetVectorData<SCAL>()` bypasses NGSolve's BaseVector
virtual dispatch overhead.

## Performance Optimization History

Measured on Hiruma mesh1_3.5T (197,395 DOFs, 168 iterations):

| Optimization | ms/iter | Cumulative Speedup |
|-------------|---------|-------------------|
| Baseline (sequential Re/Im AMS, NGSolve vector ops) | 49.2 | 1.00x |
| + Fused Re/Im fine-level SpMV | 41.1 | 1.20x |
| + Fused COCR vector updates (ParallelFor) | 36.6 | 1.34x |
| + Fused AMG DualMult (all coarse levels) | 32.7 | **1.50x** |

## Update() for Newton Iteration

Both `CompactAMSPreconditioner` (real) and `ComplexCompactAMSPreconditioner` (complex)
support `Update()` for nonlinear solvers where the system matrix changes at each step.

### What Update() preserves and rebuilds

| Category | Components | Rebuilt? |
|----------|-----------|----------|
| **Geometry** (one-time) | Pi matrices, G transposes, work vectors | NO |
| **Matrix-dependent** (per-Update) | A_bc, Galerkin projections (A_G, A_Pi), AMG hierarchies, l1 norms | YES |

This split avoids redundant recomputation of geometric data that depends only on
the mesh and coordinates, not on the matrix values.

### API

```python
# Update with current matrix (matrix was modified in-place)
pre.Update()

# Update with a new matrix object
pre.Update(new_mat)
```

### Real (magnetostatics + CG)

```python
import sparsesolv_ngsolve as ssn
from ngsolve import *
from ngsolve.krylovspace import CGSolver

fes = HCurl(mesh, order=1, nograds=True, dirichlet="outer")
G_mat, h1_fes = fes.CreateGradient()
coord_x = [mesh.ngmesh.Points()[i+1][0] for i in range(mesh.nv)]
coord_y = [mesh.ngmesh.Points()[i+1][1] for i in range(mesh.nv)]
coord_z = [mesh.ngmesh.Points()[i+1][2] for i in range(mesh.nv)]

a = BilinearForm(fes)
a += nu_cf * curl(u) * curl(v) * dx + 1e-6 * u * v * dx
a.Assemble()

pre = ssn.CompactAMSPreconditioner(a.mat, G_mat,
    freedofs=fes.FreeDofs(),
    coord_x=coord_x, coord_y=coord_y, coord_z=coord_z)

# Newton loop
for k in range(max_newton):
    a.Assemble()          # Reassemble with updated nu(B)
    pre.Update(a.mat)     # Rebuild preconditioner (geometry preserved)
    inv = CGSolver(a.mat, pre, tol=1e-8, maxiter=500)
    delta = inv * rhs
    gfu.vec.data += delta
```

### Complex (eddy current + COCR)

```python
pre = ssn.ComplexCompactAMSPreconditioner(a_real.mat, G_mat,
    freedofs=fes_real.FreeDofs(),
    coord_x=coord_x, coord_y=coord_y, coord_z=coord_z,
    ndof_complex=fes.ndof)

# Newton loop
for k in range(max_newton):
    a_real.Assemble()
    pre.Update(a_real.mat)
    inv = ssn.COCRSolver(a_complex.mat, pre, tol=1e-10)
    delta = inv * rhs
    gfu.vec.data += delta
```

## Key Design Decisions

### Why COCR?

The eddy current system A = K + jw*sigma*M satisfies A^T = A (complex symmetric).
COCR exploits this symmetry with short recurrences (3-term), requiring only 6 work
vectors regardless of iteration count.

### Why fused Re/Im?

SpMV is memory-bandwidth-limited on modern hardware. The matrix A_bc is the same
for both Re and Im parts. Loading each row from memory once and computing both
dot products halves the bandwidth cost. This applies to:
- Fine-level smoother (l1-Jacobi)
- Residual computation (b - A*x)
- Restriction (P^T * residual)
- Prolongation (x += P * correction)
- AMG V-cycle at all coarse levels (via DualMult)

### Why l1-Jacobi instead of Gauss-Seidel?

l1-Jacobi is fully parallel (no data dependency between rows), making it ideal
for TaskManager parallelization. Gauss-Seidel requires sequential or multi-color
ordering, adding complexity. The l1 norm truncation (`diag + 0.5*|off_diag|`,
capped at `4/3*diag`) provides robust smoothing without over-damping.

### Why additive nodal correction?

The Pix, Piy, Piz corrections all use the same fine-level residual, computed once.
This saves 2 fine-level SpMV evaluations per AMS cycle compared to a multiplicative
approach (which would recompute the residual between each Pi correction).

## Benchmark Results

**Test problem**: Hiruma eddy current model (SA-26-001).
Copper conductor (sigma = 5.96e7 S/m) with ferromagnetic core (mu_r = 1000).
Frequency 30 kHz, HCurl order 1, convergence tolerance 1e-10.

**Environment**: Windows Server 2022, Intel Xeon (8 cores), MSVC 2022, MKL 2024.2.

### Scaling with Mesh Size

| Mesh | Elements | HCurl DOFs | H1 DOFs | Iters | Setup [s] | Solve [s] | Total [s] | ms/iter | Memory [MB] |
|------|----------|-----------|---------|-------|-----------|-----------|-----------|---------|-------------|
| mesh1_2.5T | 130,460 | 155,527 | 23,731 | 144 | 0.7 | 3.7 | 4.5 | 25.8 | 368 |
| mesh1_3.5T | 166,198 | 197,395 | 29,901 | 168 | 0.9 | 6.3 | 7.3 | 37.7 | 460 |
| mesh1_4.5T | 203,403 | 241,129 | 36,473 | 210 | 1.3 | 9.2 | 10.5 | 43.8 | 543 |
| mesh1_5.5T | 280,278 | 331,595 | 49,643 | 249 | 1.9 | 14.3 | 16.2 | 57.3 | 725 |
| mesh1_20.5T | 1,227,241 | 1,441,102 | 211,337 | 499 | 24.9 | 197.7 | 222.6 | 396.2 | 2,933 |

All cases converged (`true ||b-Ax||/||b|| < 2e-10`).

### Comparison with IC Preconditioner (ABMC-ICCG)

IC (Incomplete Cholesky) preconditioner with ABMC (Algebraic Block Multi-Color)
parallel ordering. 30 kHz, tol=1e-10, maxiter=20,000.

| Mesh | DOFs | Method | Iters | Total [s] | Status |
|------|------|--------|-------|-----------|--------|
| mesh1_3.5T | 197k | Compact AMS + COCR | 168 | 7.2 | converged |
| mesh1_3.5T | 197k | ABMC-ICCG | 17,178 | 438.4 | not converged (res=2.8e-10) |

IC preconditioner cannot handle the curl-curl null space inherent in HCurl discretizations.
AMS resolves this with gradient and nodal auxiliary space corrections (Hiptmair-Xu 2007).

### Comparison with EMD Preconditioner (Hiruma SA-26-001)

EMD (Edge-based Magnetic field Decomposition) paper results
(Hiruma, SA-26-001, 3,670,328 DOFs, 30 kHz):

| Method | Iters | Time [s] | ms/iter |
|--------|-------|----------|---------|
| IC only | 15,838 | 5964.8 | 376.7 |
| EMD (IC + AMG V-cycle) | 4,069 | 1716.8 | 422.0 |
| EMD (IC + AMG W-cycle) | 2,935 | 1552.9 | 529.1 |
| EMD (IC + GenEO-DDM, 24 domains) | 1,004 | 550.8 | 548.6 |

Our result at 1.44M DOFs (mesh1_20.5T): 499 iter, 222.6s on 8 CPU cores.
Direct DOF-matched comparison is not possible (1.44M vs 3.67M DOFs).

## References

- R. Hiptmair, J. Xu. "Nodal Auxiliary Space Preconditioning in H(curl) and H(div) Spaces."
  SIAM J. Numer. Anal. 45(6), 2007.
- T. Kolev, P. Vassilevski. "Parallel Auxiliary Space AMG Solver for H(div) Problems."
  J. Comput. Math. 27(5), 2009.
- J. Ruge, K. Stueben. "Algebraic Multigrid." In Multigrid Methods, 1987.
- H. De Sterck, U. Yang, J. Heys. "Reducing Complexity in Parallel Algebraic Multigrid
  Preconditioners." SIAM J. Matrix Anal. Appl. 27(4), 2006.
- T. Sogabe, S.-L. Zhang. "A COCR method for solving complex symmetric linear systems."
  J. Comput. Appl. Math. 199(2), 2007.

## Benchmark Reproduction

```bash
cd examples/hiruma

# Compact AMS + COCR (single mesh)
python bench_compact_ams.py mesh1_3.5T

# Compact AMS + COCR (all standard meshes)
python bench_compact_ams.py --all

# Comparison with ABMC-ICCG
python bench_ams_vs_abmc.py mesh1_3.5T

# Results saved to results_compact_ams.json, results_ams_vs_abmc.json
```
