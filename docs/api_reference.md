# Python API Reference

## Import

```python
import ngsolve  # Must be imported first (loads shared libraries)
from sparsesolv_ngsolve import (
    # Preconditioners (IC/SGS)
    ICPreconditioner,      # Incomplete Cholesky preconditioner
    SGSPreconditioner,     # Symmetric Gauss-Seidel preconditioner

    # Compact AMG/AMS Preconditioners
    CompactAMGPreconditioner,            # Classical AMG (for H1)
    CompactAMSPreconditioner,            # Real HCurl (magnetostatics)
    CompactAMSPreconditionerImpl,        # Real AMS type (with Update())
    ComplexCompactAMSPreconditioner,     # Complex Re/Im fused AMS
    ComplexCompactAMSPreconditionerImpl, # Complex AMS type (with Update())
    has_compact_ams,                     # Check Compact AMG/AMS availability

    # Iterative Solvers
    SparseSolvSolver,      # Unified iterative solver (ICCG/SGSMRTR/CG/COCR)
    SparseSolvResult,      # Solve result
    COCRSolver,            # COCR (complex symmetric systems, native C++)
    GMRESSolver,           # GMRES (non-symmetric systems, left-preconditioned)
)
```

Factory functions automatically detect the matrix type (real/complex) via `mat.IsComplex()`.

---

## ICPreconditioner

Incomplete Cholesky (IC) preconditioner.

### Constructor

```python
pre = ICPreconditioner(mat, freedofs=None, shift=1.05)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mat` | `SparseMatrix` | - | SPD matrix (real/complex) |
| `freedofs` | `BitArray` or `None` | `None` | Free DOFs. `None` means all DOFs are free |
| `shift` | `float` | `1.05` | Shift parameter |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `shift` | `float` | Shift parameter (read/write) |
| `use_abmc` | `bool` | Enable ABMC ordering (read/write) |
| `abmc_block_size` | `int` | ABMC block size (read/write) |

### Methods

| Method | Description |
|--------|-------------|
| `Update()` | Recompute preconditioner (call after matrix changes) |

### Example

```python
pre = ICPreconditioner(a.mat, freedofs=fes.FreeDofs(), shift=1.05)
inv = CGSolver(a.mat, pre, tol=1e-10)
gfu.vec.data = inv * f.vec
```

---

## SGSPreconditioner

Symmetric Gauss-Seidel (SGS) preconditioner.

### Constructor

```python
pre = SGSPreconditioner(mat, freedofs=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mat` | `SparseMatrix` | - | SPD matrix |
| `freedofs` | `BitArray` or `None` | `None` | Free DOFs |

### Methods

| Method | Description |
|--------|-------------|
| `Update()` | Recompute preconditioner |

### Example

```python
pre = SGSPreconditioner(a.mat, freedofs=fes.FreeDofs())
inv = CGSolver(a.mat, pre, tol=1e-10)
gfu.vec.data = inv * f.vec
```

---

## SparseSolvSolver

Unified iterative solver. Supports ICCG, SGSMRTR, CG, and COCR methods.
Can be used as a `BaseMatrix`, allowing the form `gfu.vec.data = solver * f.vec`.

### Constructor

```python
solver = SparseSolvSolver(mat, method="ICCG", freedofs=None,
                           tol=1e-10, maxiter=1000, shift=1.05,
                           save_best_result=True,
                           save_residual_history=False,
                           printrates=False, conjugate=False,
                           use_abmc=False, abmc_block_size=4,
                           abmc_num_colors=4, abmc_reorder_spmv=False,
                           abmc_use_rcm=False)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mat` | `SparseMatrix` | - | SPD matrix |
| `method` | `str` | `"ICCG"` | `"ICCG"`, `"SGSMRTR"`, `"CG"`, `"COCR"` |
| `freedofs` | `BitArray` | `None` | Free DOFs |
| `tol` | `float` | `1e-10` | Convergence tolerance |
| `maxiter` | `int` | `1000` | Maximum number of iterations |
| `shift` | `float` | `1.05` | IC factorization shift (for ICCG) |
| `save_best_result` | `bool` | `True` | Track best solution |
| `save_residual_history` | `bool` | `False` | Record residual history |
| `printrates` | `bool` | `False` | Print convergence information |
| `conjugate` | `bool` | `False` | Conjugate inner product (for Hermitian systems) |

### ABMC-Related Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_abmc` | `bool` | `False` | Enable ABMC ordering |
| `abmc_block_size` | `int` | `4` | Number of rows per block |
| `abmc_num_colors` | `int` | `4` | Target number of colors |
| `abmc_reorder_spmv` | `bool` | `False` | Perform SpMV in ABMC space |
| `abmc_use_rcm` | `bool` | `False` | Apply RCM bandwidth reduction beforehand |

### Properties (Settable After Construction)

| Property | Type | Description |
|----------|------|-------------|
| `auto_shift` | `bool` | Automatic shift adjustment for IC factorization |
| `diagonal_scaling` | `bool` | Diagonal scaling |
| `divergence_check` | `bool` | Early termination on stagnation detection |
| `divergence_threshold` | `float` | Divergence detection threshold |
| `divergence_count` | `int` | Number of consecutive poor iterations before divergence is declared |
| `last_result` | `SparseSolvResult` | Most recent solve result |

### Methods

| Method | Arguments | Returns | Description |
|--------|-----------|---------|-------------|
| `Solve(rhs, sol)` | `BaseVector`, `BaseVector` | `SparseSolvResult` | Solve with initial guess |

### Example

```python
# Use as BaseMatrix (zero initial guess)
solver = SparseSolvSolver(a.mat, method="ICCG",
                           freedofs=fes.FreeDofs(), tol=1e-10)
gfu.vec.data = solver * f.vec

# Use the Solve method to obtain detailed results
result = solver.Solve(f.vec, gfu.vec)
print(f"Converged: {result.converged}, Iters: {result.iterations}")
```

---

## SparseSolvResult

A struct that stores the solve result.

| Field | Type | Description |
|-------|------|-------------|
| `converged` | `bool` | Whether the solver converged |
| `iterations` | `int` | Number of iterations |
| `final_residual` | `float` | Final relative residual |
| `residual_history` | `list[float]` | Residual at each iteration (when `save_residual_history=True`) |

```python
result = solver.Solve(f.vec, gfu.vec)
if result.converged:
    print(f"Converged in {result.iterations} iterations")
    print(f"Final residual: {result.final_residual:.2e}")
```

---

## CompactAMSPreconditioner

Compact AMS preconditioner for real HCurl systems (magnetostatic curl-curl + mass).

Header-only C++ implementation. No external libraries required.
Supports nonlinear solvers (Newton iteration): `Update()` rebuilds only the matrix-dependent parts while preserving geometric information.

### Constructor

```python
pre = CompactAMSPreconditioner(
    mat, grad_mat, freedofs=None,
    coord_x=[], coord_y=[], coord_z=[],
    cycle_type=1, print_level=0,
    subspace_solver=0, num_smooth=1)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mat` | `SparseMatrix` (real) | - | Real SPD matrix (curl-curl + mass) |
| `grad_mat` | `SparseMatrix` (real) | - | Discrete gradient G (HCurl -> H1) |
| `freedofs` | `BitArray` or `None` | `None` | Free DOFs |
| `coord_x` | `list[float]` | `[]` | Vertex x-coordinates |
| `coord_y` | `list[float]` | `[]` | Vertex y-coordinates |
| `coord_z` | `list[float]` | `[]` | Vertex z-coordinates |
| `cycle_type` | `int` | `1` | AMS cycle type (1="01210", 7="0201020") |
| `print_level` | `int` | `0` | Output verbosity level |
| `subspace_solver` | `int` | `0` | 0=CompactAMG, 1=SparseCholesky |
| `num_smooth` | `int` | `1` | Number of l1-Jacobi smoothing steps |

### Methods

| Method | Description |
|--------|-------------|
| `Update()` | Rebuild preconditioner (retains geometric information, recomputes only matrix-dependent parts) |
| `Update(new_mat)` | Rebuild preconditioner with a new matrix |

### Example

```python
import sparsesolv_ngsolve as ssn
from ngsolve import *
from ngsolve.krylovspace import CGSolver

# Magnetostatic problem
fes = HCurl(mesh, order=1, nograds=True, dirichlet="outer")
G_mat, h1_fes = fes.CreateGradient()
coord_x = [mesh.ngmesh.Points()[i+1][0] for i in range(mesh.nv)]
coord_y = [mesh.ngmesh.Points()[i+1][1] for i in range(mesh.nv)]
coord_z = [mesh.ngmesh.Points()[i+1][2] for i in range(mesh.nv)]

a = BilinearForm(fes)
a += nu * curl(u) * curl(v) * dx + 1e-6 * u * v * dx
a.Assemble()

pre = ssn.CompactAMSPreconditioner(
    a.mat, G_mat, freedofs=fes.FreeDofs(),
    coord_x=coord_x, coord_y=coord_y, coord_z=coord_z)

inv = CGSolver(a.mat, pre, tol=1e-10, maxiter=500)
gfu.vec.data = inv * f.vec
```

### Usage in Newton Iteration (Nonlinear Problems)

```python
pre = ssn.CompactAMSPreconditioner(a.mat, G_mat,
    freedofs=fes.FreeDofs(),
    coord_x=coord_x, coord_y=coord_y, coord_z=coord_z)

for k in range(max_newton):
    a.Assemble()          # Reassemble matrix with B-H curve
    pre.Update(a.mat)     # Rebuild preconditioner (geometry retained)
    inv = CGSolver(a.mat, pre, tol=1e-8, maxiter=500)
    delta = inv * rhs
    gfu.vec.data += delta
```

---

## ComplexCompactAMSPreconditioner

Compact AMS preconditioner for complex eddy current systems (fused Re/Im processing).

Header-only C++ implementation. No external libraries required.
Uses fused Re/Im SpMV to load matrix data once and process real and imaginary parts simultaneously.
Symmetric preconditioner (l1-Jacobi smoother) -- **COCRSolver recommended**.

### Constructor

```python
pre = ComplexCompactAMSPreconditioner(
    a_real_mat, grad_mat, freedofs=None,
    coord_x=[], coord_y=[], coord_z=[],
    ndof_complex=0, cycle_type=1, print_level=0,
    correction_weight=1.0, subspace_solver=0, num_smooth=1)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `a_real_mat` | `SparseMatrix` (real) | - | Real SPD auxiliary matrix (K + eps*M + \|omega\|*sigma*M) |
| `grad_mat` | `SparseMatrix` (real) | - | Discrete gradient G (HCurl -> H1) |
| `freedofs` | `BitArray` or `None` | `None` | Free DOFs |
| `coord_x` | `list[float]` | `[]` | Vertex x-coordinates |
| `coord_y` | `list[float]` | `[]` | Vertex y-coordinates |
| `coord_z` | `list[float]` | `[]` | Vertex z-coordinates |
| `ndof_complex` | `int` | `0` | Number of complex DOFs (0=auto-detect from matrix) |
| `cycle_type` | `int` | `1` | AMS cycle type (1="01210", 7="0201020") |
| `print_level` | `int` | `0` | Output verbosity level |
| `correction_weight` | `float` | `1.0` | Correction weight |
| `subspace_solver` | `int` | `0` | 0=CompactAMG, 1=SparseCholesky |
| `num_smooth` | `int` | `1` | Number of l1-Jacobi smoothing steps |

### Methods

| Method | Description |
|--------|-------------|
| `Update()` | Rebuild preconditioner (retains geometric information, recomputes only matrix-dependent parts) |
| `Update(new_a_real)` | Rebuild preconditioner with a new real auxiliary matrix |

### Example

```python
import sparsesolv_ngsolve as ssn

pre = ssn.ComplexCompactAMSPreconditioner(
    a_real_mat=a_real.mat, grad_mat=G_mat,
    freedofs=fes_real.FreeDofs(),
    coord_x=cx, coord_y=cy, coord_z=cz,
    ndof_complex=fes.ndof, cycle_type=1, print_level=0)

with TaskManager():
    inv = ssn.COCRSolver(a.mat, pre, freedofs=fes.FreeDofs(),
                          maxiter=500, tol=1e-10)
    gfu.vec.data = inv * f.vec
```

### Usage in Newton Iteration (Nonlinear Eddy Current)

```python
pre = ssn.ComplexCompactAMSPreconditioner(a_real.mat, G_mat, ...)

for k in range(max_newton):
    a_real.Assemble()
    pre.Update(a_real.mat)
    inv = ssn.COCRSolver(a_complex.mat, pre, tol=1e-10)
    delta = inv * rhs
    gfu.vec.data += delta
```

For details, see [compact_ams_cocr.md](compact_ams_cocr.md).

---

## has_compact_ams

Function to check whether Compact AMG/AMS support is available.

```python
ssn.has_compact_ams()  # -> True if Compact AMG/AMS support is available
```

---

## COCRSolver

COCR (Conjugate Orthogonal Conjugate Residual) solver. Native C++ implementation.
Optimal short-recurrence Krylov solver for complex symmetric systems (A^T=A, NOT Hermitian).

Uses non-conjugate inner product (x^T y). Minimizes ||A*r~||_2, providing smoother convergence than COCG.
Cost per iteration: 1 MatVec + 1 preconditioner application (same as CG).

**Reference**: Sogabe & Zhang (2007), J. Comput. Appl. Math., 199(2), 297-303.

### Usage 1: COCRSolver (with External Preconditioner)

Same interface as NGSolve's `CGSolver`. Use in combination with AMS preconditioner, etc.

```python
import sparsesolv_ngsolve

inv = sparsesolv_ngsolve.COCRSolver(mat, pre, freedofs=fes.FreeDofs(),
                                    maxiter=500, tol=1e-8, printrates=False)
gfu.vec.data = inv * f.vec
print(f"COCR converged in {inv.iterations} iterations")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mat` | `BaseMatrix` | - | Complex symmetric matrix |
| `pre` | `BaseMatrix` | - | Preconditioner matrix |
| `freedofs` | `BitArray` or `None` | `None` | Free DOFs. `None` means all DOFs are free |
| `maxiter` | `int` | `500` | Maximum number of iterations |
| `tol` | `float` | `1e-8` | Convergence tolerance (relative residual) |
| `printrates` | `bool` | `False` | Print convergence information |

### Usage 2: SparseSolvSolver(method="COCR")

Via the SparseSolvSolver unified interface. Uses internal IC preconditioner.

```python
solver = sparsesolv_ngsolve.SparseSolvSolver(mat, method="COCR",
    freedofs=fes.FreeDofs(), tol=1e-10, maxiter=1000)
gfu.vec.data = solver * f.vec
result = solver.last_result
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `iterations` | `int` | Number of iterations performed (COCRSolver only) |

### Convergence Criterion

```
sqrt(|rt^T * r|) / sqrt(|rt0^T * r0|) < tol
```
where `rt = M^{-1} * r` (preconditioned residual).
Equivalent convergence criterion to NGSolve's `CGSolver(conjugate=False)`.

### About COCG

COCG (Conjugate Orthogonal CG) is mathematically equivalent to `CGSolver(conjugate=False)`.
A separate class is not provided.

```python
from ngsolve.krylovspace import CGSolver
inv = CGSolver(a.mat, pre, conjugate=False, maxiter=500, tol=1e-8)
```

---

## GMRESSolver

Left-preconditioned GMRES (Generalized Minimal Residual) solver. Native C++ implementation.
Supports non-symmetric matrices. Use when the preconditioner is non-symmetric.

**Note**: For symmetric preconditioners (IC, SGS, Compact AMS), CG or COCR is recommended.
Use GMRES only when the preconditioner is non-symmetric.

### Constructor

```python
inv = sparsesolv_ngsolve.GMRESSolver(mat, pre, freedofs=None,
                                      maxiter=500, tol=1e-8,
                                      restart=30, printrates=False)
gfu.vec.data = inv * f.vec
print(f"GMRES converged in {inv.iterations} iterations")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mat` | `BaseMatrix` | - | Coefficient matrix (symmetric/non-symmetric) |
| `pre` | `BaseMatrix` | - | Left preconditioner matrix |
| `freedofs` | `BitArray` or `None` | `None` | Free DOFs. `None` means all DOFs are free |
| `maxiter` | `int` | `500` | Maximum number of iterations |
| `tol` | `float` | `1e-8` | Convergence tolerance (relative residual) |
| `restart` | `int` | `30` | Restart period |
| `printrates` | `bool` | `False` | Print convergence information |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `iterations` | `int` | Number of iterations performed |

### Example

```python
import sparsesolv_ngsolve as ssn
from ngsolve.krylovspace import CGSolver

# SGS preconditioner (non-symmetric) + GMRES
pre = ssn.SGSPreconditioner(a.mat, freedofs=fes.FreeDofs())
inv = ssn.GMRESSolver(a.mat, pre, freedofs=fes.FreeDofs(),
                       maxiter=500, tol=1e-10, restart=30)
gfu.vec.data = inv * f.vec
```

---

## CompactAMGPreconditioner

Classical AMG (Algebraic Multigrid) preconditioner. Designed for H1 Poisson systems.
Uses PMIS coarsening + classical interpolation + l1-Jacobi smoothing.

Also used internally as the subspace solver for CompactAMS.
Can be used standalone for SPD problems with H1 finite elements.

### Constructor

```python
pre = CompactAMGPreconditioner(mat, freedofs=None,
                                theta=0.25, max_levels=25,
                                min_coarse=50, num_smooth=1,
                                print_level=0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mat` | `SparseMatrix` (real) | - | SPD matrix |
| `freedofs` | `BitArray` or `None` | `None` | Free DOFs |
| `theta` | `float` | `0.25` | Strong connection threshold |
| `max_levels` | `int` | `25` | Maximum number of levels |
| `min_coarse` | `int` | `50` | Minimum DOF count at the coarsest level |
| `num_smooth` | `int` | `1` | Number of smoothing steps |
| `print_level` | `int` | `0` | Output verbosity level |

### Example

```python
import sparsesolv_ngsolve as ssn
from ngsolve.krylovspace import CGSolver

pre = ssn.CompactAMGPreconditioner(a.mat, freedofs=fes.FreeDofs())
inv = CGSolver(a.mat, pre, tol=1e-10, maxiter=500)
gfu.vec.data = inv * f.vec
```
