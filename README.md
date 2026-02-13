# SparseSolv

Header-only C++17 iterative solver library for sparse linear systems, designed for integration with [NGSolve](https://ngsolve.org/).

Fork of [JP-MARs/SparseSolv](https://github.com/JP-MARs/SparseSolv), specialized for NGSolve finite element applications.

## Features

### Preconditioners
- **IC** (Incomplete Cholesky) - shifted IC(0) with auto-shift for semi-definite systems
- **ILU** (Incomplete LU) - for general non-symmetric matrices
- **SGS** (Symmetric Gauss-Seidel) - no factorization required

### Iterative Solvers
- **CG** (Conjugate Gradient) - for SPD systems
- **MRTR** (Modified Residual Tri-diagonal Reduction) - minimizes residual norm
- **SGS-MRTR** - MRTR with built-in SGS using split formula (no separate preconditioner)

### Combined Methods
- **ICCG** - CG + IC preconditioner
- **ICMRTR** - MRTR + IC preconditioner
- **SGSMRTR** - SGS-MRTR (self-contained)

### Advanced Features
- Auto-shift IC decomposition for semi-definite matrices (curl-curl problems)
- Diagonal scaling for improved conditioning
- Numerical breakdown detection with convergence-aware recovery
- Best-result tracking (returns best iterate if solver doesn't converge)
- Residual history recording
- Template support for `double` and `std::complex<double>`

### Parallelism

All parallel operations are abstracted via `core/parallel.hpp` with compile-time dispatch:

| Build Configuration | Backend | Use Case |
|---|---|---|
| `SPARSESOLV_USE_NGSOLVE_TASKMANAGER` | NGSolve TaskManager (`ngcore::ParallelFor/ParallelReduce`) | NGSolve integration |
| `_OPENMP` | OpenMP `#pragma omp parallel for` | Standalone with OpenMP |
| (neither) | Serial loops | Standalone without threading |

Parallel operations include:
- **SpMV** (sparse matrix-vector product) - row-parallel
- **AXPY** (vector updates) - element-parallel
- **Dot products / norms** - parallel reduction (supports `complex<double>`)
- **Triangular solves** - level-scheduled wavefront parallelism (IC, ILU, SGS, SGS-MRTR)

#### Level-Scheduled Triangular Solves

Forward/backward substitution in preconditioners uses level scheduling:
rows at the same dependency level are independent and processed in parallel.

```
Level 0: [row 0, row 5, row 12, ...]   <- independent -> parallel_for
Level 1: [row 1, row 6, row 13, ...]   <- after Level 0 -> parallel_for
Level 2: [row 2, row 7, row 14, ...]   <- after Level 1 -> parallel_for
```

Level schedules are built once during preconditioner setup (O(nnz) cost)
and reused every iteration.

## Directory Structure

```
SparseSolv/
├── include/sparsesolv/         # Header-only library
│   ├── sparsesolv.hpp          # Main header (convenience includes)
│   ├── core/                   # Types, config, matrix view, preconditioner base
│   │   ├── parallel.hpp        # Parallelism abstraction (TaskManager/OpenMP/serial)
│   │   └── level_schedule.hpp  # Level scheduling for triangular solves
│   ├── preconditioners/        # IC, ILU, SGS implementations
│   ├── solvers/                # CG, MRTR, SGS-MRTR implementations
│   └── ngsolve/                # NGSolve BaseMatrix wrappers
├── ngsolve/                    # NGSolve integration reference
│   ├── python_bindings.cpp     # pybind11 code for python_linalg.cpp
│   └── README.md               # Integration guide
├── tests/                      # Test suite
│   └── test_sparsesolv.py      # NGSolve pytest (17 tests)
├── scripts/                    # Utilities
│   └── sync_to_ngsolve.py      # Sync headers to NGSolve source tree
├── CMakeLists.txt              # Header-only INTERFACE library
└── LICENSE
```

## NGSolve Integration

### Quick Start (Python)

```python
from ngsolve import *
from ngsolve.la import SparseSolvSolver

mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
fes = H1(mesh, order=2, dirichlet="left|right|top|bottom")
u, v = fes.TnT()

a = BilinearForm(fes)
a += grad(u)*grad(v)*dx
a.Assemble()

f = LinearForm(fes)
f += 1*v*dx
f.Assemble()

gfu = GridFunction(fes)
solver = SparseSolvSolver(a.mat, method="ICCG",
                          freedofs=fes.FreeDofs(), tol=1e-10)
gfu.vec.data = solver * f.vec
```

### Preconditioners with NGSolve's CGSolver

```python
from ngsolve.la import ICPreconditioner, SGSPreconditioner, ILUPreconditioner
from ngsolve.krylovspace import CGSolver

# IC preconditioner + NGSolve CG
pre = ICPreconditioner(a.mat, freedofs=fes.FreeDofs(), shift=1.05)
pre.Update()
inv = CGSolver(a.mat, pre, tol=1e-10, maxiter=2000)
gfu.vec.data = inv * f.vec
```

### 3D Curl-Curl (Semi-Definite System)

```python
from netgen.occ import Box, Pnt

box = Box(Pnt(0,0,0), Pnt(1,1,1))
for face in box.faces:
    face.name = "outer"
mesh = box.GenerateMesh(maxh=0.3)

fes = HCurl(mesh, order=1, dirichlet="outer", nograds=True)
u, v = fes.TnT()

a = BilinearForm(fes)
a += curl(u)*curl(v)*dx
a.Assemble()

f = LinearForm(fes)
f += CF((0,0,1))*v*dx
f.Assemble()

# Auto-shift IC handles semi-definite curl-curl matrix
solver = SparseSolvSolver(a.mat, method="ICCG",
                          freedofs=fes.FreeDofs(),
                          tol=1e-8, maxiter=2000, shift=1.0)
solver.auto_shift = True
solver.diagonal_scaling = True

gfu = GridFunction(fes)
gfu.vec.data = solver * f.vec
```

### SparseSolvSolver Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `method` | str | `"ICCG"` | Solver method: `ICCG`, `ICMRTR`, `SGSMRTR`, `CG`, `MRTR` |
| `tol` | float | `1e-10` | Convergence tolerance |
| `maxiter` | int | `1000` | Maximum iterations |
| `shift` | float | `1.05` | IC shift parameter (stability) |
| `auto_shift` | bool | `False` | Auto-increase shift on IC breakdown |
| `diagonal_scaling` | bool | `False` | Scale matrix by diagonal |
| `save_best_result` | bool | `True` | Track and restore best iterate |
| `save_residual_history` | bool | `False` | Record residual per iteration |
| `divergence_check` | bool | `False` | Early termination on stagnation |
| `printrates` | bool | `False` | Print convergence info to stdout |

### Available Python Classes

```python
from ngsolve.la import (
    # Factory functions (auto-dispatch real/complex based on mat.IsComplex())
    ICPreconditioner,              # Use with NGSolve's CGSolver
    SGSPreconditioner,             # Use with NGSolve's CGSolver
    ILUPreconditioner,             # Use with NGSolve's GMRESSolver
    SparseSolvSolver,              # Standalone solver (ICCG, ICMRTR, SGSMRTR)

    SparseSolvResult,              # Solve result object
)
```

Factory functions automatically dispatch to the correct type (`double` or `complex<double>`)
based on the matrix type. Direct access to typed classes is also available:

```python
from ngsolve.la import (
    # Real (double)
    SparseSolvSolverD, ICPreconditionerD, SGSPreconditionerD, ILUPreconditionerD,
    # Complex
    ComplexSparseSolvSolver, ComplexICPreconditioner, ComplexSGSPreconditioner, ComplexILUPreconditioner,
)
```

### Syncing to NGSolve

```bash
python scripts/sync_to_ngsolve.py --ngsolve-dir /path/to/ngsolve
```

See [ngsolve/README.md](ngsolve/README.md) for detailed integration instructions.

## Standalone Usage (C++)

```cpp
#include <sparsesolv/sparsesolv.hpp>

// Create CSR matrix view (zero-copy)
sparsesolv::SparseMatrixView<double> A(rows, cols, row_ptr, col_idx, values);

// Solve with ICCG
sparsesolv::SolverConfig config;
config.tolerance = 1e-10;
config.max_iterations = 1000;

auto result = sparsesolv::solve_iccg(A, b, x, size, config);
if (result.converged) {
    // solution in x
}
```

## Test Results

All tests pass on Windows Server 2022 (Intel 8-core, MSVC, NGSolve TaskManager).

### pytest (22 tests)

```
test_sparsesolv_solver_2d_poisson[ICCG]                       PASSED
test_sparsesolv_solver_2d_poisson[ICMRTR]                      PASSED
test_sparsesolv_solver_2d_poisson[SGSMRTR]                     PASSED
test_sparsesolv_solver_3d_poisson                              PASSED
test_sparsesolv_solver_vs_direct                               PASSED
test_preconditioners_with_ngsolve_cg[IC]                       PASSED
test_preconditioners_with_ngsolve_cg[ILU]                      PASSED
test_preconditioners_with_ngsolve_cg[SGS]                      PASSED
test_auto_shift_curl_curl                                      PASSED
test_residual_history / test_no_residual_history               PASSED
test_properties / test_invalid_method / test_operator_interface PASSED
test_divergence_check_early_termination                        PASSED
test_save_best_result_restores_best                            PASSED
test_factory_auto_dispatch_complex_solver                      PASSED
test_factory_auto_dispatch_complex_precond[IC/ILU/SGS]         PASSED
test_factory_returns_typed_class / _complex                    PASSED
```

### Comprehensive verification (17 tests)

| Test | Content | Result |
|---|---|---|
| All solver methods | ICCG, ICMRTR, SGSMRTR, CG, MRTR on 2D Poisson | PASS |
| Preconditioners | IC, SGS, ILU with NGSolve CGSolver | PASS |
| 3D Poisson | ICCG + SGSMRTR | PASS |
| 2D/3D Elasticity | Cantilever beam, ICCG/ICMRTR/SGSMRTR vs direct solver | PASS |
| Order convergence | p=1,2,3,4 with ICCG | PASS |
| Mesh refinement | h=0.3, 0.15, 0.075 with ICCG | PASS |
| NGSolve comparison | vs built-in Jacobi+CG, direct solver | PASS |
| 2D Magnetostatic | Iron core (mu_r=1000), material jump | PASS |
| 3D Curl-Curl | nograds=True, auto-shift + diagonal scaling | PASS |
| Best-result / history / shift sensitivity / properties | Various | PASS |

### Parallelization Benchmark

Environment: Windows Server 2022, Intel 8-core, MSVC, NGSolve TaskManager.

#### 3D Poisson (H1, order=2)

| Free DOFs | Method | 1T (ms) | 4T (ms) | Speedup |
|---|---|---|---|---|
| 7,156 | ICCG | 33.3 | 30.0 | 1.11x |
| | ICMRTR | 34.9 | 27.7 | 1.26x |
| | SGSMRTR | 28.9 | 18.7 | 1.55x |
| 10,042 | ICCG | 59.9 | 49.2 | 1.22x |
| | ICMRTR | 60.6 | 53.1 | 1.14x |
| | SGSMRTR | 33.7 | 27.6 | 1.22x |

#### 3D Elasticity (VectorH1, order=2)

| Free DOFs | Method | 1T (ms) | 4T (ms) | Speedup | 8T (ms) | Speedup |
|---|---|---|---|---|---|---|
| 14,523 | ICCG | 731 | 500 | 1.46x | 513 | 1.43x |
| | ICMRTR | 737 | 516 | 1.43x | 546 | 1.35x |
| | SGSMRTR | 326 | 180 | **1.81x** | 196 | 1.67x |

#### Analysis

- Speedup is modest at these problem sizes (< 15K DOF) due to barrier synchronization
  overhead in level-scheduled triangular solves.
- SGSMRTR shows the best scalability (no separate preconditioner setup/apply overhead).
- 8 threads provides diminishing returns beyond 4 (memory bandwidth saturation).
- Problems < 3K DOF show no benefit (parallelization overhead dominates).
- Larger problems (100K+ DOF) are expected to scale better as the
  computation-to-synchronization ratio improves.

## License

See [LICENSE](LICENSE) for details.
