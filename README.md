# SparseSolv

Header-only C++17 iterative solver library for sparse linear systems, with a standalone Python extension module for [NGSolve](https://ngsolve.org/).

Fork of [JP-MARs/SparseSolv](https://github.com/JP-MARs/SparseSolv), specialized for NGSolve finite element applications.

## Policy

**SparseSolv is provided as an independent module, separate from NGSolve.**
It is NOT embedded into NGSolve's source tree. Instead, it is built as a standalone
pybind11 extension module (`sparsesolv_ngsolve.pyd`) that links against an installed NGSolve.

## Features

### Preconditioners
- **IC** (Incomplete Cholesky) - shifted IC(0) with auto-shift for semi-definite systems
- **SGS** (Symmetric Gauss-Seidel) - no factorization required

### Iterative Solvers
- **CG** (Conjugate Gradient) - for SPD systems
- **SGS-MRTR** - MRTR with built-in SGS using split formula (no separate preconditioner)

### Combined Methods
- **ICCG** - CG + IC preconditioner
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

## Building the NGSolve Extension Module

### Prerequisites

- NGSolve installed (from source) with CMake config files
- CMake 3.16+
- C++17 compiler (MSVC 2022, GCC 10+, Clang 10+)

### Build

```bash
mkdir build && cd build
cmake .. -DSPARSESOLV_BUILD_NGSOLVE=ON \
         -DNGSOLVE_INSTALL_DIR=/path/to/ngsolve/install/cmake
cmake --build . --config Release
```

This produces `sparsesolv_ngsolve.pyd` (Windows) or `sparsesolv_ngsolve.so` (Linux/macOS).

### Installation

Copy the `.pyd`/`.so` file to a directory on your Python path, or to the NGSolve
site-packages directory.

## NGSolve Usage

### Quick Start

```python
from ngsolve import *
from sparsesolv_ngsolve import SparseSolvSolver

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
from sparsesolv_ngsolve import ICPreconditioner, SGSPreconditioner
from ngsolve.krylovspace import CGSolver

# IC preconditioner + NGSolve CG
pre = ICPreconditioner(a.mat, freedofs=fes.FreeDofs(), shift=1.05)
inv = CGSolver(a.mat, pre, tol=1e-10, maxiter=2000)
gfu.vec.data = inv * f.vec
```

### 3D Curl-Curl (Semi-Definite System)

```python
from netgen.occ import Box, Pnt

box = Box(Pnt(0,0,0), Pnt(1,1,1))
for face in box.faces:
    face.name = "outer"
mesh = Mesh(box.GenerateMesh(maxh=0.3))

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

### Application: Kelvin Transformation for Electromagnetic Analysis

SparseSolv is particularly useful for Kelvin transformation problems in
electromagnetic field analysis. The Kelvin transformation maps an infinite
exterior domain to a bounded domain using periodic boundary conditions,
enabling finite element analysis of open-boundary problems.

**Key workflow:**
1. Create interior domain and Kelvin-transformed exterior domain
2. Connect them with periodic BCs using `Identify()` + `Glue()`
3. Use NGSolve's `Periodic()` FE space
4. Solve with SparseSolv (ICCG for H-formulation, or with auto-shift for curl-curl)

Example applications:
- Magnetic sphere in uniform field (2D/3D)
- Parallel wire configurations
- Coil A-formulation with unbounded domain
- Multipole magnet analysis (dipole, quadrupole)

See `S:\NGSolve\NGSolve\2025_12_14_Kelvin変換\` for full examples.

### SparseSolvSolver Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `method` | str | `"ICCG"` | Solver method: `ICCG`, `SGSMRTR`, `CG` |
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
from sparsesolv_ngsolve import (
    # Factory functions (auto-dispatch real/complex based on mat.IsComplex())
    ICPreconditioner,              # Use with NGSolve's CGSolver
    SGSPreconditioner,             # Use with NGSolve's CGSolver
    SparseSolvSolver,              # Standalone solver (ICCG, SGSMRTR)

    SparseSolvResult,              # Solve result object
)
```

Factory functions automatically dispatch to the correct type (`double` or `complex<double>`)
based on the matrix type.

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

## Directory Structure

```
SparseSolv/
├── include/sparsesolv/         # Header-only library
│   ├── sparsesolv.hpp          # Main header (convenience includes)
│   ├── core/                   # Types, config, matrix view, preconditioner base
│   │   ├── parallel.hpp        # Parallelism abstraction (TaskManager/OpenMP/serial)
│   │   └── level_schedule.hpp  # Level scheduling for triangular solves
│   ├── preconditioners/        # IC, SGS implementations
│   ├── solvers/                # CG, SGS-MRTR implementations
│   └── ngsolve/                # NGSolve BaseMatrix wrappers + pybind11 export
├── ngsolve/
│   ├── python_module.cpp       # Standalone pybind11 module entry point
│   └── README.md               # Integration reference
├── tests/                      # Test suite
│   └── test_sparsesolv.py      # NGSolve pytest
├── CMakeLists.txt              # Header-only library + NGSolve module build
└── LICENSE
```

## License

See [LICENSE](LICENSE) for details.
