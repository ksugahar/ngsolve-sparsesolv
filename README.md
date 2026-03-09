# SparseSolv — Header-only AMS Preconditioner for NGSolve HCurl Problems

[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

**60x fewer iterations than ICCG on eddy current problems. No external dependencies.**

A header-only C++17 iterative solver library for [NGSolve](https://ngsolve.org/) finite element analysis.
Provides Compact AMS preconditioning for HCurl curl-curl problems (magnetostatics and eddy current),
IC/SGS preconditioners with ABMC parallel triangular solves, and COCR/GMRES Krylov solvers.

Supports both `double` and `std::complex<double>`, and is provided as a pybind11 extension module (`sparsesolv_ngsolve`) independent of the NGSolve source tree.

Fork of [JP-MARs/SparseSolv](https://github.com/JP-MARs/SparseSolv).

## Key Features

1. **ICCG** -- Conjugate gradient method with incomplete Cholesky (IC) preconditioning. Handles semi-definite matrices (curl-curl) via auto-shift. Supports complex symmetric matrices (unconjugated inner product)
2. **ABMC Parallelization** -- Parallel incomplete Cholesky triangular solve using Algebraic Block Multi-Color (ABMC) ordering [Iwashita et al. 2012]
3. **Compact AMS + COCR** -- Auxiliary space preconditioning + Krylov solver for HCurl problems. No external dependencies (header-only C++). Supports both real (magnetostatics + CG) and complex (eddy current + COCR). Supports nonlinear solvers (Newton iteration) via `Update()`

## Why SparseSolv?

NGSolve includes direct solvers and built-in BDDC. SparseSolv adds functionality that NGSolve does not provide by default:

1. **IC preconditioning + auto-shift** -- Curl-curl FEM matrices are semi-definite, and standard IC decomposition breaks down. SparseSolv's auto-shift IC detects breakdown and adjusts automatically

2. **ABMC parallel triangular solve** -- The triangular solve in IC preconditioning is inherently sequential. ABMC ordering eliminates this bottleneck and enables parallel forward/backward substitution [Iwashita et al. 2012]

3. **Compact AMS + COCR** -- A combination of auxiliary space preconditioning (Hiptmair-Xu 2007) and the COCR Krylov solver (Sogabe-Zhang 2007) for complex eddy current problems with HCurl finite elements (A = K + jw*sigma*M). See [docs/compact_ams_cocr.md](docs/compact_ams_cocr.md) for details

## Solver Selection Guide

| Problem | Finite Element Space | Recommended Method | Reason |
|---------|---------------------|-------------------|--------|
| Poisson | H1 | **ICCG** | Memory-efficient and fast |
| Curl-curl (real) | HCurl (`nograds=True`) | **Shifted-ICCG** | Handles semi-definite systems via auto-shift IC |
| Magnetostatics (real, large-scale) | HCurl (real, p=1) | **Compact AMS + CG** | Iteration count stable with respect to mesh size |
| Magnetostatics (nonlinear) | HCurl (real) | **Compact AMS + CG** | Supports Newton iteration via `Update()` |
| Eddy current (complex, large-scale) | HCurl (complex, p=1) | **Compact AMS + COCR** | Iteration count stable with respect to mesh size |
| Eddy current (complex, small-medium scale) | HCurl (complex) | **ICCG** (`conjugate=False`) | Memory-efficient |

## Performance

### Compact AMS + COCR (Complex Eddy Current)

Hiruma eddy current problem (copper coil + iron core mu_r=1000, 30 kHz, tol=1e-10).
**Environment**: Intel Xeon (8 cores), Windows Server 2022, MSVC 2022, MKL 2024.2.

| Mesh | HCurl DOFs | Iterations | Time | ms/iter | Memory |
|------|----------:|-----------:|-----:|--------:|-------:|
| mesh1_2.5T | 155,527 | 144 | 4.5s | 25.8 | 368 MB |
| mesh1_3.5T | 197,395 | 168 | 7.3s | 37.7 | 460 MB |
| mesh1_5.5T | 331,595 | 249 | 16.2s | 57.3 | 725 MB |
| mesh1_20.5T | 1,441,102 | 499 | 222.6s | 396.2 | 2,933 MB |

Comparison with ABMC-ICCG (IC preconditioning only) on mesh1_3.5T (197k DOFs):

| Method | Iterations | Time | Status |
|--------|----------:|-----:|--------|
| Compact AMS + COCR | 168 | 7.2s | Converged |
| ABMC-ICCG | 17,178 | 438.4s | Not converged (res=2.8e-10) |

IC preconditioning cannot handle the curl-curl null space in HCurl discretization.
AMS resolves this through discrete gradient correction and Nedelec interpolation correction.

Run benchmarks: `python examples/hiruma/bench_compact_ams.py --all`

### ABMC ICCG

3D HCurl curl-curl problem (148K DOFs, order 2, 8 threads):

| Solver | Iterations | Time | vs ICCG |
|--------|-----------|------|---------|
| ICCG | 463 | 4.4 s | 1.0x |
| ICCG + ABMC (8 colors) | 414 | 2.6 s | **1.7x** |

## Documentation

Detailed documentation is available in [docs/](docs/):
- [Architecture](docs/architecture.md) -- Source code structure and design
- [Algorithms](docs/algorithms.md) -- Algorithm descriptions (IC, SGS-MRTR, CG, ABMC)
- [Compact AMS + COCR](docs/compact_ams_cocr.md) -- Auxiliary space preconditioning for complex eddy current problems
- [ABMC Implementation Guide](docs/abmc_implementation_details.md) -- Parallel triangular solve, performance analysis
- [API Reference](docs/api_reference.md) -- Python API reference
- [Tutorials](docs/tutorials.md) -- Practical examples
- [Developer Information](docs/development.md) -- Build, testing, and development notes

## Features

### Preconditioners
- **Compact AMS** (Auxiliary-space Maxwell Solver) -- Auxiliary space preconditioning for HCurl. Header-only C++ implementation of the Hiptmair-Xu (2007) algorithm. No external dependencies
  - `CompactAMSPreconditioner`: For real HCurl (magnetostatics), used with CG solver
  - `ComplexCompactAMSPreconditioner`: For complex eddy current, fused Re/Im processing
  - `Update()`: Supports nonlinear solvers (Newton iteration). Retains geometric information and rebuilds only matrix-dependent parts
  - CompactAMG: PMIS coarsening + l1-Jacobi + V-cycle, DualMult fusion
- **IC** (Incomplete Cholesky) -- Shifted IC(0), auto-shift for semi-definite matrices
- **SGS** (Symmetric Gauss-Seidel) -- No factorization required

### Iterative Solvers
- **COCR** (Conjugate Orthogonal Conjugate Residual) -- Short-recurrence Krylov solver for complex symmetric systems (A^T=A) [Sogabe-Zhang 2007]. O(n) memory, no restart required
- **CG** (Conjugate Gradient) -- For SPD systems, supports complex symmetric (`conjugate=False`)
- **SGS-MRTR** -- MRTR with built-in split formula

### Combined Methods
- **Compact AMS + COCR** -- For complex eddy current (large-scale HCurl p=1)
- **ICCG** -- CG + IC preconditioning (optional ABMC parallel triangular solve)
- **SGSMRTR** -- SGS-MRTR (self-contained)

### Advanced Features
- Auto-shift IC decomposition for semi-definite matrices (curl-curl problems)
- Condition number improvement via diagonal scaling
- Parallel triangular solve using ABMC (Algebraic Block Multi-Color) ordering
- Numerical breakdown detection and convergence-aware recovery
- Best-result tracking (returns best iterate when not converged)

### Parallelization

All parallel processing is dispatched at compile time via `core/parallel.hpp`:

| Build Configuration | Backend | Use Case |
|--------------------|---------|----------|
| `SPARSESOLV_USE_NGSOLVE_TASKMANAGER` | NGSolve TaskManager (`ngcore::ParallelFor/ParallelReduce`) | NGSolve integration |
| `_OPENMP` | OpenMP `#pragma omp parallel for` | Standalone + OpenMP |
| (neither) | Serial execution | Standalone (no threads) |

## Installation

### Pre-built Wheel (Recommended)

Download the platform-appropriate `.whl` file from the [Releases](https://github.com/ksugahar/ngsolve-sparsesolv/releases) page and install:

```bash
pip install sparsesolv_ngsolve-2.6.0-cp312-cp312-win_amd64.whl
```

Adjust the filename to match your Python version and OS.

### Build from Source

Requires [Git for Windows](https://gitforwindows.org/) (or equivalent), CMake 3.16+,
a C++17 compiler, and NGSolve (`pip install ngsolve` or source build).

```bash
pip install git+https://github.com/ksugahar/ngsolve-sparsesolv.git
```

Or clone and build locally:

```bash
git clone https://github.com/ksugahar/ngsolve-sparsesolv.git
cd ngsolve-sparsesolv
pip install .
```

For development builds (editable install, manual CMake), see [docs/development.md](docs/development.md).

### Verification

```bash
python -c "from sparsesolv_ngsolve import SparseSolvSolver; print('OK')"
```

## Usage with NGSolve

### Quick Start (ICCG)

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

### ABMC-Parallelized ICCG

```python
# Parallelize IC preconditioning triangular solve with ABMC
solver = SparseSolvSolver(a.mat, method="ICCG",
                          freedofs=fes.FreeDofs(), tol=1e-10,
                          use_abmc=True,
                          abmc_block_size=4,
                          abmc_num_colors=4)
gfu.vec.data = solver * f.vec
```

### Compact AMS + COCR (Complex Eddy Current)

```python
import sparsesolv_ngsolve as ssn
from ngsolve import *

# Assemble complex system A = K + jw*sigma*M (omitted)
# Build real SPD auxiliary matrix
fes_real = HCurl(mesh, order=1, nograds=True,
                 dirichlet="dirichlet", complex=False)
u_r, v_r = fes_real.TnT()
a_real = BilinearForm(fes_real)
a_real += nu_cf * curl(u_r) * curl(v_r) * dx
a_real += 1e-6 * nu_cf * u_r * v_r * dx
a_real += abs(omega) * sigma_cf * u_r * v_r * dx("cond")
a_real.Assemble()

# Discrete gradient matrix and vertex coordinates
G_mat, h1_fes = fes_real.CreateGradient()
coord_x = [mesh.ngmesh.Points()[i+1][0] for i in range(mesh.nv)]
coord_y = [mesh.ngmesh.Points()[i+1][1] for i in range(mesh.nv)]
coord_z = [mesh.ngmesh.Points()[i+1][2] for i in range(mesh.nv)]

# Compact AMS preconditioner + COCR solver
pre = ssn.ComplexCompactAMSPreconditioner(
    a_real_mat=a_real.mat, grad_mat=G_mat,
    freedofs=fes_real.FreeDofs(),
    coord_x=coord_x, coord_y=coord_y, coord_z=coord_z,
    ndof_complex=fes.ndof, cycle_type=1, print_level=0)

with TaskManager():
    inv = ssn.COCRSolver(a.mat, pre,
                         freedofs=fes.FreeDofs(),
                         maxiter=500, tol=1e-10)
    gfu.vec.data = inv * f.vec

print(f"Converged in {inv.iterations} iterations")
```

See [docs/compact_ams_cocr.md](docs/compact_ams_cocr.md) for details.

### Preconditioner + NGSolve CGSolver

```python
from sparsesolv_ngsolve import ICPreconditioner, SGSPreconditioner
from ngsolve.krylovspace import CGSolver

# IC preconditioner + NGSolve CG
pre = ICPreconditioner(a.mat, freedofs=fes.FreeDofs(), shift=1.05)
inv = CGSolver(a.mat, pre, tol=1e-10, maxiter=2000)
gfu.vec.data = inv * f.vec
```

### SparseSolvSolver Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | `"ICCG"` | Solver method: `ICCG`, `SGSMRTR`, `CG`, `COCR` |
| `tol` | float | `1e-10` | Convergence tolerance |
| `maxiter` | int | `1000` | Maximum number of iterations |
| `shift` | float | `1.05` | IC shift parameter (stability) |
| `auto_shift` | bool | `False` | Automatic shift increase on IC breakdown |
| `diagonal_scaling` | bool | `False` | Diagonal scaling |
| `save_best_result` | bool | `True` | Track and restore best iterate |
| `conjugate` | bool | `False` | Conjugated inner product for Hermitian systems |
| `use_abmc` | bool | `False` | Parallel triangular solve via ABMC ordering |
| `abmc_block_size` | int | `4` | Number of rows per ABMC block |
| `abmc_num_colors` | int | `4` | Target number of ABMC colors |

### Python Class List

```python
from sparsesolv_ngsolve import (
    # Compact AMS preconditioners (for HCurl, supports Update())
    CompactAMSPreconditioner,      # Real HCurl (magnetostatics) + CG
    ComplexCompactAMSPreconditioner,   # Complex eddy current, fused Re/Im + COCR

    # IC/SGS preconditioners
    ICPreconditioner,              # For use with NGSolve CGSolver
    SGSPreconditioner,             # For use with NGSolve CGSolver

    # Solvers
    SparseSolvSolver,              # Combined solver (ICCG, SGSMRTR)
    COCRSolver,                    # COCR (complex symmetric systems, native C++)
    GMRESSolver,                   # GMRES (non-symmetric systems, for AMS preconditioning)
    SparseSolvResult,              # Solve result
)
```

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
```

## Directory Structure

```
ngsolve-sparsesolv/
├── docs/                        # Documentation
├── include/sparsesolv/         # Header-only library
│   ├── sparsesolv.hpp          # Main header
│   ├── core/                   # Types, configuration, matrix views
│   │   ├── parallel.hpp        # Parallelization abstraction layer (TaskManager/OpenMP/serial)
│   │   └── abmc_ordering.hpp   # ABMC ordering
│   ├── preconditioners/        # IC, SGS, Compact AMS implementations
│   │   ├── ic_preconditioner.hpp          # Incomplete Cholesky (auto-shift)
│   │   ├── sgs_preconditioner.hpp         # Symmetric Gauss-Seidel
│   │   ├── compact_amg.hpp                # Classical AMG (PMIS, l1-Jacobi, DualMult)
│   │   ├── compact_ams.hpp                # AMS preconditioner (Hiptmair-Xu 2007)
│   │   └── complex_compact_ams.hpp        # Fused Re/Im ComplexCompactAMS
│   ├── solvers/                # CG, COCR, SGS-MRTR implementations
│   └── ngsolve/                # NGSolve BaseMatrix wrapper + pybind11
├── examples/
│   └── hiruma/                 # Eddy current benchmark (30 kHz)
│       ├── bench_compact_ams.py    # Compact AMS + COCR benchmark
│       ├── bench_ams_vs_abmc.py    # AMS vs ABMC-ICCG comparison
│       └── eddy_current.py         # Eddy current analysis
├── tests/
│   └── test_sparsesolv.py      # Solver and preconditioner tests
├── sparsesolv_ngsolve.pyi      # Python type stubs
├── CMakeLists.txt
└── LICENSE
```

## References

1. T. Iwashita, H. Nakashima, Y. Takahashi,
   "Algebraic Block Multi-Color Ordering Method for Parallel Multi-Threaded
   Sparse Triangular Solver in ICCG Method",
   *Proc. IEEE IPDPS*, 2012.
   [DOI: 10.1109/IPDPS.2012.51](https://doi.org/10.1109/IPDPS.2012.51)

2. J.A. Meijerink, H.A. van der Vorst,
   "An iterative solution method for linear systems of which the coefficient matrix
   is a symmetric M-matrix",
   *Math. Comp.*, Vol. 31, No. 137, pp. 148-162, 1977.

3. R. Hiptmair, J. Xu,
   "Nodal auxiliary space preconditioning in H(curl) and H(div) spaces",
   *SIAM J. Numer. Anal.*, Vol. 45, No. 6, pp. 2483-2509, 2007.

4. T. Sogabe, S.-L. Zhang,
   "A COCR method for solving complex symmetric linear systems",
   *J. Comput. Appl. Math.*, Vol. 199, No. 2, pp. 297-303, 2007.

5. T.V. Kolev, P.S. Vassilevski,
   "Parallel auxiliary space AMG for H(curl) problems",
   *J. Comput. Math.*, Vol. 27, No. 5, pp. 604-623, 2009.

6. JP-MARs/SparseSolv,
   https://github.com/JP-MARs/SparseSolv

## License

This project is licensed under the [Mozilla Public License 2.0](LICENSE).
