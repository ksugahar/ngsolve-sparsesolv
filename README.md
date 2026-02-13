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

## Directory Structure

```
SparseSolv/
├── include/sparsesolv/         # Header-only library
│   ├── sparsesolv.hpp          # Main header (convenience includes)
│   ├── core/                   # Types, config, matrix view, preconditioner base
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

### Available Python Classes

```python
from ngsolve.la import (
    ICPreconditioner,      # Use with NGSolve's CGSolver
    SGSPreconditioner,     # Use with NGSolve's CGSolver
    ILUPreconditioner,     # Use with NGSolve's GMRESSolver
    SparseSolvSolver,      # Standalone solver (ICCG, ICMRTR, SGSMRTR)
    SparseSolvResult,      # Solve result object
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

## License

See [LICENSE](LICENSE) for details.
