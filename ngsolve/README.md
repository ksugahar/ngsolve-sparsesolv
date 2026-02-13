# NGSolve Integration Guide

## Overview

SparseSolv integrates into NGSolve as a header-only library providing
IC/ILU/SGS preconditioners and ICCG/ICMRTR/SGSMRTR iterative solvers.

## File Placement

Copy the following into the NGSolve source tree (`linalg/` directory):

```
ngsolve/linalg/
├── sparsesolv/                              ← from include/sparsesolv/{core,preconditioners,solvers}/ + sparsesolv.hpp
│   ├── sparsesolv.hpp
│   ├── core/
│   │   ├── types.hpp
│   │   ├── solver_config.hpp
│   │   ├── sparse_matrix_view.hpp
│   │   └── preconditioner.hpp
│   ├── preconditioners/
│   │   ├── ic_preconditioner.hpp
│   │   ├── ilu_preconditioner.hpp
│   │   └── sgs_preconditioner.hpp
│   └── solvers/
│       ├── iterative_solver.hpp
│       ├── cg_solver.hpp
│       ├── mrtr_solver.hpp
│       └── sgs_mrtr_solver.hpp
├── sparsesolv_precond.hpp                   ← from include/sparsesolv/ngsolve/sparsesolv_precond.hpp
├── la.hpp                                   ← add: #include "sparsesolv_precond.hpp"
├── python_linalg.cpp                        ← add: SparseSolv bindings (see python_bindings.cpp)
└── CMakeLists.txt                           ← add: install sparsesolv headers
```

## CMakeLists.txt Changes

Add to `linalg/CMakeLists.txt`:

```cmake
# SparseSolv header-only library
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/sparsesolv)

# Install headers
install(FILES sparsesolv_precond.hpp
        DESTINATION ${NGSOLVE_INSTALL_DIR_INCLUDE})
install(DIRECTORY sparsesolv
        DESTINATION ${NGSOLVE_INSTALL_DIR_INCLUDE}
        FILES_MATCHING PATTERN "*.hpp")
```

## python_linalg.cpp Changes

1. Add at the top (with other includes):
   ```cpp
   #include "sparsesolv_precond.hpp"
   ```

2. Append the contents of `python_bindings.cpp` before the closing brace
   of `ExportNgla()`.

## la.hpp Changes

Add:
```cpp
#include "sparsesolv_precond.hpp"
```

## sparsesolv_precond.hpp Include Path

When placed in `linalg/sparsesolv_precond.hpp`, the include path
`#include "sparsesolv/sparsesolv.hpp"` should work since the sparsesolv/
directory is at the same level.

**Note**: The copy in `include/sparsesolv/ngsolve/` uses
`#include "../sparsesolv.hpp"` (relative path within the SparseSolv repo).
The sync script automatically adjusts this path.

## Sync Script

Use `scripts/sync_to_ngsolve.py` to automate the copy:

```bash
python scripts/sync_to_ngsolve.py --ngsolve-dir /path/to/ngsolve_ksugahar
```

## Python API

After integration, the following are available in `ngsolve.la`:

```python
from ngsolve.la import (
    # Real (double) versions
    ICPreconditioner,              # Incomplete Cholesky preconditioner
    SGSPreconditioner,             # Symmetric Gauss-Seidel preconditioner
    ILUPreconditioner,             # Incomplete LU preconditioner
    SparseSolvSolver,              # Unified iterative solver

    # Complex versions
    ComplexICPreconditioner,       # IC preconditioner for complex matrices
    ComplexSGSPreconditioner,      # SGS preconditioner for complex matrices
    ComplexILUPreconditioner,      # ILU preconditioner for complex matrices
    ComplexSparseSolvSolver,       # Iterative solver for complex systems

    # Result type (shared by real and complex)
    SparseSolvResult,              # Solve result (converged, iterations, residual)
)
```
