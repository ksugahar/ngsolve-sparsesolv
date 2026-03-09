# Architecture and Source Code Guide

## Directory Structure

```
ngsolve-sparsesolv/
├── include/sparsesolv/           # Header-only library core
│   ├── sparsesolv.hpp            # Main header (includes all components)
│   ├── core/                     # Foundation components
│   │   ├── types.hpp             # Type definitions (index_t, complex_t)
│   │   ├── constants.hpp         # Numerical constants (tolerances, thresholds)
│   │   ├── solver_config.hpp     # SolverConfig struct
│   │   ├── sparse_matrix_view.hpp # Non-owning CSR matrix view (zero-copy)
│   │   ├── sparse_matrix_coo.hpp # COO format (for assembly)
│   │   ├── sparse_matrix_csr.hpp # CSR format (for storage)
│   │   ├── dense_matrix.hpp      # Dense matrix + LU inverse
│   │   ├── preconditioner.hpp    # Preconditioner base class (template)
│   │   ├── parallel.hpp          # Parallelization abstraction layer (TaskManager/OpenMP/serial)
│   │   ├── level_schedule.hpp    # Level scheduling (triangular solve parallelization)
│   │   ├── abmc_ordering.hpp     # ABMC ordering (triangular solve parallelization)
│   │   └── rcm_ordering.hpp      # RCM bandwidth reduction ordering
│   ├── preconditioners/          # Preconditioner implementations
│   │   ├── ic_preconditioner.hpp # Incomplete Cholesky (IC) factorization
│   │   ├── sgs_preconditioner.hpp # Symmetric Gauss-Seidel (SGS)
│   │   ├── compact_amg.hpp     # CompactAMG (classical AMG, header-only)
│   │   ├── compact_ams.hpp     # CompactAMS (Hiptmair-Xu auxiliary space preconditioner)
│   │   └── complex_compact_ams.hpp # ComplexCompactAMS (fused Re/Im for eddy current)
│   ├── solvers/                  # Iterative solvers
│   │   ├── iterative_solver.hpp  # Iterative solver base class
│   │   ├── cg_solver.hpp         # Conjugate Gradient (CG)
│   │   └── sgs_mrtr_solver.hpp   # SGS-MRTR (split formula)
│   └── ngsolve/                  # NGSolve integration layer
│       ├── sparsesolv_precond.hpp # BaseMatrix wrappers (IC, SGS, BDDC, Solver)
│       └── sparsesolv_python_export.hpp # pybind11 bindings + factory functions
├── ngsolve/
│   └── python_module.cpp         # pybind11 module entry point
├── tests/
│   └── test_sparsesolv.py        # Solver and preconditioner tests
├── docs/                         # Documentation (this folder)
├── CMakeLists.txt                # Build configuration
└── LICENSE
```

## Design Principles

### Header-Only

All C++ code is implemented in `.hpp` header files.
Since all code is inlined at compile time, there are no linking issues and distribution is straightforward.

### Template Design

All algorithm classes are parameterized with `template<typename Scalar>`:

```cpp
template<typename Scalar = double>
class ICPreconditioner : public Preconditioner<Scalar> { ... };
```

`Scalar` is instantiated with either `double` or `std::complex<double>`.
This allows both real and complex problems (e.g., eddy current) to be handled within a single codebase.

### Parallelization Abstraction Layer

`core/parallel.hpp` switches the backend at compile time:

| Build Configuration | Backend | Use Case |
|---|---|---|
| `SPARSESOLV_USE_NGSOLVE_TASKMANAGER` | NGSolve TaskManager | When integrated with NGSolve |
| `_OPENMP` | OpenMP | Standalone |
| (neither) | Serial execution | Debugging |

Main API:

```cpp
sparsesolv::parallel_for(n, [&](index_t i) { ... });
sparsesolv::parallel_reduce(n, init, [&](index_t i) -> T { ... });
sparsesolv::get_num_threads();
```

## Header Dependencies

```
sparsesolv.hpp (main header)
├── core/types.hpp
├── core/constants.hpp
├── core/solver_config.hpp
├── core/sparse_matrix_view.hpp ← core/parallel.hpp
├── core/preconditioner.hpp ← core/sparse_matrix_view.hpp
├── core/abmc_ordering.hpp
├── preconditioners/ic_preconditioner.hpp
│   ← core/preconditioner.hpp, core/level_schedule.hpp,
│      core/abmc_ordering.hpp, core/rcm_ordering.hpp
├── preconditioners/sgs_preconditioner.hpp
│   ← core/preconditioner.hpp
├── solvers/iterative_solver.hpp
│   ← core/preconditioner.hpp, core/sparse_matrix_view.hpp
├── solvers/cg_solver.hpp
│   ← solvers/iterative_solver.hpp
└── solvers/sgs_mrtr_solver.hpp
    ← solvers/iterative_solver.hpp, core/level_schedule.hpp
```

NGSolve integration layer (only when building with NGSolve):

```
ngsolve/sparsesolv_precond.hpp   ← sparsesolv.hpp + NGSolve headers
ngsolve/sparsesolv_python_export.hpp ← sparsesolv_precond.hpp + pybind11
```

## NGSolve Integration Layer

### SparseMatrixView (Zero-Copy Wrapper)

NGSolve's `SparseMatrix<SCAL>` has a CSR-like internal structure.
`SparseSolvPrecondBase::prepare_matrix_view()` converts it to
SparseSolv's `SparseMatrixView<SCAL>` without copying any data.

FreeDofs handling: rows for constrained DOFs are replaced with identity rows (diagonal=1, off-diagonal=0).

```cpp
// sparsesolv_precond.hpp: SparseSolvPrecondBase::prepare_matrix_view()
if (!freedofs_->Test(i)) {
    modified_values_[k] = (j == i) ? SCAL(1) : SCAL(0);  // identity row
} else if (!freedofs_->Test(j)) {
    modified_values_[k] = SCAL(0);  // zero coupling to constrained DOF
}
```

### SparseSolvPrecondBase (BaseMatrix Wrapper)

Base class for all preconditioners. Inherits from NGSolve's `BaseMatrix` and
implements `Mult()` / `MultAdd()` for compatibility with NGSolve's `CGSolver`.

```
SparseSolvPrecondBase<SCAL>  (abstract base)
├── SparseSolvICPreconditioner<SCAL>    → ICPreconditioner<SCAL>
├── SparseSolvSGSPreconditioner<SCAL>   → SGSPreconditioner<SCAL>
└── SparseSolvSolver<SCAL>              → ICCG/SGSMRTR/CG

CompactAMS  (inherits BaseMatrix directly)
  → Real HCurl AMS preconditioner (magnetostatics, supports Update())
  → Python: CompactAMSPreconditionerImpl

ComplexCompactAMS  (inherits BaseMatrix directly)
  → Complex eddy current fused Re/Im AMS preconditioner (supports Update())
  → Python: ComplexCompactAMSPreconditionerImpl
```

Nonlinear solver support via Update():
- `Update()`: Rebuilds the preconditioner with the current matrix (geometric information is retained)
- `Update(new_mat)`: Rebuilds the preconditioner with a new matrix
- Geometric information (G, Pi matrices, transposes) is computed only during initial construction

### Python Factory Functions (Auto-Dispatch)

The factory functions in `sparsesolv_python_export.hpp` automatically determine
the matrix type and create the appropriate template instance:

```cpp
// ICPreconditioner factory
m.def("ICPreconditioner", [](shared_ptr<BaseMatrix> mat, ...) {
    if (mat->IsComplex()) {
        // → SparseSolvICPreconditioner<Complex>
    } else {
        // → SparseSolvICPreconditioner<double>
    }
});
```

### Type Registration (AMS)

CompactAMS / ComplexCompactAMS register concrete types via `py::class_`,
and the factory functions return concrete types, making `Update()` accessible from Python:

```cpp
// Type registration
py::class_<CompactAMS, shared_ptr<CompactAMS>, BaseMatrix>
    (m, "CompactAMSPreconditionerImpl")
    .def("Update", py::overload_cast<>(&CompactAMS::Update))
    .def("Update", py::overload_cast<shared_ptr<SparseMatrix<double>>>(&CompactAMS::Update));

// Factory (returns concrete type)
m.def("CompactAMSPreconditioner", [...] -> shared_ptr<CompactAMS> { ... });
```

For details on the BDDC algorithm, see [algorithms.md](algorithms.md).
It uses NGSolve's built-in BDDC (`a.mat.Inverse(fes.FreeDofs(), inverse="bddc")`).
