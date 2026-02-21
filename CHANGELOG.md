# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2026-02-21

### Added
- BDDC (Balancing Domain Decomposition by Constraints) preconditioner
  - Element-by-element construction from BilinearForm
  - Wirebasket/interface DOF classification via NGSolve CouplingType
  - Coarse solver: SparseCholesky (default), PARDISO, dense LU
  - Mesh-independent convergence (2 CG iterations for all test problems)
- Japanese documentation in `docs/` (architecture, algorithms, API reference, tutorials)
- MPL 2.0 license headers on all source files
- CONTRIBUTING.md

### Changed
- Removed block elimination BDDC approach (replaced by element-by-element)
- Removed unused preconditioners (Identity, Jacobi, ILU)
- Removed MRTRSolver (SGS-MRTR is self-contained)
- Consolidated duplicate `reorder_matrix()` code in IC preconditioner
- Extracted shared `BuildSparseMatrixView()` to eliminate code duplication
- Simplified pybind11 type registrations (factories are the public API)

### Fixed
- DenseMatrix::invert() permutation matrix construction (P^T → P)
  - Caused BDDC failure on HCurl problems with non-trivial pivoting
- Complex inner product: unconjugated dot product for complex-symmetric FEM matrices
  - Fixed eddy current divergence (5000 iterations → 58 iterations)
- SGS-MRTR complex comparison: use `std::real(denom)` for threshold check

## [2.0.0] - 2026-02-19

### Added
- Standalone pybind11 module (`sparsesolv_ngsolve.pyd`) separate from NGSolve
- Factory functions with auto-dispatch (real/complex via `mat.IsComplex()`)
- Auto-shift IC decomposition for semi-definite matrices (curl-curl)
- Diagonal scaling for improved conditioning
- ABMC (Algebraic Block Multi-Color) ordering for parallel triangular solves
- RCM bandwidth reduction (optional, combined with ABMC)
- Complex number support (`std::complex<double>`) throughout
- Divergence detection with best-result recovery
- Residual history recording
- Persistent parallel region for level-scheduled triangular solves

### Changed
- Restructured as header-only C++17 library
- NGSolve integration via `SPARSESOLV_USE_NGSOLVE_TASKMANAGER` compile flag
- Parallelism abstraction: TaskManager / OpenMP / serial dispatch

## [1.0.0] - 2026-02-01

Initial fork from [JP-MARs/SparseSolv](https://github.com/JP-MARs/SparseSolv).

### Features from upstream
- IC(0) preconditioner with shift parameter
- SGS preconditioner
- CG solver
- SGS-MRTR solver with split formula
