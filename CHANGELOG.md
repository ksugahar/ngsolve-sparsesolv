# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.7.0] - 2026-03-10

### Removed
- BiCGStab C++ solver (`bicgstab_solver.hpp`, `SparseSolvSolver(method="BiCGStab")`)
  - Superseded by COCR (complex-symmetric) and GMRES (non-symmetric)

### Changed
- Translated all documentation (README + docs/) to Japanese
- Consolidated docs from 10 files to 6 files:
  - Merged `docs/benchmarks.md` into `docs/compact_ams_cocr.md`
  - Merged `docs/architecture.md` into `docs/development.md`
  - Merged `docs/abmc_implementation_details.md` into `docs/algorithms.md`
  - Removed `docs/README.md` (table of contents only)

## [2.6.0] - 2026-03-09

### Added
- **Compact AMS Preconditioner** — Header-only C++ implementation of auxiliary space preconditioning for HCurl
  - `CompactAMSPreconditioner`: For real HCurl systems (magnetostatic curl-curl + mass)
  - `ComplexCompactAMSPreconditioner`: For complex eddy current systems, halving bandwidth cost with fused Re/Im SpMV
  - `CompactAMG`: Classical AMG with PMIS coarsening + l1-Jacobi + DualMult
  - No external library dependencies, parallelized with NGSolve TaskManager
  - **`Update()` method** — Supports nonlinear solvers (Newton iterations). Retains geometric information (G, Pi matrices, transposes) while rebuilding only matrix-dependent components (Galerkin projections, AMG hierarchy, l1 norms)
  - Python type registration: `CompactAMSPreconditionerImpl`, `ComplexCompactAMSPreconditionerImpl`
- **COCR Solver (C++)** — Krylov solver for complex symmetric systems (A^T=A)
  - `COCRSolver(mat, pre)`: Native C++ implementation, compatible with NGSolve BaseMatrix
  - `SparseSolvSolver(method="COCR")`: Also available via SparseSolvSolver dispatch
  - Header-only: `include/sparsesolv/solvers/cocr_solver.hpp`
  - Uses unconjugated inner product (x^T y), optimal for complex symmetric FEM matrices in eddy current problems
- **ABMC vs AMS Benchmark** (`bench_ams_vs_abmc.py`)

### Removed
- All HYPRE AMS related code (`HypreAMSPreconditioner`, `ComplexHypreAMSPreconditioner`,
  `HypreBoomerAMGPreconditioner`, `has_hypre()`, `external/hypre/`)
- BiCGStab solver (`bicgstab_solver.py`)
- Custom AMS preconditioner (`TaskManagerAMSPreconditioner`, etc.)
- EMD benchmark (`bench_emd_comparison.py`)

### Changed
- Changed SGS preconditioner sparse matrix reference to non-owning pointer (fixes double-free bug)
- Reorganized all documentation around Compact AMS + COCR

## [2.3.0] - 2026-02-28

### Added
- Hiruma eddy current problem mesh examples (6 meshes, Gmsh v2 format, Git LFS)
- `examples/hiruma/eddy_current.py` — Eddy current analysis with A-Phi formulation

### Improved
- Kernel fusion in CG iterations — approximately 20% reduction in memory traffic
  - Fused SpMV + dot(p, Ap) into a single pass (eliminates re-reading p[] and Ap[])
  - Fused AXPY + residual norm computation into a single pass (eliminates re-reading r[])
  - Fused preconditioner application + dot(r, z) into a single pass (`apply_fused_dot`)
  - Reduced kernel launches per iteration from 7 to 4
- Converted `apply_in_reordered_space` (ABMC space CG path) to a persistent parallel region
  - Before: 2*nc parallel_for dispatches (nc = number of colors)
  - After: 1 dispatch + 2*nc SpinBarrier calls (equivalent to `apply_abmc`)
- Added auto_shift support for ABMC parallel IC factorization (restart via atomic flags)
  - Before: ABMC parallel path was unusable when auto_shift was enabled, falling back to sequential IC factorization
  - After: Detects breakdown during parallel factorization, increases shift, and restarts the entire process for full parallelization
- Hiruma HCurl p=1 eddy current problem parallel scaling improved from 1.5x to 3.14x (8 cores)
- Exponential backoff for auto_shift (`increment *= 2`) — significantly reduces restart count
- Changed default `shift_increment` from 0.01 to 0.05
- Optimized shift values in notebooks (NB02, NB04) from 1.5 to 1.15 (approximately 7% improvement in iteration count)

## [2.2.0] - 2026-02-25

### Added
- ABMC parallel IC factorization (parallel_for per color)
- Persistent parallel region for level-scheduled triangular solve (SpinBarrier)
- Python API exposure of ABMC properties (`use_abmc`, `abmc_block_size`, etc.)

### Changed
- Excluded ngsolve headers from header-only installation

## [2.1.0] - 2026-02-21

### Added
- Japanese documentation in `docs/` (architecture, algorithms, API reference, tutorials)
- MPL 2.0 license headers in all source files
- CONTRIBUTING.md

### Changed
- Removed unused preconditioner matrices (Identity, Jacobi, ILU)
- Removed MRTRSolver (SGS-MRTR is self-contained)
- Consolidated duplicate `reorder_matrix()` code in IC preconditioner matrix
- Extracted shared `BuildSparseMatrixView()` to eliminate code duplication
- Simplified pybind11 type registration (factory is the public API)

### Fixed
- DenseMatrix::invert() permutation matrix construction (P^T to P)
- Complex inner product: unconjugated inner product for complex symmetric FEM matrices
  - Fixed eddy current divergence (5000 iterations reduced to 58 iterations)
- SGS-MRTR complex comparison: use `std::real(denom)` for threshold check

## [2.0.0] - 2026-02-19

### Added
- Standalone pybind11 module independent of NGSolve (`sparsesolv_ngsolve.pyd`)
- Factory function with automatic dispatch (automatic real/complex detection via `mat.IsComplex()`)
- Automatic shift IC factorization for semi-definite matrices (curl-curl)
- Diagonal scaling for condition number improvement
- ABMC (Algebraic Block Multi-Color) ordering (for parallel triangular solve)
- RCM bandwidth reduction (optional, combinable with ABMC)
- Full complex number support (`std::complex<double>`)
- Divergence detection with best result recovery
- Residual history recording
- Persistent parallel region for level-scheduled triangular solve

### Changed
- Restructured as a header-only C++17 library
- NGSolve integration via `SPARSESOLV_USE_NGSOLVE_TASKMANAGER` compile flag
- Parallelization abstraction: TaskManager / OpenMP / serial dispatch

## [1.0.0] - 2026-02-01

Initial fork from [JP-MARs/SparseSolv](https://github.com/JP-MARs/SparseSolv).

### Features from upstream
- IC(0) preconditioner matrix with shift parameter
- SGS preconditioner matrix
- CG solver
- SGS-MRTR solver with splitting formula
