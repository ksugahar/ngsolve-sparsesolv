# ngsolve-sparsesolv Documentation

Header-only iterative solver library for NGSolve.
Provides IC factorization, ABMC parallelization, Compact AMS preconditioning, and COCR/CG solvers.

## Contents

| Document | Description |
|---|---|
| [compact_ams_cocr.md](compact_ams_cocr.md) | **Compact AMS + COCR** -- Auxiliary space preconditioning for eddy current (start here for AMS) |
| [tutorials.md](tutorials.md) | Practical tutorials with copy-pastable examples |
| [api_reference.md](api_reference.md) | Python API reference |
| [algorithms.md](algorithms.md) | Algorithm descriptions (BDDC, IC, SGS-MRTR, CG, ABMC, Compact AMS) |
| [architecture.md](architecture.md) | Source code structure and architecture |
| [benchmarks.md](benchmarks.md) | Benchmark results |
| [development.md](development.md) | Build, testing, and developer information |

## Overview

SparseSolv is an iterative solver library for large sparse linear systems arising from finite element methods (FEM).
It is a fork of [JP-MARs/SparseSolv](https://github.com/JP-MARs/SparseSolv), specialized for integration with NGSolve.

### Key Features

- **Compact AMS preconditioning**: Auxiliary space preconditioning for HCurl eddy current problems (header-only, no external dependency). 60x fewer iterations than ICCG
- **IC factorization (Incomplete Cholesky)**: Auto-shift for semi-definite curl-curl matrices
- **ABMC ordering**: Parallel triangular solve [Iwashita et al. 2012]
- **COCR**: Short-recurrence Krylov solver for complex symmetric systems (A^T=A)
- **CG**: Preconditioned conjugate gradient with complex symmetric (unconjugated inner product) support
- **SGS-MRTR**: Iterative solver with built-in diagonal scaling (DAD transformation)

### Solver Selection Guide

| Problem | Finite Element Space | Recommended Solver |
|---------|---------------------|-------------------|
| Eddy current (complex, large-scale) | HCurl (complex, p=1) | **Compact AMS + COCR** |
| Magnetostatics (real, large-scale) | HCurl (real, p=1) | **Compact AMS + CG** |
| Curl-curl (real) | HCurl (nograds=True) | Shifted-ICCG |
| Poisson equation | H1 | ICCG |
| Eddy current (complex, small-medium) | HCurl (complex) | ICCG or COCR |
