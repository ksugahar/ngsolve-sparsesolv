# ngsolve-sparsesolv Development Policy

## Architecture

This repository provides `ngsolve-sparsesolv`, a **standalone** PyPI package
that adds Compact AMS/COCR solvers to the official NGSolve.

**No forks, no monolithic bundling.** Users install official ngsolve from PyPI
and ngsolve-sparsesolv on top of it:

```bash
pip install ngsolve-sparsesolv   # pulls ngsolve>=6.2.2601 automatically
```

### Why Standalone (Not Monolithic)

As of v6.2.2601, official PyPI ngsolve already includes:
- **Intel MKL** (`Requires-Dist: mkl`, `USE_MKL=ON`)
- **PARDISO** direct solver (via MKL, `USE_PARDISO` in compile definitions)
- **Periodic BC fix** (Identify() survives Glue(), fixed in 6.2.2601)

The only remaining add-on value of this package:
1. **Compact AMS** -- Hiptmair-Xu auxiliary-space Maxwell preconditioner (HYPRE-free)
2. **COCR** -- Complex symmetric Krylov solver (Sogabe-Zhang 2007)
3. **ICCG** -- IC(0) with auto-shift for curl-curl + ABMC parallel triangular solve
4. **SGS-MRTR** -- Symmetric Gauss-Seidel MRTR

### SetGeomInfo Patch (Separate)

The SetGeomInfo API patch (netgen PR#232) is **no longer part of this package**.
It is maintained separately in the ksugahar/netgen fork until upstream adoption.
- PR: https://github.com/NGSolve/netgen/pull/232
- For users who need SetGeomInfo: install the patched netgen from the fork

## Build

- **Compiler**: MSVC (Visual Studio 2022)
- **BLAS/LAPACK**: Intel MKL (via pip `mkl-devel` at build time, `mkl` at runtime)
- **Platform**: Windows only (current)
- **Python**: 3.10+
- **Parallelization**: NGSolve TaskManager (not raw OpenMP)

### Build from source

```bash
pip install ngsolve mkl-devel pybind11 scikit-build-core
pip install .
```

### MKL DLL Policy

MKL DLLs are NOT bundled in the wheel. They are installed via pip dependency:
```
mkl >= 2024.2.0
intel-cmplr-lib-rt
```

## PyPI

- **Package name**: `ngsolve-sparsesolv`
- **Dependencies**: `ngsolve>=6.2.2601`, `mkl>=2024.2.0`, `intel-cmplr-lib-rt`, `numpy`
- **Publishing**: Automated via GitHub Actions + OIDC Trusted Publishers
- **Trigger**: Push a version tag `v*`

## Key Files

| File | Purpose |
|------|---------|
| `include/sparsesolv/` | SparseSolv header-only library (C++17) |
| `ngsolve/python_module.cpp` | SparseSolv pybind11 entry point |
| `pyproject.toml` | Build config (scikit-build-core) |
| `.github/workflows/build-wheels.yml` | Wheel build CI + PyPI publish |
| `.github/workflows/ci.yml` | Build + test CI |

## Solver Selection Guide

| Problem | FEM Space | Recommended Solver |
|---------|-----------|-------------------|
| Poisson | H1 | ICCG |
| Curl-curl (real) | HCurl (`nograds=True`) | Shifted-ICCG |
| Magnetostatic (large) | HCurl real | Compact AMS + CG |
| Eddy current (large) | HCurl complex | Compact AMS + COCR |

## Console Encoding

NEVER use Unicode mathematical symbols in print statements.
Use ASCII equivalents: `^2` not square, `->` not arrow, `<=` not le.
Windows console defaults to cp932 in Japanese environments.
