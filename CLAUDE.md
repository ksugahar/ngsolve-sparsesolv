# ngsolve-sparsesolv Development Policy

## Architecture

This repository provides a monolithic PyPI package `ngsolve-sparsesolv` that bundles:
1. **NGSolve** -- Official NGSolve (tracked via NGSOLVE_VERSION)
2. **Netgen** -- Official Netgen with SetGeomInfo patch (until PR#86 is merged upstream)
3. **SparseSolv** -- Compact AMS, COCR, IC, SGS solvers (this repo's original code)

### No Forks

This project does NOT maintain forks of NGSolve or Netgen.
- Official NGSolve is cloned at build time from https://github.com/NGSolve/ngsolve
- `NGSOLVE_VERSION` file pins the upstream version tag
- `patches/` directory contains minimal patches applied at build time
- When upstream adopts our patches (e.g., PR#86 for SetGeomInfo), the patch file is deleted

### Upstream Sync

To update to a new NGSolve release:
1. Update `NGSOLVE_VERSION` to the new tag (e.g., `v6.2.2602`)
2. Verify patches apply cleanly: `python scripts/build_monolithic.py`
3. If a patch fails, update the patch to match upstream changes
4. Run tests, tag, push

### SparseSolv Independent Build

SparseSolv can be built independently against any installed NGSolve:
```bash
pip install ngsolve
pip install .  # builds sparsesolv_ngsolve module only
```

For the monolithic wheel (netgen + ngsolve + sparsesolv):
```bash
python scripts/build_monolithic.py
pip install dist/ngsolve_sparsesolv-*.whl
```

## Build

- **Compiler**: MSVC (Visual Studio 2022)
- **BLAS/LAPACK**: Intel MKL (OpenBLAS NOT supported)
- **Platform**: Windows only (current)
- **Python**: 3.10+
- **Parallelization**: NGSolve TaskManager (not raw OpenMP)

### MKL DLL Policy

MKL DLLs are NOT bundled in the wheel. They are installed via pip dependency:
```
mkl >= 2024.2.0
intel-cmplr-lib-rt
```

## PyPI

- **Package name**: `ngsolve-sparsesolv`
- **Contents**: netgen + ngsolve + sparsesolv_ngsolve (single wheel)
- **Publishing**: Automated via GitHub Actions + OIDC Trusted Publishers
- **Trigger**: Push a version tag `v*`

### Deprecated Package

The old `sparsesolv-ngsolve` PyPI package is deprecated.
Users should migrate to `pip install ngsolve-sparsesolv`.

## Key Files

| File | Purpose |
|------|---------|
| `NGSOLVE_VERSION` | Pinned upstream NGSolve version tag |
| `patches/*.patch` | Patches applied to official ngsolve/netgen at build time |
| `scripts/build_monolithic.py` | Monolithic wheel build orchestrator |
| `include/sparsesolv/` | SparseSolv header-only library (C++17) |
| `ngsolve/python_module.cpp` | SparseSolv pybind11 entry point |
| `pyproject.toml` | Standalone SparseSolv build config |
| `.github/workflows/build-wheels.yml` | Monolithic wheel CI + PyPI publish |
| `.github/workflows/ci.yml` | Standalone SparseSolv CI test |

## Solver Selection Guide

| Problem | FEM Space | Recommended Solver |
|---------|-----------|-------------------|
| Poisson | H1 | ICCG |
| Curl-curl (real) | HCurl | Shifted-ICCG |
| Magnetostatic (large) | HCurl real | Compact AMS + CG |
| Eddy current (large) | HCurl complex | Compact AMS + COCR |

## SetGeomInfo Patch

The SetGeomInfo API adds `Element2d.SetGeomInfo(vertex_index, u, v)` to Netgen,
enabling high-order mesh curving for externally imported meshes (from Cubit, GMSH).

- **PR**: https://github.com/NGSolve/netgen/pull/232
- **Patch**: `patches/netgen-setgeominfo.patch`
- **Status**: Pending upstream adoption
- **When merged**: Delete the patch file, update this section

## Console Encoding

NEVER use Unicode mathematical symbols in print statements.
Use ASCII equivalents: `^2` not square, `->` not arrow, `<=` not le.
Windows console defaults to cp932 in Japanese environments.
