# Hiruma Eddy Current Analysis

Frequency-domain eddy current analysis (A formulation, f = 50 Hz) of a copper coil
around a ferromagnetic iron core. Mesh files provided by Hiruma-sensei.

## Mesh Files

| File | Nodes | Elements | HCurl DOFs | Description |
|------|------:|--------:|-----------:|-------------|
| `mesh1_2.5T.msh` | 23,731 | 130,460 | 155k | Coarsest |
| `mesh1_3.5T.msh` | 29,901 | 168,792 | 197k | |
| `mesh1_4.5T.msh` | 36,473 | 205,911 | 241k | |
| `mesh1_5.5T.msh` | 49,643 | 283,628 | 331k | |
| `mesh1_20.5T.msh` | 211,337 | 1,227,241 | 1.44M | Large |
| `mesh1_21.5T_HF.msh` | 691,541 | 2,605,376 | ~7.8M | Very large |

Original Gmsh v1 files are in `original_msh1/` (not tracked by git).

## Region and Material Definitions

| Region | Physical tag | mu_r | sigma [S/m] | Description |
|--------|:-----------:|-----:|------------:|-------------|
| core | 1 | 1000 | 0 | Ferromagnetic iron core |
| cond | 2 | 1 | 5.96e7 | Copper conductor (coil) |
| air | 3 | 1 | 0 | Surrounding air |
| dirichlet | 4 | - | - | Outer boundary (A=0) |
| gamma_in | 5 | - | - | Coil inlet (phi=1) |
| gamma_out | 6 | - | - | Coil outlet (phi=0) |

## Usage

```python
from netgen.read_gmsh import ReadGmsh
from ngsolve import *

m = ReadGmsh("mesh1_2.5T")
mesh = Mesh(m)
```

## Scripts

### Core Files

| Script | Description |
|--------|-------------|
| `eddy_current.py` | Original BDDC solver (MKL PARDISO coarse) |
| `bench_hypre_ams.py` | Benchmark: HYPRE AMS Python wrapper vs C++ TaskManager |

### Archived (in `_archive/`)

Development benchmarks and intermediate experiments. Kept for reference.

## HYPRE AMS Preconditioner

### Theory

For complex eddy-current systems `A = K + j*omega*sigma*M`:

1. Build a real SPD auxiliary matrix:
   `A_real = K + eps*M + |omega|*sigma*M_cond`
2. Apply HYPRE AMS to Re and Im parts independently:
   `y = HYPRE_AMS(Re(f)) + j * HYPRE_AMS(Im(f))`

`ComplexHypreAMSPreconditioner` creates TWO independent HYPRE AMS instances
and runs them concurrently via NGSolve TaskManager for ~1.5x speedup.

HYPRE AMS is non-symmetric (relax_type=3, hybrid GS) -> **GMResSolver required**.

### Usage

```python
import sparsesolv_ngsolve as ssn
from ngsolve.krylovspace import GMResSolver

pre = ssn.ComplexHypreAMSPreconditioner(
    a_real_mat=a_real.mat, grad_mat=G_mat,
    freedofs=fes_real.FreeDofs(),
    coord_x=cx, coord_y=cy, coord_z=cz,
    ndof_complex=fes.ndof, cycle_type=1, print_level=0)

with TaskManager():
    inv = GMResSolver(mat=a.mat, pre=pre, maxiter=500, tol=1e-8)
    gfu.vec.data = inv * f.vec
```

### Benchmark Results (GMRES, tol=1e-8)

| Mesh | DOFs | Python (sequential) | C++ TaskManager | Speedup |
|------|-----:|---:|---:|---:|
| 2.5T | 155k | 5.40s, 50 it | **3.43s, 50 it** | **1.57x** |
| 5.5T | 331k | 14.98s, 59 it | **10.19s, 59 it** | **1.47x** |
| 20.5T | 1.44M | 103.09s, 75 it | **69.16s, 75 it** | **1.49x** |

Iteration counts are identical - same math, only parallelism differs.

Run: `python bench_hypre_ams.py`
