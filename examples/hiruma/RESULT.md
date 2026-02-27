# Hiruma Eddy Current Analysis Results

## Problem

Frequency-domain eddy current analysis (A-Phi formulation, f = 50 Hz) of a
copper coil around a ferromagnetic iron core.

| Region | mu_r | sigma [S/m] | Description |
|--------|------|-------------|-------------|
| core | 1000 | 0 | Ferromagnetic iron core |
| cond | 1 | 5.96e7 | Copper conductor (coil) |
| air | 1 | 0 | Surrounding air |

## Formulation

**A-Phi product space** (HCurl x H1):

```
curl(nu * curl A) + j*w*sigma*(A + grad(phi)) = 0   in cond
curl(nu * curl A) + eps*nu*A = 0                      elsewhere
A x n = 0 on dirichlet;  phi = 1 on gamma_in;  phi = 0 on gamma_out
```

- `HCurl(nograds=True, complex=True)` for vector potential A
- `H1(definedon="cond", complex=True)` for scalar potential Phi
- Stabilization `1e-6 * nu * A * N * dx` for BDDC conditioning
- NGSolve BDDC preconditioner + CG (conjugate=False for complex-symmetric)

## Mesh-Size Scaling (BDDC)

| Mesh | nv | ne | DOFs | BDDC iters | setup [s] | solve [s] |
|------|------:|--------:|--------:|:---:|-----:|-----:|
| mesh1_2.5T | 23,731 | 130,460 | 179,258 | **3** | 10.1 | 1.3 |
| mesh1_3.5T | 29,901 | 166,198 | 227,296 | **3** | 15.1 | 1.9 |
| mesh1_4.5T | 36,473 | 203,403 | 277,602 | **3** | 21.4 | 2.4 |
| mesh1_5.5T | 49,643 | 280,278 | 381,238 | **3** | 47.1 | 4.2 |
| mesh1_20.5T | 211,337 | 1,227,241 | 1,652,439 | OOM | - | - |
| mesh1_21.5T_HF | 691,541 | 2,597,034 | 7,760,242 | OOM | - | - |

**BDDC iteration count is mesh-size independent (3 iterations for all sizes).**

Large meshes (20.5T, 21.5T_HF) fail with out-of-memory during complex product
space assembly.

## ||B||^2 by Region

| Mesh | ||B||^2 total | core (%) | cond (%) | air (%) |
|------|-------------|----------|----------|---------|
| 2.5T | 1.634e-01 | 5.50e-02 (33.7%) | 4.97e-03 (3.0%) | 1.03e-01 (63.3%) |
| 3.5T | 1.533e-01 | 5.63e-02 (36.7%) | 4.32e-03 (2.8%) | 9.27e-02 (60.4%) |
| 4.5T | 1.378e-01 | 5.23e-02 (37.9%) | 3.80e-03 (2.8%) | 8.17e-02 (59.3%) |
| 5.5T | 1.229e-01 | 4.73e-02 (38.5%) | 3.33e-03 (2.7%) | 7.23e-02 (58.8%) |

||B||^2 decreases ~25% from coarsest to finest mesh. This is NOT purely
h-refinement convergence — the meshes differ in geometry discretization:

| Mesh | vol_core | vol_cond | vol_air | ne_core |
|------|----------|----------|---------|---------|
| 2.5T | 9.509e-05 | 1.121e-05 | 2.025e-01 | 4,076 |
| 3.5T | 9.509e-05 | 1.472e-05 | 2.025e-01 | 4,083 |
| 4.5T | 9.509e-05 | 1.823e-05 | 2.025e-01 | 4,077 |
| 5.5T | 9.509e-05 | 2.174e-05 | 2.108e-01 | 4,078 |

- Core elements (~4,077) and core volume are identical across meshes
- Conductor volume nearly doubles (1.12e-5 → 2.17e-5)
- Air volume increases slightly for 5.5T mesh (+4%)
- These are NOT uniform h-refinements of the same geometry

## BDDC vs ICCG Comparison (mesh1_2.5T)

| Solver | Iterations | Time [s] | ||B||^2 | converged |
|--------|--------:|------:|-----------|:---------:|
| A-Phi BDDC | 3 | 13.3 | 1.634e-01 | yes |
| Two-step ICCG | 5000 | 97.0 | 1.632e-01 | no |

- ICCG uses a two-step approach: DC current distribution (Laplace) + HCurl A-only
- ICCG does not require the 1e-6 stabilization (auto_shift handles semi-definiteness)
- Despite non-convergence, ICCG ||B||^2 matches BDDC within 0.1% (best-iterate tracking)
- **BDDC is essential for HCurl problems with high-contrast materials (mur=1000)**

## VTK Output

VTK files (`eddy_current_mesh1_*.vtu`) contain:

| Field | Description |
|-------|-------------|
| B_re, B_im | Magnetic flux density (real/imaginary) |
| B_abs | |B| magnitude |
| J_re, J_im | Current density in conductor |
| Q_joule | Joule heat density [W/m^3] |
| MaterialID | 1=core, 2=cond, 3=air |

Values are unnormalized (phi=1 on gamma_in). Scale by (I_target/I_computed)^2
for physical quantities.
