# SparseSolv Benchmark Results

Benchmark results from `examples/hiruma/bench_compact_ams.py` and `bench_ams_vs_abmc.py`.

**Environment**: Intel Xeon (8C/8T), 128 GB DDR4, Windows Server 2022, MSVC 2022, Intel MKL 2024.2, NGSolve + sparsesolv_ngsolve, 2026-03-09

---

## 1. Compact AMS + COCR (Eddy Current)

Source: `examples/hiruma/bench_compact_ams.py` -> `results_compact_ams.json`

Complex eddy-current (30 kHz, mu_r=1000 core, sigma=5.96e7 Cu).
ComplexCompactAMSPreconditioner + COCRSolver, tol=1e-10.

| Mesh | Elements | HCurl DOFs | H1 DOFs | Iters | Setup [s] | Solve [s] | Total [s] | ms/iter | Memory [MB] |
|------|----------|-----------|---------|-------|-----------|-----------|-----------|---------|-------------|
| mesh1_2.5T | 130,460 | 155,527 | 23,731 | 144 | 0.7 | 3.7 | 4.5 | 25.8 | 368 |
| mesh1_3.5T | 166,198 | 197,395 | 29,901 | 168 | 0.9 | 6.3 | 7.3 | 37.7 | 460 |
| mesh1_4.5T | 203,403 | 241,129 | 36,473 | 210 | 1.3 | 9.2 | 10.5 | 43.8 | 543 |
| mesh1_5.5T | 280,278 | 331,595 | 49,643 | 249 | 1.9 | 14.3 | 16.2 | 57.3 | 725 |
| mesh1_20.5T | 1,227,241 | 1,441,102 | 211,337 | 499 | 24.9 | 197.7 | 222.6 | 396.2 | 2,933 |

All cases converged (`true ||b-Ax||/||b|| < 2e-10`).

---

## 2. Comparison with IC Preconditioner (ABMC-ICCG)

Source: `examples/hiruma/bench_ams_vs_abmc.py` -> `results_ams_vs_abmc.json`

IC (Incomplete Cholesky) preconditioner with ABMC (Algebraic Block Multi-Color)
parallel ordering. 30 kHz, tol=1e-10, maxiter=20,000.

| Mesh | DOFs | Method | Iters | Total [s] | Status |
|------|------|--------|-------|-----------|--------|
| mesh1_3.5T | 197k | Compact AMS + COCR | 168 | 7.2 | converged |
| mesh1_3.5T | 197k | ABMC-ICCG | 17,178 | 438.4 | not converged (res=2.8e-10) |

IC preconditioner cannot handle the curl-curl null space inherent in HCurl discretizations.
AMS resolves this with gradient and nodal auxiliary space corrections (Hiptmair-Xu 2007).

---

## 3. EMD Preconditioner Reference (Hiruma SA-26-001)

EMD (Edge-based Magnetic field Decomposition) paper results
(Hiruma, SA-26-001, 3,670,328 DOFs, 30 kHz):

| Method | Iters | Time [s] | ms/iter |
|--------|-------|----------|---------|
| IC only | 15,838 | 5964.8 | 376.7 |
| EMD (IC + AMG V-cycle) | 4,069 | 1716.8 | 422.0 |
| EMD (IC + AMG W-cycle) | 2,935 | 1552.9 | 529.1 |
| EMD (IC + GenEO-DDM, 24 domains) | 1,004 | 550.8 | 548.6 |

Our result at 1.44M DOFs (mesh1_20.5T): 499 iter, 222.6s on 8 CPU cores.
Direct DOF-matched comparison is not possible (1.44M vs 3.67M DOFs).

---

## 4. AMS vs ICCG Scaling (Iron Core Eddy Current)

Comparison of Compact AMS + COCR vs ICCG on an iron core eddy current problem
with material contrast (mu_r=1000 iron core, sigma=1e6 S/m, air gap, 30 kHz).

This benchmark demonstrates the key advantage of AMS: **mesh-size independent iteration count**.

| Mesh | HCurl DOFs | Method | Iters | Solve [s] | Notes |
|------|-----------|--------|-------|-----------|-------|
| Small | 2,728 | ICCG | 97 | 0.03 | |
| Small | 2,728 | AMS + COCR | 46 | 0.11 | 1.5x fewer iters |
| Medium | 6,382 | ICCG | 147 | 0.09 | |
| Medium | 6,382 | AMS + COCR | 52 | 0.13 | 2.8x fewer iters |
| Large | 19,357 | ICCG | 234 | 0.59 | |
| Large | 19,357 | AMS + COCR | 52 | 0.23 | **4.5x fewer iters, 2.6x faster** |
**Key observations**:

1. **AMS iteration count is stable** (46 → 52 → 52) regardless of mesh size
2. **ICCG iteration count grows** (97 → 147 → 234) as mesh is refined — O(h^{-1}) scaling
3. **AMS wall time advantage increases with problem size** — at 19K DOFs, AMS is already 2.6x faster
4. For larger problems (>100K DOFs), the ICCG iteration growth makes AMS essential (see Section 2: 17,178 iters at 197K DOFs)

Setup: `maxh` = 0.08/0.06/0.04, `order=1`, `nograds=True`, `tol=1e-8`, `maxiter=10000`.

---

## JSON Output Policy

All benchmark scripts output results in JSON format with required fields:

| Field | Description |
|-------|-------------|
| `peak_memory_mb` | Peak working set (Windows) / RSS (Linux) via psutil |
| `t_setup` | Preconditioner setup time [s] |
| `t_solve` | Linear solver execution time [s] |
| `iterations` | Iteration count |
| `converged` | Convergence flag |

JSON files are saved alongside scripts as `results_{benchmark_name}.json`.
