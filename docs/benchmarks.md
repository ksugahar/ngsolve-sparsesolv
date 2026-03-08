# SparseSolv Benchmark Results

Benchmark results from `examples/hiruma/bench_hypre_ams.py` and `bench_emd_comparison.py`.

**Environment**: Intel Core i7-9700K (8C/8T, 3.60 GHz), 128 GB DDR4, Windows Server 2022, Intel MKL 2024.2, NGSolve + sparsesolv_ngsolve, 2026-03-08

---

## 1. HYPRE AMS + BiCGStab (渦電流)

Source: `examples/hiruma/bench_hypre_ams.py` -> `results_hypre_ams.json`

Complex eddy-current (50 Hz, mu_r=1000 core, sigma=5.96e7 Cu).
ComplexHypreAMSPreconditioner + BiCGStabSolver。

### Python逐次 vs C++ TaskManager (Re/Im並列)

| Mesh | HCurl DOFs | Python逐次 | C++ TaskManager | 高速化 |
|------|----------:|---:|---:|---:|
| 2.5T | 155k | 4.72s, 26 it | **2.67s, 26 it** | **1.77x** |
| 5.5T | 331k | 10.56s, 26 it | **6.49s, 26 it** | **1.63x** |
| 20.5T | 1.44M | 54.80s, 26 it | **37.17s, 26 it** | **1.47x** |

反復数は全メッシュで26回 — メッシュサイズ非依存の優れたスケーラビリティ。

### なぜ BiCGStab か？

HYPRE AMSは非対称前処理 (hybrid Gauss-Seidelスムーザ) → CG使用不可。
BiCGStabとGMRESを比較した結果、**BiCGStabが圧倒的に優れる** (30kHz、1.44M DOFs):

| ソルバー | 反復数 | 計算時間 | メモリ |
|---------|------:|--------:|------:|
| **BiCGStab** | **26** | **35.5s** | **3.8 GB** |
| GMRES | 528 | 1,112.6s | 15.3 GB |

BiCGStabは固定8ワークベクトルのみ使用 (O(N)メモリ)。
GMRESはKrylov基底をm本蓄積 (O(k*N)メモリ)、高周波数で破綻する。

### EMD前処理との比較 (SA-26-001, 30kHz)

Source: `examples/hiruma/bench_emd_comparison.py` -> `results_emd_comparison.json`

| 手法 | DOFs | 反復数 | 計算時間 | vs EMD最良 |
|------|-----:|------:|---------:|:---------:|
| EMD (IC + GenEO-DDM 24分割) | 3.67M | 1,004 | 550.8s | 1.0x |
| **HYPRE AMS + BiCGStab (推定)** | 3.67M | ~26 | **~85s** | **~6.5x** |
| HYPRE AMS + BiCGStab (実測) | 1.44M | 26 | 48.9s | — |

See [examples/hiruma/RESULT.md](../examples/hiruma/RESULT.md) for detailed results.

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
