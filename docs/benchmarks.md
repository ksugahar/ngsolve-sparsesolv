# SparseSolv Benchmark Results

Benchmark results from `examples/hiruma/bench_hypre_ams.py`.

**Environment**: Windows Server 2022, 8 cores, Intel MKL, 2026-03-07

---

## 1. HYPRE AMS (渦電流)

Source: `examples/hiruma/bench_hypre_ams.py` -> `results_hypre_ams.json`

Complex eddy-current (50 Hz, mu_r=1000 core, sigma=5.96e7 Cu).
ComplexHypreAMSPreconditioner + GMResSolver (HYPRE AMSは非対称前処理→GMRES必須)。

### Python逐次 vs C++ TaskManager (Re/Im並列)

| Mesh | HCurl DOFs | Python逐次 | C++ TaskManager | 高速化 |
|------|----------:|---:|---:|---:|
| 2.5T | 155k | 5.40s, 50 it | **3.43s, 50 it** | **1.57x** |
| 5.5T | 331k | 14.98s, 59 it | **10.19s, 59 it** | **1.47x** |
| 20.5T | 1.44M | 103.09s, 75 it | **69.16s, 75 it** | **1.49x** |

反復数は完全に同一 — 数学的に同じ前処理、TaskManager並列化のみの差。

### なぜ hybrid GS + GMRES が最適か

HYPRE AMSのHCurlスムーザには複数の選択肢がある:

| スムーザ | 対称性 | Krylov | 反復数 | 1反復コスト |
|---------|:---:|--------|:---:|:---:|
| hybrid GS (relax_type=3) | 非対称 | GMRES | **少** | 中 |
| l1-Jacobi (relax_type=2) | 対称 | CG | 多 | 低 |
| symmetric GS (relax_type=6) | 対称 | CG | 中 | 高 (逐次) |
| Chebyshev (relax_type=16) | 対称 | CG | 中 | 中 |

対称スムーザならCG互換だが、GMRESのKrylov基底メモリ増を差し引いても
**hybrid GS + GMRESが総合的に最速**。理由:
- hybrid GSはGS (プロセッサ内) + Jacobi (プロセッサ間) のハイブリッドで、
  GSの高い平滑化効果とJacobiの並列性を両立
- 反復数の削減がGMRESのメモリコストを上回る
- 複素渦電流ではRe/Im分割により実質的に実数AMSを2回適用するため、
  前処理の品質 (= 反復数の少なさ) が全体性能を支配

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
