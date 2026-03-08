# SparseSolv

[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

[NGSolve](https://ngsolve.org/) 有限要素解析向けのヘッダオンリー C++17 反復法ソルバーライブラリ。

**主な成果**: HCurl有限要素の複素渦電流問題において、[HYPRE](https://github.com/hypre-space/hypre) AMS + BiCGStabの組み合わせが、EMD前処理 (比留間, SA-26-001) 比で推定6.5倍高速であることをベンチマークで実証した。1.44M DOFsで26反復・47秒収束 (30kHz)。`ComplexHypreAMSPreconditioner` によるRe/Im TaskManager並列で逐次比1.5倍の追加高速化を達成。

- **HYPRE AMS前処理 (メイン)** — [HYPRE](https://github.com/hypre-space/hypre) のAMS (Auxiliary-space Maxwell Solver) をNGSolve `BaseMatrix` にラップ。複素渦電流向けRe/Im 2インスタンス並列 (`ComplexHypreAMSPreconditioner`)。BiCGStab外部ソルバーと組み合わせて使用。
- **ABMC並列化ICCG** — 代数的ブロックマルチカラー (ABMC) 順序付けによる並列不完全コレスキーCG。メモリ効率に優れ、中規模問題に有効。

`double` と `std::complex<double>` の両方をサポートし、NGSolveのソースツリーとは独立したpybind11拡張モジュール (`sparsesolv_ngsolve`) として提供する。

[JP-MARs/SparseSolv](https://github.com/JP-MARs/SparseSolv) のfork。

## なぜSparseSolvか？

NGSolveには直接法ソルバーと組込みBDDCが搭載されている。SparseSolvはNGSolveが標準では提供しない機能を追加する:

1. **HYPRE AMS + BiCGStabの性能実証** — 複素渦電流 (HCurl p=1) の大規模FEM問題で、HYPRE AMSのhybrid Gauss-Seidelスムーザ + BiCGStabの組み合わせが、EMD前処理 (GenEO-DDM 24分割) 比で推定6.5倍高速であることを実証した。BiCGStabは固定8ワークベクトルのみ使用し、GMRESのKrylov基底蓄積によるメモリ爆発 (O(k*N)) を回避する。`ComplexHypreAMSPreconditioner` はRe/Im 2インスタンスをTaskManagerで並列処理し、Python逐次ラッパーに対して約1.5倍の追加高速化を達成する。

2. **電磁界 (HCurl) 問題への堅牢性** — curl-curl FEM行列は半正定値であり、標準のIC分解は破綻する。SparseSolvのauto-shift ICは破綻を検出し自動調整する。渦電流問題では導電率項 sigma|u|^2 が自然に正則化として機能するため、curl-curl 単独の場合に比べてICCGでも安定する。純粋なcurl-curl問題では小さな正則化項 `sigma*u*v*dx` (sigma = 1e-6) を加えることでICCGの安定性を確保できる。

3. **ABMCマルチカラー順序付けによる並列ICCG** — IC前処理の三角解法は本質的に逐次的である。ABMC順序付けがこのボトルネックを解消し、並列前進・後退代入を実現する。メモリ効率に優れ (CG: ~5ベクトル)、中規模問題に有効。

## ソルバー選択ガイド

| 問題 | 有限要素空間 | 推奨手法 | 理由 |
|------|------------|---------|------|
| 渦電流 (複素数、大規模) | HCurl (complex, p=1) | **HYPRE AMS+BiCGStab** | 1.44M DOFsで26反復収束、EMD比6.5倍高速 |
| Curl-curl (実数) | HCurl (`nograds=True`) | **Shifted-ICCG** or **HYPRE AMS** | auto-shift ICで半正定値対応 |
| Poisson | H1 | **ICCG** | メモリ効率に優れ高速 |
| 渦電流 (複素数、小中規模) | HCurl (complex) | **ICCG** (`conjugate=False`) or **COCR** | 複素対称行列対応 |
| 小規模 (< 1K DOFs) | 任意 | 直接法 | 反復法のオーバーヘッド大 |

チュートリアルと計算時間の比較は [tutorials](docs/tutorials.md) を参照。

## 性能

### HYPRE AMS + BiCGStab (複素渦電流) — メイン結果

HCurl p=1の大規模複素渦電流問題 (A定式化) に対するベンチマーク。

#### なぜ BiCGStab か？

HYPRE AMSは非対称前処理 (hybrid Gauss-Seidelスムーザ) のため、CGは使用不可。
非対称前処理対応のKrylovソルバーとしてGMRESとBiCGStabを比較した結果、
**BiCGStabが圧倒的に優れる** (30kHz渦電流、1.44M DOFs):

| ソルバー | 反復数 | 計算時間 | メモリ | ms/反復 |
|---------|------:|--------:|------:|--------:|
| **BiCGStab** | **26** | **35.5s** | **3.8 GB** | 1,367 |
| GMRES | 528 | 1,112.6s | 15.3 GB | 2,107 |

**BiCGStabはGMRES比33倍高速**。理由:
- BiCGStabは固定8ワークベクトル — 反復数に依存しないO(N)メモリ
- GMRESはKrylov基底を蓄積 (528本) — O(k*N)メモリ、Gram-Schmidt直交化のO(k^2)コスト
- 高周波数 (30kHz) で反復数が増加すると、GMRESのメモリ・計算コストが爆発的に増大

#### EMD前処理との比較 (Hiruma SA-26-001, 30kHz)

EMD (Electro-Magnetic Decoupling) 前処理 (比留間, IEEJ SA-26-001/RM-26-001, 2026.3.5) との比較:

| 手法 | DOFs | 反復数 | 計算時間 | vs EMD最良 |
|------|-----:|------:|---------:|:---------:|
| EMD (IC + GenEO-DDM 24分割) | 3.67M | 1,004 | 550.8s | 1.0x |
| **HYPRE AMS + BiCGStab (推定)** | 3.67M | ~26 | **~85s** | **~6.5x** |
| HYPRE AMS + BiCGStab (実測) | 1.44M | 26 | 48.9s | — |

HYPRE AMSはHCurlの離散勾配構造 (G行列) を直接利用してcurl-curl核を除去するため、
構造を知らないIC前処理より反復数が40倍少ない (26 vs 1,004)。

#### ComplexHypreAMSPreconditioner ベンチマーク (BiCGStab, tol=1e-8)

Hiruma渦電流問題 (銅コイル+鉄芯、f=50Hz):
**計算環境**: Intel i7-9700K (8C/8T), 128GB RAM, Windows Server 2022, MKL 2024.2

| メッシュ | DOFs | Python逐次 | C++ TaskManager | 高速化 |
|---------|-----:|---:|---:|---:|
| 2.5T | 155k | 4.72s, 26 it | **2.67s, 26 it** | **1.77x** |
| 5.5T | 331k | 10.56s, 26 it | **6.49s, 26 it** | **1.63x** |
| 20.5T | 1.44M | 54.80s, 26 it | **37.17s, 26 it** | **1.47x** |

反復数は完全に同一 — 数学的に同じ前処理、TaskManager並列化のみの差。

ベンチマーク実行: `python examples/hiruma/bench_hypre_ams.py`

### ABMC ICCG (curl-curl 実数)

3D HCurl curl-curl問題でのベンチマーク (`nograds=True`)。

**トロイダルコイル** (148K DOFs, order 2, 8スレッド — div-freeソース):

| ソルバー | 反復数 | 計算時間 | vs ICCG |
|---------|--------|---------|---------|
| ICCG | 463 | 4.4 s | 1.0x |
| ICCG + ABMC (8色) | 414 | 2.6 s | **1.7x** |

## ドキュメント

[docs/](docs/) に詳細ドキュメント (日本語) を整備:
- [アーキテクチャ](docs/architecture.md) — ソースコード構成と設計
- [アルゴリズム](docs/algorithms.md) — アルゴリズム解説 (IC, SGS-MRTR, CG, ABMC)
- [ABMC実装ガイド](docs/abmc_implementation_details.md) — 並列三角解法、性能解析
- [APIリファレンス](docs/api_reference.md) — Python APIリファレンス
- [チュートリアル](docs/tutorials.md) — 実践例 (全ソルバー比較)
- [開発者向け情報](docs/development.md) — ビルド、テスト、開発ノート

## 機能

### 前処理
- **HYPRE AMS** (Auxiliary-space Maxwell Solver) — HCurl p=1渦電流向け、**本ライブラリのメイン機能**
  - `HypreAMSPreconditioner`: 実数SPD行列向け
  - `ComplexHypreAMSPreconditioner`: TaskManager並列Re/Im (1.5x高速化)
  - BiCGStab外部ソルバー推奨 (非対称前処理)
- **IC** (不完全コレスキー) — shifted IC(0)、半正定値行列向けauto-shift
- **SGS** (対称ガウス・ザイデル) — 分解不要

### 反復法ソルバー
- **BiCGStab** — 非対称前処理 (HYPRE AMS) と組み合わせ (`examples/hiruma/bicgstab_solver.py`)。固定8ワークベクトル、O(N)メモリ
- **CG** (共役勾配法) — SPD系、複素対称 (`conjugate=False`) 対応
- **COCR** (Conjugate Orthogonal Conjugate Residual) — C++ネイティブ、複素対称系 (A^T=A) の最適短漸化式Krylovソルバー (Sogabe-Zhang 2007)
- **SGS-MRTR** — split formula内蔵のMRTR

### 統合手法
- **HYPRE AMS+BiCGStab** — HYPRE AMS前処理 + BiCGStabSolver (大規模HCurl渦電流、**推奨**)
- **ICCG** — CG + IC前処理 (オプションでABMC並列三角解法)
- **SGSMRTR** — SGS-MRTR (自己完結型)

### 高度な機能
- 半正定値行列 (curl-curl問題) 向けauto-shift IC分解
- 対角スケーリングによる条件数改善
- ABMC (代数的ブロックマルチカラー) 順序付けによる並列三角解法
- 数値的破綻検出と収束認識型回復
- best-result追跡 (未収束時は最良反復解を返却)
- 残差履歴記録

### 並列化

全並列処理は `core/parallel.hpp` によりコンパイル時にディスパッチ:

| ビルド設定 | バックエンド | 用途 |
|-----------|------------|------|
| `SPARSESOLV_USE_NGSOLVE_TASKMANAGER` | NGSolve TaskManager (`ngcore::ParallelFor/ParallelReduce`) | NGSolve統合 |
| `_OPENMP` | OpenMP `#pragma omp parallel for` | スタンドアロン+OpenMP |
| (どちらもなし) | シリアル実行 | スタンドアロン (スレッドなし) |

## インストール

### ビルド済みWheel (推奨)

[Releases](https://github.com/ksugahar/ngsolve-sparsesolv/releases) ページからプラットフォーム対応の `.whl` ファイルをダウンロードしてインストール:

```bash
pip install sparsesolv_ngsolve-2.1.0-cp312-cp312-win_amd64.whl
```

ファイル名はPythonバージョンとOSに合わせて適宜変更。

### ソースからビルド

[Git for Windows](https://gitforwindows.org/) (または同等品)、CMake 3.16+、
C++17コンパイラ、NGSolve (`pip install ngsolve` またはソースビルド) が必要。

```bash
pip install git+https://github.com/ksugahar/ngsolve-sparsesolv.git
```

またはクローンしてローカルビルド:

```bash
git clone https://github.com/ksugahar/ngsolve-sparsesolv.git
cd ngsolve-sparsesolv
pip install .
```

開発ビルド (editable install, 手動CMake) は [docs/development.md](docs/development.md) を参照。

### 動作確認

```bash
python -c "from sparsesolv_ngsolve import SparseSolvSolver; print('OK')"
```

## NGSolveでの使い方

### クイックスタート (ICCG)

```python
from ngsolve import *
from sparsesolv_ngsolve import SparseSolvSolver

mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
fes = H1(mesh, order=2, dirichlet="left|right|top|bottom")
u, v = fes.TnT()

a = BilinearForm(fes)
a += grad(u)*grad(v)*dx
a.Assemble()

f = LinearForm(fes)
f += 1*v*dx
f.Assemble()

gfu = GridFunction(fes)
solver = SparseSolvSolver(a.mat, method="ICCG",
                          freedofs=fes.FreeDofs(), tol=1e-10)
gfu.vec.data = solver * f.vec
```

### HYPRE AMS + BiCGStab (複素渦電流、推奨)

HCurl p=1の複素渦電流問題に対するHYPRE AMS前処理。
`ComplexHypreAMSPreconditioner` がRe/Im部分をTaskManagerで並列処理する。

```python
import sparsesolv_ngsolve as ssn
from ngsolve import *
from bicgstab_solver import BiCGStabSolver  # examples/hiruma/

# 複素系 A = K + jw*sigma*M を組み立て (省略)
# 実数SPD補助行列を構築
fes_real = HCurl(mesh, order=1, nograds=True,
                 dirichlet="dirichlet", complex=False)
u_r, v_r = fes_real.TnT()
a_real = BilinearForm(fes_real)
a_real += nu_cf * curl(u_r) * curl(v_r) * dx
a_real += 1e-6 * nu_cf * u_r * v_r * dx
a_real += abs(omega) * sigma_cf * u_r * v_r * dx("cond")
a_real.Assemble()

# HYPRE AMS構成要素: 離散勾配、頂点座標
G_mat, h1_fes = fes_real.CreateGradient()

nv = mesh.nv
cx, cy, cz = [0.0]*nv, [0.0]*nv, [0.0]*nv
for i in range(nv):
    pt = mesh.ngmesh.Points()[i + 1]
    cx[i], cy[i], cz[i] = pt[0], pt[1], pt[2]

# ComplexHypreAMSPreconditioner (TaskManager並列Re/Im)
pre = ssn.ComplexHypreAMSPreconditioner(
    a_real_mat=a_real.mat, grad_mat=G_mat,
    freedofs=fes_real.FreeDofs(),
    coord_x=cx, coord_y=cy, coord_z=cz,
    ndof_complex=fes.ndof, cycle_type=1, print_level=0)

# HYPRE AMSは非対称前処理 → BiCGStab推奨 (CGは使用不可)
with TaskManager():
    inv = BiCGStabSolver(mat=a.mat, pre=pre, maxiter=500, tol=1e-8)
    gfu.vec.data = inv * f.vec
```

### ABMC並列化ICCG

```python
# ABMCでIC前処理の三角解法を並列化
solver = SparseSolvSolver(a.mat, method="ICCG",
                          freedofs=fes.FreeDofs(), tol=1e-10,
                          use_abmc=True,
                          abmc_block_size=4,
                          abmc_num_colors=4)
gfu.vec.data = solver * f.vec
```

メモリ効率に優れる (CG: ~5ベクトル)。
148K DOF HCurl問題 (8スレッド) でICCG計算時間を4.4秒から2.6秒に短縮 (1.7倍高速化)。

### COCRソルバー (複素対称系)

複素対称FEM行列 (渦電流等) にはCOCRが最適。C++ネイティブ実装。
非共役内積 (x^T y) を使用、||A*r~||_2を最小化するためCGより滑らかな収束。

```python
import sparsesolv_ngsolve

# 外部前処理 + COCRソルバー (対称前処理が利用可能な場合に推奨)
inv = sparsesolv_ngsolve.COCRSolver(a.mat, pre, maxiter=500, tol=1e-8)
gfu.vec.data = inv * f.vec
print(f"COCR: {inv.iterations} iterations")
```

### 前処理 + NGSolve CGSolver

```python
from sparsesolv_ngsolve import ICPreconditioner, SGSPreconditioner
from ngsolve.krylovspace import CGSolver

# IC前処理 + NGSolve CG
pre = ICPreconditioner(a.mat, freedofs=fes.FreeDofs(), shift=1.05)
inv = CGSolver(a.mat, pre, tol=1e-10, maxiter=2000)
gfu.vec.data = inv * f.vec
```

### 3D curl-curl (半正定値系)

```python
from netgen.occ import Box, Pnt

box = Box(Pnt(0,0,0), Pnt(1,1,1))
for face in box.faces:
    face.name = "outer"
mesh = Mesh(box.GenerateMesh(maxh=0.3))

fes = HCurl(mesh, order=1, dirichlet="outer", nograds=True)
u, v = fes.TnT()

a = BilinearForm(fes)
a += curl(u)*curl(v)*dx
a.Assemble()

f = LinearForm(fes)
f += CF((0,0,1))*v*dx
f.Assemble()

# auto-shift ICで半正定値curl-curl行列に対応
solver = SparseSolvSolver(a.mat, method="ICCG",
                          freedofs=fes.FreeDofs(),
                          tol=1e-8, maxiter=2000, shift=1.0)
solver.auto_shift = True
solver.diagonal_scaling = True

gfu = GridFunction(fes)
gfu.vec.data = solver * f.vec
```

### 複素対称系 vs エルミート系

SparseSolvは複素対称系とエルミート系の両方を `conjugate` パラメータで切り替える:

```python
# 複素対称 (A^T = A): 渦電流 (curl-curl + iσ mass) など
# デフォルト: conjugate=False (非共役内積 a^T * b)
solver = SparseSolvSolver(a.mat, method="ICCG",
                          freedofs=fes.FreeDofs(), tol=1e-8)

# エルミート (A^H = A): 実係数問題を複素FE空間で解く場合など
# conjugate=True (共役内積 a^H * b)
solver = SparseSolvSolver(a.mat, method="ICCG",
                          freedofs=fes.FreeDofs(), tol=1e-8,
                          conjugate=True)
```

NGSolveの `CGSolver(conjugate=True/False)` に対応。

### SparseSolvSolver パラメータ

| パラメータ | 型 | 既定値 | 説明 |
|-----------|------|--------|------|
| `method` | str | `"ICCG"` | ソルバー手法: `ICCG`, `SGSMRTR`, `CG` |
| `tol` | float | `1e-10` | 収束許容値 |
| `maxiter` | int | `1000` | 最大反復回数 |
| `shift` | float | `1.05` | ICシフトパラメータ (安定性) |
| `auto_shift` | bool | `False` | IC破綻時のシフト自動増加 |
| `diagonal_scaling` | bool | `False` | 対角スケーリング |
| `save_best_result` | bool | `True` | 最良反復解の追跡・復元 |
| `save_residual_history` | bool | `False` | 各反復の残差を記録 |
| `conjugate` | bool | `False` | エルミート系用の共役内積 |
| `divergence_check` | bool | `False` | 停滞時の早期終了 |
| `printrates` | bool | `False` | 収束情報のstdout出力 |
| `use_abmc` | bool | `False` | ABMC順序付けによる並列三角解法 |
| `abmc_block_size` | int | `4` | ABMCブロックあたりの行数 |
| `abmc_num_colors` | int | `4` | ABMC目標色数 |

### Pythonクラス一覧

```python
from sparsesolv_ngsolve import (
    # HYPRE AMS前処理 (HCurl渦電流向け — メイン機能)
    HypreAMSPreconditioner,              # 実数HYPRE AMS
    ComplexHypreAMSPreconditioner,       # 複素Re/Im TaskManager並列
    HypreBoomerAMGPreconditioner,        # H1 AMG
    has_hypre,                           # HYPRE利用可否

    # IC/SGS前処理
    ICPreconditioner,              # NGSolve CGSolverと併用
    SGSPreconditioner,             # NGSolve CGSolverと併用

    # ソルバー
    SparseSolvSolver,              # 統合ソルバー (ICCG, SGSMRTR)
    COCRSolver,                    # COCR (複素対称系、C++ネイティブ)
    SparseSolvResult,              # ソルブ結果
)
```

Factory関数は行列の型に基づいて適切なテンプレート (`double` or `complex<double>`) に自動ディスパッチ。

## スタンドアロン使用 (C++)

```cpp
#include <sparsesolv/sparsesolv.hpp>

// CSR行列ビューを作成 (ゼロコピー)
sparsesolv::SparseMatrixView<double> A(rows, cols, row_ptr, col_idx, values);

// ICCGで解く
sparsesolv::SolverConfig config;
config.tolerance = 1e-10;
config.max_iterations = 1000;

auto result = sparsesolv::solve_iccg(A, b, x, size, config);
if (result.converged) {
    // 解は x に格納
}
```

## ディレクトリ構成

```
ngsolve-sparsesolv/
├── docs/                        # ドキュメント (日本語)
├── include/sparsesolv/         # ヘッダオンリーライブラリ
│   ├── sparsesolv.hpp          # メインヘッダ (全コンポーネントをインクルード)
│   ├── core/                   # 型、設定、行列ビュー、前処理基底
│   │   ├── dense_matrix.hpp    # 小規模密行列 + LU逆行列
│   │   ├── sparse_matrix_coo.hpp # COO疎行列 (組立用)
│   │   ├── sparse_matrix_csr.hpp # CSR疎行列 (格納用)
│   │   ├── parallel.hpp        # 並列化抽象レイヤ (TaskManager/OpenMP/serial)
│   │   ├── level_schedule.hpp  # レベルスケジューリング (三角解法並列化)
│   │   └── abmc_ordering.hpp   # ABMC順序付け (三角解法並列化)
│   ├── preconditioners/        # IC, SGS, AMS実装
│   │   ├── ic_preconditioner.hpp          # 不完全コレスキー (auto-shift)
│   │   ├── sgs_preconditioner.hpp         # 対称ガウス・ザイデル
│   │   ├── hypre_ams_preconditioner.hpp   # HYPRE AMS + ComplexHypreAMS
│   │   └── hypre_boomeramg_preconditioner.hpp  # HYPRE BoomerAMG
│   ├── solvers/                # CG, COCR, SGS-MRTR実装
│   └── ngsolve/                # NGSolve BaseMatrixラッパー + pybind11エクスポート
├── ngsolve/
│   └── python_module.cpp       # pybind11モジュールエントリポイント
├── examples/
│   └── hiruma/                 # 渦電流ベンチマーク (Hiruma問題)
│       ├── bicgstab_solver.py  # BiCGStab (Van der Vorst 1992) NGSolve実装
│       ├── bench_hypre_ams.py  # HYPRE AMS Python vs C++ TaskManager比較
│       ├── bench_emd_comparison.py # EMD前処理 (Hiruma SA-26-001) との比較
│       └── eddy_current.py     # A-Phi定式化の渦電流解析
├── tests/
│   └── test_sparsesolv.py      # ソルバー・前処理テスト
├── sparsesolv_ngsolve.pyi      # Python型スタブ
├── CMakeLists.txt              # ヘッダオンリーライブラリ + NGSolveモジュールビルド
└── LICENSE
```

## 参考文献

1. T. Iwashita, H. Nakashima, Y. Takahashi,
   "Algebraic Block Multi-Color Ordering Method for Parallel Multi-Threaded
   Sparse Triangular Solver in ICCG Method",
   *Proc. IEEE 26th International Parallel and Distributed Processing Symposium (IPDPS)*, 2012.
   [DOI: 10.1109/IPDPS.2012.51](https://doi.org/10.1109/IPDPS.2012.51)
   — ABMCオーダリングアルゴリズム。

2. J.A. Meijerink, H.A. van der Vorst,
   "An iterative solution method for linear systems of which the coefficient matrix
   is a symmetric M-matrix",
   *Math. Comp.*, Vol. 31, No. 137, pp. 148–162, 1977.
   [DOI: 10.1090/S0025-5718-1977-0438681-4](https://doi.org/10.1090/S0025-5718-1977-0438681-4)
   — 不完全コレスキー (IC) 分解。

3. M.R. Hestenes, E. Stiefel,
   "Methods of conjugate gradients for solving linear systems",
   *J. Res. Nat. Bur. Standards*, Vol. 49, No. 6, pp. 409–436, 1952.
   [DOI: 10.6028/jres.049.044](https://doi.org/10.6028/jres.049.044)
   — 共役勾配 (CG) 法。

4. E. Cuthill, J. McKee,
   "Reducing the bandwidth of sparse symmetric matrices",
   *Proc. 24th National Conference of the ACM*, pp. 157–172, 1969.
   [DOI: 10.1145/800195.805928](https://doi.org/10.1145/800195.805928)
   — Reverse Cuthill-McKee (RCM) 帯域縮小順序付け。

5. 圓谷友紀, 三船泰, 岩下武史, 高橋英治,
   "MRTR法に基づく前処理付き反復法の数値実験",
   *電気学会研究会資料*, SA-12-64, 2012.
   — SGS-MRTR法: 国内研究会論文。

6. T. Tsuburaya, Y. Okamoto, K. Fujiwara, S. Sato,
   "Improvement of the Preconditioned MRTR Method With Eisenstat's Technique
   in Real Symmetric Sparse Matrices",
   *IEEE Transactions on Magnetics*, Vol. 49, No. 5, pp. 1641–1644, 2013.
   [DOI: 10.1109/TMAG.2013.2240283](https://doi.org/10.1109/TMAG.2013.2240283)
   — 前処理MRTR法の高速化。

7. 圓谷友紀,
   "大規模電磁界問題の有限要素解析のための反復法の開発",
   *博士論文*, 宇都宮大学, 2016.
   — 大規模電磁界FEMの反復法に関する包括的参考文献。

8. R. Hiptmair, J. Xu,
    "Nodal auxiliary space preconditioning in H(curl) and H(div) spaces",
    *SIAM J. Numer. Anal.*, Vol. 45, No. 6, pp. 2483–2509, 2007.
    [DOI: 10.1137/060660588](https://doi.org/10.1137/060660588)
    — AMS (Auxiliary-space Maxwell Solver) の理論的基礎。

9. T.V. Kolev, P.S. Vassilevski,
    "Parallel auxiliary space AMG for H(curl) problems",
    *J. Comput. Math.*, Vol. 27, No. 5, pp. 604–623, 2009.
    [DOI: 10.4208/jcm.2009.27.5.013](https://doi.org/10.4208/jcm.2009.27.5.013)
    — HYPRE AMSの並列実装。HCurlスムーザ、Pi/G補正の実装詳細。

10. R.D. Falgout, U.M. Yang,
    "hypre: A library of high performance preconditioners",
    *Proc. ICCS 2002*, LNCS 2331, pp. 632–641, 2002.
    [DOI: 10.1007/3-540-47789-6_66](https://doi.org/10.1007/3-540-47789-6_66)
    — HYPREライブラリ (BoomerAMG, AMS等) の概要。

11. T. Sogabe, S.-L. Zhang,
    "A COCR method for solving complex symmetric linear systems",
    *J. Comput. Appl. Math.*, Vol. 199, No. 2, pp. 297–303, 2007.
    [DOI: 10.1016/j.cam.2005.07.032](https://doi.org/10.1016/j.cam.2005.07.032)
    — COCR法: 複素対称行列の最適短漸化式Krylovソルバー。

12. H.A. van der Vorst,
    "Bi-CGSTAB: A Fast and Smoothly Converging Variant of Bi-CG
    for the Solution of Nonsymmetric Linear Systems",
    *SIAM J. Sci. Stat. Comput.*, Vol. 13, No. 2, pp. 631–644, 1992.
    [DOI: 10.1137/0913035](https://doi.org/10.1137/0913035)
    — BiCGStab法: 非対称前処理対応、固定メモリ (8ワークベクトル)。

13. JP-MARs/SparseSolv,
    https://github.com/JP-MARs/SparseSolv

## ライセンス

本プロジェクトは [Mozilla Public License 2.0](LICENSE) の下でライセンスされている。

SparseSolvはNGSolveとは独立したモジュールとして提供される。
インストール済みNGSolveに対してリンクするスタンドアロンpybind11拡張モジュールとしてビルドされる。
