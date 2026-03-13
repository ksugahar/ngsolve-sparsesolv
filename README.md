# ngsolve-sparsesolv — NGSolve + SparseSolv モノリシックパッケージ

[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)
[![PyPI](https://img.shields.io/pypi/v/ngsolve-sparsesolv)](https://pypi.org/project/ngsolve-sparsesolv/)

**渦電流問題でICCGの60倍少ない反復回数。外部依存なし。**

公式 [NGSolve](https://ngsolve.org/) に追随しつつ、以下の付加価値を提供するモノリシックパッケージ:

1. **Intel MKL build** — MKL BLAS/LAPACK + PARDISO直接法
2. **SetGeomInfo API** — 外部メッシュ（Cubit, GMSH）の高次曲面要素対応 (PR#86)
3. **SparseSolv** — Compact AMS, COCR等のHCurl向け反復法ソルバー

`pip install ngsolve-sparsesolv` で netgen + ngsolve + sparsesolv_ngsolve が全て入る。

SparseSolvは [JP-MARs/SparseSolv](https://github.com/JP-MARs/SparseSolv) からのフォーク。
ヘッダオンリーC++17、`double` と `std::complex<double>` の両方に対応。

## 主な特徴

1. **ICCG** — 不完全コレスキー(IC)前処理付き共役勾配法。auto-shiftによる半正定値行列（curl-curl）対応。複素対称行列（非共役内積）対応
2. **ABMC並列化** — ABMC (Algebraic Block Multi-Color) 順序付け [Iwashita et al. 2012] による並列不完全コレスキー三角求解
3. **Compact AMS + COCR** — HCurl問題向け補助空間前処理＋クリロフソルバー。外部依存なし（ヘッダオンリーC++）。実数（静磁界＋CG）・複素数（渦電流＋COCR）対応。`Update()` による非線形ソルバー（ニュートン反復）対応

## なぜSparseSolv？

NGSolveは直接法とBDDCを内蔵しているが、SparseSolvはNGSolveが標準提供しない機能を追加する：

1. **IC前処理 + auto-shift** — curl-curl FEM行列は半正定値であり、通常のIC分解は破綻する。SparseSolvのauto-shift ICは破綻を検知し自動的にシフトを増加
2. **ABMC並列三角求解** — IC前処理の三角求解は本質的に逐次的である。ABMC順序付けがこのボトルネックを除去し、並列前進・後退代入を実現 [Iwashita et al. 2012]
3. **Compact AMS + COCR** — 補助空間前処理 (Hiptmair-Xu 2007) とCOCRクリロフソルバー (Sogabe-Zhang 2007) の組合せ。HCurl有限要素による複素渦電流問題 (A = K + jω·σ·M) 向け。詳細は [docs/compact_ams_cocr.md](docs/compact_ams_cocr.md) を参照

## ソルバー選択指針

| 問題 | 有限要素空間 | 推奨手法 | 理由 |
|------|------------|---------|------|
| ポアソン方程式 | H1 | **ICCG** | メモリ効率が高く高速 |
| Curl-curl（実数） | HCurl (`nograds=True`) | **Shifted-ICCG** | auto-shift ICで半正定値対応 |
| 静磁界（実数・大規模） | HCurl (実数, p=1) | **Compact AMS + CG** | メッシュサイズに依存しない反復回数 |
| 静磁界（非線形） | HCurl (実数) | **Compact AMS + CG** | `Update()` でニュートン反復対応 |
| 渦電流（複素・大規模） | HCurl (複素, p=1) | **Compact AMS + COCR** | メッシュサイズに依存しない反復回数 |
| 渦電流（複素・中小規模） | HCurl (複素) | **ICCG** (`conjugate=False`) | メモリ効率が高い |

## 性能

### Compact AMS + COCR（複素渦電流）

比留間渦電流問題（銅コイル＋鉄心 μ_r=1000、30 kHz、tol=1e-10）。
**環境**: Intel Xeon (8コア)、Windows Server 2022、MSVC 2022、MKL 2024.2。

| メッシュ | HCurl自由度 | 反復回数 | 時間 | ms/反復 | メモリ |
|---------|----------:|---------:|----:|-------:|------:|
| mesh1_2.5T | 155,527 | 144 | 4.5s | 25.8 | 368 MB |
| mesh1_3.5T | 197,395 | 168 | 7.3s | 37.7 | 460 MB |
| mesh1_5.5T | 331,595 | 249 | 16.2s | 57.3 | 725 MB |
| mesh1_20.5T | 1,441,102 | 499 | 222.6s | 396.2 | 2,933 MB |

ABMC-ICCG（IC前処理のみ）との比較（mesh1_3.5T、197k自由度）：

| 手法 | 反復回数 | 時間 | 状態 |
|------|--------:|----:|------|
| Compact AMS + COCR | 168 | 7.2s | 収束 |
| ABMC-ICCG | 17,178 | 438.4s | 非収束 (残差=2.8e-10) |

IC前処理はHCurl離散化のcurl-curl零空間を扱えない。
AMSは離散勾配補正とNedelec補間補正によりこれを解決する。

ベンチマーク実行: `python examples/hiruma/bench_compact_ams.py --all`

### ABMC ICCG

3D HCurl curl-curl問題（148K自由度、次数2、8スレッド）：

| ソルバー | 反復回数 | 時間 | vs ICCG |
|---------|---------|------|---------|
| ICCG | 463 | 4.4 s | 1.0x |
| ICCG + ABMC (8色) | 414 | 2.6 s | **1.7x** |

## ドキュメント

詳細なドキュメントは [docs/](docs/) にある：
- [Compact AMS + COCR](docs/compact_ams_cocr.md) — 複素渦電流問題向け補助空間前処理
- [アルゴリズム](docs/algorithms.md) — アルゴリズム詳説（IC、SGS-MRTR、CG、ABMC、Compact AMS）
- [APIリファレンス](docs/api_reference.md) — Python API
- [チュートリアル](docs/tutorials.md) — 実用例
- [開発者情報](docs/development.md) — ビルド、テスト、アーキテクチャ

## 機能

### 前処理
- **Compact AMS** (Auxiliary-space Maxwell Solver) — HCurl向け補助空間前処理。Hiptmair-Xu (2007) アルゴリズムのヘッダオンリーC++実装。外部依存なし
  - `CompactAMSPreconditioner`: 実数HCurl（静磁界）向け、CGソルバーと併用
  - `ComplexCompactAMSPreconditioner`: 複素渦電流向け、Re/Im融合処理
  - `Update()`: 非線形ソルバー（ニュートン反復）対応。幾何情報を保持し行列依存部分のみ再構築
  - CompactAMG: PMISコースニング + l1-ヤコビ + Vサイクル、DualMult融合
- **IC** (不完全コレスキー) — シフト付きIC(0)、半正定値行列向けauto-shift
- **SGS** (対称ガウスザイデル) — 分解不要

### 反復法ソルバー
- **COCR** (Conjugate Orthogonal Conjugate Residual) — 複素対称系（A^T=A）向け短漸化式クリロフソルバー [Sogabe-Zhang 2007]。O(n)メモリ、リスタート不要
- **CG** (共役勾配法) — SPD系向け、複素対称（`conjugate=False`）対応
- **SGS-MRTR** — 分割公式内蔵MRTR法

### 組合せ手法
- **Compact AMS + COCR** — 複素渦電流（大規模HCurl p=1）向け
- **ICCG** — CG + IC前処理（ABMC並列三角求解オプション）
- **SGSMRTR** — SGS-MRTR（自己完結型）

### 高度な機能
- 半正定値行列（curl-curl問題）向けauto-shift IC分解
- 対角スケーリングによる条件数改善
- ABMC (Algebraic Block Multi-Color) 順序付けによる並列三角求解
- 数値的破綻検知と収束ベスト結果復元
- ベスト結果追跡（非収束時にベスト反復結果を返却）

### 並列化

すべての並列処理は `core/parallel.hpp` によりコンパイル時にディスパッチ：

| ビルド設定 | バックエンド | 用途 |
|-----------|------------|------|
| `SPARSESOLV_USE_NGSOLVE_TASKMANAGER` | NGSolve TaskManager (`ngcore::ParallelFor/ParallelReduce`) | NGSolve統合 |
| `_OPENMP` | OpenMP `#pragma omp parallel for` | スタンドアロン + OpenMP |
| （なし） | 逐次実行 | スタンドアロン（スレッドなし） |

## インストール

### PyPI（推奨）

```bash
pip install ngsolve-sparsesolv
```

これ一つで netgen + ngsolve (MKL) + sparsesolv_ngsolve が全てインストールされる。

> **注意**: 旧パッケージ `sparsesolv-ngsolve` は廃止。`pip install ngsolve-sparsesolv` に移行してください。

### ソースからビルド（SparseSolv単体）

既にNGSolveがインストール済みの場合、SparseSolvのみビルド可能：

```bash
pip install ngsolve  # 公式NGSolve
pip install git+https://github.com/ksugahar/ngsolve-sparsesolv.git
```

### モノリシックWheel（ソースからフルビルド）

```bash
git clone https://github.com/ksugahar/ngsolve-sparsesolv.git
cd ngsolve-sparsesolv
python scripts/build_monolithic.py
pip install dist/ngsolve_sparsesolv-*.whl
```

### 動作確認

```bash
python -c "import ngsolve; from sparsesolv_ngsolve import SparseSolvSolver; print('OK')"
```

## NGSolveでの使用

### クイックスタート（ICCG）

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

### ABMC並列化ICCG

```python
# ABMC順序付けによるIC前処理三角求解の並列化
solver = SparseSolvSolver(a.mat, method="ICCG",
                          freedofs=fes.FreeDofs(), tol=1e-10,
                          use_abmc=True,
                          abmc_block_size=4,
                          abmc_num_colors=4)
gfu.vec.data = solver * f.vec
```

### Compact AMS + COCR（複素渦電流）

```python
import sparsesolv_ngsolve as ssn
from ngsolve import *

# 複素系 A = K + jω·σ·M のアセンブル（省略）
# 実数SPD補助行列の構築
fes_real = HCurl(mesh, order=1, nograds=True,
                 dirichlet="dirichlet", complex=False)
u_r, v_r = fes_real.TnT()
a_real = BilinearForm(fes_real)
a_real += nu_cf * curl(u_r) * curl(v_r) * dx
a_real += 1e-6 * nu_cf * u_r * v_r * dx
a_real += abs(omega) * sigma_cf * u_r * v_r * dx("cond")
a_real.Assemble()

# 離散勾配行列と頂点座標
G_mat, h1_fes = fes_real.CreateGradient()
coord_x = [mesh.ngmesh.Points()[i+1][0] for i in range(mesh.nv)]
coord_y = [mesh.ngmesh.Points()[i+1][1] for i in range(mesh.nv)]
coord_z = [mesh.ngmesh.Points()[i+1][2] for i in range(mesh.nv)]

# Compact AMS前処理 + COCRソルバー
pre = ssn.ComplexCompactAMSPreconditioner(
    a_real_mat=a_real.mat, grad_mat=G_mat,
    freedofs=fes_real.FreeDofs(),
    coord_x=coord_x, coord_y=coord_y, coord_z=coord_z,
    ndof_complex=fes.ndof, cycle_type=1, print_level=0)

with TaskManager():
    inv = ssn.COCRSolver(a.mat, pre,
                         freedofs=fes.FreeDofs(),
                         maxiter=500, tol=1e-10)
    gfu.vec.data = inv * f.vec

print(f"Converged in {inv.iterations} iterations")
```

詳細は [docs/compact_ams_cocr.md](docs/compact_ams_cocr.md) を参照。

### 前処理 + NGSolve CGSolver

```python
from sparsesolv_ngsolve import ICPreconditioner, SGSPreconditioner
from ngsolve.krylovspace import CGSolver

# IC前処理 + NGSolve CG
pre = ICPreconditioner(a.mat, freedofs=fes.FreeDofs(), shift=1.05)
inv = CGSolver(a.mat, pre, tol=1e-10, maxiter=2000)
gfu.vec.data = inv * f.vec
```

### SparseSolvSolverパラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|------|---------|------|
| `method` | str | `"ICCG"` | ソルバー手法: `ICCG`, `SGSMRTR`, `CG`, `COCR` |
| `tol` | float | `1e-10` | 収束判定閾値 |
| `maxiter` | int | `1000` | 最大反復回数 |
| `shift` | float | `1.05` | ICシフトパラメータ（安定化） |
| `auto_shift` | bool | `False` | IC破綻時の自動シフト増加 |
| `diagonal_scaling` | bool | `False` | 対角スケーリング |
| `save_best_result` | bool | `True` | ベスト反復結果の追跡・復元 |
| `conjugate` | bool | `False` | エルミート系向け共役内積 |
| `use_abmc` | bool | `False` | ABMC順序付けによる並列三角求解 |
| `abmc_block_size` | int | `4` | ABMCブロック当たりの行数 |
| `abmc_num_colors` | int | `4` | ABMC目標色数 |

### Pythonクラス一覧

```python
from sparsesolv_ngsolve import (
    # Compact AMS前処理（HCurl向け、Update()対応）
    CompactAMSPreconditioner,      # 実数HCurl（静磁界）+ CG
    ComplexCompactAMSPreconditioner,   # 複素渦電流、Re/Im融合 + COCR

    # IC/SGS前処理
    ICPreconditioner,              # NGSolve CGSolverと併用
    SGSPreconditioner,             # NGSolve CGSolverと併用

    # ソルバー
    SparseSolvSolver,              # 統合ソルバー (ICCG, SGSMRTR)
    COCRSolver,                    # COCR（複素対称系、C++ネイティブ）
    GMRESSolver,                   # GMRES（非対称系、左前処理）
    SparseSolvResult,              # 解結果
)
```

## スタンドアロン使用（C++）

```cpp
#include <sparsesolv/sparsesolv.hpp>

// CSR行列ビューの作成（ゼロコピー）
sparsesolv::SparseMatrixView<double> A(rows, cols, row_ptr, col_idx, values);

// ICCGで求解
sparsesolv::SolverConfig config;
config.tolerance = 1e-10;
config.max_iterations = 1000;

auto result = sparsesolv::solve_iccg(A, b, x, size, config);
```

## ディレクトリ構成

```
ngsolve-sparsesolv/
├── include/sparsesolv/         # ヘッダオンリーライブラリ
│   ├── sparsesolv.hpp          # メインヘッダ
│   ├── core/                   # 型定義、設定、行列ビュー
│   │   ├── parallel.hpp        # 並列化抽象層 (TaskManager/OpenMP/逐次)
│   │   └── abmc_ordering.hpp   # ABMC順序付け
│   ├── preconditioners/        # IC, SGS, Compact AMS実装
│   │   ├── ic_preconditioner.hpp          # 不完全コレスキー (auto-shift)
│   │   ├── sgs_preconditioner.hpp         # 対称ガウスザイデル
│   │   ├── compact_amg.hpp                # 古典的AMG (PMIS, l1-ヤコビ, DualMult)
│   │   ├── compact_ams.hpp                # AMS前処理 (Hiptmair-Xu 2007)
│   │   └── complex_compact_ams.hpp        # Re/Im融合ComplexCompactAMS
│   ├── solvers/                # CG, COCR, SGS-MRTR実装
│   └── ngsolve/                # NGSolve BaseMatrixラッパー + pybind11
├── patches/                    # 本家への適用パッチ
│   └── netgen-setgeominfo.patch    # SetGeomInfo API (PR#86採択まで)
├── scripts/
│   └── build_monolithic.py     # モノリシックwheel構築スクリプト
├── examples/
│   └── hiruma/                 # 渦電流ベンチマーク (30 kHz)
├── tests/
│   └── test_sparsesolv.py      # ソルバー・前処理テスト
├── docs/                       # ドキュメント
├── NGSOLVE_VERSION             # 追随する公式NGSolveバージョン
├── sparsesolv_ngsolve.pyi      # Python型スタブ
├── CMakeLists.txt              # SparseSolv独立ビルド
├── CLAUDE.md                   # 開発ポリシー
└── LICENSE
```

## 参考文献

1. T. Iwashita, H. Nakashima, Y. Takahashi,
   "Algebraic Block Multi-Color Ordering Method for Parallel Multi-Threaded
   Sparse Triangular Solver in ICCG Method",
   *Proc. IEEE IPDPS*, 2012.
   [DOI: 10.1109/IPDPS.2012.51](https://doi.org/10.1109/IPDPS.2012.51)

2. J.A. Meijerink, H.A. van der Vorst,
   "An iterative solution method for linear systems of which the coefficient matrix
   is a symmetric M-matrix",
   *Math. Comp.*, Vol. 31, No. 137, pp. 148-162, 1977.

3. R. Hiptmair, J. Xu,
   "Nodal auxiliary space preconditioning in H(curl) and H(div) spaces",
   *SIAM J. Numer. Anal.*, Vol. 45, No. 6, pp. 2483-2509, 2007.

4. T. Sogabe, S.-L. Zhang,
   "A COCR method for solving complex symmetric linear systems",
   *J. Comput. Appl. Math.*, Vol. 199, No. 2, pp. 297-303, 2007.

5. T.V. Kolev, P.S. Vassilevski,
   "Parallel auxiliary space AMG for H(curl) problems",
   *J. Comput. Math.*, Vol. 27, No. 5, pp. 604-623, 2009.

6. JP-MARs/SparseSolv,
   https://github.com/JP-MARs/SparseSolv

## ライセンス

[Mozilla Public License 2.0](LICENSE) のもとで提供。
