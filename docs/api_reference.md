# Python APIリファレンス

## インポート

```python
import ngsolve  # 先にインポート必須 (共有ライブラリのロード)
from sparsesolv_ngsolve import (
    # 前処理 (IC/SGS)
    ICPreconditioner,      # 不完全コレスキー前処理
    SGSPreconditioner,     # 対称ガウス・ザイデル前処理

    # HYPRE AMS前処理 (HCurl渦電流向け)
    HypreAMSPreconditioner,              # 実数HYPRE AMS
    ComplexHypreAMSPreconditioner,       # 複素Re/Im TaskManager並列
    HypreBoomerAMGPreconditioner,        # H1スカラー楕円系向けAMG
    has_hypre,                           # HYPRE利用可否チェック

    # 反復法ソルバー
    SparseSolvSolver,      # 統合反復法ソルバー (ICCG/SGSMRTR/CG/COCR)
    SparseSolvResult,      # ソルブ結果
    COCRSolver,            # COCR (複素対称系、C++ネイティブ)
)
```

Factory関数は行列の型 (実数/複素数) を `mat.IsComplex()` で自動判定する。

---

## ICPreconditioner

不完全コレスキー (IC) 前処理。

### コンストラクタ

```python
pre = ICPreconditioner(mat, freedofs=None, shift=1.05)
```

| 引数 | 型 | 既定値 | 説明 |
|------|------|--------|------|
| `mat` | `SparseMatrix` | - | SPD行列 (実数/複素数) |
| `freedofs` | `BitArray` or `None` | `None` | 自由DOF。`None`で全DOF自由 |
| `shift` | `float` | `1.05` | シフトパラメータ |

### プロパティ

| プロパティ | 型 | 説明 |
|-----------|------|------|
| `shift` | `float` | シフトパラメータ (読み書き) |

### メソッド

| メソッド | 説明 |
|---------|------|
| `Update()` | 前処理を再計算 (行列変更後に呼ぶ) |

### 使用例

```python
pre = ICPreconditioner(a.mat, freedofs=fes.FreeDofs(), shift=1.05)
inv = CGSolver(a.mat, pre, tol=1e-10)
gfu.vec.data = inv * f.vec
```

---

## SGSPreconditioner

対称ガウス・ザイデル (SGS) 前処理。

### コンストラクタ

```python
pre = SGSPreconditioner(mat, freedofs=None)
```

| 引数 | 型 | 既定値 | 説明 |
|------|------|--------|------|
| `mat` | `SparseMatrix` | - | SPD行列 |
| `freedofs` | `BitArray` or `None` | `None` | 自由DOF |

### メソッド

| メソッド | 説明 |
|---------|------|
| `Update()` | 前処理を再計算 |

### 使用例

```python
pre = SGSPreconditioner(a.mat, freedofs=fes.FreeDofs())
inv = CGSolver(a.mat, pre, tol=1e-10)
gfu.vec.data = inv * f.vec
```

---

## SparseSolvSolver

統合反復法ソルバー。ICCG, SGSMRTR, CG, COCR を選択可能。
`BaseMatrix` として使用できるため、`gfu.vec.data = solver * f.vec` の形で呼べる。

### コンストラクタ

```python
solver = SparseSolvSolver(mat, method="ICCG", freedofs=None,
                           tol=1e-10, maxiter=1000, shift=1.05,
                           save_best_result=True,
                           save_residual_history=False,
                           printrates=False, conjugate=False,
                           use_abmc=False, abmc_block_size=4,
                           abmc_num_colors=4, abmc_reorder_spmv=False,
                           abmc_use_rcm=False)
```

### パラメータ一覧

| パラメータ | 型 | 既定値 | 説明 |
|-----------|------|--------|------|
| `mat` | `SparseMatrix` | - | SPD行列 |
| `method` | `str` | `"ICCG"` | `"ICCG"`, `"SGSMRTR"`, `"CG"`, `"COCR"` |
| `freedofs` | `BitArray` | `None` | 自由DOF |
| `tol` | `float` | `1e-10` | 収束許容値 |
| `maxiter` | `int` | `1000` | 最大反復回数 |
| `shift` | `float` | `1.05` | IC分解シフト (ICCG用) |
| `save_best_result` | `bool` | `True` | 最良解の追跡 |
| `save_residual_history` | `bool` | `False` | 残差履歴の記録 |
| `printrates` | `bool` | `False` | 収束情報の表示 |
| `conjugate` | `bool` | `False` | 共役内積 (Hermitian用) |

### ABMC関連パラメータ

| パラメータ | 型 | 既定値 | 説明 |
|-----------|------|--------|------|
| `use_abmc` | `bool` | `False` | ABMC順序付け有効化 |
| `abmc_block_size` | `int` | `4` | ブロックあたりの行数 |
| `abmc_num_colors` | `int` | `4` | 目標色数 |
| `abmc_reorder_spmv` | `bool` | `False` | ABMC空間でSpMV |
| `abmc_use_rcm` | `bool` | `False` | RCM帯域縮小の前適用 |

### プロパティ (構築後に設定可能)

| プロパティ | 型 | 説明 |
|-----------|------|------|
| `auto_shift` | `bool` | IC分解の自動シフト調整 |
| `diagonal_scaling` | `bool` | 対角スケーリング |
| `divergence_check` | `bool` | 停滞検出による早期終了 |
| `divergence_threshold` | `float` | 発散検出閾値 |
| `divergence_count` | `int` | 発散判定までの連続不良反復数 |
| `last_result` | `SparseSolvResult` | 最新のソルブ結果 |

### メソッド

| メソッド | 引数 | 戻り値 | 説明 |
|---------|------|--------|------|
| `Solve(rhs, sol)` | `BaseVector`, `BaseVector` | `SparseSolvResult` | 初期値付きソルブ |

### 使用例

```python
# BaseMatrixとして使用 (初期値ゼロ)
solver = SparseSolvSolver(a.mat, method="ICCG",
                           freedofs=fes.FreeDofs(), tol=1e-10)
gfu.vec.data = solver * f.vec

# Solveメソッドで詳細結果を取得
result = solver.Solve(f.vec, gfu.vec)
print(f"Converged: {result.converged}, Iters: {result.iterations}")
```

---

## SparseSolvResult

ソルブ結果を格納する構造体。

| フィールド | 型 | 説明 |
|-----------|------|------|
| `converged` | `bool` | 収束したか |
| `iterations` | `int` | 反復回数 |
| `final_residual` | `float` | 最終相対残差 |
| `residual_history` | `list[float]` | 各反復の残差 (`save_residual_history=True`時) |

```python
result = solver.Solve(f.vec, gfu.vec)
if result.converged:
    print(f"Converged in {result.iterations} iterations")
    print(f"Final residual: {result.final_residual:.2e}")
```

---

## ComplexHypreAMSPreconditioner

複素渦電流系向けのHYPRE AMS前処理 (TaskManager並列Re/Im)。

2つの独立HYPRE AMSインスタンスを作成し、Re/Im部分をNGSolve TaskManagerで
並列に処理する。Python Re/Im wrapperに対して約1.5x高速化。

HYPRE AMSは非対称前処理 (relax_type=3, hybrid GS) → **BiCGStabSolver推奨**。

### コンストラクタ

```python
pre = ComplexHypreAMSPreconditioner(
    a_real_mat, grad_mat, freedofs=None,
    coord_x=[], coord_y=[], coord_z=[],
    ndof_complex=0, cycle_type=1, print_level=0)
```

| 引数 | 型 | 既定値 | 説明 |
|------|------|--------|------|
| `a_real_mat` | `SparseMatrix` (実数) | - | 実数SPD補助行列 (K + eps*M + \|omega\|*sigma*M) |
| `grad_mat` | `SparseMatrix` (実数) | - | 離散勾配 G (HCurl -> H1) |
| `freedofs` | `BitArray` or `None` | `None` | 自由DOF |
| `coord_x` | `list[float]` | `[]` | 頂点x座標 |
| `coord_y` | `list[float]` | `[]` | 頂点y座標 |
| `coord_z` | `list[float]` | `[]` | 頂点z座標 |
| `ndof_complex` | `int` | `0` | 複素DOF数 (`fes.ndof`) |
| `cycle_type` | `int` | `1` | HYPRE AMS cycle type |
| `print_level` | `int` | `0` | HYPRE出力レベル |

### 使用例

```python
import sparsesolv_ngsolve as ssn
from bicgstab_solver import BiCGStabSolver

pre = ssn.ComplexHypreAMSPreconditioner(
    a_real_mat=a_real.mat, grad_mat=G_mat,
    freedofs=fes_real.FreeDofs(),
    coord_x=cx, coord_y=cy, coord_z=cz,
    ndof_complex=fes.ndof, cycle_type=1, print_level=0)

with TaskManager():
    inv = BiCGStabSolver(mat=a.mat, pre=pre, maxiter=500, tol=1e-8)
    gfu.vec.data = inv * f.vec
```

### ベンチマーク結果 (BiCGStab, tol=1e-8)

| メッシュ | DOFs | Python (逐次) | C++ TaskManager | 高速化 |
|---------|-----:|---:|---:|---:|
| 2.5T | 155k | 4.72s, 26 it | **2.67s, 26 it** | **1.77x** |
| 5.5T | 331k | 10.56s, 26 it | **6.49s, 26 it** | **1.63x** |
| 20.5T | 1.44M | 54.80s, 26 it | **37.17s, 26 it** | **1.47x** |

反復数は全メッシュで26回 (メッシュサイズ非依存)。GMRES比33x高速化。

---

## HypreAMSPreconditioner

HYPRE AMS前処理 (オプション)。`SPARSESOLV_USE_HYPRE=ON` でビルド時のみ利用可能。
`has_hypre()` で利用可否を確認。

### コンストラクタ

```python
pre = HypreAMSPreconditioner(
    mat, grad_mat, freedofs=None,
    coord_x=[], coord_y=[], coord_z=[],
    cycle_type=1, print_level=0)
```

| 引数 | 型 | 既定値 | 説明 |
|------|------|--------|------|
| `mat` | `SparseMatrix` (実数) | - | 実数SPD行列 |
| `grad_mat` | `SparseMatrix` (実数) | - | 離散勾配 G |
| `freedofs` | `BitArray` or `None` | `None` | 自由DOF |
| `coord_x` | `list[float]` | `[]` | 頂点x座標 |
| `coord_y` | `list[float]` | `[]` | 頂点y座標 |
| `coord_z` | `list[float]` | `[]` | 頂点z座標 |
| `cycle_type` | `int` | `1` | HYPRE AMS cycle type |
| `print_level` | `int` | `0` | HYPRE出力レベル |

### 使用例

```python
import sparsesolv_ngsolve as ssn

if ssn.has_hypre():
    pre = ssn.HypreAMSPreconditioner(
        a_real.mat, G_mat, fes_real.FreeDofs(),
        cx, cy, cz, cycle_type=7, print_level=0)
```

---

## has_hypre

HYPRE利用可否の確認関数。

```python
ssn.has_hypre()  # -> True if built with SPARSESOLV_USE_HYPRE
```

---

## COCRSolver

COCR (Conjugate Orthogonal Conjugate Residual) ソルバー。C++ネイティブ実装。
複素対称系 (A^T=A, NOT Hermitian) の最適短漸化式Krylovソルバー。

非共役内積 (x^T y) を使用。||A*r~||_2を最小化するためCOCGより滑らかな収束。
1反復あたり: 1 MatVec + 1 前処理適用 (CGと同コスト)。

**参考文献**: Sogabe & Zhang (2007), J. Comput. Appl. Math., 199(2), 297-303.

### 使用方法1: COCRSolver (外部前処理付き)

NGSolveの `CGSolver` と同じインタフェース。AMS前処理等と組み合わせて使用。

```python
import sparsesolv_ngsolve

inv = sparsesolv_ngsolve.COCRSolver(mat, pre, maxiter=500, tol=1e-8, printrates=False)
gfu.vec.data = inv * f.vec
print(f"COCR converged in {inv.iterations} iterations")
```

| 引数 | 型 | 既定値 | 説明 |
|------|------|--------|------|
| `mat` | `BaseMatrix` | - | 複素対称行列 |
| `pre` | `BaseMatrix` | - | 前処理行列 |
| `maxiter` | `int` | `500` | 最大反復回数 |
| `tol` | `float` | `1e-8` | 収束許容値 (相対残差) |
| `printrates` | `bool` | `False` | 収束情報の表示 |

### 使用方法2: SparseSolvSolver(method="COCR")

SparseSolvSolverの統合インタフェース経由。内部IC前処理。

```python
solver = sparsesolv_ngsolve.SparseSolvSolver(mat, method="COCR",
    freedofs=fes.FreeDofs(), tol=1e-10, maxiter=1000)
gfu.vec.data = solver * f.vec
result = solver.last_result
```

### プロパティ

| プロパティ | 型 | 説明 |
|-----------|------|------|
| `iterations` | `int` | 実行反復回数 (COCRSolverのみ) |

### 収束判定

```
sqrt(|rt^T * r|) / sqrt(|rt0^T * r0|) < tol
```
ここで `rt = M^{-1} * r` (前処理残差)。
NGSolveの `CGSolver(conjugate=False)` と同等の収束基準。

### COCG について

COCG (Conjugate Orthogonal CG) は `CGSolver(conjugate=False)` と数学的に等価。
別途クラスは提供しない。

```python
from ngsolve.krylovspace import CGSolver
inv = CGSolver(a.mat, pre, conjugate=False, maxiter=500, tol=1e-8)
```
