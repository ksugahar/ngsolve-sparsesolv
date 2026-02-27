# Python APIリファレンス

## インポート

```python
import ngsolve  # 先にインポート必須 (共有ライブラリのロード)
from sparsesolv_ngsolve import (
    BDDCPreconditioner,    # BDDC領域分割前処理
    ICPreconditioner,      # 不完全コレスキー前処理
    SGSPreconditioner,     # 対称ガウス・ザイデル前処理
    SparseSolvSolver,      # 統合反復法ソルバー
    SparseSolvResult,      # ソルブ結果
)
```

Factory関数は行列の型 (実数/複素数) を `mat.IsComplex()` で自動判定する。

---

## BDDCPreconditioner

BDDC (Balancing Domain Decomposition by Constraints) 前処理。
NGSolveの `CGSolver` と組み合わせて使用する。

### コンストラクタ

```python
pre = BDDCPreconditioner(a, fes)
```

| 引数 | 型 | 説明 |
|------|------|------|
| `a` | `BilinearForm` | **組立済み**の双線形形式 (`a.Assemble()` 済み) |
| `fes` | `FESpace` | 有限要素空間 |

粗空間ソルバーにはMKL PARDISOを使用（内部で自動選択）。

**内部処理**:
1. `fes.CouplingType` からDOF分類 (wirebasket/interface) を取得
2. `BilinearForm.Integrators()` から要素行列を `CalcElementMatrix` で計算
3. 要素ごとにSchur補体を計算し、BDDC前処理を構築
4. 粗空間ソルバー (SparseCholesky等) を構築

### プロパティ

| プロパティ | 型 | 説明 |
|-----------|------|------|
| `num_wirebasket_dofs` | `int` | Wirebasket (粗空間) DOF数 |
| `num_interface_dofs` | `int` | Interface (局所) DOF数 |

### 使用例

```python
from ngsolve import *
from ngsolve.krylovspace import CGSolver
from sparsesolv_ngsolve import BDDCPreconditioner

# 問題設定
fes = H1(mesh, order=3, dirichlet="outer")
u, v = fes.TnT()
a = BilinearForm(fes)
a += InnerProduct(grad(u), grad(v)) * dx
a.Assemble()

# BDDC + CG
pre = BDDCPreconditioner(a, fes)
inv = CGSolver(a.mat, pre, tol=1e-10, maxiter=500)
gfu.vec.data = inv * f.vec
print(f"Iterations: {inv.iterations}")
```

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

統合反復法ソルバー。ICCG, SGSMRTR, CG を選択可能。
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
| `method` | `str` | `"ICCG"` | `"ICCG"`, `"SGSMRTR"`, `"CG"` |
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
