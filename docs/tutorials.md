# 実践チュートリアル

コピペ実行可能な完全例。各例ではセットアップ時間、CG反復回数、
ソルブ時間、解の精度を出力する。

---

## 1. H1 Poisson 3D

3次元単位立方体上のPoisson方程式。BDDC, IC, SGS の3手法を比較する。

```python
import time
import ngsolve
from ngsolve import *
from ngsolve.krylovspace import CGSolver
from netgen.occ import Box, Pnt
from sparsesolv_ngsolve import (
    BDDCPreconditioner, ICPreconditioner, SGSPreconditioner
)

# メッシュ生成
box = Box(Pnt(0, 0, 0), Pnt(1, 1, 1))
for face in box.faces:
    face.name = "outer"
mesh = box.GenerateMesh(maxh=0.3)

# H1空間 (order=3)
fes = H1(mesh, order=3, dirichlet="outer")
u, v = fes.TnT()
print(f"DOF: {fes.ndof}")

# 組立
a = BilinearForm(fes)
a += InnerProduct(grad(u), grad(v)) * dx
a.Assemble()

f = LinearForm(fes)
f += 1 * v * dx
f.Assemble()

# 参照解 (直接法)
gfu_ref = GridFunction(fes)
gfu_ref.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f.vec
ref_norm = sqrt(Integrate(gfu_ref**2, mesh))

# --- BDDC + CG ---
t0 = time.perf_counter()
pre_bddc = BDDCPreconditioner(a, fes)
t_setup = time.perf_counter() - t0

gfu = GridFunction(fes)
inv = CGSolver(a.mat, pre_bddc, printrates=False, tol=1e-10, maxiter=500)
t0 = time.perf_counter()
gfu.vec.data = inv * f.vec
t_solve = time.perf_counter() - t0

err = sqrt(Integrate((gfu - gfu_ref)**2, mesh)) / ref_norm
print(f"BDDC+CG: setup={t_setup:.3f}s, solve={t_solve:.3f}s, "
      f"iters={inv.iterations}, error={err:.2e}")

# --- IC + CG ---
t0 = time.perf_counter()
pre_ic = ICPreconditioner(a.mat, freedofs=fes.FreeDofs(), shift=1.05)
t_setup = time.perf_counter() - t0

gfu = GridFunction(fes)
inv = CGSolver(a.mat, pre_ic, printrates=False, tol=1e-10, maxiter=2000)
t0 = time.perf_counter()
gfu.vec.data = inv * f.vec
t_solve = time.perf_counter() - t0

err = sqrt(Integrate((gfu - gfu_ref)**2, mesh)) / ref_norm
print(f"IC+CG:   setup={t_setup:.3f}s, solve={t_solve:.3f}s, "
      f"iters={inv.iterations}, error={err:.2e}")

# --- SGS + CG ---
t0 = time.perf_counter()
pre_sgs = SGSPreconditioner(a.mat, freedofs=fes.FreeDofs())
t_setup = time.perf_counter() - t0

gfu = GridFunction(fes)
inv = CGSolver(a.mat, pre_sgs, printrates=False, tol=1e-10, maxiter=2000)
t0 = time.perf_counter()
gfu.vec.data = inv * f.vec
t_solve = time.perf_counter() - t0

err = sqrt(Integrate((gfu - gfu_ref)**2, mesh)) / ref_norm
print(f"SGS+CG:  setup={t_setup:.3f}s, solve={t_solve:.3f}s, "
      f"iters={inv.iterations}, error={err:.2e}")
```

**典型的な出力** (maxh=0.3, order=3):
```
DOF: ~8000
BDDC+CG: setup=0.15s, solve=0.01s, iters=2,   error=2.5e-11
IC+CG:   setup=0.02s, solve=0.08s, iters=58,  error=3.1e-11
SGS+CG:  setup=0.00s, solve=0.25s, iters=180, error=4.2e-11
```

---

## 2. VectorH1 弾性体

3次元弾性体問題。全面固定、体積力 (0, 0, -1)。

```python
import time
import ngsolve
from ngsolve import *
from ngsolve.krylovspace import CGSolver
from netgen.occ import Box, Pnt
from sparsesolv_ngsolve import BDDCPreconditioner, ICPreconditioner

box = Box(Pnt(0, 0, 0), Pnt(1, 1, 1))
for face in box.faces:
    face.name = "outer"
mesh = box.GenerateMesh(maxh=0.3)

fes = VectorH1(mesh, order=2, dirichlet="outer")
u, v = fes.TnT()
print(f"DOF: {fes.ndof}")

# 弾性定数
E, nu = 1.0, 0.3
mu = E / (2 * (1 + nu))
lam = E * nu / ((1 + nu) * (1 - 2 * nu))

def eps(u):
    return 0.5 * (Grad(u) + Grad(u).trans)

def sigma(u):
    return 2 * mu * eps(u) + lam * Trace(eps(u)) * Id(3)

a = BilinearForm(fes)
a += InnerProduct(sigma(u), eps(v)) * dx
a.Assemble()

f = LinearForm(fes)
f += CF((0, 0, -1)) * v * dx
f.Assemble()

# 参照解
gfu_ref = GridFunction(fes)
gfu_ref.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f.vec
ref_norm = sqrt(Integrate(InnerProduct(gfu_ref, gfu_ref), mesh))

# BDDC
t0 = time.perf_counter()
pre = BDDCPreconditioner(a, fes)
t_setup = time.perf_counter() - t0

gfu = GridFunction(fes)
inv = CGSolver(a.mat, pre, tol=1e-10, maxiter=500)
t0 = time.perf_counter()
gfu.vec.data = inv * f.vec
t_solve = time.perf_counter() - t0

err = sqrt(Integrate(InnerProduct(gfu - gfu_ref, gfu - gfu_ref), mesh)) / ref_norm
print(f"BDDC: setup={t_setup:.3f}s, solve={t_solve:.3f}s, "
      f"iters={inv.iterations}, error={err:.2e}")

# IC
t0 = time.perf_counter()
pre_ic = ICPreconditioner(a.mat, freedofs=fes.FreeDofs())
t_setup = time.perf_counter() - t0

gfu = GridFunction(fes)
inv = CGSolver(a.mat, pre_ic, tol=1e-10, maxiter=2000)
t0 = time.perf_counter()
gfu.vec.data = inv * f.vec
t_solve = time.perf_counter() - t0

err = sqrt(Integrate(InnerProduct(gfu - gfu_ref, gfu - gfu_ref), mesh)) / ref_norm
print(f"IC:   setup={t_setup:.3f}s, solve={t_solve:.3f}s, "
      f"iters={inv.iterations}, error={err:.2e}")
```

---

## 3. HCurl curl-curl (実数)

磁気ベクトルポテンシャル問題。正則化項 σ|u|^2 を加える。
ソース項は curl-based (離散div-free保証)。
B = curl(A) で精度評価する。

**注意**:
- `nograds=True` で勾配DOFを除外 (DOF数削減 + 条件数改善)
- 正則化項 σ は小さい値 (1e-6) で十分

```python
import time
import ngsolve
from ngsolve import *
from ngsolve.krylovspace import CGSolver
from netgen.occ import Box, Pnt
from sparsesolv_ngsolve import BDDCPreconditioner, SparseSolvSolver

box = Box(Pnt(0, 0, 0), Pnt(1, 1, 1))
for face in box.faces:
    face.name = "outer"
mesh = box.GenerateMesh(maxh=0.3)

fes = HCurl(mesh, order=2, dirichlet="outer", nograds=True)
u, v = fes.TnT()
print(f"DOF: {fes.ndof}")

sigma_val = 1e-6
a = BilinearForm(fes)
a += InnerProduct(curl(u), curl(v)) * dx
a += sigma_val * InnerProduct(u, v) * dx
a.Assemble()

# Curl-based source (離散的にdiv J = 0 が保証される)
T = CF((y*(1-y)*z*(1-z), z*(1-z)*x*(1-x), x*(1-x)*y*(1-y)))
f = LinearForm(fes)
f += InnerProduct(T, curl(v)) * dx
f.Assemble()

# 参照解
gfu_ref = GridFunction(fes)
gfu_ref.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f.vec
B_ref = curl(gfu_ref)
B_norm = sqrt(Integrate(InnerProduct(B_ref, B_ref), mesh))

# BDDC + CG
t0 = time.perf_counter()
pre = BDDCPreconditioner(a, fes)
t_setup = time.perf_counter() - t0

gfu = GridFunction(fes)
inv = CGSolver(a.mat, pre, tol=1e-10, maxiter=500)
t0 = time.perf_counter()
gfu.vec.data = inv * f.vec
t_solve = time.perf_counter() - t0

B = curl(gfu)
B_err = sqrt(Integrate(InnerProduct(B - B_ref, B - B_ref), mesh)) / B_norm
print(f"BDDC: setup={t_setup:.3f}s, solve={t_solve:.3f}s, "
      f"iters={inv.iterations}, B error={B_err:.2e}")

# Shifted-ICCG (auto-shift)
t0 = time.perf_counter()
solver = SparseSolvSolver(a.mat, method="ICCG",
                           freedofs=fes.FreeDofs(), tol=1e-10,
                           maxiter=2000, shift=1.0)
solver.auto_shift = True
solver.diagonal_scaling = True
t_setup = time.perf_counter() - t0

gfu = GridFunction(fes)
t0 = time.perf_counter()
gfu.vec.data = solver * f.vec
t_solve = time.perf_counter() - t0

B = curl(gfu)
B_err = sqrt(Integrate(InnerProduct(B - B_ref, B - B_ref), mesh)) / B_norm
iters = solver.last_result.iterations
print(f"ICCG: setup={t_setup:.3f}s, solve={t_solve:.3f}s, "
      f"iters={iters}, B error={B_err:.2e}")
```

---

## 4. HCurl 渦電流 (複素数)

渦電流問題 (curl-curl + iσ mass)。複素対称行列。
`conjugate=False` が重要 (非共役内積)。

```python
import time
import ngsolve
from ngsolve import *
from ngsolve.krylovspace import CGSolver
from netgen.occ import Box, Pnt
from sparsesolv_ngsolve import BDDCPreconditioner

box = Box(Pnt(0, 0, 0), Pnt(1, 1, 1))
for face in box.faces:
    face.name = "outer"
mesh = box.GenerateMesh(maxh=0.4)

fes = HCurl(mesh, order=2, dirichlet="outer", nograds=True, complex=True)
u, v = fes.TnT()
print(f"DOF: {fes.ndof}")

a = BilinearForm(fes)
a += InnerProduct(curl(u), curl(v)) * dx
a += 1j * InnerProduct(u, v) * dx   # iσ mass term
a.Assemble()

f = LinearForm(fes)
f += CF((0, 0, 1)) * v * dx
f.Assemble()

# 参照解
gfu_ref = GridFunction(fes)
gfu_ref.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f.vec
B_ref = curl(gfu_ref)
B_norm = sqrt(abs(Integrate(InnerProduct(B_ref, B_ref), mesh)))

# BDDC + CG (conjugate=False: 複素対称)
t0 = time.perf_counter()
pre = BDDCPreconditioner(a, fes)
t_setup = time.perf_counter() - t0

gfu = GridFunction(fes)
inv = CGSolver(a.mat, pre, tol=1e-10, maxiter=500, conjugate=False)
t0 = time.perf_counter()
gfu.vec.data = inv * f.vec
t_solve = time.perf_counter() - t0

B = curl(gfu)
B_err = sqrt(abs(Integrate(InnerProduct(B - B_ref, B - B_ref), mesh))) / B_norm
print(f"BDDC: setup={t_setup:.3f}s, solve={t_solve:.3f}s, "
      f"iters={inv.iterations}, B error={B_err:.2e}")
```

**注意**: 複素数の渦電流では `conjugate=False` が必須。
FEM行列は複素対称 (A^T = A) であり、エルミート (A^H = A) ではない。
`conjugate=True` にすると発散する。

---

## 5. ソルバー選択ガイド

| 問題の特徴 | 推奨手法 | 理由 |
|-----------|---------|------|
| H1, 高次 (order ≥ 3) | **BDDC+CG** | 反復2回。メッシュ非依存 |
| H1, 低次 (order 1-2) | **ICCG** | p=1: BDDC≈PARDISO直接法、ICCG が 10倍以上高速 |
| VectorH1 弾性体 | **BDDC+CG** | ICCGは弾性体で反復数増加 |
| HCurl curl-curl (実数) | **BDDC+CG** or **Shifted-ICCG** | auto_shift有効化 |
| HCurl 渦電流 (複素数) | **BDDC+CG** (conjugate=False) | SGS-MRTRは精度不足 |
| 小規模 (DOF < 1000) | 直接法 (`sparsecholesky`) | 反復法のオーバーヘッド大 |
| IC前処理の並列化 | ABMC (`use_abmc=True`) | 三角解法の並列度向上 |

### HCurl問題のソース構成

HCurl (curl-curl) 問題では、右辺のソース項が離散的にdiv-freeでないと
反復法が収束しない:

| ソース構成 | 離散div-free | 収束性 |
|-----------|------------|--------|
| `int J . v dx` (直接) | 不完全 | 収束しない場合あり |
| `int T . curl(v) dx` (curl-based) | 保証される | 常に収束 |
| Helmholtz補正後のJ | 保証される | 常に収束 |

渦電流問題では iσ 項が自然に正則化するため、この問題は発生しない。
