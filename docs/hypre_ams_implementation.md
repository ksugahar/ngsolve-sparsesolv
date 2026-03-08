# HYPRE AMS 実装解説

ComplexHypreAMSPreconditioner の内部実装を解説する。
HYPRE AMS (Auxiliary-space Maxwell Solver) を NGSolve の HCurl 渦電流問題に統合し、
Re/Im を TaskManager で並列適用する。

---

## 1. アーキテクチャ概要

```
Python (NGSolve)                     C++ (sparsesolv_ngsolve)
─────────────────────                ──────────────────────────────────
                                     ComplexHypreAMSPreconditioner
BiCGStabSolver                         ├── ams_re_: HypreAMSPreconditioner
  mat = a.mat (complex)                │     └── HYPRE AMS instance #1
  pre = ComplexHypreAMS ──────────────→│          ├── parcsr_A_ (real SPD)
                                       │          ├── parcsr_G_ (gradient)
  inv = BiCGStab(mat, pre)             │          └── par_coords[3]
  gfu.vec.data = inv * f.vec           │
                                       └── ams_im_: HypreAMSPreconditioner
                                             └── HYPRE AMS instance #2
                                                  ├── parcsr_A_ (同一行列)
                                                  ├── parcsr_G_ (同一行列)
                                                  └── par_coords[3]
```

**設計方針**: 2つの独立した HYPRE AMS インスタンスを保持し、
Re/Im を `ParallelFor(2, ...)` で同時適用する。
HYPRE 内部の作業ベクトル (b, x) がインスタンスごとに独立なため、
ロックやバリアなしで安全に並列化できる。

---

## 2. HYPRE AMS の数学的背景

### 2.1 HCurl curl-curl 問題

渦電流問題の複素 FEM 行列:

```
A = K + eps*M + j*omega*sigma*M_cond
```

| 項 | 意味 | 式 |
|---|------|---|
| K | curl-curl 剛性 | int nu * curl(u) . curl(v) dx |
| eps*M | 安定化質量 | int eps*nu * u . v dx |
| j*omega*sigma*M_cond | 渦電流 | int j*omega*sigma * u . v dx("cond") |

K は大きなカーネル (grad H1) を持ち、標準的な前処理 (IC, Jacobi) では
収束が極めて遅い。AMS はこのカーネルを離散勾配 G で直接扱う。

### 2.2 AMS の原理

AMS は Hiptmair-Xu (2007) の **補助空間前処理** に基づく:

```
HCurl 空間 V_h = Nedelec 辺要素
H1 空間 S_h = 節点ラグランジュ

離散勾配: G : S_h -> V_h  (curl(G*phi) = 0)
```

AMS の 1 V-cycle は以下の 3 段階:

1. **スムーザ**: hybrid Gauss-Seidel (HCurl 空間で直接)
2. **Pi 補助空間補正**: `Pi = [G, x*G, y*G, z*G]` を用いた AMG
3. **G 補助空間補正**: gradient カーネルに対する AMG

cycle_type=1 (加法型) では:

```
M^{-1} = S + Pi * B_Pi^{-1} * Pi^T + G * B_G^{-1} * G^T
```

ここで S はスムーザ、B_Pi と B_G は BoomerAMG V-cycle。

### 2.3 前処理の非対称性

**重要**: HYPRE AMS の hybrid Gauss-Seidel スムーザ (`relax_type=3`) は
前進掃引+後退掃引の順序が固定されており、**非対称前処理**となる。
そのため CG は使えず、BiCGStab (または GMRES) が必要。

---

## 3. NGSolve 行列から HYPRE 行列への変換

### 3.1 IJ 形式と ParCSR 形式

HYPRE は 2 段階の行列構築を要求する:

```
IJ (構築用) → Assemble → ParCSR (計算用)
```

```cpp
// 1. IJ 行列を作成 (行範囲指定、シリアルでは [0, n-1])
HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, ndof-1, 0, ncols-1, &hyp_A_);
HYPRE_IJMatrixSetObjectType(hyp_A_, HYPRE_PARCSR);
HYPRE_IJMatrixInitialize(hyp_A_);

// 2. 行ごとにエントリを追加
for (int k = 0; k < ndof; k++) {
    auto cols = mat->GetRowIndices(k);
    auto vals = mat->GetRowValues(k);
    HYPRE_IJMatrixAddToValues(hyp_A_, 1, &ncols_k, &k, cols, vals);
}

// 3. アセンブル → ParCSR 抽出
HYPRE_IJMatrixAssemble(hyp_A_);
HYPRE_IJMatrixGetObject(hyp_A_, (void**)&parcsr_A_);
```

### 3.2 Dirichlet DOF の処理

freedofs (自由 DOF) マスクに基づいて、Dirichlet 行を単位行列行に置換:

```cpp
if (freedofs_ && !freedofs_->Test(k)) {
    // Dirichlet 行: A[k,k] = 1, 他は 0
    HYPRE_Int col = k;
    double val = 1.0;
    HYPRE_IJMatrixAddToValues(hyp_A_, 1, &one, &k, &col, &val);
} else {
    // 自由行: NGSolve のエントリをそのままコピー
    // ...
}
```

**勾配行列 G**: Dirichlet フィルタリングなしで全エントリをコピーする。
HYPRE AMS 内部が G の構造を利用して grad カーネルを処理するため、
G を変更すると AMS の動作が破壊される。

### 3.3 対称格納の展開 (BoomerAMG)

NGSolve の SparseMatrix は上三角のみを格納する場合がある (対称格納)。
BoomerAMG はフル格納を要求するため、下三角を明示的に復元する:

```cpp
// 対称格納の検出: 中間行に自身より小さい列番号があるか
int mid = ndof_ / 2;
auto mid_row = mat_->GetRowIndices(mid);
bool is_symmetric_storage = true;
for (int j = 0; j < mid_row.Size(); j++)
    if (mid_row[j] < mid) { is_symmetric_storage = false; break; }

// 対称格納の場合: 下三角エントリを収集して追加
if (is_symmetric_storage) {
    for (int k = 0; k < ndof_; k++)
        for each (col, val) in row k where col > k:
            lower_entries[col].push_back({k, val});  // 転置を保存
    // 後で AddToValues で追加
}
```

---

## 4. ComplexHypreAMSPreconditioner の実装

### 4.1 複素 → 実数分離

複素渦電流系 A*x = b を解くとき、前処理は実部と虚部を独立に処理:

```
f = f_re + j * f_im  (複素 RHS)

u_re = AMS^{-1} * f_re    ← instance #1
u_im = AMS^{-1} * f_im    ← instance #2  (TaskManager 並列)

u = u_re + j * u_im
```

### 4.2 Mult() の実装

```cpp
void Mult(const BaseVector& f, BaseVector& u) const override {
    // (1) 複素入力を Re/Im に分離
    auto fv_f = f.FVComplex();
    for (int i = 0; i < ndof_real_; i++) {
        re_in_[i] = fv_f[i].real();
        im_in_[i] = fv_f[i].imag();
    }

    // (2) TaskManager で 2 タスクを並列実行
    ParallelFor(2, [&](int task) {
        if (task == 0)
            ams_re_->Mult(*re_in_, *re_out_);   // Re 側
        else
            ams_im_->Mult(*im_in_, *im_out_);   // Im 側
    });

    // (3) Re/Im を複素出力に統合
    auto fv_u = u.FVComplex();
    for (int i = 0; i < ndof_real_; i++)
        fv_u[i] = Complex(re_out_[i], im_out_[i]);
}
```

### 4.3 並列化の詳細

| 項目 | 値 |
|------|---|
| タスク数 | 2 (Re, Im) |
| 同期 | `ParallelFor` 内部バリアのみ |
| データ競合 | なし (独立インスタンス) |
| TaskManager 不使用時 | 逐次実行 (正しい結果) |

**2 インスタンスが必要な理由**: HYPRE AMS の `Mult()` は内部作業ベクトル
(`hyp_b_`, `hyp_x_`) を書き換える。1 インスタンスを 2 スレッドから呼ぶと
データ競合が発生する。行列 (A, G) と座標は読み取り専用だが、
HYPRE の内部状態は共有不可。

### 4.4 補助行列 A_real の構成

複素行列 A をそのまま HYPRE に渡せないため、
等価な実数 SPD 補助行列を構成する:

```python
A_real = K + eps*M + |omega| * sigma * M_cond
```

| 元の項 | 補助行列での対応 |
|--------|----------------|
| K (curl-curl) | そのまま (実数、SPD) |
| eps*M (安定化) | そのまま |
| j*omega*sigma*M | |omega|*sigma*M (虚部の絶対値) |

この A_real は SPD なので HYPRE AMS のセットアップに使える。
前処理の「質」は A_real が A の実数スペクトルをどれだけ近似するかで決まる。

---

## 5. HYPRE AMS セットアップオプション

### 5.1 AMS 設定

```cpp
HYPRE_AMSCreate(&precond_);
HYPRE_AMSSetDimension(precond_, 3);              // 3次元問題
HYPRE_AMSSetTol(precond_, 0);                    // 前処理モード (収束判定なし)
HYPRE_AMSSetMaxIter(precond_, 1);                // 1 V-cycle のみ
HYPRE_AMSSetCycleType(precond_, cycle_type);     // 1=加法, 2=乗法
HYPRE_AMSSetDiscreteGradient(precond_, parcsr_G_);
HYPRE_AMSSetCoordinateVectors(precond_, cx, cy, cz);
HYPRE_AMSSetup(precond_, parcsr_A_, par_b, par_x);
```

### 5.2 cycle_type の選択

| cycle_type | 名称 | パターン | 特徴 |
|-----------|------|---------|------|
| 1 | 加法型 | 01210 | デフォルト、安定、やや多い反復 |
| 2 | 乗法 V-cycle | — | 反復数減、非対称度が増す |
| 7 | 変形加法 | 0201020 | 大規模問題向け |

ベンチマークでは cycle_type=1 で 26 反復 (全メッシュ共通)。

### 5.3 スムーザ (relax_type)

HYPRE AMS のデフォルトスムーザは `relax_type=3` (hybrid Gauss-Seidel):

```
hybrid GS = forward GS + backward GS (固定順序)
```

前進と後退の順序が固定されているため **非対称前処理** となり、
CG は使えない。BiCGStab が必要な根本的理由。

`relax_type=6` (対称 GS) に変更すれば対称になるが、
AMS の反復数が増加する (推奨しない)。

---

## 6. BiCGStab との組み合わせ

### 6.1 BiCGStab アルゴリズム概要

Van der Vorst (1992) のアルゴリズム。非対称前処理に対応:

```
作業ベクトル: r, r0, p, v, s, t, p_hat, s_hat  (固定 8 本)

r = b - A*x, r0 = r の影ベクトル (固定)
rho = alpha = omega = 1, p = v = 0

反復 k = 1, 2, ...:
    rho_new = <r0, r>
    beta = (rho_new/rho) * (alpha/omega)
    p = r + beta*(p - omega*v)

    p_hat = M^{-1} * p          ← AMS 1回適用
    v = A * p_hat                ← SpMV 1回

    alpha = rho_new / <r0, v>
    s = r - alpha*v

    s_hat = M^{-1} * s          ← AMS 1回適用
    t = A * s_hat                ← SpMV 1回

    omega = <t, s> / <t, t>
    x += alpha*p_hat + omega*s_hat
    r = s - omega*t
```

### 6.2 1 反復あたりのコスト

| 操作 | 回数 | 備考 |
|------|------|------|
| SpMV (A*x) | 2 | 複素行列 × 複素ベクトル |
| 前処理適用 (M^{-1}) | 2 | ComplexHypreAMS Mult() |
| 内積 | 3 | <r0,r>, <r0,v>, <t,s>+<t,t> |
| AXPY | 5 | ベクトル更新 |

**メモリ**: 8 ワークベクトル × ndof × 16 bytes (complex) = 固定 O(N)

### 6.3 GMRES との比較

| 特性 | BiCGStab | GMRES(m) |
|------|---------|----------|
| メモリ | 8 ベクトル (固定) | m+1 ベクトル (反復数依存) |
| SpMV/反復 | 2 | 1 |
| 前処理/反復 | 2 | 1 |
| 直交化コスト | なし | O(m^2) Gram-Schmidt |
| リスタート | 不要 | 必要 (情報損失) |
| 収束保証 | なし (breakdown 可能) | 残差単調減少 |

渦電流問題 (30kHz, 1.44M DOFs) での実測:
- BiCGStab: 26 反復, 35.5s, 3.8 GB
- GMRES: 528 反復, 1112.6s, 15.3 GB → **BiCGStab が 33x 高速**

---

## 7. メモリレイアウト

### 7.1 HYPRE 側のメモリ

```
HypreAMSPreconditioner (1 インスタンス):
  parcsr_A_     : nnz(A_real) × 12 bytes (index + value)
  parcsr_G_     : nnz(G) × 12 bytes
  par_coords[3] : 3 × ndof_h1 × 8 bytes
  hyp_b_, hyp_x_: 2 × ndof_hc × 8 bytes
  AMS hierarchy : BoomerAMG levels (粗いレベルは小さい)

ComplexHypreAMSPreconditioner:
  2 × HypreAMS メモリ
  + 4 × ndof_real × 8 bytes (re_in, im_in, re_out, im_out)
```

### 7.2 実測メモリ (1.44M DOFs)

| 項目 | サイズ |
|------|--------|
| 複素 FEM 行列 | ~2.0 GB |
| 実数補助行列 | ~1.0 GB |
| HYPRE AMS × 2 | ~0.5 GB |
| BiCGStab 8 ベクトル | ~184 MB |
| **合計** | **~3.8 GB** |

---

## 8. 離散勾配 G と頂点座標の取得

### 8.1 NGSolve での G 行列生成

```python
fes_real = HCurl(mesh, order=1, nograds=True, complex=False)
G_mat, h1_fes = fes_real.CreateGradient()
```

`CreateGradient()` は HCurl → H1 の離散勾配行列 G を返す:
- G の形状: ndof_hcurl × ndof_h1
- G[e,v] = +1 or -1 (辺 e の端点 v に対する勾配)
- `curl(G * phi) = 0` (勾配場は curl-free)

### 8.2 頂点座標

```python
nv = mesh.nv
coord_x = [0.0] * nv
coord_y = [0.0] * nv
coord_z = [0.0] * nv
for i in range(nv):
    p = mesh.ngmesh.Points()[i + 1]   # Netgen は 1-indexed
    coord_x[i] = p[0]
    coord_y[i] = p[1]
    coord_z[i] = p[2]
```

**注意**: `mesh.ngmesh.Points()` は **1-indexed** (Netgen の仕様)。
`i + 1` で 0-indexed の Python 配列に格納する。

座標は AMS 内部の Pi 補助空間 `Pi = [G, x*G, y*G, z*G]` の構成に使用される。

---

## 9. HYPRE オブジェクトのライフサイクル

```
構築:
  HYPRE_IJMatrixCreate()   → IJ ハンドル
  HYPRE_IJMatrixSetObjectType(, HYPRE_PARCSR)
  HYPRE_IJMatrixInitialize()

データ設定:
  HYPRE_IJMatrixAddToValues()  × 行数回
  HYPRE_IJMatrixAssemble()     → ParCSR 内部構築

抽出:
  HYPRE_IJMatrixGetObject()    → ParCSR ポインタ (IJ が所有)

使用:
  HYPRE_AMSSetup(solver, A, b, x)  ← ParCSR を渡す
  HYPRE_AMSSolve(solver, A, b, x)  ← 前処理適用

破棄:
  HYPRE_AMSDestroy(solver)     ← 内部階層を解放
  HYPRE_IJMatrixDestroy(A)     ← IJ + ParCSR 両方解放
  HYPRE_IJVectorDestroy(b)
```

デストラクタ `Cleanup()` で全 HYPRE オブジェクトを解放:

```cpp
~HypreAMSPreconditioner() { Cleanup(); }

void Cleanup() {
    if (precond_) { HYPRE_AMSDestroy(precond_); precond_ = nullptr; }
    if (hyp_A_)   { HYPRE_IJMatrixDestroy(hyp_A_); hyp_A_ = nullptr; }
    if (hyp_G_)   { HYPRE_IJMatrixDestroy(hyp_G_); hyp_G_ = nullptr; }
    if (hyp_b_)   { HYPRE_IJVectorDestroy(hyp_b_); hyp_b_ = nullptr; }
    if (hyp_x_)   { HYPRE_IJVectorDestroy(hyp_x_); hyp_x_ = nullptr; }
    for (int d = 0; d < 3; d++)
        if (hyp_coord_[d]) { HYPRE_IJVectorDestroy(hyp_coord_[d]); hyp_coord_[d] = nullptr; }
}
```

---

## 10. ベンチマーク結果

**計算環境**: Intel Core i7-9700K (8C/8T, 3.60 GHz), 128 GB DDR4,
Windows Server 2022, Intel MKL 2024.2

### 10.1 Python 逐次 vs C++ TaskManager 並列

BiCGStab + ComplexHypreAMS, 50 Hz 渦電流, tol=1e-8:

| メッシュ | DOFs | Python 逐次 | C++ TaskManager | 高速化 |
|---------|-----:|---:|---:|---:|
| 2.5T | 155k | 4.72s, 26 it | 2.67s, 26 it | 1.77x |
| 5.5T | 331k | 10.56s, 26 it | 6.49s, 26 it | 1.63x |
| 20.5T | 1.44M | 54.80s, 26 it | 37.17s, 26 it | 1.47x |

反復数は全メッシュで **26 回** (メッシュサイズ非依存)。

### 10.2 EMD 前処理との比較 (30 kHz)

| 手法 | DOFs | 反復数 | 時間 |
|------|-----:|------:|-----:|
| EMD: IC only (SA-26-001) | 3.67M | 15,838 | 5964.8s |
| EMD: IC+AMG W-cycle | 3.67M | 2,935 | 1552.9s |
| EMD: IC+GenEO-DDM 24 | 3.67M | 1,004 | 550.8s |
| **HYPRE AMS + BiCGStab** | 1.44M | **26** | **35.5s** |

HYPRE AMS は HCurl の勾配構造 (G 行列) を直接利用するため、
構造を知らない IC 前処理より反復数が **40 倍少ない** (26 vs 1,004)。

---

## 参考文献

1. Kolev, Tz. V. and Vassilevski, P. S., "Parallel Auxiliary Space AMG for H(curl) Problems",
   *J. Comput. Math.*, 27(5), 604-623, 2009.
2. Hiptmair, R. and Xu, J., "Nodal Auxiliary Space Preconditioning in H(curl) and H(div) Spaces",
   *SIAM J. Numer. Anal.*, 45(6), 2483-2509, 2007.
3. Van der Vorst, H. A., "Bi-CGSTAB: A Fast and Smoothly Converging Variant of Bi-CG
   for the Solution of Nonsymmetric Linear Systems", *SIAM J. Sci. Stat. Comput.*,
   13(2), 631-644, 1992.
4. HYPRE Reference Manual, https://hypre.readthedocs.io/
