# アルゴリズム解説

## 1. BDDC (Balancing Domain Decomposition by Constraints)

### 1.1 概要

BDDCは有限要素法の大規模問題に対する**領域分割前処理**である [Dohrmann 2003]。
メッシュを要素ごとに「サブドメイン」と見なし、
サブドメイン間の結合を**粗空間 (coarse space)** で処理する。
代数的定式化は [Mandel, Dohrmann, Tezaur 2005] を参照。

**特徴**:
- 反復回数がメッシュサイズ *h* にほぼ依存しない (スケーラブル)
- 要素行列から直接構築できる (組立済み行列不要)

**注**: SparseSolvの独自BDDC実装は削除済み。以下はアルゴリズムの解説として残す。
NGSolveの組込みBDDC (`a.mat.Inverse(fes.FreeDofs(), inverse="bddc")`) を使用すること。

### 1.2 DOF分類

有限要素空間の自由度 (DOF) を2種類に分類する:

| 種類 | 英語名 | 役割 | NGSolve CouplingType |
|------|--------|------|---------------------|
| **Wirebasket** | 粗空間DOF | 頂点・辺のDOF。グローバルに連成 | `WIREBASKET_DOF` |
| **Interface** | 局所DOF | 面・内部のDOF。サブドメイン内で消去 | それ以外 |

高次有限要素 (order ≥ 2) では、wirebasket DOFは少数の低次DOFであり、
interface DOFが大多数を占める。BDDCの効率はこの階層性に依存する。

### 1.3 要素レベルのSchur補体

各要素 *e* の剛性行列を wirebasket (w) と interface (i) に分割する:

```
K^(e) = | K_ww  K_wi |
        | K_iw  K_ii |
```

**Schur補体** (wirebasket上の等価行列):

```
S^(e) = K_ww - K_wi * K_ii^{-1} * K_iw
```

**Harmonic extension** (wirebasketの値からinterface値を復元):

```
he^(e) = -K_ii^{-1} * K_iw
```

疑似コード:
```
K_ii_inv = LU分解(K_ii)        // PA=LU 部分ピボット
he = -K_ii_inv * K_iw           // harmonic extension
S = K_ww + K_wi * he            // Schur complement
```

### 1.4 重み付け

共有DOF (複数要素に属するDOF) の寄与を正しく分配するため、
**重み付け (averaging)** を行う。

各要素のinterface DOF *k* に対する重み:

```
w_k^(e) = |K_ii^(e)(k,k)|   (K_iiの対角成分の絶対値)
```

重みの正規化: 全要素からの寄与を合計し、逆数をとる:

```
W_k = 1 / Σ_e w_k^(e)
```

この正規化後の重みで harmonic extension, inner solve, harmonic extension transpose
の行列要素をスケーリングする。


### 1.5 粗空間ソルバー

全要素のSchur補体を組み立てた **wirebasket行列** S_global を直接法で解く。

直接法ソルバー (SparseCholesky, MKL PARDISO等) を使用する。
wirebasket DOF数は全DOFに比べて少数であるため、直接法のコストは許容範囲内。
NGSolveの組込みBDDCでは、粗空間ソルバーとしてSparseCholeskyまたはPARDISOが利用可能。

### 1.6 Applyアルゴリズム (5ステップ)

BDDCの前処理適用は以下の5ステップで行う。
入力 *r* は残差ベクトル、出力 *y* は前処理済みベクトル。

```
Step 1: y = r                              (コピー)
Step 2: y += H^T * r                       (harmonic extension transpose)
Step 3: y_wb = S^{-1} * y_wb              (wirebasket成分の粗空間ソルブ)
Step 4: y += K_inner * r                   (inner solve: interface成分)
Step 5: y = y + H * y                      (harmonic extension)
```

ここで:
- H: harmonic extension行列 (全要素の he^(e) を重み付けで組み立てたもの)
- H^T: その転置
- K_inner: 全要素の K_ii^{-1} を重み付けで組み立てたもの

NGSolveでの使用例:
```python
# NGSolve組込みBDDC
inv = a.mat.Inverse(fes.FreeDofs(), inverse="bddc")
gfu.vec.data = inv * f.vec
```

---

## 2. IC分解 (不完全コレスキー分解)

### 2.1 概要

対称正定値行列 *A* に対して近似分解を求める [Meijerink, van der Vorst 1977]:

```
A ≈ L * D * L^T
```

- *L*: 下三角行列 (Aと同じスパースパターン)
- *D*: 対角行列

前処理の適用: y = (LDL^T)^{-1} * x を前進代入・対角スケーリング・後退代入の3段階で解く。

**実装ファイル**: `preconditioners/ic_preconditioner.hpp`

### 2.2 シフトパラメータ

IC分解の対角に**シフト**を加えて安定化する:

```
d_i = α * a_ii - Σ_{k<i} l_ik^2 * d_k^{-1}
```

ここで α はシフトパラメータ (既定: 1.05)。

- α = 1.0: 標準IC(0)。不定値行列では破綻する場合がある
- α = 1.05: デフォルト。やや安定
- α = 1.1〜1.2: 困難な問題向け

### 2.3 Auto-shift

半正定値行列 (curl-curl問題) では、対角が小さくなりすぎて
IC分解が破綻することがある。Auto-shiftはこれを自動検出し、
シフトを増加させて分解を再試行する:

```cpp
// ic_preconditioner.hpp: compute_ic_factorization()
if (abs_s < config_.min_diagonal_threshold && abs_orig > 0.0) {
    shift += increment;     // シフト増加
    increment *= 2;         // 指数バックオフ
    restart = true;          // 分解を再開
}
```

### 2.4 対角スケーリング

行列の条件数を改善するため、対角スケーリングを適用できる:

```
scaling[i] = 1 / sqrt(|A[i,i]|)
A_scaled[i,j] = scaling[i] * A[i,j] * scaling[j]
```

実装箇所: `ICPreconditioner::compute_scaling_factors()`, `apply_scaling_to_L()`

### 2.5 三角解法の並列化

IC分解の適用 (前進代入・後退代入) は本質的に逐次的であるが、
データ依存性解析により部分的に並列化できる。

2つの方式を提供する:

| 方式 | ファイル | 並列粒度 | 特徴 |
|------|---------|---------|------|
| レベルスケジューリング | `level_schedule.hpp` | 行単位 | 単純、低オーバーヘッド |
| ABMC順序付け | `abmc_ordering.hpp` | ブロック単位 | 高並列性 |

詳細はセクション4 (ABMCオーダリング) を参照。

---

## 3. SGS-MRTR

### 3.1 概要

SGS-MRTR は**対称ガウス・ザイデル (SGS)** 前処理を**MRTR反復法**に
組み込んだソルバーである。分割公式 (split formula) により、
前処理の前進部 *L* と後退部 *L^T* を別々に適用する。

**特徴**:
- IC分解が不要 (分解コストなし)
- DAD変換 (対角スケーリング) が内蔵されている
- 複素対称行列にも対応 (ただし精度に制限あり)

**実装ファイル**: `solvers/sgs_mrtr_solver.hpp`

**参考文献**:
- 圓谷友紀, 三船泰, 岩下武史, 高橋英治,
  "MRTR法に基づく前処理付き反復法の数値実験",
  *電気学会研究会資料*, SA-12-64, 2012.
- T. Tsuburaya, Y. Okamoto, K. Fujiwara, S. Sato,
  "Improvement of the Preconditioned MRTR Method With Eisenstat's Technique
  in Real Symmetric Sparse Matrices",
  *IEEE Trans. Magnetics*, Vol. 49, No. 5, pp. 1641-1644, 2013.
  [DOI: 10.1109/TMAG.2013.2240283](https://doi.org/10.1109/TMAG.2013.2240283)

### 3.2 DAD変換

行列 *A* を対角スケーリングする:

```
D = diag(|A|)^{-1/2}
A' = D * A * D
b' = D * b
x  = D * x'
```

スケーリング後の行列 A' の対角は全て1に近くなり、
条件数が改善される。

### 3.3 split formula

SGSの前進部 L と後退部 L^T を個別に適用し、
MRTR反復の各ステップで:

```
rd = L^{-1} * r          (前進ソルブ)
u  = L^{-T} * rd         (後退ソルブ)
ARd = u + L^{-1}(rd - u) (近似 M^{-1}Ard)
```

### 3.4 MRTR反復

MRTRは最小残差型の反復法で、2項の再帰で解を更新する:

```
p_{k+1} = u_k + (η_k ζ_{k-1} / ζ_k) p_k
x_{k+1} = x_k + ζ_k p_k
```

ここで ζ_k, η_k は最適化パラメータ。

### 3.5 複素数対応の注意

SGS-MRTRの ζ_k 計算で除算が出現する際、複素数の比較に `std::real()` を使用:

```cpp
// sgs_mrtr_solver.hpp
if (std::abs(denom) < constants::DENOMINATOR_BREAKDOWN) {
    denom = (std::real(denom) >= 0) ? ... : ...;
}
```

複素対称行列 (A^T = A) の渦電流問題では、DAD変換が最適でない可能性があり、
収束精度が5%程度にとどまる場合がある。こうした場合はICCG+ABMC、COCR、
またはHYPRE AMS+GMRESを推奨する。

---

## 4. ABMC順序付け (Algebraic Block Multi-Color Ordering)

### 4.1 概要

三角解法 (前進代入・後退代入) の**並列化**のための行順序付けアルゴリズム。
レベルスケジューリングの問題点 (FEM行列での並列度不足) を解消する。

**実装ファイル**: `core/abmc_ordering.hpp`

**参考文献**:
- 岩下武史, 中島浩, 高橋康人,
  "Algebraic Block Multi-Color Ordering Method for Parallel
  Multi-Threaded Sparse Triangular Solver in ICCG Method",
  *Proc. IEEE IPDPS*, 2012.

### 4.2 アルゴリズム

ABMCは2段階で構成される:

#### Stage 1: 代数的ブロッキング (BFS集約)

行列のグラフ上でBFS (幅優先探索) により、近傍の行をブロックにまとめる。

```
入力: 行列グラフ G, ブロックサイズ B
出力: ブロック割当て block[i] (行i → ブロックID)

1. 未割当ての行 seed から BFS開始
2. BFS探索中に B 行を集約してブロック形成
3. 全行が割り当てられるまで繰り返し
```

ブロックサイズ `B` (既定: 4) は、ブロック内の行が逐次処理されるため、
キャッシュ効率とのバランスで決まる。

#### Stage 2: マルチカラーリング (グリーディ彩色)

ブロック隣接グラフ (ブロック間に下三角依存関係があるとき辺を張る) を
グリーディ彩色する。

```
入力: ブロック隣接グラフ, 目標色数 C
出力: ブロック色 color[b] (ブロックb → 色ID)

1. ブロック依存関係の隣接グラフ構築
2. グリーディ彩色: 各ブロックに未使用の最小色を割当て
```

目標色数 `C` (既定: 4) は並列粒度を制御する。

#### 彩色アルゴリズムの詳細

彩色は2段階で行われる:

1. **最小色数の決定**: `num_colors = max(target_colors, 1 + max_lower_triangular_degree)`
2. **Forbidden-Color-Set 法**: 各ブロックの下三角隣接色を forbidden 配列にマークし、最小の未使用色を割り当て。全色が衝突した場合、新しい色を追加。

`target_colors` は**下限**であり保証値ではない。実際の色数はブロックグラフの構造に依存する。

#### 既知の制限

- 複雑なメッシュ（ヘリカルコイル等）では色数が `target_colors` を大幅に超える可能性がある
- BFS ブロッキングは行列グラフの局所性に依存。帯域幅の大きい行列ではブロック間依存が密になり彩色数が増加

### 4.3 三角解法の並列実行

ABMC順序付け後の三角解法は3階層で実行される:

```
for 色 c = 0 to C-1:            ← 逐次 (色間依存性)
    parallel_for ブロック b ∈ 色c:  ← 並列 (同色ブロックは独立)
        for 行 i ∈ ブロックb:       ← 逐次 (ブロック内依存性)
            前進代入の1行処理
```

**レベルスケジューリングとの比較**:

| 特性 | レベルスケジューリング | ABMC |
|------|---------------------|------|
| 並列粒度 | 行単位 | ブロック単位 |
| 同期回数 | レベル数 (数百) | 色数 (4程度) |
| FEMでの並列度 | 低い (深い依存チェーン) | 高い |
| オーバーヘッド | 低い | 行列並べ替えコスト |

### 4.4 Persistent parallel region (レベルスケジューリング高速化)

標準のレベルスケジューリングでは各レベルで `parallel_for` を呼び出すため、
スレッドプール起動のオーバーヘッドが数百回発生する。

`forward_substitution_persistent()` / `backward_substitution_persistent()` は、
単一の `parallel_for(nthreads, ...)` 内で全レベルを処理し、
レベル間はスピンバリアで同期する:

```cpp
// ic_preconditioner.hpp
parallel_for(nthreads, [&](index_t thread_id) {
    for (int lev = 0; lev < num_levels; ++lev) {
        // 各スレッドが担当行を処理
        barrier.wait();  // レベル間同期
    }
});
```

### 4.5 RCM順序付けとの組み合わせ

RCM (Reverse Cuthill-McKee) 帯域縮小順序付けをABMCの前に適用することで、
行列の帯域幅を削減し、SpMV (行列ベクトル積) のキャッシュ効率を改善できる。

3つの実行パスがある:

| パス | SpMV行列 | 前処理 | 設定 |
|------|---------|--------|------|
| 標準 | 元の行列 | ABMC順序で三角解法 | `use_abmc=True` |
| RCM+ABMC | RCM行列 | RCM→ABMCの合成順序 | `use_abmc=True, abmc_use_rcm=True` |
| ABMC全空間 | ABMC行列 | 順序なし (直接) | `abmc_reorder_spmv=True` |

### 4.6 性能特性

ABMCの比較対象は**persistent parallel region付きレベルスケジューリング**である
（逐次レベルスケジューリングではない）。

3D HCurl curl-curl (order=2, 8スレッド) での実測:

| DOFs | Level Sched. | ABMC (best) | Speedup |
|------|-------------|-------------|---------|
| 11K | 0.044s | 0.051s | 0.86x |
| 27K | 0.158s | 0.154s | 1.02x |
| 82K | 0.763s | 0.650s | 1.17x |
| 186K | 2.505s | 1.962s | 1.28x |

**損益分岐点: 約 25K〜30K DOFs** (8スレッド)。小規模問題ではABMCの
セットアップコストとベクトル置換オーバーヘッドが三角解法の高速化を上回る。

問題構造にも依存し、トロイダルコイル (148K DOFs, 帯域幅大) では1.8xの高速化を
達成している ([02_performance_comparison.ipynb](02_performance_comparison.ipynb))。

詳細は [abmc_implementation_details.md](abmc_implementation_details.md) のセクション7を参照。

---

## 5. CG法 (共役勾配法)

### 5.1 概要

対称正定値 (SPD) 行列に対する**前処理付き共役勾配法 (PCG)** [Hestenes, Stiefel 1952]。

**実装ファイル**: `solvers/cg_solver.hpp`

### 5.2 アルゴリズム

```
r_0 = b - A*x_0
z_0 = M^{-1} * r_0
p_0 = z_0

for k = 0, 1, 2, ...:
    α_k = (r_k, z_k) / (p_k, A*p_k)
    x_{k+1} = x_k + α_k * p_k
    r_{k+1} = r_k - α_k * A*p_k
    z_{k+1} = M^{-1} * r_{k+1}
    β_k = (r_{k+1}, z_{k+1}) / (r_k, z_k)
    p_{k+1} = z_{k+1} + β_k * p_k
```

### 5.3 複素対称行列への対応

FEM行列は複素対称 (A^T = A) であり、**エルミートではない** (A^H ≠ A)。
例: 渦電流方程式 curl-curl + iσ mass。

この場合、内積は**非共役** (unconjugated) にする必要がある:

```
(a, b) = Σ a_i * b_i       (非共役: complex-symmetric)
(a, b) = Σ conj(a_i) * b_i  (共役: Hermitian)
```

実装での切替え:

```cpp
// iterative_solver.hpp: dot_product()
if (config_.conjugate)
    sum += std::conj(a[i]) * b[i];  // Hermitian
else
    sum += a[i] * b[i];  // complex-symmetric (default)
```

NGSolveの `CGSolver(conjugate=False)` に対応する。

### 5.4 収束判定

相対残差ノルムで判定:

```
||r_k|| / ||r_0|| < tol
```

追加機能:
- **best-result追跡**: 最良の解を保持し、未収束時はそれを返す
- **発散検出**: 残差が最良値の閾値倍を超えると早期終了
- **残差履歴**: 各反復の残差を記録

---

## 6. DenseMatrix LU逆行列

### 6.1 概要

小規模密行列の LU分解と逆行列計算。COO/CSR行列ビルダー内部で使用。
PA = LU 形式の部分ピボット付きLU分解を行う。

**実装ファイル**: `core/dense_matrix.hpp`

### 6.2 置換行列の構築

部分ピボット付きLU分解では、行交換を記録する置換ベクトル `piv[]` が得られる。
置換行列 P は次のように構築する:

```
P[k, piv[k]] = 1   (正しい: PA = LU)
```

**注意**: `P[piv[k], k] = 1` は P^T (転置) であり、間違い。
この区別は対角優位な行列 (H1) では影響しにくいが、
非自明なピボット交換が発生する行列 (HCurl) では致命的になる。

```cpp
// dense_matrix.hpp: invert()
DenseMatrix inv(n, n);
for (index_t k = 0; k < n; ++k) {
    inv(k, piv[k]) = Scalar(1);  // P[k, piv[k]] = 1
}
```

### 6.3 逆行列計算の3段階

```
1. LU分解: PA = LU (部分ピボット)
2. 前進代入: L * Y = P (P は置換行列)
3. 後退代入: U * X = Y → X = A^{-1}
```

---

## 7. コード由来と貢献者

ngsolve-sparsesolvは複数の研究者のコードを統合・拡張したものである。

### 7.1 由来の対応表

| コンポーネント | 由来 | 原著者 | 元ファイル (JP-MARs/SparseSolv) |
|---|---|---|---|
| CG反復ソルバ | 佐藤 | 佐藤 (tsato) | `MatSolvers_ICCG.cpp` |
| IC分解・前後進代入（逐次） | 佐藤 | 佐藤 (tsato) | `MatSolvers_ICCG.cpp` |
| ABMCオーダリング | 比留間 | 比留間真吾 (Shingo Hiruma) | `MatSolvers_ABMCICCG.cpp` |
| ABMC並列IC分解・並列前後進代入 | 比留間 | 比留間真吾 | `MatSolvers_ABMCICCG.cpp` |
| IC-MRTR反復 | 圓谷→佐藤 | 圓谷友紀 (原理), 佐藤 (実装) | `MatSolvers_ICMRTR.cpp` |
| SGS-MRTR反復（Eisenstat法） | 圓谷→佐藤 | 圓谷友紀 (原理), 佐藤 (実装) | `MatSolvers_SGSMRTR.cpp` |

### 7.2 オリジナルコード

- **圓谷コード**: `S:/NGSolve/01_GitHub/2024_0405_疎行列ソルバ(逐次版)/`
  - IC-MRTRとSGS-MRTRの反復公式のリファレンス実装（C言語、1-based indexing）
  - シフトパラメータ付きIC分解の auto-shift ループ（gamma = 1.05, +0.05刻み）はこのコードに由来

- **JP-MARs/SparseSolv**: `https://github.com/JP-MARs/SparseSolv`
  - 佐藤 (tsato) による疎行列ソルバフレームワーク（C++, Eigenベース）
  - 比留間によるABMC並列化の追加
  - ngsolve-sparsesolvのfork元

### 7.3 ngsolve-sparsesolv での拡張

以下はfork後に追加された機能:

| 機能 | 説明 |
|---|---|
| IC auto-shift | 半正定値行列への自動シフト調整 |
| 対角スケーリング | DAD変換による条件数改善 |
| ゼロ対角処理 | curl-curl行列のゼロ対角DOF対応 |
| Localized IC (Block IC) | Fukuhara 2009のパーティションレベルIC |
| Persistent parallel region | SpinBarrier同期によるレベルスケジューリング高速化 |
| RCM順序付け | ABMC前のReverse Cuthill-McKee帯域縮小 |
| 複素対称対応 | COCG/complex-symmetric CG |
| NGSolve統合 | pybind11モジュール、BaseMatrix互換 |

---

## 8. HYPRE AMS (Auxiliary-space Maxwell Solver)

### 8.1 概要

HYPRE AMS は**実数SPD HCurl curl-curl + mass 系**に対する
**補助空間前処理** [Kolev, Vassilevski 2009] である。
Lawrence Livermore国立研究所の HYPRE ライブラリに実装されている。

**特徴**:
- HCurl有限要素のcurl-curl問題に特化した代数的マルチグリッド
- 離散勾配行列 G と頂点座標のみ必要 (要素行列不要)
- BoomerAMGを内部的に使用した粗空間ソルバー
- メッシュ非依存の反復数 (理論的)

**制限**:
- **実数行列のみ** (複素系には直接使えない → Re/Im splitting)
- 逐次モード (`HYPRE_SEQUENTIAL`) でMPIスタブを使用
- cycle_type=1 (additive) は大規模問題で不安定になる場合がある

**実装ファイル**: `preconditioners/hypre_ams_preconditioner.hpp`

### 8.2 AMS理論

HCurl空間の Helmholtz分解:

```
H(curl) = grad(H^1) + (H(curl) ∩ ker(div))
```

curl-curl行列 K のカーネルは勾配空間 grad(H^1) に含まれる。
AMS はこの空間分解を利用して2つの補助空間で前処理する:

1. **G補正** (gradient correction): `G * A_G^{-1} * G^T` で grad(H^1) 成分を処理
2. **Pi補正** (Nedelec interpolation): `Pi * A_Pi^{-1} * Pi^T` でcurl成分を処理

ここで:
- G: 離散勾配行列 (HCurl -> H1), `fes.CreateGradient()` で取得
- Pi: Nedelec内挿行列, 頂点座標 (x, y, z) から自動構築
- A_G, A_Pi: 補助空間上の粗グリッド行列 (BoomerAMGで近似逆)

### 8.3 Cycle Type

| cycle_type | 名称 | 構造 | 特性 |
|:---:|--------|------|------|
| 1 | Additive | smoother + G + Pi を加法的に合成 | 単純、安定性は問題依存 |
| 2 | Multiplicative | smoother → G → smoother → Pi → smoother | 安定、高コスト |
| 5 | V-cycle (2+3) | cycle 2 と 3 の組合せ | 推奨 |
| 7 | V-cycle (12+13) | 対称版 | 最も安定 |

cycle_type=1 がデフォルトだが、渦電流問題の大規模メッシュでは**不安定**になることがある
(1.44M DOFs で非収束)。cycle_type=7 が安定だが、コストが約2倍。

### 8.4 NGSolve統合 (C++ラッパー)

`HypreAMSPreconditioner` クラスは NGSolve の `BaseMatrix` を継承し、
HYPRE の IJ/ParCSR 形式への変換を行う。

**データフロー**:

```
NGSolve SparseMatrix<double> ──→ HYPRE IJ Matrix ──→ HYPRE ParCSR Matrix
  ConvertSparseMatrix()              │                       │
                                     │                       ▼
NGSolve BitArray (FreeDofs) ─────────┘              HYPRE_AMSSetup()
  Dirichlet行: identity row                                  │
  拘束列: ゼロに設定                                          ▼
                                                    HYPRE_AMSSolve()
NGSolve vector ──→ HYPRE IJVector ──→ ParVector ──→     │
  Mult(): f → u                                     ──→ NGSolve vector
```

**Dirichlet DOFの処理**:
- 行列: 拘束行は identity row (対角=1, 非対角=0)
- RHS: 拘束DOFはゼロに設定
- 出力: 全DOFをコピー (拘束DOFは自動的に0)

NGSolve の `hypre_ams_precond.cpp` と同等の処理。

**離散勾配行列の変換**:
- G は `is_gradient=true` で変換 (FreeDOFsフィルタリングなし)
- 完全な勾配構造を保持する必要がある (拘束列もそのまま)

### 8.5 ビルド設定

HYPRE は `external/hypre/` に配置し、逐次モード (MPI不要) でビルドする:

```bash
cd external/hypre
mkdir build && cd build
cmake ../src -DHYPRE_WITH_MPI=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

CMake オプション:

```cmake
cmake -DSPARSESOLV_USE_HYPRE=ON ..
```

`SPARSESOLV_USE_HYPRE=OFF` (デフォルト) ではHYPRE関連コードは全てコンパイルされない。
`has_hypre()` 関数で実行時に利用可能性を確認できる。

### 8.6 Python API

```python
import sparsesolv_ngsolve

# 実数SPD補助行列を組み立て
fes_real = HCurl(mesh, order=1, nograds=True, dirichlet="dirichlet", complex=False)
u, v = fes_real.TnT()
a_real = BilinearForm(fes_real)
a_real += nu * curl(u) * curl(v) * dx
a_real += eps * nu * u * v * dx
a_real += abs(omega) * sigma * u * v * dx("cond")
a_real.Assemble()

# 離散勾配行列と頂点座標
G, h1_fes = fes_real.CreateGradient()
cx, cy, cz = [], [], []
for i in range(mesh.nv):
    p = mesh.ngmesh.Points()[i + 1]  # 1-indexed
    cx.append(p[0]); cy.append(p[1]); cz.append(p[2])

# HYPRE AMS前処理生成
pre = sparsesolv_ngsolve.HypreAMSPreconditioner(
    a_real.mat, G, fes_real.FreeDofs(),
    cx, cy, cz, cycle_type=1, print_level=0)
```

**複素渦電流問題への適用**: Re/Im splitting で複素系に使う (セクション9参照)。

---

## 9. ComplexHypreAMSPreconditioner (TaskManager並列)

### 9.1 概要

`ComplexHypreAMSPreconditioner` は**複素対称渦電流問題**に対する前処理であり、
HYPRE AMSの2インスタンスをNGSolve TaskManagerで並列実行する。

**特徴**:
- HYPRE AMSの高品質な反復数 (成熟した20年以上の開発)
- TaskManagerによるRe/Im並列化で約1.5x高速化
- HYPRE AMSは非対称 → GMResSolver必須

**実装ファイル**: `preconditioners/hypre_ams_preconditioner.hpp`

### 9.2 定式化

渦電流方程式 (周波数領域、A単独定式化):

```
curl(nu * curl A) + eps*nu*A + jw*sigma*A = -jw*sigma*grad(phi)
```

離散化すると**複素対称系**:

```
A_complex = K + eps*M + jw*sigma*M_cond
```

### 9.3 Re/Im splitting

複素対称系に実数SPD前処理を適用する手法。

**実数SPD補助行列**:

```
A_real = K + eps*M + |w|*sigma*M_cond
```

**適用**: 複素ベクトル x = x_re + j*x_im に対し:

```
y_re = HYPRE_AMS(x_re)    (Re用HYPRE AMSインスタンス)
y_im = HYPRE_AMS(x_im)    (Im用HYPRE AMSインスタンス)
y = y_re + j * y_im        (TaskManagerで並列実行)
```

2つのHYPRE AMSインスタンスが必要な理由:
HYPREの内部ベクトル (`hyp_b_`, `hyp_x_`) はスレッドセーフではないため、
同一インスタンスの並列呼び出しは不可。

### 9.4 ベンチマーク結果 (GMRES, tol=1e-8)

| メッシュ | DOFs | Python (逐次) | C++ TaskManager | 高速化 |
|---------|-----:|---:|---:|---:|
| 2.5T | 155k | 5.40s, 50 it | **3.43s, 50 it** | **1.57x** |
| 5.5T | 331k | 14.98s, 59 it | **10.19s, 59 it** | **1.47x** |
| 20.5T | 1.44M | 103.09s, 75 it | **69.16s, 75 it** | **1.49x** |

反復数は同一 — 数学的に同じ処理、並列化のみ異なる。

### 9.5 Python使用例

```python
import sparsesolv_ngsolve as ssn
from ngsolve.krylovspace import GMResSolver

pre = ssn.ComplexHypreAMSPreconditioner(
    a_real_mat=a_real.mat, grad_mat=G_mat,
    freedofs=fes_real.FreeDofs(),
    coord_x=cx, coord_y=cy, coord_z=cz,
    ndof_complex=fes.ndof, cycle_type=1, print_level=0)

with TaskManager():
    inv = GMResSolver(mat=a.mat, pre=pre, maxiter=500, tol=1e-8)
    gfu.vec.data = inv * f.vec
```

---

## 10. ソルバー比較: ABMC ICCG vs BDDC vs HYPRE AMS

### 10.1 手法の位置づけ

3つの異なる前処理戦略の比較。それぞれ異なるレベルで問題構造を利用する:

| 特性 | ABMC ICCG | BDDC (NGSolve組込み) | HYPRE AMS |
|------|-----------|------|-----------|
| **対象** | 汎用SPD | 汎用SPD (高次向き) | HCurl curl-curl + mass |
| **空間構造の利用** | なし (代数的) | 要素構造 (wirebasket) | Helmholtz分解 (G, Pi) |
| **多段階性** | 1レベル (IC前処理) | 2レベル (粗空間+局所) | 多レベル (AMG階層) |
| **前処理の対称性** | **対称** | **対称** | **非対称** (デフォルト) |
| **Krylovソルバー** | CG / COCR | CG / COCR | **GMRES必須** |
| **メモリ効率** | **最良** | 中 | 大 (GMRES基底 + AMG階層) |
| **セットアップコスト** | 低 | 高 | 中 |
| **反復数スケーリング** | h依存 (悪化) | h非依存 | h準非依存 |

### 10.2 メモリ使用量の比較

**ABMC ICCG** が最もメモリ効率が良い:
- IC(0): A と同じ非ゼロパターン (追加メモリほぼゼロ)
- CG: 約5本の作業ベクトル (O(N))
- ABMC: 順序付けテーブル (O(N) 整数)

**BDDC**:
- 粗空間行列 (wirebasket DOFs の密行列): O(N_wb^2)
- 要素ごとの Schur 補体: O(Σ n_wb^(e)^2)
- CG: 約5本の作業ベクトル

**HYPRE AMS + GMRES**:
- GMRES Krylov基底: m反復 × N DOFs × 16 bytes (complex)
  - 例: 75反復 × 1.44M DOFs × 16B ≈ **1.7 GB**
- HYPRE内部AMG階層: 2つの BoomerAMG (Pi, G)
- 2インスタンス (Re/Im並列) で上記の2倍
- CG比で約10-20倍のメモリ

**トレードオフ**: HYPRE AMS+GMRESはメモリ使用量でABMC ICCGに劣るが、
大規模HCurl問題での反復数の安定性が決定的な利点となる。

### 10.3 なぜ HYPRE AMS が HCurl 渦電流に有効か

ABMC ICCGやBDDC (NGSolve組込み) はHCurl渦電流問題でも動作するが、**問題の物理構造を利用しない**。
HYPRE AMSは3つの構造的優位性を持つ:

#### (a) Helmholtz分解の直接活用

HCurl空間には `u = grad(phi) + z` という直交分解がある。
curl-curl行列 K はカーネル (勾配空間) を持つため、本質的に半正定値であり、
ICCGには特別な処理 (auto-shift) が必要。

AMSはこの分解を**離散勾配行列 G** と **Nedelec内挿行列 Pi** で直接捉える:
- G補正: grad(H1) 成分を H1 AMG で効率処理
- Pi補正: curl-free complement をベクトルH1 AMG で処理

ICCGもBDDCもこの分解を知らず、一般的な疎行列として処理する。

#### (b) 代数的マルチグリッドの多段階補正

IC前処理は行列の近似逆を1レベルで構築するため、
低周波成分 (メッシュスケールの変動) の処理が不十分。
メッシュ細分化で反復数が増加する根本原因。

BDDCは2レベル (局所+粗空間) で改善するが、粗空間は wirebasket DOFs に限定。

AMSは BoomerAMG による**多段階**の粗グリッド補正を行い、
全スケールの成分を効率的に処理する。これが大規模問題でのh非依存収束の鍵。

#### (c) 渦電流問題の正則化活用

渦電流系 `K + jw*sigma*M` では:
- 導電領域: σ項が正則化として機能し、curl-curl単独より条件が改善
- 非導電領域: curl-curl が支配的でカーネルが存在

AMSはこの混合構造を補助行列 `A_real = K + eps*M + |w|*sigma*M_cond` で適切に反映する。
ICCGのauto-shiftは一様なシフトしかできず、領域ごとの最適化が不可能。

### 10.4 HYPRE AMS の対称化と CG 互換性

HYPRE AMS のデフォルト設定では内部 BoomerAMG が hybrid GS (relax_type=3) を使用するため、
前処理全体が非対称となり GMRES が必須。

ただし、全スムーザを対称型に設定すれば CG 互換になる:

| スムーザ設定 | 対称性 | Krylov |
|------------|:---:|--------|
| relax_type=2 (l1-Jacobi) + alpha/beta relax_type=6 | **対称** | CG / COCR |
| relax_type=16 (Chebyshev) + alpha/beta relax_type=16 | **対称** | CG / COCR |
| relax_type=3 (hybrid GS, デフォルト) | 非対称 | GMRES |

対称設定にすれば CG の O(1) メモリで利用可能だが、
反復数が増加する可能性がある。現時点ではこれらのパラメータは未公開
(Step 1 of plan で追加予定)。

### 10.5 推奨選択

| 条件 | 推奨 | 理由 |
|------|------|------|
| H1 低次 (p=1-2) | **ABMC ICCG** | セットアップ最速、メモリ最小 |
| H1/HCurl 高次 (p≥3) | **BDDC** (NGSolve組込み) | h非依存反復、高次で真価 |
| HCurl 渦電流 (大規模 p=1) | **HYPRE AMS+GMRES** | 反復数安定 (1.44M DOFsで75反復) |
| HCurl 渦電流 (中規模) | **BDDC+CG** (NGSolve組込み) | メモリ効率、対称前処理 |
| メモリ制約厳しい | **ABMC ICCG** | CG≈5ベクトル vs GMRES≈mベクトル |

---

## 参考文献

1. C. R. Dohrmann,
   "A Preconditioner for Substructuring Based on Constrained Energy Minimization",
   *SIAM J. Sci. Comput.*, Vol. 25, No. 1, pp. 246–258, 2003.
   [DOI: 10.1137/S1064827502412887](https://doi.org/10.1137/S1064827502412887)

2. J. Mandel, C. R. Dohrmann, R. Tezaur,
   "An Algebraic Theory for Primal and Dual Substructuring Methods
   by Constraints",
   *Appl. Numer. Math.*, Vol. 54, No. 2, pp. 167–193, 2005.
   [DOI: 10.1016/j.apnum.2004.09.022](https://doi.org/10.1016/j.apnum.2004.09.022)

3. J. A. Meijerink, H. A. van der Vorst,
   "An Iterative Solution Method for Linear Systems of Which the
   Coefficient Matrix is a Symmetric M-Matrix",
   *Math. Comp.*, Vol. 31, No. 137, pp. 148–162, 1977.
   [DOI: 10.1090/S0025-5718-1977-0438681-4](https://doi.org/10.1090/S0025-5718-1977-0438681-4)

4. M. R. Hestenes, E. Stiefel,
   "Methods of Conjugate Gradients for Solving Linear Systems",
   *J. Research of the National Bureau of Standards*,
   Vol. 49, No. 6, pp. 409–436, 1952.
   [DOI: 10.6028/jres.049.044](https://doi.org/10.6028/jres.049.044)

5. T. Iwashita, H. Nakashima, Y. Takahashi,
   "Algebraic Block Multi-Color Ordering Method for Parallel
   Multi-Threaded Sparse Triangular Solver in ICCG Method",
   *Proc. IEEE IPDPS*, 2012.
   [DOI: 10.1109/IPDPS.2012.51](https://doi.org/10.1109/IPDPS.2012.51)

6. 圓谷友紀, 三船泰, 岩下武史, 高橋英治,
   "MRTR法に基づく前処理付き反復法の数値実験",
   *電気学会研究会資料*, SA-12-64, 2012.

7. T. Tsuburaya, Y. Okamoto, K. Fujiwara, S. Sato,
   "Improvement of the Preconditioned MRTR Method With Eisenstat's Technique
   in Real Symmetric Sparse Matrices",
   *IEEE Trans. Magnetics*, Vol. 49, No. 5, pp. 1641–1644, 2013.
   [DOI: 10.1109/TMAG.2013.2240283](https://doi.org/10.1109/TMAG.2013.2240283)

8. T. V. Kolev, P. S. Vassilevski,
   "Parallel Auxiliary Space AMG for H(curl) Problems",
   *J. Comput. Math.*, Vol. 27, No. 5, pp. 604–623, 2009.

9. R. D. Falgout, U. M. Yang,
   "hypre: A Library of High Performance Preconditioners",
   *Computational Science -- ICCS 2002*, Lecture Notes in Computer Science,
   Vol. 2331, pp. 632–641, 2002.
   [DOI: 10.1007/3-540-47789-6_66](https://doi.org/10.1007/3-540-47789-6_66)

10. R. Hiptmair, J. Xu,
    "Nodal Auxiliary Space Preconditioning in H(curl) and H(div) Spaces",
    *SIAM J. Numer. Anal.*, Vol. 45, No. 6, pp. 2483–2509, 2007.
    [DOI: 10.1137/060660588](https://doi.org/10.1137/060660588)
