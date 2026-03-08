# アーキテクチャとソースコード解説

## ディレクトリ構成

```
ngsolve-sparsesolv/
├── include/sparsesolv/           # ヘッダオンリーライブラリ本体
│   ├── sparsesolv.hpp            # メインヘッダ (全コンポーネントをインクルード)
│   ├── core/                     # 基盤コンポーネント
│   │   ├── types.hpp             # 型定義 (index_t, complex_t)
│   │   ├── constants.hpp         # 数値定数 (許容値、閾値)
│   │   ├── solver_config.hpp     # SolverConfig 構造体
│   │   ├── sparse_matrix_view.hpp # CSR行列の非所有ビュー (ゼロコピー)
│   │   ├── sparse_matrix_coo.hpp # COO形式 (組立用)
│   │   ├── sparse_matrix_csr.hpp # CSR形式 (格納用)
│   │   ├── dense_matrix.hpp      # 密行列 + LU逆行列
│   │   ├── preconditioner.hpp    # 前処理基底クラス (テンプレート)
│   │   ├── parallel.hpp          # 並列化抽象レイヤ (TaskManager/OpenMP/serial)
│   │   ├── level_schedule.hpp    # レベルスケジューリング (三角解法並列化)
│   │   ├── abmc_ordering.hpp     # ABMC順序付け (三角解法並列化)
│   │   └── rcm_ordering.hpp      # RCM帯域縮小順序付け
│   ├── preconditioners/          # 前処理実装
│   │   ├── ic_preconditioner.hpp # 不完全コレスキー (IC) 分解
│   │   ├── sgs_preconditioner.hpp # 対称ガウス・ザイデル (SGS)
│   │   └── hypre_ams_preconditioner.hpp # HYPRE AMS (条件付: SPARSESOLV_USE_HYPRE)
│   ├── solvers/                  # 反復法ソルバー
│   │   ├── iterative_solver.hpp  # 反復法基底クラス
│   │   ├── cg_solver.hpp         # 共役勾配法 (CG)
│   │   └── sgs_mrtr_solver.hpp   # SGS-MRTR (split formula)
│   └── ngsolve/                  # NGSolve統合レイヤ
│       ├── sparsesolv_precond.hpp # BaseMatrix wrappers (IC, SGS, BDDC, Solver)
│       └── sparsesolv_python_export.hpp # pybind11バインディング + factory関数
├── ngsolve/
│   └── python_module.cpp         # pybind11モジュールエントリポイント
├── tests/
│   └── test_sparsesolv.py        # ソルバー・前処理テスト
├── docs/                         # ドキュメント (本フォルダ)
├── external/
│   └── hypre/                    # HYPRE ライブラリ (SPARSESOLV_USE_HYPRE=ON時)
├── CMakeLists.txt                # ビルド設定
└── LICENSE
```

## 設計原則

### ヘッダオンリー

全てのC++コードは `.hpp` ヘッダファイルに実装されている。
コンパイル時に全コードがインライン展開されるため、リンク問題がなく配布が容易。

### テンプレート設計

全アルゴリズムクラスは `template<typename Scalar>` でパラメータ化されている:

```cpp
template<typename Scalar = double>
class ICPreconditioner : public Preconditioner<Scalar> { ... };
```

`Scalar` には `double` と `std::complex<double>` が使用される。
これにより実数問題と複素数問題 (渦電流等) を単一コードベースで処理する。

### 並列化抽象レイヤ

`core/parallel.hpp` がコンパイル時にバックエンドを切り替える:

| ビルド設定 | バックエンド | 用途 |
|---|---|---|
| `SPARSESOLV_USE_NGSOLVE_TASKMANAGER` | NGSolve TaskManager | NGSolve統合時 |
| `_OPENMP` | OpenMP | スタンドアロン |
| (どちらもなし) | シリアル実行 | デバッグ |

主要API:

```cpp
sparsesolv::parallel_for(n, [&](index_t i) { ... });
sparsesolv::parallel_reduce(n, init, [&](index_t i) -> T { ... });
sparsesolv::get_num_threads();
```

## ヘッダ依存関係

```
sparsesolv.hpp (メインヘッダ)
├── core/types.hpp
├── core/constants.hpp
├── core/solver_config.hpp
├── core/sparse_matrix_view.hpp ← core/parallel.hpp
├── core/preconditioner.hpp ← core/sparse_matrix_view.hpp
├── core/abmc_ordering.hpp
├── preconditioners/ic_preconditioner.hpp
│   ← core/preconditioner.hpp, core/level_schedule.hpp,
│      core/abmc_ordering.hpp, core/rcm_ordering.hpp
├── preconditioners/sgs_preconditioner.hpp
│   ← core/preconditioner.hpp
├── solvers/iterative_solver.hpp
│   ← core/preconditioner.hpp, core/sparse_matrix_view.hpp
├── solvers/cg_solver.hpp
│   ← solvers/iterative_solver.hpp
└── solvers/sgs_mrtr_solver.hpp
    ← solvers/iterative_solver.hpp, core/level_schedule.hpp
```

NGSolve統合レイヤ (NGSolveビルド時のみ):

```
ngsolve/sparsesolv_precond.hpp   ← sparsesolv.hpp + NGSolve headers
ngsolve/sparsesolv_python_export.hpp ← sparsesolv_precond.hpp + pybind11
```

## NGSolve統合レイヤ

### SparseMatrixView (ゼロコピーラッパー)

NGSolveの `SparseMatrix<SCAL>` はCSRライクな内部構造を持つ。
`SparseSolvPrecondBase::prepare_matrix_view()` が、データコピーなしに
SparseSolvの `SparseMatrixView<SCAL>` に変換する。

FreeDofs処理: 拘束DOFの行は単位行列 (対角=1, 非対角=0) に置換。

```cpp
// sparsesolv_precond.hpp: SparseSolvPrecondBase::prepare_matrix_view()
if (!freedofs_->Test(i)) {
    modified_values_[k] = (j == i) ? SCAL(1) : SCAL(0);  // identity row
} else if (!freedofs_->Test(j)) {
    modified_values_[k] = SCAL(0);  // zero coupling to constrained DOF
}
```

### SparseSolvPrecondBase (BaseMatrix wrapper)

全前処理クラスの基底。NGSolveの `BaseMatrix` を継承し、
`Mult()` / `MultAdd()` を実装して NGSolve の `CGSolver` と互換にする。

```
SparseSolvPrecondBase<SCAL>  (抽象基底)
├── SparseSolvICPreconditioner<SCAL>    → ICPreconditioner<SCAL>
├── SparseSolvSGSPreconditioner<SCAL>   → SGSPreconditioner<SCAL>
└── SparseSolvSolver<SCAL>              → ICCG/SGSMRTR/CG

HypreAMSPreconditioner  (BaseMatrix直接継承, 実数のみ)
  → HYPRE AMS (条件付: SPARSESOLV_USE_HYPRE)
```

### Python factory関数 (auto-dispatch)

`sparsesolv_python_export.hpp` の factory関数は、行列の型を自動判定して
適切なテンプレートインスタンスを生成する:

```cpp
// ICPreconditioner factory
m.def("ICPreconditioner", [](shared_ptr<BaseMatrix> mat, ...) {
    if (mat->IsComplex()) {
        // → SparseSolvICPreconditioner<Complex>
    } else {
        // → SparseSolvICPreconditioner<double>
    }
});
```

BDDCアルゴリズムの詳細は [algorithms.md](algorithms.md) を参照。
NGSolveの組込みBDDC (`a.mat.Inverse(fes.FreeDofs(), inverse="bddc")`) を使用する。
