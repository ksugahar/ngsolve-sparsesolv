# Changelog

このプロジェクトの注目すべき変更点はすべてこのファイルに記録されます。

形式は[Keep a Changelog](https://keepachangelog.com/en/1.1.0/)に基づいており、
このプロジェクトは[Semantic Versioning](https://semver.org/spec/v2.0.0.html)に従っています。

## [2.3.0] - 2026-02-28

### 追加
- Hiruma渦電流問題のメッシュ例題 (6メッシュ、Gmsh v2形式、Git LFS)
- `examples/hiruma/eddy_current.py` — A-Phi定式化の渦電流解析
- `examples/hiruma/bench_parallel.py` — BDDC vs ABMC+ICCG 並列スケーリングベンチマーク

### 改善
- CG反復のカーネル融合 — メモリトラフィック約20%削減
  - SpMV + dot(p, Ap) を1パスに融合 (p[], Ap[] の再読込排除)
  - AXPY + 残差ノルム計算を1パスに融合 (r[] の再読込排除)
  - 反復あたりのカーネル起動を7回から5回に削減
- ABMC並列IC分解でauto_shift対応 (アトミックフラグによるリスタート)
  - 従来: auto_shift有効時はABMC並列パスが使えず逐次IC分解にフォールバック
  - 改善: 並列分解中にbreakdownを検出→シフト増加→全体リスタートで完全並列化
- Hiruma HCurl p=1 渦電流問題で並列スケーリング 1.5x → 2.85x (8コア)

## [2.2.0] - 2026-02-25

### 追加
- BDDC前処理: MKL PARDISO直接統合による粗解法
- ABMC並列IC分解 (色ごとのparallel_for)
- レベルスケジューリング三角解法の持続的並列領域 (SpinBarrier)
- ABMCプロパティのPython API公開 (`use_abmc`, `abmc_block_size`, etc.)

### 変更
- BDDC粗解法をNGSolve SparseCholeskyからMKL PARDISOに変更
- ヘッダオンリーインストールからngsolve/ヘッダを除外

## [2.1.0] - 2026-02-21

### 追加
- BDDC (Balancing Domain Decomposition by Constraints) 前処理行列
  - BilinearForm からの要素ごとの構成
  - NGSolve CouplingType による Wirebasket/interface DOF 分類
  - 粗解法: SparseCholesky (デフォルト)、PARDISO
  - メッシュ非依存収束 (すべてのテスト問題で CG 2 反復)
- `docs/` の日本語ドキュメント (アーキテクチャ、アルゴリズム、API リファレンス、チュートリアル)
- すべてのソースファイルに MPL 2.0 ライセンスヘッダ
- CONTRIBUTING.md

### 変更
- ブロック消去 BDDC アプローチを削除 (要素ごとのアプローチに置き換え)
- 未使用の前処理行列を削除 (Identity、Jacobi、ILU)
- MRTRSolver を削除 (SGS-MRTR は自己完結)
- IC 前処理行列の重複する `reorder_matrix()` コードを統合
- 共有 `BuildSparseMatrixView()` を抽出してコード重複を排除
- pybind11 型登録を簡潔化 (ファクトリが公開 API)

### 修正
- DenseMatrix::invert() 置換行列の構成 (P^T → P)
  - 自明でないピボッティングの HCurl 問題で BDDC 失敗を引き起こす
- 複素数内積: 複素対称 FEM 行列の非共役内積
  - 渦電流の発散を修正 (5000 反復 → 58 反復)
- SGS-MRTR 複素数比較: 閾値チェックに `std::real(denom)` を使用

## [2.0.0] - 2026-02-19

### 追加
- NGSolve とは独立したスタンドアロン pybind11 モジュール (`sparsesolv_ngsolve.pyd`)
- 自動ディスパッチ機能を備えたファクトリ関数 (`mat.IsComplex()` による実数/複素数の自動判定)
- 半正定値行列の自動シフト IC 分解 (curl-curl)
- 条件数改善のための対角スケーリング
- ABMC (Algebraic Block Multi-Color) 順序付け (並列三角解法用)
- RCM 帯域幅削減 (オプション、ABMC と組み合わせ)
- 複素数対応 (`std::complex<double>`) を完全サポート
- 発散検出と最良結果の復帰
- 残差履歴記録
- レベルスケジュール三角解法の永続並列領域

### 変更
- ヘッダオンリー C++17 ライブラリとして再構成
- `SPARSESOLV_USE_NGSOLVE_TASKMANAGER` コンパイルフラグによる NGSolve 統合
- 並列化の抽象化: TaskManager / OpenMP / シリアル ディスパッチ

## [1.0.0] - 2026-02-01

[JP-MARs/SparseSolv](https://github.com/JP-MARs/SparseSolv) からの初回フォーク。

### 上流からの機能
- シフトパラメータ付き IC(0) 前処理行列
- SGS 前処理行列
- CG ソルバー
- 分割公式による SGS-MRTR ソルバー
