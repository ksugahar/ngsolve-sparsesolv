# ngsolve-sparsesolv ドキュメント

NGSolve向けヘッダオンリー反復法ソルバーライブラリ。
BDDC前処理を中心に、IC分解・SGS-MRTR・CG法を提供する。

## 目次

| ドキュメント | 内容 |
|---|---|
| [architecture.md](architecture.md) | ソースコード構成とアーキテクチャ |
| [algorithms.md](algorithms.md) | アルゴリズム解説 (BDDC, IC, SGS-MRTR, CG) |
| [api_reference.md](api_reference.md) | Python APIリファレンス |
| [tutorials.md](tutorials.md) | 実践チュートリアル (コピペ実行可能) |
| [development.md](development.md) | ビルド・テスト・開発者向け情報 |

## 概要

SparseSolvは有限要素法 (FEM) の大規模疎行列連立方程式を解くための
反復法ライブラリである。[JP-MARs/SparseSolv](https://github.com/JP-MARs/SparseSolv)
のforkであり、NGSolveとの統合に特化している。

### 主な機能

- **BDDC前処理**: 要素行列から領域分割前処理を構築。メッシュ非依存の反復回数
- **IC分解 (不完全コレスキー)**: auto-shift対応。curl-curl半正定値行列にも対応
- **SGS-MRTR**: DAD変換による対角スケーリング内蔵の反復法
- **CG法**: 前処理付き共役勾配法。複素対称系 (非共役内積) 対応
- **ABMC順序付け**: 三角解法の並列化

### 対応する問題の種類

| 問題 | 有限要素空間 | 推奨ソルバー |
|------|------------|------------|
| Poisson方程式 | H1 | BDDC+CG or ICCG |
| 弾性体問題 | VectorH1 | BDDC+CG |
| Curl-curl (実数) | HCurl (nograds=True) | BDDC+CG or Shifted-ICCG |
| 渦電流 (複素数) | HCurl (complex) | BDDC+CG (conjugate=False) |
