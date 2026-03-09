# Developer Guide

## Build Instructions

### Prerequisites

- NGSolve (source build or pip install) + CMake configuration files
- CMake 3.16 or later
- C++17-compatible compiler (MSVC 2022, GCC 10+, Clang 10+)
- pybind11 (automatically fetched by CMake)

### Build Commands

```bash
git clone https://github.com/ksugahar/ngsolve-sparsesolv.git
cd ngsolve-sparsesolv
mkdir build && cd build

cmake .. -DSPARSESOLV_BUILD_NGSOLVE=ON \
         -DNGSOLVE_INSTALL_DIR=/path/to/ngsolve/install/cmake

cmake --build . --config Release
```

**When using Intel MKL** (BLAS used internally by CompactAMG/AMS):

```bash
cmake .. -DSPARSESOLV_BUILD_NGSOLVE=ON \
         -DNGSOLVE_INSTALL_DIR=/path/to/ngsolve/install/cmake \
         -DCMAKE_PREFIX_PATH="C:/Program Files (x86)/Intel/oneAPI/mkl/latest"
```

### Installation

```bash
# Copy to NGSolve site-packages
SITE_PACKAGES=$(python -c "import ngsolve, pathlib; print(pathlib.Path(ngsolve.__file__).parent.parent)")
mkdir -p "$SITE_PACKAGES/sparsesolv_ngsolve"
cp build/Release/sparsesolv_ngsolve*.pyd "$SITE_PACKAGES/sparsesolv_ngsolve/"
echo "from .sparsesolv_ngsolve import *" > "$SITE_PACKAGES/sparsesolv_ngsolve/__init__.py"
```

---

## Running Tests

```bash
python -m pytest tests/test_sparsesolv.py -v --tb=short
```

Test structure:
- `test_sparsesolv.py`: Solver and preconditioner tests (46 cases)
  - ICCG, SGSMRTR, CG, IC, SGS for various problems
  - 2D/3D, H1/VectorH1/HCurl, real/complex
  - ABMC ordering, diagonal scaling, auto-shift

---

## Past Bugs and Lessons Learned

### 1. DenseMatrix P^T Bug

**Problem**: In the dense LU inverse, the permutation matrix was constructed as P^T instead of P.

```cpp
// Wrong (P^T):
inv(piv[j], j) = Scalar(1);

// Correct (P):
inv(k, piv[k]) = Scalar(1);
```

**Lesson**: In PA = LU, P is the matrix that "moves row k to row piv[k]".
The correct construction is P[k, piv[k]] = 1.

### 2. Complex Inner Product (Non-conjugate vs Conjugate)

**Problem**: The CG solver was using the conjugate inner product `std::conj(a[i]) * b[i]`.

FEM matrices are **complex-symmetric** (A^T = A), not Hermitian (A^H = A).
Example: the eddy current equation curl-curl + i*sigma*mass.

```cpp
// Wrong (for Hermitian matrices):
sum += std::conj(a[i]) * b[i];

// Correct (for complex-symmetric matrices):
sum += a[i] * b[i];
```

**Symptom**: Divergence after 5000 iterations on eddy current problems.
After the fix: convergence in 58 iterations.

**Lesson**: Always use the non-conjugate inner product for complex FEM problems.
This corresponds to NGSolve's `CGSolver(conjugate=False)`.

### 3. SGS-MRTR Complex Comparison

**Problem**: The zeta calculation in SGS-MRTR used `denom >= 0`, but
`std::complex<double>` does not have a `>=` operator.

```cpp
// Fixed:
denom = (std::real(denom) >= 0) ? ... : ...;
```

---

## NGSolve API Notes

### AutoVector

NGSolve's `BaseVector::CreateVector()` returns an `AutoVector`.
It has shared_ptr-like semantics but with its own move/copy conventions.

### CalcElementMatrix

Computing element matrices requires a `LocalHeap`:

```cpp
LocalHeap lh(10000000, "name", true);  // mult_by_threads=true

IterateElements(*fes, VOL, lh,
    [&](FESpace::Element el, LocalHeap& lh_thread) {
        auto& fe = el.GetFE();
        auto& trafo = el.GetTrafo();
        FlatMatrix<SCAL> elmat(ndof, ndof, lh_thread);
        integrator->CalcElementMatrix(fe, trafo, elmat, lh_thread);
    });
```

### IsRegularDof

An element's DOF list may contain invalid DOFs (< 0).
Use `IsRegularDof(dnum)` to filter them:

```cpp
auto dnums = el.GetDofs();
for (int i = 0; i < dnums.Size(); ++i) {
    if (IsRegularDof(dnums[i])) {
        // Valid DOF
    }
}
```

---

## References

1. J. A. Meijerink, H. A. van der Vorst,
   "An Iterative Solution Method for Linear Systems of Which the
   Coefficient Matrix is a Symmetric M-Matrix",
   *Math. Comp.*, Vol. 31, No. 137, pp. 148-162, 1977.
   [DOI: 10.1090/S0025-5718-1977-0438681-4](https://doi.org/10.1090/S0025-5718-1977-0438681-4)

2. M. R. Hestenes, E. Stiefel,
   "Methods of Conjugate Gradients for Solving Linear Systems",
   *J. Research of the National Bureau of Standards*,
   Vol. 49, No. 6, pp. 409-436, 1952.
   [DOI: 10.6028/jres.049.044](https://doi.org/10.6028/jres.049.044)

3. T. Iwashita, H. Nakashima, Y. Takahashi,
   "Algebraic Block Multi-Color Ordering Method for Parallel Multi-Threaded
   Sparse Triangular Solver in ICCG Method",
   *Proc. IEEE IPDPS*, 2012.
   [DOI: 10.1109/IPDPS.2012.51](https://doi.org/10.1109/IPDPS.2012.51)

4. E. Cuthill, J. McKee,
   "Reducing the Bandwidth of Sparse Symmetric Matrices",
   *Proc. 24th Nat. Conf. ACM*, pp. 157-172, 1969.
   [DOI: 10.1145/800195.805928](https://doi.org/10.1145/800195.805928)

5. 圓谷友紀 (T. Tsuburaya), 三船泰 (Y. Mifune), 岩下武史 (T. Iwashita), 高橋英治 (E. Takahashi),
   "MRTR法に基づく前処理付き反復法の数値実験"
   (Numerical experiments on preconditioned iterative methods based on the MRTR method),
   *電気学会研究会資料* (IEEJ Technical Meeting Papers), SA-12-64, 2012.

6. T. Tsuburaya, Y. Okamoto, K. Fujiwara, S. Sato,
   "Improvement of the Preconditioned MRTR Method With Eisenstat's Technique
   in Real Symmetric Sparse Matrices",
   *IEEE Trans. Magnetics*, Vol. 49, No. 5, pp. 1641-1644, 2013.
   [DOI: 10.1109/TMAG.2013.2240283](https://doi.org/10.1109/TMAG.2013.2240283)

7. 圓谷友紀 (T. Tsuburaya),
   "大規模電磁界問題の有限要素解析のための反復法の開発"
   (Development of iterative methods for finite element analysis of large-scale electromagnetic field problems),
   Doctoral dissertation, 宇都宮大学 (Utsunomiya University), 2016.

8. JP-MARs/SparseSolv,
    https://github.com/JP-MARs/SparseSolv
