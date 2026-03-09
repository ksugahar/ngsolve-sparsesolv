# Algorithm Reference

## 1. BDDC (Balancing Domain Decomposition by Constraints)

### 1.1 Overview

BDDC is a **domain decomposition preconditioner** for large-scale finite element problems [Dohrmann 2003].
It treats each element of the mesh as a "subdomain" and handles
inter-subdomain coupling through a **coarse space**.
See [Mandel, Dohrmann, Tezaur 2005] for the algebraic formulation.

**Features**:
- Iteration count is nearly independent of mesh size *h* (scalable)
- Can be constructed directly from element matrices (no assembled global matrix required)

**Note**: The custom BDDC implementation in SparseSolv has been removed. The description below is retained as an algorithm reference.
Use NGSolve's built-in BDDC (`a.mat.Inverse(fes.FreeDofs(), inverse="bddc")`) instead.

### 1.2 DOF Classification

Degrees of freedom (DOFs) in the finite element space are classified into two categories:

| Category | Name | Role | NGSolve CouplingType |
|----------|------|------|---------------------|
| **Wirebasket** | Coarse space DOFs | Vertex and edge DOFs. Globally coupled | `WIREBASKET_DOF` |
| **Interface** | Local DOFs | Face and interior DOFs. Eliminated within each subdomain | All others |

For higher-order finite elements (order >= 2), wirebasket DOFs are a small number of low-order DOFs,
while interface DOFs constitute the majority. BDDC's efficiency depends on this hierarchy.

### 1.3 Element-Level Schur Complement

The stiffness matrix of each element *e* is partitioned into wirebasket (w) and interface (i) blocks:

```
K^(e) = | K_ww  K_wi |
        | K_iw  K_ii |
```

**Schur complement** (equivalent matrix on the wirebasket):

```
S^(e) = K_ww - K_wi * K_ii^{-1} * K_iw
```

**Harmonic extension** (recover interface values from wirebasket values):

```
he^(e) = -K_ii^{-1} * K_iw
```

Pseudocode:
```
K_ii_inv = LU_factorize(K_ii)   // PA=LU with partial pivoting
he = -K_ii_inv * K_iw            // harmonic extension
S = K_ww + K_wi * he             // Schur complement
```

### 1.4 Weighting

To correctly distribute contributions from shared DOFs (DOFs belonging to multiple elements),
**weighting (averaging)** is applied.

Weight for interface DOF *k* on element *e*:

```
w_k^(e) = |K_ii^(e)(k,k)|   (absolute value of the K_ii diagonal entry)
```

Weight normalization: sum the contributions from all elements and take the reciprocal:

```
W_k = 1 / Sigma_e w_k^(e)
```

These normalized weights are used to scale the matrix entries of the
harmonic extension, inner solve, and harmonic extension transpose.


### 1.5 Coarse Space Solver

The **wirebasket matrix** S_global, assembled from the Schur complements of all elements,
is solved using a direct method.

A direct solver (SparseCholesky, MKL PARDISO, etc.) is used.
Since the number of wirebasket DOFs is small compared to the total DOFs,
the cost of the direct solve is acceptable.
In NGSolve's built-in BDDC, SparseCholesky or PARDISO are available as coarse space solvers.

### 1.6 Apply Algorithm (5 Steps)

The BDDC preconditioner application consists of the following 5 steps.
Input *r* is the residual vector; output *y* is the preconditioned vector.

```
Step 1: y = r                              (copy)
Step 2: y += H^T * r                       (harmonic extension transpose)
Step 3: y_wb = S^{-1} * y_wb              (coarse space solve on wirebasket component)
Step 4: y += K_inner * r                   (inner solve: interface component)
Step 5: y = y + H * y                      (harmonic extension)
```

Where:
- H: harmonic extension matrix (assembled from all element he^(e) with weighting)
- H^T: its transpose
- K_inner: assembled from all element K_ii^{-1} with weighting

Usage example in NGSolve:
```python
# NGSolve built-in BDDC
inv = a.mat.Inverse(fes.FreeDofs(), inverse="bddc")
gfu.vec.data = inv * f.vec
```

---

## 2. IC Factorization (Incomplete Cholesky Factorization)

### 2.1 Overview

Computes an approximate factorization of a symmetric positive definite matrix *A* [Meijerink, van der Vorst 1977]:

```
A ~ L * D * L^T
```

- *L*: lower triangular matrix (same sparsity pattern as A)
- *D*: diagonal matrix

Preconditioner application: solve y = (LDL^T)^{-1} * x via three stages: forward substitution, diagonal scaling, and backward substitution.

**Implementation file**: `preconditioners/ic_preconditioner.hpp`

### 2.2 Shift Parameter

A **shift** is added to the diagonal of the IC factorization for stabilization:

```
d_i = alpha * a_ii - Sigma_{k<i} l_ik^2 * d_k^{-1}
```

where alpha is the shift parameter (default: 1.05).

- alpha = 1.0: Standard IC(0). May break down for indefinite matrices
- alpha = 1.05: Default. Slightly more stable
- alpha = 1.1 to 1.2: For difficult problems

### 2.3 Auto-shift

For semi-definite matrices (curl-curl problems), diagonal entries can become
too small and cause the IC factorization to break down. Auto-shift automatically
detects this and retries the factorization with an increased shift:

```cpp
// ic_preconditioner.hpp: compute_ic_factorization()
if (abs_s < config_.min_diagonal_threshold && abs_orig > 0.0) {
    shift += increment;     // increase shift
    increment *= 2;         // exponential backoff
    restart = true;          // restart factorization
}
```

### 2.4 Diagonal Scaling

Diagonal scaling can be applied to improve the condition number of the matrix:

```
scaling[i] = 1 / sqrt(|A[i,i]|)
A_scaled[i,j] = scaling[i] * A[i,j] * scaling[j]
```

Implementation: `ICPreconditioner::compute_scaling_factors()`, `apply_scaling_to_L()`

### 2.5 Parallelization of Triangular Solves

Triangular solve application (forward and backward substitution) is inherently sequential,
but can be partially parallelized through data dependency analysis.

Two approaches are provided:

| Approach | File | Parallelism Granularity | Characteristics |
|----------|------|------------------------|-----------------|
| Level scheduling | `level_schedule.hpp` | Row-level | Simple, low overhead |
| ABMC ordering | `abmc_ordering.hpp` | Block-level | Higher parallelism |

See Section 4 (ABMC Ordering) for details.

---

## 3. SGS-MRTR

### 3.1 Overview

SGS-MRTR is a solver that incorporates a **Symmetric Gauss-Seidel (SGS)** preconditioner
into the **MRTR iterative method**. Using a split formula, the forward part *L* and
backward part *L^T* of the preconditioner are applied separately.

**Features**:
- No IC factorization required (no factorization cost)
- DAD transformation (diagonal scaling) is built in
- Supports complex symmetric matrices (with limited accuracy)

**Implementation file**: `solvers/sgs_mrtr_solver.hpp`

**References**:
- T. Tsuburaya, Y. Mifune, T. Iwashita, E. Takahashi,
  "Numerical Experiments on Preconditioned Iterative Methods Based on the MRTR Method",
  *IEEJ Technical Meeting on Static Apparatus*, SA-12-64, 2012.
- T. Tsuburaya, Y. Okamoto, K. Fujiwara, S. Sato,
  "Improvement of the Preconditioned MRTR Method With Eisenstat's Technique
  in Real Symmetric Sparse Matrices",
  *IEEE Trans. Magnetics*, Vol. 49, No. 5, pp. 1641-1644, 2013.
  [DOI: 10.1109/TMAG.2013.2240283](https://doi.org/10.1109/TMAG.2013.2240283)

### 3.2 DAD Transformation

Diagonal scaling of matrix *A*:

```
D = diag(|A|)^{-1/2}
A' = D * A * D
b' = D * b
x  = D * x'
```

After scaling, all diagonal entries of A' are close to 1,
which improves the condition number.

### 3.3 Split Formula

The forward part L and backward part L^T of SGS are applied separately.
At each step of the MRTR iteration:

```
rd = L^{-1} * r          (forward solve)
u  = L^{-T} * rd         (backward solve)
ARd = u + L^{-1}(rd - u) (approximate M^{-1}Ard)
```

### 3.4 MRTR Iteration

MRTR is a minimum residual type iterative method that updates the solution using a two-term recurrence:

```
p_{k+1} = u_k + (eta_k zeta_{k-1} / zeta_k) p_k
x_{k+1} = x_k + zeta_k p_k
```

where zeta_k and eta_k are optimization parameters.

### 3.5 CSR Diagonal Position Assumption

The forward and backward substitutions in SGS-MRTR assume that the **diagonal element
exists at the row index position** in each row of the CSR matrix. NGSolve's `SparseMatrix`
satisfies this condition, but this is not guaranteed for arbitrary CSR matrices.

Specifically, the column indices `col_idx[row_ptr[i]:row_ptr[i+1]]` for row *i*
must be sorted, and the diagonal entry `A[i,i]` must exist.

### 3.6 Notes on Complex Number Support

When division occurs in the zeta_k computation of SGS-MRTR, `std::real()` is used for complex comparisons:

```cpp
// sgs_mrtr_solver.hpp
if (std::abs(denom) < constants::DENOMINATOR_BREAKDOWN) {
    denom = (std::real(denom) >= 0) ? ... : ...;
}
```

For complex symmetric matrix (A^T = A) eddy current problems, the DAD transformation
may not be optimal, and convergence accuracy may be limited to around 5%.
In such cases, ICCG+ABMC, COCR, or Compact AMS+COCR are recommended.

---

## 4. ABMC Ordering (Algebraic Block Multi-Color Ordering)

### 4.1 Overview

A row ordering algorithm for **parallelizing** triangular solves (forward and backward substitution).
It addresses the issue of insufficient parallelism that level scheduling exhibits for FEM matrices.

**Implementation file**: `core/abmc_ordering.hpp`

**References**:
- T. Iwashita, H. Nakashima, Y. Takahashi,
  "Algebraic Block Multi-Color Ordering Method for Parallel
  Multi-Threaded Sparse Triangular Solver in ICCG Method",
  *Proc. IEEE IPDPS*, 2012.

### 4.2 Algorithm

ABMC consists of two stages:

#### Stage 1: Algebraic Blocking (BFS Aggregation)

Nearby rows are grouped into blocks via BFS (breadth-first search) on the matrix graph.

```
Input: Matrix graph G, block size B
Output: Block assignment block[i] (row i -> block ID)

1. Start BFS from an unassigned row seed
2. Aggregate B rows during BFS traversal to form a block
3. Repeat until all rows are assigned
```

Block size `B` (default: 4) is determined by the balance between cache efficiency
and the fact that rows within a block are processed sequentially.

#### Stage 2: Multi-Coloring (Greedy Coloring)

The block adjacency graph (with edges where lower-triangular dependencies exist
between blocks) is colored using a greedy algorithm.

```
Input: Block adjacency graph, target number of colors C
Output: Block color color[b] (block b -> color ID)

1. Build the block dependency adjacency graph
2. Greedy coloring: assign each block the smallest unused color
```

Target number of colors `C` (default: 4) controls the parallelism granularity.

#### Coloring Algorithm Details

Coloring is performed in two steps:

1. **Determine minimum number of colors**: `num_colors = max(target_colors, 1 + max_lower_triangular_degree)`
2. **Forbidden-Color-Set method**: For each block, mark adjacent lower-triangular colors in a forbidden array and assign the smallest unused color. If all colors conflict, add a new color.

`target_colors` is a **lower bound**, not a guarantee. The actual number of colors depends on the structure of the block graph.

#### Known Limitations

- For complex meshes (e.g., helical coils), the number of colors may significantly exceed `target_colors`
- BFS blocking depends on the locality of the matrix graph. Matrices with large bandwidth tend to have dense inter-block dependencies, increasing the number of colors

### 4.3 Parallel Execution of Triangular Solves

After ABMC ordering, the triangular solve is executed in three hierarchical levels:

```
for color c = 0 to C-1:               <- sequential (inter-color dependency)
    parallel_for block b in color c:   <- parallel (same-color blocks are independent)
        for row i in block b:          <- sequential (intra-block dependency)
            process one row of forward substitution
```

**Comparison with level scheduling**:

| Characteristic | Level Scheduling | ABMC |
|----------------|-----------------|------|
| Parallelism granularity | Row-level | Block-level |
| Number of synchronizations | Number of levels (hundreds) | Number of colors (~4) |
| Parallelism for FEM | Low (deep dependency chains) | High |
| Overhead | Low | Matrix reordering cost |

### 4.4 Persistent Parallel Region (Level Scheduling Acceleration)

In standard level scheduling, `parallel_for` is called at each level, resulting in
thread pool startup overhead occurring hundreds of times.

`forward_substitution_persistent()` / `backward_substitution_persistent()` process
all levels within a single `parallel_for(nthreads, ...)`, using spin barriers for
inter-level synchronization:

```cpp
// ic_preconditioner.hpp
parallel_for(nthreads, [&](index_t thread_id) {
    for (int lev = 0; lev < num_levels; ++lev) {
        // each thread processes its assigned rows
        barrier.wait();  // inter-level synchronization
    }
});
```

### 4.5 Combination with RCM Ordering

Applying RCM (Reverse Cuthill-McKee) bandwidth reduction ordering before ABMC
reduces the matrix bandwidth and improves cache efficiency for SpMV (sparse matrix-vector product).

Three execution paths are available:

| Path | SpMV Matrix | Preconditioner | Configuration |
|------|-------------|----------------|---------------|
| Standard | Original matrix | Triangular solve with ABMC ordering | `use_abmc=True` |
| RCM+ABMC | RCM-ordered matrix | Composite RCM->ABMC ordering | `use_abmc=True, abmc_use_rcm=True` |
| ABMC full space | ABMC-ordered matrix | No ordering (direct) | `abmc_reorder_spmv=True` |

### 4.6 Performance Characteristics

The comparison baseline for ABMC is **level scheduling with persistent parallel region**
(not sequential level scheduling).

Measured results for 3D HCurl curl-curl (order=2, 8 threads):

| DOFs | Level Sched. | ABMC (best) | Speedup |
|------|-------------|-------------|---------|
| 11K | 0.044s | 0.051s | 0.86x |
| 27K | 0.158s | 0.154s | 1.02x |
| 82K | 0.763s | 0.650s | 1.17x |
| 186K | 2.505s | 1.962s | 1.28x |

**Break-even point: approximately 25K-30K DOFs** (8 threads). For small problems,
ABMC's setup cost and vector permutation overhead outweigh the speedup of the triangular solve.

Performance also depends on problem structure; for a toroidal coil (148K DOFs, large bandwidth),
a 1.8x speedup was achieved ([02_performance_comparison.ipynb](02_performance_comparison.ipynb)).

See Section 7 of [abmc_implementation_details.md](abmc_implementation_details.md) for details.

---

## 5. CG Method (Conjugate Gradient Method)

### 5.1 Overview

**Preconditioned Conjugate Gradient (PCG)** method for symmetric positive definite (SPD) matrices [Hestenes, Stiefel 1952].

**Implementation file**: `solvers/cg_solver.hpp`

### 5.2 Algorithm

```
r_0 = b - A*x_0
z_0 = M^{-1} * r_0
p_0 = z_0

for k = 0, 1, 2, ...:
    alpha_k = (r_k, z_k) / (p_k, A*p_k)
    x_{k+1} = x_k + alpha_k * p_k
    r_{k+1} = r_k - alpha_k * A*p_k
    z_{k+1} = M^{-1} * r_{k+1}
    beta_k = (r_{k+1}, z_{k+1}) / (r_k, z_k)
    p_{k+1} = z_{k+1} + beta_k * p_k
```

### 5.3 Support for Complex Symmetric Matrices

FEM matrices are complex symmetric (A^T = A) and are **not Hermitian** (A^H != A).
Example: eddy current equation curl-curl + i*sigma*mass.

In this case, the inner product must be **unconjugated**:

```
(a, b) = Sigma a_i * b_i       (unconjugated: complex-symmetric)
(a, b) = Sigma conj(a_i) * b_i  (conjugated: Hermitian)
```

Switching in the implementation:

```cpp
// iterative_solver.hpp: dot_product()
if (config_.conjugate)
    sum += std::conj(a[i]) * b[i];  // Hermitian
else
    sum += a[i] * b[i];  // complex-symmetric (default)
```

This corresponds to NGSolve's `CGSolver(conjugate=False)`.

### 5.4 Convergence Criterion

Convergence is determined by the relative residual norm:

```
||r_k|| / ||r_0|| < tol
```

Additional features:
- **Best-result tracking**: retains the best solution and returns it when convergence is not achieved
- **Divergence detection**: early termination if the residual exceeds a threshold multiple of the best value
- **Residual history**: records the residual at each iteration

---

## 6. DenseMatrix LU Inverse

### 6.1 Overview

LU factorization and inverse computation for small dense matrices. Used internally by COO/CSR matrix builders.
Performs PA = LU factorization with partial pivoting.

**Implementation file**: `core/dense_matrix.hpp`

### 6.2 Permutation Matrix Construction

LU factorization with partial pivoting produces a permutation vector `piv[]` that records row swaps.
The permutation matrix P is constructed as follows:

```
P[k, piv[k]] = 1   (correct: PA = LU)
```

**Caution**: `P[piv[k], k] = 1` is P^T (transpose) and is incorrect.
This distinction has little effect for diagonally dominant matrices (H1),
but becomes critical for matrices with non-trivial pivoting (HCurl).

```cpp
// dense_matrix.hpp: invert()
DenseMatrix inv(n, n);
for (index_t k = 0; k < n; ++k) {
    inv(k, piv[k]) = Scalar(1);  // P[k, piv[k]] = 1
}
```

### 6.3 Three Stages of Inverse Computation

```
1. LU factorization: PA = LU (partial pivoting)
2. Forward substitution: L * Y = P (P is the permutation matrix)
3. Backward substitution: U * X = Y -> X = A^{-1}
```

---

## 7. Code Origins and Contributors

ngsolve-sparsesolv integrates and extends code from multiple researchers.

### 7.1 Origin Mapping

| Component | Origin | Original Author | Original File (JP-MARs/SparseSolv) |
|---|---|---|---|
| CG iterative solver | Sato | Sato (tsato) | `MatSolvers_ICCG.cpp` |
| IC factorization and sequential forward/backward substitution | Sato | Sato (tsato) | `MatSolvers_ICCG.cpp` |
| ABMC ordering | Hiruma | Shingo Hiruma | `MatSolvers_ABMCICCG.cpp` |
| ABMC parallel IC factorization and parallel forward/backward substitution | Hiruma | Shingo Hiruma | `MatSolvers_ABMCICCG.cpp` |
| IC-MRTR iteration | Tsuburaya -> Sato | Tomoki Tsuburaya (theory), Sato (implementation) | `MatSolvers_ICMRTR.cpp` |
| SGS-MRTR iteration (Eisenstat technique) | Tsuburaya -> Sato | Tomoki Tsuburaya (theory), Sato (implementation) | `MatSolvers_SGSMRTR.cpp` |

### 7.2 Original Code

- **Tsuburaya's code**: Reference implementation of IC-MRTR and SGS-MRTR iteration formulas (C language, 1-based indexing).
  The auto-shift loop for IC factorization with shift parameter (gamma = 1.05, +0.05 increments) originates from this code

- **JP-MARs/SparseSolv**: `https://github.com/JP-MARs/SparseSolv`
  - Sparse matrix solver framework by Sato (tsato) (C++, Eigen-based)
  - ABMC parallelization added by Hiruma
  - Fork base of ngsolve-sparsesolv

### 7.3 Extensions in ngsolve-sparsesolv

The following features were added after forking:

| Feature | Description |
|---|---|
| IC auto-shift | Automatic shift adjustment for semi-definite matrices |
| Diagonal scaling | Condition number improvement via DAD transformation |
| Zero diagonal handling | Support for zero-diagonal DOFs in curl-curl matrices |
| Localized IC (Block IC) | Partition-level IC based on Fukuhara 2009 |
| Persistent parallel region | Level scheduling acceleration with SpinBarrier synchronization |
| RCM ordering | Reverse Cuthill-McKee bandwidth reduction before ABMC |
| Complex symmetric support | COCG / complex-symmetric CG |
| NGSolve integration | pybind11 module, BaseMatrix compatibility |

---

## 8. Compact AMS + COCR (For Complex Eddy Current Problems)

### 8.1 Overview

Compact AMS is a header-only C++ implementation of the **Auxiliary Space Preconditioning** method
by Hiptmair-Xu (2007). It requires no external libraries and uses NGSolve TaskManager for parallelization.

**Features**:
- Specialized for curl-curl + mass problems with HCurl finite elements (supports both real and complex)
- Requires only the discrete gradient matrix G and vertex coordinates (no element matrices needed)
- Uses CompactAMG (classical AMG) as the auxiliary space solver
- `Update()` supports nonlinear solvers (Newton iteration): retains geometric information and rebuilds only matrix-dependent parts
- Real magnetostatics: `CompactAMSPreconditioner` + CG
- Complex eddy currents: `ComplexCompactAMSPreconditioner` + COCR

**Implementation files**: `preconditioners/compact_amg.hpp`, `compact_ams.hpp`, `complex_compact_ams.hpp`

See [compact_ams_cocr.md](compact_ams_cocr.md) for details.

### 8.2 AMS Theory

Helmholtz decomposition of the HCurl space:

```
H(curl) = grad(H^1) + (H(curl) intersection ker(div))
```

The kernel of the curl-curl matrix K is contained in the gradient space grad(H^1).
AMS exploits this space decomposition by preconditioning in two auxiliary spaces:

1. **G correction** (gradient correction): processes the grad(H^1) component via `G * A_G^{-1} * G^T`
2. **Pi correction** (Nedelec interpolation): processes the curl component via `Pi * A_Pi^{-1} * Pi^T`

Where:
- G: discrete gradient matrix (HCurl -> H1), obtained via `fes.CreateGradient()`
- Pi: Nedelec interpolation matrix, automatically constructed from vertex coordinates (x, y, z)
- A_G, A_Pi: coarse grid matrices on the auxiliary spaces (approximate inverse via CompactAMG)

### 8.3 Application to Complex Eddy Currents

For complex symmetric systems A = K + jw*sigma*M, Re/Im splitting is used.
The AMS preconditioner is built from the real SPD auxiliary matrix
`A_real = K + eps*M + |w|*sigma*M_cond`, and the same AMS instance is applied
to the real and imaginary parts of the complex vector.

**Fused Re/Im processing**: Since SpMV is memory bandwidth-bound, loading the matrix data once
and processing both Re/Im simultaneously halves the bandwidth cost. This optimization is applied
to the fine-level smoother, residual computation, restriction/prolongation, and all levels of
the AMG V-cycle (DualMult).

### 8.4 AMG Coarse Grid Construction Complexity

`BuildClassicalInterp()` (classical interpolation matrix construction) scans the
non-zero pattern of strongly connected columns for each row, resulting in a worst-case
complexity of **O(nnz^2/n)**.
For matrices with dense coupling patterns (3D higher-order elements), the coarsening
setup may become dominant.

Currently this cost is within acceptable bounds, but for large-scale problems,
care must be taken regarding the trade-off with PMIS coarsening quality
(coarse grid size).

### 8.5 Recommended Selection

| Condition | Recommendation | Reason |
|-----------|---------------|--------|
| H1 low order (p=1-2) | **ABMC ICCG** | Fastest setup, minimum memory |
| H1/HCurl high order (p>=3) | **BDDC** (NGSolve built-in) | h-independent iterations, excels at high order |
| HCurl magnetostatics (real, p=1) | **Compact AMS+CG** | AMS auxiliary space handles curl-curl null space |
| HCurl nonlinear (Newton iteration) | **Compact AMS+CG** | `Update()` rebuilds preconditioner |
| HCurl eddy current (large-scale, p=1) | **Compact AMS+COCR** | Fused Re/Im + AMS auxiliary space |
| HCurl eddy current (medium-scale) | **BDDC+CG** (NGSolve built-in) | Memory efficient, symmetric preconditioner |
| Tight memory constraints | **ABMC ICCG** | CG/COCR: approximately 5-6 work vectors |

---

## References

1. C. R. Dohrmann,
   "A Preconditioner for Substructuring Based on Constrained Energy Minimization",
   *SIAM J. Sci. Comput.*, Vol. 25, No. 1, pp. 246-258, 2003.
   [DOI: 10.1137/S1064827502412887](https://doi.org/10.1137/S1064827502412887)

2. J. Mandel, C. R. Dohrmann, R. Tezaur,
   "An Algebraic Theory for Primal and Dual Substructuring Methods
   by Constraints",
   *Appl. Numer. Math.*, Vol. 54, No. 2, pp. 167-193, 2005.
   [DOI: 10.1016/j.apnum.2004.09.022](https://doi.org/10.1016/j.apnum.2004.09.022)

3. J. A. Meijerink, H. A. van der Vorst,
   "An Iterative Solution Method for Linear Systems of Which the
   Coefficient Matrix is a Symmetric M-Matrix",
   *Math. Comp.*, Vol. 31, No. 137, pp. 148-162, 1977.
   [DOI: 10.1090/S0025-5718-1977-0438681-4](https://doi.org/10.1090/S0025-5718-1977-0438681-4)

4. M. R. Hestenes, E. Stiefel,
   "Methods of Conjugate Gradients for Solving Linear Systems",
   *J. Research of the National Bureau of Standards*,
   Vol. 49, No. 6, pp. 409-436, 1952.
   [DOI: 10.6028/jres.049.044](https://doi.org/10.6028/jres.049.044)

5. T. Iwashita, H. Nakashima, Y. Takahashi,
   "Algebraic Block Multi-Color Ordering Method for Parallel
   Multi-Threaded Sparse Triangular Solver in ICCG Method",
   *Proc. IEEE IPDPS*, 2012.
   [DOI: 10.1109/IPDPS.2012.51](https://doi.org/10.1109/IPDPS.2012.51)

6. T. Tsuburaya, Y. Mifune, T. Iwashita, E. Takahashi,
   "Numerical Experiments on Preconditioned Iterative Methods Based on the MRTR Method",
   *IEEJ Technical Meeting on Static Apparatus*, SA-12-64, 2012.

7. T. Tsuburaya, Y. Okamoto, K. Fujiwara, S. Sato,
   "Improvement of the Preconditioned MRTR Method With Eisenstat's Technique
   in Real Symmetric Sparse Matrices",
   *IEEE Trans. Magnetics*, Vol. 49, No. 5, pp. 1641-1644, 2013.
   [DOI: 10.1109/TMAG.2013.2240283](https://doi.org/10.1109/TMAG.2013.2240283)

8. R. Hiptmair, J. Xu,
   "Nodal Auxiliary Space Preconditioning in H(curl) and H(div) Spaces",
   *SIAM J. Numer. Anal.*, Vol. 45, No. 6, pp. 2483-2509, 2007.
   [DOI: 10.1137/060660588](https://doi.org/10.1137/060660588)

9. T. V. Kolev, P. S. Vassilevski,
   "Parallel Auxiliary Space AMG for H(curl) Problems",
   *J. Comput. Math.*, Vol. 27, No. 5, pp. 604-623, 2009.

10. T. Sogabe, S.-L. Zhang,
    "A COCR method for solving complex symmetric linear systems",
    *J. Comput. Appl. Math.*, Vol. 199, No. 2, pp. 297-303, 2007.
