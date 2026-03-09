# Algebraic Block Multi-Color (ABMC) Ordering: Implementation Details

This document describes the mathematical background and C++ implementation details of the Algebraic Block Multi-Color (ABMC) ordering method used in SparseSolv. ABMC is a technique for parallelizing the forward and backward substitution steps in Incomplete Cholesky (IC) preconditioning.

**Implementation files:**
- `core/abmc_ordering.hpp` -- ABMC schedule construction (BFS aggregation + graph coloring)
- `core/level_schedule.hpp` -- Level scheduling (comparison baseline)
- `core/parallel.hpp` -- Parallel primitives (`parallel_for`, `SpinBarrier`, `get_num_threads`)
- `core/solver_config.hpp` -- ABMC parameter configuration
- `preconditioners/ic_preconditioner.hpp` -- Triangular solve execution, apply path selection

---

## 1. Theoretical Background

### 1.1 Challenges of Parallel Triangular Solves

In iterative solvers such as ICCG, applying the preconditioner requires solving triangular systems:
- Forward substitution: solve $L y = x$
- Backward substitution: solve $L^T z = y$

Computing $y_i$ depends on all $y_j$ for which $L_{ij} \neq 0$ and $j < i$. This data dependency is the bottleneck for parallelization.

### 1.2 Level Scheduling

A basic technique for resolving dependencies. The level (dependency depth) of each row is computed, and rows at the same level are processed in parallel:

```
level[i] = max(level[j] for j in L[i,:] where j < i) + 1
```

**Problem**: FEM matrices have wide bandwidth and deep dependency chains. The number of levels reaches hundreds, and each level contains few rows, resulting in poor parallel efficiency.

### 1.3 Multi-Coloring

An approach that breaks dependencies through graph coloring. Nodes are colored so that no two adjacent nodes share the same color, and nodes of the same color are processed in parallel. However, classical point-based multi-coloring leads to poor cache locality because memory accesses jump across distant locations based on color assignments.

### 1.4 ABMC: Block + Multi-Color

ABMC [Iwashita, Nakashima, Takahashi 2012] introduces a two-level hierarchy to achieve both parallelism and cache locality:

1. **Blocking**: Group nearby rows into blocks -- maintains cache locality
2. **Coloring**: Color inter-block dependencies -- enables parallelism with few synchronization points

It requires no geometric information and operates solely on the CSR matrix pattern.

---

## 2. Implementation Pipeline

The ABMC schedule is constructed during `ICPreconditioner::setup()`. `ABMCSchedule::build()` executes four stages in sequence (`abmc_ordering.hpp`).

### Stage 1: Algebraic Blocking (BFS Aggregation)

**Function**: `make_algebraic_blocks()`

Groups nearby rows into blocks using BFS (breadth-first search) on the matrix graph.

```
Input: CSR matrix (row_ptr, col_idx), block size B
Output: raw_block_list[b] = {set of original row indices}
        block_assign[i] = block ID for row i

1. Initialize block_assign to -1 (unassigned)
2. Start a new block with the first unassigned row as seed
3. Add seed's neighbors to the BFS queue (mark -2: in queue)
4. Dequeue and add to block until block size B is reached
5. Reset remaining queued nodes to -1 (for the next block)
6. Repeat until all rows are assigned
```

**Sentinel value usage:**
| Value | Meaning |
|-------|---------|
| `-1` | Unassigned |
| `-2` | In BFS queue (prevents duplicate insertion) |
| `>= 0` | Assigned to a block ID |

**Complexity**: O(nnz) -- each nonzero element is traversed at most twice

Block size `B` (default: 4, `SolverConfig::abmc_block_size`) balances cache efficiency and parallel granularity, since rows within a block are processed sequentially.

### Stage 2: Block Adjacency Graph Construction

**Function**: `build_block_graph()`

Two blocks $A$ and $B$ are adjacent if any row in block $A$ has a nonzero connection to any row in block $B$.

```
Input: raw_block_list, block_assign, CSR matrix
Output: Block adjacency graph (blk_row_ptr, blk_col_idx) -- CSR format

for each block bi:
    for each row row in bi:
        for each nonzero column j in row:
            bj = block_assign[j]
            if bj != bi and last_seen[bj] != bi:
                add bj to neighbors[bi]
                last_seen[bj] = bi  <-- deduplication sentinel
```

**Complexity**: O(nnz). The `last_seen` sentinel ensures each block pair is added only once.

### Stage 3: Greedy Coloring

**Function**: `color_graph()`

Colors the block adjacency graph so that blocks with lower-triangular dependencies have different colors.

**Two-phase algorithm:**

```
Phase 1: Determine minimum number of colors
    num_colors = target_colors  (SolverConfig::abmc_num_colors, default: 4)
    for each block i:
        lower_count = |{j : j < i and (i,j) are adjacent}|
        num_colors = max(num_colors, lower_count + 1)

Phase 2: Greedy coloring with forbidden-color-set
    forbidden[c] = i means "color c is used by a lower-triangular neighbor of block i"
    for each block i (in order):
        Mark colors of lower-triangular neighbors j < i as forbidden
        Select the smallest unused color
        If all colors are used: ++num_colors (add a new color)
```

**Important**: `target_colors` is a **lower bound**, not a guarantee. The actual number of colors is:

```
num_colors >= max(target_colors, 1 + max_i |{j < i : (i,j) in E}|)
```

It may further increase in Phase 2 (Phase 1 computes based on lower-triangular degree only, but the coloring order may require additional colors).

**Complexity**: O(sum of degrees) -- each block's neighbors are scanned once

### Stage 4: Row-Level Ordering Construction

**Function**: `build_row_ordering()`

Constructs flat row-level arrays from the block coloring result.

```
Output:
  color_offsets[c] -- starting block position for color c (CSR)
  color_blocks[k] -- global block ID
  block_offsets[b] -- starting row position for block b (CSR)
  ordering[old] = new -- old-to-new permutation
  reverse_ordering[new] = old -- new-to-old inverse permutation

Traversal order: color 0 -> color 1 -> ... -> color C-1
  Within each color: block order
    Within each block: sorted by original index (std::sort)
```

**Consecutive numbering**: New row indices are assigned consecutively in traversal order. Rows in block $b$ span from `block_offsets[b]` to `block_offsets[b+1]-1` contiguously, making a separate row-index array unnecessary. The triangular solve can directly loop as `for (i = block_offsets[blk]; i < block_offsets[blk+1]; ++i)`.

The purpose of intra-block sorting is to ensure that rows within the same block have nearby indices in the original matrix, reducing TLB misses and improving cache line utilization during preconditioner application.

---

## 3. Parallel Triangular Solve Logic

### 3.1 Data Structures

Data involved in the triangular solve after ABMC scheduling:

| Data | Space | Description |
|------|-------|-------------|
| `L_`, `Lt_`, `inv_diag_` | ABMC space | IC factorization results (lower/upper triangular factors, inverse diagonal) |
| `abmc_x_perm_` | ABMC space | Input vector permutation buffer (size n) |
| `abmc_y_perm_` | ABMC space | Output vector permutation buffer (size n) |
| `work_temp_` | ABMC space | Intermediate result vector (size n, shared by ABMC/level scheduling) |
| `work_temp2_` | ABMC space | Second intermediate result (diagonal scaling only, size n) |
| `composite_perm_` | Original -> ABMC | Composite permutation |
| `composite_scaling_` | Original space | Diagonal scaling coefficients (original space) |

**Work vector sharing**: `work_temp_` is shared between the ABMC path and the level scheduling path. It is allocated once during `setup()`, eliminating heap allocations during `apply()`.

### 3.2 Forward Substitution ($L y = x$)

`forward_substitution_abmc()` -- 3-level loop:

```cpp
for (color c = 0; c < num_colors; ++c) {            // Sequential: inter-color dependency
    parallel_for(num blocks in color c, [&](bidx) {  // Parallel: same-color blocks are independent
        blk = color_blocks[blk_begin + bidx];
        row_begin = block_offsets[blk];
        row_end = block_offsets[blk + 1];
        for (i = row_begin; i < row_end; ++i) {      // Sequential: intra-block dependency
            s = x[i];                                 // Direct use of ABMC-space row index
            for (k : off-diagonal entries in row i of L_)
                s -= L_.values[k] * y[L_.col_idx[k]];
            y[i] = s / L_.values[diag];
        }
    });
    // Implicit synchronization barrier (parallel_for completion = proceed to next color)
}
```

Since `block_offsets` defines contiguous index ranges, rows within a block can be looped directly from `row_begin` to `row_end`. No indirect-reference arrays are needed.

### 3.3 Backward Substitution ($L^T z = D^{-1} y$)

`backward_substitution_abmc()` -- reverse order of forward substitution:
- Colors are traversed in reverse order: C-1 -> 0
- Rows within blocks are also traversed in reverse (`row_end -> row_begin`)
- Uses $L^T$ (upper triangular)

```cpp
for (c = nc; c-- > 0;) {                              // Colors in reverse order
    parallel_for(num blocks in color c, [&](bidx) {
        for (i = row_end; i-- > row_begin;) {          // Rows in reverse order
            s = 0;
            for (k : off-diagonal entries in row i of Lt_)
                s -= Lt_.values[k] * y[Lt_.col_idx[k]];
            y[i] = s * inv_diag_[i] + x[i];           // Apply D^{-1} + add source
        }
    });
}
```

### 3.4 Correctness Justification

ABMC coloring ensures that no lower-triangular dependencies exist between blocks of the same color. Therefore:
- Forward substitution for blocks of color c depends only on results from colors < c
- Blocks of the same color c are mutually independent -- can be executed in parallel
- Intra-block dependencies from the original matrix are preserved -- sequential execution is correct

---

## 4. Apply Path Selection

`ICPreconditioner` provides three apply paths, differing in the "space" of input/output vectors.

### 4.1 Standard Path: `apply_abmc()` (Recommended)

Used when CG operates in the original space. Permutation is performed during preconditioner application.

```
x (original space) -> permute with composite_perm_ -> abmc_x_perm_ (ABMC space)
  -> forward_substitution_abmc -> work_temp_
  -> backward_substitution_abmc -> abmc_y_perm_ (ABMC space)
y (original space) <- inverse permute with composite_perm_ <- abmc_y_perm_
```

When diagonal scaling is enabled, scaling is applied simultaneously with permutation:
```cpp
abmc_x_perm_[perm[i]] = x[i] * composite_scaling_[i];
y[i] = abmc_y_perm_[perm[i]] * composite_scaling_[i];
```

`composite_scaling_[i] = scaling_[composite_perm_[i]]` -- precomputed.

### 4.2 ABMC Full-Space Path: `apply_in_reordered_space()`

Used when CG operates in ABMC space (`abmc_reorder_spmv=True`). No permutation is needed since input/output are already in ABMC space.

### 4.3 RCM+ABMC Separated Path: `apply_rcm_abmc()`

Used when CG operates in RCM space. Only RCM-to-ABMC permutation is needed:
```
x (RCM space) -> permute with composite_perm_rcm_ -> ABMC space
  -> triangular solve
y (RCM space) <- inverse permute with composite_perm_rcm_ <- ABMC space
```

### 4.4 Path Selection Criteria

| Path | SpMV Matrix | Preconditioner | Configuration |
|------|-------------|----------------|---------------|
| Standard (4.1) | Original matrix | Composite permutation + ABMC triangular solve | `use_abmc=True` |
| RCM+ABMC (4.3) | RCM matrix | RCM-to-ABMC permutation | `use_abmc=True, abmc_use_rcm=True` |
| ABMC full-space (4.2) | ABMC matrix | No permutation | `abmc_reorder_spmv=True` |

In the NGSolve integration (`sparsesolv_ngsolve` module), SpMV is handled by NGSolve's `BaseMatrix::Mult`, which operates in the original matrix space, so the **standard path (4.1)** is used.

---

## 5. Combination with RCM

### 5.1 Composite Permutation

When using RCM+ABMC, a two-stage permutation is required:

```
Original row i -> RCM space: rcm_ordering_[i]
RCM space -> ABMC space: abmc_ord[rcm_ordering_[i]]
```

Rather than performing two levels of indirection each time, a composite permutation is precomputed during setup:

```cpp
// build_composite_permutations()
composite_perm_[i] = abmc_ord[rcm_ordering_[i]];  // Original -> ABMC (single step)
composite_perm_rcm_ = abmc_ord;                    // RCM -> ABMC (single step)
```

### 5.2 Matrix Copy Flow

The RCM+ABMC path involves two matrix copies:

```
Original matrix A
  | reorder_matrix_with_perm(A, rcm_ordering_)
RCM matrix rcm_csr_
  | reorder_matrix(rcm_view)  [reorder with ABMC ordering]
ABMC matrix reordered_csr_
  | extract_lower_triangular()
L_ (for IC factorization)
```

The ABMC-only path requires only one copy (`A -> reordered_csr_ -> L_`).

### 5.3 Memory Management

After factorization is complete, unnecessary matrix copies are freed:
```cpp
if (!config_.abmc_reorder_spmv)
    reordered_csr_.clear();  // SpMV uses the original matrix
if (!config_.abmc_use_rcm)
    rcm_csr_.clear();
```

---

## 6. Comparison with Level Scheduling

### 6.1 Differences in Parallelization Approaches

SparseSolv provides two parallel triangular solve methods. When ABMC is not used, **level scheduling with a persistent parallel region** is employed.

| Characteristic | Level Scheduling (persistent) | ABMC |
|---------------|-------------------------------|------|
| Parallel granularity | Row-level | Block-level |
| Synchronization count | Number of levels (hundreds) | Number of colors (4 to tens) |
| Synchronization mechanism | SpinBarrier (within a single parallel_for) | parallel_for calls (per color) |
| Parallelism for FEM | Low (deep dependency chains) | High |
| Setup cost | O(nnz) -- level computation only | O(nnz) -- BFS + coloring + matrix reordering |
| Additional memory | Level arrays only | Work vectors 2*n + matrix copy |
| Allocation during apply | None (pre-allocated) | None (pre-allocated) |

**Additional memory breakdown (ABMC path)**:
- `abmc_x_perm_` (n), `abmc_y_perm_` (n) -- ABMC-specific
- `work_temp_` (n), `work_temp2_` (n) -- shared by both paths
- `composite_perm_` (n), `composite_scaling_` (n) -- permutation/scaling
- `reordered_csr_` -- reordered matrix (freed after factorization when `abmc_reorder_spmv=false`)

### 6.2 Persistent Parallel Region

A naive implementation of level scheduling calls `parallel_for` for each level. When the number of levels reaches hundreds, the overhead of launching the thread pool becomes problematic.

`forward_substitution_persistent()` processes all levels within a single `parallel_for(nthreads, ...)`, synchronizing between levels with SpinBarrier:

```cpp
parallel_for(nthreads, [&](index_t thread_id) {
    for (int lev = 0; lev < num_levels; ++lev) {
        // Each thread processes its assigned range of rows
        const index_t my_start = level_size * thread_id / nthreads;
        const index_t my_end = level_size * (thread_id + 1) / nthreads;
        for (idx = my_start; idx < my_end; ++idx) { ... }
        barrier.wait();  // Inter-level synchronization
    }
});
```

This path is automatically selected when the thread count is greater than 1 (`get_num_threads() > 1`).

### 6.3 SpinBarrier Implementation

`SpinBarrier` (`parallel.hpp`) uses a sense-reversing algorithm:

```cpp
class SpinBarrier {
    alignas(64) std::atomic<int> count_;   // Number of arrived threads
    alignas(64) std::atomic<int> sense_;   // Current generation number
    int num_threads_;
public:
    void wait() {
        int my_sense = sense_.load(acquire);
        if (count_.fetch_add(1, acq_rel) == num_threads_ - 1) {
            // Last thread: reset counter + flip sense
            count_.store(0, relaxed);
            sense_.fetch_add(1, release);
        } else {
            // Spin-wait (yield after 4096 iterations)
            while (sense_.load(acquire) == my_sense) { ... }
        }
    }
};
```

**Cache line separation**: `count_` and `sense_` are placed on separate cache lines using `alignas(64)` to prevent false sharing.

**Spin-to-yield strategy**: After 4096 spin iterations, `std::this_thread::yield()` is called. This spins for short synchronizations and yields the CPU for longer ones.

---

## 7. Performance Characteristics

### 7.1 Benchmark Conditions

- 8 threads (NGSolve TaskManager)
- 3D HCurl curl-curl problem (order=2, `nograds=True`)
- Unit cube mesh, shifted-ICCG (`shift=1.05, auto_shift=True, diagonal_scaling=True`)
- Baseline: level scheduling with persistent parallel region
- Minimum of 3 runs

### 7.2 Scaling Results

| DOFs | Level Sched. | ABMC (best) | Speedup | Iterations |
|------|-------------|-------------|---------|------------|
| 11K | 0.044s | 0.051s | **0.86x** | 31 |
| 27K | 0.158s | 0.154s | **1.02x** | 39 |
| 82K | 0.763s | 0.650s | **1.17x** | 59 |
| 186K | 2.505s | 1.962s | **1.28x** | 79 |

### 7.3 Comparison on NB02 Toroidal Coil

On the toroidal coil problem from [02_performance_comparison.ipynb](02_performance_comparison.ipynb) (148K DOFs):

| Solver | Iterations | Wall Time | Speedup |
|--------|-----------|-----------|---------|
| ICCG (level sched.) | 463 | 10.0s | 1.0x |
| ICCG + ABMC (8 colors) | 415 | 6.0s | **1.7x** |

On the unit cube (186K DOFs) the speedup is 1.28x, while on the toroidal coil (148K DOFs) it is 1.7x. The factors behind this difference are:
- **Iteration count**: 463 vs 79. A higher iteration count increases the time fraction spent in triangular solves, making ABMC's effect more pronounced
- **Matrix structure**: The toroidal coil has wider bandwidth, leading to more levels in level scheduling -- greater synchronization overhead -- greater advantage for ABMC

### 7.4 Break-Even Point

ABMC has the following overheads:
1. **Setup**: BFS aggregation, block graph construction, coloring, matrix reordering
2. **Per-apply**: Vector permutation (2 `parallel_for` calls for scatter/gather)

In regimes where these overheads exceed the speedup from faster triangular solves, ABMC becomes counterproductive.

**Break-even point: approximately 25K--30K DOFs** (8 threads, unit cube, HCurl order 2)

- DOFs < 25K: Level scheduling is faster
- DOFs > 30K: ABMC is advantageous. The benefit grows proportionally with problem size

### 7.5 Factors Affecting Effectiveness

| Factor | Favorable for ABMC | Unfavorable for ABMC |
|--------|-------------------|---------------------|
| Problem size | Large (>30K DOFs) | Small (<25K DOFs) |
| Iteration count | High (>100) | Low (<30) |
| Matrix bandwidth | Wide (deep dependency chains) | Narrow |
| ABMC color count | Few (4--8) | Many (>20) |
| Thread count | Many (>=4) | Few (1--2) |

---

## 8. Configuration Parameters

ABMC-related parameters controlled by `SolverConfig`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_abmc` | bool | false | Enable ABMC |
| `abmc_block_size` | int | 4 | Block size for BFS aggregation (rows/block) |
| `abmc_num_colors` | int | 4 | Target number of colors (lower bound, auto-expanded) |
| `abmc_reorder_spmv` | bool | false | Run CG in ABMC space (SpMV also reordered) |
| `abmc_use_rcm` | bool | false | Apply RCM bandwidth reduction before ABMC |

**Recommended settings**: `use_abmc=True` alone (with other defaults) is sufficient.

- `abmc_block_size`: Performance varies by only a few percent across values of 4--16
- `abmc_num_colors`: Almost always auto-expanded, so the specified value has little impact
- `abmc_reorder_spmv`: Usually false to preserve the cache locality of FEM mesh ordering
- `abmc_use_rcm`: Usually false due to the additional matrix copy and setup cost

---

## 9. ABMCSchedule Data Structure

Members of the `ABMCSchedule` struct (`abmc_ordering.hpp`):

```cpp
struct ABMCSchedule {
    // Color -> blocks (CSR format)
    std::vector<index_t> color_offsets;    // size: num_colors + 1
    std::vector<index_t> color_blocks;     // List of block IDs per color

    // Block -> rows (CSR format, effectively offsets only due to consecutive numbering)
    std::vector<index_t> block_offsets;    // size: num_blocks + 1

    // Row permutation
    std::vector<index_t> ordering;         // ordering[old_row] = new_row
    std::vector<index_t> reverse_ordering; // reverse_ordering[new_row] = old_row
};
```

**Traversal pattern in triangular solve**:
```
for c = 0 to num_colors - 1:
    blk_begin = color_offsets[c]
    blk_end = color_offsets[c + 1]
    parallel_for(blk_begin .. blk_end):
        blk = color_blocks[bidx]
        for i = block_offsets[blk] to block_offsets[blk+1] - 1:
            // Process row i in ABMC space
```

**Memory efficiency**: Since `block_offsets` defines contiguous row ranges, no row-to-block mapping array is needed. The rows of block $b$ are fully determined by `[block_offsets[b], block_offsets[b+1])`.

---

## 10. CG Kernel Fusion (v2.3.0)

### 10.1 Motivation

CG iterations are memory-bandwidth bound (arithmetic intensity ~0.33 FLOP/byte), and re-reading vectors between kernels is the bottleneck. Before fusion, a CG iteration involves 7 kernel launches with a total memory traffic of nnz + 12n reads and 4n writes (excluding IC apply).

### 10.2 Fused Kernels

**Phase 1: SpMV + dot fusion** -- `cg_solver.hpp`

```cpp
Scalar pAp = parallel_reduce_sum<Scalar>(n, [&](index_t i) -> Scalar {
    Scalar s = Scalar(0);
    for (index_t k = A_rowptr[i]; k < A_rowptr[i + 1]; ++k)
        s += A_vals[k] * p[A_colidx[k]];
    Ap[i] = s;
    return p[i] * s;  // Partial sum for dot(p, Ap)
});
```

Computes SpMV (`Ap = A*p`) and the inner product (`pAp = dot(p, Ap)`) in a single pass. Eliminates re-reading of p[] and Ap[] (2n reads).

**Phase 2: AXPY + norm fusion**

```cpp
double norm_r_sq = parallel_reduce_sum<double>(n, [&](index_t i) -> double {
    x[i] += alpha * p[i];
    r[i] -= alpha * Ap[i];
    return std::norm(r[i]);  // |r[i]|^2
});
```

Executes vector updates (`x += alpha*p`, `r -= alpha*Ap`) and residual norm computation in a single pass. Eliminates re-reading of r[] (n reads).

**Phase 3: Preconditioner apply + dot(r, z) fusion** -- `preconditioner.hpp`, `ic_preconditioner.hpp`

```cpp
// Preconditioner::apply_fused_dot() -- virtual method in the base class
Scalar apply_fused_dot(const Scalar* r_for_dot, const Scalar* x,
                       Scalar* y, index_t size, bool conjugate) const {
    apply(x, y, size);
    return parallel_reduce_sum<Scalar>(size, [&](index_t i) -> Scalar {
        return conjugate ? std::conj(r_for_dot[i]) * y[i]
                         : r_for_dot[i] * y[i];
    });
}
```

Each IC preconditioning path overrides `apply_fused_dot()` to compute `dot(r, z)` simultaneously during the output phase of backward substitution. In the ABMC-space path (Path 1), thread-local partial sums are accumulated within the persistent parallel region and reduced serially at the end.

**Effect**: 7 kernels per iteration reduced to 4, with approximately 25% reduction in memory traffic.

### 10.3 ABMC Parallel auto_shift

Before v2.3.0, ABMC parallel IC factorization was unavailable when auto_shift was enabled, falling back to the sequential path. In v2.3.0, an atomic flag (`std::atomic<bool>`) restart mechanism was introduced:

1. Execute ABMC color-parallel IC factorization
2. If any thread detects a diagonal breakdown, set the atomic flag
3. At the end of the color loop, check the flag -- increase the shift (exponential backoff) -- restart from scratch with the original values

The shift increase uses exponential backoff (`increment *= 2`), reaching an appropriate shift value with few restarts.
Since the ABMC ordering is pattern-based, no recomputation is needed upon restart.

**Effect**: For the Hiruma eddy current problem (HCurl p=1), parallel scaling improved from 1.5x to 2.6x (8 cores).

### 10.4 ABMC-Space CG (`abmc_reorder_spmv=True`)

A mode that runs CG entirely in ABMC-reordered space. SpMV, IC apply, and all vector operations are performed in ABMC space.

Advantages:
- Eliminates input/output vector permutation (scatter/gather) during IC apply
- SpMV cache locality aligns with the ABMC block structure
- Enabled with parameter `abmc_reorder_spmv=True`

**Effect**: 2.7x to 2.89x on the Hiruma problem (8 cores).

### 10.5 Persistent Parallel Region + dot(r,z) Fusion (`apply_fused_dot`)

`apply_in_reordered_space()` (ABMC-space CG path) originally called `forward_substitution_abmc()` / `backward_substitution_abmc()` for each color, resulting in a total of 2*nc `parallel_for` dispatches. This was converted to a persistent parallel region (1 dispatch + 2*nc barriers), similar to `apply_abmc()`, and additionally computes `dot(r, z)` during the final output of backward substitution.

Structure of `apply_in_reordered_space_fused_dot()`:

```cpp
parallel_for(nthreads, [&](index_t tid) {
    // Forward substitution (color-sequential, block-parallel)
    for (index_t c = 0; c < nc; ++c) {
        // ... my blocks in color c ...
        barrier.wait();
    }
    // Backward substitution + dot accumulation
    Scalar local_dot = 0;
    for (index_t c = nc; c-- > 0;) {
        // ... backward solve for my blocks ...
        // On final write: local_dot += r[i] * y[i];
        barrier.wait();
    }
    partial_dots[tid] = local_dot;
});
// Serial reduction of partial sums
```

Supported across all 3 paths:
- **Path 1** (`abmc_reorder_spmv=True`): `apply_in_reordered_space_fused_dot()`
- **Path 2** (RCM+ABMC): `apply_rcm_abmc_fused_dot()`
- **Path 3** (Standard ABMC): `apply_abmc_fused_dot()` -- dot fusion during Phase 4 (output permutation)

**Effect**: 2.89x to 3.14x on the Hiruma problem (8 cores, 79% of the theoretical maximum of 4x).

---

## 11. Known Limitations

### 11.1 Unpredictability of Color Count

`target_colors` is a lower bound for greedy coloring, and it is auto-expanded when inter-block dependencies are dense. For complex meshes (e.g., helical coils), the actual count may significantly exceed the specified value.

**Potential improvement**: Log the actual color count so users can verify it.

### 11.2 No Automatic Block Size Tuning

Currently, `block_size` (default: 4) is a user-specified fixed value. Automatic tuning based on matrix structure is not implemented.

Benchmark results show that the difference across block sizes 4--16 is small (within a few percent).

### 11.3 Limited Effectiveness of RCM Pre-Reordering

RCM may reduce the ABMC color count through bandwidth reduction, but it incurs additional matrix copy costs and setup time. In current benchmarks, the difference between `abmc_use_rcm=False` and `True` is within the margin of error, so the default is disabled.

---

## 12. Improvement History

This section records improvements made based on implementation review.

### 12.1 Removal of `block_rows` Array

**Issue**: The `block_rows[ridx]` array constructed in `build_row_ordering()` was always equal to `ridx` (since new indices are assigned consecutively, it was the identity mapping). Instead of `i = block_rows[ridx]`, one could simply write `i = ridx`.

**Resolution**: Removed the `block_rows` member from `ABMCSchedule`. Simplified the triangular solve loop to `for (i = row_begin; i < row_end; ++i)`. O(n) memory reduction.

### 12.2 Pre-Allocation and Sharing of Work Vectors

**Issue**: The level scheduling path's `apply_level_schedule()` heap-allocated `std::vector<Scalar> temp(size)` on every call. The ABMC path pre-allocated during `setup()`, but used a different variable name (`abmc_temp_`).

**Resolution**: Unified `work_temp_` (and `work_temp2_`) as shared member variables for both paths. Allocated once during `setup()`, completely eliminating heap allocations during `apply()`.

### 12.3 Removal of Unused Parameters from `build_row_ordering`

**Issue**: There were unused output parameters in `color_graph()` and unused parameters in `build_row_ordering()`.

**Resolution**: Removed unnecessary output parameters from `color_graph()`. Simplified the signature of `build_row_ordering()`.

---

## Acknowledgements

The ABMC ordering implementation is based on code provided by Yuki Tsurutani (Fukuoka University) at [JP-MARs/SparseSolv](https://github.com/JP-MARs/SparseSolv). This code implements the ABMC method [1] and SGS-MRTR preconditioning research from the group of Takeshi Iwashita (Kyoto University). This repository adds its own extensions including header-only restructuring, NGSolve integration, auto-shift IC, Compact AMS preconditioning, and COCR solver.

---

## References

1. T. Iwashita, H. Nakashima, Y. Takahashi,
   "Algebraic Block Multi-Color Ordering Method for Parallel
   Multi-Threaded Sparse Triangular Solver in ICCG Method",
   *Proc. IEEE IPDPS*, 2012.
   [DOI: 10.1109/IPDPS.2012.51](https://doi.org/10.1109/IPDPS.2012.51)

2. E. Cuthill, J. McKee,
   "Reducing the Bandwidth of Sparse Symmetric Matrices",
   *Proc. 24th Nat. Conf. ACM*, pp. 157--172, 1969.
   [DOI: 10.1145/800195.805928](https://doi.org/10.1145/800195.805928)

3. J. A. Meijerink, H. A. van der Vorst,
   "An Iterative Solution Method for Linear Systems of Which the
   Coefficient Matrix is a Symmetric M-Matrix",
   *Math. Comp.*, Vol. 31, No. 137, pp. 148--162, 1977.
   [DOI: 10.1090/S0025-5718-1977-0438681-4](https://doi.org/10.1090/S0025-5718-1977-0438681-4)
