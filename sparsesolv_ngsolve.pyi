"""Type stubs for sparsesolv_ngsolve — SparseSolv iterative solvers for NGSolve.

Provides IC/SGS preconditioners, ICCG/SGSMRTR iterative solvers,
and HYPRE AMS preconditioners for HCurl eddy-current problems.
"""

from ngsolve import BaseMatrix, BaseVector, BitArray

__all__ = [
    "SparseSolvResult",
    "ICPreconditionerD",
    "ICPreconditionerC",
    "SGSPreconditionerD",
    "SGSPreconditionerC",
    "SparseSolvSolverD",
    "SparseSolvSolverC",
    "ICPreconditioner",
    "SGSPreconditioner",
    "SparseSolvSolver",
    "COCRSolverD",
    "COCRSolverC",
    "COCRSolver",
    "HypreAMSPreconditioner",
    "ComplexHypreAMSPreconditioner",
    "HypreBoomerAMGPreconditioner",
    "has_hypre",
]

# =============================================================================
# Result type
# =============================================================================

class SparseSolvResult:
    """Result of a SparseSolv iterative solve."""

    converged: bool
    """Whether the solver converged within tolerance."""
    iterations: int
    """Number of iterations performed."""
    final_residual: float
    """Final relative residual (or best residual if save_best_result enabled)."""
    residual_history: list[float]
    """Residual at each iteration (empty unless save_residual_history enabled)."""

# =============================================================================
# IC Preconditioner types
# =============================================================================

class ICPreconditionerD(BaseMatrix):
    """Incomplete Cholesky preconditioner for real (double) matrices."""

    shift: float
    """IC shift parameter."""

    def Update(self) -> None:
        """Recompute the IC factorization with current shift."""
        ...

class ICPreconditionerC(BaseMatrix):
    """Incomplete Cholesky preconditioner for complex matrices."""

    shift: float
    """IC shift parameter."""

    def Update(self) -> None:
        """Recompute the IC factorization with current shift."""
        ...

# =============================================================================
# SGS Preconditioner types
# =============================================================================

class SGSPreconditionerD(BaseMatrix):
    """Symmetric Gauss-Seidel preconditioner for real (double) matrices."""

    def Update(self) -> None:
        """Recompute the SGS factorization."""
        ...

class SGSPreconditionerC(BaseMatrix):
    """Symmetric Gauss-Seidel preconditioner for complex matrices."""

    def Update(self) -> None:
        """Recompute the SGS factorization."""
        ...

# =============================================================================
# Solver types
# =============================================================================

class SparseSolvSolverD(BaseMatrix):
    """Iterative solver for real (double) matrices."""

    method: str
    """Solver method: ``"ICCG"``, ``"SGSMRTR"``, ``"CG"``, or ``"COCR"``."""
    tol: float
    """Convergence tolerance (relative residual)."""
    maxiter: int
    """Maximum iterations."""
    shift: float
    """IC shift parameter."""
    auto_shift: bool
    """Auto-increase shift on IC breakdown (for semi-definite systems)."""
    diagonal_scaling: bool
    """Diagonal scaling for improved conditioning."""
    save_best_result: bool
    """Track and return the best iterate found."""
    save_residual_history: bool
    """Record residual at each iteration."""
    printrates: bool
    """Print convergence info to stdout."""
    conjugate: bool
    """Use conjugated inner product for Hermitian systems."""
    divergence_check: bool
    """Enable stagnation-based divergence detection."""
    divergence_threshold: float
    """Threshold for divergence detection."""
    divergence_count: int
    """Iterations before declaring divergence."""
    use_abmc: bool
    """Enable ABMC ordering for parallel triangular solves."""
    abmc_block_size: int
    """Rows per block in ABMC aggregation."""
    abmc_num_colors: int
    """Target number of colors for ABMC coloring."""
    abmc_reorder_spmv: bool
    """Reorder SpMV in ABMC space (experimental)."""
    abmc_use_rcm: bool
    """Use RCM ordering (experimental)."""

    @property
    def last_result(self) -> SparseSolvResult:
        """Result from the last Solve() call."""
        ...

    def Solve(self, rhs: BaseVector, sol: BaseVector) -> SparseSolvResult:
        """Solve Ax = b.

        Args:
            rhs: Right-hand side vector.
            sol: Solution vector (input: initial guess, output: solution).

        Returns:
            SparseSolvResult with convergence info.
        """
        ...

class SparseSolvSolverC(BaseMatrix):
    """Iterative solver for complex matrices."""

    method: str
    tol: float
    maxiter: int
    shift: float
    auto_shift: bool
    diagonal_scaling: bool
    save_best_result: bool
    save_residual_history: bool
    printrates: bool
    conjugate: bool
    divergence_check: bool
    divergence_threshold: float
    divergence_count: int
    use_abmc: bool
    abmc_block_size: int
    abmc_num_colors: int
    abmc_reorder_spmv: bool
    abmc_use_rcm: bool

    @property
    def last_result(self) -> SparseSolvResult: ...

    def Solve(self, rhs: BaseVector, sol: BaseVector) -> SparseSolvResult:
        """Solve Ax = b.

        Args:
            rhs: Right-hand side vector.
            sol: Solution vector (input: initial guess, output: solution).

        Returns:
            SparseSolvResult with convergence info.
        """
        ...

# =============================================================================
# Factory functions (auto-dispatch real/complex based on mat.IsComplex())
# =============================================================================

def ICPreconditioner(
    mat: BaseMatrix,
    freedofs: BitArray | None = None,
    shift: float = 1.05,
) -> ICPreconditionerD | ICPreconditionerC:
    """Incomplete Cholesky (IC) Preconditioner.

    Auto-dispatches to real/complex based on ``mat.IsComplex()``.
    Calls ``Update()`` automatically on construction.

    Args:
        mat: SPD sparse matrix (real or complex).
        freedofs: Free DOFs. Constrained DOFs treated as identity.
        shift: Shift parameter for stability (default: 1.05).
    """
    ...

def SGSPreconditioner(
    mat: BaseMatrix,
    freedofs: BitArray | None = None,
) -> SGSPreconditionerD | SGSPreconditionerC:
    """Symmetric Gauss-Seidel (SGS) Preconditioner.

    Auto-dispatches to real/complex based on ``mat.IsComplex()``.
    Calls ``Update()`` automatically on construction.

    Args:
        mat: SPD sparse matrix (real or complex).
        freedofs: Free DOFs. Constrained DOFs treated as identity.
    """
    ...

def SparseSolvSolver(
    mat: BaseMatrix,
    method: str = "ICCG",
    freedofs: BitArray | None = None,
    tol: float = 1e-10,
    maxiter: int = 1000,
    shift: float = 1.05,
    save_best_result: bool = True,
    save_residual_history: bool = False,
    printrates: bool = False,
    conjugate: bool = False,
    use_abmc: bool = False,
    abmc_block_size: int = 4,
    abmc_num_colors: int = 4,
    abmc_reorder_spmv: bool = False,
    abmc_use_rcm: bool = False,
) -> SparseSolvSolverD | SparseSolvSolverC:
    """Iterative solver (ICCG / SGSMRTR / CG / COCR).

    Can be used as a BaseMatrix inverse operator (``solver * rhs``) or
    via ``Solve()`` for detailed convergence results.

    Auto-dispatches to real/complex based on ``mat.IsComplex()``.

    Args:
        mat: SPD sparse matrix (real or complex).
        method: ``"ICCG"``, ``"SGSMRTR"``, ``"CG"``, or ``"COCR"``.
        freedofs: Free DOFs.
        tol: Convergence tolerance (default: 1e-10).
        maxiter: Maximum iterations (default: 1000).
        shift: IC shift parameter (default: 1.05).
        save_best_result: Track and return best iterate (default: True).
        save_residual_history: Record residual per iteration (default: False).
        printrates: Print convergence info to stdout (default: False).
        conjugate: Conjugated inner product for Hermitian systems (default: False).
        use_abmc: Enable ABMC parallel ordering (default: False).
        abmc_block_size: Rows per ABMC block (default: 4).
        abmc_num_colors: Target ABMC colors (default: 4).
        abmc_reorder_spmv: Reorder SpMV in ABMC space (default: False).
        abmc_use_rcm: Use RCM ordering (default: False).
    """
    ...

# =============================================================================
# COCR Solver (C++ native, NGSolve BaseMatrix interface)
# =============================================================================

class COCRSolverD(BaseMatrix):
    """COCR solver for real (double) matrices."""

    @property
    def iterations(self) -> int:
        """Number of iterations performed in last solve."""
        ...

class COCRSolverC(BaseMatrix):
    """COCR solver for complex matrices."""

    @property
    def iterations(self) -> int:
        """Number of iterations performed in last solve."""
        ...

def COCRSolver(
    mat: BaseMatrix,
    pre: BaseMatrix,
    maxiter: int = 500,
    tol: float = 1e-8,
    printrates: bool = False,
) -> COCRSolverD | COCRSolverC:
    """COCR (Conjugate Orthogonal Conjugate Residual) solver.

    For complex-symmetric systems (A^T = A, NOT Hermitian).
    Uses unconjugated inner products. Minimizes ||A r~||_2.

    Auto-dispatches to real/complex based on ``mat.IsComplex()``.

    Usage (same as NGSolve CGSolver):
        inv = COCRSolver(mat, pre, maxiter=500, tol=1e-8)
        gfu.vec.data = inv * rhs.vec

    For COCG, use ``CGSolver(mat, pre, conjugate=False)`` instead.

    Args:
        mat: System matrix (real or complex).
        pre: Preconditioner (BaseMatrix).
        maxiter: Maximum iterations (default: 500).
        tol: Relative convergence tolerance (default: 1e-8).
        printrates: Print convergence info (default: False).
    """
    ...

# =============================================================================
# ComplexHypreAMSPreconditioner (TaskManager parallel Re/Im)
# =============================================================================

def ComplexHypreAMSPreconditioner(
    a_real_mat: BaseMatrix,
    grad_mat: BaseMatrix,
    freedofs: BitArray | None = None,
    coord_x: list[float] = ...,
    coord_y: list[float] = ...,
    coord_z: list[float] = ...,
    ndof_complex: int = 0,
    cycle_type: int = 1,
    print_level: int = 0,
) -> BaseMatrix:
    """Complex HYPRE AMS with TaskManager-parallel Re/Im splitting.

    Creates TWO independent HYPRE AMS instances and applies them
    to Re and Im parts concurrently via NGSolve TaskManager.
    Only available when built with ``SPARSESOLV_USE_HYPRE=ON``.

    For complex eddy-current systems ``A = K + jw*sigma*M``.
    Use with ``GMResSolver`` (HYPRE AMS is non-symmetric).

    Args:
        a_real_mat: Real SPD auxiliary matrix (K + eps*M + |omega|*sigma*M).
        grad_mat: Discrete gradient G (HCurl -> H1).
        freedofs: Free DOFs mask.
        coord_x: Vertex x-coordinates.
        coord_y: Vertex y-coordinates.
        coord_z: Vertex z-coordinates.
        ndof_complex: Complex DOF count.
        cycle_type: HYPRE AMS cycle type (default: 1).
        print_level: HYPRE print level (default: 0).
    """
    ...

# =============================================================================
# HYPRE AMS Preconditioner (conditional: SPARSESOLV_USE_HYPRE)
# =============================================================================

def HypreAMSPreconditioner(
    mat: BaseMatrix,
    grad_mat: BaseMatrix,
    freedofs: BitArray | None = None,
    coord_x: list[float] = ...,
    coord_y: list[float] = ...,
    coord_z: list[float] = ...,
    cycle_type: int = 1,
    print_level: int = 0,
) -> BaseMatrix:
    """HYPRE AMS (Auxiliary-space Maxwell Solver) Preconditioner.

    Uses HYPRE's BoomerAMG-based AMS for real HCurl systems.
    Only available when built with ``SPARSESOLV_USE_HYPRE=ON``.
    Check availability with ``has_hypre()``.

    Args:
        mat: Real SPD sparse matrix.
        grad_mat: Discrete gradient G (HCurl -> H1).
        freedofs: Free DOFs mask.
        coord_x: Vertex x-coordinates.
        coord_y: Vertex y-coordinates.
        coord_z: Vertex z-coordinates.
        cycle_type: HYPRE AMS cycle type (default: 1).
        print_level: HYPRE print level (default: 0).
    """
    ...

def HypreBoomerAMGPreconditioner(
    mat: BaseMatrix,
    freedofs: BitArray | None = None,
    print_level: int = 0,
    relax_type: int = 6,
    coarsen_type: int = 10,
    num_functions: int = 1,
    dof_func: list[int] | None = None,
) -> BaseMatrix:
    """HYPRE BoomerAMG Preconditioner for H1 scalar elliptic systems.

    Only available when built with ``SPARSESOLV_USE_HYPRE=ON``.

    Args:
        mat: Real SPD sparse matrix.
        freedofs: Free DOFs mask.
        print_level: HYPRE print level (default: 0).
        relax_type: Smoother type (default: 6, symmetric GS).
        coarsen_type: Coarsening type (default: 10, HMIS).
        num_functions: Block size for systems AMG (default: 1).
        dof_func: DOF-to-function mapping for systems AMG.
    """
    ...

def has_hypre() -> bool:
    """Returns True if HYPRE support is available (built with SPARSESOLV_USE_HYPRE)."""
    ...
