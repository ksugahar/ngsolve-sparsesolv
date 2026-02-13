/**
 * @file python_bindings.cpp
 * @brief SparseSolv pybind11 bindings for NGSolve integration
 *
 * REFERENCE COPY - This file is NOT compiled directly.
 * It shows the pybind11 code that should be appended to NGSolve's
 * linalg/python_linalg.cpp inside the ExportNgla() function.
 *
 * Prerequisites in python_linalg.cpp:
 *   #include "sparsesolv_precond.hpp"  // at the top of the file
 *
 * The code below should be placed before the closing brace of ExportNgla().
 */

// ==========================================================================
// SparseSolv Preconditioners (double)
// ==========================================================================

  py::class_<SparseSolvICPreconditioner<double>,
             shared_ptr<SparseSolvICPreconditioner<double>>,
             BaseMatrix>(m, "ICPreconditioner",
    R"raw_string(
Incomplete Cholesky (IC) Preconditioner using shifted IC decomposition.

Based on SparseSolv library by JP-MARs.

Supports Dirichlet boundary conditions through the freedofs parameter.

Example usage:

.. code-block:: python

    from ngsolve import *
    from ngsolve.krylovspace import CGSolver

    # Create bilinear form with Dirichlet BC
    fes = H1(mesh, order=2, dirichlet="left|right|top|bottom")
    u, v = fes.TnT()
    a = BilinearForm(fes)
    a += grad(u)*grad(v)*dx
    a.Assemble()

    # Create IC preconditioner with FreeDofs
    pre = ICPreconditioner(a.mat, freedofs=fes.FreeDofs(), shift=1.1)
    pre.Update()

    # Use with CGSolver
    inv = CGSolver(a.mat, pre, printrates=True, tol=1e-10)
    gfu.vec.data = inv * f.vec

Parameters:

mat : ngsolve.la.SparseMatrix
  The sparse matrix to precondition (must be SPD on free DOFs)

freedofs : ngsolve.BitArray, optional
  BitArray indicating free DOFs. Constrained DOFs are treated as identity.
  If None, all DOFs are considered free.

shift : float
  Shift parameter for IC decomposition (default: 1.05).
  Larger values improve stability but reduce preconditioner quality.

)raw_string")
    .def(py::init([](py::object mat, py::object freedofs, double shift) {
      auto sp_mat = py::cast<shared_ptr<SparseMatrix<double>>>(mat);
      shared_ptr<BitArray> sp_freedofs = nullptr;
      if (!freedofs.is_none()) {
        sp_freedofs = py::cast<shared_ptr<BitArray>>(freedofs);
      }
      return make_shared<SparseSolvICPreconditioner<double>>(sp_mat, sp_freedofs, shift);
    }), py::arg("mat"), py::arg("freedofs") = py::none(), py::arg("shift") = 1.05,
        "Create IC preconditioner from SparseMatrix")
    .def("Update", &SparseSolvICPreconditioner<double>::Update,
        "Update preconditioner (recompute factorization after matrix change)")
    .def_property("shift",
        &SparseSolvICPreconditioner<double>::GetShift,
        &SparseSolvICPreconditioner<double>::SetShift,
        "Shift parameter for IC decomposition");

  py::class_<SparseSolvSGSPreconditioner<double>,
             shared_ptr<SparseSolvSGSPreconditioner<double>>,
             BaseMatrix>(m, "SGSPreconditioner",
    R"raw_string(
Symmetric Gauss-Seidel (SGS) Preconditioner.

Based on SparseSolv library by JP-MARs.

Supports Dirichlet boundary conditions through the freedofs parameter.

Example usage:

.. code-block:: python

    from ngsolve import *
    from ngsolve.krylovspace import CGSolver

    # Problem with Dirichlet BC
    fes = H1(mesh, order=2, dirichlet="left|right|top|bottom")
    u, v = fes.TnT()
    a = BilinearForm(fes)
    a += grad(u)*grad(v)*dx
    a.Assemble()

    pre = SGSPreconditioner(a.mat, freedofs=fes.FreeDofs())
    pre.Update()
    inv = CGSolver(a.mat, pre, printrates=True)
    gfu.vec.data = inv * f.vec

Parameters:

mat : ngsolve.la.SparseMatrix
  The sparse matrix to precondition (must be SPD on free DOFs)

freedofs : ngsolve.BitArray, optional
  BitArray indicating free DOFs. Constrained DOFs are treated as identity.
  If None, all DOFs are considered free.

)raw_string")
    .def(py::init([](py::object mat, py::object freedofs) {
      auto sp_mat = py::cast<shared_ptr<SparseMatrix<double>>>(mat);
      shared_ptr<BitArray> sp_freedofs = nullptr;
      if (!freedofs.is_none()) {
        sp_freedofs = py::cast<shared_ptr<BitArray>>(freedofs);
      }
      return make_shared<SparseSolvSGSPreconditioner<double>>(sp_mat, sp_freedofs);
    }), py::arg("mat"), py::arg("freedofs") = py::none(),
        "Create SGS preconditioner from SparseMatrix")
    .def("Update", &SparseSolvSGSPreconditioner<double>::Update,
        "Update preconditioner (recompute after matrix change)");

  py::class_<SparseSolvILUPreconditioner<double>,
             shared_ptr<SparseSolvILUPreconditioner<double>>,
             BaseMatrix>(m, "ILUPreconditioner",
    R"raw_string(
Incomplete LU (ILU) Preconditioner using shifted ILU decomposition.

Based on SparseSolv library by JP-MARs.

Suitable for general (non-symmetric) matrices. For symmetric positive definite
matrices, ICPreconditioner is more efficient.

Supports Dirichlet boundary conditions through the freedofs parameter.

Example usage:

.. code-block:: python

    from ngsolve import *
    from ngsolve.krylovspace import GMRESSolver

    # Problem with Dirichlet BC
    fes = H1(mesh, order=2, dirichlet="left|right|top|bottom")
    u, v = fes.TnT()
    a = BilinearForm(fes)
    a += grad(u)*grad(v)*dx + u*v*dx  # Non-symmetric possible
    a.Assemble()

    pre = ILUPreconditioner(a.mat, freedofs=fes.FreeDofs(), shift=1.1)
    pre.Update()
    inv = GMRESSolver(a.mat, pre, printrates=True, tol=1e-10)
    gfu.vec.data = inv * f.vec

Parameters:

mat : ngsolve.la.SparseMatrix
  The sparse matrix to precondition

freedofs : ngsolve.BitArray, optional
  BitArray indicating free DOFs. Constrained DOFs are treated as identity.
  If None, all DOFs are considered free.

shift : float
  Shift parameter for ILU decomposition (default: 1.05).
  Larger values improve stability but reduce preconditioner quality.

)raw_string")
    .def(py::init([](py::object mat, py::object freedofs, double shift) {
      auto sp_mat = py::cast<shared_ptr<SparseMatrix<double>>>(mat);
      shared_ptr<BitArray> sp_freedofs = nullptr;
      if (!freedofs.is_none()) {
        sp_freedofs = py::cast<shared_ptr<BitArray>>(freedofs);
      }
      return make_shared<SparseSolvILUPreconditioner<double>>(sp_mat, sp_freedofs, shift);
    }), py::arg("mat"), py::arg("freedofs") = py::none(), py::arg("shift") = 1.05,
        "Create ILU preconditioner from SparseMatrix")
    .def("Update", &SparseSolvILUPreconditioner<double>::Update,
        "Update preconditioner (recompute factorization after matrix change)")
    .def_property("shift",
        &SparseSolvILUPreconditioner<double>::GetShift,
        &SparseSolvILUPreconditioner<double>::SetShift,
        "Shift parameter for ILU decomposition");

  // ==========================================================================
  // SparseSolv Solver Result
  // ==========================================================================

  py::class_<SparseSolvResult>(m, "SparseSolvResult",
    R"raw_string(
Result of a SparseSolv iterative solve.

Attributes:

converged : bool
  Whether the solver converged within tolerance.

iterations : int
  Number of iterations performed.

final_residual : float
  Final relative residual. If save_best_result was enabled and the solver
  did not converge, this is the best residual achieved during iteration.

residual_history : list[float]
  Residual at each iteration (only if save_residual_history=True).

)raw_string")
    .def_readonly("converged", &SparseSolvResult::converged,
        "Whether the solver converged within tolerance")
    .def_readonly("iterations", &SparseSolvResult::iterations,
        "Number of iterations performed")
    .def_readonly("final_residual", &SparseSolvResult::final_residual,
        "Final relative residual (or best residual if save_best_result enabled)")
    .def_readonly("residual_history", &SparseSolvResult::residual_history,
        "Residual at each iteration (if save_residual_history enabled)")
    .def("__repr__", [](const SparseSolvResult& r) {
      return string("SparseSolvResult(converged=") +
             (r.converged ? "True" : "False") +
             ", iterations=" + std::to_string(r.iterations) +
             ", residual=" + std::to_string(r.final_residual) + ")";
    });

  // ==========================================================================
  // SparseSolv Iterative Solver (double)
  // ==========================================================================

  py::class_<SparseSolvSolver<double>,
             shared_ptr<SparseSolvSolver<double>>,
             BaseMatrix>(m, "SparseSolvSolver",
    R"raw_string(
Iterative solver using the SparseSolv library by JP-MARs.

Supports multiple solver methods for symmetric positive definite systems:
- ICCG: Conjugate Gradient + Incomplete Cholesky preconditioner
- ICMRTR: MRTR (Modified Residual Tri-diagonal Reduction) + IC preconditioner
- SGSMRTR: MRTR with built-in Symmetric Gauss-Seidel (split formula)
- CG: Conjugate Gradient without preconditioner
- MRTR: MRTR without preconditioner

Key features:
- save_best_result (default: True): During iteration, tracks the best
  solution vector (lowest residual). If the solver doesn't converge, the
  best solution found is returned instead of the last iterate.
- save_residual_history: Records residual at every iteration for analysis.
- FreeDofs support for Dirichlet boundary conditions.

Can be used as an inverse operator (BaseMatrix) or with Solve() method.

Example usage as inverse operator:

.. code-block:: python

    from ngsolve import *

    fes = H1(mesh, order=2, dirichlet="left|right|top|bottom")
    u, v = fes.TnT()
    a = BilinearForm(fes)
    a += grad(u)*grad(v)*dx
    a.Assemble()
    f = LinearForm(fes)
    f += 1*v*dx
    f.Assemble()

    gfu = GridFunction(fes)
    solver = SparseSolvSolver(a.mat, method="ICCG",
                              freedofs=fes.FreeDofs(), tol=1e-10)
    gfu.vec.data = solver * f.vec

Example usage with Solve() for detailed results:

.. code-block:: python

    solver = SparseSolvSolver(a.mat, method="ICCG",
                              freedofs=fes.FreeDofs(), tol=1e-10,
                              save_residual_history=True)
    result = solver.Solve(f.vec, gfu.vec)
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Final residual: {result.final_residual}")

Parameters:

mat : ngsolve.la.SparseMatrix
  The sparse system matrix (must be SPD for CG/MRTR methods).

method : str
  Solver method. One of: "ICCG", "ICMRTR", "SGSMRTR", "CG", "MRTR".

freedofs : ngsolve.BitArray, optional
  BitArray indicating free DOFs. Constrained DOFs are treated as identity.

tol : float
  Relative convergence tolerance (default: 1e-10).

maxiter : int
  Maximum number of iterations (default: 1000).

shift : float
  Shift parameter for IC preconditioner (default: 1.05).
  Only used with ICCG and ICMRTR methods.

save_best_result : bool
  Track best solution during iteration (default: True).

save_residual_history : bool
  Record residual at each iteration (default: False).

printrates : bool
  Print convergence information after solve (default: False).

)raw_string")
    .def(py::init([](py::object mat, const string& method, py::object freedofs,
                     double tol, int maxiter, double shift,
                     bool save_best_result, bool save_residual_history,
                     bool printrates) {
      auto sp_mat = py::cast<shared_ptr<SparseMatrix<double>>>(mat);
      shared_ptr<BitArray> sp_freedofs = nullptr;
      if (!freedofs.is_none()) {
        sp_freedofs = py::cast<shared_ptr<BitArray>>(freedofs);
      }
      return make_shared<SparseSolvSolver<double>>(
          sp_mat, method, sp_freedofs, tol, maxiter, shift,
          save_best_result, save_residual_history, printrates);
    }),
        py::arg("mat"),
        py::arg("method") = "ICCG",
        py::arg("freedofs") = py::none(),
        py::arg("tol") = 1e-10,
        py::arg("maxiter") = 1000,
        py::arg("shift") = 1.05,
        py::arg("save_best_result") = true,
        py::arg("save_residual_history") = false,
        py::arg("printrates") = false,
        "Create SparseSolv iterative solver")
    .def("Solve", [](SparseSolvSolver<double>& self,
                     const BaseVector& rhs, BaseVector& sol) {
      return self.Solve(rhs, sol);
    }, py::arg("rhs"), py::arg("sol"),
        R"raw_string(
Solve the linear system Ax = b with initial guess.

Parameters:

rhs : ngsolve.BaseVector
  Right-hand side vector b.

sol : ngsolve.BaseVector
  Solution vector x. Used as initial guess on input, contains
  solution on output. For constrained DOFs (not in freedofs),
  values are preserved.

Returns:

SparseSolvResult
  Result object with convergence info, iteration count, and residual.
)raw_string")
    .def_property("method",
        &SparseSolvSolver<double>::GetMethod,
        &SparseSolvSolver<double>::SetMethod,
        "Solver method: ICCG, ICMRTR, SGSMRTR, CG, MRTR")
    .def_property("tol",
        &SparseSolvSolver<double>::GetTolerance,
        &SparseSolvSolver<double>::SetTolerance,
        "Relative convergence tolerance")
    .def_property("maxiter",
        &SparseSolvSolver<double>::GetMaxIterations,
        &SparseSolvSolver<double>::SetMaxIterations,
        "Maximum number of iterations")
    .def_property("shift",
        &SparseSolvSolver<double>::GetShift,
        &SparseSolvSolver<double>::SetShift,
        "Shift parameter for IC preconditioner")
    .def_property("save_best_result",
        &SparseSolvSolver<double>::GetSaveBestResult,
        &SparseSolvSolver<double>::SetSaveBestResult,
        "Track best solution during iteration")
    .def_property("save_residual_history",
        &SparseSolvSolver<double>::GetSaveResidualHistory,
        &SparseSolvSolver<double>::SetSaveResidualHistory,
        "Record residual at each iteration")
    .def_property("printrates",
        &SparseSolvSolver<double>::GetPrintRates,
        &SparseSolvSolver<double>::SetPrintRates,
        "Print convergence information after solve")
    .def_property("auto_shift",
        &SparseSolvSolver<double>::GetAutoShift,
        &SparseSolvSolver<double>::SetAutoShift,
        "Enable automatic shift adjustment for IC decomposition (for semi-definite matrices)")
    .def_property("diagonal_scaling",
        &SparseSolvSolver<double>::GetDiagonalScaling,
        &SparseSolvSolver<double>::SetDiagonalScaling,
        "Enable diagonal scaling (1/sqrt(A[i,i])) for IC preconditioner")
    .def_property_readonly("last_result",
        &SparseSolvSolver<double>::GetLastResult,
        "Result from the last Solve() or Mult() call");

  // ==========================================================================
  // SparseSolv Preconditioners (Complex)
  // ==========================================================================

  py::class_<SparseSolvICPreconditioner<Complex>,
             shared_ptr<SparseSolvICPreconditioner<Complex>>,
             BaseMatrix>(m, "ComplexICPreconditioner",
    R"raw_string(
Incomplete Cholesky (IC) Preconditioner for complex-valued matrices.

Based on SparseSolv library by JP-MARs.

Complex version of ICPreconditioner. Suitable for Hermitian positive definite
matrices from eddy current or electromagnetic problems.

Parameters:

mat : ngsolve.la.SparseMatrix
  Complex sparse matrix to precondition

freedofs : ngsolve.BitArray, optional
  BitArray indicating free DOFs. Constrained DOFs are treated as identity.

shift : float
  Shift parameter for IC decomposition (default: 1.05).

)raw_string")
    .def(py::init([](py::object mat, py::object freedofs, double shift) {
      auto sp_mat = py::cast<shared_ptr<SparseMatrix<Complex>>>(mat);
      shared_ptr<BitArray> sp_freedofs = nullptr;
      if (!freedofs.is_none()) {
        sp_freedofs = py::cast<shared_ptr<BitArray>>(freedofs);
      }
      return make_shared<SparseSolvICPreconditioner<Complex>>(sp_mat, sp_freedofs, shift);
    }), py::arg("mat"), py::arg("freedofs") = py::none(), py::arg("shift") = 1.05,
        "Create complex IC preconditioner from SparseMatrix")
    .def("Update", &SparseSolvICPreconditioner<Complex>::Update,
        "Update preconditioner (recompute factorization after matrix change)")
    .def_property("shift",
        &SparseSolvICPreconditioner<Complex>::GetShift,
        &SparseSolvICPreconditioner<Complex>::SetShift,
        "Shift parameter for IC decomposition");

  py::class_<SparseSolvSGSPreconditioner<Complex>,
             shared_ptr<SparseSolvSGSPreconditioner<Complex>>,
             BaseMatrix>(m, "ComplexSGSPreconditioner",
    R"raw_string(
Symmetric Gauss-Seidel (SGS) Preconditioner for complex-valued matrices.

Based on SparseSolv library by JP-MARs.

Complex version of SGSPreconditioner.

Parameters:

mat : ngsolve.la.SparseMatrix
  Complex sparse matrix to precondition

freedofs : ngsolve.BitArray, optional
  BitArray indicating free DOFs. Constrained DOFs are treated as identity.

)raw_string")
    .def(py::init([](py::object mat, py::object freedofs) {
      auto sp_mat = py::cast<shared_ptr<SparseMatrix<Complex>>>(mat);
      shared_ptr<BitArray> sp_freedofs = nullptr;
      if (!freedofs.is_none()) {
        sp_freedofs = py::cast<shared_ptr<BitArray>>(freedofs);
      }
      return make_shared<SparseSolvSGSPreconditioner<Complex>>(sp_mat, sp_freedofs);
    }), py::arg("mat"), py::arg("freedofs") = py::none(),
        "Create complex SGS preconditioner from SparseMatrix")
    .def("Update", &SparseSolvSGSPreconditioner<Complex>::Update,
        "Update preconditioner (recompute after matrix change)");

  py::class_<SparseSolvILUPreconditioner<Complex>,
             shared_ptr<SparseSolvILUPreconditioner<Complex>>,
             BaseMatrix>(m, "ComplexILUPreconditioner",
    R"raw_string(
Incomplete LU (ILU) Preconditioner for complex-valued matrices.

Based on SparseSolv library by JP-MARs.

Complex version of ILUPreconditioner. Suitable for general (non-Hermitian)
complex matrices from electromagnetic problems.

Parameters:

mat : ngsolve.la.SparseMatrix
  Complex sparse matrix to precondition

freedofs : ngsolve.BitArray, optional
  BitArray indicating free DOFs. Constrained DOFs are treated as identity.

shift : float
  Shift parameter for ILU decomposition (default: 1.05).

)raw_string")
    .def(py::init([](py::object mat, py::object freedofs, double shift) {
      auto sp_mat = py::cast<shared_ptr<SparseMatrix<Complex>>>(mat);
      shared_ptr<BitArray> sp_freedofs = nullptr;
      if (!freedofs.is_none()) {
        sp_freedofs = py::cast<shared_ptr<BitArray>>(freedofs);
      }
      return make_shared<SparseSolvILUPreconditioner<Complex>>(sp_mat, sp_freedofs, shift);
    }), py::arg("mat"), py::arg("freedofs") = py::none(), py::arg("shift") = 1.05,
        "Create complex ILU preconditioner from SparseMatrix")
    .def("Update", &SparseSolvILUPreconditioner<Complex>::Update,
        "Update preconditioner (recompute factorization after matrix change)")
    .def_property("shift",
        &SparseSolvILUPreconditioner<Complex>::GetShift,
        &SparseSolvILUPreconditioner<Complex>::SetShift,
        "Shift parameter for ILU decomposition");

  // ==========================================================================
  // SparseSolv Iterative Solver (Complex)
  // ==========================================================================

  py::class_<SparseSolvSolver<Complex>,
             shared_ptr<SparseSolvSolver<Complex>>,
             BaseMatrix>(m, "ComplexSparseSolvSolver",
    R"raw_string(
Iterative solver for complex-valued systems using the SparseSolv library.

Complex version of SparseSolvSolver. Suitable for electromagnetic problems
such as eddy current analysis, frequency-domain Maxwell's equations, etc.

Supports the same solver methods as SparseSolvSolver:
- ICCG, ICMRTR, SGSMRTR, CG, MRTR

Example usage:

.. code-block:: python

    from ngsolve import *

    fes = HCurl(mesh, order=2, complex=True)
    u, v = fes.TnT()
    a = BilinearForm(fes)
    a += curl(u)*curl(v)*dx + 1j*sigma*u*v*dx
    a.Assemble()

    solver = ComplexSparseSolvSolver(a.mat, method="ICCG",
                                     freedofs=fes.FreeDofs(), tol=1e-10)
    gfu.vec.data = solver * f.vec

Parameters:

mat : ngsolve.la.SparseMatrix
  Complex sparse system matrix.

method : str
  Solver method. One of: "ICCG", "ICMRTR", "SGSMRTR", "CG", "MRTR".

freedofs : ngsolve.BitArray, optional
  BitArray indicating free DOFs.

tol : float
  Relative convergence tolerance (default: 1e-10).

maxiter : int
  Maximum number of iterations (default: 1000).

shift : float
  Shift parameter for IC preconditioner (default: 1.05).

save_best_result : bool
  Track best solution during iteration (default: True).

save_residual_history : bool
  Record residual at each iteration (default: False).

printrates : bool
  Print convergence information after solve (default: False).

)raw_string")
    .def(py::init([](py::object mat, const string& method, py::object freedofs,
                     double tol, int maxiter, double shift,
                     bool save_best_result, bool save_residual_history,
                     bool printrates) {
      auto sp_mat = py::cast<shared_ptr<SparseMatrix<Complex>>>(mat);
      shared_ptr<BitArray> sp_freedofs = nullptr;
      if (!freedofs.is_none()) {
        sp_freedofs = py::cast<shared_ptr<BitArray>>(freedofs);
      }
      return make_shared<SparseSolvSolver<Complex>>(
          sp_mat, method, sp_freedofs, tol, maxiter, shift,
          save_best_result, save_residual_history, printrates);
    }),
        py::arg("mat"),
        py::arg("method") = "ICCG",
        py::arg("freedofs") = py::none(),
        py::arg("tol") = 1e-10,
        py::arg("maxiter") = 1000,
        py::arg("shift") = 1.05,
        py::arg("save_best_result") = true,
        py::arg("save_residual_history") = false,
        py::arg("printrates") = false,
        "Create complex SparseSolv iterative solver")
    .def("Solve", [](SparseSolvSolver<Complex>& self,
                     const BaseVector& rhs, BaseVector& sol) {
      return self.Solve(rhs, sol);
    }, py::arg("rhs"), py::arg("sol"),
        R"raw_string(
Solve the complex linear system Ax = b with initial guess.

Parameters:

rhs : ngsolve.BaseVector
  Right-hand side vector b (complex).

sol : ngsolve.BaseVector
  Solution vector x (complex). Used as initial guess on input.

Returns:

SparseSolvResult
  Result object with convergence info, iteration count, and residual.
)raw_string")
    .def_property("method",
        &SparseSolvSolver<Complex>::GetMethod,
        &SparseSolvSolver<Complex>::SetMethod,
        "Solver method: ICCG, ICMRTR, SGSMRTR, CG, MRTR")
    .def_property("tol",
        &SparseSolvSolver<Complex>::GetTolerance,
        &SparseSolvSolver<Complex>::SetTolerance,
        "Relative convergence tolerance")
    .def_property("maxiter",
        &SparseSolvSolver<Complex>::GetMaxIterations,
        &SparseSolvSolver<Complex>::SetMaxIterations,
        "Maximum number of iterations")
    .def_property("shift",
        &SparseSolvSolver<Complex>::GetShift,
        &SparseSolvSolver<Complex>::SetShift,
        "Shift parameter for IC preconditioner")
    .def_property("save_best_result",
        &SparseSolvSolver<Complex>::GetSaveBestResult,
        &SparseSolvSolver<Complex>::SetSaveBestResult,
        "Track best solution during iteration")
    .def_property("save_residual_history",
        &SparseSolvSolver<Complex>::GetSaveResidualHistory,
        &SparseSolvSolver<Complex>::SetSaveResidualHistory,
        "Record residual at each iteration")
    .def_property("printrates",
        &SparseSolvSolver<Complex>::GetPrintRates,
        &SparseSolvSolver<Complex>::SetPrintRates,
        "Print convergence information after solve")
    .def_property("auto_shift",
        &SparseSolvSolver<Complex>::GetAutoShift,
        &SparseSolvSolver<Complex>::SetAutoShift,
        "Enable automatic shift adjustment for IC decomposition")
    .def_property("diagonal_scaling",
        &SparseSolvSolver<Complex>::GetDiagonalScaling,
        &SparseSolvSolver<Complex>::SetDiagonalScaling,
        "Enable diagonal scaling for IC preconditioner")
    .def_property_readonly("last_result",
        &SparseSolvSolver<Complex>::GetLastResult,
        "Result from the last Solve() or Mult() call");
