/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/// @file sparsesolv_python_export.hpp
/// @brief Pybind11 bindings: type registration + factory functions with auto-dispatch

#ifndef NGSOLVE_SPARSESOLV_PYTHON_EXPORT_HPP
#define NGSOLVE_SPARSESOLV_PYTHON_EXPORT_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "sparsesolv_precond.hpp"
#include "sparsesolv_solvers.hpp"
#include <comp.hpp>
#include <type_traits>

#ifdef SPARSESOLV_USE_HYPRE
#include "sparsesolv/preconditioners/hypre_ams_preconditioner.hpp"
#include "sparsesolv/preconditioners/hypre_boomeramg_preconditioner.hpp"
#endif


namespace py = pybind11;

namespace ngla {

// ============================================================================
// Internal: SparseSolvResult (non-templated, called once)
// ============================================================================

inline void ExportSparseSolvResult_impl(py::module& m) {
  py::class_<SparseSolvResult>(m, "SparseSolvResult",
    "Result of a SparseSolv iterative solve (converged, iterations, final_residual, residual_history).")
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
}

// ============================================================================
// Internal: Type registration (D/C suffix, no constructors — use factories)
// ============================================================================

template<typename SCAL>
void ExportSparseSolvTyped(py::module& m, const std::string& suffix) {

  // IC Preconditioner type registration (factory creates instances, methods accessible via downcast)
  {
    std::string cls_name = "ICPreconditioner" + suffix;
    py::class_<SparseSolvICPreconditioner<SCAL>,
               shared_ptr<SparseSolvICPreconditioner<SCAL>>,
               BaseMatrix>(m, cls_name.c_str())
      .def("Update", &SparseSolvICPreconditioner<SCAL>::Update)
      .def_property("shift",
          &SparseSolvICPreconditioner<SCAL>::GetShift,
          &SparseSolvICPreconditioner<SCAL>::SetShift)
      .def_property("use_abmc",
          &SparseSolvICPreconditioner<SCAL>::GetUseABMC,
          &SparseSolvICPreconditioner<SCAL>::SetUseABMC)
      .def_property("abmc_block_size",
          &SparseSolvICPreconditioner<SCAL>::GetABMCBlockSize,
          &SparseSolvICPreconditioner<SCAL>::SetABMCBlockSize)
      .def_property("abmc_num_colors",
          &SparseSolvICPreconditioner<SCAL>::GetABMCNumColors,
          &SparseSolvICPreconditioner<SCAL>::SetABMCNumColors)
      .def_property("diagonal_scaling",
          &SparseSolvICPreconditioner<SCAL>::GetDiagonalScaling,
          &SparseSolvICPreconditioner<SCAL>::SetDiagonalScaling);
  }

  // SGS Preconditioner type registration
  {
    std::string cls_name = "SGSPreconditioner" + suffix;
    py::class_<SparseSolvSGSPreconditioner<SCAL>,
               shared_ptr<SparseSolvSGSPreconditioner<SCAL>>,
               BaseMatrix>(m, cls_name.c_str())
      .def("Update", &SparseSolvSGSPreconditioner<SCAL>::Update);
  }

  // SparseSolv Solver type registration
  {
    std::string cls_name = "SparseSolvSolver" + suffix;
    py::class_<SparseSolvSolver<SCAL>,
               shared_ptr<SparseSolvSolver<SCAL>>,
               BaseMatrix>(m, cls_name.c_str())
      .def("Solve", [](SparseSolvSolver<SCAL>& self,
                       const BaseVector& rhs, BaseVector& sol) {
        return self.Solve(rhs, sol);
      }, py::arg("rhs"), py::arg("sol"))
      .def_property("method",
          &SparseSolvSolver<SCAL>::GetMethod,
          &SparseSolvSolver<SCAL>::SetMethod)
      .def_property("tol",
          &SparseSolvSolver<SCAL>::GetTolerance,
          &SparseSolvSolver<SCAL>::SetTolerance)
      .def_property("maxiter",
          &SparseSolvSolver<SCAL>::GetMaxIterations,
          &SparseSolvSolver<SCAL>::SetMaxIterations)
      .def_property("shift",
          &SparseSolvSolver<SCAL>::GetShift,
          &SparseSolvSolver<SCAL>::SetShift)
      .def_property("save_best_result",
          &SparseSolvSolver<SCAL>::GetSaveBestResult,
          &SparseSolvSolver<SCAL>::SetSaveBestResult)
      .def_property("save_residual_history",
          &SparseSolvSolver<SCAL>::GetSaveResidualHistory,
          &SparseSolvSolver<SCAL>::SetSaveResidualHistory)
      .def_property("printrates",
          &SparseSolvSolver<SCAL>::GetPrintRates,
          &SparseSolvSolver<SCAL>::SetPrintRates)
      .def_property("auto_shift",
          &SparseSolvSolver<SCAL>::GetAutoShift,
          &SparseSolvSolver<SCAL>::SetAutoShift)
      .def_property("diagonal_scaling",
          &SparseSolvSolver<SCAL>::GetDiagonalScaling,
          &SparseSolvSolver<SCAL>::SetDiagonalScaling)
      .def_property("divergence_check",
          &SparseSolvSolver<SCAL>::GetDivergenceCheck,
          &SparseSolvSolver<SCAL>::SetDivergenceCheck)
      .def_property("divergence_threshold",
          &SparseSolvSolver<SCAL>::GetDivergenceThreshold,
          &SparseSolvSolver<SCAL>::SetDivergenceThreshold)
      .def_property("divergence_count",
          &SparseSolvSolver<SCAL>::GetDivergenceCount,
          &SparseSolvSolver<SCAL>::SetDivergenceCount)
      .def_property("conjugate",
          &SparseSolvSolver<SCAL>::GetConjugate,
          &SparseSolvSolver<SCAL>::SetConjugate)
      .def_property("use_abmc",
          &SparseSolvSolver<SCAL>::GetUseABMC,
          &SparseSolvSolver<SCAL>::SetUseABMC)
      .def_property("abmc_block_size",
          &SparseSolvSolver<SCAL>::GetABMCBlockSize,
          &SparseSolvSolver<SCAL>::SetABMCBlockSize)
      .def_property("abmc_num_colors",
          &SparseSolvSolver<SCAL>::GetABMCNumColors,
          &SparseSolvSolver<SCAL>::SetABMCNumColors)
      .def_property("abmc_reorder_spmv",
          &SparseSolvSolver<SCAL>::GetABMCReorderSpMV,
          &SparseSolvSolver<SCAL>::SetABMCReorderSpMV)
      .def_property("abmc_use_rcm",
          &SparseSolvSolver<SCAL>::GetABMCUseRCM,
          &SparseSolvSolver<SCAL>::SetABMCUseRCM)
      .def_property_readonly("last_result",
          &SparseSolvSolver<SCAL>::GetLastResult);
  }
}

// ============================================================================
// Internal helpers
// ============================================================================

inline shared_ptr<BitArray> ExtractFreeDofs(py::object freedofs) {
  if (freedofs.is_none()) return nullptr;
  return py::cast<shared_ptr<BitArray>>(freedofs);
}

// ============================================================================
// Internal: Factory functions with auto-dispatch via mat->IsComplex()
// ============================================================================

inline void ExportSparseSolvFactories(py::module& m) {

  // ---- ICPreconditioner factory ----
  m.def("ICPreconditioner", [](shared_ptr<BaseMatrix> mat,
                                py::object freedofs, double shift) {
    auto sp_freedofs = ExtractFreeDofs(freedofs);
    shared_ptr<BaseMatrix> result;
    if (mat->IsComplex()) {
      auto sp = dynamic_pointer_cast<SparseMatrix<Complex>>(mat);
      if (!sp) throw py::type_error("ICPreconditioner: expected SparseMatrix");
      auto p = make_shared<SparseSolvICPreconditioner<Complex>>(sp, sp_freedofs, shift);
      p->Update();
      result = p;
    } else {
      auto sp = dynamic_pointer_cast<SparseMatrix<double>>(mat);
      if (!sp) throw py::type_error("ICPreconditioner: expected SparseMatrix");
      auto p = make_shared<SparseSolvICPreconditioner<double>>(sp, sp_freedofs, shift);
      p->Update();
      result = p;
    }
    return result;
  },
  py::arg("mat"), py::arg("freedofs") = py::none(), py::arg("shift") = 1.05,
  R"raw_string(
Incomplete Cholesky (IC) Preconditioner.

Parameters:

mat : SparseMatrix
  SPD matrix (real or complex, auto-detected).
freedofs : BitArray, optional
  Free DOFs. Constrained DOFs treated as identity.
shift : float
  Shift parameter (default: 1.05).
)raw_string");

  // ---- SGSPreconditioner factory ----
  m.def("SGSPreconditioner", [](shared_ptr<BaseMatrix> mat,
                                  py::object freedofs) {
    auto sp_freedofs = ExtractFreeDofs(freedofs);
    shared_ptr<BaseMatrix> result;
    if (mat->IsComplex()) {
      auto sp = dynamic_pointer_cast<SparseMatrix<Complex>>(mat);
      if (!sp) throw py::type_error("SGSPreconditioner: expected SparseMatrix");
      auto p = make_shared<SparseSolvSGSPreconditioner<Complex>>(sp, sp_freedofs);
      p->Update();
      result = p;
    } else {
      auto sp = dynamic_pointer_cast<SparseMatrix<double>>(mat);
      if (!sp) throw py::type_error("SGSPreconditioner: expected SparseMatrix");
      auto p = make_shared<SparseSolvSGSPreconditioner<double>>(sp, sp_freedofs);
      p->Update();
      result = p;
    }
    return result;
  },
  py::arg("mat"), py::arg("freedofs") = py::none(),
  R"raw_string(
Symmetric Gauss-Seidel (SGS) Preconditioner.

Parameters:

mat : SparseMatrix
  SPD matrix (real or complex, auto-detected).
freedofs : BitArray, optional
  Free DOFs. Constrained DOFs treated as identity.
)raw_string");

  // ---- SparseSolvSolver factory ----
  m.def("SparseSolvSolver", [](shared_ptr<BaseMatrix> mat,
                                 const string& method, py::object freedofs,
                                 double tol, int maxiter, double shift,
                                 bool save_best_result, bool save_residual_history,
                                 bool printrates, bool conjugate,
                                 bool use_abmc, int abmc_block_size, int abmc_num_colors,
                                 bool abmc_reorder_spmv, bool abmc_use_rcm) {
    auto sp_freedofs = ExtractFreeDofs(freedofs);

    auto configure = [&](auto& solver) {
      solver->SetConjugate(conjugate);
      solver->SetUseABMC(use_abmc);
      solver->SetABMCBlockSize(abmc_block_size);
      solver->SetABMCNumColors(abmc_num_colors);
      solver->SetABMCReorderSpMV(abmc_reorder_spmv);
      solver->SetABMCUseRCM(abmc_use_rcm);
    };

    shared_ptr<BaseMatrix> result;
    if (mat->IsComplex()) {
      auto sp = dynamic_pointer_cast<SparseMatrix<Complex>>(mat);
      if (!sp) throw py::type_error("SparseSolvSolver: expected SparseMatrix");
      auto solver = make_shared<SparseSolvSolver<Complex>>(
          sp, method, sp_freedofs, tol, maxiter, shift,
          save_best_result, save_residual_history, printrates);
      configure(solver);
      result = solver;
    } else {
      auto sp = dynamic_pointer_cast<SparseMatrix<double>>(mat);
      if (!sp) throw py::type_error("SparseSolvSolver: expected SparseMatrix");
      auto solver = make_shared<SparseSolvSolver<double>>(
          sp, method, sp_freedofs, tol, maxiter, shift,
          save_best_result, save_residual_history, printrates);
      configure(solver);
      result = solver;
    }
    return result;
  },
  py::arg("mat"),
  py::arg("method") = "ICCG",
  py::arg("freedofs") = py::none(),
  py::arg("tol") = 1e-10,
  py::arg("maxiter") = 1000,
  py::arg("shift") = 1.05,
  py::arg("save_best_result") = true,
  py::arg("save_residual_history") = false,
  py::arg("printrates") = false,
  py::arg("conjugate") = false,
  py::arg("use_abmc") = false,
  py::arg("abmc_block_size") = 4,
  py::arg("abmc_num_colors") = 4,
  py::arg("abmc_reorder_spmv") = false,
  py::arg("abmc_use_rcm") = false,
  R"raw_string(
Iterative solver (ICCG / SGSMRTR / CG / COCR). Auto-detects real/complex.

Can be used as BaseMatrix (inverse operator) or via Solve() for detailed results.

Parameters:

mat : SparseMatrix
  SPD matrix (real or complex).
method : str
  "ICCG", "SGSMRTR", "CG", or "COCR".
  COCR: Conjugate Orthogonal Conjugate Residual for complex-symmetric A^T=A.
freedofs : BitArray, optional
  Free DOFs.
tol : float
  Convergence tolerance (default: 1e-10).
maxiter : int
  Max iterations (default: 1000).
shift : float
  IC shift parameter (default: 1.05).
save_best_result : bool
  Track best solution (default: True).
save_residual_history : bool
  Record residual history (default: False).
printrates : bool
  Print convergence info (default: False).
conjugate : bool
  Conjugated inner product for Hermitian systems (default: False).

Properties (set after construction):
  auto_shift, diagonal_scaling, divergence_check, divergence_threshold,
  divergence_count, use_abmc, abmc_block_size, abmc_num_colors,
  abmc_reorder_spmv, abmc_use_rcm.
)raw_string");
}

// ============================================================================
// HYPRE AMS factory (conditional on SPARSESOLV_USE_HYPRE)
// ============================================================================

#ifdef SPARSESOLV_USE_HYPRE
inline void ExportHypreAMS(py::module& m) {
  m.def("HypreAMSPreconditioner",
    [](shared_ptr<BaseMatrix> mat,
       shared_ptr<BaseMatrix> grad_mat,
       py::object freedofs_obj,
       py::list coord_x_list,
       py::list coord_y_list,
       py::list coord_z_list,
       int cycle_type,
       int print_level) -> shared_ptr<BaseMatrix>
    {
      auto sp_mat = dynamic_pointer_cast<SparseMatrix<double>>(mat);
      if (!sp_mat)
        throw py::type_error("HypreAMSPreconditioner: system matrix must be real SparseMatrix<double>");

      auto sp_grad = dynamic_pointer_cast<SparseMatrix<double>>(grad_mat);
      if (!sp_grad)
        throw py::type_error("HypreAMSPreconditioner: gradient matrix must be real SparseMatrix<double>");

      auto sp_freedofs = ExtractFreeDofs(freedofs_obj);

      // Convert Python lists to std::vector<double>
      auto to_vec = [](py::list lst) {
        std::vector<double> v(lst.size());
        for (size_t i = 0; i < lst.size(); i++)
          v[i] = lst[i].cast<double>();
        return v;
      };

      return make_shared<HypreAMSPreconditioner>(
          sp_mat, sp_grad, sp_freedofs,
          to_vec(coord_x_list), to_vec(coord_y_list), to_vec(coord_z_list),
          cycle_type, print_level);
    },
    py::arg("mat"),
    py::arg("grad_mat"),
    py::arg("freedofs") = py::none(),
    py::arg("coord_x") = py::list(),
    py::arg("coord_y") = py::list(),
    py::arg("coord_z") = py::list(),
    py::arg("cycle_type") = 1,
    py::arg("print_level") = 0,
    R"raw_string(
HYPRE AMS (Auxiliary-space Maxwell Solver) Preconditioner.

Uses HYPRE's AMS for real HCurl curl-curl + mass systems.
For complex problems, use Re/Im splitting at the Python level.

Parameters:

mat : SparseMatrix (real)
  Assembled system matrix (must be nonsymmetric storage).
grad_mat : SparseMatrix (real)
  Discrete gradient matrix (HCurl -> H1).
freedofs : BitArray, optional
  Free DOFs mask.
coord_x, coord_y, coord_z : list of float
  Vertex coordinates (length = number of H1 DOFs).
cycle_type : int
  AMS cycle type (1=additive, 2=multiplicative, default=1).
print_level : int
  HYPRE print level (0=silent, default=0).
)raw_string");

  m.def("has_hypre", []() { return true; },
    "Returns True if HYPRE support is available.");

  // BoomerAMG standalone preconditioner for H1 systems
  m.def("HypreBoomerAMGPreconditioner",
    [](shared_ptr<BaseMatrix> mat,
       py::object freedofs_obj,
       int print_level,
       int coarsen_type,
       int relax_type,
       int agg_levels,
       double strong_threshold,
       int interp_type,
       int max_levels,
       int num_sweeps,
       int coarse_relax_type,
       int num_functions,
       py::object dof_func_obj) -> shared_ptr<BaseMatrix>
    {
      auto sp_mat = dynamic_pointer_cast<SparseMatrix<double>>(mat);
      if (!sp_mat)
        throw py::type_error("HypreBoomerAMGPreconditioner: expected real SparseMatrix<double>");
      auto sp_freedofs = ExtractFreeDofs(freedofs_obj);

      std::vector<HYPRE_Int> dof_func;
      if (!dof_func_obj.is_none()) {
        auto dof_func_list = dof_func_obj.cast<py::list>();
        dof_func.resize(dof_func_list.size());
        for (size_t i = 0; i < dof_func_list.size(); i++)
          dof_func[i] = dof_func_list[i].cast<HYPRE_Int>();
      }

      return make_shared<HypreBoomerAMGPreconditioner>(
          sp_mat, sp_freedofs, print_level,
          coarsen_type, relax_type, agg_levels, strong_threshold,
          interp_type, max_levels, num_sweeps, coarse_relax_type,
          num_functions, std::move(dof_func));
    },
    py::arg("mat"),
    py::arg("freedofs") = py::none(),
    py::arg("print_level") = 0,
    py::arg("coarsen_type") = 10,
    py::arg("relax_type") = 6,
    py::arg("agg_levels") = 1,
    py::arg("strong_threshold") = 0.25,
    py::arg("interp_type") = 0,
    py::arg("max_levels") = 25,
    py::arg("num_sweeps") = 1,
    py::arg("coarse_relax_type") = 9,
    py::arg("num_functions") = 1,
    py::arg("dof_func") = py::none(),
    R"raw_string(
HYPRE BoomerAMG Preconditioner for H1 scalar elliptic systems.

Standalone BoomerAMG for use as h1_inv in Custom AMS preconditioner.
With relax_type=6 (symmetric GS), the V-cycle is CG-compatible.

Parameters:

mat : SparseMatrix (real)
  H1 system matrix.
freedofs : BitArray, optional
  Free DOFs mask.
print_level : int
  0=silent (default).
coarsen_type : int
  10=HMIS (default, good for 3D).
relax_type : int
  6=symmetric GS (default, CG-safe), 16=Chebyshev.
agg_levels : int
  Aggressive coarsening levels (default=1).
strong_threshold : float
  Strength threshold (default=0.25 for 3D).
interp_type : int
  Interpolation type (default=0, classical).
max_levels : int
  Maximum AMG levels (default=25).
num_sweeps : int
  Smoother sweeps per level (default=1).
coarse_relax_type : int
  Coarsest level smoother (default=9, Gaussian elimination).
)raw_string");

  // Complex HYPRE AMS with TaskManager Re/Im parallelism
  m.def("ComplexHypreAMSPreconditioner",
    [](shared_ptr<BaseMatrix> a_real_mat,
       shared_ptr<BaseMatrix> grad_mat,
       py::object freedofs_obj,
       py::list coord_x_list,
       py::list coord_y_list,
       py::list coord_z_list,
       int ndof_complex,
       int cycle_type,
       int print_level) -> shared_ptr<BaseMatrix>
    {
      auto sp_mat = dynamic_pointer_cast<SparseMatrix<double>>(a_real_mat);
      if (!sp_mat)
        throw py::type_error("ComplexHypreAMSPreconditioner: a_real_mat must be real SparseMatrix<double>");

      auto sp_grad = dynamic_pointer_cast<SparseMatrix<double>>(grad_mat);
      if (!sp_grad)
        throw py::type_error("ComplexHypreAMSPreconditioner: grad_mat must be real SparseMatrix<double>");

      auto sp_freedofs = ExtractFreeDofs(freedofs_obj);

      auto to_vec = [](py::list lst) {
        std::vector<double> v(lst.size());
        for (size_t i = 0; i < lst.size(); i++)
          v[i] = lst[i].cast<double>();
        return v;
      };

      return make_shared<ComplexHypreAMSPreconditioner>(
          sp_mat, sp_grad, sp_freedofs,
          to_vec(coord_x_list), to_vec(coord_y_list), to_vec(coord_z_list),
          ndof_complex, cycle_type, print_level);
    },
    py::arg("a_real_mat"),
    py::arg("grad_mat"),
    py::arg("freedofs") = py::none(),
    py::arg("coord_x") = py::list(),
    py::arg("coord_y") = py::list(),
    py::arg("coord_z") = py::list(),
    py::arg("ndof_complex") = 0,
    py::arg("cycle_type") = 1,
    py::arg("print_level") = 0,
    R"raw_string(
Complex HYPRE AMS preconditioner with TaskManager Re/Im parallelism.

For complex eddy current problems (A = K + jw*sigma*M). Creates TWO
independent HYPRE AMS solver instances and applies them to the real
and imaginary parts in parallel via NGSolve TaskManager.

Use with GMResSolver on the complex system matrix.

Parameters:

a_real_mat : SparseMatrix (real)
  Real auxiliary matrix (K + eps*M + |omega|*sigma*M_cond).
grad_mat : SparseMatrix (real)
  Discrete gradient matrix (HCurl -> H1).
freedofs : BitArray, optional
  Free DOFs mask for HCurl space.
coord_x, coord_y, coord_z : list of float
  Vertex coordinates (length = number of H1 DOFs).
ndof_complex : int
  Number of complex DOFs (= fes.ndof for complex HCurl space).
cycle_type : int
  AMS cycle type (1=01210, 7=0201020, default=1).
print_level : int
  HYPRE print level (0=silent, default=0).
)raw_string");
}
#else
inline void ExportHypreAMS(py::module& m) {
  m.def("has_hypre", []() { return false; },
    "Returns True if HYPRE support is available.");
}
#endif

// ============================================================================
// COCR solver (NGSolve BaseMatrix interface, accepts external preconditioner)
// ============================================================================

inline void ExportCOCRSolver(py::module& m) {
  // Register C++ types so .iterations property works
  py::class_<COCRSolverNGS<double>, shared_ptr<COCRSolverNGS<double>>, BaseMatrix>
      (m, "COCRSolverD")
      .def_property_readonly("iterations", &COCRSolverNGS<double>::GetIterations);

  py::class_<COCRSolverNGS<Complex>, shared_ptr<COCRSolverNGS<Complex>>, BaseMatrix>
      (m, "COCRSolverC")
      .def_property_readonly("iterations", &COCRSolverNGS<Complex>::GetIterations);

  m.def("COCRSolver",
    [](shared_ptr<BaseMatrix> mat,
       shared_ptr<BaseMatrix> pre,
       int maxiter,
       double tol,
       bool printrates) -> shared_ptr<BaseMatrix>
    {
      if (mat->IsComplex()) {
        return make_shared<COCRSolverNGS<Complex>>(
            mat, pre, maxiter, tol, printrates);
      } else {
        return make_shared<COCRSolverNGS<double>>(
            mat, pre, maxiter, tol, printrates);
      }
    },
    py::arg("mat"),
    py::arg("pre"),
    py::arg("maxiter") = 500,
    py::arg("tol") = 1e-8,
    py::arg("printrates") = false,
    R"raw_string(
COCR (Conjugate Orthogonal Conjugate Residual) solver for complex-symmetric systems.

For A^T = A (NOT Hermitian). Uses unconjugated inner products (x^T y).
Minimizes ||A r~||_2 for smoother convergence than COCG/CG.

When to use COCRSolver vs SparseSolvSolver(method="COCR"):
  - COCRSolver(mat, pre): accepts any external BaseMatrix preconditioner
    (e.g., IC, HYPRE AMS). Same interface as NGSolve CGSolver/GMResSolver.
  - SparseSolvSolver(method="COCR"): uses internal IC preconditioner with
    auto-shift, ABMC ordering, divergence detection. Unified solver interface.

Usage (same as NGSolve CGSolver):
  inv = COCRSolver(mat, pre, maxiter=500, tol=1e-8)
  gfu.vec.data = inv * rhs.vec

For COCG, use CGSolver(mat, pre, conjugate=False) instead.

Parameters:

mat : BaseMatrix
  System matrix (real or complex).
pre : BaseMatrix
  Preconditioner (must be symmetric for COCR).
maxiter : int
  Maximum iterations (default: 500).
tol : float
  Relative convergence tolerance (default: 1e-8).
printrates : bool
  Print convergence info (default: False).

Reference: Sogabe & Zhang (2007), J. Comput. Appl. Math., 199(2), 297-303.
)raw_string");
}

// ============================================================================
// Public API: Single entry point for NGSolve integration
// ============================================================================

/// Register all SparseSolv Python bindings (type registration + factory functions)
inline void ExportSparseSolvBindings(py::module& m) {
  ExportSparseSolvResult_impl(m);
  ExportSparseSolvTyped<double>(m, "D");
  ExportSparseSolvTyped<Complex>(m, "C");
  ExportSparseSolvFactories(m);
  ExportHypreAMS(m);
  ExportCOCRSolver(m);
}

} // namespace ngla

#endif // NGSOLVE_SPARSESOLV_PYTHON_EXPORT_HPP
