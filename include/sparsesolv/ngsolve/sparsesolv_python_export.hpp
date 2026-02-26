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
#include <comp.hpp>
#include <type_traits>

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

  // BDDC Preconditioner type registration
  {
    std::string cls_name = "BDDCPreconditioner" + suffix;
    py::class_<SparseSolvBDDCPreconditioner<SCAL>,
               shared_ptr<SparseSolvBDDCPreconditioner<SCAL>>,
               BaseMatrix>(m, cls_name.c_str())
      .def_property_readonly("num_wirebasket_dofs",
          &SparseSolvBDDCPreconditioner<SCAL>::NumWirebasketDofs)
      .def_property_readonly("num_interface_dofs",
          &SparseSolvBDDCPreconditioner<SCAL>::NumInterfaceDofs);
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
// Internal: BDDC factory from BilinearForm (C++ element matrix extraction)
// ============================================================================

template<typename SCAL>
shared_ptr<BaseMatrix> CreateBDDCFromBilinearForm(
    shared_ptr<ngcomp::BilinearForm> bfa,
    shared_ptr<ngcomp::FESpace> fes,
    const std::string& coarse_inverse)
{
    auto mat = dynamic_pointer_cast<SparseMatrix<SCAL>>(bfa->GetMatrixPtr());
    if (!mat) throw py::type_error("BDDCPreconditioner: matrix type mismatch");
    auto freedofs = fes->GetFreeDofs(false);  // include LOCAL_DOFs for order >= 3
    auto mesh = fes->GetMeshAccess();
    size_t ndof = fes->GetNDof();
    size_t ne = mesh->GetNE(ngfem::VOL);

    // Extract DOF classification (parallel)
    std::vector<sparsesolv::DOFType> dof_types(ndof);
    ParallelFor(ndof, [&](size_t d) {
        auto ct = fes->GetDofCouplingType(d);
        dof_types[d] = (ct == ngcomp::WIREBASKET_DOF)
            ? sparsesolv::DOFType::Wirebasket
            : sparsesolv::DOFType::Interface;
    });

    // Extract element DOFs and element matrices (parallel via IterateElements)
    std::vector<std::vector<sparsesolv::index_t>> element_dofs(ne);
    std::vector<sparsesolv::DenseMatrix<SCAL>> element_matrices(ne);
    const auto& integrators = bfa->Integrators();

    LocalHeap lh(10000000, "bddc_setup", true);  // mult_by_threads=true

    // Filter to VOL integrators only
    Array<shared_ptr<ngfem::BilinearFormIntegrator>> vol_integrators;
    for (auto& integ : integrators) {
        if (integ->VB() == ngfem::VOL)
            vol_integrators.Append(integ);
    }

    ngcomp::IterateElements(*fes, ngfem::VOL, lh,
        [&](ngcomp::FESpace::Element el, LocalHeap& lh_thread) {
            size_t elnr = el.Nr();

            // Get DOFs for this element
            auto dnums = el.GetDofs();

            // Filter to valid DOFs (>= 0)
            std::vector<int> valid_local;
            std::vector<sparsesolv::index_t> valid_global;
            for (int i = 0; i < dnums.Size(); ++i) {
                if (ngcomp::IsRegularDof(dnums[i])) {
                    valid_local.push_back(i);
                    valid_global.push_back(static_cast<sparsesolv::index_t>(dnums[i]));
                }
            }
            element_dofs[elnr] = std::move(valid_global);

            // Get FE and transformation
            auto& fe = el.GetFE();
            auto& trafo = el.GetTrafo();

            int ndof_el = dnums.Size();
            FlatMatrix<SCAL> elmat(ndof_el, ndof_el, lh_thread);
            elmat = SCAL(0);

            // Sum contributions from VOL integrators only
            for (auto& integrator : vol_integrators) {
                FlatMatrix<SCAL> contrib(ndof_el, ndof_el, lh_thread);
                contrib = SCAL(0);
                integrator->CalcElementMatrix(fe, trafo, contrib, lh_thread);
                elmat += contrib;
            }

            // Extract valid-DOF submatrix
            int nvalid = static_cast<int>(valid_local.size());
            sparsesolv::DenseMatrix<SCAL> dm(nvalid, nvalid);
            for (int i = 0; i < nvalid; ++i)
                for (int j = 0; j < nvalid; ++j)
                    dm(i, j) = elmat(valid_local[i], valid_local[j]);
            element_matrices[elnr] = std::move(dm);
        });

    auto p = make_shared<SparseSolvBDDCPreconditioner<SCAL>>(
        mat, freedofs, std::move(element_dofs),
        std::move(dof_types), std::move(element_matrices),
        coarse_inverse);
    p->Update();
    return p;
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

  // ---- BDDCPreconditioner factory ----
  m.def("BDDCPreconditioner", [](py::object first_arg,
                                  py::object second_arg,
                                  std::string coarse_inverse) {
    // BilinearForm API: BDDCPreconditioner(a, fes, coarse_inverse=...)
    try {
      auto bfa = py::cast<shared_ptr<ngcomp::BilinearForm>>(first_arg);
      auto fes = py::cast<shared_ptr<ngcomp::FESpace>>(second_arg);
      if (bfa->GetMatrixPtr()->IsComplex())
        return CreateBDDCFromBilinearForm<Complex>(bfa, fes, coarse_inverse);
      else
        return CreateBDDCFromBilinearForm<double>(bfa, fes, coarse_inverse);
    } catch (py::cast_error&) {
      throw py::type_error(
          "BDDCPreconditioner(a, fes): expected BilinearForm and FESpace");
    }
  },
  py::arg("a"),
  py::arg("fes"),
  py::arg("coarse_inverse") = "sparsecholesky",
  R"raw_string(
BDDC (Balancing Domain Decomposition by Constraints) Preconditioner.

Extracts element matrices from BilinearForm and builds element-by-element BDDC.

Parameters:

a : BilinearForm
  Assembled BilinearForm.
fes : FESpace
  Finite element space.
coarse_inverse : str
  Coarse solver: "sparsecholesky" (default), "pardiso".
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
    shared_ptr<BaseMatrix> result;
    if (mat->IsComplex()) {
      auto sp = dynamic_pointer_cast<SparseMatrix<Complex>>(mat);
      if (!sp) throw py::type_error("SparseSolvSolver: expected SparseMatrix");
      auto solver = make_shared<SparseSolvSolver<Complex>>(
          sp, method, sp_freedofs, tol, maxiter, shift,
          save_best_result, save_residual_history, printrates);
      solver->SetConjugate(conjugate);
      solver->SetUseABMC(use_abmc);
      solver->SetABMCBlockSize(abmc_block_size);
      solver->SetABMCNumColors(abmc_num_colors);
      solver->SetABMCReorderSpMV(abmc_reorder_spmv);
      solver->SetABMCUseRCM(abmc_use_rcm);
      result = solver;
    } else {
      auto sp = dynamic_pointer_cast<SparseMatrix<double>>(mat);
      if (!sp) throw py::type_error("SparseSolvSolver: expected SparseMatrix");
      auto solver = make_shared<SparseSolvSolver<double>>(
          sp, method, sp_freedofs, tol, maxiter, shift,
          save_best_result, save_residual_history, printrates);
      solver->SetUseABMC(use_abmc);
      solver->SetABMCBlockSize(abmc_block_size);
      solver->SetABMCNumColors(abmc_num_colors);
      solver->SetABMCReorderSpMV(abmc_reorder_spmv);
      solver->SetABMCUseRCM(abmc_use_rcm);
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
Iterative solver (ICCG / SGSMRTR / CG). Auto-detects real/complex.

Can be used as BaseMatrix (inverse operator) or via Solve() for detailed results.

Parameters:

mat : SparseMatrix
  SPD matrix (real or complex).
method : str
  "ICCG", "SGSMRTR", or "CG".
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
// Public API: Single entry point for NGSolve integration
// ============================================================================

/// Register all SparseSolv Python bindings (type registration + factory functions)
inline void ExportSparseSolvBindings(py::module& m) {
  ExportSparseSolvResult_impl(m);
  ExportSparseSolvTyped<double>(m, "D");
  ExportSparseSolvTyped<Complex>(m, "C");
  ExportSparseSolvFactories(m);
}

} // namespace ngla

#endif // NGSOLVE_SPARSESOLV_PYTHON_EXPORT_HPP
