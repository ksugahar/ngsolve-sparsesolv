/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/// @file hypre_boomeramg_preconditioner.hpp
/// @brief HYPRE BoomerAMG preconditioner wrapper for NGSolve H1 systems.
///
/// Standalone BoomerAMG for H1 scalar elliptic systems.
/// With symmetric smoothing (relax_type=6), the V-cycle is symmetric
/// and can be used inside CG-compatible preconditioners.

#ifndef SPARSESOLV_HYPRE_BOOMERAMG_PRECONDITIONER_HPP
#define SPARSESOLV_HYPRE_BOOMERAMG_PRECONDITIONER_HPP

#ifdef SPARSESOLV_USE_HYPRE

#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"

#include <comp.hpp>  // NGSolve
#include <vector>
#include <stdexcept>
#include <iostream>

namespace ngla {

/// HYPRE BoomerAMG preconditioner for real H1 scalar elliptic systems.
///
/// Primary use case: H1 scalar elliptic systems.
/// Useful for custom preconditioning outside of AMS context.
///
/// With relax_type=6 (symmetric Gauss-Seidel), the V-cycle is symmetric,
/// making the overall AMS preconditioner CG-compatible.
class HypreBoomerAMGPreconditioner : public BaseMatrix {
public:
    /// @param mat              H1 system matrix (SparseMatrix<double>)
    /// @param freedofs         Free DOFs mask (Dirichlet boundary)
    /// @param print_level      0=silent, 1=setup info, 2+=verbose
    /// @param coarsen_type     10=HMIS (recommended for 3D)
    /// @param relax_type       6=symmetric GS (CG-safe!), 16=Chebyshev
    /// @param agg_levels       Number of aggressive coarsening levels
    /// @param strong_threshold Strength threshold (0.25 for 3D)
    /// @param interp_type      Interpolation type (0=classical)
    /// @param max_levels       Maximum AMG hierarchy depth
    /// @param num_sweeps       Number of smoother sweeps per level
    /// @param coarse_relax_type  Smoother for coarsest level (9=Gaussian elim)
    /// @param num_functions    Number of functions for nodal/block coarsening
    ///                         (1=scalar, 3=vector for combined Pi systems)
    HypreBoomerAMGPreconditioner(
        shared_ptr<SparseMatrix<double>> mat,
        shared_ptr<BitArray> freedofs,
        int print_level = 0,
        int coarsen_type = 10,
        int relax_type = 6,
        int agg_levels = 1,
        double strong_threshold = 0.25,
        int interp_type = 0,
        int max_levels = 25,
        int num_sweeps = 1,
        int coarse_relax_type = 9,
        int num_functions = 1,
        std::vector<HYPRE_Int> dof_func = {})
        : mat_(mat), freedofs_(freedofs), ndof_(mat->Height()),
          dof_func_(std::move(dof_func))
    {
        Setup(coarsen_type, relax_type, agg_levels, strong_threshold,
              interp_type, max_levels, num_sweeps, coarse_relax_type,
              print_level, num_functions);
    }

    ~HypreBoomerAMGPreconditioner() {
        Cleanup();
    }

    int VHeight() const override { return ndof_; }
    int VWidth() const override { return ndof_; }
    bool IsComplex() const override { return false; }

    AutoVector CreateRowVector() const override {
        return mat_->CreateRowVector();
    }
    AutoVector CreateColVector() const override {
        return mat_->CreateColVector();
    }

    void Mult(const BaseVector& f, BaseVector& u) const override {
        HYPRE_IJVectorInitialize(hyp_b_);
        HYPRE_IJVectorInitialize(hyp_x_);

        auto fv_f = f.FVDouble();
        auto fv_u = u.FVDouble();

        buf_b_.resize(ndof_);
        buf_x_.resize(ndof_);
        for (int i = 0; i < ndof_; i++) {
            buf_b_[i] = (freedofs_ && !freedofs_->Test(i)) ? 0.0 : fv_f[i];
            buf_x_[i] = 0.0;
        }

        HYPRE_IJVectorSetValues(hyp_b_, ndof_, indices_.data(), buf_b_.data());
        HYPRE_IJVectorSetValues(hyp_x_, ndof_, indices_.data(), buf_x_.data());
        HYPRE_IJVectorAssemble(hyp_b_);
        HYPRE_IJVectorAssemble(hyp_x_);

        HYPRE_ParVector par_b, par_x;
        HYPRE_IJVectorGetObject(hyp_b_, (void**)&par_b);
        HYPRE_IJVectorGetObject(hyp_x_, (void**)&par_x);

        HYPRE_BoomerAMGSolve(solver_, parcsr_A_, par_b, par_x);

        HYPRE_IJVectorGetValues(hyp_x_, ndof_, indices_.data(), buf_x_.data());
        fv_u = 0.0;
        for (int i = 0; i < ndof_; i++)
            fv_u[i] = buf_x_[i];
    }

    /// Symmetric preconditioner: MultTrans = Mult
    void MultTrans(const BaseVector& f, BaseVector& u) const override {
        Mult(f, u);
    }

private:
    shared_ptr<SparseMatrix<double>> mat_;
    shared_ptr<BitArray> freedofs_;
    int ndof_;

    HYPRE_Solver solver_ = nullptr;
    HYPRE_IJMatrix hyp_A_ = nullptr;
    HYPRE_ParCSRMatrix parcsr_A_ = nullptr;
    HYPRE_IJVector hyp_b_ = nullptr;
    HYPRE_IJVector hyp_x_ = nullptr;

    std::vector<HYPRE_Int> indices_;
    std::vector<HYPRE_Int> dof_func_;
    mutable std::vector<double> buf_b_, buf_x_;

    /// Convert NGSolve SparseMatrix to HYPRE ParCSR, handling symmetric storage.
    ///
    /// H1 BilinearForm may use symmetric storage (upper triangle only).
    /// HYPRE expects full CSR, so we symmetrize: emit (i,j,v) and (j,i,v).
    void ConvertSparseMatrix() {
        HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, ndof_ - 1,
                             0, ndof_ - 1, &hyp_A_);
        HYPRE_IJMatrixSetObjectType(hyp_A_, HYPRE_PARCSR);
        HYPRE_IJMatrixInitialize(hyp_A_);

        // Check if matrix is symmetric storage by comparing nnz pattern:
        // In symmetric storage, row k only has entries with column index >= k
        bool is_symmetric_storage = false;
        if (ndof_ > 1) {
            auto first_row = mat_->GetRowIndices(0);
            // If first row has entries, check if any col < row_index
            // For row 0, all cols >= 0, so check a middle row
            int mid = ndof_ / 2;
            auto mid_row = mat_->GetRowIndices(mid);
            bool has_lower = false;
            for (int j = 0; j < mid_row.Size(); j++) {
                if (mid_row[j] < mid) { has_lower = true; break; }
            }
            is_symmetric_storage = !has_lower && mid_row.Size() > 0;
        }

        // Temporary storage for off-diagonal transpose entries
        // For symmetric storage, we need to add (j, k, val) for each (k, j, val) with k != j
        std::vector<std::vector<std::pair<HYPRE_Int, double>>> lower_entries;
        if (is_symmetric_storage)
            lower_entries.resize(ndof_);

        for (int k = 0; k < ndof_; k++) {
            auto row_cols = mat_->GetRowIndices(k);
            auto row_vals = mat_->GetRowValues(k);
            HYPRE_Int row = k;

            if (freedofs_ && !freedofs_->Test(k)) {
                // Dirichlet row: identity
                HYPRE_Int col = k;
                double val = 1.0;
                int one = 1;
                HYPRE_IJMatrixAddToValues(hyp_A_, 1, &one, &row, &col, &val);
            } else {
                // Active row: copy entries, skipping constrained columns
                for (int j = 0; j < row_cols.Size(); j++) {
                    HYPRE_Int col = row_cols[j];
                    double val = row_vals[j];

                    if (freedofs_ && !freedofs_->Test(col))
                        continue;

                    int one = 1;
                    HYPRE_IJMatrixAddToValues(hyp_A_, 1, &one, &row, &col, &val);

                    // For symmetric storage, also add transpose entry
                    if (is_symmetric_storage && col != k) {
                        lower_entries[col].push_back({k, val});
                    }
                }
            }
        }

        // Add lower-triangle entries (transpose of upper)
        if (is_symmetric_storage) {
            for (int k = 0; k < ndof_; k++) {
                if (freedofs_ && !freedofs_->Test(k))
                    continue;
                for (auto& [col, val] : lower_entries[k]) {
                    if (freedofs_ && !freedofs_->Test(col))
                        continue;
                    HYPRE_Int row = k;
                    int one = 1;
                    HYPRE_IJMatrixAddToValues(hyp_A_, 1, &one, &row, &col, &val);
                }
            }
        }

        HYPRE_IJMatrixAssemble(hyp_A_);
        HYPRE_IJMatrixGetObject(hyp_A_, (void**)&parcsr_A_);

        if (is_symmetric_storage) {
            std::cout << "  BoomerAMG: symmetric storage detected, symmetrized "
                      << ndof_ << " x " << ndof_ << " matrix" << std::endl;
        }
    }

    void CreateIJVector(HYPRE_IJVector& vec) {
        HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, ndof_ - 1, &vec);
        HYPRE_IJVectorSetObjectType(vec, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(vec);
    }

    void Setup(int coarsen_type, int relax_type, int agg_levels,
               double strong_threshold, int interp_type, int max_levels,
               int num_sweeps, int coarse_relax_type, int print_level,
               int num_functions = 1) {
        // Index array
        indices_.resize(ndof_);
        for (int i = 0; i < ndof_; i++)
            indices_[i] = i;

        // Convert matrix
        ConvertSparseMatrix();

        // Create working vectors
        CreateIJVector(hyp_b_);
        {
            std::vector<double> zeros(ndof_, 0.0);
            HYPRE_IJVectorSetValues(hyp_b_, ndof_, indices_.data(), zeros.data());
        }
        HYPRE_IJVectorAssemble(hyp_b_);

        CreateIJVector(hyp_x_);
        {
            std::vector<double> zeros(ndof_, 0.0);
            HYPRE_IJVectorSetValues(hyp_x_, ndof_, indices_.data(), zeros.data());
        }
        HYPRE_IJVectorAssemble(hyp_x_);

        // Create BoomerAMG solver
        HYPRE_BoomerAMGCreate(&solver_);

        // Coarsening
        HYPRE_BoomerAMGSetCoarsenType(solver_, coarsen_type);
        HYPRE_BoomerAMGSetAggNumLevels(solver_, agg_levels);
        HYPRE_BoomerAMGSetStrongThreshold(solver_, strong_threshold);

        // Interpolation
        HYPRE_BoomerAMGSetInterpType(solver_, interp_type);

        // Hierarchy
        HYPRE_BoomerAMGSetMaxLevels(solver_, max_levels);
        HYPRE_BoomerAMGSetMinCoarseSize(solver_, 2);

        // Smoother: CRITICAL for CG compatibility
        // relax_type=6: symmetric GS (forward + backward) -> symmetric V-cycle
        // relax_type=16: Chebyshev -> also symmetric
        HYPRE_BoomerAMGSetRelaxType(solver_, relax_type);
        HYPRE_BoomerAMGSetNumSweeps(solver_, num_sweeps);

        // Coarsest level: Gaussian elimination (exact)
        HYPRE_BoomerAMGSetCycleRelaxType(solver_, coarse_relax_type, 3);

        // Block/nodal coarsening for vector systems (e.g., combined Pi)
        if (num_functions > 1) {
            HYPRE_BoomerAMGSetNumFunctions(solver_, num_functions);
            if (!dof_func_.empty()) {
                // Custom DOF-to-function mapping (for blocked DOF ordering)
                HYPRE_BoomerAMGSetDofFunc(solver_, dof_func_.data());
            }
            // Nodal coarsening: coarsen unknowns together
            HYPRE_BoomerAMGSetNodal(solver_, 1);
        }

        // Preconditioner mode: single V-cycle, no tolerance
        HYPRE_BoomerAMGSetTol(solver_, 0.0);
        HYPRE_BoomerAMGSetMaxIter(solver_, 1);

        HYPRE_BoomerAMGSetPrintLevel(solver_, print_level);

        // Setup
        HYPRE_ParVector par_b, par_x;
        HYPRE_IJVectorGetObject(hyp_b_, (void**)&par_b);
        HYPRE_IJVectorGetObject(hyp_x_, (void**)&par_x);
        HYPRE_BoomerAMGSetup(solver_, parcsr_A_, par_b, par_x);

        if (print_level > 0)
            std::cout << "  BoomerAMG: setup complete (ndof=" << ndof_
                      << " coarsen=" << coarsen_type
                      << " relax=" << relax_type << ")" << std::endl;
    }

    void Cleanup() {
        if (solver_) { HYPRE_BoomerAMGDestroy(solver_); solver_ = nullptr; }
        if (hyp_A_)  { HYPRE_IJMatrixDestroy(hyp_A_); hyp_A_ = nullptr; }
        if (hyp_b_)  { HYPRE_IJVectorDestroy(hyp_b_); hyp_b_ = nullptr; }
        if (hyp_x_)  { HYPRE_IJVectorDestroy(hyp_x_); hyp_x_ = nullptr; }
    }
};

}  // namespace ngla

#endif  // SPARSESOLV_USE_HYPRE
#endif  // SPARSESOLV_HYPRE_BOOMERAMG_PRECONDITIONER_HPP
