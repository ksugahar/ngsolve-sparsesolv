/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/// @file sparsesolv_solvers.hpp
/// @brief SparseSolv iterative solvers (ICCG, SGSMRTR, CG, COCR) as NGSolve BaseMatrix wrappers
///
/// Separated from sparsesolv_precond.hpp for clarity of responsibility:
/// - sparsesolv_precond.hpp: preconditioners (IC, SGS, BDDC)
/// - sparsesolv_solvers.hpp: iterative solvers (SparseSolvSolver, COCRSolverNGS)

#ifndef NGSOLVE_SPARSESOLV_SOLVERS_HPP
#define NGSOLVE_SPARSESOLV_SOLVERS_HPP

#include "sparsesolv_precond.hpp"  // for GetVectorData, BuildSparseMatrixView

namespace ngla {

// ============================================================================
// Solver Result (alias for sparsesolv::SolverResult)
// ============================================================================
using SparseSolvResult = sparsesolv::SolverResult;

// ============================================================================
// SparseSolv Iterative Solver
// ============================================================================

/// Unified iterative solver: ICCG, SGSMRTR, CG, COCR.
/// Use as BaseMatrix (inv * rhs) or call .Solve() for detailed results.
///
/// Uses internal preconditioner (IC for ICCG, SGS for SGSMRTR, none for CG/COCR).
/// For external preconditioner with COCR, use COCRSolverNGS instead.
template<typename SCAL = double>
class SparseSolvSolver : public BaseMatrix {
public:
    SparseSolvSolver(shared_ptr<SparseMatrix<SCAL>> mat,
                      const string& method = "ICCG",
                      shared_ptr<BitArray> freedofs = nullptr,
                      double tol = 1e-10,
                      int maxiter = 1000,
                      double shift = 1.05,
                      bool save_best_result = true,
                      bool save_residual_history = false,
                      bool printrates = false)
        : mat_(mat)
        , method_(method)
        , freedofs_(freedofs ? make_shared<BitArray>(*freedofs) : nullptr)
        , height_(static_cast<sparsesolv::index_t>(mat->Height()))
        , width_(static_cast<sparsesolv::index_t>(mat->Width()))
        , printrates_(printrates)
    {
        config_.tolerance = tol;
        config_.max_iterations = maxiter;
        config_.shift_parameter = shift;
        config_.save_best_result = save_best_result;
        config_.save_residual_history = save_residual_history;
    }

    /// Solve Ax = b, x initialized to zero
    void Mult(const BaseVector& x, BaseVector& y) const override {
        y = 0.0;
        Solve(x, y);
    }

    void MultAdd(double s, const BaseVector& x, BaseVector& y) const override {
        auto temp = y.CreateVector();
        temp = 0.0;
        Solve(x, *temp);
        y += s * *temp;
    }

    /// Solve Ax = b with initial guess, returns SparseSolvResult
    SparseSolvResult Solve(const BaseVector& rhs, BaseVector& sol) const {
        // Prepare matrix view (with FreeDofs handling)
        auto view = prepare_matrix();

        // Get vector data
        const SCAL* b = GetVectorData<SCAL>(rhs);
        SCAL* x = GetVectorData<SCAL>(sol);

        // For FreeDofs: work on copies with constrained DOFs zeroed
        std::vector<SCAL> b_mod;
        std::vector<SCAL> x_mod;
        const SCAL* b_ptr = b;
        SCAL* x_ptr = x;

        if (freedofs_) {
            b_mod.resize(height_);
            x_mod.resize(height_);
            sparsesolv::parallel_for(height_, [&](sparsesolv::index_t i) {
                if (freedofs_->Test(i)) {
                    b_mod[i] = b[i];
                    x_mod[i] = x[i];
                } else {
                    b_mod[i] = SCAL(0);
                    x_mod[i] = SCAL(0);
                }
            });
            b_ptr = b_mod.data();
            x_ptr = x_mod.data();
        }

        // Configure solver
        sparsesolv::SolverConfig config = config_;
        if (printrates_) {
            config.save_residual_history = true;
        }

        // Dispatch to appropriate solver
        sparsesolv::SolverResult result;

        if (method_ == "ICCG" || method_ == "iccg") {
            result = sparsesolv::solve_iccg(view, b_ptr, x_ptr, height_, config);
        } else if (method_ == "SGSMRTR" || method_ == "sgsmrtr") {
            result = sparsesolv::solve_sgsmrtr(view, b_ptr, x_ptr, height_, config);
        } else if (method_ == "CG" || method_ == "cg") {
            sparsesolv::CGSolver<SCAL> solver;
            solver.set_config(config);
            result = solver.solve(view, b_ptr, x_ptr, height_, nullptr);
        } else if (method_ == "COCR" || method_ == "cocr") {
            sparsesolv::COCRSolver<SCAL> solver;
            solver.set_config(config);
            result = solver.solve(view, b_ptr, x_ptr, height_, nullptr);
        } else {
            throw std::runtime_error(
                "SparseSolvSolver: Unknown method '" + method_ +
                "'. Available: ICCG, SGSMRTR, CG, COCR");
        }

        // Copy solution back (only free DOFs)
        if (freedofs_) {
            sparsesolv::parallel_for(height_, [&](sparsesolv::index_t i) {
                if (freedofs_->Test(i)) {
                    x[i] = x_ptr[i];
                }
            });
        }

        // Print convergence info if requested
        if (printrates_ && !result.residual_history.empty()) {
            for (size_t i = 0; i < result.residual_history.size(); ++i) {
                std::cout << method_ << " iteration " << i
                          << ", residual = " << result.residual_history[i]
                          << std::endl;
            }
            if (result.converged) {
                std::cout << method_ << " converged in "
                          << result.iterations << " iterations, residual = "
                          << result.final_residual << std::endl;
            } else {
                std::cout << method_ << " NOT converged after "
                          << result.iterations << " iterations, residual = "
                          << result.final_residual << std::endl;
            }
        }

        // Build and store result
        last_result_ = std::move(result);
        return last_result_;
    }

    int VHeight() const override { return static_cast<int>(height_); }
    int VWidth() const override { return static_cast<int>(width_); }
    AutoVector CreateRowVector() const override { return mat_->CreateRowVector(); }
    AutoVector CreateColVector() const override { return mat_->CreateColVector(); }

    // Property accessors
    const string& GetMethod() const { return method_; }
    void SetMethod(const string& method) { method_ = method; }
    double GetTolerance() const { return config_.tolerance; }
    void SetTolerance(double tol) { config_.tolerance = tol; }
    int GetMaxIterations() const { return config_.max_iterations; }
    void SetMaxIterations(int maxiter) { config_.max_iterations = maxiter; }
    double GetShift() const { return config_.shift_parameter; }
    void SetShift(double shift) { config_.shift_parameter = shift; }
    bool GetSaveBestResult() const { return config_.save_best_result; }
    void SetSaveBestResult(bool save) { config_.save_best_result = save; }
    bool GetSaveResidualHistory() const { return config_.save_residual_history; }
    void SetSaveResidualHistory(bool save) { config_.save_residual_history = save; }
    bool GetPrintRates() const { return printrates_; }
    void SetPrintRates(bool print) { printrates_ = print; }
    bool GetAutoShift() const { return config_.auto_shift; }
    void SetAutoShift(bool enable) { config_.auto_shift = enable; }
    bool GetDiagonalScaling() const { return config_.diagonal_scaling; }
    void SetDiagonalScaling(bool enable) { config_.diagonal_scaling = enable; }

    // Complex inner product
    bool GetConjugate() const { return config_.conjugate; }
    void SetConjugate(bool enable) { config_.conjugate = enable; }

    // ABMC ordering
    bool GetUseABMC() const { return config_.use_abmc; }
    void SetUseABMC(bool enable) { config_.use_abmc = enable; }
    int GetABMCBlockSize() const { return config_.abmc_block_size; }
    void SetABMCBlockSize(int bs) { config_.abmc_block_size = bs; }
    int GetABMCNumColors() const { return config_.abmc_num_colors; }
    void SetABMCNumColors(int nc) { config_.abmc_num_colors = nc; }
    bool GetABMCReorderSpMV() const { return config_.abmc_reorder_spmv; }
    void SetABMCReorderSpMV(bool enable) { config_.abmc_reorder_spmv = enable; }
    bool GetABMCUseRCM() const { return config_.abmc_use_rcm; }
    void SetABMCUseRCM(bool enable) { config_.abmc_use_rcm = enable; }

    // Divergence detection
    bool GetDivergenceCheck() const {
        return config_.divergence_check == sparsesolv::DivergenceCheck::StagnationCount;
    }
    void SetDivergenceCheck(bool enable) {
        config_.divergence_check = enable
            ? sparsesolv::DivergenceCheck::StagnationCount
            : sparsesolv::DivergenceCheck::None;
    }
    double GetDivergenceThreshold() const { return config_.divergence_threshold; }
    void SetDivergenceThreshold(double threshold) { config_.divergence_threshold = threshold; }
    int GetDivergenceCount() const { return config_.divergence_count; }
    void SetDivergenceCount(int count) { config_.divergence_count = count; }

    const SparseSolvResult& GetLastResult() const { return last_result_; }

private:
    sparsesolv::SparseMatrixView<SCAL> prepare_matrix() const {
        return BuildSparseMatrixView<SCAL>(
            *mat_, freedofs_.get(), height_, width_,
            row_ptr_, col_idx_, modified_values_);
    }

    shared_ptr<SparseMatrix<SCAL>> mat_;
    string method_;
    shared_ptr<BitArray> freedofs_;
    sparsesolv::SolverConfig config_;
    sparsesolv::index_t height_;
    sparsesolv::index_t width_;
    bool printrates_;

    // Mutable for use in const Mult/Solve methods
    mutable std::vector<sparsesolv::index_t> row_ptr_;
    mutable std::vector<sparsesolv::index_t> col_idx_;
    mutable std::vector<SCAL> modified_values_;
    mutable SparseSolvResult last_result_;
};

// ============================================================================
// COCR Solver with External Preconditioner
// ============================================================================

/// COCR solver as NGSolve BaseMatrix, accepting an external preconditioner.
///
/// Use this when you have a custom preconditioner (any BaseMatrix).
/// For internal IC/SGS preconditioner, use SparseSolvSolver(method="COCR") instead.
///
/// Note on convergence criterion: uses preconditioned residual sqrt(|rt^T * r|)
/// where rt = M^{-1}*r. This differs from the raw COCRSolver<SCAL> which uses
/// unpreconditioned ||r||. The preconditioned residual is standard for
/// preconditioned Krylov solvers (consistent with NGSolve's CGSolver).
///
/// Usage: inv = COCRSolverNGS(mat, pre, maxiter, tol); gfu.vec = inv * rhs;
template<typename SCAL>
class COCRSolverNGS : public BaseMatrix {
public:
    COCRSolverNGS(shared_ptr<BaseMatrix> mat,
                  shared_ptr<BaseMatrix> pre,
                  int maxiter = 500,
                  double tol = 1e-8,
                  bool printrates = false)
        : mat_(std::move(mat)), pre_(std::move(pre)),
          maxiter_(maxiter), tol_(tol), printrates_(printrates),
          iterations_(0) {}

    /// Solve: y = A^{-1} * x  (y initialized to zero)
    void Mult(const BaseVector& rhs, BaseVector& sol) const override {
        sol = 0.0;

        auto r = rhs.CreateVector();
        auto rt = rhs.CreateVector();
        auto p = rhs.CreateVector();
        auto q = rhs.CreateVector();
        auto qt = rhs.CreateVector();
        auto t = rhs.CreateVector();

        // r = rhs - A*sol
        *r = rhs;
        *r -= *mat_ * sol;

        // rt = M^{-1} * r
        *rt = *pre_ * *r;

        // p = rt
        *p = *rt;

        // q = A * p
        *q = *mat_ * *p;

        // rho = rt^T * q (unconjugated)
        SCAL rho = InnerProduct<SCAL>(*rt, *q, false);

        // Convergence measure: sqrt(|rt^T * r|) (preconditioned residual)
        SCAL wdn = InnerProduct<SCAL>(*rt, *r, false);
        double res0 = std::sqrt(std::abs(wdn));

        if (res0 < 1e-30) {
            iterations_ = 0;
            return;
        }

        double final_res = res0 * tol_;
        if (printrates_)
            std::cout << "COCR iter 0: res = " << res0 << std::endl;

        int j = 0;
        for (j = 0; j < maxiter_; ++j) {
            // qt = M^{-1} * q
            *qt = *pre_ * *q;

            // mu = qt^T * q (unconjugated)
            SCAL mu = InnerProduct<SCAL>(*qt, *q, false);

            if (std::abs(mu) < 1e-30) break;

            SCAL alpha = rho / mu;

            // x += alpha * p
            sol += alpha * *p;

            // r -= alpha * q
            *r -= alpha * *q;

            // rt -= alpha * qt (vector update, NOT precond solve)
            *rt -= alpha * *qt;

            // Convergence: sqrt(|rt^T * r|) (preconditioned residual)
            wdn = InnerProduct<SCAL>(*rt, *r, false);
            double res = std::sqrt(std::abs(wdn));

            if (printrates_)
                std::cout << "COCR iter " << (j+1) << ": res = " << res << std::endl;

            if (res <= final_res) {
                iterations_ = j + 1;
                return;
            }

            // t = A * rt
            *t = *mat_ * *rt;

            // rho_new = rt^T * t (unconjugated)
            SCAL rho_new = InnerProduct<SCAL>(*rt, *t, false);

            if (std::abs(rho) < 1e-30) break;

            SCAL beta = rho_new / rho;
            rho = rho_new;

            // p = rt + beta * p
            *p *= beta;
            *p += *rt;

            // q = t + beta * q
            *q *= beta;
            *q += *t;
        }
        iterations_ = j;
    }

    int VHeight() const override { return mat_->VHeight(); }
    int VWidth() const override { return mat_->VWidth(); }
    AutoVector CreateRowVector() const override { return mat_->CreateRowVector(); }
    AutoVector CreateColVector() const override { return mat_->CreateColVector(); }
    bool IsComplex() const override { return mat_->IsComplex(); }

    int GetIterations() const { return iterations_; }

private:
    shared_ptr<BaseMatrix> mat_;
    shared_ptr<BaseMatrix> pre_;
    int maxiter_;
    double tol_;
    bool printrates_;
    mutable int iterations_;
};

} // namespace ngla

#endif // NGSOLVE_SPARSESOLV_SOLVERS_HPP
