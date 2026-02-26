/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/// @file pardiso_solver.hpp
/// @brief MKL PARDISO direct solver wrapper (NGSolve-independent)

#ifndef SPARSESOLV_DIRECT_PARDISO_SOLVER_HPP
#define SPARSESOLV_DIRECT_PARDISO_SOLVER_HPP

#include "../core/types.hpp"
#include <mkl_pardiso.h>
#include <complex>
#include <vector>
#include <stdexcept>
#include <cstring>

namespace sparsesolv {

namespace detail {

/// PARDISO matrix type for real SPD
template<typename Scalar>
struct pardiso_mtype {
    static constexpr MKL_INT value = 2;  // real symmetric positive definite
};

/// PARDISO matrix type for complex symmetric
template<typename T>
struct pardiso_mtype<std::complex<T>> {
    static constexpr MKL_INT value = 6;  // complex symmetric
};

} // namespace detail

/// MKL PARDISO direct solver wrapper
///
/// Wraps Intel MKL PARDISO for solving symmetric systems A*x = b.
/// Supports real SPD (mtype=2) and complex symmetric (mtype=6).
/// Input: full symmetric CSR (0-based). Upper triangle extracted internally.
template<typename Scalar>
class PardisoSolver {
public:
    PardisoSolver() {
        std::memset(pt_, 0, sizeof(pt_));
        mtype_ = detail::pardiso_mtype<Scalar>::value;
        pardisoinit(pt_, &mtype_, iparm_);

        // 0-based indexing
        iparm_[34] = 1;

        // Disable iterative-direct algorithm
        iparm_[3] = 0;
    }

    ~PardisoSolver() {
        if (n_ > 0) {
            MKL_INT phase = -1;  // release
            MKL_INT maxfct = 1, mnum = 1, nrhs = 1, msglvl = 0, error = 0;
            pardiso(pt_, &maxfct, &mnum, &mtype_, &phase, &n_,
                    nullptr, ia_.data(), ja_.data(),
                    nullptr, &nrhs, iparm_, &msglvl,
                    nullptr, nullptr, &error);
        }
    }

    // Non-copyable, movable
    PardisoSolver(const PardisoSolver&) = delete;
    PardisoSolver& operator=(const PardisoSolver&) = delete;
    PardisoSolver(PardisoSolver&&) = default;
    PardisoSolver& operator=(PardisoSolver&&) = default;

    /// Factorize from CSR matrix (0-based indexing, full symmetric storage).
    /// Extracts upper triangular part internally for PARDISO.
    void factorize(index_t n,
                   const index_t* row_ptr,
                   const index_t* col_idx,
                   const Scalar* values) {
        n_ = static_cast<MKL_INT>(n);

        // Extract upper triangular (col >= row) from full CSR
        extract_upper_triangle(n, row_ptr, col_idx, values);

        // Phase 11: Symbolic factorization (analysis)
        MKL_INT phase = 11;
        MKL_INT maxfct = 1, mnum = 1, nrhs = 1, msglvl = 0, error = 0;
        pardiso(pt_, &maxfct, &mnum, &mtype_, &phase, &n_,
                a_.data(), ia_.data(), ja_.data(),
                nullptr, &nrhs, iparm_, &msglvl,
                nullptr, nullptr, &error);
        if (error != 0)
            throw std::runtime_error(
                "PARDISO analysis failed, error=" + std::to_string(error));

        // Phase 22: Numerical factorization
        phase = 22;
        pardiso(pt_, &maxfct, &mnum, &mtype_, &phase, &n_,
                a_.data(), ia_.data(), ja_.data(),
                nullptr, &nrhs, iparm_, &msglvl,
                nullptr, nullptr, &error);
        if (error != 0)
            throw std::runtime_error(
                "PARDISO factorization failed, error=" + std::to_string(error));
    }

    /// Solve A*x = rhs (uses previously computed factorization)
    void solve(const Scalar* rhs, Scalar* sol) const {
        MKL_INT phase = 33;  // solve + iterative refinement
        MKL_INT maxfct = 1, mnum = 1, nrhs = 1, msglvl = 0, error = 0;

        // PARDISO modifies iparm during solve, so cast away const
        pardiso(const_cast<void**>(pt_), &maxfct, &mnum,
                const_cast<MKL_INT*>(&mtype_), &phase,
                const_cast<MKL_INT*>(&n_),
                const_cast<Scalar*>(a_.data()),
                const_cast<MKL_INT*>(ia_.data()),
                const_cast<MKL_INT*>(ja_.data()),
                nullptr, &nrhs,
                const_cast<MKL_INT*>(iparm_),
                &msglvl,
                const_cast<Scalar*>(rhs), sol, &error);
        if (error != 0)
            throw std::runtime_error(
                "PARDISO solve failed, error=" + std::to_string(error));
    }

private:
    /// Extract upper triangular part from full symmetric CSR
    void extract_upper_triangle(index_t n,
                                const index_t* row_ptr,
                                const index_t* col_idx,
                                const Scalar* values) {
        // Count upper triangular entries per row
        ia_.resize(n + 1);
        ia_[0] = 0;
        for (index_t i = 0; i < n; ++i) {
            MKL_INT count = 0;
            for (index_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
                if (col_idx[k] >= i)
                    ++count;
            }
            ia_[i + 1] = ia_[i] + count;
        }

        // Fill upper triangular entries
        MKL_INT nnz_upper = ia_[n];
        ja_.resize(nnz_upper);
        a_.resize(nnz_upper);

        for (index_t i = 0; i < n; ++i) {
            MKL_INT pos = ia_[i];
            for (index_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
                if (col_idx[k] >= i) {
                    ja_[pos] = static_cast<MKL_INT>(col_idx[k]);
                    a_[pos] = values[k];
                    ++pos;
                }
            }
        }
    }

    mutable void* pt_[64];          // PARDISO internal state
    mutable MKL_INT iparm_[64];     // PARDISO control parameters
    MKL_INT mtype_;                 // matrix type (2=real SPD, 6=complex symmetric)
    MKL_INT n_ = 0;                 // matrix dimension

    // Upper-triangular CSR (0-based indexing)
    std::vector<MKL_INT> ia_;       // row_ptr
    std::vector<MKL_INT> ja_;       // col_idx
    std::vector<Scalar> a_;         // values
};

} // namespace sparsesolv

#endif // SPARSESOLV_DIRECT_PARDISO_SOLVER_HPP
