/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/**
 * @file bicgstab_solver.hpp
 * @brief Preconditioned BiCGStab solver for non-symmetric linear systems
 *
 * Reference: Van der Vorst (1992), "Bi-CGSTAB: A Fast and Smoothly
 * Converging Variant of Bi-CG for the Solution of Nonsymmetric Linear
 * Systems", SIAM J. Sci. Stat. Comput., 13(2), 631-644.
 *
 * For complex systems, uses conjugated inner products (a^H * b).
 * Fixed memory: 9 work vectors regardless of iteration count.
 *
 * Performance: Fused SpMV+dot kernels minimize memory passes.
 * Each iteration: 2 SpMV + 2 precond + ~5 vector passes (vs ~9 naive).
 * MKL note: Fused kernels are faster than separate MKL BLAS1 calls
 * because they avoid multiple passes over large vectors. MKL sparse
 * BLAS (mkl_sparse_d_mv) could improve SpMV but conflicts with the
 * fusion and TaskManager parallelization already in use.
 */

#ifndef SPARSESOLV_SOLVERS_BICGSTAB_SOLVER_HPP
#define SPARSESOLV_SOLVERS_BICGSTAB_SOLVER_HPP

#include "iterative_solver.hpp"
#include "../core/constants.hpp"

namespace sparsesolv {

/**
 * @brief Preconditioned BiCGStab for non-symmetric systems (right preconditioning)
 *
 * Solves A*x = b with optional right preconditioner M:
 *   Solve A*M^{-1}*(M*x) = b
 *
 * Key advantages over GMRES:
 * - Fixed 9 work vectors (GMRES: 1 vector per iteration + restart overhead)
 * - No Gram-Schmidt orthogonalization (O(k^2*N) for GMRES(k))
 * - O(N) memory regardless of iteration count
 *
 * Key advantage over COCR:
 * - Works for general non-symmetric systems (COCR requires A^T = A)
 * - Suitable for eddy current with AMS preconditioner (non-symmetric)
 *
 * Cost per iteration: 2 SpMV + 2 preconditioner applies + O(N) vector ops
 *
 * @tparam Scalar The scalar type (double or complex<double>)
 */
template<typename Scalar = double>
class BiCGStabSolver : public IterativeSolver<Scalar> {
public:
    std::string name() const override { return "BiCGStab"; }

protected:
    void allocate_work_vectors() override {
        IterativeSolver<Scalar>::allocate_work_vectors();
        r0_.resize(this->size_);
        s_.resize(this->size_);
        t_.resize(this->size_);
        p_hat_.resize(this->size_);
        s_hat_.resize(this->size_);
    }

    SolverResult do_iterate() override {
        const index_t n = this->size_;
        auto& r = this->r_;       // residual (from base class)
        auto& p = this->p_;       // search direction (from base class)
        auto& v = this->Ap_;      // A*p_hat (reuse base class Ap_)
        Scalar* x = this->x_;
        const auto& config = this->config_;

        // Shadow residual: r0 = r (fixed throughout iteration)
        std::copy(r.begin(), r.end(), r0_.begin());

        // Initialize scalars
        Scalar rho(1), alpha(1), omega(1);
        std::fill(v.begin(), v.end(), Scalar(0));
        std::fill(p.begin(), p.end(), Scalar(0));

        // Cache CSR pointers for fused SpMV+dot
        const auto* rowptr = this->A_->row_ptr();
        const auto* colidx = this->A_->col_idx();
        const auto* vals = this->A_->values();

        for (int iter = 0; iter < config.max_iterations; ++iter) {
            Scalar rho_old = rho;

            // rho = <r0, r> (conjugated for complex)
            rho = conj_dot(r0_.data(), r.data(), n);

            if (std::abs(rho) < constants::BREAKDOWN_THRESHOLD) {
                double nr = this->compute_norm(r.data(), n);
                return this->build_result(
                    this->check_convergence(nr, iter), iter, nr);
            }

            // Direction update
            if (iter == 0) {
                std::copy(r.begin(), r.end(), p.begin());
            } else {
                if (std::abs(rho_old * omega) < constants::BREAKDOWN_THRESHOLD) {
                    double nr = this->compute_norm(r.data(), n);
                    return this->build_result(false, iter, nr);
                }
                Scalar beta = (rho / rho_old) * (alpha / omega);
                // p = r + beta * (p - omega * v)
                parallel_for(n, [&](index_t i) {
                    p[i] = r[i] + beta * (p[i] - omega * v[i]);
                });
            }

            // Precondition: p_hat = M^{-1} * p
            if (this->precond_) {
                this->precond_->apply(p.data(), p_hat_.data(), n);
            } else {
                std::copy(p.begin(), p.end(), p_hat_.begin());
            }

            // Fused: v = A*p_hat AND <r0, v> (one memory pass over SpMV)
            Scalar r0v = fused_spmv_conj_dot(
                rowptr, colidx, vals, n,
                p_hat_.data(), v.data(), r0_.data());

            if (std::abs(r0v) < constants::BREAKDOWN_THRESHOLD) {
                double nr = this->compute_norm(r.data(), n);
                return this->build_result(false, iter, nr);
            }
            alpha = rho / r0v;

            // Fused: s = r - alpha*v AND ||s||
            double norm_s = fused_update_norm(
                n, r.data(), Scalar(-1) * alpha, v.data(), s_.data());

            // Early convergence on s
            if (norm_s / this->normalizer_ < config.tolerance ||
                norm_s < config.abs_tolerance) {
                parallel_for(n, [&](index_t i) {
                    x[i] += alpha * p_hat_[i];
                });
                this->check_convergence(norm_s, iter);
                return this->build_result(true, iter + 1, norm_s);
            }

            // Precondition: s_hat = M^{-1} * s
            if (this->precond_) {
                this->precond_->apply(s_.data(), s_hat_.data(), n);
            } else {
                std::copy(s_.begin(), s_.end(), s_hat_.begin());
            }

            // Fused: t = A*s_hat AND ||t||^2
            double t_norm_sq;
            fused_spmv_norm_sq(
                rowptr, colidx, vals, n,
                s_hat_.data(), t_.data(), t_norm_sq);

            if (t_norm_sq < constants::BREAKDOWN_THRESHOLD) {
                parallel_for(n, [&](index_t i) {
                    x[i] += alpha * p_hat_[i];
                });
                this->check_convergence(norm_s, iter);
                return this->build_result(false, iter + 1, norm_s);
            }

            // omega = <t, s> / ||t||^2
            Scalar tds = conj_dot(t_.data(), s_.data(), n);
            omega = tds / Scalar(t_norm_sq);

            // Fused: x += alpha*p_hat + omega*s_hat; r = s - omega*t; ||r||
            double norm_r = fused_solution_update(
                n, x, alpha, p_hat_.data(),
                omega, s_hat_.data(),
                r.data(), s_.data(), t_.data());

            if (this->check_convergence(norm_r, iter)) {
                return this->build_result(true, iter + 1, norm_r);
            }

            if (this->is_diverging()) {
                return this->build_result(false, iter + 1, norm_r);
            }

            // Detect numerical blowup
            if (std::isnan(norm_r) || std::isinf(norm_r)) {
                return this->build_result(false, iter + 1, norm_r);
            }
        }

        double fnorm = this->compute_norm(r.data(), n);
        return this->build_result(false, config.max_iterations, fnorm);
    }

private:
    /**
     * @brief Conjugated dot product: conj(a)^T * b
     *
     * For real Scalar: standard dot product.
     * For complex Scalar: sesquilinear (a^H * b).
     */
    Scalar conj_dot(const Scalar* a, const Scalar* b, index_t n) const {
        if constexpr (std::is_same_v<Scalar, double>) {
            return parallel_reduce_sum<Scalar>(n, [a, b](index_t i) -> Scalar {
                return a[i] * b[i];
            });
        } else {
            return parallel_reduce_sum<Scalar>(n, [a, b](index_t i) -> Scalar {
                return std::conj(a[i]) * b[i];
            });
        }
    }

    /**
     * @brief Fused SpMV + conjugated dot: y = A*x; return conj(d)^T * y
     *
     * Combines sparse matrix-vector multiply with dot product in a single
     * parallel pass. The dot product reads d[i] sequentially (free, hidden
     * by SpMV's random-access latency to x via col_idx).
     */
    Scalar fused_spmv_conj_dot(
        const index_t* rowptr, const index_t* colidx, const Scalar* vals,
        index_t n, const Scalar* x, Scalar* y, const Scalar* d) const
    {
        return parallel_reduce_sum<Scalar>(n, [&](index_t i) -> Scalar {
            Scalar s = Scalar(0);
            for (index_t k = rowptr[i]; k < rowptr[i + 1]; ++k)
                s += vals[k] * x[colidx[k]];
            y[i] = s;
            if constexpr (std::is_same_v<Scalar, double>)
                return d[i] * s;
            else
                return std::conj(d[i]) * s;
        });
    }

    /**
     * @brief Fused: y = a + scale*b AND return ||y||
     *
     * Computes vector update and norm in one pass.
     */
    double fused_update_norm(
        index_t n, const Scalar* a, Scalar scale,
        const Scalar* b, Scalar* y) const
    {
        double sq = parallel_reduce_sum<double>(n, [&](index_t i) -> double {
            Scalar yi = a[i] + scale * b[i];
            y[i] = yi;
            return std::norm(yi);
        });
        return std::sqrt(sq);
    }

    /**
     * @brief Fused SpMV + norm: y = A*x AND norm_sq = ||y||^2
     */
    void fused_spmv_norm_sq(
        const index_t* rowptr, const index_t* colidx, const Scalar* vals,
        index_t n, const Scalar* x, Scalar* y, double& norm_sq) const
    {
        norm_sq = parallel_reduce_sum<double>(n, [&](index_t i) -> double {
            Scalar s = Scalar(0);
            for (index_t k = rowptr[i]; k < rowptr[i + 1]; ++k)
                s += vals[k] * x[colidx[k]];
            y[i] = s;
            return std::norm(s);
        });
    }

    /**
     * @brief Fused solution update: x += a*ph + w*sh; r = s - w*t; return ||r||
     *
     * Combines three vector operations and norm in one parallel pass:
     * 1. Solution update: x[i] += alpha*p_hat[i] + omega*s_hat[i]
     * 2. Residual update: r[i] = s[i] - omega*t[i]
     * 3. Residual norm: sum |r[i]|^2
     */
    double fused_solution_update(
        index_t n, Scalar* x, Scalar a, const Scalar* ph,
        Scalar w, const Scalar* sh,
        Scalar* r, const Scalar* s, const Scalar* t) const
    {
        double sq = parallel_reduce_sum<double>(n, [&](index_t i) -> double {
            x[i] += a * ph[i] + w * sh[i];
            Scalar ri = s[i] - w * t[i];
            r[i] = ri;
            return std::norm(ri);
        });
        return std::sqrt(sq);
    }

    std::vector<Scalar> r0_;     // shadow residual (fixed throughout)
    std::vector<Scalar> s_;      // intermediate residual
    std::vector<Scalar> t_;      // A * s_hat
    std::vector<Scalar> p_hat_;  // preconditioned p
    std::vector<Scalar> s_hat_;  // preconditioned s
};

} // namespace sparsesolv

#endif // SPARSESOLV_SOLVERS_BICGSTAB_SOLVER_HPP
