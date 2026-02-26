/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/// @file bddc_preconditioner.hpp
/// @brief Element-by-element BDDC preconditioner matching NGSolve's BDDC

#ifndef SPARSESOLV_PRECONDITIONERS_BDDC_PRECONDITIONER_HPP
#define SPARSESOLV_PRECONDITIONERS_BDDC_PRECONDITIONER_HPP

#include "../core/preconditioner.hpp"
#include "../core/dense_matrix.hpp"
#include "../core/sparse_matrix_view.hpp"
#include "../core/sparse_matrix_csr.hpp"
#include "../core/sparse_matrix_csr_builder.hpp"
#include "../core/parallel.hpp"
#include "../direct/pardiso_solver.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>

namespace sparsesolv {

/// DOF classification for BDDC
enum class DOFType : uint8_t {
    Wirebasket = 0,  ///< Vertex/edge DOFs (coarse space)
    Interface  = 1   ///< Face/interior DOFs (eliminated locally)
};

/// Element-by-element BDDC preconditioner
template<typename Scalar = double>
class BDDCPreconditioner : public Preconditioner<Scalar> {
public:
    BDDCPreconditioner() = default;

    void set_element_info(std::vector<std::vector<index_t>> element_dofs,
                          std::vector<DOFType> dof_types) {
        element_dofs_ = std::move(element_dofs);
        dof_types_ = std::move(dof_types);
    }

    void set_free_dofs(std::vector<bool> free_dofs) {
        free_dofs_ = std::move(free_dofs);
    }

    void set_element_matrices(std::vector<DenseMatrix<Scalar>> element_matrices) {
        element_matrices_ = std::move(element_matrices);
    }

    /// Access wirebasket DOF mapping (compact_idx -> global_dof)
    const std::vector<index_t>& wirebasket_dofs() const { return wb_dofs_; }

    void setup(const SparseMatrixView<Scalar>& A) override {
        n_total_ = A.rows();
        build_dof_maps();
        setup_element_bddc();
        build_pardiso_coarse_solver();
        this->is_setup_ = true;
    }

    void apply(const Scalar* x, Scalar* y, index_t size) const override {
        apply_element_bddc(x, y, size);
    }

    std::string name() const override { return "BDDC"; }

    index_t num_wirebasket_dofs() const { return n_wb_; }
    index_t num_interface_dofs() const { return n_if_; }

private:
    // Input data
    std::vector<std::vector<index_t>> element_dofs_;
    std::vector<DOFType> dof_types_;
    std::vector<bool> free_dofs_;

    // Dimensions
    index_t n_total_ = 0;
    index_t n_wb_ = 0;
    index_t n_if_ = 0;

    // DOF mapping (only free DOFs get compact indices)
    std::vector<index_t> wb_dofs_;   // compact_wb_idx -> global_dof
    std::vector<index_t> if_dofs_;   // compact_if_idx -> global_dof
    std::vector<index_t> wb_map_;    // global_dof -> compact_wb_idx (-1 if not wb)

    // Element-by-element mode data
    std::vector<DenseMatrix<Scalar>> element_matrices_;
    SparseMatrixCSR<Scalar> he_csr_;       // harmonic extension (full space, if->wb)
    SparseMatrixCSR<Scalar> het_csr_;      // harmonic extension transpose (full space, wb->if)
    SparseMatrixCSR<Scalar> is_csr_;       // inner solve (full space, if->if)
    SparseMatrixCSR<Scalar> wb_csr_;       // wirebasket Schur complement (compact n_wb x n_wb)
    std::unique_ptr<PardisoSolver<Scalar>> pardiso_solver_;  // coarse solver
    std::vector<double> weight_;           // DOF weights

    // Work vectors for apply (mutable for const apply)
    mutable std::vector<Scalar> work1_, work2_;
    mutable std::vector<Scalar> wb_work1_, wb_work2_;

    void build_dof_maps() {
        wb_dofs_.clear();
        if_dofs_.clear();
        wb_map_.assign(n_total_, -1);
        n_wb_ = 0;
        n_if_ = 0;

        for (index_t d = 0; d < n_total_; ++d) {
            if (!free_dofs_.empty() &&
                d < static_cast<index_t>(free_dofs_.size()) &&
                !free_dofs_[d])
                continue;

            if (d < static_cast<index_t>(dof_types_.size()) &&
                dof_types_[d] == DOFType::Interface) {
                if_dofs_.push_back(d);
                n_if_++;
            } else {
                wb_map_[d] = n_wb_;
                wb_dofs_.push_back(d);
                n_wb_++;
            }
        }
    }

    void setup_element_bddc() {
        int nthreads = get_num_threads();
        index_t n_elements = static_cast<index_t>(element_dofs_.size());

        // Thread-local CSR builders and weight vectors
        std::vector<SparseMatrixCSRBuilder<Scalar>> tl_he, tl_het, tl_is, tl_wb;
        std::vector<std::vector<double>> tl_weight;
        tl_he.reserve(nthreads);
        tl_het.reserve(nthreads);
        tl_is.reserve(nthreads);
        tl_wb.reserve(nthreads);
        tl_weight.reserve(nthreads);
        for (int t = 0; t < nthreads; ++t) {
            tl_he.emplace_back(n_total_, n_total_);
            tl_het.emplace_back(n_total_, n_total_);
            tl_is.emplace_back(n_total_, n_total_);
            tl_wb.emplace_back(n_wb_, n_wb_);
            tl_weight.emplace_back(n_total_, 0.0);
        }

        // Parallel element processing
        parallel_for(n_elements, [&](index_t e) {
            int tid = get_thread_id();
            process_element(e, tl_he[tid], tl_het[tid], tl_is[tid],
                            tl_wb[tid], tl_weight[tid]);
        });

        // Merge thread-local builders
        SparseMatrixCSRBuilder<Scalar> he_bld(n_total_, n_total_);
        SparseMatrixCSRBuilder<Scalar> het_bld(n_total_, n_total_);
        SparseMatrixCSRBuilder<Scalar> is_bld(n_total_, n_total_);
        SparseMatrixCSRBuilder<Scalar> wb_bld(n_wb_, n_wb_);
        for (int t = 0; t < nthreads; ++t) {
            he_bld.merge_from(std::move(tl_he[t]));
            het_bld.merge_from(std::move(tl_het[t]));
            is_bld.merge_from(std::move(tl_is[t]));
            wb_bld.merge_from(std::move(tl_wb[t]));
        }

        // Merge thread-local weights
        weight_.assign(n_total_, 0.0);
        for (int t = 0; t < nthreads; ++t)
            for (index_t i = 0; i < n_total_; ++i)
                weight_[i] += tl_weight[t][i];

        // Build CSR directly (per-row sort + dedup, no global sort)
        he_csr_ = he_bld.build();
        het_csr_ = het_bld.build();
        is_csr_ = is_bld.build();
        wb_csr_ = wb_bld.build();

        finalize_weights();

        // Free element matrices (no longer needed)
        element_matrices_.clear();
        element_matrices_.shrink_to_fit();

        // Pre-allocate work vectors
        work1_.resize(n_total_);
        work2_.resize(n_total_);
        wb_work1_.resize(n_wb_);
        wb_work2_.resize(n_wb_);
    }

    void process_element(size_t e,
                         SparseMatrixCSRBuilder<Scalar>& he_bld,
                         SparseMatrixCSRBuilder<Scalar>& het_bld,
                         SparseMatrixCSRBuilder<Scalar>& is_bld,
                         SparseMatrixCSRBuilder<Scalar>& wb_bld,
                         std::vector<double>& weight) {
        const auto& el_dofs = element_dofs_[e];
        const auto& elmat = element_matrices_[e];
        index_t nel = static_cast<index_t>(el_dofs.size());

        // Classify element DOFs into wirebasket and interface
        std::vector<index_t> local_wb, local_if;
        for (index_t k = 0; k < nel; ++k) {
            index_t d = el_dofs[k];
            if (d < 0) continue;
            if (!free_dofs_.empty() &&
                d < static_cast<index_t>(free_dofs_.size()) &&
                !free_dofs_[d])
                continue;
            if (d < static_cast<index_t>(dof_types_.size()) &&
                dof_types_[d] == DOFType::Interface)
                local_if.push_back(k);
            else
                local_wb.push_back(k);
        }

        index_t nw = static_cast<index_t>(local_wb.size());
        index_t ni = static_cast<index_t>(local_if.size());

        if (nw == 0) return;

        // Extract K_ww block
        DenseMatrix<Scalar> schur(nw, nw);
        for (index_t i = 0; i < nw; ++i)
            for (index_t j = 0; j < nw; ++j)
                schur(i, j) = elmat(local_wb[i], local_wb[j]);

        if (ni > 0) {
            DenseMatrix<Scalar> K_wi(nw, ni);
            DenseMatrix<Scalar> K_iw(ni, nw);
            DenseMatrix<Scalar> K_ii(ni, ni);

            for (index_t i = 0; i < nw; ++i)
                for (index_t j = 0; j < ni; ++j)
                    K_wi(i, j) = elmat(local_wb[i], local_if[j]);
            for (index_t i = 0; i < ni; ++i)
                for (index_t j = 0; j < nw; ++j)
                    K_iw(i, j) = elmat(local_if[i], local_wb[j]);
            for (index_t i = 0; i < ni; ++i)
                for (index_t j = 0; j < ni; ++j)
                    K_ii(i, j) = elmat(local_if[i], local_if[j]);

            // Element-level weight from K_ii diagonal
            std::vector<double> elem_weight(ni);
            for (index_t k = 0; k < ni; ++k)
                elem_weight[k] = std::abs(K_ii(k, k));

            DenseMatrix<Scalar> K_ii_inv = K_ii;
            K_ii_inv.invert();

            // harm_ext = -K_ii^{-1} * K_iw  (ni x nw)
            DenseMatrix<Scalar> harm_ext = DenseMatrix<Scalar>::multiply(K_ii_inv, K_iw);
            harm_ext.negate();

            // schur = K_ww - K_wi * K_ii^{-1} * K_iw
            DenseMatrix<Scalar>::multiply_add(K_wi, harm_ext, schur);

            // harm_ext_t = -K_wi * K_ii^{-1}  (nw x ni)
            DenseMatrix<Scalar> harm_ext_t = DenseMatrix<Scalar>::multiply(K_wi, K_ii_inv);
            harm_ext_t.negate();

            // Apply element weights (following NGSolve's AddMatrix)
            for (index_t k = 0; k < ni; ++k)
                harm_ext.scale_row(k, static_cast<Scalar>(elem_weight[k]));
            for (index_t l = 0; l < ni; ++l)
                harm_ext_t.scale_col(l, static_cast<Scalar>(elem_weight[l]));
            for (index_t k = 0; k < ni; ++k)
                K_ii_inv.scale_row(k, static_cast<Scalar>(elem_weight[k]));
            for (index_t l = 0; l < ni; ++l)
                K_ii_inv.scale_col(l, static_cast<Scalar>(elem_weight[l]));

            // Accumulate global weight for interface DOFs
            for (index_t k = 0; k < ni; ++k)
                weight[el_dofs[local_if[k]]] += elem_weight[k];

            // Map to global DOF indices and accumulate into CSR builder
            std::vector<index_t> g_if(ni), g_wb(nw);
            for (index_t k = 0; k < ni; ++k) g_if[k] = el_dofs[local_if[k]];
            for (index_t k = 0; k < nw; ++k) g_wb[k] = el_dofs[local_wb[k]];

            he_bld.add_submatrix(g_if.data(), ni, g_wb.data(), nw, harm_ext);
            het_bld.add_submatrix(g_wb.data(), nw, g_if.data(), ni, harm_ext_t);
            is_bld.add_submatrix(g_if.data(), ni, g_if.data(), ni, K_ii_inv);

            std::vector<index_t> c_wb(nw);
            for (index_t k = 0; k < nw; ++k) c_wb[k] = wb_map_[g_wb[k]];
            wb_bld.add_submatrix(c_wb.data(), nw, c_wb.data(), nw, schur);
        } else {
            // No interface DOFs: add K_ww directly to wirebasket
            std::vector<index_t> g_wb(nw), c_wb(nw);
            for (index_t k = 0; k < nw; ++k) {
                g_wb[k] = el_dofs[local_wb[k]];
                c_wb[k] = wb_map_[g_wb[k]];
            }
            wb_bld.add_submatrix(c_wb.data(), nw, c_wb.data(), nw, schur);
        }
    }

    void finalize_weights() {
        for (index_t i = 0; i < n_total_; ++i)
            if (weight_[i] > 0.0)
                weight_[i] = 1.0 / weight_[i];

        // Scale inner_solve: is[i,j] *= w[i] * w[j]
        for (index_t i = 0; i < is_csr_.rows; ++i) {
            double wi = weight_[i];
            for (index_t k = is_csr_.row_ptr[i]; k < is_csr_.row_ptr[i + 1]; ++k) {
                index_t j = is_csr_.col_idx[k];
                is_csr_.values[k] *= static_cast<Scalar>(wi * weight_[j]);
            }
        }

        // Scale harmonic_ext: he[i,:] *= w[i]
        for (index_t i = 0; i < he_csr_.rows; ++i) {
            double wi = weight_[i];
            for (index_t k = he_csr_.row_ptr[i]; k < he_csr_.row_ptr[i + 1]; ++k)
                he_csr_.values[k] *= static_cast<Scalar>(wi);
        }

        // Scale harmonic_ext_trans: het[:,j] *= w[j]
        for (index_t i = 0; i < het_csr_.rows; ++i) {
            for (index_t k = het_csr_.row_ptr[i]; k < het_csr_.row_ptr[i + 1]; ++k) {
                index_t j = het_csr_.col_idx[k];
                het_csr_.values[k] *= static_cast<Scalar>(weight_[j]);
            }
        }
    }

    void build_pardiso_coarse_solver() {
        pardiso_solver_ = std::make_unique<PardisoSolver<Scalar>>();
        pardiso_solver_->factorize(
            n_wb_,
            wb_csr_.row_ptr.data(),
            wb_csr_.col_idx.data(),
            wb_csr_.values.data());
    }

    /// Apply BDDC: y = (I + he) * (S_wb^{-1} * (I + he^T) * x + is * x)
    /// Fused SpMV-add with empty row skip: het has wb rows only, he/is have if rows only.
    void apply_element_bddc(const Scalar* x, Scalar* y, index_t size) const {
        // Step 1: y = x
        std::copy(x, x + size, y);

        // Step 2: y += het * x (wirebasket rows only — het has entries in wb rows)
        parallel_for(n_wb_, [&](index_t k) {
            index_t i = wb_dofs_[k];
            Scalar sum = Scalar(0);
            for (index_t p = het_csr_.row_ptr[i]; p < het_csr_.row_ptr[i + 1]; ++p)
                sum += het_csr_.values[p] * x[het_csr_.col_idx[p]];
            y[i] += sum;
        });

        // Step 3: Coarse solve (wirebasket inverse)
        for (index_t k = 0; k < n_wb_; ++k)
            wb_work1_[k] = y[wb_dofs_[k]];

        pardiso_solver_->solve(wb_work1_.data(), wb_work2_.data());

        std::fill(work1_.begin(), work1_.end(), Scalar(0));
        for (index_t k = 0; k < n_wb_; ++k)
            work1_[wb_dofs_[k]] = wb_work2_[k];

        // Step 4: work1_ += is * x (interface rows only — is has entries in if rows)
        parallel_for(n_if_, [&](index_t k) {
            index_t i = if_dofs_[k];
            Scalar sum = Scalar(0);
            for (index_t p = is_csr_.row_ptr[i]; p < is_csr_.row_ptr[i + 1]; ++p)
                sum += is_csr_.values[p] * x[is_csr_.col_idx[p]];
            work1_[i] += sum;
        });

        // Step 5: y = work1_ + he * work1_
        // Wirebasket rows: he has no entries, so y = work1_ directly
        parallel_for(n_wb_, [&](index_t k) {
            index_t i = wb_dofs_[k];
            y[i] = work1_[i];
        });
        // Interface rows: fused SpMV-add (he has entries in if rows only)
        parallel_for(n_if_, [&](index_t k) {
            index_t i = if_dofs_[k];
            Scalar sum = Scalar(0);
            for (index_t p = he_csr_.row_ptr[i]; p < he_csr_.row_ptr[i + 1]; ++p)
                sum += he_csr_.values[p] * work1_[he_csr_.col_idx[p]];
            y[i] = work1_[i] + sum;
        });
    }
};

} // namespace sparsesolv

#endif // SPARSESOLV_PRECONDITIONERS_BDDC_PRECONDITIONER_HPP
