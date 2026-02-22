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
#include "../core/sparse_matrix_coo.hpp"
#include "../core/sparse_matrix_csr.hpp"
#include <vector>
#include <functional>
#include <algorithm>
#include <cmath>

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

    /// Set external coarse solver: void(const Scalar* rhs, Scalar* sol) on compact wb vectors
    void set_coarse_solver(std::function<void(const Scalar*, Scalar*)> solver) {
        coarse_solver_ = std::move(solver);
    }

    /// Access wirebasket CSR matrix (available after setup, compact n_wb x n_wb)
    const SparseMatrixCSR<Scalar>& wirebasket_csr() const { return wb_csr_; }

    /// Access wirebasket DOF mapping (compact_idx -> global_dof)
    const std::vector<index_t>& wirebasket_dofs() const { return wb_dofs_; }

    void setup(const SparseMatrixView<Scalar>& A) override {
        n_total_ = A.rows();
        build_dof_maps();
        setup_element_bddc();
        this->is_setup_ = true;
    }

    void apply(const Scalar* x, Scalar* y, index_t size) const override {
        apply_element_bddc(x, y, size);
    }

    std::string name() const override { return "BDDC"; }

    index_t num_wirebasket_dofs() const { return n_wb_; }
    index_t num_interface_dofs() const { return n_if_; }

private:
    std::function<void(const Scalar*, Scalar*)> coarse_solver_;

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
        // Initialize COO accumulators
        SparseMatrixCOO<Scalar> he_coo(n_total_, n_total_);
        SparseMatrixCOO<Scalar> het_coo(n_total_, n_total_);
        SparseMatrixCOO<Scalar> is_coo(n_total_, n_total_);
        SparseMatrixCOO<Scalar> wb_coo(n_wb_, n_wb_);

        weight_.assign(n_total_, 0.0);

        for (size_t e = 0; e < element_dofs_.size(); ++e) {
            process_element(e, he_coo, het_coo, is_coo, wb_coo);
        }

        // Convert COO to CSR (sums duplicate entries)
        he_csr_ = he_coo.to_csr();
        het_csr_ = het_coo.to_csr();
        is_csr_ = is_coo.to_csr();
        wb_csr_ = wb_coo.to_csr();

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
                         SparseMatrixCOO<Scalar>& he_coo,
                         SparseMatrixCOO<Scalar>& het_coo,
                         SparseMatrixCOO<Scalar>& is_coo,
                         SparseMatrixCOO<Scalar>& wb_coo) {
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
                weight_[el_dofs[local_if[k]]] += elem_weight[k];

            // Map to global DOF indices and accumulate into COO
            std::vector<index_t> g_if(ni), g_wb(nw);
            for (index_t k = 0; k < ni; ++k) g_if[k] = el_dofs[local_if[k]];
            for (index_t k = 0; k < nw; ++k) g_wb[k] = el_dofs[local_wb[k]];

            he_coo.add_submatrix(g_if.data(), ni, g_wb.data(), nw, harm_ext);
            het_coo.add_submatrix(g_wb.data(), nw, g_if.data(), ni, harm_ext_t);
            is_coo.add_submatrix(g_if.data(), ni, g_if.data(), ni, K_ii_inv);

            std::vector<index_t> c_wb(nw);
            for (index_t k = 0; k < nw; ++k) c_wb[k] = wb_map_[g_wb[k]];
            wb_coo.add_submatrix(c_wb.data(), nw, c_wb.data(), nw, schur);
        } else {
            // No interface DOFs: add K_ww directly to wirebasket
            std::vector<index_t> g_wb(nw), c_wb(nw);
            for (index_t k = 0; k < nw; ++k) {
                g_wb[k] = el_dofs[local_wb[k]];
                c_wb[k] = wb_map_[g_wb[k]];
            }
            wb_coo.add_submatrix(c_wb.data(), nw, c_wb.data(), nw, schur);
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

    /// Apply BDDC: y = (I + he) * (S_wb^{-1} * (I + he^T) * x + is * x)
    void apply_element_bddc(const Scalar* x, Scalar* y, index_t size) const {
        // Step 1: y = x
        std::copy(x, x + size, y);

        // Step 2: y += het * x
        het_csr_.multiply(x, work1_.data());
        for (index_t i = 0; i < size; ++i)
            y[i] += work1_[i];

        // Step 3: Coarse solve (wirebasket inverse)
        for (index_t k = 0; k < n_wb_; ++k)
            wb_work1_[k] = y[wb_dofs_[k]];

        coarse_solver_(wb_work1_.data(), wb_work2_.data());

        std::fill(work1_.begin(), work1_.end(), Scalar(0));
        for (index_t k = 0; k < n_wb_; ++k)
            work1_[wb_dofs_[k]] = wb_work2_[k];

        // Step 4: tmp += inner_solve * x
        is_csr_.multiply(x, work2_.data());
        for (index_t i = 0; i < size; ++i)
            work1_[i] += work2_[i];

        // Step 5: y = tmp + he * tmp
        he_csr_.multiply(work1_.data(), work2_.data());
        for (index_t i = 0; i < size; ++i)
            y[i] = work1_[i] + work2_[i];
    }
};

} // namespace sparsesolv

#endif // SPARSESOLV_PRECONDITIONERS_BDDC_PRECONDITIONER_HPP
