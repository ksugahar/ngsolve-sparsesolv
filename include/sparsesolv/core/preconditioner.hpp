/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/// @file preconditioner.hpp
/// @brief Base class for preconditioners

#ifndef SPARSESOLV_CORE_PRECONDITIONER_HPP
#define SPARSESOLV_CORE_PRECONDITIONER_HPP

#include "types.hpp"
#include "constants.hpp"
#include "parallel.hpp"
#include "sparse_matrix_view.hpp"
#include <string>
#include <memory>

namespace sparsesolv {

/// Abstract base class for preconditioners: setup(A), then apply(x, y) = M^{-1}*x
template<typename Scalar = double>
class Preconditioner {
public:
    virtual ~Preconditioner() = default;

    virtual void setup(const SparseMatrixView<Scalar>& A) = 0;
    virtual void apply(const Scalar* x, Scalar* y, index_t size) const = 0;

    void apply(const std::vector<Scalar>& x, std::vector<Scalar>& y) const {
        assert(x.size() == y.size());
        apply(x.data(), y.data(), static_cast<index_t>(x.size()));
    }

    virtual std::string name() const = 0;
    virtual bool is_ready() const { return is_setup_; }

protected:
    bool is_setup_ = false;
};

} // namespace sparsesolv

#endif // SPARSESOLV_CORE_PRECONDITIONER_HPP
