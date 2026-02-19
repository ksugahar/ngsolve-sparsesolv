/**
 * @file python_module.cpp
 * @brief Standalone pybind11 module for SparseSolv NGSolve integration
 *
 * Builds as an independent .pyd/.so that links against NGSolve.
 * Usage: import sparsesolv_ngsolve
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "sparsesolv/ngsolve/sparsesolv_python_export.hpp"

namespace py = pybind11;

PYBIND11_MODULE(sparsesolv_ngsolve, m) {
    // Import NGSolve's la module so pybind11 knows about BaseMatrix, SparseMatrix, etc.
    py::module_::import("ngsolve.la");

    m.doc() = "SparseSolv iterative solvers for NGSolve\n\n"
              "Provides IC/SGS preconditioners and ICCG/SGSMRTR iterative solvers\n"
              "for use with NGSolve's sparse linear algebra.\n\n"
              "Based on JP-MARs/SparseSolv (https://github.com/JP-MARs/SparseSolv)";

    ngla::ExportSparseSolvBindings(m);
}
