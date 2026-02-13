/**
 * @file python_bindings.cpp
 * @brief SparseSolv pybind11 bindings for NGSolve integration
 *
 * REFERENCE COPY - This file is NOT compiled directly.
 * It documents how to add SparseSolv Python bindings to NGSolve's
 * linalg/python_linalg.cpp.
 *
 * Integration steps:
 * 1. Add to the includes at the top of python_linalg.cpp:
 *      #include "sparsesolv_python_export.hpp"
 *
 * 2. Add to the ExportNgla() function (before closing brace):
 *      // SparseSolv Preconditioners and Solvers
 *      // Based on JP-MARs/SparseSolv (https://github.com/JP-MARs/SparseSolv)
 *      ExportSparseSolvBindings(m);
 *
 * This registers the following Python classes and factory functions:
 *
 *   Factory functions (auto-dispatch via mat->IsComplex()):
 *   - ICPreconditioner(mat, freedofs, shift)
 *   - SGSPreconditioner(mat, freedofs)
 *   - ILUPreconditioner(mat, freedofs, shift)
 *   - SparseSolvSolver(mat, method, freedofs, tol, ...)
 *
 *   Internal typed classes (D=double, C=Complex):
 *   - ICPreconditionerD, ICPreconditionerC
 *   - SGSPreconditionerD, SGSPreconditionerC
 *   - ILUPreconditionerD, ILUPreconditionerC
 *   - SparseSolvSolverD, SparseSolvSolverC
 *   - SparseSolvResult
 *
 * See sparsesolv_python_export.hpp for the full implementation.
 */
