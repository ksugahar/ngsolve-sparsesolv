/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/**
 * @file diffop_simd_stubs.cpp
 * @brief Stub implementations for DifferentialOperator SIMD virtual methods
 *
 * When building sparsesolv_ngsolve.pyd against a freshly-built NGSolve wheel,
 * the 6 SIMD virtual methods of DifferentialOperator may not be exported from
 * libngsolve.dll (Windows DLL export policy gap). These base-class methods
 * just throw ExceptionNOSIMD in the official NGSolve source, so providing
 * identical stubs here satisfies the linker without changing runtime behavior.
 *
 * The vtable entries for these methods will point to our local stubs. Since
 * derived classes override them with actual SIMD implementations (which ARE
 * exported from libngsolve.dll), these stubs are never called at runtime.
 *
 * If NGSolve fixes its DLL export policy in a future release, these stubs
 * can be removed.
 */

// Define NGS_EXPORTS so NGS_DLL_HEADER expands to __declspec(dllexport)
// instead of __declspec(dllimport). This allows us to provide local
// definitions of the 6 methods.
#define NGS_EXPORTS

#include <fem.hpp>

namespace ngfem {

  void DifferentialOperator::CalcMatrix(
      const FiniteElement & fel,
      const SIMD_BaseMappedIntegrationRule & mir,
      BareSliceMatrix<SIMD<double>> mat) const
  {
    throw ExceptionNOSIMD(string("DifferentialOperator::CalcMatrix(SIMD<double>) stub - type = ")
                          + typeid(*this).name());
  }

  void DifferentialOperator::CalcMatrix(
      const FiniteElement & fel,
      const SIMD_BaseMappedIntegrationRule & mir,
      BareSliceMatrix<SIMD<Complex>> mat) const
  {
    throw ExceptionNOSIMD(string("DifferentialOperator::CalcMatrix(SIMD<Complex>) stub - type = ")
                          + typeid(*this).name());
  }

  void DifferentialOperator::Apply(
      const FiniteElement & bfel,
      const SIMD_BaseMappedIntegrationRule & bmir,
      BareSliceVector<double> x,
      BareSliceMatrix<SIMD<double>> flux) const
  {
    throw ExceptionNOSIMD(string("DifferentialOperator::Apply(SIMD<double>) stub - type = ")
                          + typeid(*this).name());
  }

  void DifferentialOperator::Apply(
      const FiniteElement & bfel,
      const SIMD_BaseMappedIntegrationRule & bmir,
      BareSliceVector<Complex> x,
      BareSliceMatrix<SIMD<Complex>> flux) const
  {
    throw ExceptionNOSIMD(string("DifferentialOperator::Apply(SIMD<Complex>) stub - type = ")
                          + typeid(*this).name());
  }

  void DifferentialOperator::AddTrans(
      const FiniteElement & bfel,
      const SIMD_BaseMappedIntegrationRule & bmir,
      BareSliceMatrix<SIMD<double>> flux,
      BareSliceVector<double> x) const
  {
    throw ExceptionNOSIMD(string("DifferentialOperator::AddTrans(SIMD<double>) stub - type = ")
                          + typeid(*this).name());
  }

  void DifferentialOperator::AddTrans(
      const FiniteElement & bfel,
      const SIMD_BaseMappedIntegrationRule & bmir,
      BareSliceMatrix<SIMD<Complex>> flux,
      BareSliceVector<Complex> x) const
  {
    throw ExceptionNOSIMD(string("DifferentialOperator::AddTrans(SIMD<Complex>) stub - type = ")
                          + typeid(*this).name());
  }

}  // namespace ngfem
