# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
Tests for SparseSolv BDDC preconditioner.

Tests element-by-element BDDC against NGSolve's built-in BDDC on 3D problems.
"""

import pytest
import numpy as np

import ngsolve
from ngsolve import *
from ngsolve.krylovspace import CGSolver
from netgen.occ import Box, Pnt
from sparsesolv_ngsolve import BDDCPreconditioner


def create_3d_poisson(maxh=0.4, order=3):
    """Create 3D Poisson problem on unit cube."""
    box = Box(Pnt(0, 0, 0), Pnt(1, 1, 1))
    for face in box.faces:
        face.name = "outer"
    mesh = box.GenerateMesh(maxh=maxh)

    fes = H1(mesh, order=order, dirichlet="outer")
    u, v = fes.TnT()

    a = BilinearForm(fes)
    a += InnerProduct(grad(u), grad(v)) * dx
    a.Assemble()

    f = LinearForm(fes)
    f += 1 * v * dx
    f.Assemble()

    return mesh, fes, a, f


def test_bddc_3d_poisson():
    """Test SparseSolv BDDC on 3D Poisson with H1 order=3."""
    mesh, fes, a, f = create_3d_poisson(maxh=0.4, order=3)

    pre = BDDCPreconditioner(a, fes)

    gfu = GridFunction(fes)
    inv = CGSolver(a.mat, pre, printrates=False, tol=1e-10, maxiter=500)
    gfu.vec.data = inv * f.vec
    print(f"\nSparseSolv BDDC: {inv.iterations} iterations, DOF={fes.ndof}")

    assert inv.iterations < 500, "SparseSolv BDDC did not converge"

    # Reference: direct solver
    gfu_ref = GridFunction(fes)
    gfu_ref.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f.vec

    err = sqrt(Integrate((gfu - gfu_ref)**2, mesh))
    ref_norm = sqrt(Integrate(gfu_ref**2, mesh))
    rel_err = err / ref_norm if ref_norm > 0 else err
    print(f"  Relative error vs direct: {rel_err:.2e}")
    assert rel_err < 1e-6, f"Solution error too large: {rel_err}"


def test_bddc_vs_ngsolve_bddc():
    """Compare SparseSolv BDDC with NGSolve's built-in BDDC."""
    mesh, fes, a, f = create_3d_poisson(maxh=0.35, order=3)
    print(f"\nDOF={fes.ndof}")

    # SparseSolv BDDC
    pre_ss = BDDCPreconditioner(a, fes)

    gfu_ss = GridFunction(fes)
    inv_ss = CGSolver(a.mat, pre_ss, printrates=False, tol=1e-10, maxiter=500)
    gfu_ss.vec.data = inv_ss * f.vec
    iters_ss = inv_ss.iterations

    # NGSolve BDDC
    a2 = BilinearForm(fes)
    a2 += InnerProduct(grad(fes.TnT()[0]), grad(fes.TnT()[1])) * dx
    c_ng = Preconditioner(a2, type="bddc")
    a2.Assemble()

    gfu_ng = GridFunction(fes)
    inv_ng = CGSolver(a2.mat, c_ng.mat, printrates=False, tol=1e-10, maxiter=500)
    gfu_ng.vec.data = inv_ng * f.vec
    iters_ng = inv_ng.iterations

    print(f"SparseSolv BDDC: {iters_ss} iterations")
    print(f"NGSolve BDDC:    {iters_ng} iterations")

    # Both should produce similar solutions
    err = sqrt(Integrate((gfu_ss - gfu_ng)**2, mesh))
    ref_norm = sqrt(Integrate(gfu_ng**2, mesh))
    rel_err = err / ref_norm if ref_norm > 0 else err
    print(f"Relative difference: {rel_err:.2e}")
    assert rel_err < 1e-6, f"Solutions differ too much: {rel_err}"

    # Iteration count should be comparable
    assert iters_ss <= 2 * iters_ng + 5, \
        f"SparseSolv BDDC needs too many iterations: {iters_ss} vs NGSolve {iters_ng}"


def test_bddc_mesh_independence():
    """Verify that BDDC iteration count is roughly mesh-independent."""
    iters_list = []
    for maxh in [0.4, 0.3, 0.25]:
        mesh, fes, a, f = create_3d_poisson(maxh=maxh, order=3)

        pre = BDDCPreconditioner(a, fes)

        gfu = GridFunction(fes)
        inv = CGSolver(a.mat, pre, printrates=False, tol=1e-10, maxiter=500)
        gfu.vec.data = inv * f.vec

        iters_list.append(inv.iterations)
        print(f"maxh={maxh}, DOF={fes.ndof}, iters={inv.iterations}")

    # Iterations should not grow significantly with refinement
    assert max(iters_list) < 3 * min(iters_list), \
        f"Iterations not mesh-independent: {iters_list}"


def test_bddc_shifted_curl_curl():
    """Shifted BDDC on HCurl curl-curl problem."""
    box = Box(Pnt(0, 0, 0), Pnt(1, 1, 1))
    for face in box.faces:
        face.name = "outer"
    mesh = box.GenerateMesh(maxh=0.3)

    fes = HCurl(mesh, order=1, dirichlet="outer", nograds=True)
    u, v = fes.TnT()

    # Regularized matrix for BDDC
    sigma = 1e-6
    a = BilinearForm(fes)
    a += InnerProduct(curl(u), curl(v)) * dx
    a += sigma * InnerProduct(u, v) * dx
    a.Assemble()

    # Curl-based source (div J = 0 guaranteed)
    T = CF((y * (1 - y) * z * (1 - z),
            z * (1 - z) * x * (1 - x),
            x * (1 - x) * y * (1 - y)))
    f_form = LinearForm(fes)
    f_form += InnerProduct(T, curl(v)) * dx
    f_form.Assemble()

    print(f"\nHCurl DOF={fes.ndof}")

    pre = BDDCPreconditioner(a, fes)

    gfu = GridFunction(fes)
    inv = CGSolver(a.mat, pre, printrates=False, tol=1e-10, maxiter=500)
    gfu.vec.data = inv * f_form.vec
    print(f"Shifted SparseSolv BDDC: {inv.iterations} iterations")

    # Reference: direct solver
    gfu_ref = GridFunction(fes)
    gfu_ref.vec.data = a.mat.Inverse(fes.FreeDofs(),
                                      inverse="sparsecholesky") * f_form.vec

    B = curl(gfu)
    B_ref = curl(gfu_ref)
    B_err = sqrt(Integrate(InnerProduct(B - B_ref, B - B_ref), mesh))
    B_norm = sqrt(Integrate(InnerProduct(B_ref, B_ref), mesh))
    rel_err = B_err / B_norm if B_norm > 0 else B_err
    print(f"B = curl(A) relative error: {rel_err:.2e}")

    assert inv.iterations < 500, "Shifted BDDC did not converge"
    assert rel_err < 1e-4, f"B error too large: {rel_err}"


def test_bddc_eddy_current():
    """BDDC on complex HCurl eddy current problem."""
    box = Box(Pnt(0, 0, 0), Pnt(1, 1, 1))
    for face in box.faces:
        face.name = "outer"
    mesh = box.GenerateMesh(maxh=0.4)

    fes = HCurl(mesh, order=2, dirichlet="outer", nograds=True, complex=True)
    u, v = fes.TnT()

    a = BilinearForm(fes)
    a += InnerProduct(curl(u), curl(v)) * dx
    a += 1j * InnerProduct(u, v) * dx
    a.Assemble()

    f = LinearForm(fes)
    f += CF((0, 0, 1)) * v * dx
    f.Assemble()

    print(f"\nComplex HCurl DOF={fes.ndof}")

    # SparseSolv BDDC
    pre = BDDCPreconditioner(a, fes)
    gfu = GridFunction(fes)
    inv = CGSolver(a.mat, pre, printrates=False, tol=1e-10, maxiter=500,
                   conjugate=False)
    gfu.vec.data = inv * f.vec
    print(f"SparseSolv BDDC: {inv.iterations} iterations")

    # NGSolve BDDC
    a2 = BilinearForm(fes)
    a2 += InnerProduct(curl(u), curl(v)) * dx
    a2 += 1j * InnerProduct(u, v) * dx
    c_ng = Preconditioner(a2, "bddc")
    a2.Assemble()

    gfu_ng = GridFunction(fes)
    inv_ng = CGSolver(a2.mat, c_ng.mat, printrates=False, tol=1e-10, maxiter=500,
                      conjugate=False)
    gfu_ng.vec.data = inv_ng * f.vec
    print(f"NGSolve BDDC:    {inv_ng.iterations} iterations")

    assert inv.iterations < 500, "SparseSolv BDDC did not converge"
    assert inv.iterations <= 2 * inv_ng.iterations + 5, \
        f"Too many iterations: {inv.iterations} vs NGSolve {inv_ng.iterations}"

    # Compare solutions via direct solver
    gfu_ref = GridFunction(fes)
    gfu_ref.vec.data = a.mat.Inverse(fes.FreeDofs(),
                                      inverse="sparsecholesky") * f.vec
    ref_norm = sqrt(abs(Integrate(InnerProduct(curl(gfu_ref), curl(gfu_ref)), mesh)))
    err = sqrt(abs(Integrate(InnerProduct(curl(gfu) - curl(gfu_ref),
                                           curl(gfu) - curl(gfu_ref)), mesh)))
    rel_err = err / ref_norm if ref_norm > 0 else err
    print(f"Relative error vs direct: {rel_err:.2e}")
    assert rel_err < 1e-6, f"Solution error too large: {rel_err}"


def test_bddc_vs_ngsolve_eddy_current():
    """Compare SparseSolv and NGSolve BDDC on eddy current with mesh refinement."""
    for maxh in [0.4, 0.3]:
        box = Box(Pnt(0, 0, 0), Pnt(1, 1, 1))
        for face in box.faces:
            face.name = "outer"
        mesh = box.GenerateMesh(maxh=maxh)

        fes = HCurl(mesh, order=2, dirichlet="outer", nograds=True, complex=True)
        u, v = fes.TnT()

        a = BilinearForm(fes)
        a += InnerProduct(curl(u), curl(v)) * dx
        a += 1j * InnerProduct(u, v) * dx
        a.Assemble()

        f = LinearForm(fes)
        f += CF((0, 0, 1)) * v * dx
        f.Assemble()

        # SparseSolv
        pre_ss = BDDCPreconditioner(a, fes)
        gfu_ss = GridFunction(fes)
        inv_ss = CGSolver(a.mat, pre_ss, printrates=False, tol=1e-10,
                          maxiter=500, conjugate=False)
        gfu_ss.vec.data = inv_ss * f.vec

        # NGSolve
        a2 = BilinearForm(fes)
        a2 += InnerProduct(curl(u), curl(v)) * dx
        a2 += 1j * InnerProduct(u, v) * dx
        c_ng = Preconditioner(a2, "bddc")
        a2.Assemble()
        gfu_ng = GridFunction(fes)
        inv_ng = CGSolver(a2.mat, c_ng.mat, printrates=False, tol=1e-10,
                          maxiter=500, conjugate=False)
        gfu_ng.vec.data = inv_ng * f.vec

        print(f"maxh={maxh}, DOF={fes.ndof}: SS={inv_ss.iterations} iters, "
              f"NG={inv_ng.iterations} iters")

        assert inv_ss.iterations < 500
        assert inv_ss.iterations <= 2 * inv_ng.iterations + 5



if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
