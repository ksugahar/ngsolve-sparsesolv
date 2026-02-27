"""
Mesh-size scaling benchmark for Hiruma eddy current problem.

Runs A-Phi BDDC across all mesh sizes.
Optionally runs two-step ICCG for smaller meshes.
"""

import sys
import time
import numpy as np
from netgen.read_gmsh import ReadGmsh
from ngsolve import *
from ngsolve.krylovspace import CGSolver

mu0 = 4e-7 * np.pi
freq = 50.0
s = 1j * 2 * np.pi * freq
order = 1

meshes = [
    "mesh1_2.5T",
    "mesh1_3.5T",
    "mesh1_4.5T",
    "mesh1_5.5T",
    "mesh1_20.5T",
    "mesh1_21.5T_HF",
]

results = []

for mesh_name in meshes:
    print(f"\n{'='*60}")
    print(f"Mesh: {mesh_name}")
    print(f"{'='*60}")
    sys.stdout.flush()

    try:
        m = ReadGmsh(mesh_name)
    except Exception as e:
        print(f"  SKIP (cannot load: {e})")
        continue

    mesh = Mesh(m)
    print(f"  nv={mesh.nv:,}  ne={mesh.ne:,}")

    nu_cf = mesh.MaterialCF({
        "core": 1 / (1000 * mu0), "cond": 1 / mu0, "air": 1 / mu0,
    })
    sigma_cf = mesh.MaterialCF({"cond": 5.96e7}, default=0)

    # --- A-Phi BDDC ---
    fesA = HCurl(mesh, order=order, nograds=True,
                 dirichlet="dirichlet", complex=True)
    fesPhi = H1(mesh, order=order, definedon="cond",
                dirichlet="gamma_in|gamma_out", complex=True)
    fes = fesA * fesPhi
    (A, phi), (N, psi) = fes.TnT()
    ndof = fes.ndof
    print(f"  DOFs: {ndof:,}  (HCurl {fesA.ndof:,} + H1 {fesPhi.ndof:,})")
    sys.stdout.flush()

    gf = GridFunction(fes)
    _, gfPhi = gf.components
    gfPhi.Set(1, definedon=mesh.Boundaries("gamma_in"))

    a = BilinearForm(fes)
    a += nu_cf * curl(A) * curl(N) * dx
    a += 1e-6 * nu_cf * A * N * dx
    a += s * sigma_cf * (A + grad(phi)) * (N + grad(psi)) * dx("cond")

    c = Preconditioner(a, "bddc")

    try:
        t0 = time.perf_counter()
        a.Assemble()
        c.Update()
        t_setup = time.perf_counter() - t0
    except Exception as e:
        print(f"  BDDC assembly FAILED: {e}")
        results.append({"mesh": mesh_name, "nv": mesh.nv, "ne": mesh.ne,
                        "ndof": ndof, "bddc_iters": None})
        continue

    rhs = gf.vec.CreateVector()
    rhs.data = -a.mat * gf.vec

    t0 = time.perf_counter()
    inv = CGSolver(a.mat, c.mat, maxiter=2000, tol=1e-8,
                   printrates=False, conjugate=False)
    gf.vec.data += inv * rhs
    t_solve = time.perf_counter() - t0

    gfA_sol, gfPhi_sol = gf.components
    B = curl(gfA_sol)
    B2 = Integrate(B * Conj(B), mesh).real

    print(f"  BDDC: {inv.iterations} iters, "
          f"setup {t_setup:.1f}s, solve {t_solve:.1f}s, ||B||^2={B2:.6e}")

    # VTK output
    B_abs = sqrt(B.real * B.real + B.imag * B.imag)
    J_cond = s * sigma_cf * (gfA_sol + grad(gfPhi_sol))
    mat_id = mesh.MaterialCF({mat: i + 1 for i, mat in enumerate(mesh.GetMaterials())})

    vtk = VTKOutput(mesh,
                    coefs=[B.real, B.imag, B_abs, J_cond.real, J_cond.imag, mat_id],
                    names=["B_re", "B_im", "B_abs", "J_re", "J_im", "MaterialID"],
                    filename=f"eddy_current_{mesh_name}",
                    subdivision=0, legacy=False)
    vtk.Do()
    print(f"  VTK: eddy_current_{mesh_name}.vtu")

    results.append({
        "mesh": mesh_name, "nv": mesh.nv, "ne": mesh.ne, "ndof": ndof,
        "bddc_iters": inv.iterations,
        "bddc_setup": t_setup, "bddc_solve": t_solve, "B2": B2,
    })
    sys.stdout.flush()

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*72}")
print(f"{'Mesh':<20} {'nv':>8} {'ne':>9} {'ndof':>9} "
      f"{'iters':>5} {'setup':>8} {'solve':>8} {'||B||^2':>12}")
print(f"{'-'*72}")
for r in results:
    if r.get("bddc_iters") is None:
        print(f"{r['mesh']:<20} {r['nv']:>8,} {r['ne']:>9,} {r['ndof']:>9,}  FAIL")
    else:
        print(f"{r['mesh']:<20} {r['nv']:>8,} {r['ne']:>9,} {r['ndof']:>9,} "
              f"{r['bddc_iters']:>5} {r['bddc_setup']:>7.1f}s {r['bddc_solve']:>7.1f}s "
              f"{r['B2']:>12.6e}")
print(f"{'='*72}")
