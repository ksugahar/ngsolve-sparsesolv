"""Benchmark: HYPRE AMS + GMRES for complex eddy current problems.

Compares:
  1. HYPRE AMS (real) + Python Re/Im wrapper + GMRES (sequential)
  2. ComplexHypreAMSPreconditioner + GMRES (TaskManager parallel Re/Im)

Usage:
    python bench_hypre_ams.py [mesh_name ...]
"""

import json
import os
import sys
import time
from datetime import datetime

import numpy as np
from netgen.read_gmsh import ReadGmsh
from ngsolve import *
from ngsolve.krylovspace import GMResSolver
import sparsesolv_ngsolve as ssn

mu0 = 4e-7 * np.pi
omega = 2 * np.pi * 50.0
eps = 1e-6
maxiter = 500
tol = 1e-8


def setup_problem(mesh_name):
    mesh_file = os.path.join(os.path.dirname(__file__), mesh_name)
    mesh = Mesh(ReadGmsh(mesh_file))

    nu_cf = 1.0 / (mu0 * IfPos(mesh.MaterialCF({"core": 1}), 1000, 1.0))
    sigma_cf = mesh.MaterialCF({"cond": 5.96e7}, default=0)

    # Complex HCurl
    fes = HCurl(mesh, order=1, nograds=True,
                dirichlet="dirichlet", complex=True)
    u, v = fes.TnT()
    a = BilinearForm(fes)
    a += nu_cf * curl(u) * curl(v) * dx
    a += eps * nu_cf * u * v * dx
    a += 1j * omega * sigma_cf * u * v * dx("cond")
    a.Assemble()

    f = LinearForm(fes)
    f += nu_cf * CF((0, 0, 1)) * v * dx("cond")
    f.Assemble()

    # Real auxiliary for AMS
    fes_real = HCurl(mesh, order=1, nograds=True,
                     dirichlet="dirichlet", complex=False)
    u_r, v_r = fes_real.TnT()
    a_real = BilinearForm(fes_real)
    a_real += nu_cf * curl(u_r) * curl(v_r) * dx
    a_real += eps * nu_cf * u_r * v_r * dx
    a_real += abs(omega) * sigma_cf * u_r * v_r * dx("cond")
    a_real.Assemble()

    G_mat, h1_fes = fes_real.CreateGradient()

    # HYPRE AMS coordinates
    nv = mesh.nv
    coord_x = [0.0] * nv
    coord_y = [0.0] * nv
    coord_z = [0.0] * nv
    for i in range(nv):
        p = mesh.ngmesh.Points()[i + 1]
        coord_x[i] = p[0]
        coord_y[i] = p[1]
        coord_z[i] = p[2]

    return {
        "mesh": mesh, "fes": fes, "a": a, "f": f,
        "a_real": a_real, "fes_real": fes_real,
        "G_mat": G_mat, "h1_fes": h1_fes,
        "coord_x": coord_x, "coord_y": coord_y, "coord_z": coord_z,
    }


def run_solver(p, label, pre, maxiter=500, tol=1e-8):
    gfu = GridFunction(p["fes"])
    with TaskManager():
        t0 = time.perf_counter()
        inv = GMResSolver(mat=p["a"].mat, pre=pre, maxiter=maxiter,
                          printrates=False, tol=tol)
        gfu.vec.data = inv * p["f"].vec
        t_solve = time.perf_counter() - t0

    iters = inv.iterations
    ms_it = (t_solve / iters * 1000) if iters > 0 else 0
    status = "OK" if iters < maxiter else "FAIL"
    print(f"  {label:55s}  {iters:4d} it  {t_solve:8.2f}s  "
          f"{ms_it:7.1f}ms/it [{status}]", flush=True)
    return {
        "label": label,
        "iterations": iters,
        "t_solve": round(t_solve, 4),
        "ms_per_iter": round(ms_it, 1),
        "converged": iters < maxiter,
    }


class PythonComplexWrapper(BaseMatrix):
    """Python-level Re/Im splitting (sequential HYPRE calls)."""
    def __init__(self, pre_real, ndof_complex):
        super().__init__()
        self._is_complex = True
        self._ndof = ndof_complex
        self._pre = pre_real
        self._re_in = pre_real.CreateColVector()
        self._im_in = pre_real.CreateColVector()
        self._re_out = pre_real.CreateColVector()
        self._im_out = pre_real.CreateColVector()

    def Height(self): return self._ndof
    def Width(self): return self._ndof
    def IsComplex(self): return True

    def Mult(self, x, y):
        x_np = x.FV().NumPy()
        self._re_in.FV().NumPy()[:] = x_np.real
        self._im_in.FV().NumPy()[:] = x_np.imag
        self._re_out.data = self._pre * self._re_in
        self._im_out.data = self._pre * self._im_in
        y.FV().NumPy()[:] = (self._re_out.FV().NumPy()
                             + 1j * self._im_out.FV().NumPy())

    def MultTrans(self, x, y):
        self.Mult(x, y)


def run_benchmark(mesh_name):
    p = setup_problem(mesh_name)
    ndof_hc = p["fes"].ndof
    ndof_h1 = p["h1_fes"].ndof
    print(f"\nMesh: {mesh_name}  DOFs: HCurl={ndof_hc:,}  H1={ndof_h1:,}",
          flush=True)

    results = []

    # 1. Python wrapper (sequential Re/Im)
    t0 = time.perf_counter()
    pre_real = ssn.HypreAMSPreconditioner(
        p["a_real"].mat, p["G_mat"], p["fes_real"].FreeDofs(),
        p["coord_x"], p["coord_y"], p["coord_z"],
        1, 0)
    t_setup1 = time.perf_counter() - t0

    pre_py = PythonComplexWrapper(pre_real, p["fes"].ndof)
    r = run_solver(p, "Python Re/Im wrapper (sequential) + GMRES", pre_py)
    r["t_setup"] = round(t_setup1, 4)
    results.append(r)

    # 2. C++ ComplexHypreAMSPreconditioner (TaskManager parallel Re/Im)
    t0 = time.perf_counter()
    pre_cpp = ssn.ComplexHypreAMSPreconditioner(
        a_real_mat=p["a_real"].mat,
        grad_mat=p["G_mat"],
        freedofs=p["fes_real"].FreeDofs(),
        coord_x=p["coord_x"],
        coord_y=p["coord_y"],
        coord_z=p["coord_z"],
        ndof_complex=p["fes"].ndof,
        cycle_type=1,
        print_level=0)
    t_setup2 = time.perf_counter() - t0

    r = run_solver(p, "C++ ComplexHypreAMS (TaskManager) + GMRES", pre_cpp)
    r["t_setup"] = round(t_setup2, 4)
    results.append(r)

    # Summary
    print(f"\n{'='*90}")
    print(f"{'Method':>60s} {'Iters':>6s} {'Time':>8s} {'ms/it':>8s}")
    print(f"{'='*90}")
    for r in results:
        print(f"  {r['label']:>60s} {r['iterations']:>5d} "
              f"{r['t_solve']:>7.2f}s {r['ms_per_iter']:>7.1f}")

    out = {
        "timestamp": datetime.now().isoformat(),
        "benchmark": "bench_hypre_ams",
        "mesh_name": mesh_name,
        "ndof_hcurl": ndof_hc,
        "ndof_h1": ndof_h1,
        "results": results,
    }
    json_path = os.path.join(os.path.dirname(__file__),
                             "results_hypre_ams.json")
    with open(json_path, "w") as fp:
        json.dump(out, fp, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    mesh_names = sys.argv[1:] if len(sys.argv) > 1 else ["mesh1_2.5T"]
    print(f"\n{'='*70}")
    print(f"HYPRE AMS Benchmark: Python wrapper vs C++ TaskManager")
    print(f"{'='*70}")
    for mesh_name in mesh_names:
        run_benchmark(mesh_name)
