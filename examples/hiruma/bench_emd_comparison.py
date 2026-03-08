"""Benchmark: HYPRE AMS + BiCGStab vs EMD preconditioner (Hiruma, SA-26-001).

Reproduces the eddy current benchmark from:
  "A study on iterative solutions using EMD preconditioning
   for the full Maxwell equations and Darwin model"
  Shingo Hiruma, IEE Japan SA-26-001/RM-26-001, 2026.3.5

Problem: 21-turn coil model, eddy current formulation
  - sigma = 5.96e7 S/m (copper conductor)
  - mu_r = 1000 (iron core)
  - f = 30 kHz (omega = 2*pi*30000)

EMD paper results (Table 1, eddy current model, 3,670,328 DOFs):
  IC only:                15,838 iter, 5964.8 s
  EMD (IC + AMG V-cycle):  4,069 iter, 1716.8 s  (3.47x)
  EMD (IC + AMG W-cycle):  2,935 iter, 1552.9 s  (3.84x)
  EMD (IC + GenEO-DDM 24):  1,004 iter,  550.8 s (10.83x)

BiCGStab advantage over GMRES:
  - Fixed 8 work vectors (vs GMRES: 1 vector per iteration)
  - No Gram-Schmidt orthogonalization cost
  - O(N) memory regardless of iteration count

Usage:
    python bench_emd_comparison.py [mesh_name]
    python bench_emd_comparison.py mesh1_2.5T          # quick test
    python bench_emd_comparison.py mesh1_20.5T         # full comparison
"""

import json
import os
import platform
import sys
import time
from datetime import datetime

import numpy as np

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from netgen.read_gmsh import ReadGmsh
from ngsolve import *
import sparsesolv_ngsolve as ssn
from bicgstab_solver import BiCGStabSolver

# Physical parameters (match Hiruma SA-26-001)
mu0 = 4e-7 * np.pi
freq = 30e3                         # 30 kHz (induction heating range)
omega = 2 * np.pi * freq
eps = 1e-6                          # HCurl stabilization
sigma_cu = 5.96e7                   # copper conductivity [S/m]
mu_r_core = 1000                    # iron core relative permeability

# Solver parameters
maxiter = 2000
tol = 1e-10                         # match Hiruma: relative residual 1e-10


def get_peak_memory_mb():
    if HAS_PSUTIL:
        mem = psutil.Process(os.getpid()).memory_info()
        return getattr(mem, 'peak_wset', mem.rss) / (1024 * 1024)
    return 0.0


def setup_problem(mesh_name):
    """Setup eddy current problem matching Hiruma's formulation."""
    mesh_file = os.path.join(os.path.dirname(__file__), mesh_name)
    print(f"Loading mesh: {mesh_file}", flush=True)
    t0 = time.perf_counter()
    mesh = Mesh(ReadGmsh(mesh_file))
    t_mesh = time.perf_counter() - t0
    print(f"  Mesh loaded in {t_mesh:.1f}s  "
          f"(nv={mesh.nv:,}, ne={mesh.ne:,})", flush=True)

    # Material coefficients
    nu_cf = 1.0 / (mu0 * IfPos(mesh.MaterialCF({"core": 1}), mu_r_core, 1.0))
    sigma_cf = mesh.MaterialCF({"cond": sigma_cu}, default=0)

    # Complex HCurl space (A-formulation, eddy current)
    fes = HCurl(mesh, order=1, nograds=True,
                dirichlet="dirichlet", complex=True)
    u, v = fes.TnT()

    print(f"  HCurl DOFs: {fes.ndof:,}", flush=True)

    # Bilinear form: curl-curl + stabilization + eddy current
    t0 = time.perf_counter()
    a = BilinearForm(fes)
    a += nu_cf * curl(u) * curl(v) * dx
    a += eps * nu_cf * u * v * dx
    a += 1j * omega * sigma_cf * u * v * dx("cond")
    a.Assemble()
    t_asm = time.perf_counter() - t0
    print(f"  Complex matrix assembled in {t_asm:.1f}s", flush=True)

    # RHS: uniform current source in conductor
    f = LinearForm(fes)
    f += nu_cf * CF((0, 0, 1)) * v * dx("cond")
    f.Assemble()

    # Real auxiliary matrix for AMS preconditioner
    fes_real = HCurl(mesh, order=1, nograds=True,
                     dirichlet="dirichlet", complex=False)
    u_r, v_r = fes_real.TnT()
    a_real = BilinearForm(fes_real)
    a_real += nu_cf * curl(u_r) * curl(v_r) * dx
    a_real += eps * nu_cf * u_r * v_r * dx
    a_real += abs(omega) * sigma_cf * u_r * v_r * dx("cond")
    a_real.Assemble()
    print(f"  Real auxiliary matrix assembled", flush=True)

    # Discrete gradient for AMS
    G_mat, h1_fes = fes_real.CreateGradient()
    print(f"  H1 DOFs: {h1_fes.ndof:,}", flush=True)

    # Vertex coordinates for HYPRE AMS
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
        "t_assembly": t_asm,
    }


def run_bicgstab(p, cycle_type=1, print_level=0):
    """Run HYPRE AMS + BiCGStab solver."""
    label = f"HYPRE AMS (cycle={cycle_type}) + BiCGStab"
    print(f"\n  Running: {label}", flush=True)

    # Setup HYPRE AMS preconditioner
    t0 = time.perf_counter()
    pre = ssn.ComplexHypreAMSPreconditioner(
        a_real_mat=p["a_real"].mat,
        grad_mat=p["G_mat"],
        freedofs=p["fes_real"].FreeDofs(),
        coord_x=p["coord_x"],
        coord_y=p["coord_y"],
        coord_z=p["coord_z"],
        ndof_complex=p["fes"].ndof,
        cycle_type=cycle_type,
        print_level=print_level)
    t_setup = time.perf_counter() - t0

    # Solve with BiCGStab
    gfu = GridFunction(p["fes"])
    with TaskManager():
        t0 = time.perf_counter()
        inv = BiCGStabSolver(mat=p["a"].mat, pre=pre,
                             maxiter=maxiter, tol=tol, printrates=False)
        gfu.vec.data = inv * p["f"].vec
        t_solve = time.perf_counter() - t0
    mem_after = get_peak_memory_mb()

    iters = inv.iterations
    converged = iters < maxiter
    ms_it = (t_solve / iters * 1000) if iters > 0 else 0
    status = "CONVERGED" if converged else "FAILED"

    print(f"    {iters:5d} iterations, {t_solve:.2f}s solve, "
          f"{ms_it:.1f}ms/iter [{status}]", flush=True)
    print(f"    Setup: {t_setup:.2f}s", flush=True)

    return {
        "method": label,
        "iterations": iters,
        "t_setup": round(t_setup, 3),
        "t_solve": round(t_solve, 3),
        "t_total": round(t_setup + t_solve, 3),
        "ms_per_iter": round(ms_it, 1),
        "converged": converged,
        "peak_memory_mb": round(mem_after, 1),
        "cycle_type": cycle_type,
    }


def run_iccg(p):
    """Run ICCG solver (IC preconditioner) for baseline comparison."""
    label = "ICCG (IC preconditioner)"
    print(f"\n  Running: {label}", flush=True)

    t0 = time.perf_counter()
    solver = ssn.SparseSolvSolver(
        p["a_real"].mat, "CG", p["fes_real"].FreeDofs(),
        tol=tol, maxiter=maxiter, shift=1.05,
        printrates=False)
    t_setup = time.perf_counter() - t0

    # For complex system, we need Re/Im wrapper
    class ICComplexWrapper(BaseMatrix):
        def __init__(self, solver, ndof):
            super().__init__()
            self._is_complex = True
            self._ndof = ndof
            self._solver = solver
            self._re_in = solver.CreateColVector()
            self._im_in = solver.CreateColVector()
            self._re_out = solver.CreateColVector()
            self._im_out = solver.CreateColVector()

        def Height(self): return self._ndof
        def Width(self): return self._ndof
        def IsComplex(self): return True

        def Mult(self, x, y):
            x_np = x.FV().NumPy()
            self._re_in.FV().NumPy()[:] = x_np.real
            self._im_in.FV().NumPy()[:] = x_np.imag
            self._re_out.data = self._solver * self._re_in
            self._im_out.data = self._solver * self._im_in
            y.FV().NumPy()[:] = (self._re_out.FV().NumPy()
                                 + 1j * self._im_out.FV().NumPy())

        def MultTrans(self, x, y):
            self.Mult(x, y)

    pre = ICComplexWrapper(solver, p["fes"].ndof)

    gfu = GridFunction(p["fes"])
    with TaskManager():
        t0 = time.perf_counter()
        inv = BiCGStabSolver(mat=p["a"].mat, pre=pre,
                             maxiter=maxiter, tol=tol, printrates=False)
        gfu.vec.data = inv * p["f"].vec
        t_solve = time.perf_counter() - t0

    iters = inv.iterations
    converged = iters < maxiter
    ms_it = (t_solve / iters * 1000) if iters > 0 else 0
    status = "CONVERGED" if converged else f"FAILED ({iters} iters)"

    print(f"    {iters:5d} iterations, {t_solve:.2f}s solve, "
          f"{ms_it:.1f}ms/iter [{status}]", flush=True)

    return {
        "method": label,
        "iterations": iters,
        "t_setup": round(t_setup, 3),
        "t_solve": round(t_solve, 3),
        "t_total": round(t_setup + t_solve, 3),
        "ms_per_iter": round(ms_it, 1),
        "converged": converged,
    }


def print_comparison_table(results, ndof):
    """Print results comparison with EMD paper data."""
    # EMD paper results (Table 1, eddy current model, 3,670,328 DOFs)
    emd_results = [
        {"method": "EMD: IC only", "iterations": 15838,
         "t_solve": 5964.8, "speedup": 1.0},
        {"method": "EMD: IC + AMG V-cycle", "iterations": 4069,
         "t_solve": 1716.8, "speedup": 3.47},
        {"method": "EMD: IC + AMG W-cycle", "iterations": 2935,
         "t_solve": 1552.9, "speedup": 3.84},
        {"method": "EMD: IC + GenEO-DDM (24)", "iterations": 1004,
         "t_solve": 550.8, "speedup": 10.83},
    ]

    print(f"\n{'='*90}")
    print(f"Comparison with EMD paper (SA-26-001, 3,670,328 DOFs, 30kHz)")
    print(f"Current benchmark: {ndof:,} DOFs")
    print(f"{'='*90}")
    print(f"{'Method':<45s} {'Iters':>7s} {'Time[s]':>9s} {'ms/it':>8s}")
    print(f"{'-'*90}")

    # EMD results (from paper, only if same DOF count)
    if abs(ndof - 3670328) < 100000:
        print("--- EMD paper (Hiruma, SA-26-001) ---")
        for r in emd_results:
            ms_it = r["t_solve"] / r["iterations"] * 1000
            print(f"  {r['method']:<43s} {r['iterations']:>7d} "
                  f"{r['t_solve']:>9.1f} {ms_it:>8.1f}")
        print()

    # Our results
    print("--- This benchmark (HYPRE AMS + BiCGStab) ---")
    for r in results:
        print(f"  {r['method']:<43s} {r['iterations']:>7d} "
              f"{r['t_solve']:>9.1f} {r['ms_per_iter']:>8.1f}")
    print(f"{'='*90}")


def main():
    mesh_name = sys.argv[1] if len(sys.argv) > 1 else "mesh1_2.5T"

    print(f"\n{'='*70}")
    print(f"EMD Comparison Benchmark")
    print(f"Problem: Eddy current, f={freq/1e3:.0f} kHz, "
          f"sigma={sigma_cu:.2e} S/m, mu_r={mu_r_core}")
    print(f"Solver: HYPRE AMS + BiCGStab, tol={tol}, maxiter={maxiter}")
    print(f"{'='*70}")

    p = setup_problem(mesh_name)
    ndof = p["fes"].ndof
    results = []

    # Run BiCGStab + HYPRE AMS
    for ct in [1]:
        r = run_bicgstab(p, cycle_type=ct)
        results.append(r)

    print_comparison_table(results, ndof)

    # Save results
    out = {
        "timestamp": datetime.now().isoformat(),
        "hostname": platform.node(),
        "benchmark": "bench_emd_comparison",
        "problem": {
            "mesh": mesh_name,
            "ndof_hcurl": ndof,
            "ndof_h1": p["h1_fes"].ndof,
            "frequency_hz": freq,
            "omega": omega,
            "sigma": sigma_cu,
            "mu_r_core": mu_r_core,
            "tol": tol,
            "maxiter": maxiter,
        },
        "emd_paper": {
            "title": "SA-26-001 (Hiruma, 2026.3.5)",
            "ndof": 3670328,
            "best_method": "EMD (IC + GenEO-DDM, 24 domains)",
            "best_iterations": 1004,
            "best_time_s": 550.8,
        },
        "results": results,
    }

    json_path = os.path.join(os.path.dirname(__file__),
                             "results_emd_comparison.json")
    with open(json_path, "w") as fp:
        json.dump(out, fp, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
