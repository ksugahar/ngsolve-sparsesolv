"""
Parallel scaling benchmark: SparseSolv BDDC vs ABMC+ICCG
on Hiruma eddy current problem (HCurl A-only, complex).

Demonstrates that BDDC with regularization has superior parallel
performance compared to ABMC+ICCG on curl-curl systems with
high-contrast materials (mur=1000).

Usage:
    python bench_parallel.py                  # default mesh
    python bench_parallel.py mesh1_3.5T       # specify mesh
"""

import sys
import os
import time
import numpy as np
from netgen.read_gmsh import ReadGmsh
from ngsolve import *
from ngsolve.krylovspace import CGSolver
import sparsesolv_ngsolve as ss

# ---------- parameters ----------
mu0 = 4e-7 * np.pi
freq = 50.0
s = 1j * 2 * np.pi * freq
order = 1
mesh_name = sys.argv[1] if len(sys.argv) > 1 else "mesh1_2.5T"
iccg_maxiter = 5000

# ---------- mesh ----------
m = ReadGmsh(mesh_name)
mesh = Mesh(m)
print(f"Mesh: {mesh_name}  nv={mesh.nv:,}  ne={mesh.ne:,}")

# ---------- material properties ----------
nu_cf = mesh.MaterialCF({
    "core": 1 / (1000 * mu0), "cond": 1 / mu0, "air": 1 / mu0,
})
sigma_cf = mesh.MaterialCF({"cond": 5.96e7}, default=0)

# ---------- DC phi solve (common preprocessing) ----------
fes_phi = H1(mesh, order=order, definedon="cond",
             dirichlet="gamma_in|gamma_out")
u_phi, v_phi = fes_phi.TnT()
a_phi = BilinearForm(fes_phi)
a_phi += sigma_cf * grad(u_phi) * grad(v_phi) * dx("cond")
a_phi.Assemble()
gf_phi = GridFunction(fes_phi)
gf_phi.Set(1, definedon=mesh.Boundaries("gamma_in"))
r_phi = gf_phi.vec.CreateVector()
r_phi.data = -a_phi.mat * gf_phi.vec
gf_phi.vec.data += a_phi.mat.Inverse(fes_phi.FreeDofs(),
                                      inverse="sparsecholesky") * r_phi

# ---------- HCurl space ----------
fesA = HCurl(mesh, order=order, nograds=True,
             dirichlet="dirichlet", complex=True)
u, v = fesA.TnT()
print(f"HCurl DOFs: {fesA.ndof:,}")

# Assemble RHS (common)
f = LinearForm(fesA)
f += (-s * sigma_cf) * grad(gf_phi) * v * dx("cond")
f.Assemble()


def run_bddc(use_taskmanager):
    """SparseSolv BDDC with regularization."""
    a_bddc = BilinearForm(fesA)
    a_bddc += nu_cf * curl(u) * curl(v) * dx
    a_bddc += 1e-6 * nu_cf * u * v * dx  # regularization
    a_bddc += s * sigma_cf * u * v * dx("cond")

    if use_taskmanager:
        with TaskManager(pajetrace=False):
            a_bddc.Assemble()
    else:
        a_bddc.Assemble()

    if use_taskmanager:
        with TaskManager(pajetrace=False):
            t0 = time.perf_counter()
            pre = ss.BDDCPreconditioner(a_bddc, fesA)
            t_setup = time.perf_counter() - t0

            gfu = GridFunction(fesA)
            t0 = time.perf_counter()
            inv = CGSolver(a_bddc.mat, pre, maxiter=200, tol=1e-8,
                           printrates=False, conjugate=False)
            gfu.vec.data = inv * f.vec
            t_solve = time.perf_counter() - t0
    else:
        t0 = time.perf_counter()
        pre = ss.BDDCPreconditioner(a_bddc, fesA)
        t_setup = time.perf_counter() - t0

        gfu = GridFunction(fesA)
        t0 = time.perf_counter()
        inv = CGSolver(a_bddc.mat, pre, maxiter=200, tol=1e-8,
                       printrates=False, conjugate=False)
        gfu.vec.data = inv * f.vec
        t_solve = time.perf_counter() - t0

    B2 = Integrate(curl(gfu) * Conj(curl(gfu)), mesh).real
    return t_setup, t_solve, inv.iterations, B2


def run_iccg(use_taskmanager):
    """SparseSolv ICCG with ABMC + auto_shift."""
    a_iccg = BilinearForm(fesA)
    a_iccg += nu_cf * curl(u) * curl(v) * dx
    a_iccg += s * sigma_cf * u * v * dx("cond")

    if use_taskmanager:
        with TaskManager(pajetrace=False):
            a_iccg.Assemble()
    else:
        a_iccg.Assemble()

    solver = ss.SparseSolvSolver(
        a_iccg.mat, method="ICCG",
        freedofs=fesA.FreeDofs(),
        tol=1e-6, maxiter=iccg_maxiter,
        save_best_result=True,
        use_abmc=True,
        abmc_block_size=2, abmc_num_colors=4,
        abmc_reorder_spmv=True)
    solver.auto_shift = True

    gfu = GridFunction(fesA)
    if use_taskmanager:
        with TaskManager(pajetrace=False):
            t0 = time.perf_counter()
            gfu.vec.data = solver * f.vec
            t_total = time.perf_counter() - t0
    else:
        t0 = time.perf_counter()
        gfu.vec.data = solver * f.vec
        t_total = time.perf_counter() - t0

    result = solver.last_result
    B2 = Integrate(curl(gfu) * Conj(curl(gfu)), mesh).real
    return t_total, result.iterations, result.converged, B2


# ============================================================
# Run benchmarks
# ============================================================
print(f"\nOMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'not set')}")
print(f"CPU count: {os.cpu_count()}")
print()

# --- Serial (1 thread) ---
print("--- Serial (1 thread) ---")
sys.stdout.flush()

t_bddc_setup_1, t_bddc_solve_1, bddc_iters, bddc_B2 = run_bddc(False)
print(f"  BDDC:  setup={t_bddc_setup_1*1000:.0f}ms  "
      f"solve={t_bddc_solve_1*1000:.0f}ms  "
      f"total={(t_bddc_setup_1+t_bddc_solve_1)*1000:.0f}ms  "
      f"iters={bddc_iters}  ||B||^2={bddc_B2:.6e}")
sys.stdout.flush()

t_iccg_1, iccg_iters, iccg_conv, iccg_B2 = run_iccg(False)
print(f"  ICCG:  total={t_iccg_1*1000:.0f}ms  "
      f"iters={iccg_iters}  conv={iccg_conv}  ||B||^2={iccg_B2:.6e}")
sys.stdout.flush()

# --- Parallel (N threads) ---
print(f"\n--- Parallel ({os.cpu_count()} threads) ---")
sys.stdout.flush()

# Warmup
run_bddc(True)
run_iccg(True)

t_bddc_setup_N, t_bddc_solve_N, bddc_iters_N, bddc_B2_N = run_bddc(True)
print(f"  BDDC:  setup={t_bddc_setup_N*1000:.0f}ms  "
      f"solve={t_bddc_solve_N*1000:.0f}ms  "
      f"total={(t_bddc_setup_N+t_bddc_solve_N)*1000:.0f}ms  "
      f"iters={bddc_iters_N}  ||B||^2={bddc_B2_N:.6e}")
sys.stdout.flush()

t_iccg_N, iccg_iters_N, iccg_conv_N, iccg_B2_N = run_iccg(True)
print(f"  ICCG:  total={t_iccg_N*1000:.0f}ms  "
      f"iters={iccg_iters_N}  conv={iccg_conv_N}  ||B||^2={iccg_B2_N:.6e}")
sys.stdout.flush()

# ============================================================
# Summary
# ============================================================
bddc_total_1 = t_bddc_setup_1 + t_bddc_solve_1
bddc_total_N = t_bddc_setup_N + t_bddc_solve_N

print(f"\n{'='*70}")
print(f"{'Method':<16} {'1T [ms]':>10} {'NT [ms]':>10} {'Speedup':>10} "
      f"{'Iters':>6} {'||B||^2':>12}")
print(f"{'-'*70}")
print(f"{'BDDC (reg)':<16} {bddc_total_1*1000:>10.0f} {bddc_total_N*1000:>10.0f} "
      f"{bddc_total_1/bddc_total_N:>10.2f}x "
      f"{bddc_iters:>6} {bddc_B2:>12.6e}")
print(f"{'ABMC+ICCG':<16} {t_iccg_1*1000:>10.0f} {t_iccg_N*1000:>10.0f} "
      f"{t_iccg_1/t_iccg_N:>10.2f}x "
      f"{iccg_iters:>6} {iccg_B2:>12.6e}")
print(f"{'='*70}")
print(f"\nBDDC/ICCG time ratio (serial):   {bddc_total_1/t_iccg_1:.2f}x")
print(f"BDDC/ICCG time ratio (parallel): {bddc_total_N/t_iccg_N:.2f}x")
if not iccg_conv:
    print(f"\nNote: ICCG did not converge in {iccg_maxiter} iterations.")
    print(f"  ||B||^2 rel diff: {abs(bddc_B2-iccg_B2)/bddc_B2:.2e} "
          f"(best iterate)")
