"""Complex BiCGStab solver for NGSolve.

Implements preconditioned BiCGStab (Van der Vorst, 1992) using NGSolve's
BaseMatrix/BaseVector interface. Designed for complex eddy current problems
with HYPRE AMS preconditioner.

Key advantage over GMRES: fixed memory (8 work vectors) regardless of
iteration count. GMRES stores one vector per iteration, causing O(k*N)
memory growth that dominates cost at high iteration counts.

Reference implementation: Radia rad_relaxation_methods.cpp, line 2776
"""

import math
from ngsolve import Norm
from ngsolve.krylovspace import LinearSolver


class BiCGStabSolver(LinearSolver):
    """Preconditioned BiCGStab for complex non-symmetric systems.

    Usage:
        pre = ssn.ComplexHypreAMSPreconditioner(...)
        solver = BiCGStabSolver(mat=A, pre=pre, tol=1e-10, maxiter=2000)
        gfu.vec.data = solver * f.vec
    """

    name = "BiCGStab"

    def _SolveImpl(self, rhs, sol):
        A = self.mat
        M = self.pre  # preconditioner (None = identity)
        is_complex = rhs.is_complex

        # Allocate 8 work vectors (fixed, independent of iteration count)
        r = rhs.CreateVector()
        r0 = rhs.CreateVector()
        p = rhs.CreateVector()
        v = rhs.CreateVector()
        s = rhs.CreateVector()
        t = rhs.CreateVector()
        p_hat = rhs.CreateVector()
        s_hat = rhs.CreateVector()

        # Initialize: r = b - A*x0
        r.data = rhs - A * sol
        r0.data = r

        rho = 1.0
        alpha = 1.0
        omega = 1.0
        p[:] = 0.0
        v[:] = 0.0

        rhs_norm = Norm(rhs)
        if rhs_norm < 1e-30:
            rhs_norm = 1.0

        r_norm = Norm(r)
        self.CheckResidual(r_norm)

        for k in range(1, self.maxiter + 1):
            rho_old = rho
            rho = r0.InnerProduct(r, conjugate=is_complex)

            if abs(rho) < 1e-30:
                if self.printrates:
                    print(f"BiCGStab breakdown: rho=0 at iter {k}")
                break

            if k == 1:
                p.data = r
            else:
                if abs(rho_old * omega) < 1e-30:
                    if self.printrates:
                        print(f"BiCGStab breakdown: rho_old*omega=0 at iter {k}")
                    break
                beta = (rho / rho_old) * (alpha / omega)
                # p = r + beta * (p - omega * v)
                p.data -= omega * v
                p.data *= beta
                p.data += r

            # Precondition: p_hat = M^{-1} p
            if M is not None:
                p_hat.data = M * p
            else:
                p_hat.data = p

            # v = A * p_hat
            v.data = A * p_hat

            r0_dot_v = r0.InnerProduct(v, conjugate=is_complex)
            if abs(r0_dot_v) < 1e-30:
                if self.printrates:
                    print(f"BiCGStab breakdown: r0_dot_v=0 at iter {k}")
                break
            alpha = rho / r0_dot_v

            # s = r - alpha * v
            s.data = r - alpha * v

            s_norm = Norm(s)
            if s_norm / rhs_norm < self.tol:
                sol.data += alpha * p_hat
                self.CheckResidual(s_norm)
                break

            # Precondition: s_hat = M^{-1} s
            if M is not None:
                s_hat.data = M * s
            else:
                s_hat.data = s

            # t = A * s_hat
            t.data = A * s_hat

            t_dot_s = t.InnerProduct(s, conjugate=is_complex)
            t_dot_t = t.InnerProduct(t, conjugate=is_complex)
            if abs(t_dot_t) < 1e-30:
                sol.data += alpha * p_hat
                self.CheckResidual(s_norm)
                break
            omega = t_dot_s / t_dot_t

            # Update solution: x += alpha * p_hat + omega * s_hat
            sol.data += alpha * p_hat + omega * s_hat

            # Update residual: r = s - omega * t
            r.data = s - omega * t

            r_norm = Norm(r)
            if self.CheckResidual(r_norm):
                break

            # Detect numerical blowup
            if math.isnan(r_norm) or math.isinf(r_norm) or r_norm > 1e15 * rhs_norm:
                if self.printrates:
                    print(f"BiCGStab: numerical blowup at iter {k}")
                break
