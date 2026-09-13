"""Proof of concept: N=4 two-mode closure in HOMOGENEOUS coordinates.

Instead of feeding the ratios U_j/U_0 (singular at every node of a standing
wave) to a closure, fit the full moment vector to two eigenmodes,

    g_1 A(zeta_1) + g_2 A(zeta_2) = (U_0, U_1, U_2, U_3),   A = (1, a_1, a_2, a_3),

which is 4 complex equations for 4 complex unknowns and never divides by
U_0, then close with U_4 = g_1 a_4(zeta_1) + g_2 a_4(zeta_2).  Variable
projection: for given zetas the g's are a linear least-squares solve; the
zetas are found by Levenberg-Marquardt warm-started from the previous RHS
evaluation.  This is the map a node-regular N=4 NN would have to amortize
(cf. bot/closures/dressed_pole_r2.py for the slab-ITG version).

Runs the k lambda_D = 0.4 standing wave (density IC, continuum included) and
compares with the kinetic solution and the ratio-based NN (N=4).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

import bot.figs.fig_standing_wave as F
from bot.closures.inference import alpha as a3, alpha4 as a4
from bot.closures.u2 import beta as a2

om = F.landau_root()
zL = om / F.K


def basis(z):
    return np.array([1.0, z, a2(z), a3(z)], dtype=complex)


class TwoModeHomog:
    """Callable closure U -> U_4 with a warm-started variable-projection solve."""

    def __init__(self, z0):
        self.z = np.array(z0, dtype=complex)
        self.n_calls = 0
        self.max_resid = 0.0
        self.trace = []

    # pole search box: generous around the training rectangle, but keeps
    # Z'(zeta) finite (it overflows like exp(Im zeta^2) far below the axis)
    LO = np.array([-6.0, -2.5, -6.0, -2.5])
    HI = np.array([6.0, 2.5, 6.0, 2.5])

    def _resid(self, x, U):
        z1, z2 = x[0] + 1j * x[1], x[2] + 1j * x[3]
        with np.errstate(all="ignore"):
            A = np.column_stack([basis(z1), basis(z2)])
        if not np.all(np.isfinite(A)):
            return np.full(8, 1e3)
        g, *_ = np.linalg.lstsq(A, U, rcond=None)
        r = U - A @ g
        return np.concatenate([r.real, r.imag]) / (np.linalg.norm(U) + 1e-300)

    def __call__(self, U):
        x0 = np.array([self.z[0].real, self.z[0].imag, self.z[1].real, self.z[1].imag])
        x0 = np.clip(x0, self.LO + 1e-6, self.HI - 1e-6)
        sol = least_squares(self._resid, x0, args=(U,), method="trf",
                            bounds=(self.LO, self.HI),
                            xtol=1e-12, ftol=1e-12, max_nfev=200)
        z1, z2 = sol.x[0] + 1j * sol.x[1], sol.x[2] + 1j * sol.x[3]
        self.z = np.array([z1, z2])
        A = np.column_stack([basis(z1), basis(z2)])
        g, *_ = np.linalg.lstsq(A, U, rcond=None)
        self.n_calls += 1
        self.max_resid = max(self.max_resid, np.linalg.norm(sol.fun))
        return g[0] * a4(z1) + g[1] * a4(z2)


def run(z0):
    cl = TwoModeHomog(z0)
    U0 = F.evolve(4, cl, rtol=1e-8, atol=1e-12)
    return U0, cl


if __name__ == "__main__":
    print(f"k lambda_D = {F.K_LAMBDA_D}: Landau root omega = {om.real:.4f} {om.imag:+.4f}i, "
          f"zeta_L = {zL:.4f}")
    U_kin = F.kinetic()

    for name, z0 in [("warm start near the root pair (2-0.1i, -2-0.1i)", (2.0 - 0.1j, -2.0 - 0.1j)),
                     ("generic warm start (1.5+0.2i, -0.5-0.5i)", (1.5 + 0.2j, -0.5 - 0.5j))]:
        U0, cl = run(z0)
        w, g = F.fit_mode(U0)
        i5 = np.searchsorted(F.T, 5.0)
        late = slice(i5, None)
        err_late = np.max(np.abs(U0[late] - U_kin[late])) / F.EPS
        print(f"\n{name}")
        print(f"  omega {w:.4f} ({100*(w/om.real-1):+.2f}%)   gamma {g:+.4f} ({100*(g/om.imag-1):+.1f}%)")
        print(f"  max |U0 - kinetic|/eps for t>5: {err_late:.2e}   (RHS calls {cl.n_calls}, "
              f"max fit residual {cl.max_resid:.1e})")
        print(f"  final fitted zetas: {cl.z[0]:.4f}, {cl.z[1]:.4f}   (exact: {zL:.4f}, {-np.conj(zL):.4f})")

    # reference: the ratio-based NN N=4 for the same problem (from fig_standing_wave)
    from bot.closures.n4_nn import load_super
    U_nn = F.evolve(4, F.nn4_closure(load_super()))
    w, g = F.fit_mode(U_nn)
    print(f"\nratio-based NN (N=4), same IC: omega {w:.4f} ({100*(w/om.real-1):+.1f}%)  "
          f"gamma {g:+.4f} ({100*(g/om.imag-1):+.1f}%)")
