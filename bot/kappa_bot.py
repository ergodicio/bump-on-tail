"""Bump-on-tail with a kappa-distributed beam: kinetic reference and closures.

Demonstrates the equilibrium-generality of the EKR closure.  The N = 2 fluid
system is structurally IDENTICAL to the Maxwellian case -- the moment
hierarchy only needs M_0 = 1 and M_1 = 0 to close at the temperature moment
-- so switching equilibrium changes only the closure function,
U_2/U_0 = beta_kappa(zeta) = zeta^2 + 1/(2 R_kappa(zeta)).

The Hammett-Perkins closure, by contrast, cannot represent the change at all:
its coefficients are fixed by the asymptotics of the Maxwellian Z', and the
fluid matrix it closes sees the equilibrium only through M_0, M_1 (both of
which are unchanged by our normalization).  Its predicted growth rate for a
kappa beam is therefore identical to its Maxwellian prediction, while the
kinetic answer moves.

Dispersion relation (derived from the linearized beam Vlasov equation plus
the modified-Langmuir wave equation, with zeta = omega/k - u_b):

    omega^2 [ 1 + 2 eps R(zeta) / k^2 ] = 1

which reproduces bot.kinetic for R = -Z'/2.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root as _scipy_root

from bot.closures.kappa import F_kappa, R_kappa, beta_kappa, dF_kappa
from bot.closed_loop import _safe_xi_ratio


def s_grid(N_v: int, V: float):
    s = np.linspace(-V, V, N_v)
    return s, s[1] - s[0]


# ----- kinetic reference ---------------------------------------------------

def kinetic_system_kappa(k: float, u_b: float, eps: float, kappa: int,
                         N_v: int = 400, V: float = 20.0) -> np.ndarray:
    """(u, E, F[s_0..s_{N_v-1}]) matrix for a kappa-distributed beam.

    Identical to bot.kinetic.kinetic_system_sspace except the velocity-space
    source is -dF_kappa/ds rather than the Maxwellian 2s exp(-s^2)/sqrt(pi).
    A wider, finer grid is used because the kappa tails decay as a power law.
    """
    s, ds = s_grid(N_v, V)
    dim = 2 + N_v
    A = np.zeros((dim, dim), dtype=complex)
    A[0, 1] = -1.0
    A[1, 0] = +1.0
    A[1, 2:] = -eps * (u_b + s) * ds
    src = -dF_kappa(s, kappa)
    for j in range(N_v):
        A[2 + j, 2 + j] = -1j * k * (u_b + s[j])
        A[2 + j, 1] = src[j]
    return A


def kinetic_evolve_kappa(k, u_b, eps, kappa, t_grid, y0,
                         N_v: int = 400, V: float = 20.0) -> np.ndarray:
    A = kinetic_system_kappa(k, u_b, eps, kappa, N_v, V)
    lam, V_ = np.linalg.eig(A)
    c = np.linalg.solve(V_, y0)
    return np.array([(V_ @ (c * np.exp(lam * t)))[1] for t in t_grid])


def kinetic_ic_kappa(k, u_b, eps, kappa, omega, amp=1e-3,
                     N_v: int = 400, V: float = 20.0) -> np.ndarray:
    """Exact single-mode IC: F(s) = -i E src(s) / (k (s - zeta))."""
    s, _ds = s_grid(N_v, V)
    zeta = (omega - k * u_b) / k
    E0 = float(amp)
    y0 = np.zeros(2 + N_v, dtype=complex)
    y0[0] = E0 / (1j * omega)
    y0[1] = E0
    y0[2:] = -1j * E0 * (-dF_kappa(s, kappa)) / (k * (s - zeta))
    return y0


# ----- analytic dispersion relation ---------------------------------------

def D_kappa(omega: complex, k: float, u_b: float, eps: float,
            kappa: int) -> complex:
    zeta = omega / k - u_b
    return omega**2 * (1.0 + 2.0 * eps * R_kappa(zeta, kappa) / k**2) - 1.0


def find_root_kappa(k, u_b, eps, kappa, omega0: complex) -> complex:
    def f(x):
        d = D_kappa(x[0] + 1j * x[1], k, u_b, eps, kappa)
        return [d.real, d.imag]
    sol = _scipy_root(f, [omega0.real, omega0.imag], tol=1e-13)
    return sol.x[0] + 1j * sol.x[1] if sol.success else np.nan


def peak_k_kappa(u_b, eps, kappa, k_lo=0.1, k_hi=0.9, n_k=41,
                 omega_seed=1.0 + 0.05j):
    """Scan k for the maximum kinetic growth rate of the analytic root."""
    best = (None, -np.inf, None)
    for k in np.linspace(k_lo, k_hi, n_k):
        w = find_root_kappa(k, u_b, eps, kappa, omega_seed)
        if np.isnan(w) or w.imag > 5.0:
            continue
        if w.imag > best[1]:
            best = (k, w.imag, w)
    return best[0], best[2]


# ----- EKR closure (N = 2) -------------------------------------------------

def _ekr_rhs(t, y_real, k, u_b, eps, kappa):
    y = y_real[:4] + 1j * y_real[4:]
    u, E, U0, U1 = y
    U2 = U0 * beta_kappa(_safe_xi_ratio(U1, U0), kappa)
    du = -E
    dE = u - eps * (u_b * U0 + U1)
    dU0 = -1j * k * u_b * U0 - 1j * k * U1
    dU1 = -1j * k * u_b * U1 - 1j * k * U2 + 1.0 * E      # M_0 = 1
    dy = np.array([du, dE, dU0, dU1])
    return np.concatenate([dy.real, dy.imag])


def ekr_evolve_kappa(k, u_b, eps, kappa, t_grid, y0,
                     rtol=1e-9, atol=1e-12) -> np.ndarray:
    y0r = np.concatenate([y0.real, y0.imag])
    sol = solve_ivp(_ekr_rhs, (t_grid[0], t_grid[-1]), y0r, t_eval=t_grid,
                    args=(k, u_b, eps, kappa), method="RK45",
                    rtol=rtol, atol=atol)
    if not sol.success:
        print(f"  ekr_evolve_kappa WARNING: {sol.message}")
    return sol.y[1] + 1j * sol.y[5]


def fluid_ic_kappa(k, u_b, kappa, omega, amp=1e-3) -> np.ndarray:
    """N=2 fluid eigenmode IC consistent with the kinetic one."""
    zeta = (omega - k * u_b) / k
    R = R_kappa(zeta, kappa)
    E0 = float(amp)
    U0 = -2j * E0 * R / k
    return np.array([E0 / (1j * omega), E0, U0, zeta * U0], dtype=complex)
