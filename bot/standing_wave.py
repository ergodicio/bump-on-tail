"""Single-species Vlasov-Poisson density perturbation for Figure 6.

v_t=omega_pe=1, k=sqrt(2)*k_lambda_D. Poisson gives E=-i U_0/k
and Phi=2 U_0/k^2. U_n(0)=epsilon*M_n, a standing wave with density nodes.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import newton

from bot.closed_loop import _safe_xi_ratio
from bot.closures.response import Z, beta

K_LAMBDA_D = 0.4
K = K_LAMBDA_D * np.sqrt(2.0)
EPS = 1e-3
T_END = 40.0
T = np.linspace(0.0, T_END, 2001)
M = np.array([1.0, 0.0, 0.5, 0.0])


def R(z):
    return 1.0 + z * Z(z)


def landau_root() -> complex:
    disp = lambda z: 1.0 + (2.0 / K**2) * R(z)
    zeta = newton(disp, 2.3 - 0.1j, tol=1e-12)
    return zeta * K  # omega, in omega_p units


def kinetic(N_v: int = 1000, V: float = 8.0) -> np.ndarray:
    v = np.linspace(-V, V, N_v)
    dv = v[1] - v[0]
    w = np.full(N_v, dv)
    w[0] = w[-1] = dv / 2
    f0 = np.exp(-(v**2)) / np.sqrt(np.pi)
    df0 = -2.0 * v * f0
    A = np.diag(-1j * K * v) + (1j / K) * np.outer(df0, w)
    lam, Vec = np.linalg.eig(A)
    c = np.linalg.solve(Vec, EPS * f0)
    return np.array([w @ (Vec @ (c * np.exp(lam * tt))) for tt in T])


def evolve(N: int, closure, rtol=1e-9, atol=1e-13, U_init: np.ndarray | None = None) -> np.ndarray:
    """Return U_0(t) for the N-moment system closed by closure(U) -> U_N.

    Default initial condition is the density perturbation U_n(0) = eps M_n;
    pass U_init (length N, complex) to start elsewhere.
    """

    def rhs(t, y):
        U = y[:N] + 1j * y[N:]
        UN = closure(U)
        E = -(1j / K) * U[0]
        dU = np.empty(N, dtype=complex)
        for n in range(N):
            nxt = U[n + 1] if n + 1 < N else UN
            dU[n] = -1j * K * nxt + n * M[n - 1] * E if n > 0 else -1j * K * nxt
        return np.concatenate([dU.real, dU.imag])

    U0 = EPS * M[:N] if U_init is None else np.asarray(U_init, dtype=complex)
    sol = solve_ivp(
        rhs,
        (0.0, T_END),
        np.concatenate([U0.real, U0.imag]),
        t_eval=T,
        method="RK45",
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"{N}-moment integration: {sol.message}")
    out = np.full(len(T), np.nan, dtype=complex)
    out[: sol.y.shape[1]] = sol.y[0] + 1j * sol.y[N]
    return out


def linear_closure(a: np.ndarray):
    a = np.asarray(a, dtype=complex)
    return lambda U: a @ U


def ekr_closure(U):
    return U[0] * beta(_safe_xi_ratio(U[1], U[0]))


def fit_mode(U0: np.ndarray, t0: float = 5.0):
    """(omega, gamma) from zero-crossing spacing and the log-envelope slope."""
    m = (T >= t0) & np.isfinite(U0)
    t, x = T[m], U0.real[m]
    s = np.sign(x)
    ic = np.where(s[1:] * s[:-1] < 0)[0]
    if len(ic) < 3:
        return np.nan, np.nan
    tc = t[ic] - x[ic] * (t[ic + 1] - t[ic]) / (x[ic + 1] - x[ic])
    omega = np.pi / np.mean(np.diff(tc))
    env_t, env_v = [], []
    for i0, i1 in zip(ic[:-1], ic[1:]):
        j = i0 + np.argmax(np.abs(x[i0 : i1 + 1]))
        env_t.append(t[j])
        env_v.append(np.abs(x[j]))
    gamma = np.polyfit(env_t, np.log(env_v), 1)[0]
    return omega, gamma
