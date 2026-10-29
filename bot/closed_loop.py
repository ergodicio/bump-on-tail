"""Time integration for the cold background plus warm beam.

State y=(u,E,U_0,...); frequencies use exp(-i*omega*t). Linear systems
support eigendecomposition or RK45; the EKR closure uses RK45 and the
same ratio floor/cap as the published calculation.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from bot.closures.response import Zprime, mode_ratios
from bot.closures.response import beta as _u2_beta
from bot.fluid import fluid_system
from bot.kinetic import kinetic_system_sspace

_XI_RATIO_MAX = 12.0


def kinetic_evolve(
    k: float,
    u_b: float,
    eps: float,
    t_grid: np.ndarray,
    y0: np.ndarray,
    N_v: int = 96,
    V: float = 6.0,
) -> np.ndarray:
    """Return E(t) for the s-space kinetic system."""
    A = kinetic_system_sspace(k, u_b, eps, N_v, V)
    eigvals, evecs = np.linalg.eig(A)
    coefs = np.linalg.solve(evecs, y0)
    E_t = np.empty(len(t_grid), dtype=complex)
    for i, ti in enumerate(t_grid):
        y = evecs @ (coefs * np.exp(eigvals * ti))
        E_t[i] = y[1]
    return E_t


def _linear_rhs(t, y_real, A):
    n = A.shape[0]
    y = y_real[:n] + 1j * y_real[n:]
    dy = A @ y
    return np.concatenate([dy.real, dy.imag])


def pade_evolve(
    k: float,
    u_b: float,
    eps: float,
    a: np.ndarray,
    t_grid: np.ndarray,
    y0: np.ndarray,
    method: str = "eigen",
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> np.ndarray:
    """Return E(t) for the closed fluid system with Padé closure a.

    method="eigen" (default): exact via eigendecomposition (matrix
    exponential), no time-stepping error.
    method="rk45": genuinely time-stepped via RK45, matching the numerical
    scheme used for the nonlinear direct-beta/alpha closures (fair
    side-by-side comparison of the transient, not just the asymptotic
    rate).
    """
    A = fluid_system(k, u_b, eps, a)
    if method == "eigen":
        eigvals, evecs = np.linalg.eig(A)
        coefs = np.linalg.solve(evecs, y0)
        E_t = np.empty(len(t_grid), dtype=complex)
        for i, ti in enumerate(t_grid):
            y = evecs @ (coefs * np.exp(eigvals * ti))
            E_t[i] = y[1]
        return E_t
    elif method == "rk45":
        n = A.shape[0]
        y0_real = np.concatenate([y0.real, y0.imag])
        sol = solve_ivp(
            _linear_rhs,
            (t_grid[0], t_grid[-1]),
            y0_real,
            t_eval=t_grid,
            args=(A,),
            method="RK45",
            rtol=rtol,
            atol=atol,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        m = len(sol.t)
        y_complex = sol.y[:n, :m] + 1j * sol.y[n:, :m]
        return y_complex[1]
    else:
        raise ValueError(f"unknown method {method!r}")


def _safe_xi_ratio(
    U1: complex, U0: complex, U0_floor: float = 1e-15, max_mag: float = _XI_RATIO_MAX
) -> complex:
    """Return U_1/U_0, clipped in magnitude to avoid feeding a Z'-based
    closure an argument that blows up near a U_0 node."""
    if abs(U0) < U0_floor:
        return 0.0 + 0.0j
    xi = U1 / U0
    mag = abs(xi)
    if mag > max_mag:
        xi = xi * (max_mag / mag)
    return xi


def _direct_u2_rhs(t, y_real, k: float, u_b: float, eps: float):
    """RHS for N=2 fluid system with exact per-mode beta(U_1/U_0) closure.

    State y = (u, E, U_0, U_1),  stacked as real-imaginary pairs.
    Closure: U_2 = U_0 * beta(U_1/U_0)  where beta(xi) = xi^2 - 1/Z'(xi).

    Off the single-eigenmode manifold (e.g. generic delta-E IC), U_1/U_0 is
    not equal to xi_b for any single mode; beta is evaluated at the effective
    ratio and acts as the off-manifold extension.
    """
    y = y_real[:4] + 1j * y_real[4:]
    u, E, U0, U1 = y
    U2 = U0 * _u2_beta(_safe_xi_ratio(U1, U0))
    du = -E
    dE = u - eps * (u_b * U0 + U1)
    dU0 = -1j * k * u_b * U0 - 1j * k * U1
    dU1 = -1j * k * u_b * U1 - 1j * k * U2 + 1.0 * E  # M_0 = 1
    dy = np.array([du, dE, dU0, dU1])
    return np.concatenate([dy.real, dy.imag])


def direct_u2_evolve(
    k: float,
    u_b: float,
    eps: float,
    t_grid: np.ndarray,
    y0: np.ndarray,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> np.ndarray:
    """Return E(t) for the N=2 fluid system with the exact beta closure."""
    y0_real = np.concatenate([y0.real, y0.imag])
    sol = solve_ivp(
        _direct_u2_rhs,
        (t_grid[0], t_grid[-1]),
        y0_real,
        t_eval=t_grid,
        args=(k, u_b, eps),
        method="RK45",
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    y_complex = sol.y[:4] + 1j * sol.y[4:]
    return y_complex[1]  # E(t)


def fluid_eigenmode_ic(k, u_b, omega, order, amp=1e-3):
    """Kinetic eigenmode moments projected onto an order-moment fluid state.

    The initial state is shared across closures of the same order. A linear
    Padé closure subsequently projects it onto its own supported modes.
    """
    zeta = (omega - k * u_b) / k
    density = 1j * Zprime(zeta) * amp / k
    return np.concatenate(([amp / (1j * omega), amp], density * mode_ratios(zeta, order - 1)))
