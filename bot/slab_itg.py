"""Slab ITG validation used in Figure 1.

Time is k_parallel*v_ti*t; quasineutrality gives phi=U_0/tau.
The drive S(w) and moment sources are derived from the same drift-kinetic
model. The EKR argument includes the diamagnetic frequency shift.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from bot.closed_loop import _safe_xi_ratio
from bot.closures.pade import maxwellian_moment
from bot.closures.response import Z

SQRT_PI = float(np.sqrt(np.pi))
CHI1_HP = 2.0 / SQRT_PI


def manifold_Un_over_phi(z, zeta_star: float, eta: float, n_max: int = 3):
    """U_n/phi on the kinetic ITG manifold, n = 0..n_max (needs I_{n_max+2})."""
    I = [Z(z)]
    for n in range(n_max + 2):
        I.append(z * I[-1] + maxwellian_moment(n))
    return [
        zeta_star * (1.0 - 0.5 * eta) * I[n] + zeta_star * eta * I[n + 2] - I[n + 1]
        for n in range(n_max + 1)
    ]


def beta_itg(z, zeta_star: float, eta: float):
    """Exact U_2/U_0 on the ITG manifold at frequency z (cf. beta_closure)."""
    U = manifold_Un_over_phi(z, zeta_star, eta, n_max=2)
    den = U[0]
    if abs(den) < 1e-12:
        den = 1e-12 * (den / abs(den) if den != 0 else 1.0)
    return U[2] / den


def R_kinetic(zeta, zeta_star: float, eta: float):
    """DKE-derived ion response R = -U_0/phi (has the eta*zeta*zeta_* term)."""
    return (
        1.0
        - eta * zeta * zeta_star
        + (zeta - zeta_star * (1.0 - 0.5 * eta + eta * zeta**2)) * Z(zeta)
    )


def D_kinetic(zeta, zeta_star: float, eta: float, tau: float = 1.0):
    return R_kinetic(zeta, zeta_star, eta) + tau


def eta_threshold(zeta_star: float, tau: float = 1.0) -> float:
    """Marginal eta of the DKE-derived dispersion (analytic, real-zeta)."""
    return 1.0 + np.sqrt(1.0 + 2.0 * tau * (1.0 + tau) / zeta_star**2)


def find_mode(D_func, zeta0: complex, zeta_star: float, eta: float, tau: float = 1.0) -> complex:
    """Newton-solve D(zeta) = 0 near zeta0; nan on failure."""

    def res(x):
        v = D_func(complex(x[0], x[1]), zeta_star, eta, tau)
        return [v.real, v.imag]

    sol = root(res, [zeta0.real, zeta0.imag], method="hybr", tol=1e-13)
    if not sol.success:
        return complex(float("nan"), float("nan"))
    return complex(sol.x[0], sol.x[1])


def max_growing_root(
    D_func,
    zeta_star: float,
    eta: float,
    tau: float = 1.0,
    re_lo: float = -3.0,
    re_hi: float = 3.0,
    im_lo: float = 0.001,
    im_hi: float = 3.0,
    n_re: int = 25,
    n_im: int = 15,
) -> complex:
    """Most-unstable root of D in the upper half plane (grid + Newton).

    Returns complex(nan, 0) if no growing root is found.
    """
    best = complex(float("nan"), 0.0)
    for re_v in np.linspace(re_lo, re_hi, n_re):
        for im_v in np.geomspace(im_lo, im_hi, n_im):
            zt = find_mode(D_func, complex(re_v, im_v), zeta_star, eta, tau)
            if np.isnan(zt.real) or zt.imag <= best.imag:
                continue
            if abs(D_func(zt, zeta_star, eta, tau)) < 1e-9:
                best = zt
    return best


def w_grid(N_w: int, W: float = 8.0):
    w = np.linspace(-W, W, N_w)
    return w, w[1] - w[0]


def drive_S(w: np.ndarray, zeta_star: float, eta: float) -> np.ndarray:
    """S(w) = (zeta_*[1 + eta(w^2 - 1/2)] - w) F0(w)."""
    F0 = np.exp(-(w**2)) / SQRT_PI
    return (zeta_star * (1.0 + eta * (w**2 - 0.5)) - w) * F0


def kinetic_system(
    zeta_star: float, eta: float, tau: float = 1.0, N_w: int = 600, W: float = 8.0
) -> np.ndarray:
    """Linear ODE matrix A for dg/dt = A g on the w-grid.

    dg_j/dt = -i w_j g_j + i S(w_j) phi,   phi = U_0/tau = (dw sum_l g_l)/tau
    """
    w, dw = w_grid(N_w, W)
    A = np.diag(-1j * w).astype(complex)
    A += np.outer(1j * drive_S(w, zeta_star, eta), np.full(N_w, dw / tau))
    return A


def kinetic_evolve(
    zeta_star: float,
    eta: float,
    tau: float,
    t_grid: np.ndarray,
    g0: np.ndarray,
    N_w: int = 600,
    W: float = 8.0,
) -> dict:
    """Evolve the discretized DKE from g0 via eigendecomposition.

    Returns dict with t, U0(t), U1(t), U2(t), phi(t).
    """
    w, dw = w_grid(N_w, W)
    A = kinetic_system(zeta_star, eta, tau, N_w, W)
    lam, V = np.linalg.eig(A)
    c = np.linalg.solve(V, g0)
    U = np.empty((3, len(t_grid)), dtype=complex)
    for i, t in enumerate(t_grid):
        g = V @ (c * np.exp(lam * t))
        U[0, i] = dw * np.sum(g)
        U[1, i] = dw * np.sum(w * g)
        U[2, i] = dw * np.sum(w**2 * g)
    return {"t": t_grid, "U0": U[0], "U1": U[1], "U2": U[2], "phi": U[0] / tau}


def fluid_hp_system(
    zeta_star: float, eta: float, tau: float = 1.0, Gamma: float = 3.0, chi1: float = CHI1_HP
) -> np.ndarray:
    """3x3 matrix for d/dt (U_0, U_1, U_2) with the HP heat-flux closure.

        dU_0 = -i U_1 + i zeta_* phi
        dU_1 = -i U_2 - (i/2) phi
        dU_2 = -i U_3 + i zeta_*(1+eta)/2 phi
        U_3  = (Gamma/2) U_1 - i (chi1/2) (2 U_2 - U_0),   phi = U_0/tau

    Gamma = 3 is the exact 1D hierarchy coefficient (HP recommended); the
    Braginskii-like variant uses Gamma = 5/3.
    """
    A = np.zeros((3, 3), dtype=complex)
    # dU_0
    A[0, 1] = -1j
    A[0, 0] = 1j * zeta_star / tau
    # dU_1
    A[1, 2] = -1j
    A[1, 0] = -0.5j / tau
    # dU_2 = -i U_3 + source
    A[2, 1] = -1j * (Gamma / 2.0)
    A[2, 2] = -1j * (-1j * chi1)  # -i * (-i chi1/2 * 2) U_2
    A[2, 0] = -1j * (1j * chi1 / 2.0)  # -i * (+i chi1/2) U_0
    A[2, 0] += 1j * zeta_star * (1.0 + eta) / (2.0 * tau)
    return A


def fluid_hp_modes(
    zeta_star: float, eta: float, tau: float = 1.0, Gamma: float = 3.0, chi1: float = CHI1_HP
) -> np.ndarray:
    """All zeta = i * eigval (3 modes)."""
    return 1j * np.linalg.eigvals(fluid_hp_system(zeta_star, eta, tau, Gamma, chi1))


def _linear_rhs(t, y_real, A):
    n = A.shape[0]
    y = y_real[:n] + 1j * y_real[n:]
    dy = A @ y
    return np.concatenate([dy.real, dy.imag])


def linear_evolve_rk45(
    A: np.ndarray, t_grid: np.ndarray, y0: np.ndarray, rtol: float = 1e-9, atol: float = 1e-12
) -> np.ndarray:
    """RK45 time-stepping of dy/dt = A y (same physics as eigendecomposition,
    but genuinely integrated -- for apples-to-apples comparison against a
    nonlinear closure that has no choice but to be time-stepped)."""
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
    return sol.y[:n, :m] + 1j * sol.y[n:, :m]


def fluid_hp_evolve(
    zeta_star: float,
    eta: float,
    tau: float,
    t_grid: np.ndarray,
    y0: np.ndarray,
    Gamma: float = 3.0,
    chi1: float = CHI1_HP,
    method: str = "eigen",
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> dict:
    """Evolve the HP 3-moment system from y0 = (U_0, U_1, U_2).

    method="eigen" (default): exact via eigendecomposition (matrix
    exponential), no time-stepping error.
    method="rk45": genuinely time-stepped, matching the numerical scheme
    used for the nonlinear direct-beta/alpha closures (fair side-by-side
    comparison of the transient, not just the asymptotic rate).
    """
    A = fluid_hp_system(zeta_star, eta, tau, Gamma, chi1)
    if method == "eigen":
        lam, V = np.linalg.eig(A)
        c = np.linalg.solve(V, y0)
        Y = np.empty((3, len(t_grid)), dtype=complex)
        for i, t in enumerate(t_grid):
            Y[:, i] = V @ (c * np.exp(lam * t))
    elif method == "rk45":
        Y = linear_evolve_rk45(A, t_grid, y0, rtol=rtol, atol=atol)
    else:
        raise ValueError(f"unknown method {method!r}")
    return {"t": t_grid, "U0": Y[0], "U1": Y[1], "U2": Y[2], "phi": Y[0] / tau}


def density_ic_kinetic(amp: float = 1e-3, N_w: int = 600, W: float = 8.0) -> np.ndarray:
    """Generic density perturbation: g0 = amp F0(w)  (U_0 = amp, U_1 = 0...)."""
    w, _dw = w_grid(N_w, W)
    return amp * np.exp(-(w**2)) / SQRT_PI


def density_ic_fluid(amp: float = 1e-3, N: int = 3) -> np.ndarray:
    """Matching fluid IC: U_n = amp * M_n for n = 0..N-1 (moments of amp*F0(w)).

    Exact for any N (was previously hardcoded/incomplete for N > 3, silently
    zeroing M_4, M_6, ... instead of their true nonzero Maxwellian values).
    """
    return np.array([amp * maxwellian_moment(n) for n in range(N)], dtype=complex)


def fit_mode(t: np.ndarray, x: np.ndarray, frac: tuple[float, float] = (0.5, 1.0)) -> complex:
    """Fit zeta = omega_r + i gamma from a complex signal x(t) ~ e^{-i zeta t}.

    gamma from the log-|x| slope, omega_r from the unwrapped-phase slope,
    over the window frac of the record.
    """
    i0, i1 = int(frac[0] * len(t)), int(frac[1] * len(t))
    tt, xx = t[i0:i1], x[i0:i1]
    mag = np.abs(xx)
    if mag.max() <= 0 or np.any(mag < 1e-300):
        return complex(float("nan"), float("nan"))
    gamma = np.polyfit(tt, np.log(mag), 1)[0]
    omega_r = -np.polyfit(tt, np.unwrap(np.angle(xx)), 1)[0]
    return complex(omega_r, gamma)


def direct_beta_evolve(zeta_star, eta, tau, t_grid, y0, rtol=1e-9, atol=1e-12):
    """N=2 EKR with ITG response; return the complex density and velocity."""

    def rhs(t, y):
        U0, U1 = y[:2] + 1j * y[2:]
        phi = U0 / tau
        zeta = _safe_xi_ratio(U1, U0) - zeta_star / tau
        U2 = U0 * beta_itg(zeta, zeta_star, eta)
        dy = np.array([-1j * U1 + 1j * zeta_star * phi, -1j * U2 - 0.5j * phi])
        return np.concatenate((dy.real, dy.imag))

    sol = solve_ivp(
        rhs,
        (t_grid[0], t_grid[-1]),
        np.concatenate((y0.real, y0.imag)),
        t_eval=t_grid,
        rtol=rtol,
        atol=atol,
        method="RK45",
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    state = sol.y[:2] + 1j * sol.y[2:]
    return dict(t=sol.t, U0=state[0], U1=state[1], phi=state[0] / tau)
