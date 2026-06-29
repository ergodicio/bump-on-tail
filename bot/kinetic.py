"""Kinetic ground-truth utilities for the bump-on-tail problem.

Setup: cold background plasma (u, E) + warm beam (Maxwellian at u_b with
thermal velocity v_b). Units: omega_p = v_b = 1.

Beam-frame dimensionless frequency
    xi_b = (omega - k u_b) / k

Linear kinetic dispersion at scales >> Debye length:
    D_kin(omega, k) = 1 - 1/omega^2 - (eps/k^2) Z'(xi_b)   ;   eps = n_b/n_0

Z(xi) = i sqrt(pi) w(xi)   (Faddeeva)
Z'(xi) = -2 (1 + xi Z(xi))
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import root
from scipy.special import wofz


SQRT_PI = float(np.sqrt(np.pi))


def Z(xi):
    """Plasma dispersion function via Faddeeva."""
    return 1j * SQRT_PI * wofz(xi)


def Zprime(xi):
    return -2.0 * (1.0 + xi * Z(xi))


def kinetic_dispersion(omega, k, u_b, eps):
    """D_kin(omega, k). Roots in upper half omega-plane are unstable modes."""
    xi = (omega - k * u_b) / k
    return 1.0 - 1.0 / omega**2 - (eps / k**2) * Zprime(xi)


def find_kinetic_mode(k, u_b, eps, omega0, max_iter: int = 20, tol: float = 1e-12):
    """Complex Newton iteration for a root of D_kin near omega0."""
    omega = complex(omega0)
    h = 1e-7
    for _ in range(max_iter):
        D = kinetic_dispersion(omega, k, u_b, eps)
        if abs(D) < tol:
            break
        dD = (kinetic_dispersion(omega + h, k, u_b, eps) -
              kinetic_dispersion(omega - h, k, u_b, eps)) / (2 * h)
        step = D / dD
        # guard against runaway steps
        if abs(step) > 0.5:
            step *= 0.5 / abs(step)
        omega -= step
    return omega


def trace_kinetic(ks, u_b, eps, omega0=1.0 + 0.01j):
    """Continuation in k: track one root from omega0."""
    out = np.empty(len(ks), dtype=complex)
    w = omega0
    for i, k in enumerate(ks):
        w = find_kinetic_mode(k, u_b, eps, w)
        out[i] = w
    return out


# ----- s-space discretized kinetic system (gives all modes via eigvals) -----

def s_grid(N_v: int, V: float = 6.0):
    s = np.linspace(-V, V, N_v)
    return s, s[1] - s[0]


def kinetic_system_sspace(k: float, u_b: float, eps: float,
                           N_v: int = 96, V: float = 6.0) -> np.ndarray:
    """Linear ODE matrix in (u, E, F[s_0], ..., F[s_{N_v-1}]) space.

    F = f_b1 / n_b (perturbed beam, beam-frame velocity coords). Phase mixing
    across the discrete velocity modes produces effective Landau damping for
    t < t_recurrence; for N_v large enough the most-unstable eigenvalue
    converges to the kinetic Langmuir / BoT mode.
    """
    s, ds = s_grid(N_v, V)
    dim = 2 + N_v
    A = np.zeros((dim, dim), dtype=complex)
    A[0, 1] = -1.0
    A[1, 0] = +1.0
    A[1, 2:] = -eps * (u_b + s) * ds
    src = (2.0 * s / SQRT_PI) * np.exp(-s**2)
    for j in range(N_v):
        A[2 + j, 2 + j] = -1j * k * (u_b + s[j])
        A[2 + j, 1] = src[j]
    return A


def kinetic_modes_sspace(k: float, u_b: float, eps: float,
                          N_v: int = 96, V: float = 6.0) -> np.ndarray:
    """All omega = i * eigval(A_sspace). Robust most-unstable-mode finder."""
    return 1j * np.linalg.eigvals(kinetic_system_sspace(k, u_b, eps, N_v, V))


def most_unstable_kinetic(k: float, u_b: float, eps: float,
                           N_v: int = 96, V: float = 6.0,
                           omega_re_lo: float = 0.3, omega_re_hi: float = 3.0
                           ) -> complex:
    """Return the most-unstable Langmuir-like kinetic mode at k.

    Filters out the spurious phase-mixed modes (very large Im, very large Re)
    and selects the omega with largest Im in a sensible Re-window.
    Result is Newton-refined on the analytic dispersion to machine precision.
    """
    omegas = kinetic_modes_sspace(k, u_b, eps, N_v, V)
    mask = (omegas.real > omega_re_lo) & (omegas.real < omega_re_hi)
    cand = omegas[mask]
    omega0 = cand[np.argmax(cand.imag)] if len(cand) > 0 else omegas[np.argmax(omegas.imag)]
    return find_kinetic_mode(k, u_b, eps, omega0)
