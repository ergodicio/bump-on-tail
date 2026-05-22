"""Linearized two-fluid bump-on-tail with N-moment closure on the beam.

State per Fourier mode k:  y = (u, E, U_0, ..., U_{N-1})  in C^{N+2}

    d_t u   = -E                                       (cold bulk momentum)
    d_t E   = u - eps (u_b U_0 + U_1)                  (Ampere; eps = n_b/n_0)
    d_t U_n = -ik u_b U_n - ik U_{n+1} + n M_{n-1} E   (beam moment, n < N-1)
    closure: U_N -> sum_i a_i U_i  in the n = N-1 row

Units: omega_p = v_b = 1.
"""

from __future__ import annotations

import numpy as np

from bot.closures.pade import maxwellian_moment


def fluid_system(k: float, u_b: float, eps: float, a: np.ndarray) -> np.ndarray:
    """Build the (N+2) x (N+2) linear ODE matrix A(k) for d_t y = A y."""
    a = np.asarray(a, dtype=complex)
    N = len(a)
    dim = 2 + N
    A = np.zeros((dim, dim), dtype=complex)
    # bulk
    A[0, 1] = -1.0
    A[1, 0] = +1.0
    A[1, 2 + 0] = -eps * u_b
    if N >= 2:
        A[1, 2 + 1] = -eps
    # beam moments, rows 0..N-2
    for n in range(N - 1):
        A[2 + n, 2 + n] = -1j * k * u_b
        A[2 + n, 2 + n + 1] = -1j * k
        if n >= 1:
            A[2 + n, 1] = n * maxwellian_moment(n - 1)
    # closure row, n = N-1
    n = N - 1
    A[2 + n, 2 + n] = -1j * k * u_b
    if n >= 1:
        A[2 + n, 1] = n * maxwellian_moment(n - 1)
    for i in range(N):
        A[2 + n, 2 + i] += -1j * k * a[i]
    return A


def fluid_modes(k: float, u_b: float, eps: float, a: np.ndarray) -> np.ndarray:
    """All mode frequencies omega = i * eigval(A). length = N + 2."""
    return 1j * np.linalg.eigvals(fluid_system(k, u_b, eps, a))


def trace_fluid(ks, u_b, eps, a, ws_ref) -> np.ndarray:
    """For each k, pick the fluid eigenmode closest to ws_ref[k] (continuation)."""
    out = np.empty(len(ks), dtype=complex)
    for i, k in enumerate(ks):
        cands = fluid_modes(k, u_b, eps, a)
        j = int(np.argmin(np.abs(cands - ws_ref[i])))
        out[i] = cands[j]
    return out
