"""Padé closure for the N-moment bump-on-tail beam fluid.

Beam moment hierarchy from linearized Vlasov + integration by parts:
    xi U_n - U_{n+1} = n M_{n-1} Phi      n = 0, ..., N-1
with M_k = <s^k>_{f_b0} Maxwellian moments (M_0 = 1, M_1 = 0, M_2 = 1/2, ...).

A linear closure U_N = sum_{i<N} a_i U_i yields a rational approximant of the
kinetic response U_0/Phi = -sqrt(pi) Z'(xi). The Padé closure picks (a_i) to
match the Taylor expansion of -sqrt(pi) Z'(xi) at xi=0 to order N.

For N=3 the closure has three complex coefficients and produces a 3-pole
rational of degree (1, 3) in xi. This is the BoT analog of HP-class closures
(Hunana's R_3,2 family; specific coefficients differ from Maxwellian-bulk HP
because the dimensionless freq variable here is xi_b = (omega - k u_b)/(k v_b)).
"""

from __future__ import annotations

from math import factorial

import numpy as np


SQRT_PI = float(np.sqrt(np.pi))


def maxwellian_moment(k: int) -> float:
    """M_k = <s^k> for a unit-width Maxwellian; zero for odd k."""
    if k < 0 or k % 2 == 1:
        return 0.0
    j = k // 2
    return factorial(2 * j) / (4**j * factorial(j))


def _kinetic_taylor(K: int) -> np.ndarray:
    """First K Taylor coefficients c_0..c_{K-1} of -sqrt(pi) Z'(xi) at xi=0.

    Z(xi) coefficients:
        z_{2n}   = i sqrt(pi) (-1)^n / n!
        z_{2n+1} = -2 (-1)^n / (3/2)_n
    Then c_0 = 2 sqrt(pi), c_k = 2 sqrt(pi) * z_{k-1} (since Z' = -2 - 2 xi Z).
    """
    z = np.zeros(K, dtype=complex)
    for k in range(K):
        if k % 2 == 0:
            n = k // 2
            z[k] = 1j * SQRT_PI * (-1) ** n / factorial(n)
        else:
            n = (k - 1) // 2
            poch = 1.0
            for j in range(n):
                poch *= 1.5 + j        # (3/2)_n
            z[k] = -2.0 * (-1) ** n / poch
    c = np.zeros(K, dtype=complex)
    c[0] = 2.0 * SQRT_PI
    for k in range(1, K):
        c[k] = 2.0 * SQRT_PI * z[k - 1]
    return c


def _Q_taylor(N: int, K: int) -> np.ndarray:
    """Taylor coefs [Q_i]_k for i=0..N, k=0..K-1, used in the Padé matching.

    Recurrence Q_{i+1}(xi) = xi Q_i(xi) - i M_{i-1} gives
        [Q_i]_k = [Q_{i-1}]_{k-1} - (i-1) M_{i-2} * delta_{k,0}.
    """
    Q = np.zeros((N + 1, K), dtype=complex)
    for i in range(1, N + 1):
        Q[i, 1:] = Q[i - 1, :-1]
        Q[i, 0] -= (i - 1) * maxwellian_moment(i - 2)
    return Q


def pade_coefficients(N: int) -> np.ndarray:
    """Padé closure: solve a so fluid_U0(xi; a) matches -sqrt(pi) Z'(xi) to O(xi^N).

    Returns array of complex coefficients (a_0, ..., a_{N-1}).
    """
    c = _kinetic_taylor(N)
    Q = _Q_taylor(N, N)
    A = np.zeros((N, N), dtype=complex)
    b = np.zeros(N, dtype=complex)
    for k in range(N):
        for i in range(N):
            A[k, i] = Q[i, k] + (c[k - i] if k - i >= 0 else 0.0)
        b[k] = Q[N, k] + (c[k - N] if k - N >= 0 else 0.0)
    return np.linalg.solve(A, b)


def fluid_U0(xi, a) -> np.ndarray:
    """Closed fluid response U_0(xi)/Phi for closure U_N = sum_i a_i U_i.

    Builds and solves the moment system at every xi:
        rows 0..N-2:  xi U_n - U_{n+1} = n M_{n-1} Phi
        row    N-1:   xi U_{N-1} - sum a_i U_i = (N-1) M_{N-2} Phi
    Phi normalized to 1.
    """
    a = np.asarray(a, dtype=complex)
    N = len(a)
    xi = np.asarray(xi, dtype=complex)
    M = np.zeros(xi.shape + (N, N), dtype=complex)
    b = np.zeros(N, dtype=complex)
    for n in range(N - 1):
        M[..., n, n] = xi
        M[..., n, n + 1] = -1.0
        b[n] = n * maxwellian_moment(n - 1) if n > 0 else 0.0
    M[..., N - 1, N - 1] = xi
    M[..., N - 1, :] -= a[None, :] if xi.ndim == 0 else a
    b[N - 1] = (N - 1) * maxwellian_moment(N - 2) if N > 0 else 0.0
    U = np.linalg.solve(M, b)
    return U[..., 0]
