"""Padé closure for the N-moment bump-on-tail beam fluid.

Beam moment hierarchy from linearised Vlasov + integration by parts:
    xi U_n - U_{n+1} = n M_{n-1} Phi      n = 0, ..., N-1
with M_k = <s^k>_{f_b0} Maxwellian moments (M_0 = 1, M_1 = 0, M_2 = 1/2, ...).

The kinetic beam response (derived from Vlasov with Maxwellian beam f_b0) is
    U_0/Phi = Z'(xi_b)
where Z' = -2(1 + xi Z(xi)) is the derivative of the plasma dispersion function
and xi_b = (omega - k u_b)/(k v_b).  This is consistent with the kinetic
dispersion  D = 1 - 1/omega^2 - (eps/k^2) Z'(xi_b) = 0  in kinetic.py.

For N = 3, the linear closure U_3 = a_0 U_0 + a_1 U_1 + a_2 U_2 yields the
rational approximant
    U_0/Phi = (xi - a_2) / (xi^3 - a_2 xi^2 - a_1 xi - a_0)
which always behaves as +1/xi^2 for large xi, matching the leading term of
    Z'(xi) ~ 1/xi^2 + 3/(2 xi^4) + 15/(4 xi^6) + ...   (large-arg)

HP (Hammett-Perkins 1990) construction for N = 3
-------------------------------------------------
Following HP1990 (Eq. 9 with mu_1=0, Gamma=3, chi_1=2/sqrt(pi)):

  1. Large-argument O(1/xi^4) term forces  a_1 = 3/2   (Gamma=3 analogue).
  2. The relation  a_0 = -a_2/2  comes from the correct high-frequency
     structure of the closure  (mu_1=0 analogue).
  3. Matching Im[Z'(0+)] to model Landau damping gives  a_2 = -2i/sqrt(pi)
     (chi_1 = 2/sqrt(pi) analogue).

Result:
    a = (a_0, a_1, a_2) = (i/sqrt(pi),  3/2,  -2i/sqrt(pi))

Verification:
  * Z'(0) = -2;   f(0) = a_2/a_0 = (-2i/sqrt(pi))/(i/sqrt(pi)) = -2  ✓
  * Large arg:  f(xi) = 1/xi^2 + (3/2)/xi^4 + O(1/xi^6)  ✓
  * Small arg:  f(xi) ≈ -2 - 2i*sqrt(pi)*xi  =  Z'(xi)|_{xi→0}  ✓

For N != 3 a Taylor-at-zero matching to Z'(xi) is used as a fallback.
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


def _zprime_taylor(K: int) -> np.ndarray:
    """First K Taylor coefficients c_0..c_{K-1} of Z'(xi) at xi=0.

    Recurrence (from Z^{(n+1)}(0) = -2n Z^{(n-1)}(0)):
        c_n = -2 c_{n-2} / (n-1)   for n >= 2
    with  c_0 = Z'(0) = -2,  c_1 = Z''(0)/1! = -2i sqrt(pi).
    """
    c = np.zeros(K, dtype=complex)
    if K == 0:
        return c
    c[0] = -2.0
    if K == 1:
        return c
    c[1] = -2.0j * SQRT_PI
    for n in range(2, K):
        c[n] = -2.0 * c[n - 2] / (n - 1)
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
    """Padé closure coefficients a = (a_0, ..., a_{N-1}) for the N-moment beam fluid.

    For N=3: uses the Hammett-Perkins (1990) construction — large-argument
    asymptotic matching first, then small-argument imaginary matching for
    Landau damping.  See module docstring for derivation.

    For N != 3: falls back to Taylor-at-zero matching of the rational
    U_0/Phi to Z'(xi) (the correct kinetic target).
    """
    if N == 3:
        chi1 = 2.0 / SQRT_PI          # = 2/sqrt(pi)
        a0   =  0.5j * chi1            # = i/sqrt(pi)
        a1   =  1.5                    # = 3/2
        a2   = -1j  * chi1            # = -2i/sqrt(pi)
        return np.array([a0, a1, a2], dtype=complex)

    # General N: Taylor matching at xi=0 for Z'(xi)
    c = _zprime_taylor(N)
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
    M[..., N - 1, :] -= a          # fix: broadcast correctly for scalar and array xi
    b[N - 1] = (N - 1) * maxwellian_moment(N - 2) if N > 0 else 0.0
    U = np.linalg.solve(M, b)
    return U[..., 0]
