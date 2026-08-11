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


def aaa_coefficients_n4() -> np.ndarray:
    """N=4 closure coefficients from an AAA rational fit to Z'(xi), least-
    squares projected onto the N=4 moment-closure family.

    Unlike pade_coefficients(3) (HP: matched at exactly 2 points, xi=0 and
    xi->infinity) or pade_coefficients(N!=3) (Taylor-matched at the single
    point xi=0), this targets a GLOBAL fit over the real axis:

      1. scipy.interpolate.AAA(max_terms=5) fits Z'(xi) for xi in [-6,6]
         (real axis; AAA greedily selects support points to minimize
         absolute error, so a real-valued domain is used to avoid the
         exponential blow-up of Z' for large negative Im(xi) dominating
         the fit).  This recovers 4 poles and 4 residues, matching Z' to
         ~1-5% relative error at O(1) |xi| (vs HP's 25-70% there) -- but
         that AAA rational function has 2*4-1=7 complex degrees of freedom
         (poles AND residues both free), strictly more than this closure
         family's N=4 complex dof (residues are NOT independently free
         here -- fixing the closure's implied poles via the moment
         recursion also fixes its residues, so matching AAA's poles alone
         gives only ~20-25% accuracy, no better than HP).
      2. So a_0..a_3 are instead found by nonlinear least-squares,
         directly minimizing |fluid_U0(xi; a) - Z'(xi)| / (|Z'(xi)|+0.3)
         over 300 points on xi in [-4.5, 4.5] (scipy.optimize.least_squares,
         multiple random restarts around a HP-extended initial guess).

    Result: median relative error ~1.6% on the real axis (matching AAA's
    own quality there), degrading off-axis with growing |Im(xi)| (since the
    fit only used real-axis data) -- e.g. ~7% at xi=1.512-0.297j, ~21% at
    xi=0.580-0.461j -- still a large improvement over HP N=3's 64-68% at
    those same points.  Coefficients alternate real/imaginary (a_0, a_2
    real; a_1, a_3 imaginary), the same parity pattern as HP's own
    (a_0, a_2 imaginary; a_1 real) -- both reflect Z'(-conj(xi)) =
    conj(Z'(xi)).
    """
    return np.array([-2.08597030, 5.06030815j, 5.61145296, -2.92600838j],
                    dtype=complex)


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


# ---------------------------------------------------------------------------
# Two-point (HP / Hunana style) construction: n_zero conditions at xi = 0 and
# the rest against the large-argument asymptotic series of Z'.
# ---------------------------------------------------------------------------

def _zprime_asymptotic(M: int) -> np.ndarray:
    """s_1..s_M with Z'(xi) ~ sum_m s_m xi^{-2m} = 1/xi^2 + 3/(2 xi^4) + ...

    s_1 = 1, s_{m+1} = s_m (2m+1)/2.
    """
    s = np.zeros(M + 1)
    if M >= 1:
        s[1] = 1.0
    for m in range(1, M):
        s[m + 1] = s[m] * (2 * m + 1) / 2.0
    return s


def _zero_rows(N: int, n_zero: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows enforcing R_N(xi) = Z'(xi) + O(xi^n_zero) at xi = 0.

    Same linearised conditions pade_coefficients(N != 3) uses, but only the
    first n_zero of them.
    """
    K = max(n_zero, N + 1)
    c = _zprime_taylor(K)
    Q = _Q_taylor(N, K)                       # Q_i = -q_i (sign convention)
    A = np.zeros((n_zero, N), dtype=complex)
    b = np.zeros(n_zero, dtype=complex)
    for k in range(n_zero):
        for i in range(N):
            A[k, i] = Q[i, k] + (c[k - i] if k - i >= 0 else 0.0)
        b[k] = Q[N, k] + (c[k - N] if k - N >= 0 else 0.0)
    return A, b


def _inf_rows(N: int, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows enforcing R_N(xi) = Z'_asym(xi) + O(xi^{-...}) as xi -> infinity.

    Condition at the coefficient of xi^p (p = N-2, N-3, ...) of
    N(xi) - Z'_asym(xi) D(xi), with N = q_N - sum a_i q_i and
    D = xi^N - sum a_i xi^i:

        sum_i a_i ( s_{(i-p)/2} - [q_i]_p ) = s_{(N-p)/2} - [q_N]_p,

    the s terms present only when the index is a positive integer.  For N >= 4
    the first two of these rows come out identically zero: the moment
    hierarchy already forces R_N ~ 1/xi^2 + (3/2)/xi^4 for ANY coefficients,
    so those orders cost no degrees of freedom.  Null rows are dropped here.
    """
    qs = _q_polys(N)
    s = _zprime_asymptotic(N + n_rows + 2)

    def s_at(num: int) -> float:
        if num >= 2 and num % 2 == 0:
            m = num // 2
            return s[m] if m < len(s) else 0.0
        return 0.0

    rows, rhs = [], []
    p = N - 2
    while len(rows) < n_rows and p > -2 * (N + n_rows):
        row = np.array([s_at(i - p) - _coef(qs[i], p) for i in range(N)],
                       dtype=complex)
        val = s_at(N - p) - _coef(qs[N], p)
        if np.any(np.abs(row) > 1e-13) or abs(val) > 1e-13:
            rows.append(row)
            rhs.append(val)
        p -= 1
    if len(rows) < n_rows:
        raise RuntimeError(f"could not build {n_rows} asymptotic rows for N={N}")
    return np.array(rows), np.array(rhs)


def _q_polys(N: int) -> list[np.ndarray]:
    """q_0..q_N in increasing powers; q_0 = 0, q_{n+1} = xi q_n + n M_{n-1}."""
    qs = [np.zeros(1)]
    for n in range(N):
        prev = qs[-1]
        nxt = np.zeros(len(prev) + 1)
        nxt[1:] = prev
        nxt[0] += n * maxwellian_moment(n - 1)
        qs.append(nxt)
    return qs


def _coef(poly: np.ndarray, p: int) -> float:
    return float(poly[p]) if 0 <= p < len(poly) else 0.0


def two_point_coefficients(N: int, n_zero: int = 2) -> np.ndarray:
    """Closure coefficients from an (n_zero at 0, N - n_zero at infinity) match.

    This is the Hammett-Perkins / Hunana construction in its general form: fix
    some Taylor coefficients of the response at xi = 0 (which is what sets the
    Landau damping, since Im Z'(0) = -2 sqrt(pi) xi is the leading imaginary
    part) and spend the remaining freedom on the large-argument asymptotic
    series (which sets the fluid limit).

      two_point_coefficients(3, 2)  reproduces pade_coefficients(3), i.e. HP
                                    1990 with Gamma = 3, chi_1 = 2/sqrt(pi)
      two_point_coefficients(N, N)  reproduces pade_coefficients(N != 3), the
                                    pure Taylor-at-zero match
      two_point_coefficients(4, 2)  the direct N=4 analogue of HP's N=3 split

    Note that for N >= 4 the two leading asymptotic orders (1/xi^2 and
    3/(2 xi^4)) are automatic, so "N - n_zero asymptotic conditions" means
    N - n_zero conditions BEYOND those.
    """
    if not 0 <= n_zero <= N:
        raise ValueError(f"n_zero must be in [0, {N}], got {n_zero}")
    A0, b0 = _zero_rows(N, n_zero)
    if n_zero == N:
        return np.linalg.solve(A0, b0)
    Ai, bi = _inf_rows(N, N - n_zero)
    A = np.vstack([A0, Ai]) if n_zero else Ai
    b = np.concatenate([b0, bi]) if n_zero else bi
    return np.linalg.solve(A, b)
