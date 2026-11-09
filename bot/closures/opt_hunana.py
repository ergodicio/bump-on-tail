"""Learned Padé coefficients fitted at the NN training frequencies.

The fit minimizes relative single-mode U_N error with parity-constrained
coefficients. Cached coefficients below are exactly those used in the paper.
"""

import numpy as np

from bot.closures.pade import maxwellian_moment
from bot.closures.response import Zprime


def q_polynomials(N: int) -> list[np.ndarray]:
    """q_0..q_N as coefficient arrays in INCREASING powers of xi.

    q_0 = 0,  q_{n+1} = xi q_n + n M_{n-1}.
    """
    qs = [np.zeros(1)]
    for n in range(N):
        prev = qs[-1]
        nxt = np.zeros(len(prev) + 1)
        nxt[1:] = prev  # xi * q_n
        nxt[0] += n * maxwellian_moment(n - 1)
        qs.append(nxt)
    return qs


def _polyval_inc(coeffs: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Evaluate a polynomial given in increasing-power order."""
    return np.polyval(coeffs[::-1].astype(complex), xi)


def alpha_basis(xi, N: int) -> tuple[np.ndarray, np.ndarray]:
    """(B, y): B[:, i] = alpha_i(xi) for i < N, y = alpha_N(xi).

    alpha_n(xi) = xi^n - q_n(xi)/Z'(xi) is the exact U_n/U_0 on the kinetic
    eigenmode manifold: alpha_0 = 1, alpha_1 = xi, alpha_2 = beta (u2.beta),
    alpha_3 = alpha (inference.alpha), alpha_4 = alpha4 (inference.alpha4).
    """
    xi = np.asarray(xi, dtype=complex)
    qs = q_polynomials(N)
    Zp = Zprime(xi)
    B = np.stack([xi**i - _polyval_inc(qs[i], xi) / Zp for i in range(N)], axis=-1)
    y = xi**N - _polyval_inc(qs[N], xi) / Zp
    return B, y


def parity_mask(N: int) -> np.ndarray:
    """Unit multipliers m_i with a_i = t_i * m_i, t_i real: 1 or 1j."""
    return np.array([1.0 + 0j if (N - i) % 2 == 0 else 1j for i in range(N)])


PAPER_COEFFICIENTS = {
    3: np.array([0.9093528338j, 2.0264692241, -1.7922778048j]),
    4: np.array([-1.9743053346, 4.9642714024j, 5.3063482891, -2.9500004238j]),
}


def opt_coefficients(order):
    """Copy of the published coefficient vector; U_N=a dot U."""
    return PAPER_COEFFICIENTS[order].copy()


def fit_coefficients(order, frequencies):
    """Relative-error real least squares with the Maxwellian parity symmetry."""
    basis, target = alpha_basis(frequencies, order)
    if np.any(abs(target) == 0):
        raise ValueError("Relative loss is undefined at a zero target")
    weight = 1 / abs(target)
    parity = parity_mask(order)
    design = basis * weight[:, None] * parity
    design = np.concatenate((design.real, design.imag))
    rhs = target * weight
    parameters = np.linalg.lstsq(design, np.concatenate((rhs.real, rhs.imag)), rcond=None)[0]
    return parameters * parity
