"""Kappa-distributed beam: exact response function and EKR closure.

The EKR closure needs only the kinetic response function R(zeta) of the
equilibrium, so it applies to any f_0 admitting an analytic continuation --
unlike Landau fluid closures, whose coefficients are derived from the
Maxwellian asymptotics of Z'.  This module supplies R for the 1D kappa
distribution, normalized to the same convention as the rest of the repo
(bot.closures.pade.maxwellian_moment): M_0 = 1, M_2 = 1/2, so that
v_t^2 = 2 n^-1 int (v-u)^2 f_0 dv as in the paper.

    F_kappa(s) = C_kappa (1 + s^2/b^2)^-(kappa+1),   b^2 = kappa - 1/2

    C_kappa = Gamma(kappa+1) / (b sqrt(pi) Gamma(kappa+1/2))

For integer kappa, F is rational with poles of order kappa+1 at s = +-i b, so

    D(zeta) = int ds F(s)/(s-zeta)^2 = -2 pi i Res_{s=-ib}[F(s)/(s-zeta)^2]

(closing in the lower half-plane for Im zeta > 0; the residue formula then
provides the Landau analytic continuation to Im zeta < 0 for free).  The
response in the paper's sign convention is

    R_kappa(zeta) = -D(zeta)/2,

reducing to R = -Z'/2 = 1 + zeta Z(zeta) as kappa -> infinity.

Note the moment hierarchy itself only requires M_0 = 1 and M_1 = 0 to close
at N = 2, so the EKR closure is well defined even for small kappa where
M_n diverges for n >= 2 kappa + 1 -- a regime where increasing the order of
an asymptotically matched Pade closure is not an option.
"""

from __future__ import annotations

from math import comb, factorial

import numpy as np
from scipy.special import gammaln


def kappa_b(kappa: float) -> float:
    """Width parameter b with M_2 = 1/2 (matches the Maxwellian convention)."""
    if kappa <= 0.5:
        raise ValueError("kappa > 1/2 required for a finite second moment")
    return np.sqrt(kappa - 0.5)


def kappa_norm(kappa: float) -> float:
    """C_kappa such that int F_kappa ds = 1."""
    b = kappa_b(kappa)
    log_c = (gammaln(kappa + 1.0) - gammaln(kappa + 0.5)
             - np.log(b) - 0.5 * np.log(np.pi))
    return float(np.exp(log_c))


def F_kappa(s, kappa: float):
    """Normalized 1D kappa distribution on the beam-frame velocity grid."""
    b = kappa_b(kappa)
    return kappa_norm(kappa) * (1.0 + (s / b) ** 2) ** (-(kappa + 1.0))


def dF_kappa(s, kappa: float):
    """d F_kappa / ds."""
    b = kappa_b(kappa)
    return (kappa_norm(kappa) * (-(kappa + 1.0))
            * (1.0 + (s / b) ** 2) ** (-(kappa + 2.0)) * (2.0 * s / b**2))


def _scaled_residue(zeta: complex, kappa: int, power: int) -> complex:
    """b^(2k+2) * Res_{s=-ib}[ (s-ib)^-(k+1) (s-zeta)^-power ].

    The b powers are combined term-by-term with the (s-ib)^-(k+1+j) factors
    (which carry b^-(k+1+j)), leaving b^(k+1-j) per term -- the unscaled
    prefactor b^(2k+2) overflows for kappa beyond ~30 while the residue
    itself underflows, so the product must be formed inside the sum.
    """
    k = int(kappa)
    b = kappa_b(kappa)
    d2 = -1j * b - zeta            # (s - zeta) at s = -ib
    total = 0.0 + 0.0j
    for j in range(k + 1):
        # d^j (s-ib)^-(k+1): coefficient (-1)^j (k+1)...(k+j), phase from -2i
        c1 = ((-1.0) ** j) * float(np.prod([float(k + 1 + m)
                                            for m in range(j)])) if j else 1.0
        t1 = c1 * (-2j) ** (-(k + 1 + j)) * b ** (k + 1 - j)
        # d^m (s-zeta)^-power = (-1)^m (power)...(power+m-1) (s-zeta)^-(power+m)
        m = k - j
        c2 = ((-1.0) ** m) * float(np.prod([float(power + q)
                                            for q in range(m)])) if m else 1.0
        t2 = c2 * d2 ** (-(power + m))
        total += comb(k, j) * t1 * t2
    return total / factorial(k)


def R_kappa(zeta, kappa: int):
    """Kinetic response R = -U_0/Phi_hat for a kappa-distributed equilibrium.

    Valid in both half-planes: the residue form IS the Landau continuation.
    """
    C = kappa_norm(kappa)
    zeta = np.asarray(zeta, dtype=complex)
    scalar = zeta.ndim == 0
    flat = np.atleast_1d(zeta)
    out = np.empty(flat.shape, dtype=complex)
    for idx, z in np.ndenumerate(flat):
        D = -2j * np.pi * C * _scaled_residue(z, kappa, power=2)
        out[idx] = -0.5 * D
    return complex(out.ravel()[0]) if scalar else out


def beta_kappa(zeta, kappa: int):
    """EKR closure ratio U_2/U_0 = zeta^2 + 1/(2 R_kappa(zeta))."""
    return np.asarray(zeta, dtype=complex) ** 2 + 1.0 / (2.0 * R_kappa(zeta, kappa))
