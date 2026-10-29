"""Maxwellian kinetic response and exact single-mode moment ratios.

Z is the Landau continuation supplied by scipy's Faddeeva function.
R=1+zeta*Z=-Z'/2. Ratios can have poles and do not define a globally
holomorphic closure on moment space; the numerical ratio guard is explicit.
"""

import numpy as np
from scipy.special import wofz


def Z(zeta):
    return 1j * np.sqrt(np.pi) * wofz(zeta)


def Zprime(zeta):
    return -2.0 * (1.0 + zeta * Z(zeta))


def R(zeta):
    return 1.0 + zeta * Z(zeta)


def beta(zeta):
    """U_2/U_0 on one eigenmode."""
    return zeta**2 - 1.0 / Zprime(zeta)


def alpha(zeta):
    """U_3/U_0 on one eigenmode."""
    return zeta**3 - zeta / Zprime(zeta)


def alpha4(zeta):
    """U_4/U_0 on one eigenmode."""
    return zeta**4 - (zeta**2 + 1.5) / Zprime(zeta)


def mode_ratios(zeta, order):
    """Return U_0/U_0 through U_order/U_0 using the moment recurrence."""
    from .pade import maxwellian_moment

    ratios = [1.0 + 0j]
    for n in range(order):
        ratios.append(zeta * ratios[-1] - n * maxwellian_moment(n - 1) / Zprime(zeta))
    return np.array(ratios)
