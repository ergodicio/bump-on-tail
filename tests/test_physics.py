"""Physics checks independent of figure styling and training implementation."""

import numpy as np
from scipy.integrate import quad

from bot.closed_loop import _direct_u2_rhs, fluid_eigenmode_ic
from bot.closures.kappa import F_kappa, R_kappa
from bot.closures.pade import maxwellian_moment, pade_coefficients, two_point_coefficients
from bot.closures.response import R, Zprime, alpha, alpha4, beta, mode_ratios
from bot.kinetic import kinetic_dispersion, most_unstable_kinetic
from bot.slab_itg import D_kinetic, eta_threshold
from bot.standing_wave import landau_root


def test_response_and_moment_recurrence():
    z = 0.7 - 0.2j
    assert abs(Zprime(z) + 2 * R(z)) < 1e-14
    np.testing.assert_allclose(
        mode_ratios(z, 4), [1, z, beta(z), alpha(z), alpha4(z)], rtol=1e-14, atol=1e-14
    )
    assert maxwellian_moment(2) == 0.5
    assert maxwellian_moment(4) == 0.75
    np.testing.assert_allclose(two_point_coefficients(3, 2), pade_coefficients(3), atol=1e-14)


def test_kappa_normalization_and_temperature():
    for kappa in (2, 4):
        assert abs(quad(lambda s: F_kappa(s, kappa), -np.inf, np.inf)[0] - 1) < 1e-10
        assert abs(quad(lambda s: s * s * F_kappa(s, kappa), -np.inf, np.inf)[0] - 0.5) < 1e-10
    z = 0.4 + 0.3j
    assert abs(R_kappa(z, 40) - R(z)) < abs(R_kappa(z, 20) - R(z))


def test_ekr_supports_kinetic_eigenmode():
    k, u_b, eps = 0.295, 4.0, 0.02
    omega = most_unstable_kinetic(k, u_b, eps)
    assert abs(kinetic_dispersion(omega, k, u_b, eps)) < 1e-10
    initial = fluid_eigenmode_ic(k, u_b, omega, 2)
    rhs = _direct_u2_rhs(0.0, np.concatenate((initial.real, initial.imag)), k, u_b, eps)
    derivative = rhs[:4] + 1j * rhs[4:]
    np.testing.assert_allclose(derivative, -1j * omega * initial, rtol=1e-9, atol=1e-12)


def test_independent_validation_roots():
    threshold = eta_threshold(1.0, 1.0)
    assert abs(threshold - (1 + np.sqrt(5))) < 1e-14
    # On the real marginal root, 1 - eta*zeta*zeta_star = -tau.
    marginal = 2.0 / threshold
    assert abs(D_kinetic(marginal, 1.0, threshold, 1.0)) < 1e-12
    root = landau_root()
    assert abs(root - (1.2851 - 0.0661j)) < 1e-4
