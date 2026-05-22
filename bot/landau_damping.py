"""Application: Landau damping of Langmuir waves on a single warm Maxwellian.

Test of generalization. The NN closure (closures/naive_nn.py) was trained on
xi_b-sampled data for the BoT 3-moment closure. The training set is the
linear response of a Maxwellian -- no reference to u_b, eps, or k. So the
same checkpoint should work for any 3-moment fluid system over a Maxwellian
equilibrium, including pure-bulk Langmuir damping. This module confirms it
numerically.

Setup
-----
Strip the BoT system to a single warm Maxwellian:
    d_t E    = -U_1
    d_t U_0  = -ik U_1
    d_t U_1  = -ik U_2 + E          (M_0 = 1)
    d_t U_2  = -ik U_3                (M_1 = 0, closure provides U_3)

Kinetic Langmuir dispersion in BoT-paper normalization (Maxwellian variance
1/2, v_th = 1/sqrt(2), lambda_D = 1/sqrt(2)):
    k_code^2 = Z'(omega/k_code)
Converted to standard plasma-physics  k * lambda_D  by:
    k_lambdaD_std = k_code / sqrt(2)
i.e. k_code = sqrt(2) * k_lambdaD_std.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import wofz

from bot.closures.naive_nn import naive_predict_alpha
from bot.closures.pade import pade_coefficients


SQRT_PI = float(np.sqrt(np.pi))
SQRT2 = float(np.sqrt(2.0))


def Z(xi):
    """Plasma dispersion function. Complex input."""
    return 1j * SQRT_PI * wofz(xi)


def Zprime(xi):
    return -2.0 * (1.0 + xi * Z(xi))


def kinetic_omega(k):
    """Solve k^2 = Z'(omega/k) for the Langmuir mode with Landau damping.

    Newton iteration from a Bohm-Gross seed. Returns complex omega.
    `k` is in code units; convert with k_code = sqrt(2) * k*lambda_D_std.
    """
    omega = complex(np.sqrt(1.0 + 3.0 * k**2), -0.01)
    for _ in range(300):
        xi = omega / k
        f = k**2 - Zprime(xi)
        d = 1e-7
        fp = ((k**2 - Zprime((omega + d) / k)) - f) / d
        if abs(fp) < 1e-18:
            break
        domega = -f / fp
        omega += domega
        if abs(domega) < 1e-13:
            break
    return omega


def pade_matrix(k, a):
    """4x4 linear-evolution matrix for (E, U_0, U_1, U_2) with linear closure a."""
    A = np.zeros((4, 4), dtype=complex)
    A[0, 2] = -1.0
    A[1, 2] = -1j * k
    A[2, 0] = 1.0
    A[2, 3] = -1j * k
    A[3, 1] = -1j * k * a[0]
    A[3, 2] = -1j * k * a[1]
    A[3, 3] = -1j * k * a[2]
    return A


def langmuir_eigenvector(k, omega_kin):
    """(E, U_0, U_1, U_2) kinetic Langmuir eigenvector at the analytic omega."""
    xi = omega_kin / k
    Zp = Zprime(xi)
    U0 = 1.0 + 0j
    U1 = xi * U0
    U2 = (xi**2 - 1.0 / Zp) * U0
    # d_t E = -U_1 with E ~ exp(-i omega t)  =>  E = U_1 / (i omega)
    E = U1 / (1j * omega_kin)
    return np.array([E, U0, U1, U2], dtype=complex)


def _nn_rhs(t, y_real, k, model):
    y = y_real[:4] + 1j * y_real[4:]
    E, U0, U1, U2 = y
    if abs(U0) < 1e-15:
        U3 = 0.0 + 0.0j
    else:
        a_eff = complex(naive_predict_alpha(model, U1 / U0, U2 / U0))
        U3 = a_eff * U0
    dE = -U1
    dU0 = -1j * k * U1
    dU1 = -1j * k * U2 + E
    dU2 = -1j * k * U3
    dy = np.array([dE, dU0, dU1, dU2])
    return np.concatenate([dy.real, dy.imag])


def fit_omega(t, E, t_lo, t_hi):
    """Fit complex omega from log-magnitude slope + real-part zero crossings."""
    mask = (t >= t_lo) & (t <= t_hi)
    Esel = E[mask]; tsel = t[mask]
    floor = 1e-13 * max(np.abs(E).max(), 1e-30)
    finite = np.abs(Esel) > floor
    if finite.sum() < 50:
        return float('nan'), float('nan')
    Esel = Esel[finite]; tsel = tsel[finite]
    gamma = float(np.polyfit(tsel, np.log(np.abs(Esel)), 1)[0])
    sign = np.sign(Esel.real)
    cross = np.where(np.diff(sign) != 0)[0]
    if len(cross) >= 3:
        period = 2.0 * np.diff(tsel[cross]).mean()
        omega_re = 2.0 * np.pi / period
    else:
        omega_re = float('nan')
    return omega_re, gamma


def _time_grid(omega_kin):
    """t_end ~ a few e-folds and several oscillations."""
    g = omega_kin.imag
    t_decay = -4.0 / g if g < 0 else 40.0
    t_osc = 6 * 2 * np.pi / abs(omega_kin.real)
    t_end = min(max(t_decay, t_osc), 80.0)
    return np.linspace(0.0, t_end, 1001)


def nn_langmuir(k, omega_kin, model, amp=1e-3):
    """Time-domain NN-closed integration from a Langmuir-projected IC.

    Returns the fitted complex omega (omega_re, gamma).
    """
    v = langmuir_eigenvector(k, omega_kin)
    v *= amp / abs(v[0])
    t = _time_grid(omega_kin)
    y0r = np.concatenate([v.real, v.imag])
    sol = solve_ivp(_nn_rhs, (t[0], t[-1]), y0r, t_eval=t,
                    args=(k, model), method="RK45", rtol=1e-10, atol=1e-13)
    E = (sol.y[:4] + 1j * sol.y[4:])[0]
    t_lo, t_hi = 0.1 * t[-1], 0.9 * t[-1]
    om_re, gam = fit_omega(t, E, t_lo, t_hi)
    return complex(om_re, gam)


def pade_langmuir(k, a, omega_kin, amp=1e-3):
    """Time-domain Padé-closed integration from a Langmuir-projected IC.

    Linear system, integrated via matrix exponential for accuracy. Returns
    fitted complex omega.
    """
    v = langmuir_eigenvector(k, omega_kin)
    v *= amp / abs(v[0])
    t = _time_grid(omega_kin)
    A = pade_matrix(k, a)
    ev, V = np.linalg.eig(A)
    c = np.linalg.solve(V, v)
    E = np.array([(V @ (c * np.exp(ev * ti)))[0] for ti in t])
    t_lo, t_hi = 0.1 * t[-1], 0.9 * t[-1]
    om_re, gam = fit_omega(t, E, t_lo, t_hi)
    return complex(om_re, gam)


def sweep(kld_std, model, a_pade=None):
    """Sweep over standard k*lambda_D values; return arrays of (omega_kin,
    omega_nn, omega_pade) and the corresponding code-normalization k."""
    if a_pade is None:
        a_pade = pade_coefficients(3)
    ks_code = SQRT2 * kld_std
    n = len(ks_code)
    omega_kin = np.empty(n, dtype=complex)
    omega_nn = np.empty(n, dtype=complex)
    omega_pade = np.empty(n, dtype=complex)
    for i, k in enumerate(ks_code):
        omega_kin[i] = kinetic_omega(k)
        omega_nn[i] = nn_langmuir(k, omega_kin[i], model)
        omega_pade[i] = pade_langmuir(k, a_pade, omega_kin[i])
    return omega_kin, omega_nn, omega_pade, ks_code
