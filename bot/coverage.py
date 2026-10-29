"""Training-domain coverage diagnostics for Figure 3.

Instantaneous zeta=U_1/U_0 is evaluated on kinetic beam states, excluding t=0
where both moments vanish. The quadrature spacing cancels in this ratio.
"""

import numpy as np
from scipy.optimize import newton

from bot.closures.response import Z
from bot.kinetic import kinetic_system_sspace, most_unstable_kinetic, s_grid
from bot.modes import damped_pair_candidates
from bot.paths import DATA

RE_RANGE, IM_RANGE = (-3.0, 3.0), (-1.0, 1.5)
NPZ_SCAN = DATA / "growth_scan.npz"
OMEGA_RE_HI = 1.5
TWOMODE_CASES = ((1.0, 0.50, 0.05), (1.5, 0.40, 0.05))
K_LAMBDA_D = 0.4


def zeta_b(omega: complex, k: float, u_b: float) -> complex:
    return (omega - k * u_b) / k


def in_rectangle(z):
    return (
        (z.real > RE_RANGE[0])
        & (z.real < RE_RANGE[1])
        & (z.imag > IM_RANGE[0])
        & (z.imag < IM_RANGE[1])
    )


def scan_modes() -> np.ndarray:
    d = np.load(NPZ_SCAN)
    out = []
    for i, u_b in enumerate(d["u_b_vals"]):
        for j, eps in enumerate(d["eps_vals"]):
            k = float(d["k_star"][i, j])
            om = most_unstable_kinetic(k, u_b, eps, N_v=96, V=6.0, omega_re_hi=OMEGA_RE_HI)
            out.append(zeta_b(om, k, u_b))
    return np.array(out)


def scan_trajectories(
    t_end_min: float = 60.0,
    t_fit_min: float = 25.0,
    n_efold: float = 5.0,
    dt: float = 0.1,
    amp: float = 1e-3,
) -> list[np.ndarray]:
    d = np.load(NPZ_SCAN)
    s, _ = s_grid(96, 6.0)
    out = []
    for i, u_b in enumerate(d["u_b_vals"]):
        for j, eps in enumerate(d["eps_vals"]):
            k = float(d["k_star"][i, j])
            om = most_unstable_kinetic(k, u_b, eps, N_v=96, V=6.0, omega_re_hi=OMEGA_RE_HI)
            t_end = max(t_end_min, t_fit_min + n_efold / om.imag)
            t = np.arange(1, int(t_end / dt) + 1) * dt
            lam, vec = np.linalg.eig(kinetic_system_sspace(k, u_b, eps, 96, 6.0))
            y0 = np.zeros(len(lam), dtype=complex)
            y0[1] = amp
            F = (vec @ (np.linalg.solve(vec, y0)[:, None] * np.exp(lam[:, None] * t[None, :])))[2:]
            out.append((s[:, None] * F).sum(0) / F.sum(0))
    return out


def twomode_modes() -> np.ndarray:
    out = []
    for u_b, k, eps in TWOMODE_CASES:
        out += [zeta_b(w, k, u_b) for w in damped_pair_candidates(k, u_b, eps)[:2]]
    return np.array(out)


def standing_wave_roots() -> np.ndarray:
    K = K_LAMBDA_D * np.sqrt(2.0)
    z = newton(lambda z: 1.0 + (2.0 / K**2) * (1.0 + z * Z(z)), 2.3 - 0.1j, tol=1e-12)
    return np.array([z, -np.conj(z)])
