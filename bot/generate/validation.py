"""Generate the kappa and ITG validation data plotted in Figure 1."""

import numpy as np

from bot.closed_loop import pade_evolve
from bot.closures.pade import pade_coefficients
from bot.kappa_bot import ekr_evolve_kappa, find_root_kappa, kinetic_evolve_kappa
from bot.kinetic import most_unstable_kinetic
from bot.paths import DATA
from bot.slab_itg import (
    D_kinetic,
    density_ic_fluid,
    density_ic_kinetic,
    direct_beta_evolve,
    fit_mode,
    fluid_hp_evolve,
    fluid_hp_modes,
    kinetic_evolve,
    max_growing_root,
)

U_B, EPS, K = 4.0, 0.02, 0.295
AMP, T_END, N_V, V = 1e-3, 110.0, 500, 25.0
KAPPA_SCAN = (2, 3, 4, 6, 8, 12, 20, 40)
KAPPA_TRACE = 2
ZETA_STAR, TAU, N_W = 1.0, 1.0, 600


def fit_gamma(t, E, frac=(0.5, 1.0)):
    m = (t >= frac[0] * t[-1]) & (t <= frac[1] * t[-1])
    return np.polyfit(t[m], np.log(np.abs(E[m])), 1)[0]


def kappa_data(output):
    t = np.linspace(0.0, T_END, 1101)
    a_hp = pade_coefficients(3)
    y0_hp = np.array([0.0, AMP, 0.0, 0.0, 0.0], dtype=complex)
    E_hp = pade_evolve(K, U_B, EPS, a_hp, t, y0_hp, method="rk45")
    g_hp = fit_gamma(t, E_hp)
    w_maxw = most_unstable_kinetic(K, U_B, EPS)

    g_kin, g_ekr = [], []
    for kappa in KAPPA_SCAN:
        w = find_root_kappa(K, U_B, EPS, kappa, 1.0 + 0.05j)
        y0_ekr = np.array([0.0, AMP, 0.0, 0.0], dtype=complex)
        ge = fit_gamma(t, ekr_evolve_kappa(K, U_B, EPS, kappa, t, y0_ekr))
        g_kin.append(w.imag)
        g_ekr.append(ge)
        print(
            f"  kappa={kappa:>3}: kin={w.imag:.6f}  EKR={ge:.6f} "
            f"({100 * (ge / w.imag - 1):+.2g}%)  HP={100 * (g_hp / w.imag - 1):+.1f}%"
        )

    y0_kin = np.zeros(2 + N_V, dtype=complex)
    y0_kin[1] = AMP
    E_kin = kinetic_evolve_kappa(K, U_B, EPS, KAPPA_TRACE, t, y0_kin, N_v=N_V, V=V)
    y0_ekr = np.array([0.0, AMP, 0.0, 0.0], dtype=complex)
    E_ekr2 = ekr_evolve_kappa(K, U_B, EPS, KAPPA_TRACE, t, y0_ekr)

    d = dict(
        t=t,
        E_kin=E_kin,
        E_hp=E_hp,
        E_ekr2=E_ekr2,
        g_hp=g_hp,
        g_maxw=w_maxw.imag,
        g_kin=np.array(g_kin),
        g_ekr=np.array(g_ekr),
    )
    np.savez_compressed(output / "validation_kappa.npz", **d)
    return d


def scan(eta_curve, eta_fit):
    out = {"eta_curve": eta_curve, "eta_fit": eta_fit}

    print("dispersion-root curves...")
    out["z_kin"] = np.array([max_growing_root(D_kinetic, ZETA_STAR, e, TAU) for e in eta_curve])
    zs = []
    for e in eta_curve:
        m = fluid_hp_modes(ZETA_STAR, e, TAU, Gamma=3.0)
        zs.append(m[int(np.argmax(m.imag))])
    out["z_hp"] = np.array(zs)

    print("time-domain fits...")
    t = np.linspace(0.0, 70.0, 701)
    g0 = density_ic_kinetic(N_w=N_W)
    y2 = density_ic_fluid(N=2)
    y3 = density_ic_fluid(N=3)
    fits = {k: [] for k in ("kin", "hp", "beta")}
    for e in eta_fit:
        rk = kinetic_evolve(ZETA_STAR, e, TAU, t, g0, N_w=N_W)
        fits["kin"].append(fit_mode(rk["t"], rk["U0"], frac=(0.5, 1.0)))
        rh = fluid_hp_evolve(ZETA_STAR, e, TAU, t, y3)
        fits["hp"].append(fit_mode(rh["t"], rh["U0"], frac=(0.5, 1.0)))
        rb = direct_beta_evolve(ZETA_STAR, e, TAU, t, y2)
        fits["beta"].append(fit_mode(rb["t"], rb["U0"], frac=(0.5, 1.0)))
        print(
            f"  eta={e:4.2f}  kin={fits['kin'][-1]:.4f}  "
            f"hp={fits['hp'][-1]:.4f}  beta={fits['beta'][-1]:.4f}"
        )
    for k, v in fits.items():
        out[f"fit_{k}"] = np.array(v)
    return out


def main(output):
    kappa_data(output)
    time = np.linspace(0.0, 25.0, 501)
    kin = kinetic_evolve(1.0, 10.0, 1.0, time, density_ic_kinetic(N_w=N_W), N_w=N_W)["U0"]
    hp = fluid_hp_evolve(1.0, 10.0, 1.0, time, density_ic_fluid(N=3), method="rk45")["U0"]
    ekr = direct_beta_evolve(1.0, 10.0, 1.0, time, density_ic_fluid(N=2))["U0"]
    np.savez_compressed(
        output / "validation_itg_trace.npz",
        t_itg1=time,
        itg1_params=[1.0, 1.0, 10.0],
        itg1_kin=kin,
        itg1_hp=hp,
        itg1_beta=ekr,
        itg1_zeta_kin=max_growing_root(D_kinetic, 1.0, 10.0, 1.0, im_hi=6.0),
    )
    with np.load(DATA / "validation_itg_scan.npz") as original:
        result = scan(original["eta_curve"], original["eta_fit"])
    np.savez_compressed(output / "validation_itg_scan.npz", **result)
