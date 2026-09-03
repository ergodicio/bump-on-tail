"""Driver: slab-ITG growth rate / frequency vs eta — kinetic vs fluid closures.

Generates bot/figures/fig_slab_itg_growth.png and bot/runs/slab_itg_growth.npz.

Curves: DKE-derived kinetic dispersion root (bot/slab_itg.D_kinetic) and the
HP 3-moment fluid eigenvalue (Gamma = 3).  Markers: time-domain fits of
U_0(t) from initial-value solves (kinetic DKE, HP fluid, EKR N=2) started
from a generic density perturbation — the rigorous test that the
initial-value problem actually selects these roots.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bot.slab_itg import (D_kinetic, max_growing_root, eta_threshold,
                          fluid_hp_modes, kinetic_evolve, fluid_hp_evolve,
                          direct_beta_evolve,
                          density_ic_kinetic, density_ic_fluid, fit_mode)

ZETA_STAR = 1.0
TAU = 1.0
N_W = 600
SAVE_PNG = "bot/figures/fig_slab_itg_growth.png"
SAVE_NPZ = "bot/runs/slab_itg_growth.npz"


def scan(eta_curve, eta_fit):
    out = {"eta_curve": eta_curve, "eta_fit": eta_fit}

    print("dispersion-root curves...")
    out["z_kin"] = np.array([max_growing_root(D_kinetic, ZETA_STAR, e, TAU)
                             for e in eta_curve])
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
        rb = direct_beta_evolve(ZETA_STAR, e, TAU, t, y2, variant="itg")
        fits["beta"].append(fit_mode(rb["t"], rb["U0"], frac=(0.5, 1.0)))
        print(f"  eta={e:4.2f}  kin={fits['kin'][-1]:.4f}  "
              f"hp={fits['hp'][-1]:.4f}  beta={fits['beta'][-1]:.4f}")
    for k, v in fits.items():
        out[f"fit_{k}"] = np.array(v)
    return out


def make_fig(r):
    eta_c, eta_f = r["eta_curve"], r["eta_fit"]
    eta_th = eta_threshold(ZETA_STAR, TAU)

    matplotlib.rcParams.update({"font.family": "serif",
                                "mathtext.fontset": "cm",
                                "font.size": 16, "axes.titlesize": 16})
    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    def gr(z):
        return np.where(np.isnan(z.real), 0.0, np.maximum(z.imag, 0.0))

    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(eta_th, color="gray", lw=0.8, ls="--",
               label=rf"$\eta_{{th}} = 1+\sqrt{{5}} \approx {eta_th:.3f}$")
    ax.plot(eta_c, gr(r["z_kin"]), "k-", lw=2.5, label="Kinetic ground truth")
    ax.plot(eta_c, gr(r["z_hp"]), "C1-", lw=1.8,
            label=r"Hammett-Perkins ($N=3$)")
    ax.plot(eta_f, np.maximum(r["fit_hp"].imag, 0), "C1s", ms=6, mfc="none",
            label=r"Hammett-Perkins ($N=3$), time domain")
    ax.plot(eta_f, np.maximum(r["fit_beta"].imag, 0), "C0^", ms=6,
            label=r"EKR ($N=2$), time domain")
    ax.set_xlabel(r"$\eta = L_n/L_T$")
    ax.set_ylabel(r"$\gamma / (|k_\parallel| v_{ti})$")
    ax.legend(fontsize=11, loc="upper left")

    plt.tight_layout()
    plt.savefig(SAVE_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {SAVE_PNG}")


if __name__ == "__main__":
    eta_curve = np.linspace(0.5, 6.5, 61)
    eta_fit = np.array([1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])
    r = scan(eta_curve, eta_fit)
    np.savez_compressed(SAVE_NPZ, **r)
    print(f"Saved {SAVE_NPZ}")
    make_fig(r)
    eta_th = eta_threshold(ZETA_STAR, TAU)
    print(f"\nanalytic kinetic threshold: eta_th = {eta_th:.6f}")
