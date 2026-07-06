"""Driver: slab-ITG growth rate / frequency vs eta — kinetic vs fluid closures.

Generates bot/figures/fig_slab_itg_growth.png and bot/runs/slab_itg_growth.npz.

Curves: DKE-derived kinetic dispersion root (bot/slab_itg.D_kinetic), the
legacy itg_test.py variant (missing-zeta form, for reference), and the HP
3-moment fluid eigenvalue (Gamma = 3 and 5/3).  Markers: time-domain fits of
U_0(t) from initial-value solves (kinetic DKE, HP fluid, direct beta N=2,
direct alpha N=3) started from a generic density perturbation — the rigorous
test that the initial-value problem actually selects these roots.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bot.slab_itg import (D_kinetic, D_legacy, max_growing_root, eta_threshold,
                          fluid_hp_modes, kinetic_evolve, fluid_hp_evolve,
                          direct_beta_evolve, direct_alpha_evolve,
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
    out["z_leg"] = np.array([max_growing_root(D_legacy, ZETA_STAR, e, TAU)
                             for e in eta_curve])
    for name, Gamma in [("z_hp", 3.0), ("z_brag", 5.0 / 3.0)]:
        zs = []
        for e in eta_curve:
            m = fluid_hp_modes(ZETA_STAR, e, TAU, Gamma=Gamma)
            zs.append(m[int(np.argmax(m.imag))])
        out[name] = np.array(zs)

    print("time-domain fits...")
    t = np.linspace(0.0, 70.0, 701)
    g0 = density_ic_kinetic(N_w=N_W)
    y2 = density_ic_fluid(N=2)
    y3 = density_ic_fluid(N=3)
    fits = {k: [] for k in ("kin", "hp", "beta", "alpha")}
    for e in eta_fit:
        rk = kinetic_evolve(ZETA_STAR, e, TAU, t, g0, N_w=N_W)
        fits["kin"].append(fit_mode(rk["t"], rk["U0"], frac=(0.5, 1.0)))
        rh = fluid_hp_evolve(ZETA_STAR, e, TAU, t, y3)
        fits["hp"].append(fit_mode(rh["t"], rh["U0"], frac=(0.5, 1.0)))
        rb = direct_beta_evolve(ZETA_STAR, e, TAU, t, y2)
        fits["beta"].append(fit_mode(rb["t"], rb["U0"], frac=(0.5, 1.0)))
        ra = direct_alpha_evolve(ZETA_STAR, e, TAU, t, y3)
        fits["alpha"].append(fit_mode(ra["t"], ra["U0"], frac=(0.5, 1.0)))
        print(f"  eta={e:4.2f}  kin={fits['kin'][-1]:.4f}  "
              f"hp={fits['hp'][-1]:.4f}  beta={fits['beta'][-1]:.4f}  "
              f"alpha={fits['alpha'][-1]:.4f}")
    for k, v in fits.items():
        out[f"fit_{k}"] = np.array(v)
    return out


def make_fig(r):
    eta_c, eta_f = r["eta_curve"], r["eta_fit"]
    eta_th = eta_threshold(ZETA_STAR, TAU)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    def gr(z):
        return np.where(np.isnan(z.real), 0.0, np.maximum(z.imag, 0.0))

    ax = axes[0]
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(eta_th, color="gray", lw=0.8, ls="--",
               label=rf"$\eta_{{th}} = 1+\sqrt{{5}} \approx {eta_th:.3f}$")
    ax.axvline(2.0, color="gray", lw=0.8, ls=":",
               label=r"$\eta = 2$ (legacy claim)")
    ax.plot(eta_c, gr(r["z_kin"]), "k-", lw=2.5, label="kinetic (DKE dispersion)")
    ax.plot(eta_c, gr(r["z_leg"]), color="0.6", ls="--", lw=1.5,
            label=r"legacy $R_s$ (itg_test, missing $\zeta$)")
    ax.plot(eta_c, gr(r["z_hp"]), "C1-", lw=1.8, label=r"HP fluid ($\Gamma=3$)")
    ax.plot(eta_c, gr(r["z_brag"]), "C2:", lw=1.5, label=r"HP fluid ($\Gamma=5/3$)")
    ax.plot(eta_f, np.maximum(r["fit_kin"].imag, 0), "ko", ms=6, mfc="none",
            label="kinetic time-domain fit")
    ax.plot(eta_f, np.maximum(r["fit_hp"].imag, 0), "C1s", ms=5, mfc="none",
            label="HP time-domain fit")
    ax.plot(eta_f, np.maximum(r["fit_beta"].imag, 0), "C0^", ms=5,
            label=r"direct $\beta$ (N=2) fit")
    ax.plot(eta_f, np.maximum(r["fit_alpha"].imag, 0), "C3v", ms=5,
            label=r"direct $\alpha$ (N=3) fit")
    ax.set_xlabel(r"$\eta_i = L_n/L_T$", fontsize=12)
    ax.set_ylabel(r"$\gamma / (|k_\parallel| v_{ti}\sqrt{2})$", fontsize=11)
    ax.set_title("Growth rate", fontsize=12)
    ax.legend(fontsize=7.5, loc="upper left")

    ax = axes[1]
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(eta_th, color="gray", lw=0.8, ls="--")
    for z, sty, lab in [(r["z_kin"], dict(c="k", ls="-", lw=2.5), "kinetic"),
                        (r["z_leg"], dict(c="0.6", ls="--", lw=1.5), "legacy"),
                        (r["z_hp"], dict(c="C1", ls="-", lw=1.8), "HP fluid")]:
        m = ~np.isnan(z.real)
        ax.plot(eta_c[m], z.real[m], **sty, label=lab)
    ax.plot(eta_f, r["fit_kin"].real, "ko", ms=6, mfc="none")
    ax.plot(eta_f, r["fit_hp"].real, "C1s", ms=5, mfc="none")
    ax.plot(eta_f, r["fit_beta"].real, "C0^", ms=5)
    ax.plot(eta_f, r["fit_alpha"].real, "C3v", ms=5)
    ax.set_xlabel(r"$\eta_i = L_n/L_T$", fontsize=12)
    ax.set_ylabel(r"$\omega_r / (|k_\parallel| v_{ti}\sqrt{2})$", fontsize=11)
    ax.set_title("Real frequency (most-unstable / fitted mode)", fontsize=12)
    ax.legend(fontsize=8)

    fig.suptitle(rf"Slab ITG, time-domain tests  ($\zeta_*={ZETA_STAR}$, "
                 rf"$\tau=T_i/T_e={TAU}$)", fontsize=12)
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
