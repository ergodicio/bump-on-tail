"""Driver: slab-ITG time-domain traces |U_0(t)| — kinetic DKE vs fluid closures.

Generates bot/figures/fig_slab_itg_timedomain.png and
bot/runs/slab_itg_timedomain.npz.

Four eta panels spanning stable / near-threshold / unstable; all systems start
from the same generic density perturbation (g0 = amp*F0, i.e. U_0 = amp,
U_1 = 0, U_2 = amp/2).  Dashed guide shows exp(gamma_kin t) from the
DKE-derived dispersion root.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bot.slab_itg import (D_kinetic, max_growing_root, kinetic_evolve,
                          fluid_hp_evolve, direct_beta_evolve,
                          direct_alpha_evolve, density_ic_kinetic,
                          density_ic_fluid, eta_threshold)

ZETA_STAR = 1.0
TAU = 1.0
N_W = 600
AMP = 1e-3
ETAS = [2.5, 3.5, 4.5, 6.0]
SAVE_PNG = "bot/figures/fig_slab_itg_timedomain.png"
SAVE_NPZ = "bot/runs/slab_itg_timedomain.npz"


def run():
    t = np.linspace(0.0, 60.0, 601)
    g0 = density_ic_kinetic(amp=AMP, N_w=N_W)
    y2 = density_ic_fluid(amp=AMP, N=2)
    y3 = density_ic_fluid(amp=AMP, N=3)
    out = {"t": t, "etas": np.array(ETAS)}
    for e in ETAS:
        print(f"eta = {e} ...")
        out[f"kin_{e}"] = kinetic_evolve(ZETA_STAR, e, TAU, t, g0, N_w=N_W)["U0"]
        out[f"hp_{e}"] = fluid_hp_evolve(ZETA_STAR, e, TAU, t, y3)["U0"]
        out[f"beta_{e}"] = direct_beta_evolve(ZETA_STAR, e, TAU, t, y2)["U0"]
        out[f"alpha_{e}"] = direct_alpha_evolve(ZETA_STAR, e, TAU, t, y3)["U0"]
        out[f"zkin_{e}"] = max_growing_root(D_kinetic, ZETA_STAR, e, TAU)
    return out


def make_fig(r):
    t = r["t"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for ax, e in zip(axes.flat, ETAS):
        ax.semilogy(t, np.abs(r[f"kin_{e}"]), "k-", lw=2.2, label="kinetic DKE")
        ax.semilogy(t, np.abs(r[f"hp_{e}"]), "C1--", lw=1.8,
                    label=r"HP fluid ($\Gamma=3$)")
        ax.semilogy(t, np.abs(r[f"beta_{e}"]), "C0-.", lw=1.6,
                    label=r"direct $\beta$ (N=2)")
        ax.semilogy(t, np.abs(r[f"alpha_{e}"]), "C3:", lw=1.8,
                    label=r"direct $\alpha$ (N=3)")
        zk = complex(r[f"zkin_{e}"])
        if not np.isnan(zk.real) and zk.imag > 0:
            guide = np.abs(r[f"kin_{e}"])[len(t) // 3] * \
                np.exp(zk.imag * (t - t[len(t) // 3]))
            ax.semilogy(t, guide, color="0.5", ls="--", lw=1.0,
                        label=rf"$e^{{\gamma_{{kin}} t}}$, $\gamma={zk.imag:.3f}$")
        ax.set_title(rf"$\eta_i = {e}$", fontsize=12)
        ax.set_ylabel(r"$|\tilde n_i / n_0|$", fontsize=11)
        ax.set_ylim(1e-6, None)
        ax.legend(fontsize=7.5, loc="upper left")
    for ax in axes[1]:
        ax.set_xlabel(r"$t \, |k_\parallel| v_{ti} \sqrt{2}$", fontsize=11)
    eta_th = eta_threshold(ZETA_STAR, TAU)
    fig.suptitle(rf"Slab ITG initial-value evolution, density-perturbation IC"
                 rf"  ($\zeta_*={ZETA_STAR}$, $\tau={TAU}$; "
                 rf"kinetic $\eta_{{th}}={eta_th:.3f}$)", fontsize=12)
    plt.tight_layout()
    plt.savefig(SAVE_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {SAVE_PNG}")


if __name__ == "__main__":
    r = run()
    np.savez_compressed(SAVE_NPZ, **{k: v for k, v in r.items()})
    print(f"Saved {SAVE_NPZ}")
    make_fig(r)
