"""Driver: combined slab-ITG figure -- time trace + growth rate versus eta.

Generates bot/figures/fig_slab_itg_combined.png.

Panel (a): |U_{0,i}|(t) for the ITG case zeta_* = 1, tau = 1, eta = 10 from a
generic initial condition (data from bot/runs/bot_vs_itg_closures.npz, the
ITG half of the former fig_bot_vs_itg_closures_scaled).
Panel (b): growth rate versus eta at zeta_* = 1, tau = 1 (data from
bot/runs/slab_itg_growth.npz, the former fig_slab_itg_growth) -- curves are
the kinetic dispersion root and the HP fluid eigenvalue, markers are
time-domain fits from a generic density perturbation.

Re-run bot/figs/fig_bot_vs_itg_closures.py and bot/figs/fig_slab_itg_growth.py
first if either npz is missing.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bot.slab_itg import eta_threshold, fit_mode

NPZ_TRACE = "bot/runs/bot_vs_itg_closures.npz"
NPZ_SCAN = "bot/runs/slab_itg_growth.npz"
SAVE_PNG = "bot/figures/fig_slab_itg_combined.png"

ZETA_STAR, TAU = 1.0, 1.0

RC = {"font.family": "serif", "mathtext.fontset": "cm",
      "font.size": 7, "axes.titlesize": 7}

STYLES = [("kin", dict(color="k", ls="none", marker="o", ms=1.3,
                       markevery=20, zorder=10), "Kinetic"),
          ("hp", dict(color="C1", ls="--", lw=1.0, zorder=5),
           r"HP ($N{=}3$)"),
          ("beta", dict(color="C0", ls="-", lw=1.0, zorder=4),
           r"EKR ($N{=}2$)")]


def main() -> None:
    r = dict(np.load(NPZ_TRACE, allow_pickle=True))
    s = dict(np.load(NPZ_SCAN, allow_pickle=True))

    with matplotlib.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.95))

        # --- (a) time trace ----------------------------------------------
        ax = axes[0]
        zs, tau, eta = r["itg1_params"]
        t = r["t_itg1"]
        ref = complex(r["itg1_zeta_kin"])
        print(f"ITG trace: zeta_*={zs:g}, tau={tau:g}, eta={eta:g}, "
              f"zeta_kin={ref:.4f}")
        for key, sty, lab in STYLES:
            x = r[f"itg1_{key}"]
            zf = fit_mode(t, x, frac=(0.5, 1.0))
            if key != "kin":
                print(f"  {lab}: gamma mismatch "
                      f"{100*(zf.imag/ref.imag - 1):+.2g}%")
            ax.semilogy(t, np.abs(x), **sty, label=lab)
        ax.set_title(rf"(a) $\zeta_*={zs:g}$, $\tau={tau:g}$, $\eta={eta:g}$")
        ax.set_xlabel(r"$t\,k_\parallel v_{ti}$")
        ax.set_ylabel(r"$|U_{0,i}|$")
        ax.set_ylim(1e-6, None)
        ax.legend(fontsize=5.5, handlelength=1.4, labelspacing=0.2, borderpad=0.25, handletextpad=0.4, framealpha=0.85, loc="lower right")

        # --- (b) growth rate vs eta --------------------------------------
        ax = axes[1]
        eta_c, eta_f = s["eta_curve"], s["eta_fit"]
        eta_th = eta_threshold(ZETA_STAR, TAU)

        def gr(z):
            return np.where(np.isnan(z.real), 0.0, np.maximum(z.imag, 0.0))

        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(eta_th, color="gray", lw=0.8, ls="--")
        ax.plot(eta_c, gr(s["z_kin"]), "k-", lw=1.2, zorder=4,
                label="Kinetic")
        ax.plot(eta_c, gr(s["z_hp"]), "C1--", lw=1.0, zorder=3,
                label=r"HP ($N{=}3$)")
        ax.plot(eta_f, np.maximum(s["fit_hp"].imag, 0), "C1s", ms=3.2, mew=0.8,
                mfc="none", zorder=5, label=r"HP, fit")
        ax.plot(eta_f, np.maximum(s["fit_beta"].imag, 0), "C0^", ms=4,
                zorder=5, label=r"EKR, fit")
        ax.set_title(rf"(b) $\zeta_* = {ZETA_STAR:g}$, $\tau = {TAU:g}$")
        ax.set_xlabel(r"$\eta = L_n/L_T$")
        ax.set_ylabel(r"$\gamma / (|k_\parallel| v_{ti})$")
        ax.legend(fontsize=5.5, handlelength=1.4, labelspacing=0.2, borderpad=0.25, handletextpad=0.4, framealpha=0.85, loc="upper left")

        fig.tight_layout()
        fig.savefig(SAVE_PNG, dpi=400, bbox_inches="tight")
        print(f"Saved {SAVE_PNG}")
        plt.close(fig)


if __name__ == "__main__":
    main()
