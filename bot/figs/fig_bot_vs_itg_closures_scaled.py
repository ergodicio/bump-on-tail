"""Driver: scaled-down 2-panel version of fig_bot_vs_itg_closures.png.

Reuses the data already computed/saved by fig_bot_vs_itg_closures.py (does
not recompute) -- BoT case 2 (u_b=4, eps=0.02, largest disagreement) and
ITG case 1 (zeta_*=1, tau=1, eta=10, moderate disagreement).

Generates bot/figures/fig_bot_vs_itg_closures_scaled.png.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bot.slab_itg import fit_mode

NPZ = "bot/runs/bot_vs_itg_closures.npz"
SAVE_PNG = "bot/figures/fig_bot_vs_itg_closures_scaled.png"

STYLES = [("kin", dict(color="k", ls="-", lw=2.0), "Kinetic ground truth"),
          ("hp", dict(color="C1", ls="--", lw=1.6), r"Hammett-Perkins ($N=3$)"),
          ("beta", dict(color="C0", ls="-.", lw=1.4),
           r"Exact kinetic response ($N=2$)")]

# Computer Modern (LaTeX-style) fonts to match fig_hp_beta_sweep; mathtext
# 'cm' rather than usetex so no external latex install is needed.  Growth-
# rate mismatches and case parameters are printed to stdout (they live in
# the paper caption, not the figure).
RC = {"font.family": "serif", "mathtext.fontset": "cm",
      "font.size": 16, "axes.titlesize": 16}


def make_fig(r):
    with matplotlib.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))

        ax = axes[0]
        u_b, eps, k = r["bot2_params"]
        t = r["t_bot2"]
        print(f"BoT params: u_b={u_b:g}, eps={eps:g}, k={k:.3f}")
        for key, sty, lab in STYLES:
            x = r[f"bot2_{key}"]
            zf = fit_mode(t, x, frac=(0.5, 1.0))
            if key != "kin":
                mismatch = zf.imag / complex(r["bot2_omega_kin"]).imag - 1
                print(f"  BoT {key}: gamma mismatch {100*mismatch:+.2g}%")
            ax.semilogy(t, np.abs(x), **sty, label=lab)
        ax.set_title("Bump-on-tail")
        ax.set_xlabel(r"$t\,\omega_{pe}$")
        ax.set_ylabel("$|E|$")
        ax.set_ylim(1e-6, None)
        ax.legend(fontsize=12, loc="lower right")

        ax = axes[1]
        zs, tau, eta = r["itg1_params"]
        t = r["t_itg1"]
        ref = complex(r["itg1_zeta_kin"])
        print(f"ITG params: zeta_*={zs:g}, tau={tau:g}, eta={eta:g}")
        for key, sty, lab in STYLES:
            x = r[f"itg1_{key}"]
            zf = fit_mode(t, x, frac=(0.5, 1.0))
            if key != "kin":
                mismatch = zf.imag / ref.imag - 1
                print(f"  ITG {key}: gamma mismatch {100*mismatch:+.2g}%")
            ax.semilogy(t, np.abs(x), **sty, label=lab)
        ax.set_title("Slab ITG")
        ax.set_xlabel(r"$t\,k_\parallel v_{ti}$")
        ax.set_ylabel(r"$|U_{0,i}|$")
        ax.set_ylim(1e-6, None)
        ax.legend(fontsize=12, loc="lower right")

        plt.tight_layout()
        plt.savefig(SAVE_PNG, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {SAVE_PNG}")


if __name__ == "__main__":
    r = dict(np.load(NPZ, allow_pickle=True))
    make_fig(r)
