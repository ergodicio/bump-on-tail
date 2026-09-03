"""Driver: bump-on-tail with a kappa-distributed beam, kinetic vs HP vs EKR.

Generates bot/figures/fig_kappa_bot.png.

Same (u_b, eps, k) as the BoT panel of fig_bot_vs_itg_closures_scaled, with
the Maxwellian beam replaced by a kappa distribution normalized to the same
density and temperature.  The HP fluid system is blind to this change -- its
matrix sees the equilibrium only through M_0 and M_1, and its closure
coefficients come from the Maxwellian asymptotics of Z' -- so its prediction
is a constant, independent of kappa, while the kinetic answer varies by
~20% over the range shown.  The EKR closure simply uses R_kappa in place of
-Z'/2 and follows the kinetic curve.

Panel (a): time traces at kappa = 2.
Panel (b): growth rate versus kappa.  Note HP crosses the kinetic curve near
kappa ~ 4, where its Maxwellian bias happens to cancel the non-Maxwellian
shift -- accidental agreement at a single point, not accuracy.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bot.closed_loop import pade_evolve
from bot.closures.pade import pade_coefficients
from bot.kappa_bot import (ekr_evolve_kappa, find_root_kappa,
                           kinetic_evolve_kappa)
from bot.kinetic import most_unstable_kinetic

U_B, EPS, K = 4.0, 0.02, 0.295
KAPPA_TRACE = 2
KAPPA_SCAN = (2, 3, 4, 6, 8, 12, 20, 40)
AMP = 1e-3
T_END = 110.0
N_V, V = 500, 25.0

RC = {"font.family": "serif", "mathtext.fontset": "cm",
      "font.size": 7, "axes.titlesize": 7}


def fit_gamma(t, E, frac=(0.5, 1.0)):
    m = (t >= frac[0] * t[-1]) & (t <= frac[1] * t[-1])
    return np.polyfit(t[m], np.log(np.abs(E[m])), 1)[0]


def main() -> None:
    t = np.linspace(0.0, T_END, 1101)
    a_hp = pade_coefficients(3)

    y0_hp = np.array([0.0, AMP, 0.0, 0.0, 0.0], dtype=complex)
    E_hp = pade_evolve(K, U_B, EPS, a_hp, t, y0_hp, method="rk45")
    g_hp = fit_gamma(t, E_hp)          # kappa-independent

    w_maxw = most_unstable_kinetic(K, U_B, EPS)
    print(f"HP (kappa-independent): {g_hp:.6f}")
    print(f"Maxwellian kinetic:     {w_maxw.imag:.6f}")

    # --- scan over kappa -------------------------------------------------
    g_kin, g_ekr = [], []
    for kappa in KAPPA_SCAN:
        w = find_root_kappa(K, U_B, EPS, kappa, 1.0 + 0.05j)
        y0_ekr = np.array([0.0, AMP, 0.0, 0.0], dtype=complex)
        E_ekr = ekr_evolve_kappa(K, U_B, EPS, kappa, t, y0_ekr)
        ge = fit_gamma(t, E_ekr)
        g_kin.append(w.imag)
        g_ekr.append(ge)
        print(f"  kappa={kappa:>3}: kin={w.imag:.6f}  EKR={ge:.6f} "
              f"({100*(ge/w.imag-1):+.2g}%)  HP={100*(g_hp/w.imag-1):+.1f}%")

    with matplotlib.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.95))

        # --- (a) time traces at kappa = 2 --------------------------------
        ax = axes[0]
        y0_kin = np.zeros(2 + N_V, dtype=complex)
        y0_kin[1] = AMP
        E_kin = kinetic_evolve_kappa(K, U_B, EPS, KAPPA_TRACE, t, y0_kin,
                                     N_v=N_V, V=V)
        y0_ekr = np.array([0.0, AMP, 0.0, 0.0], dtype=complex)
        E_ekr2 = ekr_evolve_kappa(K, U_B, EPS, KAPPA_TRACE, t, y0_ekr)

        ax.semilogy(t, np.abs(E_kin), "ko", ms=1.3, markevery=44,
                    zorder=10, label="Kinetic")
        ax.semilogy(t, np.abs(E_hp), "C1--", lw=1.0, zorder=5,
                    label=r"HP ($N{=}3$)")
        ax.semilogy(t, np.abs(E_ekr2), "C0-", lw=1.0, zorder=4,
                    label=r"EKR ($N{=}2$)")
        ax.set_title(rf"(a) $\kappa = {KAPPA_TRACE}$")
        ax.set_xlabel(r"$t\,\omega_{pe}$")
        ax.set_ylabel(r"$|E|$")
        ax.set_ylim(1e-5, None)
        ax.legend(fontsize=5.5, handlelength=1.4, labelspacing=0.2, borderpad=0.25, handletextpad=0.4, framealpha=0.85, loc="lower right")

        # --- (b) growth rate vs kappa ------------------------------------
        ax = axes[1]
        ax.plot(KAPPA_SCAN, g_kin, "k-o", lw=1.1, ms=3,
                label="Kinetic")
        ax.axhline(g_hp, color="C1", ls="--", lw=1.0,
                   label=r"HP ($N{=}3$)")
        ax.plot(KAPPA_SCAN, g_ekr, "C0^", ms=4, label=r"EKR ($N{=}2$)")
        ax.axhline(w_maxw.imag, color="0.6", ls=":", lw=0.9,
                   label="Maxwellian")
        ax.set_xscale("log")
        ax.set_xticks(list(KAPPA_SCAN))
        ax.set_xticklabels([str(k) for k in KAPPA_SCAN])
        ax.set_title("(b) Growth rate")
        ax.set_xlabel(r"$\kappa$")
        ax.set_ylabel(r"$\gamma / \omega_{pe}$")
        ax.legend(fontsize=5.5, handlelength=1.4, labelspacing=0.2, borderpad=0.25, handletextpad=0.4, framealpha=0.85, loc="upper right")

        fig.tight_layout()
        out = "bot/figures/fig_kappa_bot.png"
        fig.savefig(out, dpi=400, bbox_inches="tight")
        print(f"Saved {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
