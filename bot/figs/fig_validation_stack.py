"""Driver: single-column, four-panel validation figure (panels stacked 4x1).

Generates bot/figures/fig_validation_stack.png.

Same data as fig_kappa_bot.py and fig_slab_itg_combined.py, re-laid out as
one column-width figure with the four panels stacked vertically so each
panel gets the full column width:

  (a) bump-on-tail |E|(t) for a kappa = 2 beam
  (b) slab-ITG |U_{0,i}|(t) for zeta_* = tau = 1, eta = 10
  (c) bump-on-tail growth rate versus kappa (gray dotted: Maxwellian limit)
  (d) ITG growth rate versus eta (gray dotted: eta_th = 1 + sqrt 5)

Panel order and line styles follow the caption of fig:validation in
apssamp.tex.  The kappa-scan solves are cached in
bot/runs/kappa_bot_cache.npz (delete it to force a re-run); the ITG panels
read bot/runs/bot_vs_itg_closures.npz and bot/runs/slab_itg_growth.npz as in
fig_slab_itg_combined.py.

Under the APS length formula a single-column figure counts 150/AR + 20
words with AR = width/height, so the panel height (PANEL_H) is the knob that
trades legibility against the word count; the script prints both.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bot.closed_loop import pade_evolve
from bot.closures.pade import pade_coefficients
from bot.kappa_bot import (ekr_evolve_kappa, find_root_kappa,
                           kinetic_evolve_kappa)
from bot.kinetic import most_unstable_kinetic
from bot.slab_itg import eta_threshold, fit_mode

# --- kappa bump-on-tail (fig_kappa_bot.py) ----------------------------------
U_B, EPS, K = 4.0, 0.02, 0.295
KAPPA_TRACE = 2
KAPPA_SCAN = (2, 3, 4, 6, 8, 12, 20, 40)
AMP = 1e-3
T_END = 110.0
N_V, V = 500, 25.0
NPZ_KAPPA = "bot/runs/kappa_bot_cache.npz"

# --- slab ITG (fig_slab_itg_combined.py) ------------------------------------
NPZ_TRACE = "bot/runs/bot_vs_itg_closures.npz"
NPZ_SCAN = "bot/runs/slab_itg_growth.npz"
ZETA_STAR, TAU = 1.0, 1.0

SAVE_PNG = "bot/figures/fig_validation_stack.png"

FIG_W = 3.4          # REVTeX column width, inches
PANEL_H = 1.15       # per-panel height, inches

RC = {"font.family": "serif", "mathtext.fontset": "cm",
      "font.size": 7, "axes.titlesize": 7}
LEG = dict(fontsize=5.5, handlelength=1.4, labelspacing=0.2, borderpad=0.25,
           handletextpad=0.4, framealpha=0.85)


def fit_gamma(t, E, frac=(0.5, 1.0)):
    m = (t >= frac[0] * t[-1]) & (t <= frac[1] * t[-1])
    return np.polyfit(t[m], np.log(np.abs(E[m])), 1)[0]


def kappa_data():
    if os.path.exists(NPZ_KAPPA):
        d = dict(np.load(NPZ_KAPPA))
        print(f"Loaded {NPZ_KAPPA}")
        return d
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
        print(f"  kappa={kappa:>3}: kin={w.imag:.6f}  EKR={ge:.6f} "
              f"({100*(ge/w.imag-1):+.2g}%)  HP={100*(g_hp/w.imag-1):+.1f}%")

    y0_kin = np.zeros(2 + N_V, dtype=complex)
    y0_kin[1] = AMP
    E_kin = kinetic_evolve_kappa(K, U_B, EPS, KAPPA_TRACE, t, y0_kin,
                                 N_v=N_V, V=V)
    y0_ekr = np.array([0.0, AMP, 0.0, 0.0], dtype=complex)
    E_ekr2 = ekr_evolve_kappa(K, U_B, EPS, KAPPA_TRACE, t, y0_ekr)

    d = dict(t=t, E_kin=E_kin, E_hp=E_hp, E_ekr2=E_ekr2, g_hp=g_hp,
             g_maxw=w_maxw.imag, g_kin=np.array(g_kin),
             g_ekr=np.array(g_ekr))
    np.savez(NPZ_KAPPA, **d)
    print(f"Saved {NPZ_KAPPA}")
    return d


def main() -> None:
    kd = kappa_data()
    r = dict(np.load(NPZ_TRACE, allow_pickle=True))
    s = dict(np.load(NPZ_SCAN, allow_pickle=True))

    with matplotlib.rc_context(RC):
        fig, axes = plt.subplots(4, 1, figsize=(FIG_W, 4 * PANEL_H))

        # --- (a) BoT time trace, kappa = 2 -------------------------------
        ax = axes[0]
        t = kd["t"]
        ax.semilogy(t, np.abs(kd["E_kin"]), "ko", ms=1.3, markevery=44,
                    zorder=10, label="Kinetic")
        ax.semilogy(t, np.abs(kd["E_hp"]), "C1--", lw=1.0, zorder=5,
                    label=r"HP ($N{=}3$)")
        ax.semilogy(t, np.abs(kd["E_ekr2"]), "C0-", lw=1.0, zorder=4,
                    label=r"EKR ($N{=}2$)")
        ax.set_title(rf"(a) Bump-on-tail, $\kappa = {KAPPA_TRACE}$")
        ax.set_xlabel(r"$t\,\omega_{pe}$")
        ax.set_ylabel(r"$|E|$")
        ax.set_ylim(1e-5, None)
        ax.legend(loc="lower right", **LEG)

        # --- (b) ITG time trace, eta = 10 --------------------------------
        ax = axes[1]
        zs, tau, eta = r["itg1_params"]
        ti = r["t_itg1"]
        ref = complex(r["itg1_zeta_kin"])
        styles = [("kin", dict(color="k", ls="none", marker="o", ms=1.3,
                               markevery=20, zorder=10), "Kinetic"),
                  ("hp", dict(color="C1", ls="--", lw=1.0, zorder=5),
                   r"HP ($N{=}3$)"),
                  ("beta", dict(color="C0", ls="-", lw=1.0, zorder=4),
                   r"EKR ($N{=}2$)")]
        for key, sty, lab in styles:
            x = r[f"itg1_{key}"]
            if key != "kin":
                zf = fit_mode(ti, x, frac=(0.5, 1.0))
                print(f"  ITG {lab}: gamma mismatch "
                      f"{100*(zf.imag/ref.imag - 1):+.2g}%")
            ax.semilogy(ti, np.abs(x), **sty, label=lab)
        ax.set_title(rf"(b) Slab ITG, $\zeta_*={zs:g}$, $\tau={tau:g}$, "
                     rf"$\eta={eta:g}$")
        ax.set_xlabel(r"$t\,k_\parallel v_{ti}$")
        ax.set_ylabel(r"$|U_{0,i}|$")
        ax.set_ylim(1e-6, None)
        ax.legend(loc="lower right", **LEG)

        # --- (c) BoT growth rate vs kappa --------------------------------
        ax = axes[2]
        ax.plot(KAPPA_SCAN, kd["g_kin"], "k-o", lw=1.1, ms=3, label="Kinetic")
        ax.axhline(kd["g_hp"], color="C1", ls="--", lw=1.0,
                   label=r"HP ($N{=}3$)")
        ax.plot(KAPPA_SCAN, kd["g_ekr"], "C0^", ms=4, label=r"EKR ($N{=}2$)")
        ax.axhline(kd["g_maxw"], color="0.6", ls=":", lw=0.9,
                   label="Maxwellian")
        ax.set_xscale("log")
        ax.set_xticks(list(KAPPA_SCAN))
        ax.set_xticklabels([str(k) for k in KAPPA_SCAN])
        ax.set_title("(c) Bump-on-tail growth rate")
        ax.set_xlabel(r"$\kappa$")
        ax.set_ylabel(r"$\gamma / \omega_{pe}$")
        # headroom so the legend clears the HP and Maxwellian lines
        lo = min(kd["g_maxw"], kd["g_kin"].min())
        hi = max(kd["g_kin"].max(), kd["g_hp"])
        ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.75 * (hi - lo))
        ax.legend(loc="upper right", ncol=2, **LEG)

        # --- (d) ITG growth rate vs eta ----------------------------------
        ax = axes[3]
        eta_c, eta_f = s["eta_curve"], s["eta_fit"]
        eta_th = eta_threshold(ZETA_STAR, TAU)

        def gr(z):
            return np.where(np.isnan(z.real), 0.0, np.maximum(z.imag, 0.0))

        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(eta_th, color="0.6", lw=0.9, ls=":")
        ax.plot(eta_c, gr(s["z_kin"]), "k-", lw=1.2, zorder=4,
                label="Kinetic")
        ax.plot(eta_c, gr(s["z_hp"]), "C1--", lw=1.0, zorder=3,
                label=r"HP ($N{=}3$)")
        ax.plot(eta_f, np.maximum(s["fit_hp"].imag, 0), "C1s", ms=3.2,
                mew=0.8, mfc="none", zorder=5, label=r"HP, fit")
        ax.plot(eta_f, np.maximum(s["fit_beta"].imag, 0), "C0^", ms=4,
                zorder=5, label=r"EKR, fit")
        ax.set_title(rf"(d) ITG growth rate, $\zeta_* = {ZETA_STAR:g}$, "
                     rf"$\tau = {TAU:g}$")
        ax.set_xlabel(r"$\eta = L_n/L_T$")
        ax.set_ylabel(r"$\gamma / (|k_\parallel| v_{ti})$")
        ax.legend(loc="upper left", ncol=2, **LEG)

        fig.tight_layout(h_pad=0.6)
        fig.savefig(SAVE_PNG, dpi=400, bbox_inches="tight")
        plt.close(fig)

    from PIL import Image
    w, h = Image.open(SAVE_PNG).size
    ar = w / h
    print(f"Saved {SAVE_PNG}: {w}x{h} px, AR = {ar:.3f}, "
          f"APS single-column word equivalent = {150/ar + 20:.0f}")


if __name__ == "__main__":
    main()
