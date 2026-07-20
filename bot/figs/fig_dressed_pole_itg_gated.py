"""Driver: dressed-pole M3-pre gated closure on the ITG generic-IC case
that hung a naive adaptive integrator (see dressed_pole_closure_handoff.md
addendum, M3-pre).

Generates bot/figures/fig_dressed_pole_itg_gated.png.

Case: zeta_* = 1, tau = 1, eta = 10, generic density-only IC.  Plain r=1
(no gate) hangs a naive RK45 solve for hours (fit's zeta(t) jumps
discontinuously between local minima -- a genuine rank>1 moment state, not
a fitting bug).  The gated closure (dressed_pole_r1_gated_next: blend
continuously toward the linear HP closure by relative fit residual)
completes in seconds and matches the kinetic growth rate to ~1e-10%.

Panels: (a) |U_0(t)| -- kinetic, HP, direct-alpha (itg variant), gated
dressed-pole; (b) relative fit residual (settledness diagnostic, log
scale); (c) blend weight (0 = trust the fit, 1 = trust HP); (d) fitted
zeta(t) trajectory in the complex plane vs the true kinetic root.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bot.closures.dressed_pole import itg_evolve
from bot.slab_itg import (kinetic_evolve, fluid_hp_evolve, direct_alpha_evolve,
                          density_ic_kinetic, density_ic_fluid,
                          max_growing_root, D_kinetic, fit_mode)

ZETA_STAR, TAU, ETA = 1.0, 1.0, 10.0
AMP = 1e-3
N_W = 600
T_END = 60.0

SAVE_PNG = "bot/figures/fig_dressed_pole_itg_gated.png"


def run():
    zk = max_growing_root(D_kinetic, ZETA_STAR, ETA, TAU, im_hi=6.0)
    t = np.linspace(0.0, T_END, int(20 * T_END) + 1)

    r_kin = kinetic_evolve(ZETA_STAR, ETA, TAU, t,
                           density_ic_kinetic(AMP, N_w=N_W), N_w=N_W)
    r_hp = fluid_hp_evolve(ZETA_STAR, ETA, TAU, t, density_ic_fluid(AMP, N=3),
                           method="rk45")
    r_alpha = direct_alpha_evolve(ZETA_STAR, ETA, TAU, t,
                                  density_ic_fluid(AMP, N=3), variant="itg")
    r_dp = itg_evolve(ZETA_STAR, ETA, TAU, t, density_ic_fluid(AMP, N=3))

    zf_dp = fit_mode(r_dp["t"], r_dp["U0"], frac=(0.5, 1.0))
    print(f"kinetic root:            {zk:.6f}")
    print(f"gated dressed-pole fit:   {zf_dp:.6f}  "
          f"(rel err {100*(zf_dp.imag/zk.imag-1):+.2e} %)")
    print(f"nfev-equivalent log points: {len(r_dp['log_t'])}")

    return {"zk": zk, "t": t, "kin": r_kin, "hp": r_hp, "alpha": r_alpha,
            "dp": r_dp}


def make_fig(d):
    zk = d["zk"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    ax = axes[0, 0]
    ax.semilogy(d["t"], np.abs(d["kin"]["U0"]), "k-", lw=2.2, label="kinetic")
    ax.semilogy(d["t"], np.abs(d["hp"]["U0"]), "C1--", lw=1.6, label="HP N=3")
    ax.semilogy(d["t"], np.abs(d["alpha"]["U0"]), "C3:", lw=1.6,
                label=r"direct $\alpha$ (spurious)")
    ax.semilogy(d["dp"]["t"], np.abs(d["dp"]["U0"]), "C2-", lw=1.8,
                label="dressed-pole, gated (M3-pre)")
    ax.set_xlabel(r"$t\,k_\parallel v_{ti}$")
    ax.set_ylabel(r"$|U_{0,i}|$")
    ax.set_title("Time-domain trace (generic density IC)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    log_t = d["dp"]["log_t"]
    # relative residual rr = residual/||U|| is what the gate actually acts
    # on; recover it from the logged blend (blend = rr^2/(rr^2+rel_tol^2))
    # rather than plotting the raw (absolute) residual, which grows in
    # lockstep with the exponentially-growing state and would misleadingly
    # look like the fit degrades late in the run when it does not (the
    # blend panel, which IS relative, correctly stays ~0 there).
    blend_arr = d["dp"]["log_blend"]
    rel_tol = 0.3
    rr = rel_tol * np.sqrt(np.clip(blend_arr, 0, 1 - 1e-12) /
                           (1 - np.clip(blend_arr, 0, 1 - 1e-12)))
    ax.semilogy(log_t, rr, color="C2", lw=1.2)
    ax.axhline(rel_tol, color="0.5", ls="--", lw=1,
              label=rf"rel\_tol$={rel_tol}$")
    ax.set_xlabel(r"$t\,k_\parallel v_{ti}$")
    ax.set_ylabel(r"relative fit residual $\|U-w\,m(\zeta)\|_2/\|U\|$")
    ax.set_title("Settledness diagnostic (no equivalent in direct-$\\beta/\\alpha$)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    log_blend = d["dp"]["log_blend"]
    ax.plot(log_t, log_blend, color="C2", lw=1.2)
    ax.set_xlabel(r"$t\,k_\parallel v_{ti}$")
    ax.set_ylabel("blend weight (0=fit, 1=HP)")
    ax.set_title("M3-pre gate: continuous HP blend")
    ax.set_ylim(-0.05, 1.05)

    ax = axes[1, 1]
    zeta_log = d["dp"]["log_zeta"]
    sc = ax.scatter(zeta_log.real, zeta_log.imag, c=log_t, cmap="viridis",
                    s=6, lw=0)
    ax.plot(zk.real, zk.imag, "r*", ms=16, label="true kinetic root")
    fig.colorbar(sc, ax=ax, label="$t$")
    ax.set_xlabel(r"$\mathrm{Re}\,\zeta$")
    ax.set_ylabel(r"$\mathrm{Im}\,\zeta$")
    ax.set_title(r"Fitted $\zeta(t)$ trajectory vs true root")
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        r"Dressed-pole r=1, M3-pre gated closure  "
        rf"($\zeta_*={ZETA_STAR}$, $\tau={TAU}$, $\eta={ETA}$, generic IC)"
        "\nplain r=1 hangs a naive integrator here; gating completes and "
        "matches kinetic to ~1e-10%", fontsize=11)
    plt.tight_layout()
    plt.savefig(SAVE_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {SAVE_PNG}")


if __name__ == "__main__":
    d = run()
    make_fig(d)
