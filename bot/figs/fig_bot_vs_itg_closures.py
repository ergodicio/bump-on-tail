"""Driver: same closures, same strategy — bump-on-tail vs slab ITG, time domain.

Generates bot/figures/fig_bot_vs_itg_closures.png and
bot/runs/bot_vs_itg_closures.npz.

Both problems are solved as initial-value problems from a generic IC and
compared against their kinetic ground truth, with the *same three closures*:

  * HP 3-pole heat-flux closure  U_3 = (i chi1/2) U_0 + (3/2) U_1 - i chi1 U_2
    (on the BoT side this is exactly pade_coefficients(3); on the ITG side
    fluid_hp_system with Gamma = 3)
  * direct beta  (N=2):  U_2 = beta(U_1/U_0) U_0
  * direct alpha (N=3):  U_3 = alpha(U_1/U_0) U_0

BoT case: u_b = 5, eps = 0.05, k = 0.24 (canonical growing case), delta-E IC.
ITG case: zeta_* = 1, tau = 1, eta = 4.5 (above threshold), density IC.

Expected outcome (see bot/paper/notes_slab_itg.md §8): the closure ranking
flips between the two problems.  On BoT the hierarchy is undriven at n = 0,
so U_1/U_0 = xi_b on-manifold and the direct closures are exact per mode,
while HP is a ~5% gamma overshoot.  On ITG the gradient sources enter every
moment equation (U_1/U_0 = zeta + zeta_*/tau on-manifold), the direct
closures are evaluated off their manifold and fail qualitatively, while HP
stays within ~7% of the kinetic growth rate.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bot.closed_loop import (kinetic_evolve as bot_kinetic_evolve,
                             pade_evolve as bot_pade_evolve,
                             direct_u2_evolve as bot_direct_beta_evolve,
                             direct_alpha_evolve as bot_direct_alpha_evolve,
                             generic_dE_ic, generic_dE_ic_kinetic)
from bot.closures.pade import pade_coefficients
from bot.kinetic import most_unstable_kinetic as bot_kinetic_mode
from bot.slab_itg import (D_kinetic, max_growing_root, kinetic_evolve,
                          fluid_hp_evolve, direct_beta_evolve,
                          direct_alpha_evolve, density_ic_kinetic,
                          density_ic_fluid, fit_mode)

# BoT canonical growing case
U_B, EPS, K_BOT = 5.0, 0.05, 0.24
# ITG case (above the kinetic threshold 1+sqrt(5))
ZETA_STAR, TAU, ETA = 1.0, 1.0, 4.5
AMP = 1e-3
N_W = 600

SAVE_PNG = "bot/figures/fig_bot_vs_itg_closures.png"
SAVE_NPZ = "bot/runs/bot_vs_itg_closures.npz"


def run_bot():
    t = np.linspace(0.0, 80.0, 801)
    a_hp = pade_coefficients(3)   # == [i chi1/2, 3/2, -i chi1]: HP 3-pole
    out = {"t_bot": t}
    out["bot_kin"] = bot_kinetic_evolve(K_BOT, U_B, EPS, t,
                                        generic_dE_ic_kinetic(AMP))
    out["bot_hp"] = bot_pade_evolve(K_BOT, U_B, EPS, a_hp, t,
                                    generic_dE_ic(AMP))
    out["bot_beta"] = bot_direct_beta_evolve(
        K_BOT, U_B, EPS, t, np.array([0, AMP, 0, 0], dtype=complex))
    out["bot_alpha"] = bot_direct_alpha_evolve(K_BOT, U_B, EPS, t,
                                               generic_dE_ic(AMP))
    out["bot_omega_kin"] = bot_kinetic_mode(K_BOT, U_B, EPS)
    return out


def run_itg():
    t = np.linspace(0.0, 60.0, 601)
    out = {"t_itg": t}
    out["itg_kin"] = kinetic_evolve(ZETA_STAR, ETA, TAU, t,
                                    density_ic_kinetic(AMP, N_w=N_W),
                                    N_w=N_W)["U0"]
    out["itg_hp"] = fluid_hp_evolve(ZETA_STAR, ETA, TAU, t,
                                    density_ic_fluid(AMP, N=3))["U0"]
    out["itg_beta"] = direct_beta_evolve(ZETA_STAR, ETA, TAU, t,
                                         density_ic_fluid(AMP, N=2))["U0"]
    out["itg_alpha"] = direct_alpha_evolve(ZETA_STAR, ETA, TAU, t,
                                           density_ic_fluid(AMP, N=3))["U0"]
    out["itg_zeta_kin"] = max_growing_root(D_kinetic, ZETA_STAR, ETA, TAU)
    return out


STYLES = [("kin", dict(color="k", ls="-", lw=2.2), "kinetic"),
          ("hp", dict(color="C1", ls="--", lw=1.8), "HP 3-pole"),
          ("beta", dict(color="C0", ls="-.", lw=1.6), r"direct $\beta$ (N=2)"),
          ("alpha", dict(color="C3", ls=":", lw=1.8), r"direct $\alpha$ (N=3)")]


def make_fig(r):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    for ax, prob, t_key, sig, ref, title in [
        (axes[0], "bot", "t_bot", "|E|",
         complex(r["bot_omega_kin"]),
         rf"Bump-on-tail  ($u_b={U_B}$, $\epsilon={EPS}$, $k={K_BOT}$), "
         r"$\delta E$ IC"),
        (axes[1], "itg", "t_itg", r"|\tilde n_i/n_0|",
         complex(r["itg_zeta_kin"]),
         rf"Slab ITG  ($\zeta_*={ZETA_STAR}$, $\tau={TAU}$, $\eta={ETA}$), "
         r"$\delta n$ IC"),
    ]:
        t = r[t_key]
        for key, sty, lab in STYLES:
            x = r[f"{prob}_{key}"]
            zf = fit_mode(t, x, frac=(0.5, 1.0))
            ax.semilogy(t, np.abs(x), **sty,
                        label=rf"{lab}  ($\gamma_{{fit}}={zf.imag:+.4f}$)")
        ax.text(0.02, 0.98, rf"kinetic mode: $\gamma = {ref.imag:.4f}$",
                transform=ax.transAxes, va="top", fontsize=9, color="0.3")
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("$t$ (problem units)", fontsize=11)
        ax.set_ylabel(rf"${sig}$", fontsize=11)
        ax.set_ylim(1e-6, None)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Identical closures, identical strategy: time-domain "
                 "initial-value tests on both problems", fontsize=12)
    plt.tight_layout()
    plt.savefig(SAVE_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {SAVE_PNG}")


def report(r):
    print("\n── fitted mode (omega_r + i gamma), second half of record ──")
    print(f"  BoT kinetic reference:  {complex(r['bot_omega_kin']):.5f}")
    for key, _sty, lab in STYLES:
        zf = fit_mode(r["t_bot"], r[f"bot_{key}"], frac=(0.5, 1.0))
        print(f"    BoT {lab:22s} {zf:.5f}")
    print(f"  ITG kinetic reference:  {complex(r['itg_zeta_kin']):.5f}")
    for key, _sty, lab in STYLES:
        zf = fit_mode(r["t_itg"], r[f"itg_{key}"], frac=(0.5, 1.0))
        print(f"    ITG {lab:22s} {zf:.5f}")


if __name__ == "__main__":
    r = {**run_bot(), **run_itg()}
    np.savez_compressed(SAVE_NPZ, **r)
    print(f"Saved {SAVE_NPZ}")
    make_fig(r)
    report(r)
