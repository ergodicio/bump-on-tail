"""Driver: same closures, same strategy — bump-on-tail vs slab ITG, time domain.

Generates bot/figures/fig_bot_vs_itg_closures.png and
bot/runs/bot_vs_itg_closures.npz.

Each problem is solved as an initial-value problem from a generic IC at
three parameter points spanning good -> moderate -> large disagreement
between the HP 3-pole closure and kinetic (direct alpha is computed for
report() but excluded from the figure; see notes_slab_itg.md for why):

  * HP 3-pole heat-flux closure  U_3 = (i chi1/2) U_0 + (3/2) U_1 - i chi1 U_2
    (on the BoT side this is exactly pade_coefficients(3); on the ITG side
    fluid_hp_system with Gamma = 3).  HP is linear and would normally be
    solved exactly via eigendecomposition (no time-stepping error), but is
    run here with method="rk45" (same RK45 solver, same rtol/atol as the
    nonlinear direct-beta/alpha closures below) so both fluid closures are
    genuinely time-stepped the same way -- an apples-to-apples comparison
    of the transient, not just the asymptotic growth rate.
  * direct beta (N=2):  U_2 = beta(U_1/U_0) U_0 on BoT (undriven n=0 row,
    exact per mode, always); U_2 = beta_itg(zeta_hat; zeta_*, eta) U_0 on
    ITG, the zeta_hat-shift + ITG-manifold-ratio repair from
    notes_slab_itg.md §9 / Appendix A (variant="itg") -- also exact.
    Necessarily time-stepped (RK45): both closures are nonlinear functions
    of the state (through the moment ratio), so there is no fixed matrix
    to diagonalize the way there is for HP or kinetic.

Case selection (see conversation / eigenvalue scans, not yet in notes):
  BoT   (u_b, eps): (5, 0.05) original benchmark [+5.0%], (3, 0.05) moderate
        [+6.9%], (4, 0.02) largest found with a reasonable growth rate
        [+8.3%].  Scanning (u_b, eps) in [2.5,15] x [0.002,0.3] at the
        kinetic-peak k did not turn up HP overshoots above ~12% anywhere
        (eigenvalue-based, not time-domain-fit-based -- the old
        sweep_hp_beta.npz fit_gamma numbers are contaminated by slowly-
        decaying subdominant Pade modes and read far too high, e.g. 31% at
        u_b=5,eps=0.05 vs the true eigenvalue overshoot of 5%).
  ITG   (zeta_*, tau, eta), all at zeta_* = 1: (1, 4.5) original benchmark
        [-7.0%], (1, 10.0) moderate [-9.7%], (0.5, 10.32) largest disagreement
        found for which a generic density IC still converges to the growing
        kinetic mode [-10.8%].  A much larger disagreement (-35%) exists at
        zeta_* = 0.1, tau = 0.3, eta = 12.85, but a generic density IC does
        NOT reach the growing mode there within any practical integration
        time -- it decays to the numerical noise floor (or, elsewhere in a
        zeta_*=1, tau=0.3 scan, locks onto an unrelated branch instead).
        This basin-of-attraction failure was checked broadly, not just at
        that one point (eta in [5,12] at zeta_*=0.5, and tau=0.3 at
        zeta_*=1 both fail the same way), so the largest disagreement
        *reachable by a verified generic-IC convergence test* is ~11%, well
        below BoT's ~12% ceiling -- i.e. once both sides are held to the
        same honest standard (density IC, not eigenmode-seeded), the two
        problems look comparably well-approximated by HP in this
        near-threshold-to-moderate regime; ITG's much larger disagreement
        at small zeta_*/sub-unity tau is real (verified at the eigenvalue
        level) but is not a generic-IC-reachable operating point for the
        N=2 direct-beta closure.
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
from bot.sweep import kinetic_peak
from bot.slab_itg import (D_kinetic, max_growing_root, kinetic_evolve,
                          fluid_hp_evolve, direct_beta_evolve,
                          direct_alpha_evolve, density_ic_kinetic,
                          density_ic_fluid, fit_mode)

AMP = 1e-3
N_W = 600

# (u_b, eps, t_end) -- t_end lengthened for the slower-growing case
BOT_CASES = [(5.0, 0.05, 80.0), (3.0, 0.05, 80.0), (4.0, 0.02, 110.0)]
# (zeta_star, tau, eta, t_end), all a generic density IC (verified to
# converge to the growing kinetic mode at every point below -- see the
# module docstring for the basin-of-attraction search that ruled out larger
# zeta_*=0.1-style disagreements as generic-IC-reachable).  Cases 1-2 need a
# longer window than case 0: both pass through a deep transient dip (U_0
# down to the 1e-7..1e-9 range) before locking onto the growing mode.
ITG_CASES = [(1.0, 1.0, 4.5, 60.0), (1.0, 1.0, 10.0, 25.0),
            (1.0, 0.5, 10.32, 35.0)]

SAVE_PNG = "bot/figures/fig_bot_vs_itg_closures.png"
SAVE_NPZ = "bot/runs/bot_vs_itg_closures.npz"


def run_bot():
    out = {}
    a_hp = pade_coefficients(3)   # == [i chi1/2, 3/2, -i chi1]: HP 3-pole
    for i, (u_b, eps, t_end) in enumerate(BOT_CASES):
        ks, ws, imax = kinetic_peak(u_b, eps)
        k = ks[imax]
        t = np.linspace(0.0, t_end, int(10 * t_end) + 1)
        out[f"t_bot{i}"] = t
        out[f"bot{i}_kin"] = bot_kinetic_evolve(k, u_b, eps, t,
                                                generic_dE_ic_kinetic(AMP))
        out[f"bot{i}_hp"] = bot_pade_evolve(k, u_b, eps, a_hp, t,
                                            generic_dE_ic(AMP), method="rk45")
        out[f"bot{i}_beta"] = bot_direct_beta_evolve(
            k, u_b, eps, t, np.array([0, AMP, 0, 0], dtype=complex))
        out[f"bot{i}_alpha"] = bot_direct_alpha_evolve(k, u_b, eps, t,
                                                        generic_dE_ic(AMP))
        out[f"bot{i}_omega_kin"] = bot_kinetic_mode(k, u_b, eps)
        out[f"bot{i}_params"] = np.array([u_b, eps, k])
    return out


def run_itg():
    out = {}
    for i, (zs, tau, eta, t_end) in enumerate(ITG_CASES):
        t = np.linspace(0.0, t_end, int(20 * t_end) + 1)
        out[f"t_itg{i}"] = t
        out[f"itg{i}_kin"] = kinetic_evolve(zs, eta, tau, t,
                                            density_ic_kinetic(AMP, N_w=N_W),
                                            N_w=N_W)["U0"]
        out[f"itg{i}_hp"] = fluid_hp_evolve(zs, eta, tau, t,
                                            density_ic_fluid(AMP, N=3),
                                            method="rk45")["U0"]
        out[f"itg{i}_beta"] = direct_beta_evolve(zs, eta, tau, t,
                                                 density_ic_fluid(AMP, N=2),
                                                 variant="itg")["U0"]
        out[f"itg{i}_alpha"] = direct_alpha_evolve(zs, eta, tau, t,
                                                    density_ic_fluid(AMP, N=3),
                                                    variant="itg")["U0"]
        out[f"itg{i}_zeta_kin"] = max_growing_root(D_kinetic, zs, eta, tau,
                                                   im_hi=6.0)
        out[f"itg{i}_params"] = np.array([zs, tau, eta])
    return out


STYLES = [("kin", dict(color="k", ls="-", lw=2.0), "kinetic"),
          ("hp", dict(color="C1", ls="--", lw=1.6), "HP 3-pole"),
          ("beta", dict(color="C0", ls="-.", lw=1.4), r"direct $\beta$"),
          ("alpha", dict(color="C3", ls=":", lw=1.6), r"direct $\alpha$")]

PLOT_STYLES = [s for s in STYLES if s[0] != "alpha"]


def make_fig(r):
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0))

    for i in range(3):
        ax = axes[0, i]
        t = r[f"t_bot{i}"]
        u_b, eps, k = r[f"bot{i}_params"]
        ref = complex(r[f"bot{i}_omega_kin"])
        for key, sty, lab in PLOT_STYLES:
            x = r[f"bot{i}_{key}"]
            zf = fit_mode(t, x, frac=(0.5, 1.0))
            ov = 100 * (zf.imag - ref.imag) / ref.imag if key != "kin" else 0.0
            lbl = lab if key == "kin" else rf"{lab}  ({ov:+.1f}%)"
            ax.semilogy(t, np.abs(x), **sty, label=lbl)
        ax.set_title(rf"BoT  $u_b={u_b:g}$, $\epsilon={eps:g}$, $k={k:.3f}$",
                     fontsize=10.5)
        ax.set_xlabel(r"$t\,\omega_{pe}$", fontsize=11)
        ax.set_ylabel("$|E|$", fontsize=11)
        ax.set_ylim(1e-6, None)
        ax.legend(fontsize=7.5, loc="lower right")

    for i in range(3):
        ax = axes[1, i]
        t = r[f"t_itg{i}"]
        zs, tau, eta = r[f"itg{i}_params"]
        ref = complex(r[f"itg{i}_zeta_kin"])
        for key, sty, lab in PLOT_STYLES:
            x = r[f"itg{i}_{key}"]
            zf = fit_mode(t, x, frac=(0.5, 1.0))
            ov = 100 * (zf.imag - ref.imag) / ref.imag if key != "kin" else 0.0
            lbl = lab if key == "kin" else rf"{lab}  ({ov:+.1f}%)"
            ax.semilogy(t, np.abs(x), **sty, label=lbl)
        ax.set_title(rf"ITG  $\zeta_*={zs:g}$, $\tau={tau:g}$, $\eta={eta:g}$",
                     fontsize=10.5)
        ax.set_xlabel(r"$t\,k_\parallel v_{t0}$", fontsize=11)
        ax.set_ylabel(r"$|\tilde n_i/n_0|$", fontsize=11)
        ax.set_ylim(1e-6, None)
        ax.legend(fontsize=7.5, loc="lower right")

    plt.tight_layout()
    plt.savefig(SAVE_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {SAVE_PNG}")


def report(r):
    print("\n── fitted mode (omega_r + i gamma), second half of record ──")
    for i in range(3):
        u_b, eps, k = r[f"bot{i}_params"]
        print(f"  BoT case {i}: u_b={u_b:g} eps={eps:g} k={k:.4f}  "
              f"kinetic reference {complex(r[f'bot{i}_omega_kin']):.5f}")
        for key, _sty, lab in STYLES:
            zf = fit_mode(r[f"t_bot{i}"], r[f"bot{i}_{key}"], frac=(0.5, 1.0))
            print(f"    {lab:16s} {zf:.5f}")
    for i in range(3):
        zs, tau, eta = r[f"itg{i}_params"]
        print(f"  ITG case {i}: zeta_*={zs:g} tau={tau:g} eta={eta:g}  "
              f"kinetic reference {complex(r[f'itg{i}_zeta_kin']):.5f}")
        for key, _sty, lab in STYLES:
            zf = fit_mode(r[f"t_itg{i}"], r[f"itg{i}_{key}"], frac=(0.5, 1.0))
            print(f"    {lab:16s} {zf:.5f}")


if __name__ == "__main__":
    r = {**run_bot(), **run_itg()}
    np.savez_compressed(SAVE_NPZ, **r)
    print(f"Saved {SAVE_NPZ}")
    make_fig(r)
    report(r)
