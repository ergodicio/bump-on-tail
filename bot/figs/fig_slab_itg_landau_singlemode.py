"""Driver: slab-ITG single-eigenmode test, Landau-damped (below-threshold) branch.

Generates bot/figures/fig_slab_itg_landau_singlemode.png.

Mirrors fig_u2_landau_singlemode.py on the BoT side: IC = the exact kinetic
eigenmode at a genuinely Landau-damped root of D_kinetic (eta below
eta_threshold, so there is no growing root at all -- this is the ITG-stable
regime).  No kinetic simulation is needed: the reference is the analytic
single exponential amp*exp(-i*zeta_root*t).  Closures:
  * direct beta ('itg' variant): exact-by-construction on this eigenmode
    (beta_itg is built from the same manifold moments used to construct
    the IC), so any late-time departure is basin-of-attraction root-
    switching in the nonlinear closure ODE, not approximation error.
  * HP N=3 (Pade, time-stepped RK45): a *linear* closure whose own
    dispersion relation differs from the kinetic one, so it gets an
    incorrect damping rate from the start (same mechanism documented for
    the BoT side: HP's rational response function only matches the exact
    one asymptotically at large |zeta|).
  * AAA N=4 (Pade, time-stepped RK45): same linear-closure architecture as
    HP but one moment higher, with a_i fit via scipy.interpolate.AAA (real-
    axis rational fit to Z'(xi)) + least-squares projection onto the N=4
    closure family instead of HP's 2-point Taylor/asymptotic match -- see
    bot.closures.pade.aaa_coefficients_n4 for the derivation and expected
    accuracy (~1-5% vs Z' on-axis, ~6-21% at the off-axis points tested,
    vs HP's 23-68% at the same points).
  * AAA pad (M4', fixed multi-pole state-space closure): per
    dressed_pole_closure_handoff.md's M4' addendum -- a FIXED (offline-fit)
    15-pole rational approximation of THIS problem's own U_0/phi response
    (-R_kinetic(zeta; ZETA_STAR, ETA), not the universal Z'(xi)), realized
    as auxiliary linear "pad" states w_j with dw_j/dt=-i*p_j*w_j-i*c_j*phi.
    Not constrained by the moment-ladder family (see bot.closures.aaa_pad
    module docstring), so it reaches ~0.01-0.02% error here vs HP's ~190%
    and AAA-N4's ~15-70%. Propagated by matrix exponential (method="eigen"),
    matching the doc's point that the closed pad system needs no adaptive
    integrator at all.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bot.slab_itg import (D_kinetic, min_damped_root, manifold_Un_over_phi,
                          fluid_hp_evolve, fluid_pade_evolve,
                          direct_beta_evolve, eta_threshold)
from bot.closures.pade import aaa_coefficients_n4
from bot.closures.aaa_pad import fit_aaa_pad_itg, aaa_pad_ic, aaa_pad_evolve

ZETA_STAR = 1.0
TAU = 1.0
ETA = 2.0
AMP = 1e-3
T_END = 30.0
SAVE_PNG = "bot/figures/fig_slab_itg_landau_singlemode.png"


def run():
    eta_th = eta_threshold(ZETA_STAR, TAU)
    assert ETA < eta_th, f"eta={ETA} must be below threshold {eta_th:.3f} for a damped root"

    z = min_damped_root(D_kinetic, ZETA_STAR, ETA, TAU)
    r0, r1, r2, r3 = manifold_Un_over_phi(z, ZETA_STAR, ETA, n_max=3)
    U0, U1, U2, U3 = AMP * r0, AMP * r1, AMP * r2, AMP * r3

    t = np.linspace(0.0, T_END, 1201)
    E_ref = U0 * np.exp(-1j * z * t)

    hp   = fluid_hp_evolve(ZETA_STAR, ETA, TAU, t, np.array([U0, U1, U2]),
                           method="rk45")
    beta = direct_beta_evolve(ZETA_STAR, ETA, TAU, t, np.array([U0, U1]),
                              variant="itg")
    aaa4 = fluid_pade_evolve(ZETA_STAR, ETA, TAU, t, np.array([U0, U1, U2, U3]),
                             aaa_coefficients_n4(), method="rk45")

    pad = fit_aaa_pad_itg(ZETA_STAR, ETA)
    print(f"  AAA pad: {len(pad.poles)} poles, n_pruned={pad.n_pruned}, "
          f"max_rel_err={pad.max_rel_err:.2e}")
    w0 = aaa_pad_ic(pad, z, AMP)
    pad_out = aaa_pad_evolve(pad, TAU, t, w0, method="eigen")

    return {"t": t, "zeta_root": z, "eta_th": eta_th,
            "E_ref": E_ref, "E_hp": hp["U0"], "E_beta": beta["U0"],
            "E_aaa4": aaa4["U0"], "E_pad": pad_out["U0"]}


def make_fig(r):
    t = r["t"]
    z = r["zeta_root"]

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 5.2))

    ax.semilogy(t, np.abs(r["E_ref"]), "k-", lw=2.5, zorder=6,
                label=r"$|{\rm amp}\,e^{-i\zeta t}|$ (exact single mode)")
    ax.semilogy(t, np.abs(r["E_beta"]), color="C1", lw=1.5, ls="--", zorder=4,
                label=r"Direct $\beta$ (ITG manifold, N=2)")
    ax.semilogy(t, np.abs(r["E_hp"]), color="C0", lw=1.5, ls=":", zorder=3,
                alpha=0.9, label=r"HP N=3 (Padé, RK45)")
    ax.semilogy(t, np.abs(r["E_aaa4"]), color="C2", lw=1.8, ls="-.", zorder=5,
                label=r"AAA N=4 (Padé, RK45)")
    ax.semilogy(t, np.abs(r["E_pad"]), color="C3", lw=1.4, ls="-", zorder=7,
                label=r"AAA pad (M4$'$, 15-pole, expm)")

    finite = np.concatenate([np.abs(r["E_ref"]), np.abs(r["E_beta"]),
                             np.abs(r["E_hp"]), np.abs(r["E_aaa4"]),
                             np.abs(r["E_pad"])])
    finite = finite[np.isfinite(finite) & (finite > 0)]
    ylo = 10 ** (np.floor(np.log10(finite.min())) - 0.5)
    yhi = 10 ** (np.ceil(np.log10(finite.max())) + 0.2)

    ax.set_title(
        rf"Slab ITG, Landau-damped branch: $\zeta_*={ZETA_STAR}$, $\tau={TAU}$, "
        rf"$\eta={ETA}$  (below $\eta_{{th}}={r['eta_th']:.3f}$)" + "\n"
        rf"$\zeta_{{\rm root}}={z:.4f}$",
        fontsize=10.5,
    )
    ax.set_xlim(0, T_END)
    ax.set_ylim(bottom=ylo, top=yhi)
    ax.set_xlabel(r"$t\,|k_\parallel| v_{ti}\sqrt{2}$", fontsize=11)
    ax.set_ylabel(r"$|U_{0,i}|$", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right")

    fig.tight_layout()
    fig.savefig(SAVE_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {SAVE_PNG}")


if __name__ == "__main__":
    r = run()
    make_fig(r)
