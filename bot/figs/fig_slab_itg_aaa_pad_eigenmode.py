"""Driver: slab-ITG growing-mode test with the AAA-pad (M4') closure added.

Generates bot/figures/fig_slab_itg_aaa_pad_eigenmode.png.

Same (zeta_*, tau, eta) = (1, 1, 10) as ITG case 1 in fig_bot_vs_itg_closures
/ fig_bot_vs_itg_closures_scaled ("moderate disagreement" case), so this is
directly comparable to those figures' ITG panel -- with one deliberate
change: IC = the single growing kinetic eigenmode (via manifold_Un_over_phi
at the exact root), not the generic density-only IC those figures use.

That change is necessary, not cosmetic: the AAA-pad closure
(bot.closures.aaa_pad.fit_aaa_pad_itg / aaa_pad_evolve) represents ONLY
U_0's response to phi (ITG quasineutrality is algebraic, so no separate
U_1, U_2 state is carried) -- it has no way to encode an IC where U_1, U_2
are set independently of U_0, which the density-only IC does. A first
attempt at a variant that retains the physical U_0..U_2 ladder and closes
U_3 via a per-step fixed-pole least-squares projection (bot.closures.
aaa_pad.aaa_pad_ladder_evolve) was tried on the density IC and failed
badly (wrong-sign growth rate) -- the density IC has substantial
continuum/non-pole content that a small set of discrete poles cannot
represent via a plain (minimum-norm) least-squares projection. Left
unresolved; see bot/paper/notes_slab_itg.md sec 11 for the writeup.

On the eigenmode IC the AAA pad is exact-by-construction (like direct-beta
here), so this figure demonstrates it is at least as good as direct-beta
while HP (a genuinely different rational form, not exact on this manifold)
is not.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bot.slab_itg import (D_kinetic, max_growing_root, manifold_Un_over_phi,
                          fluid_hp_evolve, direct_beta_evolve)
from bot.closures.aaa_pad import fit_aaa_pad_itg, aaa_pad_ic, aaa_pad_evolve

ZETA_STAR = 1.0
TAU = 1.0
ETA = 10.0
AMP = 1e-3
T_END = 25.0
SAVE_PNG = "bot/figures/fig_slab_itg_aaa_pad_eigenmode.png"


def run():
    z = max_growing_root(D_kinetic, ZETA_STAR, ETA, TAU)
    r0, r1, r2 = manifold_Un_over_phi(z, ZETA_STAR, ETA, n_max=2)
    U0, U1, U2 = AMP * r0, AMP * r1, AMP * r2

    t = np.linspace(0.0, T_END, 1201)
    E_ref = U0 * np.exp(-1j * z * t)

    hp   = fluid_hp_evolve(ZETA_STAR, ETA, TAU, t, np.array([U0, U1, U2]),
                           method="rk45")
    beta = direct_beta_evolve(ZETA_STAR, ETA, TAU, t, np.array([U0, U1]),
                              variant="itg")

    pad = fit_aaa_pad_itg(ZETA_STAR, ETA)
    print(f"  AAA pad: {len(pad.poles)} poles, n_pruned={pad.n_pruned}, "
          f"max_rel_err={pad.max_rel_err:.2e}")
    w0 = aaa_pad_ic(pad, z, AMP)
    pad_out = aaa_pad_evolve(pad, TAU, t, w0, method="eigen")

    return {"t": t, "zeta_root": z,
            "E_ref": E_ref, "E_hp": hp["U0"], "E_beta": beta["U0"],
            "E_pad": pad_out["U0"]}


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
    ax.semilogy(t, np.abs(r["E_pad"]), color="C3", lw=1.4, ls="-.", zorder=7,
                label=r"AAA pad (M4$'$, expm)")

    finite = np.concatenate([np.abs(r["E_ref"]), np.abs(r["E_beta"]),
                             np.abs(r["E_hp"]), np.abs(r["E_pad"])])
    finite = finite[np.isfinite(finite) & (finite > 0)]
    ylo = 10 ** (np.floor(np.log10(finite.min())) - 0.5)
    yhi = 10 ** (np.ceil(np.log10(finite.max())) + 0.2)

    ax.set_title(
        rf"Slab ITG, growing branch: $\zeta_*={ZETA_STAR}$, $\tau={TAU}$, "
        rf"$\eta={ETA}$  (same case as fig_bot_vs_itg_closures ITG-1)" + "\n"
        rf"$\zeta_{{\rm root}}={z:.4f}$  (eigenmode IC, not the generic-density IC used there)",
        fontsize=9.5,
    )
    ax.set_xlim(0, T_END)
    ax.set_ylim(bottom=ylo, top=yhi)
    ax.set_xlabel(r"$t\,k_\parallel v_{ti}$", fontsize=11)
    ax.set_ylabel(r"$|U_{0,i}|$", fontsize=11)
    ax.legend(fontsize=8.5, loc="lower right")

    fig.tight_layout()
    fig.savefig(SAVE_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {SAVE_PNG}")


if __name__ == "__main__":
    r = run()
    make_fig(r)
