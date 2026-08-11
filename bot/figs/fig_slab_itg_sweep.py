"""Driver: slab-ITG (zeta_*, eta) growth-rate sweep, HP N=3 vs EKR (beta_itg).

Generates bot/figures/fig_slab_itg_sweep.png.
Requires bot/runs/sweep_slab_itg.npz -- run
`python -m bot.sweeps.sweep_slab_itg` first if missing.

ITG counterpart of fig_hp_beta_sweep.png: relative growth-rate error
heatmaps over the drive plane, kinetic reference = Newton-polished root of
the analytic dispersion relation.  Cells below the kinetic threshold
eta_th(zeta_*) are masked (gray).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NPZ = Path(__file__).resolve().parent.parent / "runs" / "sweep_slab_itg.npz"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

RC = {"font.family": "serif", "mathtext.fontset": "cm",
      "font.size": 16, "axes.titlesize": 16}


def make_fig(protocol: str = "density") -> None:
    """protocol='density' (generic IC) or 'eigenmode' (on-manifold IC)."""
    d = np.load(NPZ)
    zs, eta = d["zeta_star_vals"], d["eta_vals"]
    suffix = "_em" if protocol == "eigenmode" else ""
    rel_hp = np.abs(d[f"gamma_hp{suffix}"] / d["gamma_kin"] - 1.0)
    rel_ekr = np.abs(d[f"gamma_ekr{suffix}"] / d["gamma_kin"] - 1.0)
    vmin, vmax = -4.0, 0.0

    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad("0.85")

    with matplotlib.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, rel, title in [(axes[0], rel_hp, r"Hammett-Perkins ($N=3$)"),
                               (axes[1], rel_ekr, r"EKR ($N=2$)")]:
            z = np.ma.masked_invalid(
                np.log10(np.maximum(rel, 10.0**vmin)))
            # eta grid is non-uniform: plot in index space with labeled ticks
            im = ax.imshow(z, origin="lower", aspect="auto",
                           cmap=cmap, vmin=vmin, vmax=vmax,
                           interpolation="nearest")
            ax.set_xticks(range(len(eta)))
            ax.set_xticklabels([f"{e:g}" for e in eta])
            ax.set_yticks(range(len(zs)))
            ax.set_yticklabels([f"{z_:g}" for z_ in zs])
            ax.set_title(title)
            ax.set_xlabel(r"$\eta$")
            ax.set_ylabel(r"$\zeta_*$")
        fig.tight_layout()

        cb = fig.colorbar(im, ax=fig.axes, shrink=0.85,
                          label=r"$\log_{10}\,|\gamma_{\rm eff}/\gamma_{\rm kin} - 1|$")
        cb.set_ticks([-4, -3, -2, -1, 0])
        cb.set_ticklabels([r"$\leq 10^{-4}$", r"$10^{-3}$", r"$10^{-2}$",
                           r"$10^{-1}$", r"$10^{0}$"])

        name = ("fig_slab_itg_sweep_eigic.png" if protocol == "eigenmode"
                else "fig_slab_itg_sweep.png")
        out = FIG_DIR / name
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"saved {out}")
        plt.close(fig)


if __name__ == "__main__":
    make_fig("density")
    make_fig("eigenmode")
