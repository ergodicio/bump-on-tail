"""Driver: refined (u_b, eps) scan over the low-u_b / low-eps corner where HP
performs worst.

Generates bot/figures/fig_opt_hunana_sweep_fine.png.
Requires bot/runs/sweep_opt_hunana_fine.npz -- run
`python -m bot.sweeps.sweep_opt_hunana --fine` first if missing.

Why this box: on the coarse grid HP's largest relative growth-rate error is
8.3% at (u_b, eps) = (4, 0.02) -- the operating point of
fig_bot_vs_itg_closures.png -- and the ridge it sits on runs off the coarse
grid's own low-u_b / low-eps edge.  This scan resolves it: u_b in [2.5, 6],
eps log-spaced over [0.005, 0.06].

Reference and metric
--------------------
The error plotted is

    |gamma_fluid_eig - gamma_kin_exact| / gamma_kin_exact

i.e. the closed fluid system's most-unstable EIGENVALUE against the
Newton-polished analytic kinetic root at k*.  Both are free of time-domain fit
error, which matters here: at weak growth the generic-delta-E fit used by the
coarse sweep (and by fig_hp_beta_sweep) does not converge within the window --
at u_b=2.5, eps=0.005 the fitted kinetic rate is 13% off the true root, which
would otherwise be charged to the closures and make even Padé N=4 look 12%
wrong when its eigenvalue error there is 0.06%.  The dashed contour marks
where that fit-window error reaches 2%, i.e. the region in which the
time-domain view of this corner cannot be trusted.

The thin solid contour is the zero crossing of the SIGNED error: HP
over-predicts gamma at (4, 0.02) by +8.3% but under-predicts by -9.6% at
(2.5, 0.005), so |error| has two ridges separated by a valley that is a sign
change, not a region of genuine accuracy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bot.figs.fig_opt_hunana_sweep import LABELS
from bot.sweeps.sweep_opt_hunana import CLOSURES

# Panels for this figure: all six LINEAR closures (the nonlinear ones have no
# closed-system eigenvalue, and this figure's metric is the eigenvalue one).
# Titles come from fig_opt_hunana_sweep.LABELS -- edit them there, once.
PANELS = list(CLOSURES)

NPZ = Path(__file__).resolve().parent.parent / "runs" / "sweep_opt_hunana_fine.npz"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
SAVE_PNG = FIG_DIR / "fig_opt_hunana_sweep_fine.png"

# Narrower than the coarse figures' [-4, -1]: over this box the four closures
# live in 1e-3..1e-1, and only the sign-change valleys dip below, so two
# decades put the actual spread across the full colour range.
VMIN, VMAX = -3.0, -1.0
MARK = (0.02, 4.0)          # (eps, u_b) of the fig_bot_vs_itg_closures trace


def signed_errors(d) -> dict:
    """Signed relative eigenvalue error against the exact kinetic root."""
    g_ref = d["gamma_kin_exact"]
    return {name: (d[f"gamma_{name}_eig"] - g_ref) / np.abs(g_ref)
            for name in PANELS}


def make_fig(path: Path = NPZ) -> None:
    d = dict(np.load(path))
    u_b, eps = d["u_b_vals"], d["eps_vals"]
    signed = signed_errors(d)

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.8), sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), PANELS):
        err = np.abs(signed[name])
        z = np.log10(np.maximum(err, 10.0**VMIN))
        im = ax.pcolormesh(eps, u_b, z, cmap="magma_r", vmin=VMIN, vmax=VMAX,
                           shading="gouraud")
        # sign change of the error
        ax.contour(eps, u_b, signed[name], levels=[0.0], colors="0.25",
                   linewidths=0.9)
        # where the time-domain protocol's kinetic reference is unreliable
        ax.contour(eps, u_b, d["converged"], levels=[0.02], colors="C0",
                   linewidths=1.2, linestyles="--")
        ax.plot(*MARK, "x", color="C2", ms=10, mew=2.4)
        ax.set_xscale("log")
        ax.set_title(f"{LABELS[name]}\n"
                     f"median {np.nanmedian(err):.2%},  max {np.nanmax(err):.2%}",
                     fontsize=9.5)
    for ax in axes[1, :]:
        ax.set_xlabel(r"$\varepsilon = n_b/n_0$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$u_b/v_b$")

    fig.suptitle("Refined scan of the low-$u_b$ / low-$\\varepsilon$ corner:  "
                 r"$|\gamma_{\rm fluid}-\gamma_{\rm kin}|/\gamma_{\rm kin}$"
                 "\n(closed-system eigenvalue vs exact kinetic root at $k_*$;  "
                 r"$\times$ = $u_b{=}4$, $\varepsilon{=}0.02$;  "
                 "thin contour = error sign change;  "
                 "dashed = 2% fit-window error)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 0.92, 0.92))

    cax = fig.add_axes((0.94, 0.10, 0.014, 0.74))
    cb = fig.colorbar(im, cax=cax,
                      label=r"$\log_{10}\,|\gamma_{\rm fluid}/\gamma_{\rm kin}-1|$")
    cb.set_ticks([-3, -2.5, -2, -1.5, -1])
    cb.set_ticklabels([r"$\leq 0.1\%$", r"$0.3\%$", r"$1\%$", r"$3\%$",
                       r"$\geq 10\%$"])

    fig.savefig(SAVE_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {SAVE_PNG}")


if __name__ == "__main__":
    if not NPZ.exists():
        print(f"Data not found: {NPZ}")
        print("Run:  python -m bot.sweeps.sweep_opt_hunana --fine")
        raise SystemExit(1)
    make_fig()
