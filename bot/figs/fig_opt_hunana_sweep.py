"""Driver: (u_b, eps) growth-rate sweep, RELATIVE rate error.

Generates bot/figures/fig_opt_hunana_sweep.png.
Requires bot/runs/sweep_opt_hunana.npz -- run
`python -m bot.sweeps.sweep_opt_hunana` first if missing.

Same grid, IC and gamma_eff fit as fig_hp_beta_sweep.png, so the panels are
directly comparable to the HP / direct-beta pair.  The absolute-error view
(matching fig_hp_beta_sweep_abs.png) is fig_opt_hunana_sweep_abs.py, which
reuses make_fig from here.

Six panels (PANELS, below), chosen to separate the three
things that can change:

  order            HP N=3 -> Hunana N=4, both with HP's own matching split
  fitting domain   HP N=3 -> opt N=3, and Hunana N=4 -> opt N=4
  architecture     all four linear closures -> NN N=4 and Direct beta N=2,
                   the two nonlinear closures

The pure Taylor-at-xi=0 N=4 closure and the (3 at 0 + 1 asym) intermediate are
computed by the sweep and stored in the npz, but are left out of this figure;
they appear in fig_opt_hunana_sweep_fine.png.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bot.sweeps.sweep_opt_hunana import NONLINEAR

NPZ = Path(__file__).resolve().parent.parent / "runs" / "sweep_opt_hunana.npz"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

# ---------------------------------------------------------------------------
# Presentation knobs.  Everything here is read at figure time, so editing any
# of it needs only the figure driver re-run -- NOT the sweep.  The npz already
# holds all eight closures (gamma_{hp3,opt3,hunana4,hunana4b,pade4,opt4,nn4,
# beta2}, plus gamma_*_eig for the six linear ones), so panels can be added,
# dropped or reordered freely.
# ---------------------------------------------------------------------------

# Which of the computed closures get panels, in order.  Any length works --
# make_fig lays them out in NCOLS columns and hides the leftover axes -- so
# panels can be added, dropped or reordered without touching the plotting code.
# Current order: the four linear closures on the top row, the three nonlinear
# ones on the bottom.
PANELS = ["hp3", "hunana4", "opt3", "opt4", "beta2", "nn3", "nn4"]
NCOLS = 4

LABELS = {
    "hp3":      r"Hammett-Perkins ($N=3$)",
    "opt3":     r"Padé opt ($N=3$)",
    "hunana4":  r"Hunana ($N=4$)",
    "hunana4b": r"Hunana ($N=4$, 3+1)",
    "pade4":    r"Padé ($N=4$, Taylor)",
    "opt4":     r"Padé opt ($N=4$)",
    "nn3":      r"NN ($N=3$)",
    "nn4":      r"NN ($N=4$)",
    "beta2":    r"EKR ($N=2$)",
}

TITLE = (r"Bump-on-tail $\gamma$ error: Hunana-family linear closures "
         "vs the nonlinear closures (blue titles)")

CMAP = "magma_r"
# Same conventions as bot.figures_u2: relative view on [-4, 1], absolute view
# narrowed to [-4, -1] where the raw rate errors actually live.
LIMITS = {"relative": (-4.0, 1.0), "absolute": (-4.0, -1.0)}
CBAR_LABEL = {
    "relative": r"$\log_{10}\,|\gamma_{\rm eff}/\gamma_{\rm kin} - 1|$",
    "absolute": r"$\log_{10}\,|\gamma_{\rm eff} - \gamma_{\rm kin}|$",
}


def error_maps(d, metric: str, eig: bool = False,
               panels: list[str] = PANELS, ref: str = "fit") -> dict:
    """Per-closure error array for the requested metric.

    eig=True is only defined for the linear closures (the nonlinear ones have
    no fixed system matrix, so the npz carries no gamma_*_eig for them).

    ref selects the kinetic reference:
      "fit"    gamma_kin -- the fitted gamma_eff of the discretised kinetic run,
               same IC and same window as the closures.  Matches sweep_hp_beta.
               The reference then carries its own transient error
               (|gamma_kin - gamma_kin_exact|: median 2.4e-5, max 2.1e-4 over
               this grid), which floors the metric: direct beta and NN N=4 sit
               BELOW that, so their numbers are protocol upper bounds.
      "exact"  gamma_kin_exact -- the Newton-polished root of the analytic
               dispersion relation at k*.  No discretisation, no window, no
               fit, so no floor.  Changes the linear closures not at all (they
               are 14-180x above the floor; per-cell ratio 1.00) and roughly
               doubles the nonlinear ones, which is the honest number: with
               this reference the residual is the fluid run's OWN transient
               contamination rather than the difference of two noisy fits.
    """
    if ref not in ("fit", "exact"):
        raise ValueError(f"ref must be 'fit' or 'exact', got {ref!r}")
    suffix = "_eig" if eig else ""
    if eig:
        g_kin = d["gamma_kin_eig"] if ref == "fit" else d["gamma_kin_exact"]
    else:
        g_kin = d["gamma_kin"] if ref == "fit" else d["gamma_kin_exact"]
    out = {}
    for name in panels:
        key = f"gamma_{name}{suffix}"
        if key not in d:
            raise KeyError(f"{key} not in npz -- {name} is a nonlinear closure "
                           f"and has no eigenvalue view")
        err = np.abs(d[key] - g_kin)
        out[name] = err / np.abs(g_kin) if metric == "relative" else err
    return out


def make_fig(metric: str = "relative", path: Path = NPZ,
             eig: bool = False, out_name: str | None = None,
             ref: str = "fit") -> None:
    d = dict(np.load(path))
    u_b, eps = d["u_b_vals"], d["eps_vals"]
    extent = [eps[0], eps[-1], u_b[0], u_b[-1]]
    vmin, vmax = LIMITS[metric]
    maps = error_maps(d, metric, eig=eig, ref=ref)

    matplotlib.rcParams.update({"font.family": "serif",
                                "mathtext.fontset": "cm",
                                "font.size": 16, "axes.titlesize": 16})
    ncols = min(NCOLS, len(PANELS))
    nrows = -(-len(PANELS) // ncols)          # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 3.9 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    flat = axes.ravel()
    for ax, name in zip(flat, PANELS):
        z = np.log10(np.maximum(maps[name], 10.0**vmin))
        im = ax.imshow(z, origin="lower", aspect="auto", extent=extent,
                       cmap=CMAP, vmin=vmin, vmax=vmax, interpolation="nearest")
        # nonlinear closures get a coloured title so the architectural
        # boundary is visible at a glance
        ax.set_title(LABELS[name], color="k")
        ax.text(0.025, 0.955,
                f"median {np.nanmedian(maps[name]):.2e}\n"
                f"max {np.nanmax(maps[name]):.2e}",
                transform=ax.transAxes, fontsize=10, va="top",
                color="k", bbox=dict(fc="w", ec="none", alpha=0.6, pad=1.5))

    for ax in flat[len(PANELS):]:             # leftover cells in the grid
        ax.set_visible(False)
    # x labels on the lowest VISIBLE axis of each column (a partly filled last
    # row would otherwise leave some columns unlabelled)
    for j in range(ncols):
        col = [axes[i, j] for i in range(nrows) if axes[i, j].get_visible()]
        if col:
            col[-1].set_xlabel(r"$\varepsilon = n_b/n_0$")
            col[-1].tick_params(labelbottom=True)
    for i in range(nrows):
        if axes[i, 0].get_visible():
            axes[i, 0].set_ylabel(r"$u_b/v_{tb}$")

    src = "closed-system eigenvalue" if eig else r"generic $\delta E$ IC, time-domain fit"
    #fig.suptitle(f"{TITLE}\n({src}, at $k_*$)", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 0.92, 0.94))

    cax = fig.add_axes((0.94, 0.10, 0.014, 0.76))
    cb = fig.colorbar(im, cax=cax, label=CBAR_LABEL[metric])
    ticks = [t for t in (-4, -3, -2, -1, 0, 1) if vmin <= t <= vmax]
    cb.set_ticks(ticks)
    cb.set_ticklabels([(rf"$\leq 10^{{{t}}}$" if t == ticks[0]
                        else rf"$10^{{{t}}}$") for t in ticks])

    if out_name is None:
        out_name = ("fig_opt_hunana_sweep"
                    + ("_abs" if metric == "absolute" else "")
                    + ("_eig" if eig else "")
                    + ("_exactref" if ref == "exact" else "") + ".png")
    out = FIG_DIR / out_name
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    if not NPZ.exists():
        print(f"Data not found: {NPZ}")
        print("Run:  python -m bot.sweeps.sweep_opt_hunana")
        raise SystemExit(1)
    make_fig("relative")
