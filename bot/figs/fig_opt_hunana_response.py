"""Driver: response-function error of Hunana-style linear closures over the
NN training rectangle, before and after fitting on that rectangle.

Generates bot/figures/fig_opt_hunana_response.png.

Four maps of the relative response error |R_N(xi;a) - Z'(xi)| / |Z'(xi)| on
Re(xi) in (-3,3) x Im(xi) in (-1,1.5) -- exactly the rectangle the NN closures
are trained on (bot.closures.n4_nn.make_dataset_n4):

  HP N=3     : coefficients from 2-point matching (xi -> inf and Im Z'(0))
  opt N=3    : same family, coefficients fit on this rectangle
  Pade N=4   : Taylor-matched at the single point xi = 0
  opt N=4    : same family, coefficients fit on this rectangle

The white crosses are the closure's own poles (roots of xi^N - sum a_i xi^i).
Every member of the family for N >= 3 puts poles inside the rectangle, and the
bright rings around them are the irreducible cost of the rational
architecture: |R_N| diverges there while Z' stays finite.  That is also why
the fit uses the pole-free closure-ratio form of the same condition (see
bot.closures.opt_hunana module docstring) rather than minimizing this map
directly.

No pre-computed data required; the fits are ordinary least squares and run in
a fraction of a second.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bot.closures.opt_hunana import (NN_RE_RANGE, NN_IM_RANGE, Zprime, response,
                                     alpha_metrics, closure_poles,
                                     fit_coefficients, response_metrics)
from bot.closures.pade import pade_coefficients

SAVE_PNG = "bot/figures/fig_opt_hunana_response.png"
N_RE, N_IM = 481, 321
VMIN, VMAX = -4.0, 0.5          # log10 relative error


def run():
    re = np.linspace(*NN_RE_RANGE, N_RE)
    im = np.linspace(*NN_IM_RANGE, N_IM)
    XI = re[None, :] + 1j * im[:, None]
    tgt = Zprime(XI)

    fits = {N: fit_coefficients(N) for N in (3, 4)}
    for N, f in fits.items():
        print(f"  opt N={N}: stable={f.stable}  "
              f"R median={f.metrics['R_median']:.2%}  "
              f"alpha median={f.metrics['A_rel_median']:.2%}")

    entries = [("HP $N=3$ (2-point match)", pade_coefficients(3)),
               ("opt $N=3$ (fit on rectangle)", fits[3].a),
               (r"Padé $N=4$ (Taylor at $\xi=0$)", pade_coefficients(4)),
               ("opt $N=4$ (fit on rectangle)", fits[4].a)]

    panels = []
    for label, a in entries:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            err = np.abs(response(XI, a) - tgt) / np.abs(tgt)
        m = {**response_metrics(a), **alpha_metrics(a)}
        panels.append({"label": label, "err": err, "poles": closure_poles(a),
                       "median": m["R_median"], "p90": m["R_p90"],
                       "alpha": m["A_rel_median"]})
    return {"re": re, "im": im, "panels": panels}


def make_fig(r):
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.0), sharex=True, sharey=True)
    extent = [r["re"][0], r["re"][-1], r["im"][0], r["im"][-1]]

    for ax, p in zip(axes.ravel(), r["panels"]):
        z = np.log10(np.clip(p["err"], 10**VMIN, None))
        im_h = ax.imshow(z, origin="lower", aspect="auto", extent=extent,
                         cmap="magma", vmin=VMIN, vmax=VMAX,
                         interpolation="nearest")
        inside = [q for q in p["poles"]
                  if extent[0] <= q.real <= extent[1]
                  and extent[2] <= q.imag <= extent[3]]
        if inside:
            ax.plot([q.real for q in inside], [q.imag for q in inside],
                    "x", color="w", ms=7, mew=1.6, ls="none")
        ax.axhline(0.0, color="w", lw=0.5, alpha=0.35)
        ax.set_title(p["label"], fontsize=10.5)
        ax.text(0.025, 0.955,
                f"$|\\Delta R|/|Z'|$  median {p['median']:.1%},  p90 {p['p90']:.0%}\n"
                f"$|\\Delta\\alpha_N|/|\\alpha_N|$  median {p['alpha']:.1%}",
                transform=ax.transAxes, fontsize=8, va="top", color="w")

    for ax in axes[1, :]:
        ax.set_xlabel(r"${\rm Re}\,\xi_b$", fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"${\rm Im}\,\xi_b$", fontsize=11)

    fig.suptitle(r"Response error $|R_N-Z'|/|Z'|$ on the NN training rectangle"
                 "\n" r"(white $\times$: the closure's own poles; "
                 r"the fit minimizes the closure-ratio error $\Delta\alpha_N$)",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 0.92, 1.0))
    cax = fig.add_axes((0.935, 0.10, 0.018, 0.78))
    cb = fig.colorbar(im_h, cax=cax)
    cb.set_label(r"$\log_{10}\,|R_N-Z'|/|Z'|$", fontsize=10)
    cb.set_ticks([-4, -3, -2, -1, 0])
    cb.set_ticklabels([r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$",
                       r"$10^{-1}$", r"$\geq 10^{0}$"])

    fig.savefig(SAVE_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {SAVE_PNG}")


if __name__ == "__main__":
    make_fig(run())
