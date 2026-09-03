"""Driver: optimized Hunana-style linear closure vs the nonlinear N=4 NN.

Generates bot/figures/fig_opt_hunana_vs_nn.png.

Both closures are now fit on the SAME data: xi_b uniform on
Re in (-3,3) x Im in (-1,1.5) (bot.closures.n4_nn.make_dataset_n4), with the
same inputs -- on the eigenmode manifold the linear closure's
sum_i a_i alpha_i(xi) and the NN's f(r_1, r_2, r_3) see identical information.
The only difference left is the function class, which is the point of the
comparison.

Panels
------
(a), (b)  Relative closure error |alpha_4_hat - alpha_4| / |alpha_4| over the
          rectangle for the optimized linear N=4 closure and for the N=4 NN,
          on a shared colour scale.
(c)       Cumulative distribution of that error for the whole family:
          HP/Padé baselines, optimized linear N=3..6, and both N=4 NNs.
(d)       Bump-on-tail growth rate: |gamma_fluid - gamma_kin| / gamma_kin
          vs k at (u_b, eps) = (5, 0.05), from a time-domain run started at
          the exact kinetic eigenmode IC and fit over the second half of the
          record -- identical protocol for every closure, so the linear runs
          get no advantage from being propagated exactly.

The takeaway is the gap between (a) and (b) -- roughly 60x in median closure
error at equal N and equal training data -- set against panel (d), where the
linear closures are already within ~1% on the eigenmode manifold: the response
function that sets the dispersion relation is far better approximated than the
closure ratio itself, so the NN's advantage does not show up here.  It shows up
off the manifold (see the two-mode sweeps).

Requires bot/runs/n4_nn.eqx and bot/runs/n4_nn_super.eqx (already trained).
Runtime is dominated by the NN time-domain runs in panel (d), ~3 s per k.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bot.closed_loop import fluid_eigenmode_ic_n4, nn_n4_evolve, pade_evolve
from bot.closures.n4_nn import load as load_n4, load_super as load_n4_super
from bot.closures.opt_hunana import (NN_RE_RANGE, NN_IM_RANGE, Zprime,
                                     alpha_basis, opt_coefficients,
                                     sample_rectangle)
from bot.closures.pade import pade_coefficients, aaa_coefficients_n4
from bot.slab_itg import fit_mode
from bot.sweep import kinetic_peak

SAVE_PNG = "bot/figures/fig_opt_hunana_vs_nn.png"

N_RE, N_IM = 361, 241
VMIN, VMAX = -4.0, 0.5           # log10 relative closure error

U_B, EPS = 5.0, 0.05
AMP = 1e-3
T_END, N_T = 60.0, 601
N_K_NN = 12                      # time-domain points (NN cost ~3 s each)


# ---------------------------------------------------------------------------
# a priori: closure error over the rectangle
# ---------------------------------------------------------------------------

def _nn_alpha4(model, xi: np.ndarray) -> np.ndarray:
    """Vectorized NN alpha4 at the exact manifold ratios of xi."""
    import jax
    import jax.numpy as jnp
    B, _ = alpha_basis(xi.ravel(), 4)
    r1, r2, r3 = B[:, 1], B[:, 2], B[:, 3]
    feats = jnp.asarray(np.stack([r1.real, r1.imag, r2.real, r2.imag,
                                  r3.real, r3.imag], axis=1), dtype=jnp.float32)
    out = np.asarray(jax.vmap(model)(feats))
    return (out[:, 0] + 1j * out[:, 1]).reshape(xi.shape)


def _linear_alpha4(a: np.ndarray, xi: np.ndarray) -> np.ndarray:
    B, _ = alpha_basis(xi.ravel(), len(a))
    return (B @ a).reshape(xi.shape)


def run_maps():
    re = np.linspace(*NN_RE_RANGE, N_RE)
    im = np.linspace(*NN_IM_RANGE, N_IM)
    XI = re[None, :] + 1j * im[:, None]
    exact = alpha_basis(XI.ravel(), 4)[1].reshape(XI.shape)

    a_opt4 = opt_coefficients(4)
    nn = load_n4()
    err_lin = np.abs(_linear_alpha4(a_opt4, XI) - exact) / np.abs(exact)
    err_nn = np.abs(_nn_alpha4(nn, XI) - exact) / np.abs(exact)
    return {"re": re, "im": im, "err_lin": err_lin, "err_nn": err_nn}


def run_cdf(n_samples: int = 20000):
    xi = sample_rectangle(n_samples, seed=123)
    curves = {}

    linear = [(r"HP $N=3$", pade_coefficients(3), "C3", ":"),
              (r"opt $N=3$", opt_coefficients(3), "C3", "-"),
              (r"Padé $N=4$", pade_coefficients(4), "C0", ":"),
              (r"AAA $N=4$", aaa_coefficients_n4(), "C0", "--"),
              (r"opt $N=4$", opt_coefficients(4), "C0", "-"),
              (r"opt $N=5$", opt_coefficients(5), "C2", "-"),
              (r"opt $N=6$", opt_coefficients(6), "C4", "-")]
    for label, a, color, ls in linear:
        exact = alpha_basis(xi, len(a))[1]
        rel = np.abs(_linear_alpha4(a, xi) - exact) / np.abs(exact)
        curves[label] = (np.sort(rel), color, ls, 1.4)

    exact4 = alpha_basis(xi, 4)[1]
    for label, model, color in [(r"NN $N=4$ (single-mode)", load_n4(), "k"),
                                (r"NN $N=4$ (super)", load_n4_super(), "0.45")]:
        rel = np.abs(_nn_alpha4(model, xi) - exact4) / np.abs(exact4)
        curves[label] = (np.sort(rel), color, "-", 2.2)
    return curves


# ---------------------------------------------------------------------------
# dynamics: gamma(k) from a common time-domain protocol
# ---------------------------------------------------------------------------

def eigenmode_ic(k: float, omega: complex, N: int) -> np.ndarray:
    """Exact kinetic eigenmode IC [u, E, U_0..U_{N-1}] for any N.

    U_n = U_0 alpha_n(xi_b) with U_0 = i Z'(xi_b) E / k; reproduces
    bot.closed_loop.fluid_eigenmode_ic_u3 / _n4 at N = 3, 4 (asserted below),
    and extends them to N > 4 so every closure starts from the same state.
    """
    xi_b = (omega - k * U_B) / k
    B, _ = alpha_basis(np.array([xi_b]), N)
    U0 = 1j * Zprime(xi_b) * AMP / k
    y = np.zeros(2 + N, dtype=complex)
    y[0] = AMP / (1j * omega)
    y[1] = AMP
    y[2:] = U0 * B[0, :]
    return y


def run_growth():
    ks, w_kin, _ = kinetic_peak(U_B, EPS, k_lo=0.12, k_hi=0.45, n_k=34)
    mask = w_kin.imag > 0.05 * w_kin.imag.max()
    ks, w_kin = ks[mask], w_kin[mask]
    t = np.linspace(0.0, T_END, N_T)

    ref = fluid_eigenmode_ic_n4(ks[0], U_B, w_kin[0], AMP)
    assert np.allclose(eigenmode_ic(ks[0], w_kin[0], 4), ref), \
        "eigenmode_ic disagrees with closed_loop.fluid_eigenmode_ic_n4"

    linear = [(r"HP $N=3$", pade_coefficients(3), "C3", ":"),
              (r"opt $N=3$", opt_coefficients(3), "C3", "-"),
              (r"Padé $N=4$", pade_coefficients(4), "C0", ":"),
              (r"opt $N=4$", opt_coefficients(4), "C0", "-"),
              (r"opt $N=5$", opt_coefficients(5), "C2", "-")]

    out = {"ks": ks, "gamma_kin": w_kin.imag,
           "k_star": ks[int(np.argmax(w_kin.imag))], "linear": []}
    for label, a, color, ls in linear:
        gam = np.empty(len(ks))
        for i, k in enumerate(ks):
            y0 = eigenmode_ic(k, w_kin[i], len(a))
            E = pade_evolve(k, U_B, EPS, a, t, y0)
            gam[i] = fit_mode(t, E, frac=(0.5, 1.0)).imag
        out["linear"].append((label, color, ls,
                              np.abs(gam - w_kin.imag) / np.abs(w_kin.imag)))
        print(f"  {label:<12} median gamma err = "
              f"{np.median(out['linear'][-1][3]):.2%}")

    # NN on a subset of k (RK45 with the NN in the RHS is the expensive path)
    idx = np.unique(np.linspace(0, len(ks) - 1, N_K_NN).astype(int))
    nn = load_n4()
    gam_nn = np.empty(len(idx))
    for j, i in enumerate(idx):
        y0 = eigenmode_ic(ks[i], w_kin[i], 4)
        E = nn_n4_evolve(ks[i], U_B, EPS, nn, t, y0)
        gam_nn[j] = fit_mode(t, E, frac=(0.5, 1.0)).imag
    out["nn_ks"] = ks[idx]
    out["nn_err"] = np.abs(gam_nn - w_kin.imag[idx]) / np.abs(w_kin.imag[idx])
    print(f"  {'NN N=4':<12} median gamma err = {np.median(out['nn_err']):.2%}")
    return out


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

def make_fig(maps, curves, growth):
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))
    extent = [maps["re"][0], maps["re"][-1], maps["im"][0], maps["im"][-1]]

    for ax, key, title in [
            (axes[0, 0], "err_lin",
             r"(a) optimized linear $N=4$  ($\sum_i a_i\alpha_i$)"),
            (axes[0, 1], "err_nn",
             r"(b) NN $N=4$  ($f(r_1,r_2,r_3)$)")]:
        z = np.log10(np.clip(maps[key], 10**VMIN, None))
        im_h = ax.imshow(z, origin="lower", aspect="auto", extent=extent,
                         cmap="magma", vmin=VMIN, vmax=VMAX,
                         interpolation="nearest")
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel(r"${\rm Re}\,\xi_b$", fontsize=10.5)
        ax.set_ylabel(r"${\rm Im}\,\xi_b$", fontsize=10.5)
        ax.text(0.025, 0.955, f"median {np.median(maps[key]):.2%}",
                transform=ax.transAxes, fontsize=9, va="top", color="w")

    ax = axes[1, 0]
    for label, (rel, color, ls, lw) in curves.items():
        frac = np.arange(1, len(rel) + 1) / len(rel)
        ax.semilogx(np.maximum(rel, 1e-6), frac, color=color, ls=ls, lw=lw,
                    label=label)
    ax.set_xlabel(r"relative closure error  $|\hat\alpha_N-\alpha_N|/|\alpha_N|$",
                  fontsize=10.5)
    ax.set_ylabel("cumulative fraction of the rectangle", fontsize=10.5)
    ax.set_xlim(1e-5, 3.0)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5, loc="upper left", ncol=2)
    ax.set_title("(c) closure error, same samples for every model", fontsize=10.5)

    ax = axes[1, 1]
    for label, color, ls, err in growth["linear"]:
        ax.semilogy(growth["ks"], np.maximum(err, 1e-6), color=color, ls=ls,
                    lw=1.5, label=label)
    ax.semilogy(growth["nn_ks"], np.maximum(growth["nn_err"], 1e-6), "k-o",
                lw=2.0, ms=4, label=r"NN $N=4$")
    ax.axvline(growth["k_star"], color="gray", lw=0.7, ls="--")
    ax.text(growth["k_star"], 0.6, r" $k_*$", fontsize=8, color="gray",
            transform=ax.get_xaxis_transform())
    ax.set_xlabel(r"$k$", fontsize=10.5)
    ax.set_ylabel(r"$|\gamma_{\rm fluid}-\gamma_{\rm kin}|/\gamma_{\rm kin}$",
                  fontsize=10.5)
    ax.set_title(rf"(d) BoT growth rate, $u_b={U_B:g}$, $\varepsilon={EPS:g}$",
                 fontsize=10.5)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5, loc="best")

    fig.suptitle("Optimized Hunana-style linear closure vs the nonlinear NN, "
                 "fit on the same rectangle", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 0.91, 0.955))
    cax = fig.add_axes((0.935, 0.545, 0.016, 0.36))
    cb = fig.colorbar(im_h, cax=cax)
    cb.set_label(r"$\log_{10}\,|\hat\alpha_4-\alpha_4|/|\alpha_4|$", fontsize=9.5)
    cb.set_ticks([-4, -3, -2, -1, 0])
    cb.set_ticklabels([r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$",
                       r"$10^{-1}$", r"$\geq 10^{0}$"])
    fig.savefig(SAVE_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {SAVE_PNG}")


if __name__ == "__main__":
    print("closure-error maps ...")
    maps = run_maps()
    print("closure-error CDFs ...")
    curves = run_cdf()
    print("growth rates ...")
    growth = run_growth()
    make_fig(maps, curves, growth)
