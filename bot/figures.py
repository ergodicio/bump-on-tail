"""Figures for the HP/Padé BoT baseline sweep."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bot.closures.pade import pade_coefficients, fluid_U0
from bot.closures.inference import alpha as inf_alpha, model_xi, moment_ratios
from bot.closures.naive_nn import (load as load_naive,
                                    naive_predict_alpha)
from bot.fluid import trace_fluid
from bot.kinetic import Zprime, most_unstable_kinetic
from bot.sweep import kinetic_peak


FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)
RUN_DIR = Path(__file__).resolve().parent / "runs"


LOG_OVER_VMIN = -4.0   # log10(|fractional error|) lower bound: 10^-4 = 0.01%
LOG_OVER_VMAX =  1.0   # upper bound: 10^1 = 1000%
LOG_OVER_CMAP = "magma_r"


def _log_overshoot_heatmap(ax, over_frac, extent, title):
    """Plot log10(|fractional overshoot|) heatmap on a shared scale."""
    z = np.log10(np.maximum(np.abs(over_frac), 1e-6))
    im = ax.imshow(z, origin="lower", aspect="auto", extent=extent,
                    cmap=LOG_OVER_CMAP, vmin=LOG_OVER_VMIN, vmax=LOG_OVER_VMAX)
    ax.set_title(title)
    ax.set_xlabel(r"$\varepsilon = n_b/n_0$"); ax.set_ylabel(r"$u_b/v_b$")
    ax.scatter([0.05], [5.0], c="white", s=60, marker="x", linewidths=2)
    ax.scatter([0.05], [5.0], c="black", s=20, marker="x", linewidths=1.5)
    return im


def _add_log_colorbar(fig, im, label=r"$\log_{10}\,|\gamma_{\rm eff}/\gamma_{\rm kin} - 1|$"):
    cb = fig.colorbar(im, ax=fig.axes, shrink=0.85, label=label)
    # tick labels: 10^-4 = 0.01%, 10^-3 = 0.1%, ..., 10^1 = 1000%
    ticks = [-4, -3, -2, -1, 0, 1]
    cb.set_ticks(ticks)
    cb.set_ticklabels([r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$",
                        r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$"])
    return cb


def fig_sweep_comparison(pade_path: Path = RUN_DIR / "sweep_pade_N3.npz",
                          td_path: Path = RUN_DIR / "sweep_timedomain.npz") -> None:
    """log10|overshoot| heatmaps: Padé N=3 (dispersion) vs NN closure (TD).

    Padé heatmap from eigenvalue sweep; NN heatmap reuses the time-domain
    sweep (identical at eigenmode-level since matrix-exp is exact).
    Shared log-error colour scale with fig_sweep_timedomain[_sim].
    """
    dP = np.load(pade_path)
    dT = np.load(td_path)
    u_b = dP["u_b_vals"]; eps = dP["eps_vals"]
    over_p = dP["overshoot"]
    over_n = dT["overshoot_naive"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    extent = [eps[0], eps[-1], u_b[0], u_b[-1]]
    _log_overshoot_heatmap(axes[0], over_p, extent, "Padé N=3")
    im1 = _log_overshoot_heatmap(axes[1], over_n, extent,
                                   r"$\xi_b$-sampled NN closure")
    fig.suptitle(r"$\gamma_{\max}$ relative error vs kinetic "
                  r"(shared log scale, $10^{-4}$ to $10^{1}$)",
                  fontsize=13, y=1.02)
    fig.tight_layout()
    _add_log_colorbar(fig, im1)
    out = FIG_DIR / "fig_sweep_comparison.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def fig_sweep_heatmaps(npz_path: Path = RUN_DIR / "sweep_pade_N3.npz") -> None:
    """3-panel: gamma_kin, gamma_fluid, overshoot vs (u_b, eps)."""
    d = np.load(npz_path)
    u_b = d["u_b_vals"]
    eps = d["eps_vals"]
    g_kin = d["gamma_kin"]
    g_flu = d["gamma_fluid"]
    over = d["overshoot"] * 100  # percent

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # γ_kin
    im0 = axes[0].imshow(g_kin, origin="lower", aspect="auto",
                          extent=[eps[0], eps[-1], u_b[0], u_b[-1]],
                          cmap="viridis")
    axes[0].set_title(r"kinetic $\gamma_{\max}$")
    axes[0].set_xlabel(r"$\varepsilon = n_b/n_0$")
    axes[0].set_ylabel(r"$u_b / v_b$")
    plt.colorbar(im0, ax=axes[0])

    # γ_fluid
    im1 = axes[1].imshow(g_flu, origin="lower", aspect="auto",
                          extent=[eps[0], eps[-1], u_b[0], u_b[-1]],
                          cmap="viridis")
    axes[1].set_title(r"Padé N=3 $\gamma_{\max}$")
    axes[1].set_xlabel(r"$\varepsilon = n_b/n_0$")
    axes[1].set_ylabel(r"$u_b / v_b$")
    plt.colorbar(im1, ax=axes[1])

    # overshoot
    im2 = axes[2].imshow(over, origin="lower", aspect="auto",
                          extent=[eps[0], eps[-1], u_b[0], u_b[-1]],
                          cmap="RdBu_r", vmin=-50, vmax=50)
    axes[2].set_title(r"overshoot $(\gamma_{\rm flu} - \gamma_{\rm kin}) / \gamma_{\rm kin}$  [%]")
    axes[2].set_xlabel(r"$\varepsilon = n_b/n_0$")
    axes[2].set_ylabel(r"$u_b / v_b$")
    plt.colorbar(im2, ax=axes[2])

    # annotate canonical point
    for ax in axes:
        ax.scatter([0.05], [5.0], c="white", s=60, marker="x", linewidths=2)
        ax.scatter([0.05], [5.0], c="black", s=20, marker="x", linewidths=1.5)

    fig.suptitle("BoT γ_max: kinetic truth vs Padé N=3 closure",
                  fontsize=13, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "fig_pade_N3_sweep.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def fig_dispersion_examples(
    cases: list[tuple[float, float]] = [(5.0, 0.05), (5.0, 0.20),
                                         (8.0, 0.05), (3.0, 0.05)],
    Ns: list[int] = [3, 4, 5, 6],
    include_inference: bool = True,
) -> None:
    """γ(k) for representative (u_b, eps) — overlay Padé hierarchy + inference vs kinetic."""
    from bot.eval_inference import evaluate_at as eval_inf
    from bot.train_inference import load as load_inference
    model = load_inference() if include_inference else None

    coeffs = {N: pade_coefficients(N) for N in Ns}
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(Ns)))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=False)
    for ax, (u_b, eps) in zip(axes.flat, cases):
        ks, ws_kin, i_max = kinetic_peak(u_b, eps, k_lo=0.10, k_hi=0.55, n_k=60)
        ax.plot(ks, ws_kin.imag, "k-", lw=2, label="kinetic")
        for N, c in zip(Ns, colors):
            ws_flu = trace_fluid(ks, u_b, eps, coeffs[N], ws_kin)
            ax.plot(ks, ws_flu.imag, "--", color=c, lw=1.3, label=f"Padé N={N}")
        if model is not None:
            r = eval_inf(u_b, eps, model=model, k_lo=0.10, k_hi=0.55, n_k=60)
            ax.plot(r["ks"], r["ws_inf"].imag, "r-", lw=1.8, label="inference")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(ks[i_max], color="gray", lw=0.5, ls=":")
        ax.set_xlabel(r"$k$")
        ax.set_ylabel(r"$\gamma = $ Im $\omega$")
        ax.set_title(rf"$u_b/v_b={u_b}$,  $\varepsilon={eps}$")
        ax.grid(alpha=0.3)
        # focus y-limits on the unstable region
        gam_max = max(ws_kin.imag.max(), 0.05)
        ax.set_ylim(-0.05, 1.5 * gam_max)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("BoT γ(k): Padé hierarchy vs kinetic", fontsize=13, y=1.00)
    fig.tight_layout()
    out = FIG_DIR / "fig_pade_dispersion.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def fig_response_function(Ns: list[int] = [3, 4, 5, 6],
                          xi_range: tuple[float, float] = (-3.0, 3.0),
                          n_xi: int = 401,
                          include_inference: bool = True) -> None:
    """Hunana-style: closure response vs exact -√π Z'(ξ) along real ξ.

    Overlays the Padé hierarchy for N in Ns. If include_inference, also
    overlays the trained inference closure (matches kinetic to NN precision).
    """
    xi = np.linspace(xi_range[0], xi_range[1], n_xi)
    sqrt_pi = float(np.sqrt(np.pi))
    target = -sqrt_pi * Zprime(xi)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(Ns)))

    axes[0].plot(xi, target.real, "k-", lw=2.2, label="kinetic")
    axes[1].plot(xi, target.imag, "k-", lw=2.2, label="kinetic")

    for N, c in zip(Ns, colors):
        a = pade_coefficients(N)
        approx = fluid_U0(xi, a)
        resid = np.abs(approx - target)
        axes[0].plot(xi, approx.real, "--", color=c, lw=1.3, label=f"Padé N={N}")
        axes[1].plot(xi, approx.imag, "--", color=c, lw=1.3, label=f"Padé N={N}")
        axes[2].semilogy(xi, resid, "-", color=c, lw=1.3, label=f"Padé N={N}")

    # NN closure overlay removed -- the closure's alpha-error is shown
    # quantitatively in the sweep/time-domain figures rather than as an
    # effective Z' on this plot.

    axes[0].set_xlabel(r"$\xi_b$")
    axes[0].set_ylabel(r"Re $[-\sqrt{\pi}\, Z'(\xi_b)]$")
    axes[0].set_title("real part")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9, loc="best")
    axes[0].axhline(0, color="gray", lw=0.5)

    axes[1].set_xlabel(r"$\xi_b$")
    axes[1].set_ylabel(r"Im $[-\sqrt{\pi}\, Z'(\xi_b)]$")
    axes[1].set_title("imaginary part  (Landau resonance)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9, loc="best")
    axes[1].axhline(0, color="gray", lw=0.5)

    axes[2].set_xlabel(r"$\xi_b$")
    axes[2].set_ylabel(r"$|R_{\rm approx} - R_{\rm kin}|$")
    axes[2].set_title("pointwise residual  (log scale)")
    axes[2].grid(alpha=0.3, which="both")
    axes[2].legend(fontsize=9, loc="best")

    fig.suptitle(r"Padé hierarchy vs kinetic $-\sqrt{\pi}\, Z'(\xi_b)$",
                  fontsize=13, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "fig_response_function.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def fig_overshoot_vs_N(u_b: float = 5.0, eps: float = 0.05,
                        Ns: list[int] = [3, 4, 5, 6, 7, 8],
                        ) -> None:
    """γ_max overshoot at canonical (u_b, eps) as a function of Padé order N."""
    from bot.sweep import measure_overshoot
    overshoots = []
    for N in Ns:
        r = measure_overshoot(u_b, eps, N=N)
        overshoots.append(r["overshoot"] * 100)
        print(f"  N={N}: γ_kin={r['gamma_kin']:+.4f}, "
              f"γ_fluid={r['gamma_fluid']:+.4f}, overshoot={overshoots[-1]:+.1f}%")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(Ns, overshoots, "o-", color="C0", lw=2, ms=8)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("Padé order N")
    ax.set_ylabel(r"$\gamma_{\max}$ overshoot  [%]")
    ax.set_title(rf"BoT γ overshoot vs Padé order at  $u_b/v_b={u_b}$, $\varepsilon={eps}$")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "fig_overshoot_vs_N.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def fig_sweep_timedomain(npz_path: Path | None = None) -> None:
    """Time-domain 2D sweep: Padé vs xi-sampled NN closure (shared log scale)."""
    if npz_path is None:
        npz_path = RUN_DIR / "sweep_timedomain.npz"
    d = np.load(npz_path)
    u_b = d["u_b_vals"]; eps = d["eps_vals"]
    over_p = d["overshoot_pade"]
    over_n = d["overshoot_naive"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    extent = [eps[0], eps[-1], u_b[0], u_b[-1]]
    _log_overshoot_heatmap(axes[0], over_p, extent, "Padé N=3")
    im1 = _log_overshoot_heatmap(axes[1], over_n, extent,
                                   r"$\xi_b$-sampled NN closure")
    fig.suptitle(r"Time-domain $\gamma_{\rm eff}$ relative error from generic $\delta E$ IC "
                  r"(shared log scale)", fontsize=12, y=1.02)
    fig.tight_layout()
    _add_log_colorbar(fig, im1)
    out = FIG_DIR / "fig_sweep_timedomain.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def fig_time_domain(npz_path: Path | None = None) -> None:
    """Two panels: |E(t)| evolution under Langmuir IC and generic δE IC.

    Each panel overlays kinetic / Padé N=3 / inference closure. Compare the
    effective growth rate to gamma_kin to assess closure quality.
    """
    if npz_path is None:
        npz_path = RUN_DIR / "time_domain.npz"
    if not npz_path.exists():
        from bot.closed_loop import run_comparison
        run_comparison()
    d = np.load(npz_path)
    t = d["t"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    g_kin = float(d["E_kin_lan"][int(len(t) * 0.6)] / d["E_kin_lan"][int(len(t) * 0.55)])
    # for ref envelope, compute γ_kin from kinetic trace late-time
    from scipy.stats import linregress
    nh = len(t) // 2
    g_kin_fit = linregress(t[nh:], np.log(np.abs(d["E_kin_lan"][nh:]))).slope
    env = np.abs(d["E_kin_lan"][0]) * np.exp(g_kin_fit * t)

    for ax, ic in zip(axes, ["lan", "gen"]):
        ax.semilogy(t, np.abs(d[f"E_kin_{ic}"]),  "k-",  lw=2.0, label="kinetic")
        ax.semilogy(t, np.abs(d[f"E_pade_{ic}"]), "C0--", lw=1.5, label="Padé N=3")
        ax.semilogy(t, np.abs(d[f"E_nai_{ic}"]),  "r--", lw=1.5, label="NN closure")
        ax.semilogy(t, env, "k:", lw=0.8, label=f"$e^{{\\gamma_{{\\rm kin}} t}}$")
        ax.set_xlabel(r"$t \cdot \omega_p$")
        ax.set_ylabel(r"$|E(t)|$")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9, loc="lower right")
    axes[0].set_title("Langmuir-projected IC")
    axes[1].set_title(r"Generic $\delta E$ IC")
    fig.suptitle(rf"Time-domain $|E(t)|$ at canonical $(u_b/v_b=5,\,\varepsilon=0.05,\,k=0.24)$",
                  fontsize=12, y=1.00)
    fig.tight_layout()
    out = FIG_DIR / "fig_time_domain.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def fig_nn_dispersion(
    cases: list[tuple[float, float]] = [(5.0, 0.05), (5.0, 0.20),
                                         (8.0, 0.05), (3.0, 0.05)],
) -> None:
    """NN closure vs kinetic γ(k) at representative operating points.

    For each (u_b, eps), integrate the NN-closed fluid system from a generic
    delta-E IC and extract γ(k) by fitting the late-time slope at each k.
    """
    from bot.closed_loop import (generic_dE_ic, generic_dE_ic_kinetic,
                                  kinetic_evolve, naive_evolve)
    from bot.closures.naive_nn import load as load_naive
    from scipy.stats import linregress
    model = load_naive()

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=False)
    for ax, (u_b, eps) in zip(axes.flat, cases):
        ks, ws_kin, i_max = kinetic_peak(u_b, eps, k_lo=0.10, k_hi=0.55, n_k=20)
        # Fit γ from late-time slope at each k via direct integration
        t_grid = np.linspace(0.0, 60.0, 601)
        gam_nn = np.empty(len(ks))
        for j, k in enumerate(ks):
            y0_fluid = generic_dE_ic(amp=1e-3)
            E = naive_evolve(k, u_b, eps, model, t_grid, y0_fluid)
            nh = len(t_grid) // 2
            gam_nn[j] = linregress(
                t_grid[nh:], np.log(np.maximum(np.abs(E[nh:]), 1e-30))
            ).slope
        ax.plot(ks, ws_kin.imag, "k-", lw=2, label="kinetic")
        ax.plot(ks, gam_nn, "r--", lw=1.6, label="NN closure")
        g_kin = ws_kin[i_max].imag
        over = (gam_nn[i_max] - g_kin) / g_kin if g_kin > 0 else np.nan
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(ks[i_max], color="gray", lw=0.5, ls=":")
        ax.set_xlabel(r"$k$"); ax.set_ylabel(r"$\gamma$ = Im $\omega$")
        ax.set_title(rf"$u_b/v_b={u_b}$, $\varepsilon={eps}$  "
                     rf"(peak overshoot: {over*100:+.3f}\%)")
        ax.grid(alpha=0.3)
        gam_max = max(ws_kin.imag.max(), 0.05)
        ax.set_ylim(-0.05, 1.3 * gam_max)
        ax.legend(fontsize=9, loc="upper right")
    fig.suptitle("NN closure vs kinetic γ(k)", fontsize=13, y=1.00)
    fig.tight_layout()
    out = FIG_DIR / "fig_nn_dispersion.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def fig_sim_dispersion(
    cases: list[tuple[float, float]] = [(5.0, 0.05), (5.0, 0.20),
                                         (8.0, 0.05), (3.0, 0.05)],
) -> None:
    """γ(k) for the sim-trained NN closure vs kinetic at representative points.

    Mirrors fig_nn_dispersion but uses the broad-sim-trained model.
    """
    from bot.closed_loop import generic_dE_ic, sim_evolve
    from bot.train_simdata import load as load_simdata
    from scipy.stats import linregress
    model_sim = load_simdata(broad=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=False)
    for ax, (u_b, eps) in zip(axes.flat, cases):
        ks, ws_kin, i_max = kinetic_peak(u_b, eps, k_lo=0.10, k_hi=0.55, n_k=20)
        t_grid = np.linspace(0.0, 60.0, 601)
        gam_sim = np.empty(len(ks))
        for j, k in enumerate(ks):
            y0_fluid = generic_dE_ic(amp=1e-3)
            E = sim_evolve(k, u_b, eps, model_sim, t_grid, y0_fluid)
            nh = len(t_grid) // 2
            gam_sim[j] = linregress(
                t_grid[nh:], np.log(np.maximum(np.abs(E[nh:]), 1e-30))
            ).slope
        ax.plot(ks, ws_kin.imag, "k-", lw=2, label="kinetic")
        ax.plot(ks, gam_sim, "--", color="C2", lw=1.6, label="sim-trained NN")
        g_kin = ws_kin[i_max].imag
        over = (gam_sim[i_max] - g_kin) / g_kin if g_kin > 0 else np.nan
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(ks[i_max], color="gray", lw=0.5, ls=":")
        ax.set_xlabel(r"$k$"); ax.set_ylabel(r"$\gamma$ = Im $\omega$")
        ax.set_title(rf"$u_b/v_b={u_b}$, $\varepsilon={eps}$  "
                     rf"(peak overshoot: {over*100:+.2f}\%)")
        ax.grid(alpha=0.3)
        gam_max = max(ws_kin.imag.max(), 0.05)
        ax.set_ylim(-0.05, 1.3 * gam_max)
        ax.legend(fontsize=9, loc="upper right")
    fig.suptitle("Sim-trained NN closure vs kinetic γ(k)  "
                  "(broad sweep: 24 op points × 5 k)", fontsize=13, y=1.00)
    fig.tight_layout()
    out = FIG_DIR / "fig_sim_dispersion.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def fig_time_domain_sim(npz_path: Path | None = None) -> None:
    """Time-domain |E(t)| at canonical: kinetic, Padé, sim-trained NN.

    Two panels: Langmuir-projected and generic δE ICs. At canonical the
    sim-trained NN tracks kinetic just like the closed-form one — the
    failure shows up off-distribution (see fig_sweep_timedomain_sim).
    """
    if npz_path is None:
        npz_path = RUN_DIR / "time_domain.npz"
    d = np.load(npz_path)
    t = d["t"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    from scipy.stats import linregress
    nh = len(t) // 2
    g_kin_fit = linregress(t[nh:], np.log(np.abs(d["E_kin_lan"][nh:]))).slope
    env = np.abs(d["E_kin_lan"][0]) * np.exp(g_kin_fit * t)
    for ax, ic in zip(axes, ["lan", "gen"]):
        ax.semilogy(t, np.abs(d[f"E_kin_{ic}"]),  "k-",  lw=2.0, label="kinetic")
        ax.semilogy(t, np.abs(d[f"E_pade_{ic}"]), "C0--", lw=1.5, label="Padé N=3")
        ax.semilogy(t, np.abs(d[f"E_sim_{ic}"]),  "--",  color="C2", lw=1.5,
                     label="sim-trained NN")
        ax.semilogy(t, env, "k:", lw=0.8, label=r"$e^{\gamma_{\rm kin} t}$")
        ax.set_xlabel(r"$t \cdot \omega_p$"); ax.set_ylabel(r"$|E(t)|$")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9, loc="lower right")
    axes[0].set_title("Langmuir-projected IC")
    axes[1].set_title(r"Generic $\delta E$ IC")
    fig.suptitle(r"Sim-trained NN closure at canonical "
                  r"$(u_b/v_b=5,\,\varepsilon=0.05,\,k=0.24)$",
                  fontsize=12, y=1.00)
    fig.tight_layout()
    out = FIG_DIR / "fig_time_domain_sim.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def fig_sweep_timedomain_sim(npz_path: Path | None = None) -> None:
    """Time-domain 2D sweep: Padé vs sim-trained NN closure (shared log scale)."""
    if npz_path is None:
        npz_path = RUN_DIR / "sweep_timedomain.npz"
    d = np.load(npz_path)
    u_b = d["u_b_vals"]; eps = d["eps_vals"]
    over_p = d["overshoot_pade"]
    over_s = d["overshoot_sim"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    extent = [eps[0], eps[-1], u_b[0], u_b[-1]]
    _log_overshoot_heatmap(axes[0], over_p, extent, "Padé N=3")
    im1 = _log_overshoot_heatmap(axes[1], over_s, extent, "sim-trained NN")
    fig.suptitle(r"Time-domain $\gamma_{\rm eff}$ relative error from generic $\delta E$ IC "
                  r"(shared log scale)", fontsize=12, y=1.02)
    fig.tight_layout()
    _add_log_colorbar(fig, im1)
    out = FIG_DIR / "fig_sweep_timedomain_sim.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def fig_xi_coverage() -> None:
    """Schematic: unstable kinetic mode curve in ξ_b plane vs training rectangle.

    Sweeps (u_b, eps, k) over a wide grid and computes ξ_b on the unstable
    mode for each. Overlays the closed-form ξ-sampling rectangle. The point:
    sim trajectories live on the 1D curve; ξ-sampling fills the rectangle.
    """
    from bot.kinetic import most_unstable_kinetic
    u_b_vals = np.linspace(3.0, 10.0, 12)
    eps_vals = np.array([0.01, 0.02, 0.05, 0.10, 0.15, 0.20])
    ks       = np.linspace(0.10, 0.45, 24)

    xis = []
    for u_b in u_b_vals:
        for eps in eps_vals:
            for k in ks:
                try:
                    om = most_unstable_kinetic(k, u_b, eps)
                    if om.imag > 0.005:   # require unstable
                        xi = (om - k*u_b) / k
                        xis.append(xi)
                except Exception:
                    pass
    xis = np.array(xis)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    # closed-form training rectangle
    rect_x = [-3, 3, 3, -3, -3]
    rect_y = [-1, -1, 1.5, 1.5, -1]
    ax.fill(rect_x, rect_y, color="C1", alpha=0.15,
             label=r"closed-form $\xi$ sampling rectangle")
    ax.plot(rect_x, rect_y, "-", color="C1", lw=1.5)
    # 1D curve traced by the unstable mode
    ax.scatter(xis.real, xis.imag, c="C2", s=14, alpha=0.7,
                label="kinetic unstable mode  $\\xi_b$  "
                       r"($u_b,\,\varepsilon,\,k$ varied)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel(r"Re $\xi_b$"); ax.set_ylabel(r"Im $\xi_b$")
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-1.2, 1.7)
    ax.set_title(r"Training-data coverage in $\xi_b$: sim trajectories "
                  r"trace a 1D curve; $\xi$-sampling fills 2D rectangle")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "fig_xi_coverage.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def fig_sweep_timedomain_three(npz_path: Path | None = None) -> None:
    """3-panel time-domain sweep: Padé N=3 / bilinear / xi-sampled NN.

    Shared log-error colour scale. The bilinear panel is the zero-parameter
    nonlinear-in-moments closure U_3 = U_1 U_2 / U_0; it beats Padé in every
    cell but stays orders of magnitude worse than the NN.
    """
    if npz_path is None:
        npz_path = RUN_DIR / "sweep_timedomain.npz"
    d = np.load(npz_path)
    u_b = d["u_b_vals"]; eps = d["eps_vals"]
    over_p = d["overshoot_pade"]
    over_b = d["overshoot_bilin"]
    over_n = d["overshoot_naive"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    extent = [eps[0], eps[-1], u_b[0], u_b[-1]]
    _log_overshoot_heatmap(axes[0], over_p, extent, "Padé N=3 (linear)")
    _log_overshoot_heatmap(axes[1], over_b, extent,
                            r"bilinear $U_3 = U_1 U_2 / U_0$ (zero-param)")
    im2 = _log_overshoot_heatmap(axes[2], over_n, extent,
                                   r"$\xi_b$-sampled NN closure")
    fig.suptitle(r"Time-domain $\gamma_{\rm eff}$ relative error, generic "
                  r"$\delta E$ IC (shared log scale)", fontsize=12, y=1.02)
    fig.tight_layout()
    _add_log_colorbar(fig, im2)
    out = FIG_DIR / "fig_sweep_timedomain_three.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def fig_offmanifold_diagnostic() -> None:
    """Interpretability figure: NN vs bilinear vs kinetic on actual physics.

    Two columns (canonical, marginal operating points). Top row: |E(t)| with
    kinetic / NN / bilinear / Padé. Bottom row: |alpha_closure - alpha_kin|
    over time, evaluated on KINETIC moments (so it is the closure function
    that is being probed, not the trajectory feedback).
    """
    from scipy.integrate import solve_ivp
    from bot.closed_loop import (bilinear_evolve, generic_dE_ic,
                                  generic_dE_ic_kinetic, kinetic_evolve,
                                  pade_evolve)
    from bot.kinetic import kinetic_system_sspace, s_grid
    from bot.closures.pade import pade_coefficients
    from bot.sweep import kinetic_peak

    model_naive = load_naive()

    def kin_full_traj(k, u_b, eps, t_grid, y0, N_v=96, V=6.0):
        A = kinetic_system_sspace(k, u_b, eps, N_v, V)
        ev, V_ = np.linalg.eig(A)
        c = np.linalg.solve(V_, y0)
        s, ds = s_grid(N_v, V)
        out = np.empty((5, len(t_grid)), dtype=complex)
        for i, ti in enumerate(t_grid):
            y = V_ @ (c * np.exp(ev * ti))
            F = y[2:]
            out[0, i] = y[1]
            out[1, i] = ds * np.sum(F)
            out[2, i] = ds * np.sum(s * F)
            out[3, i] = ds * np.sum(s**2 * F)
            out[4, i] = ds * np.sum(s**3 * F)
        return out

    def evolve_nn(k, u_b, eps, t_grid, y0):
        def rhs(t, y_real):
            y = y_real[:5] + 1j * y_real[5:]
            u, E, U0, U1, U2 = y
            if abs(U0) < 1e-15:
                U3 = 0.0 + 0.0j
            else:
                U3 = complex(naive_predict_alpha(model_naive, U1/U0, U2/U0)) * U0
            du = -E; dE = u - eps*(u_b*U0 + U1)
            dU0 = -1j*k*u_b*U0 - 1j*k*U1
            dU1 = -1j*k*u_b*U1 - 1j*k*U2 + E
            dU2 = -1j*k*u_b*U2 - 1j*k*U3
            dy = np.array([du, dE, dU0, dU1, dU2])
            return np.concatenate([dy.real, dy.imag])
        y0r = np.concatenate([y0.real, y0.imag])
        sol = solve_ivp(rhs, (t_grid[0], t_grid[-1]), y0r, t_eval=t_grid,
                        method="RK45", rtol=1e-8, atol=1e-10)
        return sol.y[:5] + 1j*sol.y[5:]

    def panel(ax_E, ax_err, k, u_b, eps, label):
        t = np.linspace(0.0, 80.0, 801)
        y0_f = generic_dE_ic(1e-3)
        y0_k = generic_dE_ic_kinetic(1e-3)
        a_pade = pade_coefficients(3)
        kin = kin_full_traj(k, u_b, eps, t, y0_k)
        E_kin, U0k, U1k, U2k, U3k = kin
        E_pade = pade_evolve(k, u_b, eps, a_pade, t, y0_f)
        E_bil = bilinear_evolve(k, u_b, eps, t, y0_f)
        Y_nn = evolve_nn(k, u_b, eps, t, y0_f)
        E_nn = Y_nn[1]

        ax_E.semilogy(t, np.abs(E_kin),  'k-',   lw=2.2, label='kinetic')
        ax_E.semilogy(t, np.abs(E_nn),   'C0-',  lw=1.4, label='NN')
        ax_E.semilogy(t, np.abs(E_bil),  'C1-',  lw=1.4, label='bilinear')
        ax_E.semilogy(t, np.abs(E_pade), 'C3--', lw=1.2, label='Padé $N=3$')
        ax_E.set_xlabel(r'$t\,\omega_p$'); ax_E.set_ylabel(r'$|E(t)|$')
        ax_E.set_title(label)
        ax_E.legend(fontsize=8, loc='lower right'); ax_E.grid(alpha=0.3)

        alpha_true = U3k / U0k
        r1k, r2k = U1k/U0k, U2k/U0k
        alpha_bil = r1k * r2k
        alpha_nn = np.array([
            complex(naive_predict_alpha(model_naive, complex(r1k[i]), complex(r2k[i])))
            for i in range(len(t))])
        rel = lambda a: np.abs(a - alpha_true) / np.abs(alpha_true)
        ax_err.semilogy(t, rel(alpha_nn),  'C0-', lw=1.6,
                         label=r'NN$(r_1^{\rm kin}, r_2^{\rm kin})$')
        ax_err.semilogy(t, rel(alpha_bil), 'C1-', lw=1.6,
                         label=r'bilinear $= r_1^{\rm kin}\!\cdot r_2^{\rm kin}$')
        ax_err.set_xlabel(r'$t\,\omega_p$')
        ax_err.set_ylabel(r'$|\alpha_{\rm closure} - \alpha_{\rm kin}|/|\alpha_{\rm kin}|$')
        ax_err.set_title('closure-function error on kinetic moments')
        ax_err.legend(fontsize=8, loc='upper right')
        ax_err.grid(alpha=0.3, which='both')
        ax_err.set_ylim(1e-5, 5.0)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    # canonical
    panel(axes[0, 0], axes[1, 0], 0.24, 5.0, 0.05,
          r'canonical: $u_b/v_b=5,\,\varepsilon=0.05,\,k=0.24$')
    # marginal
    u_b2, eps2 = 4.0, 0.02
    ks, _, i_max = kinetic_peak(u_b2, eps2)
    k2 = ks[i_max]
    panel(axes[0, 1], axes[1, 1], k2, u_b2, eps2,
          fr'marginal: $u_b/v_b={u_b2},\,\varepsilon={eps2},\,k={k2:.3f}$')
    fig.tight_layout()
    out = FIG_DIR / "fig_offmanifold_diagnostic.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def fig_landau_damping(kld_lo: float = 0.30, kld_hi: float = 0.60,
                        n_k: int = 13) -> None:
    """Landau damping on a single warm Maxwellian: NN vs Padé vs analytic.

    Same trained naive_mlp.eqx checkpoint as in the BoT figures, applied to
    a stripped-down single-species fluid system (no cold bulk, no beam, no
    eps). Tests generalization beyond BoT.
    """
    from bot.landau_damping import sweep
    model = load_naive()
    kld = np.linspace(kld_lo, kld_hi, n_k)
    omega_kin, omega_nn, omega_pade, _ = sweep(kld, model)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    ax.plot(kld, omega_kin.real, 'k-', lw=2.2, label='kinetic (analytic)')
    ax.plot(kld, omega_nn.real,  'C0o-', lw=1.4, ms=5, label='NN closure')
    ax.plot(kld, omega_pade.real,'C3s--', lw=1.4, ms=5, label='Padé $N=3$')
    ax.set_xlabel(r'$k\lambda_D$')
    ax.set_ylabel(r'$\omega_r / \omega_{pe}$')
    ax.set_title('Bohm-Gross real frequency')
    ax.legend(fontsize=9, loc='lower right'); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.semilogy(kld, -omega_kin.imag, 'k-', lw=2.2, label='kinetic (analytic)')
    ax.semilogy(kld, -omega_nn.imag,  'C0o-', lw=1.4, ms=5, label='NN closure')
    # Padé sign-handling: in this single-Maxwellian setup Padé gives
    # gamma > 0 (unstable). Note in caption instead of plotting.
    ax.set_xlabel(r'$k\lambda_D$')
    ax.set_ylabel(r'$|\gamma| / \omega_{pe}$  (damping rate)')
    ax.set_title('Landau damping rate')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3, which='both')
    ax.text(0.05, 0.95,
            r'Padé $N=3$: $\gamma>0$ (unstable in this setup, not shown)',
            transform=ax.transAxes, va='top', fontsize=8,
            color='C3')

    fig.suptitle(r'Landau damping on a single warm Maxwellian '
                  r'(same NN as BoT, no retraining)', fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "fig_landau_damping.png"
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    fig_response_function()
    fig_sweep_heatmaps()
    fig_dispersion_examples()
    fig_overshoot_vs_N()
    fig_sweep_comparison()
    fig_nn_dispersion()
    fig_sim_dispersion()
    fig_time_domain()
    fig_time_domain_sim()
    fig_sweep_timedomain()
    fig_sweep_timedomain_sim()
    fig_sweep_timedomain_three()
    fig_offmanifold_diagnostic()
    fig_landau_damping()
    fig_xi_coverage()
