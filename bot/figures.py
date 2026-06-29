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


def fig_nn_probe(nn_path: Path = RUN_DIR / "naive_mlp.eqx") -> None:
    """Probe what the N=3 naive NN learned vs the bilinear closure alpha=r1*r2.

    2x2 layout:
    A. On-manifold accuracy heatmap in xi_b plane  (NN residual ~0.3%)
    B. Jacobian mismatch: d(NN)/d(r2) / r1 along real axis  (NN is ~29% softer)
    C. Two-mode mixing: NN, bilinear, direct-alpha errors vs mixing weight
    D. Distribution of improvement across 500 random two-mode pairs

    Key findings:
    - Training target IS bilinear (r1*r2 = alpha exactly on eigenmode manifold)
    - NN reproduces bilinear with ~0.3% residual on manifold
    - NN Jacobian d(NN)/d(r2) ~ 0.71*r1 (vs bilinear = 1.0*r1): NN is "softer"
    - This softness reduces the bilinear cross-term error Cov(xi,beta) off-manifold
    - NN beats bilinear in 99% of random 2-mode pairs; median 2.7x improvement
    """
    import jax
    import jax.numpy as jnp
    from bot.closures.naive_nn import load as load_naive, naive_predict_alpha
    from bot.closures.inference import (alpha as exact_alpha, moment_ratios,
                                         Zprime as Zprime_inf)
    from matplotlib.lines import Line2D

    model = load_naive(key=jax.random.PRNGKey(0))

    def nn_alpha(r1c, r2c):
        return complex(naive_predict_alpha(model, r1c, r2c))

    # ----------------------------------------------------------------
    # Panel A: On-manifold accuracy heatmap
    # ----------------------------------------------------------------
    n_re, n_im = 60, 40
    xi_re = np.linspace(-2.8, 2.8, n_re)
    xi_im = np.linspace(-0.9, 1.4, n_im)
    err_map = np.zeros((n_im, n_re))
    for i in range(n_im):
        for j in range(n_re):
            xi = xi_re[j] + 1j * xi_im[i]
            r1, r2 = moment_ratios(xi)
            a_true = exact_alpha(xi)   # = r1*r2 (bilinear identity)
            err_map[i, j] = abs(nn_alpha(r1, r2) - a_true) / (abs(a_true) + 1e-12)

    # ----------------------------------------------------------------
    # Panel B: Jacobian mismatch on manifold (real xi_b axis)
    # ----------------------------------------------------------------
    xi_scan = np.linspace(-2.4, 2.4, 50)
    jac_ratios = []
    for xi_re_val in xi_scan:
        xi = xi_re_val + 0j
        r1, r2 = moment_ratios(xi)
        feats = jnp.array([r1.real, r1.imag, r2.real, r2.imag])
        J = jax.jacobian(lambda f: model(f))(feats)
        if abs(r1.real) > 0.05:
            jac_ratios.append(float(J[0, 2]) / r1.real)
        else:
            jac_ratios.append(np.nan)
    jac_ratios = np.array(jac_ratios)

    # ----------------------------------------------------------------
    # Panel C: Two-mode mixing errors
    # ----------------------------------------------------------------
    xi_0 = 0.9 + 0.17j
    xi_sec_list  = [-0.5 + 0.2j, 2.0 + 0.0j, 0.0 - 0.5j]
    xi_sec_labels = [r"$\xi_2\!=\!-0.5\!+\!0.2i$", r"$\xi_2\!=\!2.0$",
                     r"$\xi_2\!=\!-0.5i$"]
    xi_sec_colors = ["C1", "C2", "C3"]
    wvals = np.linspace(0.02, 0.98, 60)

    mix_curves = {}
    for xi_sec, lbl in zip(xi_sec_list, xi_sec_labels):
        r1_0, r2_0 = moment_ratios(xi_0)
        r1_s, r2_s = moment_ratios(xi_sec)
        a_0, a_s = exact_alpha(xi_0), exact_alpha(xi_sec)
        U0_0, U0_s = Zprime_inf(xi_0), Zprime_inf(xi_sec)
        e_nn, e_bil, e_da = [], [], []
        for w in wvals:
            denom = w * U0_0 + (1 - w) * U0_s
            if abs(denom) < 1e-12:
                e_nn.append(np.nan); e_bil.append(np.nan); e_da.append(np.nan)
                continue
            r1m = (w * xi_0 * U0_0 + (1 - w) * xi_sec * U0_s) / denom
            r2m = (w * r2_0 * U0_0 + (1 - w) * r2_s * U0_s) / denom
            a_true = (w * a_0 * U0_0 + (1 - w) * a_s * U0_s) / denom
            scale = abs(a_true) + 1e-12
            e_nn.append(abs(nn_alpha(r1m, r2m) - a_true) / scale)
            e_bil.append(abs(r1m * r2m - a_true) / scale)
            e_da.append(abs(exact_alpha(r1m) - a_true) / scale)
        mix_curves[lbl] = (wvals, e_nn, e_bil, e_da)

    # ----------------------------------------------------------------
    # Panel D: Distribution over 500 random 50/50 pairs
    # ----------------------------------------------------------------
    rng = np.random.default_rng(42)
    N = 500
    xi_A = rng.uniform(-2.5, 2.5, N) + 1j * rng.uniform(-0.8, 1.2, N)
    xi_B = rng.uniform(-2.5, 2.5, N) + 1j * rng.uniform(-0.8, 1.2, N)
    nn_errs, bil_errs, da_errs = [], [], []
    for xiA, xiB in zip(xi_A, xi_B):
        r1A, r2A = moment_ratios(xiA)
        r1B, r2B = moment_ratios(xiB)
        aA, aB = exact_alpha(xiA), exact_alpha(xiB)
        U0A, U0B = Zprime_inf(xiA), Zprime_inf(xiB)
        denom = 0.5 * U0A + 0.5 * U0B
        if abs(denom) < 1e-8:
            continue
        r1m = (0.5 * xiA * U0A + 0.5 * xiB * U0B) / denom
        r2m = (0.5 * r2A * U0A + 0.5 * r2B * U0B) / denom
        a_true = (0.5 * aA * U0A + 0.5 * aB * U0B) / denom
        scale = abs(a_true)
        if scale < 1e-8:
            continue
        nn_errs.append(abs(nn_alpha(r1m, r2m) - a_true) / scale)
        bil_errs.append(abs(r1m * r2m - a_true) / scale)
        da_errs.append(abs(exact_alpha(r1m) - a_true) / scale)
    nn_errs = np.array(nn_errs)
    bil_errs = np.array(bil_errs)
    da_errs = np.array(da_errs)

    # ----------------------------------------------------------------
    # Plot
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # --- Panel A: on-manifold heatmap ---
    ax = axes[0, 0]
    err_log = np.log10(np.maximum(err_map, 1e-6))
    im = ax.pcolormesh(xi_re, xi_im, err_log, cmap="RdYlGn_r",
                        vmin=-4, vmax=0, shading="auto")
    rect_kw = dict(color="white", lw=1.2, ls="--", alpha=0.75)
    ax.axhline(-1.0, **rect_kw); ax.axhline(1.5, **rect_kw)
    ax.axvline(-3.0, **rect_kw); ax.axvline(3.0, **rect_kw)
    ax.plot(0.9, 0.17, "w*", ms=11, zorder=5, label="canonical BoT mode")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel(r"Re $\xi_b$"); ax.set_ylabel(r"Im $\xi_b$")
    ax.set_title("(A)  On-manifold NN accuracy\n"
                  r"$\log_{{10}}|\hat\alpha_{{\rm NN}}(r_1,r_2) - \alpha(\xi_b)|/"
                  r"|\alpha(\xi_b)|$"
                  "\n[bilinear is exact; NN residual ~0.3%]")
    ax.legend(fontsize=8)

    # --- Panel B: Jacobian mismatch ---
    ax = axes[0, 1]
    ax.plot(xi_scan, jac_ratios, "C0-", lw=2.5,
             label=r"$\partial_{\rm Re\,r_2}\,\mathrm{Re}(\hat\alpha_{\rm NN})"
                   r"\;/\;\mathrm{Re}(r_1)$")
    ax.axhline(1.0, color="C1", lw=2, ls="--",
                label=r"Bilinear (= 1 everywhere)")
    mean_jac = float(np.nanmean(jac_ratios))
    ax.axhline(mean_jac, color="C0", lw=1.5, ls=":",
                label=f"NN mean = {mean_jac:.2f}")
    ax.fill_between(xi_scan, mean_jac, 1.0, alpha=0.12, color="C3")
    ax.set_xlabel(r"$\xi_b$  (real axis)")
    ax.set_ylabel("Jacobian ratio")
    ax.set_title(f"(B)  Jacobian: NN is softer in $r_2$\n"
                  f"Bilinear ratio = 1.00;  NN mean = {mean_jac:.2f} (std "
                  f"{float(np.nanstd(jac_ratios)):.2f})\n"
                  "[NN under-responds to r2, reducing cross-term errors]")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_ylim(-0.1, 2.0)

    # --- Panel C: Two-mode mixing ---
    ax = axes[1, 0]
    for lbl, col in zip(xi_sec_labels, xi_sec_colors):
        w, e_nn, e_bil, e_da = mix_curves[lbl]
        ax.semilogy(w, e_nn, color=col, lw=2.2)
        ax.semilogy(w, e_bil, color=col, lw=2.2, ls="--")
    ax.axhline(3e-3, color="gray", lw=1.2, ls=":", label="NN on-manifold residual")
    mode_handles = [Line2D([0],[0], color=c, lw=2, label=lbl)
                    for lbl, c in zip(xi_sec_labels, xi_sec_colors)]
    ls_handles = [Line2D([0],[0], color="k", lw=2, label="NN (solid)"),
                   Line2D([0],[0], color="k", lw=2, ls="--", label="Bilinear (dashed)")]
    ax.legend(handles=mode_handles + ls_handles, fontsize=8, ncol=2, loc="upper center")
    ax.set_xlabel(r"mixing weight $w$ (mode-1 fraction)")
    ax.set_ylabel(r"$|\hat\alpha - \alpha_{{\rm true}}|/|\alpha_{{\rm true}}|$")
    ax.set_title(r"(C)  Two-mode mixture: $\xi_1=0.9+0.17i$ + $\xi_2$"
                  r" — NN vs bilinear" + "\n"
                  r"At $w=0.5$: NN beats bilinear 2--4$\times$ across all $\xi_2$" + "\n"
                  r"[at $w=0$ or $1$: single mode, bilinear exact, NN at residual]")
    ax.set_xlim(0, 1); ax.set_ylim(1e-4, 20)
    ax.axvline(0.5, color="gray", lw=0.6, ls=":")
    ax.grid(alpha=0.3, which="both")

    # --- Panel D: Distribution over 500 pairs ---
    ax = axes[1, 1]
    bins = np.logspace(-3, 1.5, 40)
    ax.hist(bil_errs, bins=bins, alpha=0.55, color="C1", density=True,
             label=f"Bilinear (med {np.median(bil_errs):.2f})")
    ax.hist(da_errs, bins=bins, alpha=0.55, color="C3", density=True,
             label=f"Direct-alpha (med {np.median(da_errs):.2f})")
    ax.hist(nn_errs, bins=bins, alpha=0.75, color="C0", density=True,
             label=f"NN  (med {np.median(nn_errs):.2f})")
    ax.axvline(np.median(bil_errs), color="C1", lw=2.5, ls="--")
    ax.axvline(np.median(da_errs), color="C3", lw=2.5, ls="--")
    ax.axvline(np.median(nn_errs), color="C0", lw=2.5, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"$|\hat\alpha - \alpha_{{\rm true}}|/|\alpha_{{\rm true}}|$")
    ax.set_ylabel("density")
    frac_win = sum(nn_errs < bil_errs)
    ax.set_title(f"(D)  Distribution over {len(nn_errs)} random 50/50 two-mode pairs\n"
                  f"NN wins in {frac_win}/{len(nn_errs)} cases  |  "
                  f"median improvement {np.median(bil_errs)/np.median(nn_errs):.1f}x over bilinear")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        r"What the N=3 NN learned beyond bilinear  $\alpha = r_1 r_2$"
        "\n"
        "Training IS bilinear on the eigenmode manifold.  "
        "NN is a soft approximation: its Jacobian w.r.t. $r_2$ is ~0.71x bilinear, "
        "reducing the two-mode covariance error by a median 2.7x.",
        fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "fig_nn_probe.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def fig_nn_jensen_chord(nn_path: Path = RUN_DIR / "naive_mlp.eqx",
                         nn_u2_path: Path = RUN_DIR / "nn_u2_r1.eqx") -> None:
    """Chord-vs-curve: Jensen error across both N=2 (beta) and N=3 (alpha) levels.

    For a two-mode superposition with EQUAL amplitudes, the exact closure value
    required by the kinetic system is the straight chord:

        beta_exact(s)  = (1-s)*beta(xi1)  + s*beta(xi2)   [linear in s -> straight]
        alpha_exact(s) = (1-s)*alpha(xi1) + s*alpha(xi2)  [linear in s -> straight]

    Any single-mode closure evaluates at the mixed input r1(s)=(1-s)*xi1+s*xi2
    and traces a CURVE bowing away from the chord.  Gap = Jensen error.

    Layout: 3 rows x (3 data cols + 1 legend col):
      Row 0: Re(beta)  — N=2 closure level
      Row 1: Re(alpha) — N=3 closure level
      Row 2: Im(alpha) — N=3 closure level
    Colors consistent across rows: direct-beta/direct-alpha share C3 (red);
    NN_u2/NN_naive share C2 (green); bilinear (N=3 only) uses C1 (orange).
    """
    import equinox as eqx
    import jax
    from bot.closures.naive_nn import load as load_naive, naive_predict_alpha
    from bot.closures.inference import alpha as exact_alpha, moment_ratios
    from bot.closures.u2 import beta as direct_beta, model_beta as nn_u2_beta, MLP_u2

    model_n3 = load_naive()
    model_u2 = MLP_u2(hidden=(32, 32, 32), key=jax.random.PRNGKey(42))
    model_u2 = eqx.tree_deserialise_leaves(str(nn_u2_path), model_u2)

    def nn_alpha(r1c, r2c):
        return complex(naive_predict_alpha(model_n3, r1c, r2c))

    def nn_beta(r1c):
        return complex(nn_u2_beta(model_u2, r1c))

    # Three cases: covering growing, crossing, and fully damped regimes
    cases = [
        # (xi1,               xi2,                  column title)
        (0.9 + 0.17j,   2.0  + 0.0j,
         r"Growing $\to$ growing"           + "\n"
         r"$\xi_1\!=\!0.9\!+\!0.17i$,  $\xi_2\!=\!2.0$"),
        (0.9 + 0.17j,   -0.5 + 0.2j,
         r"Growing $\to$ negative-Re"       + "\n"
         r"$\xi_1\!=\!0.9\!+\!0.17i$,  $\xi_2\!=\!-0.5\!+\!0.2i$"),
        (1.225 - 0.612j, 0.164 - 0.078j,
         r"Both damped (Landau regime)"    + "\n"
         r"$\xi_1\!=\!1.22\!-\!0.61i$,  $\xi_2\!=\!0.16\!-\!0.08i$"),
    ]

    s_vals = np.linspace(0, 1, 301)

    def _annot(ax, s, chord, curve, color):
        mid = len(s) // 2
        yrange = max(abs(chord.max() - chord.min()), 1e-2)
        err = abs(curve[mid] - chord[mid])
        if err > 0.04 * yrange:
            ax.annotate(
                fr"$\Delta={err:.2f}$",
                xy=(0.5, curve[mid]),
                xytext=(0.54, curve[mid] + 0.15 * (curve[mid] - chord[mid])),
                fontsize=7, color=color, ha="left",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.7),
            )

    # 3 rows x 4 cols: Re(beta), Re(alpha), Im(alpha); col 3 = legend
    fig, axes = plt.subplots(3, 4, figsize=(16, 11),
                              gridspec_kw={"width_ratios": [1, 1, 1, 0.55]})

    h_chord_b = h_db = h_nn2 = None
    h_chord_a = h_bil = h_da = h_nn3 = None

    for col, (xi1, xi2, col_title) in enumerate(cases):
        r1_1, r2_1 = moment_ratios(xi1)   # r1=xi, r2=beta(xi)
        r1_2, r2_2 = moment_ratios(xi2)
        a1, a2 = exact_alpha(xi1), exact_alpha(xi2)
        b1, b2 = r2_1, r2_2               # beta at endpoints

        # Equal-amplitude linear mixing
        r1_s = np.array([(1 - s) * r1_1 + s * r1_2 for s in s_vals])
        r2_s = np.array([(1 - s) * r2_1 + s * r2_2 for s in s_vals])  # beta chord

        # ---- Row 0: N=2 beta closure ----
        b_chord  = np.array([(1 - s) * b1 + s * b2    for s in s_vals])  # straight!
        b_direct = np.array([direct_beta(r1_s[i])      for i in range(len(s_vals))])
        b_nn2    = np.array([nn_beta(r1_s[i])          for i in range(len(s_vals))])

        ax = axes[0, col]
        cv, dv, nv = b_chord.real, b_direct.real, b_nn2.real
        ax.fill_between(s_vals, cv, dv, alpha=0.15, color="C3", zorder=1)
        ax.fill_between(s_vals, cv, nv, alpha=0.15, color="C2", zorder=1)
        h_chord_b, = ax.plot(s_vals, cv, "k-",  lw=3.0, zorder=5)
        h_db,      = ax.plot(s_vals, dv, color="C3", lw=1.8, ls="--", zorder=3)
        h_nn2,     = ax.plot(s_vals, nv, color="C2", lw=1.8, ls="-.", zorder=4)
        ax.scatter([0, 1], [cv[0], cv[-1]], s=80, zorder=6, color="k", marker="D")
        _annot(ax, s_vals, cv, dv, "C3")
        ax.set_xlabel(r"mixing weight $s$", fontsize=8)
        ax.set_ylabel(r"Re$(\beta)$  [N=2]")
        ax.grid(alpha=0.25)
        ax.set_title(col_title, fontsize=9)

        # ---- Rows 1 & 2: N=3 alpha closure ----
        a_chord    = np.array([(1 - s) * a1 + s * a2      for s in s_vals])  # straight!
        a_bilinear = r1_s * r2_s
        a_direct   = np.array([exact_alpha(r1_s[i])        for i in range(len(s_vals))])
        a_nn       = np.array([nn_alpha(r1_s[i], r2_s[i])  for i in range(len(s_vals))])

        for row, part in enumerate(["real", "imag"]):
            ax = axes[row + 1, col]
            cv = a_chord.real    if part == "real" else a_chord.imag
            bv = a_bilinear.real if part == "real" else a_bilinear.imag
            dv = a_direct.real   if part == "real" else a_direct.imag
            nv = a_nn.real       if part == "real" else a_nn.imag

            ax.fill_between(s_vals, cv, bv, alpha=0.15, color="C1", zorder=1)
            ax.fill_between(s_vals, cv, dv, alpha=0.15, color="C3", zorder=1)
            ax.fill_between(s_vals, cv, nv, alpha=0.15, color="C2", zorder=1)
            h_chord_a, = ax.plot(s_vals, cv, "k-",  lw=3.0, zorder=5)
            h_bil,     = ax.plot(s_vals, bv, color="C1", lw=1.8, ls="--", zorder=4)
            h_da,      = ax.plot(s_vals, dv, color="C3", lw=1.8, ls="--", zorder=3)
            h_nn3,     = ax.plot(s_vals, nv, color="C2", lw=1.8, ls="-.", zorder=4)
            ax.scatter([0, 1], [cv[0], cv[-1]], s=80, zorder=6, color="k", marker="D")
            _annot(ax, s_vals, cv, bv, "C1")
            ax.set_xlabel(r"mixing weight $s$", fontsize=8)
            ax.set_ylabel(r"Re$(\alpha)$  [N=3]" if part == "real"
                          else r"Im$(\alpha)$  [N=3]")
            ax.grid(alpha=0.25)

    # --- Legend panels ---
    for row in range(3):
        axes[row, 3].axis("off")

    axes[0, 3].legend(
        [h_chord_b, h_db, h_nn2],
        [r"$\beta_{\rm exact}(s)=(1{-}s)\beta(\xi_1)+s\beta(\xi_2)$  -- chord",
         r"Direct-$\beta$:  $\beta(r_1(s))$",
         r"NN$_\beta$ (N=2):  $\hat\beta(r_1(s))$"],
        fontsize=8.5, loc="upper left",
        title="N=2 closure  (U2/U0 = beta)",
        title_fontsize=8.5, frameon=True, handlelength=2.5,
    )
    axes[1, 3].legend(
        [h_chord_a, h_bil, h_da, h_nn3],
        [r"$\alpha_{\rm exact}(s)=(1{-}s)\alpha(\xi_1)+s\alpha(\xi_2)$  -- chord",
         r"Bilinear:  $r_1(s)\cdot r_2(s)$",
         r"Direct-$\alpha$:  $\alpha(r_1(s))$",
         r"NN naive (N=3):  $\hat\alpha(r_1(s),r_2(s))$"],
        fontsize=8.5, loc="upper left",
        title=("N=3 closure  (U3/U0 = alpha)\n"
               "Endpoints (diamond): all agree at s=0,1\n"
               "(eigenmode manifold; exact by training)\n"
               "Shaded area = Jensen error"),
        title_fontsize=8, frameon=True, handlelength=2.5,
    )

    fig.suptitle(
        "Chord vs. curve: Jensen error at N=2 (top) and N=3 (middle/bottom) closure levels\n"
        r"Correct by superposition = STRAIGHT chord.  "
        r"Single-mode closures bow away; gap grows with curvature of $\beta$ or $\alpha$.",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    out = FIG_DIR / "fig_nn_jensen_chord.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def fig_n4_chord(nn_n4_path:       Path = RUN_DIR / "n4_nn.eqx",
                  nn_n4_super_path: Path = RUN_DIR / "n4_nn_super.eqx") -> None:
    """Chord vs. curve at the alpha4 (N=4 hierarchy) closure level.

    Shows three lines per panel:
      * Black solid  — correct chord (exact for two-mode superpositions)
      * Red dashed   — direct-alpha4: alpha4(r1 only), Jensen error, irreducible at N=4 too
      * Blue dash-dot  — NN_n4 (single-mode trained): Jensen error but TRAINABLE
      * Green solid  — NN_n4_super (superposition trained): dramatically reduced error

    Key distinction from N=3:
      N=3 Jensen error is IRREDUCIBLE (4 real DOF < 6 unknowns for two-mode state).
      N=4 Jensen error is TRAINABLE  (6 real DOF = 6 unknowns -> unique solution).
      The super-trained curve confirms this: superposition training reduces the gap.

    Layout: 2 rows x (3 data cols + 1 legend col).
    """
    import equinox as eqx
    import jax
    from bot.closures.inference import (alpha4 as exact_alpha4,
                                         alpha as exact_alpha,
                                         moment_ratios_n4)
    from bot.closures.n4_nn import MLP_n4, n4_predict_alpha4, load, load_super

    model_n4       = load()
    model_n4_super = load_super()

    def nn_a4(model, r1c, r2c, r3c):
        return complex(n4_predict_alpha4(model, r1c, r2c, r3c))

    from bot.closures.u2 import beta as exact_beta

    cases = [
        (0.9 + 0.17j,   2.0  + 0.0j,
         r"Growing $\to$ growing"        + "\n"
         r"$\xi_1\!=\!0.9\!+\!0.17i$,  $\xi_2\!=\!2.0$"),
        (0.9 + 0.17j,   -0.5 + 0.2j,
         r"Growing $\to$ negative-Re"    + "\n"
         r"$\xi_1\!=\!0.9\!+\!0.17i$,  $\xi_2\!=\!-0.5\!+\!0.2i$"),
        (1.225 - 0.612j, 0.164 - 0.078j,
         r"Both damped (Landau regime)"  + "\n"
         r"$\xi_1\!=\!1.22\!-\!0.61i$,  $\xi_2\!=\!0.16\!-\!0.08i$"),
    ]

    s_vals = np.linspace(0, 1, 301)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7),
                              gridspec_kw={"width_ratios": [1, 1, 1, 0.65]})

    h_chord = h_direct = h_nn4 = h_nn4s = None

    for col, (xi1, xi2, col_title) in enumerate(cases):
        r1_1, r2_1, r3_1 = moment_ratios_n4(xi1)
        r1_2, r2_2, r3_2 = moment_ratios_n4(xi2)
        a4_1, a4_2 = exact_alpha4(xi1), exact_alpha4(xi2)

        r1_s = np.array([(1-s)*r1_1 + s*r1_2 for s in s_vals])
        r2_s = np.array([(1-s)*r2_1 + s*r2_2 for s in s_vals])
        r3_s = np.array([(1-s)*r3_1 + s*r3_2 for s in s_vals])

        a4_chord  = np.array([(1-s)*a4_1 + s*a4_2 for s in s_vals])
        a4_direct = np.array([exact_alpha4(r1_s[i]) for i in range(len(s_vals))])
        a4_nn     = np.array([nn_a4(model_n4,       r1_s[i], r2_s[i], r3_s[i])
                               for i in range(len(s_vals))])
        a4_nn_s   = np.array([nn_a4(model_n4_super, r1_s[i], r2_s[i], r3_s[i])
                               for i in range(len(s_vals))])

        for row, part in enumerate(["real", "imag"]):
            ax = axes[row, col]
            cv  = a4_chord.real   if part == "real" else a4_chord.imag
            dv  = a4_direct.real  if part == "real" else a4_direct.imag
            nv  = a4_nn.real      if part == "real" else a4_nn.imag
            nsv = a4_nn_s.real    if part == "real" else a4_nn_s.imag

            ax.fill_between(s_vals, cv, dv, alpha=0.12, color="C3", zorder=1)
            ax.fill_between(s_vals, cv, nv, alpha=0.12, color="C0", zorder=1)
            ax.fill_between(s_vals, cv, nsv, alpha=0.15, color="C2", zorder=2)
            h_chord,  = ax.plot(s_vals, cv,  "k-",  lw=3.0, zorder=6)
            h_direct, = ax.plot(s_vals, dv,  color="C3", lw=1.6, ls="--",  zorder=3)
            h_nn4,    = ax.plot(s_vals, nv,  color="C0", lw=1.6, ls="-.",  zorder=4)
            h_nn4s,   = ax.plot(s_vals, nsv, color="C2", lw=2.2, ls="-",   zorder=5)
            ax.scatter([0, 1], [cv[0], cv[-1]], s=80, zorder=7, color="k", marker="D")

            ax.set_xlabel(r"mixing weight $s$", fontsize=8)
            ax.set_ylabel(r"Re$(\alpha_4)$  [N=4]" if part == "real"
                          else r"Im$(\alpha_4)$  [N=4]")
            ax.grid(alpha=0.25)
            if row == 0:
                ax.set_title(col_title, fontsize=9)

    for row in range(2):
        axes[row, 3].axis("off")

    axes[0, 3].legend(
        [h_chord, h_direct, h_nn4, h_nn4s],
        [r"$\alpha_4^{\rm exact}(s)=(1{-}s)\alpha_4(\xi_1)+s\alpha_4(\xi_2)$ — chord",
         r"Direct-$\alpha_4$:  $\alpha_4(r_1(s))$  [only $r_1$]",
         r"NN$_{n4}$ (1-mode train):  $\hat\alpha_4(r_1,r_2,r_3)$",
         r"NN$_{n4}^{\rm super}$ (superposition train)"],
        fontsize=8.5, loc="upper left",
        title="N=4 closure  (U4/U0 = alpha4)\n6 real inputs: (Re/Im of r1,r2,r3)",
        title_fontsize=8.5, frameon=True, handlelength=2.5,
    )
    axes[1, 3].legend(
        [],
        [],
        fontsize=8.5, loc="upper left", frameon=True,
        title=("Endpoints (diamond): exact\n"
               "by training (eigenmode manifold)\n\n"
               "Jensen error = gap to chord.\n\n"
               r"N=4: 6 DOF $=$ 6 unknowns" + "\n"
               r"$\Rightarrow$ superposition training" + "\n"
               r"closes the gap (NN$^{\rm super}$)." + "\n\n"
               r"N=3: 4 DOF $<$ 6 unknowns" + "\n"
               r"$\Rightarrow$ gap is irreducible."),
        title_fontsize=8.5,
    )

    fig.suptitle(
        r"Chord vs. curve at the $\alpha_4$ (N=4 hierarchy) level"
        "\n"
        r"Superposition training (NN$^{\rm super}$) substantially reduces the Jensen "
        r"error — confirming that the N=4 information content is sufficient to resolve "
        r"two-mode states.",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    out = FIG_DIR / "fig_n4_chord.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
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
    print("=== fig_n4_chord ==="); fig_n4_chord()
    print("=== fig_nn_jensen_chord ==="); fig_nn_jensen_chord()
