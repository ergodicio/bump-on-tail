"""Figures for the N=2 (U_2 closure) even-moment analysis.

Panels produced:
  fig_u2_time_domain.png   -- E(t) and closure-error traces at canonical pt
  fig_u2_sweep.png         -- 2D heatmaps: Pade N=2 vs direct-beta overshoot
  fig_u2_response.png      -- beta(xi) vs Pade N=2 rational on the real axis
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bot.closures.pade import pade_coefficients
from bot.closures.u2 import beta as u2_beta, Zprime
import equinox as eqx
import jax

from bot.closed_loop import (
    direct_u2_evolve, kinetic_evolve, pade_u2_evolve,
    langmuir_ic_kinetic, kinetic_ic_to_fluid, kinetic_ic_to_fluid_u2,
    kinetic_eigenmode_ic, fluid_eigenmode_ic_u2, fluid_eigenmode_ic_u3,
    direct_alpha_evolve, naive_evolve,
)
from bot.closures.u2 import MLP_u2, model_beta
from bot.fluid import fluid_modes
from bot.kinetic import kinetic_dispersion
from scipy.optimize import root as _scipy_root
from bot.kinetic import kinetic_system_sspace, s_grid, SQRT_PI


FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)
RUN_DIR = Path(__file__).resolve().parent / "runs"


# Shared heatmap scale
LOG_OVER_VMIN = -4.0
LOG_OVER_VMAX =  1.0
LOG_OVER_CMAP = "magma_r"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fit_gamma(t, E, t_fit_min=25.0):
    from scipy.stats import linregress
    mask = t >= t_fit_min
    logE = np.log(np.maximum(np.abs(E[mask]), 1e-30))
    slope, *_ = linregress(t[mask], logE)
    return float(slope)


def _log_heatmap(ax, over_frac, extent, title, mark=True):
    z = np.log10(np.maximum(np.abs(over_frac), 1e-6))
    im = ax.imshow(z, origin="lower", aspect="auto", extent=extent,
                   cmap=LOG_OVER_CMAP, vmin=LOG_OVER_VMIN, vmax=LOG_OVER_VMAX)
    ax.set_title(title)
    ax.set_xlabel(r"$\varepsilon = n_b/n_0$")
    ax.set_ylabel(r"$u_b/v_b$")
    if mark:
        ax.scatter([0.05], [5.0], c="white", s=60, marker="x", linewidths=2)
        ax.scatter([0.05], [5.0], c="black", s=20, marker="x", linewidths=1.5)
    return im


def _add_colorbar(fig, im):
    cb = fig.colorbar(im, ax=fig.axes, shrink=0.85,
                      label=r"$\log_{10}\,|\gamma_{\rm eff}/\gamma_{\rm kin} - 1|$")
    ticks = [-4, -3, -2, -1, 0, 1]
    cb.set_ticks(ticks)
    cb.set_ticklabels([r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$",
                       r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$"])
    return cb


# ---------------------------------------------------------------------------
# Figure 1: response function comparison on real xi axis
# ---------------------------------------------------------------------------

def fig_u2_response() -> None:
    """Plot -Z'(xi) (kinetic), Pade N=2 rational, and beta(xi) vs xi (real)."""
    xi_real = np.linspace(-3.0, 3.0, 400)

    # Kinetic response -Z'(xi) = U_0/Phi
    neg_Zprime = -np.array([Zprime(x) for x in xi_real])

    # Pade N=2 rational: 1/(xi^2 - a1*xi - a0)
    a = pade_coefficients(2)
    pade2_response = 1.0 / (xi_real**2 - a[1] * xi_real - a[0])

    # beta(xi) on real axis
    beta_real = np.array([u2_beta(x) for x in xi_real])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(xi_real, neg_Zprime.real, "k-", lw=2, label=r"$-Z'(\xi)$ (kinetic)")
    ax.plot(xi_real, pade2_response.real, "b--", lw=1.5, label="Padé N=2 rational")
    ax.set_xlim(-3, 3); ax.set_ylim(-6, 6)
    ax.set_xlabel(r"$\xi_b$ (real)"); ax.set_ylabel("Real part")
    ax.set_title(r"$U_0/\hat\Phi$ response function")
    ax.legend(fontsize=9); ax.axhline(0, color="gray", lw=0.5)

    # Padé-implied beta: U_2/U_0 = a_0 + a_1*xi  (linear in xi)
    beta_pade_re = (a[0] + a[1] * xi_real).real
    beta_pade_im = (a[0] + a[1] * xi_real).imag
    beta_real_re = beta_real.real
    beta_real_im = np.array([u2_beta(x).imag for x in xi_real])

    ax = axes[1]
    ax.plot(xi_real, beta_real_re, "r-",  lw=2,
            label=r"$\mathrm{Re}[\beta]$ exact")
    ax.plot(xi_real, beta_real_im, "r:",  lw=2,
            label=r"$\mathrm{Im}[\beta]$ exact")
    ax.plot(xi_real, beta_pade_re, "b--", lw=1.5,
            label=r"$\mathrm{Re}[\beta]$ Padé N=2")
    ax.plot(xi_real, beta_pade_im, "b-.", lw=1.5,
            label=r"$\mathrm{Im}[\beta]$ Padé N=2")
    ax.set_xlim(-3, 3); ax.set_ylim(-4, 4)
    ax.set_xlabel(r"$\xi_b$ (real)")
    ax.set_ylabel(r"$U_2/U_0$")
    ax.set_title(r"Closure ratio $\beta(\xi_b) = U_2/U_0$ on eigenmode")
    ax.legend(fontsize=8, ncol=2); ax.axhline(0, color="gray", lw=0.5)

    fig.tight_layout()
    out = FIG_DIR / "fig_u2_response.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: time-domain comparison at canonical point
# ---------------------------------------------------------------------------

def fig_u2_time_domain(u_b: float = 5.0, eps: float = 0.05,
                        k: float = 0.24, t_end: float = 80.0,
                        n_t: int = 801) -> None:
    """E(t) under Langmuir-projected and generic delta-E ICs."""
    t_grid = np.linspace(0.0, t_end, n_t)

    # Kinetic IC
    y0_kin_lan, omega_kin = langmuir_ic_kinetic(k, u_b, eps)
    y0_fluid_lan = kinetic_ic_to_fluid_u2(y0_kin_lan, k, u_b)
    # Generic delta-E IC (N=2)
    y0_fluid_gen = np.array([0.0, 1e-3, 0.0, 0.0], dtype=complex)
    N_v = 96
    y0_kin_gen = np.zeros(2 + N_v, dtype=complex); y0_kin_gen[1] = 1e-3

    print(f"  kinetic omega = {omega_kin:.4f}")
    print("  integrating...")

    E_kin_lan  = kinetic_evolve(k, u_b, eps, t_grid, y0_kin_lan)
    E_kin_gen  = kinetic_evolve(k, u_b, eps, t_grid, y0_kin_gen)
    E_p2_lan   = pade_u2_evolve(k, u_b, eps, t_grid, y0_fluid_lan)
    E_p2_gen   = pade_u2_evolve(k, u_b, eps, t_grid, y0_fluid_gen)
    E_dir_lan  = direct_u2_evolve(k, u_b, eps, t_grid, y0_fluid_lan)
    E_dir_gen  = direct_u2_evolve(k, u_b, eps, t_grid, y0_fluid_gen)

    gam_kin = _fit_gamma(t_grid, E_kin_gen)
    gam_p2  = _fit_gamma(t_grid, E_p2_gen)
    gam_dir = _fit_gamma(t_grid, E_dir_gen)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    colors = {"kin": "black", "pade2": "C0", "direct": "C1"}

    for ax, (E_kin, E_p2, E_dir, title) in zip(
        axes,
        [
            (E_kin_lan, E_p2_lan, E_dir_lan, "Langmuir-projected IC"),
            (E_kin_gen, E_p2_gen, E_dir_gen,
             r"Generic $\delta E$ IC  "
             fr"($\gamma_{{kin}}$={gam_kin:+.4f}, Padé={gam_p2:+.4f}, $\beta$={gam_dir:+.4f})"),
        ],
    ):
        ax.semilogy(t_grid, np.abs(E_kin), color=colors["kin"],  lw=1.5,
                    label="Kinetic")
        ax.semilogy(t_grid, np.abs(E_p2),  color=colors["pade2"], lw=1.2,
                    ls="--", label="Padé N=2")
        ax.semilogy(t_grid, np.abs(E_dir), color=colors["direct"], lw=1.2,
                    ls="-.", label=r"Direct $\beta$")
        ax.set_xlabel(r"$t\,[\omega_p^{-1}]$")
        ax.set_ylabel(r"$|E(t)|$")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9)

    fig.suptitle(
        fr"N=2 closure: $u_b/v_b={u_b}$, $\varepsilon={eps}$, $k={k}$",
        fontsize=12,
    )
    fig.tight_layout()
    out = FIG_DIR / "fig_u2_time_domain.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: off-manifold diagnostic along kinetic trajectory
# ---------------------------------------------------------------------------

def fig_u2_offmanifold(u_b: float = 5.0, eps: float = 0.05,
                        k: float = 0.24, t_end: float = 80.0,
                        n_t: int = 801) -> None:
    """Closure-function error |beta_closure - beta_kin| / |beta_kin|.

    Evaluated on the KINETIC moments to isolate the closure quality from
    trajectory feedback.  Both closures are assessed at the same input.

    beta_kin     = U_2^kin / U_0^kin  (exact from kinetic moments)
    beta_direct  = beta(U_1^kin / U_0^kin)  (exact formula at effective xi)
    beta_pade2   = (a_0 + a_1 * U_1^kin / U_0^kin)  (linear Pade)
    """
    t_grid = np.linspace(0.0, t_end, n_t)
    N_v = 96; V = 6.0
    s, ds = s_grid(N_v, V)

    # Generic delta-E IC — kinetic
    y0_kin = np.zeros(2 + N_v, dtype=complex); y0_kin[1] = 1e-3

    # Kinetic system matrix
    A = kinetic_system_sspace(k, u_b, eps, N_v, V)
    eigvals_lam, evecs = np.linalg.eig(A)
    coefs = np.linalg.solve(evecs, y0_kin)

    print("  computing kinetic moments along trajectory...")
    a_pade2 = pade_coefficients(2)

    U0_t = np.empty(n_t, dtype=complex)
    U1_t = np.empty(n_t, dtype=complex)
    U2_t = np.empty(n_t, dtype=complex)

    for i, ti in enumerate(t_grid):
        y = evecs @ (coefs * np.exp(eigvals_lam * ti))
        F = y[2:]
        U0_t[i] = ds * np.sum(F)
        U1_t[i] = ds * np.sum(s * F)
        U2_t[i] = ds * np.sum(s**2 * F)

    beta_kin   = U2_t / U0_t
    mask_small = np.abs(U0_t) < 1e-15

    beta_direct = np.where(
        mask_small,
        0.0 + 0.0j,
        np.array([u2_beta(U1_t[i] / U0_t[i]) if not mask_small[i] else 0j
                  for i in range(n_t)])
    )
    beta_pade2 = np.where(
        mask_small,
        0.0 + 0.0j,
        np.array([a_pade2[0] + a_pade2[1] * (U1_t[i] / U0_t[i]) if not mask_small[i] else 0j
                  for i in range(n_t)])
    )

    err_direct = np.abs(beta_direct - beta_kin) / np.maximum(np.abs(beta_kin), 1e-30)
    err_pade2  = np.abs(beta_pade2  - beta_kin) / np.maximum(np.abs(beta_kin), 1e-30)
    E_t = np.array([
        (evecs @ (coefs * np.exp(eigvals_lam * ti)))[1]
        for ti in t_grid
    ])

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax_top.semilogy(t_grid, np.abs(E_t), "k-", lw=1.5, label="Kinetic |E(t)|")
    ax_top.set_ylabel(r"$|E(t)|$")
    ax_top.legend(fontsize=9)
    ax_top.set_title(
        fr"N=2 off-manifold diagnostic: $u_b={u_b}$, $\varepsilon={eps}$, $k={k}$, generic $\delta E$ IC",
        fontsize=10,
    )

    ax_bot.semilogy(t_grid, err_pade2,  "C0-",  lw=1.5, label="Padé N=2")
    ax_bot.semilogy(t_grid, err_direct, "C1--", lw=1.5, label=r"Direct $\beta$")
    ax_bot.set_ylabel(
        r"$|\beta_{\rm closure} - \beta_{\rm kin}| \,/\, |\beta_{\rm kin}|$"
    )
    ax_bot.set_xlabel(r"$t\,[\omega_p^{-1}]$")
    ax_bot.legend(fontsize=9)
    ax_bot.set_ylim(1e-6, 1e1)

    fig.tight_layout()
    out = FIG_DIR / "fig_u2_offmanifold.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: 2D sweep heatmap (loads pre-computed npz)
# ---------------------------------------------------------------------------

def fig_u2_sweep(path: Path = RUN_DIR / "sweep_u2_timedomain.npz") -> None:
    """Log10|gamma_eff/gamma_kin - 1| heatmaps for Pade N=2 and direct-beta."""
    d = np.load(path)
    u_b = d["u_b_vals"]; eps = d["eps_vals"]
    over_p = d["overshoot_pade2"]
    over_d = d["overshoot_direct"]
    extent = [eps[0], eps[-1], u_b[0], u_b[-1]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _log_heatmap(axes[0], over_p, extent, "Padé N=2")
    im = _log_heatmap(axes[1], over_d, extent, r"Direct $\beta$ closure")
    fig.suptitle(
        r"N=2 closures: $\gamma_{\rm eff}$ relative error vs kinetic "
        r"(generic $\delta E$ IC)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    _add_colorbar(fig, im)
    out = FIG_DIR / "fig_u2_sweep.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Landau damping test — |E(t)| time domain
# ---------------------------------------------------------------------------

def fig_u2_landau_timedomain(nn_path: Path = RUN_DIR / "nn_u2_r1.eqx") -> None:
    """Time-domain |E(t)| for Landau-damped minority-species cases.

    Three cases chosen to span weak/moderate/strong damping.  Generic delta-E
    IC.  Reference slope lines show the kinetic dispersion decay rate.
    """
    from bot.closed_loop import nn_u2_evolve

    model_u2 = MLP_u2(hidden=(32, 32, 32), key=jax.random.PRNGKey(42))
    model_u2 = eqx.tree_deserialise_leaves(str(nn_path), model_u2)

    def _find_mode(k, u_b, eps, seed_im=-0.05):
        def res(x):
            D = kinetic_dispersion(x[0] + 1j*x[1], k, u_b, eps)
            return [D.real, D.imag]
        sol = _scipy_root(res, [1.0, seed_im], method='hybr')
        w = sol.x[0] + 1j*sol.x[1]
        return w if abs(kinetic_dispersion(w, k, u_b, eps)) < 1e-6 else None

    # weak / moderate / strong damping
    cases = [
        (2.0, 0.30, 0.05, r"$u_b=2.0,\;k=0.30$"),
        (1.5, 0.40, 0.05, r"$u_b=1.5,\;k=0.40$"),
        (1.0, 0.50, 0.05, r"$u_b=1.0,\;k=0.50$"),
    ]

    t_grid = np.linspace(0, 30, 601)
    amp    = 1e-3

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for ax, (u_b, k, eps, title) in zip(axes, cases):
        w_kin = _find_mode(k, u_b, eps)
        gamma_disp = w_kin.imag  # negative for damped

        y0_u2  = np.array([0, amp, 0, 0], dtype=complex)
        y0_kin = np.zeros(2 + 96, dtype=complex); y0_kin[1] = amp

        E_kin   = kinetic_evolve(k, u_b, eps, t_grid, y0_kin)
        E_dir   = direct_u2_evolve(k, u_b, eps, t_grid, y0_u2)
        E_nn    = nn_u2_evolve(k, u_b, eps, model_u2, t_grid, y0_u2)
        E_pade  = pade_u2_evolve(k, u_b, eps, t_grid, y0_u2)

        ax.semilogy(t_grid, np.abs(E_kin),  "k-",  lw=1.8, label="Kinetic",        zorder=4)
        ax.semilogy(t_grid, np.abs(E_dir),  color="C1", lw=1.4, ls="--",
                    label=r"Direct $\beta$", zorder=3)
        ax.semilogy(t_grid, np.abs(E_nn),   color="C2", lw=1.4, ls="-.",
                    label=r"NN$(r_1)$",     zorder=3)
        ax.semilogy(t_grid, np.abs(E_pade), color="C0", lw=1.4, ls=":",
                    label="Padé N=2",       zorder=3)

        # reference slope: amp * exp(gamma_disp * t)
        t_ref = np.array([0, 25])
        ax.semilogy(t_ref, amp * np.exp(gamma_disp * t_ref),
                    "k--", lw=0.9, alpha=0.5,
                    label=fr"$e^{{\gamma_{{\rm kin}}\,t}}$, $\gamma={gamma_disp:.3f}$")

        ax.set_xlabel(r"$t\;[\omega_p^{-1}]$")
        ax.set_ylabel(r"$|E(t)|$")
        ax.set_title(title + fr",  $\varepsilon=0.05$" + "\n"
                     + fr"$\xi_b = {(w_kin - k*u_b)/k:.3f}$", fontsize=9)
        ax.legend(fontsize=8, loc="lower left")
        ax.set_xlim(0, 30)
        ax.set_ylim(1e-10, 1e-1)

    fig.suptitle("N=2 closure: Landau-damped minority species — time domain\n"
                 r"(generic $\delta E$ IC)", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "fig_u2_landau_timedomain.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5b: Landau damping — time-domain with dispersion reference slope
# ---------------------------------------------------------------------------

def fig_u2_landau_dispref(nn_u2_path: Path = RUN_DIR / "nn_u2_r1.eqx",
                           nn_u3_path: Path = RUN_DIR / "naive_mlp.eqx") -> None:
    """Time-domain |E(t)| for the five Landau-damped cases in fig_u2_landau_damping.

    No kinetic simulation is shown.  Instead, the reference decay rate
    gamma_kin is taken directly from the dispersion solver and plotted as
    a black dashed slope  amp * exp(gamma_kin * t).  This avoids the
    fundamental limitation that the Landau mode is not a discrete eigenvalue
    of the finite-velocity-grid kinetic system.

    ICs: eigenmode-projected for all fluid closures (so on-eigenmode accuracy
    is cleanly separated from off-eigenmode Jensen error).

    Closures shown: Direct-beta N=2, NN(r1) N=2, Direct-alpha N=3,
                    NN naive N=3, Pade N=2 (faint, negative control).
    Layout: 2 rows x 3 columns; 5 panels filled left-to-right, last slot = legend.
    """
    from bot.closed_loop import nn_u2_evolve, inference_evolve, direct_alpha_evolve, naive_evolve
    from bot.closures.naive_nn import load as load_naive
    from bot.train_inference import load as load_inference

    # Load NN models
    model_u2  = MLP_u2(hidden=(32, 32, 32), key=jax.random.PRNGKey(42))
    model_u2  = eqx.tree_deserialise_leaves(str(nn_u2_path), model_u2)
    model_u3n = load_naive()
    model_u3i = load_inference()

    def _find_mode(k, u_b, eps, seed_im=-0.05):
        def res(x):
            D = kinetic_dispersion(x[0] + 1j*x[1], k, u_b, eps)
            return [D.real, D.imag]
        sol = _scipy_root(res, [1.0, seed_im], method='hybr')
        w = sol.x[0] + 1j*sol.x[1]
        return w if abs(kinetic_dispersion(w, k, u_b, eps)) < 1e-6 else None

    # Same 5 cases as fig_u2_landau_damping
    cases = [
        (0.5, 0.50, 0.05, r"$u_b{=}0.5,\;k{=}0.50$"),
        (1.0, 0.40, 0.05, r"$u_b{=}1.0,\;k{=}0.40$"),
        (1.0, 0.50, 0.05, r"$u_b{=}1.0,\;k{=}0.50$"),
        (1.5, 0.40, 0.05, r"$u_b{=}1.5,\;k{=}0.40$"),
        (2.0, 0.30, 0.05, r"$u_b{=}2.0,\;k{=}0.30$"),
    ]

    amp    = 1e-3
    t_end  = 80.0
    t_grid = np.linspace(0, t_end, 1601)
    t_ref  = np.array([0.0, t_end])

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.0))
    ax_flat = axes.flatten()

    # Storage for legend handles (built on first panel)
    h_dir2 = h_nn2 = h_dir3 = h_nn3n = h_pade = h_ref = None

    for idx, (u_b, k, eps, title) in enumerate(cases):
        ax = ax_flat[idx]
        w_kin      = _find_mode(k, u_b, eps)
        gamma_disp = w_kin.imag
        xi_b       = (w_kin - k * u_b) / k

        # Eigenmode ICs for each fluid system
        y0_u2 = fluid_eigenmode_ic_u2(k, u_b, w_kin, amp=amp)
        y0_u3 = fluid_eigenmode_ic_u3(k, u_b, w_kin, amp=amp)

        # Integrate fluid closures
        E_dir2 = direct_u2_evolve(k, u_b, eps, t_grid, y0_u2)
        E_nn2  = nn_u2_evolve(k, u_b, eps, model_u2, t_grid, y0_u2)
        E_dir3 = direct_alpha_evolve(k, u_b, eps, t_grid, y0_u3)
        E_nn3n = naive_evolve(k, u_b, eps, model_u3n, t_grid, y0_u3)
        E_pade = pade_u2_evolve(k, u_b, eps, t_grid, y0_u2)

        # Reference slope from dispersion solver (no kinetic simulation)
        E_slope = amp * np.exp(gamma_disp * t_ref)

        h_ref,  = ax.semilogy(t_ref,   E_slope,        "k--",  lw=2.5, zorder=6,
                              label=fr"$e^{{\gamma_{{\rm kin}}t}}$, $\gamma={gamma_disp:.3f}$")
        h_dir2, = ax.semilogy(t_grid, np.abs(E_dir2), color="C1", lw=1.5, ls="--",  zorder=4,
                              label=r"Direct $\beta$, N=2")
        h_nn2,  = ax.semilogy(t_grid, np.abs(E_nn2),  color="C2", lw=1.5, ls="-.", zorder=4,
                              label=r"NN$(r_1)$, N=2")
        h_dir3, = ax.semilogy(t_grid, np.abs(E_dir3), color="C3", lw=1.5, ls="--",  zorder=3,
                              label=r"Direct $\alpha$, N=3")
        h_nn3n, = ax.semilogy(t_grid, np.abs(E_nn3n), color="C4", lw=1.5, ls="-.", zorder=3,
                              label=r"NN naive$(r_1,r_2)$, N=3")
        h_pade, = ax.semilogy(t_grid, np.abs(E_pade), color="C0", lw=1.0, ls=":",  zorder=2,
                              alpha=0.7, label="Padé N=2")

        ax.set_title(title + fr",  $\varepsilon={eps}$" + "\n"
                     + fr"$\xi_b={xi_b:.3f}$,  $\gamma={gamma_disp:.3f}$",
                     fontsize=9)
        ax.set_xlim(0, t_end)
        ax.set_ylim(top=2e-3)
        ax.set_xlabel(r"$t\;[\omega_p^{-1}]$")
        ax.set_ylabel(r"$|E(t)|$")

    # Last panel: legend
    ax_leg = ax_flat[5]
    ax_leg.axis("off")
    handles = [h_ref, h_dir2, h_nn2, h_dir3, h_nn3n, h_pade]
    labels  = [
        r"$e^{\gamma_{\rm kin}t}$ — dispersion solver (truth)",
        r"Direct $\beta$, N=2",
        r"NN$(r_1)$, N=2",
        r"Direct $\alpha$, N=3",
        r"NN naive$(r_1,r_2)$, N=3",
        r"Padé N=2",
    ]
    ax_leg.legend(handles, labels, loc="center", fontsize=9.5,
                  title="Eigenmode IC for all fluid closures\n"
                        r"(reference $e^{\gamma t}$ from dispersion solver)",
                  title_fontsize=8.5, frameon=True)

    fig.suptitle(
        "N=2 vs N=3 closures: Landau-damped BoT — time domain\n"
        r"Reference slope $e^{\gamma_{\rm kin}t}$ from dispersion solver; "
        r"no kinetic simulation (Landau mode $\notin$ discrete spectrum of finite-grid Vlasov)",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    out = FIG_DIR / "fig_u2_landau_dispref.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: Landau damping test — dispersion comparison
# ---------------------------------------------------------------------------

def fig_u2_landau_damping(nn_path: Path = RUN_DIR / "nn_u2_r1.eqx") -> None:
    """Bar + scatter figure comparing closure growth rates for Landau-damped cases.

    Left panel : γ values for kinetic, direct-β, NN(r₁), Padé N=2 at 5 test cases.
    Right panel: ξ_b locations in the complex plane with the training rectangle.
    """

    # --- helpers ---
    def _find_mode(k, u_b, eps, seed_im=-0.05):
        def res(x):
            D = kinetic_dispersion(x[0] + 1j*x[1], k, u_b, eps)
            return [D.real, D.imag]
        sol = _scipy_root(res, [1.0, seed_im], method='hybr')
        w = sol.x[0] + 1j*sol.x[1]
        return w if abs(kinetic_dispersion(w, k, u_b, eps)) < 1e-6 else None

    def _fluid_disp(k, u_b, eps, a, omega0, n_iter=30):
        omega = omega0
        for _ in range(n_iter):
            cands = fluid_modes(k, u_b, eps, a)
            on = cands[np.argmin(np.abs(cands - omega))]
            if abs(on - omega) < 1e-10:
                break
            omega = on
        return omega

    def _direct_disp(k, u_b, eps, omega0, n_iter=30):
        omega = omega0
        for _ in range(n_iter):
            xi_b = (omega - k*u_b) / k
            a = np.array([u2_beta(xi_b), 0.0+0.0j])
            cands = fluid_modes(k, u_b, eps, a)
            on = cands[np.argmin(np.abs(cands - omega))]
            if abs(on - omega) < 1e-10:
                break
            omega = on
        return omega

    def _nn_disp(k, u_b, eps, omega0, model, n_iter=30):
        omega = omega0
        for _ in range(n_iter):
            xi_b = (omega - k*u_b) / k
            beta_hat = complex(model_beta(model, xi_b))
            a = np.array([beta_hat, 0.0+0.0j])
            cands = fluid_modes(k, u_b, eps, a)
            on = cands[np.argmin(np.abs(cands - omega))]
            if abs(on - omega) < 1e-10:
                break
            omega = on
        return omega

    # --- load NN ---
    model_u2 = MLP_u2(hidden=(32, 32, 32), key=jax.random.PRNGKey(42))
    model_u2 = eqx.tree_deserialise_leaves(str(nn_path), model_u2)

    # --- test cases ---
    cases = [
        (0.5, 0.50, 0.05),
        (1.0, 0.40, 0.05),
        (1.0, 0.50, 0.05),
        (1.5, 0.40, 0.05),
        (2.0, 0.30, 0.05),
    ]
    labels = [fr"$u_b$={u},$\,k$={k}" for u, k, _ in cases]

    g_kin, g_dir, g_nn, g_pade = [], [], [], []
    xi_b_pts = []
    a_p2 = pade_coefficients(2)

    for u_b, k, eps in cases:
        w_k = _find_mode(k, u_b, eps)
        w_d = _direct_disp(k, u_b, eps, w_k)
        w_n = _nn_disp(k, u_b, eps, w_k, model_u2)
        cands = fluid_modes(k, u_b, eps, a_p2)
        w_p = cands[np.argmin(np.abs(cands - w_k))]
        g_kin.append(w_k.imag);  g_dir.append(w_d.imag)
        g_nn.append(w_n.imag);   g_pade.append(w_p.imag)
        xi_b_pts.append((w_k - k*u_b)/k)

    n = len(cases)
    x = np.arange(n)
    w = 0.18

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5))

    # --- left: γ bar chart ---
    ax_l.bar(x - 1.5*w, g_kin,  w, label="Kinetic",        color="k",  alpha=0.85)
    ax_l.bar(x - 0.5*w, g_dir,  w, label=r"Direct $\beta$", color="C1", alpha=0.85)
    ax_l.bar(x + 0.5*w, g_nn,   w, label=r"NN$(r_1)$",      color="C2", alpha=0.85)
    ax_l.bar(x + 1.5*w, g_pade, w, label="Padé N=2",        color="C0", alpha=0.85)

    ax_l.axhline(0, color="gray", lw=0.8, ls="--")
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(labels, fontsize=8)
    ax_l.set_ylabel(r"$\gamma = \mathrm{Im}(\omega)$")
    ax_l.set_title("Landau damping: dispersion growth rates\n"
                   r"(negative = damped, positive = spurious growth)")
    ax_l.legend(fontsize=9)

    # --- right: ξ_b in complex plane ---
    # training rectangle
    from matplotlib.patches import Rectangle
    rect = Rectangle((-3, -1), 6, 2.5, linewidth=1.5, edgecolor="C0",
                     facecolor="C0", alpha=0.10, label="Training rectangle")
    ax_r.add_patch(rect)

    xi_re = [z.real for z in xi_b_pts]
    xi_im = [z.imag for z in xi_b_pts]
    sc = ax_r.scatter(xi_re, xi_im, c=g_kin, cmap="RdBu",
                      s=100, zorder=5, edgecolors="k", linewidths=0.8,
                      label=r"$\xi_b$ (colour = $\gamma_{\rm kin}$)")
    plt.colorbar(sc, ax=ax_r, label=r"$\gamma_{\rm kin}$")

    for i, (xr, xi, lbl) in enumerate(zip(xi_re, xi_im, labels)):
        ax_r.annotate(lbl, (xr, xi), textcoords="offset points",
                      xytext=(6, 4), fontsize=7)

    ax_r.axhline(0, color="gray", lw=0.8, ls="--")
    ax_r.axvline(0, color="gray", lw=0.8, ls="--")
    ax_r.set_xlim(-3.3, 3.3)
    ax_r.set_ylim(-1.3, 1.7)
    ax_r.set_xlabel(r"$\mathrm{Re}(\xi_b)$")
    ax_r.set_ylabel(r"$\mathrm{Im}(\xi_b)$")
    ax_r.set_title(r"$\xi_b$ locations vs training rectangle")
    ax_r.legend(fontsize=9, loc="upper left")

    fig.suptitle("N=2 closure: Landau-damped minority species", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "fig_u2_landau_damping.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7: Landau damping — eigenmode-projected IC (clean exponential decay)
# ---------------------------------------------------------------------------

def fig_u2_landau_eigenmode(nn_u2_path: Path = RUN_DIR / "nn_u2_r1.eqx",
                             nn_u3_path: Path = RUN_DIR / "naive_mlp.eqx") -> None:
    """Two-row figure: eigenmode IC (row 1) vs generic δE IC (row 2).

    Row 1 — eigenmode-projected IC:
        Pure eigenmode excitation; closures exact on eigenmode (direct-β/α)
        decay cleanly; NN error accumulates at late time.

    Row 2 — generic δE IC (u=U_n=0, E=amp):
        All fluid modes excited simultaneously.  This is the off-eigenmode
        (Jensen-error) regime.  Comparison shows whether the extra input r_2
        in the N=3 NN buys accuracy over the N=2 NN when the beam state is
        not on the single-eigenmode manifold.

    Closures: Direct-β N=2, NN(r1) N=2, Direct-α N=3, NN(r1,r2) N=3,
              Padé N=2 (faint reference).
    """
    from bot.closed_loop import nn_u2_evolve, inference_evolve
    from bot.closures.naive_nn import load as load_naive
    from bot.train_inference import load as load_inference

    # Load models
    model_u2  = MLP_u2(hidden=(32, 32, 32), key=jax.random.PRNGKey(42))
    model_u2  = eqx.tree_deserialise_leaves(str(nn_u2_path), model_u2)
    model_u3n = load_naive()
    model_u3i = load_inference()

    def _find_mode(k, u_b, eps, seed_im=-0.05):
        def res(x):
            D = kinetic_dispersion(x[0] + 1j*x[1], k, u_b, eps)
            return [D.real, D.imag]
        sol = _scipy_root(res, [1.0, seed_im], method='hybr')
        w = sol.x[0] + 1j*sol.x[1]
        return w if abs(kinetic_dispersion(w, k, u_b, eps)) < 1e-6 else None

    cases = [
        (2.0, 0.30, 0.05, r"$u_b=2.0,\;k=0.30$"),
        (1.5, 0.40, 0.05, r"$u_b=1.5,\;k=0.40$"),
        (1.0, 0.50, 0.05, r"$u_b=1.0,\;k=0.50$"),
    ]

    amp = 1e-3
    t_end_eig = 80.0;  t_grid_eig = np.linspace(0, t_end_eig, 1601)
    t_end_gen = 30.0;  t_grid_gen = np.linspace(0, t_end_gen,  601)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.0))
    row_labels = ["Eigenmode IC", r"Generic $\delta E$ IC"]

    # Shared legend handles (built once)
    _legend_handles = None

    for col, (u_b, k, eps, title) in enumerate(cases):
        w_kin      = _find_mode(k, u_b, eps)
        gamma_disp = w_kin.imag
        xi_b       = (w_kin - k * u_b) / k

        # ---- ICs ----
        y0_eig_kin = kinetic_eigenmode_ic(k, u_b, eps, w_kin, amp=amp)
        y0_eig_u2  = fluid_eigenmode_ic_u2(k, u_b, w_kin, amp=amp)
        y0_eig_u3  = fluid_eigenmode_ic_u3(k, u_b, w_kin, amp=amp)

        y0_gen_kin = np.zeros(2 + 96, dtype=complex); y0_gen_kin[1] = amp
        y0_gen_u2  = np.array([0, amp, 0, 0], dtype=complex)
        y0_gen_u3  = np.array([0, amp, 0, 0, 0], dtype=complex)

        # ---- row 0: eigenmode IC ----
        ax = axes[0, col]
        E_kin   = kinetic_evolve(k, u_b, eps, t_grid_eig, y0_eig_kin)
        E_dir2  = direct_u2_evolve(k, u_b, eps, t_grid_eig, y0_eig_u2)
        E_nn2   = nn_u2_evolve(k, u_b, eps, model_u2,  t_grid_eig, y0_eig_u2)
        E_dir3  = direct_alpha_evolve(k, u_b, eps, t_grid_eig, y0_eig_u3)
        E_nn3n  = naive_evolve(k, u_b, eps, model_u3n, t_grid_eig, y0_eig_u3)
        E_nn3i  = inference_evolve(k, u_b, eps, model_u3i, t_grid_eig, y0_eig_u3)
        E_pade  = pade_u2_evolve(k, u_b, eps, t_grid_eig, y0_eig_u2)

        t_ref  = np.array([0.0, t_end_eig])
        E_ref  = amp * np.exp(gamma_disp * t_ref)

        h_kin,  = ax.semilogy(t_grid_eig, np.abs(E_kin),  "k-",  lw=2.0)
        h_dir2, = ax.semilogy(t_grid_eig, np.abs(E_dir2), color="C1", lw=1.5, ls="--")
        h_nn2,  = ax.semilogy(t_grid_eig, np.abs(E_nn2),  color="C2", lw=1.5, ls="-.")
        h_dir3, = ax.semilogy(t_grid_eig, np.abs(E_dir3), color="C3", lw=1.5, ls="--")
        h_nn3n, = ax.semilogy(t_grid_eig, np.abs(E_nn3n), color="C4", lw=1.5, ls="-.")
        h_nn3i, = ax.semilogy(t_grid_eig, np.abs(E_nn3i), color="C5", lw=1.5, ls="-.")
        h_pade, = ax.semilogy(t_grid_eig, np.abs(E_pade), color="C0", lw=0.9, ls=":",
                              alpha=0.5)
        h_ref,  = ax.semilogy(t_ref, E_ref, "k--", lw=1.0, alpha=0.4)

        ax.set_title(title + fr",  $\varepsilon=0.05$" + "\n"
                     + fr"$\xi_b = {xi_b:.3f}$,  $\gamma={gamma_disp:.3f}$",
                     fontsize=9)
        ax.set_xlim(0, t_end_eig);  ax.set_ylim(1e-30, 1e-1)
        ax.set_xlabel(r"$t\;[\omega_p^{-1}]$")
        ax.set_ylabel(r"$|E(t)|$")
        if col == 0:
            ax.text(-0.18, 0.5, row_labels[0], transform=ax.transAxes,
                    fontsize=10, rotation=90, va='center', fontweight='bold')

        # ---- row 1: generic δE IC — |E(t)| with kinetic ground truth ----
        ax = axes[1, col]
        E_kin   = kinetic_evolve(k, u_b, eps, t_grid_gen, y0_gen_kin)
        E_dir2  = direct_u2_evolve(k, u_b, eps, t_grid_gen, y0_gen_u2)
        E_nn2   = nn_u2_evolve(k, u_b, eps, model_u2,  t_grid_gen, y0_gen_u2)
        E_dir3  = direct_alpha_evolve(k, u_b, eps, t_grid_gen, y0_gen_u3)
        E_nn3n  = naive_evolve(k, u_b, eps, model_u3n, t_grid_gen, y0_gen_u3)
        E_nn3i  = inference_evolve(k, u_b, eps, model_u3i, t_grid_gen, y0_gen_u3)
        E_pade  = pade_u2_evolve(k, u_b, eps, t_grid_gen, y0_gen_u2)

        ax.semilogy(t_grid_gen, np.abs(E_kin),  "k-",  lw=2.0, zorder=5)
        ax.semilogy(t_grid_gen, np.abs(E_dir2), color="C1", lw=1.5, ls="--", zorder=4)
        ax.semilogy(t_grid_gen, np.abs(E_nn2),  color="C2", lw=1.5, ls="-.", zorder=4)
        ax.semilogy(t_grid_gen, np.abs(E_dir3), color="C3", lw=1.5, ls="--", zorder=3)
        ax.semilogy(t_grid_gen, np.abs(E_nn3n), color="C4", lw=1.5, ls="-.", zorder=3)
        ax.semilogy(t_grid_gen, np.abs(E_nn3i), color="C5", lw=1.5, ls="-.", zorder=3)
        ax.semilogy(t_grid_gen, np.abs(E_pade), color="C0", lw=0.9, ls=":",
                    alpha=0.5, zorder=2)

        ax.set_xlim(0, t_end_gen);  ax.set_ylim(1e-7, 1e-1)
        ax.set_xlabel(r"$t\;[\omega_p^{-1}]$")
        ax.set_ylabel(r"$|E(t)|$")
        if col == 0:
            ax.text(-0.18, 0.5, row_labels[1], transform=ax.transAxes,
                    fontsize=10, rotation=90, va='center', fontweight='bold')

    # Shared legend
    labels = [
        "Kinetic",
        r"Direct $\beta$, N=2",
        r"NN$(r_1)$, N=2",
        r"Direct $\alpha$, N=3",
        r"NN naive$(r_1,r_2)$, N=3",
        r"NN inf$(r_1,r_2)$, N=3",
        "Padé N=2",
        r"$e^{\gamma_{\rm kin} t}$",
    ]
    handles = [h_kin, h_dir2, h_nn2, h_dir3, h_nn3n, h_nn3i, h_pade, h_ref]
    fig.legend(handles, labels, loc="lower center", ncol=4,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.04),
               title="Row 1 legend (same colors apply to row 2)", title_fontsize=8)

    fig.suptitle(
        "N=2 vs N=3 closures: Landau-damped minority species\n"
        r"Row 1: eigenmode IC (pure $e^{\gamma t}$) — "
        r"Row 2: generic $\delta E$ IC (off-eigenmode / multi-mode)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    out = FIG_DIR / "fig_u2_landau_eigenmode.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 8: Landau damping — strongly-damped cases
# ---------------------------------------------------------------------------

def fig_u2_landau_strong(nn_u2_path: Path = RUN_DIR / "nn_u2_r1.eqx",
                          nn_u3_path: Path = RUN_DIR / "naive_mlp.eqx") -> None:
    """Same two-row layout as fig_u2_landau_eigenmode but for three
    strongly-damped cases with |gamma| >> 0.17 (the max in Fig 18).

    Cases chosen so that:
      - All xi_b lie inside the NN training rectangle (Im > -1)
      - |Im(omega)| / Delta_vK > 4 so kinetic eigenmode IC is clean
      - gamma spans roughly -0.31, -0.44, -0.64 (2x to 4x Figure 18 max)

    Shorter time windows (t_eig=40, t_gen=20) because the signal decays
    several orders of magnitude per unit time.
    """
    from bot.closed_loop import nn_u2_evolve, inference_evolve
    from bot.closures.naive_nn import load as load_naive
    from bot.train_inference import load as load_inference

    # Load models
    model_u2  = MLP_u2(hidden=(32, 32, 32), key=jax.random.PRNGKey(42))
    model_u2  = eqx.tree_deserialise_leaves(str(nn_u2_path), model_u2)
    model_u3n = load_naive()
    model_u3i = load_inference()

    def _find_mode(k, u_b, eps, seed_im=-0.3):
        """Find the most-damped root within the training rectangle."""
        def res(x):
            D = kinetic_dispersion(x[0] + 1j*x[1], k, u_b, eps)
            return [D.real, D.imag]
        best = None
        for seed_re in np.linspace(0.5, 2.5, 10):
            for sim in np.linspace(-0.1, -0.9, 10):
                try:
                    sol = _scipy_root(res, [seed_re, sim], method='hybr')
                    w = sol.x[0] + 1j*sol.x[1]
                    if abs(kinetic_dispersion(w, k, u_b, eps)) > 1e-6:
                        continue
                    if w.imag >= -0.05:
                        continue
                    xi_b = (w - k*u_b)/k
                    if xi_b.imag < -1.0 or xi_b.imag > 0:
                        continue
                    if xi_b.real < 0:
                        continue
                    if best is None or w.imag < best.imag:
                        best = w
                except Exception:
                    pass
        return best

    # Three strongly-damped cases: gamma ~ -0.31, -0.44, -0.64
    # All xi_b inside training rectangle; |Im|/Delta_vK = 4.9, 5.8, 7.4
    cases = [
        (1.5, 0.50, 0.05, r"$u_b=1.5,\;k=0.50$"),
        (1.0, 0.60, 0.05, r"$u_b=1.0,\;k=0.60$"),
        (1.0, 0.70, 0.05, r"$u_b=1.0,\;k=0.70$"),
    ]

    amp       = 1e-3
    t_end_eig = 40.0;  t_grid_eig = np.linspace(0, t_end_eig, 801)
    t_end_gen = 20.0;  t_grid_gen = np.linspace(0, t_end_gen, 401)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.0))
    row_labels = ["Eigenmode IC", r"Generic $\delta E$ IC"]

    h_kin = h_dir2 = h_nn2 = h_dir3 = h_nn3n = h_nn3i = h_pade = h_ref = None

    for col, (u_b, k, eps, title) in enumerate(cases):
        w_kin      = _find_mode(k, u_b, eps)
        gamma_disp = w_kin.imag
        xi_b       = (w_kin - k * u_b) / k

        # ICs
        y0_eig_kin = kinetic_eigenmode_ic(k, u_b, eps, w_kin, amp=amp)
        y0_eig_u2  = fluid_eigenmode_ic_u2(k, u_b, w_kin, amp=amp)
        y0_eig_u3  = fluid_eigenmode_ic_u3(k, u_b, w_kin, amp=amp)

        y0_gen_kin = np.zeros(2 + 96, dtype=complex); y0_gen_kin[1] = amp
        y0_gen_u2  = np.array([0, amp, 0, 0], dtype=complex)
        y0_gen_u3  = np.array([0, amp, 0, 0, 0], dtype=complex)

        # ---- row 0: eigenmode IC ----
        ax = axes[0, col]
        E_kin  = kinetic_evolve(k, u_b, eps, t_grid_eig, y0_eig_kin)
        E_dir2 = direct_u2_evolve(k, u_b, eps, t_grid_eig, y0_eig_u2)
        E_nn2  = nn_u2_evolve(k, u_b, eps, model_u2,  t_grid_eig, y0_eig_u2)
        E_dir3 = direct_alpha_evolve(k, u_b, eps, t_grid_eig, y0_eig_u3)
        E_nn3n = naive_evolve(k, u_b, eps, model_u3n, t_grid_eig, y0_eig_u3)
        E_nn3i = inference_evolve(k, u_b, eps, model_u3i, t_grid_eig, y0_eig_u3)
        E_pade = pade_u2_evolve(k, u_b, eps, t_grid_eig, y0_eig_u2)

        t_ref = np.array([0.0, t_end_eig])
        E_ref = amp * np.exp(gamma_disp * t_ref)

        h_kin,  = ax.semilogy(t_grid_eig, np.abs(E_kin),  "k-",  lw=2.0)
        h_dir2, = ax.semilogy(t_grid_eig, np.abs(E_dir2), color="C1", lw=1.5, ls="--")
        h_nn2,  = ax.semilogy(t_grid_eig, np.abs(E_nn2),  color="C2", lw=1.5, ls="-.")
        h_dir3, = ax.semilogy(t_grid_eig, np.abs(E_dir3), color="C3", lw=1.5, ls="--")
        h_nn3n, = ax.semilogy(t_grid_eig, np.abs(E_nn3n), color="C4", lw=1.5, ls="-.")
        h_nn3i, = ax.semilogy(t_grid_eig, np.abs(E_nn3i), color="C5", lw=1.5, ls="-.")
        h_pade, = ax.semilogy(t_grid_eig, np.abs(E_pade), color="C0", lw=0.9, ls=":",
                              alpha=0.5)
        h_ref,  = ax.semilogy(t_ref, E_ref, "k--", lw=1.0, alpha=0.4)

        ax.set_title(title + fr",  $\varepsilon=0.05$" + "\n"
                     + fr"$\xi_b = {xi_b:.3f}$,  $\gamma={gamma_disp:.3f}$",
                     fontsize=9)
        ax.set_xlim(0, t_end_eig);  ax.set_ylim(1e-30, 1e-1)
        ax.set_xlabel(r"$t\;[\omega_p^{-1}]$")
        ax.set_ylabel(r"$|E(t)|$")
        if col == 0:
            ax.text(-0.18, 0.5, row_labels[0], transform=ax.transAxes,
                    fontsize=10, rotation=90, va='center', fontweight='bold')

        # ---- row 1: generic δE IC ----
        ax = axes[1, col]
        E_kin  = kinetic_evolve(k, u_b, eps, t_grid_gen, y0_gen_kin)
        E_dir2 = direct_u2_evolve(k, u_b, eps, t_grid_gen, y0_gen_u2)
        E_nn2  = nn_u2_evolve(k, u_b, eps, model_u2,  t_grid_gen, y0_gen_u2)
        E_dir3 = direct_alpha_evolve(k, u_b, eps, t_grid_gen, y0_gen_u3)
        E_nn3n = naive_evolve(k, u_b, eps, model_u3n, t_grid_gen, y0_gen_u3)
        E_nn3i = inference_evolve(k, u_b, eps, model_u3i, t_grid_gen, y0_gen_u3)
        E_pade = pade_u2_evolve(k, u_b, eps, t_grid_gen, y0_gen_u2)

        # Add reference slope on generic-IC row too for easy comparison
        t_ref_g = np.array([5.0, t_end_gen])
        # Anchor to kinetic |E| at t=5
        idx5 = np.argmin(np.abs(t_grid_gen - 5.0))
        E_ref_g = np.abs(E_kin[idx5]) * np.exp(gamma_disp * (t_ref_g - 5.0))

        ax.semilogy(t_grid_gen, np.abs(E_kin),  "k-",  lw=2.0, zorder=5)
        ax.semilogy(t_grid_gen, np.abs(E_dir2), color="C1", lw=1.5, ls="--", zorder=4)
        ax.semilogy(t_grid_gen, np.abs(E_nn2),  color="C2", lw=1.5, ls="-.", zorder=4)
        ax.semilogy(t_grid_gen, np.abs(E_dir3), color="C3", lw=1.5, ls="--", zorder=3)
        ax.semilogy(t_grid_gen, np.abs(E_nn3n), color="C4", lw=1.5, ls="-.", zorder=3)
        ax.semilogy(t_grid_gen, np.abs(E_nn3i), color="C5", lw=1.5, ls="-.", zorder=3)
        ax.semilogy(t_grid_gen, np.abs(E_pade), color="C0", lw=0.9, ls=":",
                    alpha=0.5, zorder=2)
        ax.semilogy(t_ref_g, E_ref_g, "k--", lw=1.0, alpha=0.4, zorder=1)

        ax.set_xlim(0, t_end_gen);  ax.set_ylim(1e-10, 1e-1)
        ax.set_xlabel(r"$t\;[\omega_p^{-1}]$")
        ax.set_ylabel(r"$|E(t)|$")
        if col == 0:
            ax.text(-0.18, 0.5, row_labels[1], transform=ax.transAxes,
                    fontsize=10, rotation=90, va='center', fontweight='bold')

    # Shared legend
    labels = [
        "Kinetic",
        r"Direct $\beta$, N=2",
        r"NN$(r_1)$, N=2",
        r"Direct $\alpha$, N=3",
        r"NN naive$(r_1,r_2)$, N=3",
        r"NN inf$(r_1,r_2)$, N=3",
        "Padé N=2",
        r"$e^{\gamma_{\rm kin} t}$",
    ]
    handles = [h_kin, h_dir2, h_nn2, h_dir3, h_nn3n, h_nn3i, h_pade, h_ref]
    fig.legend(handles, labels, loc="lower center", ncol=4,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.04),
               title="Row 1 legend (same colors apply to row 2)", title_fontsize=8)

    fig.suptitle(
        r"N=2 vs N=3 closures: strongly-damped minority species "
        r"($|\gamma| \gg 0.17$)" + "\n"
        r"Row 1: eigenmode IC — "
        r"Row 2: generic $\delta E$ IC",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    out = FIG_DIR / "fig_u2_landau_strong.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 9: Two-mode superposition — off-eigenmode Jensen test
# ---------------------------------------------------------------------------

def _find_landau_twomode_modes(k, u_b, eps):
    """Return all distinct Landau-damped roots within the training rectangle.

    Training-rectangle constraints (empirical from NN training range):
      xi_b.imag ∈ (-1.0, 0)    — damped but not extreme
      xi_b.real ∈ (0, 3.5)     — positive thermal speed
      omega.imag < -0.03       — clearly damped, not noise
    """
    def res(x):
        D = kinetic_dispersion(x[0] + 1j*x[1], k, u_b, eps)
        return [D.real, D.imag]

    found = []
    for seed_re in np.linspace(0.3, 2.5, 15):
        for seed_im in np.linspace(-0.05, -0.80, 15):
            try:
                sol = _scipy_root(res, [seed_re, seed_im], method='hybr')
                if not sol.success:
                    continue
                w = sol.x[0] + 1j*sol.x[1]
                if abs(kinetic_dispersion(w, k, u_b, eps)) > 1e-6:
                    continue
                if w.imag > -0.03:
                    continue
                xi_b = (w - k*u_b)/k
                if xi_b.imag < -1.0 or xi_b.imag >= 0:
                    continue
                if xi_b.real < 0 or xi_b.real > 3.5:
                    continue
                # Deduplicate: keep if not already in list (|Δω| > 0.01)
                if all(abs(w - wf) > 0.01 for wf in found):
                    found.append(w)
            except Exception:
                pass
    return sorted(found, key=lambda w: w.imag)  # most-damped first


def fig_u2_landau_twomode(nn_n4_super_path: Path = RUN_DIR / "n4_nn_super.eqx") -> None:
    """Two-mode superposition test: off-eigenmode IC with exact analytic reference.

    For each (u_b, k, eps), there exist TWO Landau-damped modes (ω₁, ω₂) at
    the same wave-number.  The initial condition is

        y0 = eigenmode_ic(ω₁, amp) + eigenmode_ic(ω₂, amp)

    which is off the eigenmode manifold of any *single-mode* closure.
    The exact analytic reference for the linear kinetic system (superposition
    principle) is

        E_ref(t) = amp·exp(−iω₁t) + amp·exp(−iω₂t)

    No kinetic simulation is needed — the reference is purely algebraic.

    We test whether each closure tracks the analytic reference, thereby
    isolating the Jensen error that arises when r₁ ≡ U₁/U₀ lies off the
    single-eigenmode manifold.

    Layout: 2×3 grid; 4 panels (one case each) + 1 legend panel.
    Closures: Direct-β N=2, NN^super N=4, HP N=3.
    """
    from bot.closed_loop import (nn_n4_evolve, fluid_eigenmode_ic_n4,
                                  pade_evolve)
    from bot.closures.n4_nn import load_super as load_n4_super

    model_n4_super = load_n4_super()
    _find_all_modes = _find_landau_twomode_modes

    # Four two-mode cases (pre-verified to have ≥2 damped modes in training rect)
    cases = [
        (1.5, 0.50, 0.05, r"$u_b{=}1.5,\;k{=}0.50$"),
        (1.0, 0.50, 0.05, r"$u_b{=}1.0,\;k{=}0.50$"),
        (1.5, 0.40, 0.05, r"$u_b{=}1.5,\;k{=}0.40$"),
        (2.0, 0.30, 0.05, r"$u_b{=}2.0,\;k{=}0.30$"),
    ]

    amp    = 1e-3
    t_end  = 60.0
    t_grid = np.linspace(0, t_end, 1201)
    n_t    = len(t_grid)

    def _pad(E):
        """Pad partial solver result with NaN to match t_grid length."""
        if len(E) >= n_t:
            return E[:n_t]
        out = np.full(n_t, np.nan, dtype=complex)
        out[:len(E)] = E
        return out

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.0))
    ax_flat = axes.flatten()

    h_ref = h_m1 = h_m2 = h_dir2 = h_nn4s = h_pade = None
    a_hp = pade_coefficients(3)

    for idx, (u_b, k, eps, title) in enumerate(cases):
        ax = ax_flat[idx]

        # Find the two modes
        modes = _find_all_modes(k, u_b, eps)
        if len(modes) < 2:
            ax.text(0.5, 0.5, f"Only {len(modes)} mode(s) found",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue

        omega1, omega2 = modes[0], modes[1]      # most-damped, second
        gamma1, gamma2 = omega1.imag, omega2.imag
        xi_b1 = (omega1 - k*u_b)/k
        xi_b2 = (omega2 - k*u_b)/k

        # ---- Analytic reference (kinetic superposition principle) ----
        E_ref = amp * np.exp(-1j*omega1*t_grid) + amp * np.exp(-1j*omega2*t_grid)

        # ---- Individual mode envelopes (for orientation) ----
        E_m1 = amp * np.exp(gamma1 * t_grid)
        E_m2 = amp * np.exp(gamma2 * t_grid)

        # ---- Two-mode IC: off-eigenmode for any single-mode closure ----
        y0_u2 = (fluid_eigenmode_ic_u2(k, u_b, omega1, amp=amp) +
                 fluid_eigenmode_ic_u2(k, u_b, omega2, amp=amp))
        y0_u3 = (fluid_eigenmode_ic_u3(k, u_b, omega1, amp=amp) +
                 fluid_eigenmode_ic_u3(k, u_b, omega2, amp=amp))
        y0_n4 = (fluid_eigenmode_ic_n4(k, u_b, omega1, amp=amp) +
                 fluid_eigenmode_ic_n4(k, u_b, omega2, amp=amp))

        # ---- Integrate fluid closures (pad with NaN if solver stops early) ----
        E_dir2 = _pad(direct_u2_evolve(k, u_b, eps, t_grid, y0_u2))
        E_nn4s  = _pad(nn_n4_evolve(k, u_b, eps, model_n4_super, t_grid, y0_n4))
        E_pade  = _pad(pade_evolve(k, u_b, eps, a_hp, t_grid, y0_u3, method="rk45"))

        # ---- Plot ----
        h_m1,  = ax.semilogy(t_grid, E_m1, color="0.70", lw=1.0, ls="--",  zorder=1,
                              label=fr"$e^{{\gamma_1 t}}$, $\gamma_1={gamma1:.3f}$")
        h_m2,  = ax.semilogy(t_grid, E_m2, color="0.50", lw=1.0, ls=":",   zorder=1,
                              label=fr"$e^{{\gamma_2 t}}$, $\gamma_2={gamma2:.3f}$")
        h_ref, = ax.semilogy(t_grid, np.abs(E_ref), "k-", lw=2.5, zorder=6,
                              label=r"$|e^{-i\omega_1 t}+e^{-i\omega_2 t}|$ (analytic)")
        h_dir2, = ax.semilogy(t_grid, np.abs(E_dir2), color="C1", lw=1.5, ls="--",  zorder=4,
                               label=r"Direct $\beta$, N=2")
        h_nn4s, = ax.semilogy(t_grid, np.abs(E_nn4s), color="C6", lw=2.0, ls="-",  zorder=5,
                               label=r"NN$^{\rm super}(r_1,r_2,r_3)$, N=4  [super train]")
        h_pade, = ax.semilogy(t_grid, np.abs(E_pade), color="C0", lw=1.0, ls=":",  zorder=2,
                               alpha=0.8, label="HP N=3")

        ax.set_title(
            title + fr",  $\varepsilon={eps}$" + "\n"
            + fr"$\omega_1={omega1:.3f}$  ($\xi_{{b1}}={xi_b1:.3f}$)" + "\n"
            + fr"$\omega_2={omega2:.3f}$  ($\xi_{{b2}}={xi_b2:.3f}$)",
            fontsize=8.5
        )
        ax.set_xlim(0, t_end)
        ax.set_ylim(bottom=1e-10, top=3e-3)
        ax.set_xlabel(r"$t\;[\omega_p^{-1}]$")
        ax.set_ylabel(r"$|E(t)|$")

    # Legend panel
    ax_leg = ax_flat[4]
    ax_leg.axis("off")
    handles = [h_ref, h_m1, h_m2, h_dir2, h_nn4s, h_pade]
    labels  = [
        r"$|$amp$\,e^{-i\omega_1 t}+$amp$\,e^{-i\omega_2 t}|$ — analytic (kinetic superposition)",
        r"amp$\,e^{\gamma_1 t}$ — mode 1 envelope",
        r"amp$\,e^{\gamma_2 t}$ — mode 2 envelope",
        r"Direct $\beta$, N=2",
        r"NN$^{\rm super}(r_1,r_2,r_3)$, N=4  [superposition training]",
        r"HP N=3 (time-stepped, RK45)",
    ]
    ax_leg.legend(handles, labels, loc="upper center", fontsize=8.5,
                  title="IC = eigenmode$_1$ + eigenmode$_2$  (off single-mode manifold)\n"
                        "Analytic ref from superposition principle — no kinetic sim\n"
                        "N=4: 6 DOFs = 6 unknowns for 2-mode state; super training reduces Jensen error",
                  title_fontsize=8, frameon=True)

    # Hide unused 6th panel
    ax_flat[5].axis("off")

    fig.suptitle(
        "Two-mode superposition test: Jensen error in off-eigenmode regime\n"
        r"IC $= $ eigenmode$(\omega_1)+$ eigenmode$(\omega_2)$ — "
        r"nonlinear closures see mixed $r_1 \neq \xi_{b1},\xi_{b2}$",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    out = FIG_DIR / "fig_u2_landau_twomode.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def _plot_landau_twomode_case(ax, u_b, k, eps, title, model_n4_super, a_hp,
                               amp=1e-3, t_end=60.0) -> None:
    """Shared single-panel plotting logic for fig_u2_landau_twomode_single.

    y-limits are clipped to the actual finite data range (plus a small
    margin) rather than a fixed value, since different (u_b, k) cases decay
    to very different noise floors within the same t_end window.
    """
    from bot.closed_loop import (nn_n4_evolve, fluid_eigenmode_ic_n4,
                                  pade_evolve)

    t_grid = np.linspace(0, t_end, 1201)
    n_t    = len(t_grid)

    def _pad(E):
        if len(E) >= n_t:
            return E[:n_t]
        out = np.full(n_t, np.nan, dtype=complex)
        out[:len(E)] = E
        return out

    modes = _find_landau_twomode_modes(k, u_b, eps)
    omega1, omega2 = modes[0], modes[1]
    gamma1, gamma2 = omega1.imag, omega2.imag
    xi_b1 = (omega1 - k*u_b)/k
    xi_b2 = (omega2 - k*u_b)/k

    E_ref = amp * np.exp(-1j*omega1*t_grid) + amp * np.exp(-1j*omega2*t_grid)
    E_m1 = amp * np.exp(gamma1 * t_grid)
    E_m2 = amp * np.exp(gamma2 * t_grid)

    y0_u2 = (fluid_eigenmode_ic_u2(k, u_b, omega1, amp=amp) +
             fluid_eigenmode_ic_u2(k, u_b, omega2, amp=amp))
    y0_u3 = (fluid_eigenmode_ic_u3(k, u_b, omega1, amp=amp) +
             fluid_eigenmode_ic_u3(k, u_b, omega2, amp=amp))
    y0_n4 = (fluid_eigenmode_ic_n4(k, u_b, omega1, amp=amp) +
             fluid_eigenmode_ic_n4(k, u_b, omega2, amp=amp))

    E_dir2 = _pad(direct_u2_evolve(k, u_b, eps, t_grid, y0_u2))
    E_nn4s = _pad(nn_n4_evolve(k, u_b, eps, model_n4_super, t_grid, y0_n4))
    E_pade = _pad(pade_evolve(k, u_b, eps, a_hp, t_grid, y0_u3, method="rk45"))

    ax.semilogy(t_grid, E_m1, color="0.70", lw=1.0, ls="--", zorder=1,
                label=fr"$e^{{\gamma_1 t}}$, $\gamma_1={gamma1:.3f}$")
    ax.semilogy(t_grid, E_m2, color="0.50", lw=1.0, ls=":", zorder=1,
                label=fr"$e^{{\gamma_2 t}}$, $\gamma_2={gamma2:.3f}$")
    ax.semilogy(t_grid, np.abs(E_ref), "k-", lw=2.5, zorder=6,
                label=r"$|e^{-i\omega_1 t}+e^{-i\omega_2 t}|$ (analytic)")
    ax.semilogy(t_grid, np.abs(E_dir2), color="C1", lw=1.5, ls="--", zorder=4,
                label=r"Direct $\beta$, N=2")
    ax.semilogy(t_grid, np.abs(E_nn4s), color="C6", lw=2.0, ls="-", zorder=5,
                label=r"NN$^{\rm super}(r_1,r_2,r_3)$, N=4")
    ax.semilogy(t_grid, np.abs(E_pade), color="C0", lw=1.0, ls=":", zorder=2,
                alpha=0.8, label="HP N=3 (RK45)")

    # Clip y-axis to the actual data range (bottom: finite-value floor of
    # the reference/closure traces; top: just above the initial amplitude).
    finite_traces = [np.abs(E_ref), np.abs(E_dir2), np.abs(E_nn4s), np.abs(E_pade)]
    all_vals = np.concatenate(finite_traces)
    all_vals = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
    ylo = 10 ** (np.floor(np.log10(all_vals.min())) - 0.5)
    yhi = 10 ** (np.ceil(np.log10(all_vals.max())) + 0.2)

    ax.set_title(
        title + fr",  $\varepsilon={eps}$" + "\n"
        + fr"$\omega_1={omega1:.3f}$  ($\xi_{{b1}}={xi_b1:.3f}$),  "
        + fr"$\omega_2={omega2:.3f}$  ($\xi_{{b2}}={xi_b2:.3f}$)",
        fontsize=9.5,
    )
    ax.set_xlim(0, t_end)
    ax.set_ylim(bottom=ylo, top=yhi)
    ax.set_xlabel(r"$t\;[\omega_p^{-1}]$")
    ax.set_ylabel(r"$|E(t)|$")
    ax.legend(fontsize=7.5, loc="upper right")


def fig_u2_landau_twomode_single(
        nn_n4_super_path: Path = RUN_DIR / "n4_nn_super.eqx") -> None:
    """Scaled-down version of fig_u2_landau_twomode: top-middle + top-right cases.

    Same computation/style as fig_u2_landau_twomode, restricted to the
    u_b=1.0, k=0.50 and u_b=1.5, k=0.40 cases, each with its own inline
    legend and its y-axis clipped to that case's actual data range (rather
    than the fixed 1e-10 floor used in the full 2x3 grid, which is far
    below where the u_b=1.5, k=0.40 traces actually decay to).
    """
    from bot.closures.n4_nn import load_super as load_n4_super

    model_n4_super = load_n4_super()
    a_hp = pade_coefficients(3)

    cases = [
        (1.0, 0.50, 0.05, r"$u_b{=}1.0,\;k{=}0.50$"),
        (1.5, 0.40, 0.05, r"$u_b{=}1.5,\;k{=}0.40$"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for ax, (u_b, k, eps, title) in zip(axes, cases):
        _plot_landau_twomode_case(ax, u_b, k, eps, title, model_n4_super, a_hp)

    fig.tight_layout()
    out = FIG_DIR / "fig_u2_landau_twomode_single.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def _plot_landau_singlemode_case(ax, u_b, k, eps, title, model_n4_super, a_hp,
                                  amp=1e-3, t_end=60.0) -> None:
    """Single-eigenmode counterpart of _plot_landau_twomode_case.

    IC = a single kinetic eigenmode (the most-damped Landau root at this
    (u_b, k, eps), i.e. the same omega1 used as "mode 1" in the two-mode
    test).  On this IC, Direct-beta/HP/NN^super are all exact-by-
    construction for the *fluid* moments they're built from -- but only
    Direct-beta's fixed point is an eigenmode of its own closed dynamical
    system.  HP N=3's closure relation differs from the exact beta(xi_b)
    used to build the IC, so the IC is off HP's own eigenmode manifold and
    it leaks into its own (generally different) eigenmodes -- same
    mechanism as the two-mode test, but here isolated with no admixture
    from a second mode.
    """
    from bot.closed_loop import (nn_n4_evolve, fluid_eigenmode_ic_n4,
                                  pade_evolve)

    t_grid = np.linspace(0, t_end, 1201)
    n_t    = len(t_grid)

    def _pad(E):
        if len(E) >= n_t:
            return E[:n_t]
        out = np.full(n_t, np.nan, dtype=complex)
        out[:len(E)] = E
        return out

    modes = _find_landau_twomode_modes(k, u_b, eps)
    omega1 = modes[0]      # most-damped Landau root
    gamma1 = omega1.imag
    xi_b1 = (omega1 - k*u_b) / k

    E_ref = amp * np.exp(-1j*omega1*t_grid)

    y0_u2 = fluid_eigenmode_ic_u2(k, u_b, omega1, amp=amp)
    y0_u3 = fluid_eigenmode_ic_u3(k, u_b, omega1, amp=amp)
    y0_n4 = fluid_eigenmode_ic_n4(k, u_b, omega1, amp=amp)

    E_dir2 = _pad(direct_u2_evolve(k, u_b, eps, t_grid, y0_u2))
    E_nn4s = _pad(nn_n4_evolve(k, u_b, eps, model_n4_super, t_grid, y0_n4))
    E_pade = _pad(pade_evolve(k, u_b, eps, a_hp, t_grid, y0_u3, method="rk45"))

    ax.semilogy(t_grid, np.abs(E_ref), "k-", lw=2.5, zorder=6,
                label=r"$|e^{-i\omega_1 t}|$ (exact single mode)")
    ax.semilogy(t_grid, np.abs(E_dir2), color="C1", lw=1.5, ls="--", zorder=4,
                label=r"Direct $\beta$, N=2")
    ax.semilogy(t_grid, np.abs(E_nn4s), color="C6", lw=2.0, ls="-", zorder=5,
                label=r"NN$^{\rm super}(r_1,r_2,r_3)$, N=4")
    ax.semilogy(t_grid, np.abs(E_pade), color="C0", lw=1.0, ls=":", zorder=2,
                alpha=0.8, label="HP N=3 (RK45)")

    finite_traces = [np.abs(E_ref), np.abs(E_dir2), np.abs(E_nn4s), np.abs(E_pade)]
    all_vals = np.concatenate(finite_traces)
    all_vals = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
    ylo = 10 ** (np.floor(np.log10(all_vals.min())) - 0.5)
    yhi = 10 ** (np.ceil(np.log10(all_vals.max())) + 0.2)

    ax.set_title(
        title + fr",  $\varepsilon={eps}$" + "\n"
        + fr"$\omega_1={omega1:.3f}$  ($\xi_{{b1}}={xi_b1:.3f}$),  $\gamma_1={gamma1:.4f}$",
        fontsize=9.5,
    )
    ax.set_xlim(0, t_end)
    ax.set_ylim(bottom=ylo, top=yhi)
    ax.set_xlabel(r"$t\;[\omega_p^{-1}]$")
    ax.set_ylabel(r"$|E(t)|$")
    ax.legend(fontsize=7.5, loc="upper right")


def fig_u2_landau_singlemode(
        nn_n4_super_path: Path = RUN_DIR / "n4_nn_super.eqx") -> None:
    """Single-eigenmode counterpart of fig_u2_landau_twomode_single.

    Same two cases (u_b=1.0, k=0.50 and u_b=1.5, k=0.40), same closures and
    styling, but IC = a single kinetic eigenmode rather than a two-mode
    superposition -- isolates each closure's behavior with no Jensen-error
    admixture from a second mode.
    """
    from bot.closures.n4_nn import load_super as load_n4_super

    model_n4_super = load_n4_super()
    a_hp = pade_coefficients(3)

    cases = [
        (1.0, 0.50, 0.05, r"$u_b{=}1.0,\;k{=}0.50$"),
        (1.5, 0.40, 0.05, r"$u_b{=}1.5,\;k{=}0.40$"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for ax, (u_b, k, eps, title) in zip(axes, cases):
        _plot_landau_singlemode_case(ax, u_b, k, eps, title, model_n4_super, a_hp)

    fig.tight_layout()
    out = FIG_DIR / "fig_u2_landau_singlemode.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: HP vs Direct β sweep heatmap
# ---------------------------------------------------------------------------

def fig_hp_beta_sweep(path: Path = RUN_DIR / "sweep_hp_beta.npz") -> None:
    """Log10|γ_eff/γ_kin − 1| heatmaps: HP Padé N=3 vs Direct β (N=2)."""
    d = np.load(path)
    u_b = d["u_b_vals"]; eps = d["eps_vals"]
    over_hp  = d["overshoot_hp"]
    over_dir = d["overshoot_direct"]
    extent = [eps[0], eps[-1], u_b[0], u_b[-1]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _log_heatmap(axes[0], over_hp,  extent, r"HP Padé $N=3$  (Hammett-Perkins)",
                mark=False)
    im = _log_heatmap(axes[1], over_dir, extent, r"Direct $\beta$ closure  ($N=2$)",
                      mark=False)
    fig.tight_layout()
    _add_colorbar(fig, im)
    out = FIG_DIR / "fig_hp_beta_sweep.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def fig_hp_beta_sweep_abs(path: Path = RUN_DIR / "sweep_hp_beta.npz") -> None:
    """Log10(ABSOLUTE growth-rate error) heatmaps: HP Padé N=3 vs Direct β.

    Same data as fig_hp_beta_sweep, but plots the raw |gamma_eff - gamma_kin|
    instead of normalizing by |gamma_kin|.  gamma_kin varies ~9x across this
    grid (0.035 at u_b=3,eps=0.01 up to 0.32 at u_b=10,eps=0.20), so the
    relative metric can make small-gamma_kin cells look disproportionately
    bad even when the absolute rate error is tiny -- this view isolates the
    raw rate error itself.
    """
    d = np.load(path)
    if "gamma_kin" not in d:
        raise KeyError("No gamma_kin key found. Re-run bot.sweeps.sweep_hp_beta.")
    u_b = d["u_b_vals"]; eps = d["eps_vals"]
    abs_hp  = np.abs(d["gamma_hp"]     - d["gamma_kin"])
    abs_dir = np.abs(d["gamma_direct"] - d["gamma_kin"])
    extent = [eps[0], eps[-1], u_b[0], u_b[-1]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _log_heatmap(axes[0], abs_hp,  extent, r"HP Padé $N=3$  (Hammett-Perkins)",
                mark=False)
    im = _log_heatmap(axes[1], abs_dir, extent, r"Direct $\beta$ closure  ($N=2$)",
                      mark=False)
    fig.tight_layout()

    cb = fig.colorbar(im, ax=fig.axes, shrink=0.85,
                      label=r"$\log_{10}\,|\gamma_{\rm eff} - \gamma_{\rm kin}|$")
    cb.set_ticks([-4, -3, -2, -1, 0, 1])
    cb.set_ticklabels([r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$",
                       r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$"])

    out = FIG_DIR / "fig_hp_beta_sweep_abs.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: Two-mode mixing sweep heatmap
# ---------------------------------------------------------------------------

def fig_twomode_sweep(path: Path = RUN_DIR / "sweep_twomode.npz") -> None:
    """Log10(rel RMSE of E(t)) heatmaps for two-mode mixing IC sweep.

    x-axis: mixing weight w ∈ [0, 1]  (w=0: pure mode 1; w=1: pure mode 2)
    y-axis: ε = n_b/n_0
    3 panels: Direct β (N=2), N=4 NN super, HP Padé N=3.
    Error metric: ||E_fluid − E_ref||_2 / ||E_ref||_2
    """
    d = np.load(path)
    w_vals   = d["w_vals"]
    eps_vals = d["eps_vals"]

    extent = [w_vals[0], w_vals[-1], eps_vals[0], eps_vals[-1]]

    def _panel(ax, err, title):
        z = np.log10(np.maximum(err, 1e-8))
        im = ax.imshow(z, origin="lower", aspect="auto", extent=extent,
                       cmap=LOG_OVER_CMAP, vmin=LOG_OVER_VMIN, vmax=LOG_OVER_VMAX,
                       interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"Mixing weight $w$")
        ax.set_ylabel(r"$\varepsilon = n_b/n_0$")
        return im

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    _panel(axes[0], d["err_dir"], r"Direct $\beta$  ($N=2$)")
    _panel(axes[1], d["err_nn"],  r"N=4 NN$^{\rm super}$  ($r_1,r_2,r_3$)")
    im = _panel(axes[2], d["err_hp"],  r"HP Padé $N=3$")

    fig.suptitle(
        r"Two-mode IC: $\|E_{\rm fluid} - E_{\rm ref}\|/\|E_{\rm ref}\|$  "
        r"($u_b=1.0,\;k=0.5$;  IC$\,=\,(1{-}w)\cdot$mode$_1 + w\cdot$mode$_2$)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()

    cb = fig.colorbar(im, ax=fig.axes, shrink=0.85,
                      label=r"$\log_{10}\,\|E_{\rm fluid} - E_{\rm ref}\|/\|E_{\rm ref}\|$")
    cb.set_ticks([-4, -3, -2, -1, 0, 1])
    cb.set_ticklabels([r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$",
                       r"$10^{-1}$", r"$10^{0}$", r"$\geq 10^{1}$"])

    out = FIG_DIR / "fig_twomode_sweep.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def fig_twomode_sweep_gamma(path: Path = RUN_DIR / "sweep_twomode.npz") -> None:
    """Log10(rel damping-rate error) heatmaps for the Landau two-mode sweep.

    Uses the late-time log|E| slope as the effective damping rate, ignoring
    phase entirely. For the Landau case (u_b=1, k=0.5, both modes damped),
    the late-time rate equals the least-damped eigenvalue's Im(ω) for w<1.
    """
    d = np.load(path)
    w_vals   = d["w_vals"]
    eps_vals = d["eps_vals"]

    if "err_dir_gamma" not in d:
        raise KeyError("No gamma-only keys found. Re-run bot.sweeps.sweep_twomode.")

    extent = [w_vals[0], w_vals[-1], eps_vals[0], eps_vals[-1]]

    def _panel(ax, err, title):
        z = np.log10(np.maximum(err, 1e-8))
        im = ax.imshow(z, origin="lower", aspect="auto", extent=extent,
                       cmap=LOG_OVER_CMAP, vmin=LOG_OVER_VMIN, vmax=LOG_OVER_VMAX,
                       interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"Mixing weight $w$")
        ax.set_ylabel(r"$\varepsilon = n_b/n_0$")
        return im

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    _panel(axes[0], d["err_dir_gamma"], r"Direct $\beta$  ($N=2$)")
    _panel(axes[1], d["err_nn_gamma"],  r"N=4 NN$^{\rm super}$  ($r_1,r_2,r_3$)")
    im = _panel(axes[2], d["err_hp_gamma"],  r"HP Padé $N=3$")

    axes[0].text(0.02, 0.97, "mode 1", transform=axes[0].transAxes,
                 fontsize=8, va="top", color="white", alpha=0.8)
    axes[0].text(0.78, 0.97, "mode 2", transform=axes[0].transAxes,
                 fontsize=8, va="top", color="white", alpha=0.8)

    fig.suptitle(
        r"Two-mode IC (Landau regime): $|\gamma_{\rm fluid} - \gamma_{\rm ref}|/|\gamma_{\rm ref}|$  "
        r"($u_b=1.0,\;k=0.5$;  both modes damped)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()

    cb = fig.colorbar(im, ax=fig.axes, shrink=0.85,
                      label=r"$\log_{10}\,|\gamma_{\rm fluid} - \gamma_{\rm ref}|/|\gamma_{\rm ref}|$")
    cb.set_ticks([-4, -3, -2, -1, 0, 1])
    cb.set_ticklabels([r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$",
                       r"$10^{-1}$", r"$10^{0}$", r"$\geq 10^{1}$"])

    out = FIG_DIR / "fig_twomode_sweep_gamma.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: Two-mode mixing sweep — GROWING regime
# ---------------------------------------------------------------------------

def fig_twomode_sweep_growing(path: Path = RUN_DIR / "sweep_twomode_growing.npz") -> None:
    """Log10(rel RMSE of E(t)) heatmaps for two-mode mixing sweep in growing regime.

    u_b=5.0, k=0.30: mode 1 is the growing beam mode, mode 2 is the damped
    upper branch.  w=0 is pure growing; w=1 is pure damped.
    """
    d = np.load(path)
    w_vals   = d["w_vals"]
    eps_vals = d["eps_vals"]

    extent = [w_vals[0], w_vals[-1], eps_vals[0], eps_vals[-1]]

    def _panel(ax, err, title):
        z = np.log10(np.maximum(err, 1e-8))
        im = ax.imshow(z, origin="lower", aspect="auto", extent=extent,
                       cmap=LOG_OVER_CMAP, vmin=LOG_OVER_VMIN, vmax=LOG_OVER_VMAX,
                       interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"Mixing weight $w$")
        ax.set_ylabel(r"$\varepsilon = n_b/n_0$")
        return im

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    _panel(axes[0], d["err_dir"], r"Direct $\beta$  ($N=2$)")
    _panel(axes[1], d["err_nn"],  r"N=4 NN$^{\rm super}$  ($r_1,r_2,r_3$)")
    im = _panel(axes[2], d["err_hp"],  r"HP Padé $N=3$")

    axes[0].text(0.02, 0.97, "growing", transform=axes[0].transAxes,
                 fontsize=8, va="top", color="white", alpha=0.8)
    axes[0].text(0.78, 0.97, "damped", transform=axes[0].transAxes,
                 fontsize=8, va="top", color="white", alpha=0.8)

    fig.suptitle(
        r"Two-mode IC (growing regime): $\|E_{\rm fluid} - E_{\rm ref}\|/\|E_{\rm ref}\|$  "
        r"($u_b=5.0,\;k=0.3$;  $w{=}0$: growing beam,  $w{=}1$: damped upper branch)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()

    cb = fig.colorbar(im, ax=fig.axes, shrink=0.85,
                      label=r"$\log_{10}\,\|E_{\rm fluid} - E_{\rm ref}\|/\|E_{\rm ref}\|$")
    cb.set_ticks([-4, -3, -2, -1, 0, 1])
    cb.set_ticklabels([r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$",
                       r"$10^{-1}$", r"$10^{0}$", r"$\geq 10^{1}$"])

    out = FIG_DIR / "fig_twomode_sweep_growing.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def fig_twomode_sweep_growing_gamma(path: Path = RUN_DIR / "sweep_twomode_growing.npz") -> None:
    """Log10(rel growth-rate error) heatmaps for the growing-regime two-mode sweep.

    Unlike fig_twomode_sweep_growing (full complex-waveform RMSE, which mixes
    growth-rate and real-frequency/phase error), this isolates growth-rate
    accuracy only: |gamma_fluid - gamma_ref| / |gamma_ref|, fit from the
    late-time log-amplitude slope of each signal. A closure can have a large
    RMSE here purely from phase drift while still tracking the correct
    instability rate -- this figure shows whether that is in fact the case.
    """
    d = np.load(path)
    w_vals   = d["w_vals"]
    eps_vals = d["eps_vals"]

    extent = [w_vals[0], w_vals[-1], eps_vals[0], eps_vals[-1]]

    def _panel(ax, err, title):
        z = np.log10(np.maximum(err, 1e-8))
        im = ax.imshow(z, origin="lower", aspect="auto", extent=extent,
                       cmap=LOG_OVER_CMAP, vmin=LOG_OVER_VMIN, vmax=LOG_OVER_VMAX,
                       interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"Mixing weight $w$")
        ax.set_ylabel(r"$\varepsilon = n_b/n_0$")
        return im

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    _panel(axes[0], d["err_dir_gamma"], r"Direct $\beta$  ($N=2$)")
    _panel(axes[1], d["err_nn_gamma"],  r"N=4 NN$^{\rm super}$  ($r_1,r_2,r_3$)")
    im = _panel(axes[2], d["err_hp_gamma"],  r"HP Padé $N=3$")

    axes[0].text(0.02, 0.97, "growing", transform=axes[0].transAxes,
                 fontsize=8, va="top", color="white", alpha=0.8)
    axes[0].text(0.78, 0.97, "damped", transform=axes[0].transAxes,
                 fontsize=8, va="top", color="white", alpha=0.8)

    fig.suptitle(
        r"Two-mode IC (growing regime): $|\gamma_{\rm fluid} - \gamma_{\rm ref}|/|\gamma_{\rm ref}|$  "
        r"($u_b=5.0,\;k=0.3$;  $w{=}0$: growing beam,  $w{=}1$: damped upper branch)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()

    cb = fig.colorbar(im, ax=fig.axes, shrink=0.85,
                      label=r"$\log_{10}\,|\gamma_{\rm fluid} - \gamma_{\rm ref}|/|\gamma_{\rm ref}|$")
    cb.set_ticks([-4, -3, -2, -1, 0, 1])
    cb.set_ticklabels([r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$",
                       r"$10^{-1}$", r"$10^{0}$", r"$\geq 10^{1}$"])

    out = FIG_DIR / "fig_twomode_sweep_growing_gamma.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def fig_twomode_sweep_growing_gamma_abs(
        path: Path = RUN_DIR / "sweep_twomode_growing.npz") -> None:
    """Log10(ABSOLUTE growth-rate error) heatmaps for the growing-regime sweep.

    Same data as fig_twomode_sweep_growing_gamma, but plots the raw
    |gamma_fluid - gamma_ref| instead of normalizing by |gamma_ref|.  Near
    the w->1 edge (pure damped upper-branch mode), gamma_ref itself shrinks
    toward zero as epsilon increases (e.g. -0.039 at eps=0.2), so the
    relative metric blows up there even when the absolute rate error is
    modest -- this view separates "the true rate is hard to resolve
    because it's tiny" from "the closure's predicted rate is actually far
    off in absolute terms."
    """
    d = np.load(path)
    if "err_dir_gamma_abs" not in d:
        raise KeyError(
            "No *_gamma_abs keys found. Re-run bot.sweeps.sweep_twomode_growing."
        )
    w_vals   = d["w_vals"]
    eps_vals = d["eps_vals"]

    extent = [w_vals[0], w_vals[-1], eps_vals[0], eps_vals[-1]]

    def _panel(ax, err, title):
        z = np.log10(np.maximum(err, 1e-8))
        im = ax.imshow(z, origin="lower", aspect="auto", extent=extent,
                       cmap=LOG_OVER_CMAP, vmin=LOG_OVER_VMIN, vmax=LOG_OVER_VMAX,
                       interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"Mixing weight $w$")
        ax.set_ylabel(r"$\varepsilon = n_b/n_0$")
        return im

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    _panel(axes[0], d["err_dir_gamma_abs"], r"Direct $\beta$  ($N=2$)")
    _panel(axes[1], d["err_nn_gamma_abs"],  r"N=4 NN$^{\rm super}$  ($r_1,r_2,r_3$)")
    im = _panel(axes[2], d["err_hp_gamma_abs"],  r"HP Padé $N=3$")

    axes[0].text(0.02, 0.97, "growing", transform=axes[0].transAxes,
                 fontsize=8, va="top", color="white", alpha=0.8)
    axes[0].text(0.78, 0.97, "damped", transform=axes[0].transAxes,
                 fontsize=8, va="top", color="white", alpha=0.8)

    fig.suptitle(
        r"Two-mode IC (growing regime): $|\gamma_{\rm fluid} - \gamma_{\rm ref}|$  "
        r"(NOT normalized by $|\gamma_{\rm ref}|$)  "
        r"($u_b=5.0,\;k=0.3$;  $w{=}0$: growing beam,  $w{=}1$: damped upper branch)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()

    cb = fig.colorbar(im, ax=fig.axes, shrink=0.85,
                      label=r"$\log_{10}\,|\gamma_{\rm fluid} - \gamma_{\rm ref}|$")
    cb.set_ticks([-4, -3, -2, -1, 0, 1])
    cb.set_ticklabels([r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$",
                       r"$10^{-1}$", r"$10^{0}$", r"$\geq 10^{1}$"])

    out = FIG_DIR / "fig_twomode_sweep_growing_gamma_abs.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: Two-mode sweeps — NN and HP Padé only (no Direct β)
# ---------------------------------------------------------------------------

def fig_twomode_sweep_nn_hp(path: Path = RUN_DIR / "sweep_twomode.npz") -> None:
    """Two-panel version of fig_twomode_sweep: NN and HP Padé only."""
    d = np.load(path)
    w_vals   = d["w_vals"]
    eps_vals = d["eps_vals"]
    extent = [w_vals[0], w_vals[-1], eps_vals[0], eps_vals[-1]]

    def _panel(ax, err, title):
        z = np.log10(np.maximum(err, 1e-8))
        im = ax.imshow(z, origin="lower", aspect="auto", extent=extent,
                       cmap=LOG_OVER_CMAP, vmin=LOG_OVER_VMIN, vmax=LOG_OVER_VMAX,
                       interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"Mixing weight $w$")
        ax.set_ylabel(r"$\varepsilon = n_b/n_0$")
        return im

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    _panel(axes[0], d["err_nn"],  r"NN$(r_1,r_2,r_3)$")
    im = _panel(axes[1], d["err_hp"],  r"HP Padé $N=3$")

    fig.tight_layout()

    cb = fig.colorbar(im, ax=fig.axes, shrink=0.85,
                      label=r"$\log_{10}\,\|E_{\rm fluid} - E_{\rm ref}\|/\|E_{\rm ref}\|$")
    cb.set_ticks([-4, -3, -2, -1, 0, 1])
    cb.set_ticklabels([r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$",
                       r"$10^{-1}$", r"$10^{0}$", r"$\geq 10^{1}$"])

    out = FIG_DIR / "fig_twomode_sweep_nn_hp.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def fig_twomode_sweep_growing_nn_hp(path: Path = RUN_DIR / "sweep_twomode_growing.npz") -> None:
    """Two-panel version of fig_twomode_sweep_growing: NN and HP Padé only."""
    d = np.load(path)
    w_vals   = d["w_vals"]
    eps_vals = d["eps_vals"]
    extent = [w_vals[0], w_vals[-1], eps_vals[0], eps_vals[-1]]

    def _panel(ax, err, title):
        z = np.log10(np.maximum(err, 1e-8))
        im = ax.imshow(z, origin="lower", aspect="auto", extent=extent,
                       cmap=LOG_OVER_CMAP, vmin=LOG_OVER_VMIN, vmax=LOG_OVER_VMAX,
                       interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"Mixing weight $w$")
        ax.set_ylabel(r"$\varepsilon = n_b/n_0$")
        return im

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    _panel(axes[0], d["err_nn"],  r"NN$(r_1,r_2,r_3)$")
    im = _panel(axes[1], d["err_hp"],  r"HP Padé $N=3$")

    axes[0].text(0.02, 0.97, "growing", transform=axes[0].transAxes,
                 fontsize=8, va="top", color="white", alpha=0.8)
    axes[0].text(0.78, 0.97, "damped", transform=axes[0].transAxes,
                 fontsize=8, va="top", color="white", alpha=0.8)

    fig.tight_layout()

    cb = fig.colorbar(im, ax=fig.axes, shrink=0.85,
                      label=r"$\log_{10}\,\|E_{\rm fluid} - E_{\rm ref}\|/\|E_{\rm ref}\|$")
    cb.set_ticks([-4, -3, -2, -1, 0, 1])
    cb.set_ticklabels([r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$",
                       r"$10^{-1}$", r"$10^{0}$", r"$\geq 10^{1}$"])

    out = FIG_DIR / "fig_twomode_sweep_growing_nn_hp.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: Time-domain traces at w=1 (pure damped upper-branch mode)
# ---------------------------------------------------------------------------

def fig_twomode_growing_w1_timedomain(
        nn_n4_super_path: Path = RUN_DIR / "n4_nn_super.eqx") -> None:
    """Time-domain E(t) traces at the w=1 edge of the growing-regime sweep.

    fig_twomode_sweep_growing_gamma.png shows a large growth-rate error for
    every closure exactly at w=1 (pure "damped upper branch" kinetic mode,
    no admixture of the growing beam mode).  This figure shows why: it
    plots |E(t)| for the kinetic reference against Direct-beta, HP N=3
    (Pade), and the N=4 NN super closure, starting from the exact single-
    mode eigenmode IC (fluid_eigenmode_ic_u2/u3/n4 at omega2 only).

    Direct-beta is exact-by-construction on this IC and tracks the
    reference almost perfectly.  HP N=3's fixed linear system has its own
    eigenmodes, none of which coincides with the true kinetic eigenmode at
    xi_b2 -- so the IC has a small but nonzero projection onto HP's own
    spurious fastest-growing eigenvector, which eventually dominates and
    diverges exponentially away from the (weakly damped) true reference.
    The N=4 NN, whose static alpha4 prediction at xi_b2 is accurate to
    ~0.1-0.5%, shows the same mechanism in a far milder, delayed form: the
    small residual closure bias compounds over the ~40-time-unit window.
    """
    from bot.closed_loop import (nn_n4_evolve, fluid_eigenmode_ic_n4,
                                  pade_evolve, direct_u2_evolve,
                                  fluid_eigenmode_ic_u2, fluid_eigenmode_ic_u3)
    from bot.closures.n4_nn import load_super as load_n4_super
    from bot.sweeps.sweep_twomode_growing import _find_two_modes, K, U_B

    model_n4_super = load_n4_super()
    a_hp = pade_coefficients(3)

    amp    = 1e-3
    t_end  = 40.0
    t_grid = np.linspace(0, t_end, 801)
    n_t    = len(t_grid)

    def _pad(E):
        if len(E) >= n_t:
            return E[:n_t]
        out = np.full(n_t, np.nan, dtype=complex)
        out[:len(E)] = E
        return out

    eps_vals = [0.02, 0.05, 0.10, 0.15, 0.20]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.0))
    ax_flat = axes.flatten()

    h_ref = h_dir = h_hp = h_nn = None

    for idx, eps in enumerate(eps_vals):
        ax = ax_flat[idx]

        modes = _find_two_modes(K, U_B, eps)
        if len(modes) < 2:
            ax.text(0.5, 0.5, f"< 2 modes found", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(fr"$\varepsilon={eps}$")
            continue
        omega1, omega2 = modes         # growing, damped upper branch
        gamma2 = omega2.imag
        xi_b2 = (omega2 - K*U_B) / K

        E_ref = amp * np.exp(-1j * omega2 * t_grid)

        y0_u2 = fluid_eigenmode_ic_u2(K, U_B, omega2, amp=amp)
        y0_u3 = fluid_eigenmode_ic_u3(K, U_B, omega2, amp=amp)
        y0_n4 = fluid_eigenmode_ic_n4(K, U_B, omega2, amp=amp)

        E_dir = _pad(direct_u2_evolve(K, U_B, eps, t_grid, y0_u2))
        E_hp  = _pad(pade_evolve(K, U_B, eps, a_hp, t_grid, y0_u3, method="rk45"))
        E_nn  = _pad(nn_n4_evolve(K, U_B, eps, model_n4_super, t_grid, y0_n4))

        h_ref, = ax.semilogy(t_grid, np.abs(E_ref), "k-", lw=2.5, zorder=6,
                              label=r"kinetic ref $|{\rm amp}\,e^{-i\omega_2 t}|$")
        h_dir, = ax.semilogy(t_grid, np.abs(E_dir), color="C1", lw=1.5, ls="--",
                              zorder=4, label=r"Direct $\beta$, N=2")
        h_hp,  = ax.semilogy(t_grid, np.abs(E_hp),  color="C0", lw=1.5, ls=":",
                              zorder=3, label="HP N=3 (RK45)")
        h_nn,  = ax.semilogy(t_grid, np.abs(E_nn),  color="C6", lw=2.0, ls="-",
                              zorder=5, label=r"N=4 NN$^{\rm super}$")

        ax.set_title(
            fr"$\varepsilon={eps}$" + "\n"
            + fr"$\omega_2={omega2:.3f}$  ($\xi_{{b2}}={xi_b2:.3f}$),  $\gamma_2={gamma2:.4f}$",
            fontsize=9,
        )
        ax.set_xlim(0, t_end)
        ax.set_ylim(bottom=1e-6, top=1e2)
        ax.set_xlabel(r"$t\;[\omega_p^{-1}]$")
        ax.set_ylabel(r"$|E(t)|$")

    ax_leg = ax_flat[5]
    ax_leg.axis("off")
    ax_leg.legend([h_ref, h_dir, h_hp, h_nn],
                  [r"kinetic ref $|{\rm amp}\,e^{-i\omega_2 t}|$ (pure damped upper-branch mode)",
                   r"Direct $\beta$, N=2",
                   r"HP N=3 (Padé, time-stepped RK45)",
                   r"N=4 NN$^{\rm super}(r_1,r_2,r_3)$ (retrained, wide $w$ coverage)"],
                  loc="upper center", fontsize=9,
                  title="IC = single kinetic eigenmode at $\\omega_2$ (w=1 edge of\n"
                        "fig_twomode_sweep_growing_gamma.png) — no growing-mode admixture",
                  title_fontsize=8.5, frameon=True)

    fig.suptitle(
        r"Growing regime, $w=1$ edge: pure damped upper-branch mode "
        r"($u_b=5.0,\;k=0.3$)" + "\n"
        r"Direct-$\beta$ is exact by construction; HP/NN blow up once a tiny "
        r"projection onto their own fastest-growing mode overtakes the true (weak) decay",
        fontsize=10.5, y=1.02,
    )
    fig.tight_layout()
    out = FIG_DIR / "fig_twomode_growing_w1_timedomain.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: Single-mode eigenmode IC sweep over u_b × eps
# ---------------------------------------------------------------------------

def fig_singlemode_sweep(path: Path = RUN_DIR / "sweep_singlemode.npz") -> None:
    """Log10(rel RMSE of E(t)) heatmaps for the single-mode eigenmode IC sweep.

    x-axis: u_b (beam velocity)
    y-axis: ε = n_b/n_0
    3 panels: Direct β (N=2), N=4 NN super, HP Padé N=3.
    IC is the fluid eigenmode for the most-unstable kinetic mode at k*(u_b, eps).
    Reference: analytic  E_ref(t) = amp * exp(-i*omega*t).
    """
    d = np.load(path)
    u_b_vals = d["u_b_vals"]
    eps_vals = d["eps_vals"]

    extent = [u_b_vals[0], u_b_vals[-1], eps_vals[0], eps_vals[-1]]

    def _panel(ax, err, title):
        z = np.log10(np.maximum(err, 1e-8))
        im = ax.imshow(z.T, origin="lower", aspect="auto", extent=extent,
                       cmap=LOG_OVER_CMAP, vmin=LOG_OVER_VMIN, vmax=LOG_OVER_VMAX,
                       interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"$u_b$  (beam velocity)")
        ax.set_ylabel(r"$\varepsilon = n_b/n_0$")
        return im

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    _panel(axes[0], d["err_dir"], r"Direct $\beta$  ($N=2$)")
    _panel(axes[1], d["err_nn"],  r"N=4 NN$^{\rm super}$  ($r_1,r_2,r_3$)")
    im = _panel(axes[2], d["err_hp"],  r"HP Padé $N=3$")

    fig.suptitle(
        r"Single-mode IC: $\|E_{\rm fluid} - E_{\rm ref}\|/\|E_{\rm ref}\|$  "
        r"(eigenmode IC at $k^*(u_b,\varepsilon)$;  "
        r"$E_{\rm ref}(t)=\mathrm{amp}\,e^{-i\omega t}$)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()

    cb = fig.colorbar(im, ax=fig.axes, shrink=0.85,
                      label=r"$\log_{10}\,\|E_{\rm fluid} - E_{\rm ref}\|/\|E_{\rm ref}\|$")
    cb.set_ticks([-4, -3, -2, -1, 0, 1])
    cb.set_ticklabels([r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$",
                       r"$10^{-1}$", r"$10^{0}$", r"$\geq 10^{1}$"])

    out = FIG_DIR / "fig_singlemode_sweep.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: HP Padé N=3 time series sweep over |ξ_b|
# ---------------------------------------------------------------------------

def fig_pade_xib_sweep(
    cases: list[tuple] | None = None,
    t_end: float | None = None,
    amp: float = 1e-3,
) -> None:
    """9-panel time series: kinetic vs HP Padé N=3 across a range of |ξ_b|.

    Each panel uses the kinetic eigenmode as initial condition and plots |E(t)|
    on a log scale.  γ_HP/γ_kin is shown in the subtitle so the reader can see
    how the growth-rate accuracy varies with |ξ_b|.

    Default cases span |ξ_b| ≈ 0.17 → 2.5, all beam-plasma unstable modes.
    """
    from scipy.stats import linregress
    from bot.closed_loop import pade_evolve, kinetic_evolve, fluid_eigenmode_ic_u3
    from bot.kinetic import most_unstable_kinetic

    if cases is None:
        cases = [
            (3.0, 0.2, 0.05),
            (3.0, 0.3, 0.05),
            (4.0, 0.2, 0.05),
            (3.5, 0.4, 0.05),
            (5.0, 0.2, 0.05),
            (5.0, 0.3, 0.05),
            (6.0, 0.2, 0.05),
            (7.0, 0.2, 0.05),
            (8.0, 0.2, 0.05),
        ]

    a_hp = pade_coefficients(3)

    def _kinetic_eigenmode_ic(k, u_b, eps, omega, N_v=96, V=6.0):
        s, ds = s_grid(N_v, V)
        xi = (omega - k * u_b) / k
        F = (2 * s / SQRT_PI) * np.exp(-s ** 2) / (k * (s - xi)) * amp
        y0 = np.zeros(2 + N_v, dtype=complex)
        y0[1] = amp
        y0[2:] = F
        return y0

    def _fit_gamma(t, E):
        t_min = t[-1] * 0.4
        mask = t >= t_min
        logE = np.log(np.maximum(np.abs(E[mask]), 1e-30))
        slope, *_ = linregress(t[mask], logE)
        return float(slope)

    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, (u_b, k, eps) in zip(axes.flat, cases):
        omega = most_unstable_kinetic(k, u_b, eps)
        xib = (omega - k * u_b) / k

        t_stop = t_end if t_end is not None else min(80.0, 4.0 / max(omega.imag, 0.01))
        t_grid = np.linspace(0, t_stop, 601)

        ic_u3  = fluid_eigenmode_ic_u3(k, u_b, omega, amp)
        ic_kin = _kinetic_eigenmode_ic(k, u_b, eps, omega)

        E_hp  = pade_evolve(k, u_b, eps, a_hp, t_grid, ic_u3)
        E_kin = kinetic_evolve(k, u_b, eps, t_grid, ic_kin)

        g_kin = _fit_gamma(t_grid, E_kin)
        g_hp  = _fit_gamma(t_grid, E_hp)
        ratio = g_hp / g_kin if abs(g_kin) > 1e-6 else np.nan

        ax.semilogy(t_grid, np.abs(E_kin), "k-",  lw=1.5, label=f"kinetic  γ={g_kin:.4f}")
        ax.semilogy(t_grid, np.abs(E_hp),  "C1--", lw=1.5, label=f"HP N=3  γ={g_hp:.4f}")
        ax.set_title(
            f"$u_b={u_b}$, $k={k}$, $\\varepsilon={eps}$  $|\\xi_b|={abs(xib):.2f}$\n"
            f"$\\gamma_{{\\rm HP}}/\\gamma_{{\\rm kin}} = {ratio:.3f}$",
            fontsize=8,
        )
        ax.set_xlabel("$t$", fontsize=8)
        ax.set_ylabel("$|E|$", fontsize=8)
        ax.legend(fontsize=6.5)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        r"HP Padé $N=3$ vs kinetic — eigenmode IC, varying $|\xi_b|$",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIG_DIR / "fig_pade_xib_sweep.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: N=3 bilinear vs N=2 Direct beta vs kinetic, generic dE IC
# ---------------------------------------------------------------------------

def fig_manifold_transient(u_b: float = 4.0, eps: float = 0.02,
                            t_end: float = 80.0, n_t: int = 801,
                            amp: float = 1e-3, w_mix: float = 0.5) -> None:
    """Compare kinetic vs N=3 bilinear vs N=2 Direct beta under three ICs.

    Direct beta enforces U_2 = U_0 beta(U_1/U_0) algebraically at every
    instant, so it is exact on the kinetic growth rate from t=0 regardless
    of IC. Bilinear N=3 instead evolves U_2 as a free dynamical variable
    (closing only U_3 = U_1 U_2 / U_0); how far U_2 starts from the
    beta-manifold value depends on the IC:

      Row 1 — Generic dE IC (u=U_n=0, E=amp): U_2 starts at 0, maximally
        off-manifold -> large overshoot before settling onto gamma_kin.
      Row 2 — Projected kinetic IC, off-eigenmode: U_n = velocity moments
        of a MIXED kinetic state (1-w_mix)*eigenvector(omega1, growing) +
        w_mix*eigenvector(omega2, damped) at the same k.  This is off the
        single-eigenmode manifold for a genuine physical reason (mode
        beating), not just finite-moment truncation.
      Row 3 — Analytic fluid eigenmode IC, off-eigenmode: same mixing
        weight applied to the closed-form single-eigenmode formulas,
        (1-w_mix)*ic(omega1) + w_mix*ic(omega2).  Exactly on the
        single-mode manifold for EITHER omega1 or omega2 alone, but the
        mix is not, since xi_hat = U_1/U_0 is not constant once two
        different complex frequencies beat against each other.

    Columns: (A) |E(t)|, (B) off-manifold distance |U_2-beta(xi_b)U_0|,
    (C) instantaneous growth rate d/dt log|E(t)|.
    """
    from scipy.integrate import solve_ivp
    from scipy.stats import linregress
    from bot.closed_loop import _bilinear_rhs, _direct_u2_rhs
    from bot.sweep import kinetic_peak
    from bot.sweeps.sweep_twomode_growing import _find_two_modes

    ks, ws_kin, i_max = kinetic_peak(u_b, eps)
    k = ks[i_max]

    # actual kinetic eigenvector + its omega (single-mode reference, row 1)
    y0_kin_eig, omega_eig = langmuir_ic_kinetic(k, u_b, eps, amp=amp)
    g_kin = omega_eig.imag

    # growing + damped pair at the same k, for the off-eigenmode rows 2 & 3
    omega1, omega2 = _find_two_modes(k, u_b, eps)  # omega1 == omega_eig

    t_grid = np.linspace(0.0, t_end, n_t)

    def run_bilinear(y0_bil):
        y0r = np.concatenate([y0_bil.real, y0_bil.imag])
        sol = solve_ivp(_bilinear_rhs, (t_grid[0], t_grid[-1]), y0r,
                         t_eval=t_grid, args=(k, u_b, eps),
                         method="RK45", rtol=1e-8, atol=1e-10)
        y = sol.y[:5] + 1j * sol.y[5:]
        return y[1], y[2], y[3], y[4]   # E, U0, U1, U2

    def run_direct(y0_dir):
        y0r = np.concatenate([y0_dir.real, y0_dir.imag])
        sol = solve_ivp(_direct_u2_rhs, (t_grid[0], t_grid[-1]), y0r,
                         t_eval=t_grid, args=(k, u_b, eps),
                         method="RK45", rtol=1e-8, atol=1e-10)
        y = sol.y[:4] + 1j * sol.y[4:]
        E, U0, U1 = y[1], y[2], y[3]
        U2 = U0 * u2_beta(U1 / U0)
        return E, U0, U1, U2

    def inst_gamma(E):
        return np.gradient(np.log(np.maximum(np.abs(E), 1e-300)), t_grid)

    def off_manifold(U0, U1, U2):
        with np.errstate(invalid="ignore", divide="ignore"):
            d = np.abs(U2 - U0 * u2_beta(U1 / U0))
        d[0] = np.nan
        return d

    # ---- three IC strategies -------------------------------------------
    y0_kin_generic = np.zeros(2 + 96, dtype=complex)
    y0_kin_generic[1] = amp
    y0_bil_generic = np.array([0.0, amp, 0.0, 0.0, 0.0], dtype=complex)
    y0_dir_generic = np.array([0.0, amp, 0.0, 0.0], dtype=complex)

    # off-eigenmode mix: (1-w_mix)*growing-mode + w_mix*damped-mode
    y0_kin1 = kinetic_eigenmode_ic(k, u_b, eps, omega1, amp)
    y0_kin2 = kinetic_eigenmode_ic(k, u_b, eps, omega2, amp)
    y0_kin_mix = (1.0 - w_mix) * y0_kin1 + w_mix * y0_kin2

    y0_bil_proj = kinetic_ic_to_fluid(y0_kin_mix, k, u_b)
    y0_dir_proj = y0_bil_proj[:4]

    y0_bil_eigm = ((1.0 - w_mix) * fluid_eigenmode_ic_u3(k, u_b, omega1, amp) +
                   w_mix * fluid_eigenmode_ic_u3(k, u_b, omega2, amp))
    y0_dir_eigm = ((1.0 - w_mix) * fluid_eigenmode_ic_u2(k, u_b, omega1, amp) +
                   w_mix * fluid_eigenmode_ic_u2(k, u_b, omega2, amp))

    rows = [
        ("Generic " r"$\delta E$ IC", y0_kin_generic, y0_bil_generic, y0_dir_generic),
        ("Projected kinetic IC\n(off-eigenmode)", y0_kin_mix, y0_bil_proj, y0_dir_proj),
        ("Analytic eigenmode IC\n(off-eigenmode)", y0_kin_mix, y0_bil_eigm, y0_dir_eigm),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12.5))
    nh = 3 * n_t // 4

    for row, (row_label, y0_kin, y0_bil, y0_dir) in enumerate(rows):
        E_kin = kinetic_evolve(k, u_b, eps, t_grid, y0_kin)
        E_bil, U0_bil, U1_bil, U2_bil = run_bilinear(y0_bil)
        E_dir, U0_dir, U1_dir, U2_dir = run_direct(y0_dir)

        dist_bil = off_manifold(U0_bil, U1_bil, U2_bil)
        dist_dir = off_manifold(U0_dir, U1_dir, U2_dir)

        gam_kin = inst_gamma(E_kin)
        gam_bil = inst_gamma(E_bil)
        gam_dir = inst_gamma(E_dir)

        g_bil_fit = linregress(t_grid[nh:], np.log(np.abs(E_bil[nh:]))).slope
        overshoot_pct = (g_bil_fit - g_kin) / g_kin * 100.0

        ax = axes[row, 0]
        ax.semilogy(t_grid, np.abs(E_kin), "k-", lw=2.0, label="Kinetic")
        ax.semilogy(t_grid, np.abs(E_bil), "--", color="C1", lw=1.5, label="Bilinear N=3")
        ax.semilogy(t_grid, np.abs(E_dir), "--", color="C0", lw=1.5, label="Direct β N=2")
        ax.set_xlabel(r"$t$"); ax.set_ylabel(f"{row_label}\n" + r"$|E(t)|$", fontsize=9)
        title = f"Bilinear overshoot ~{overshoot_pct:.0f}%"
        if row == 0:
            title = "(A)  $|E(t)|$\n" + title
        ax.set_title(title, fontsize=9)
        ax.text(0.03, 0.92, rf"$\gamma_{{\rm kin}}={g_kin:.3f}$",
                transform=ax.transAxes, fontsize=9, va="top")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3, which="both")

        ax = axes[row, 1]
        ax.semilogy(t_grid, np.maximum(dist_bil, 1e-16), "--", color="C1", lw=1.5,
                    label=r"Bilinear N=3")
        ax.semilogy(t_grid, np.maximum(dist_dir, 1e-16), "--", color="C0", lw=1.5,
                    label=r"Direct β N=2: $\equiv 0$")
        ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$|U_2 - \beta(\xi_b)\,U_0|$", fontsize=9)
        if row == 0:
            ax.set_title("(B)  Off-manifold distance", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3, which="both")

        ax = axes[row, 2]
        ax.plot(t_grid, gam_kin, "k:", lw=1.6, label="Kinetic")
        ax.plot(t_grid, gam_bil, "--", color="C1", lw=1.5, label="Bilinear N=3")
        ax.plot(t_grid, gam_dir, "--", color="C0", lw=1.5, label="Direct β N=2")
        ax.axhline(g_kin, color="gray", lw=0.8)
        ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$d\log|E|/dt$", fontsize=9)
        if row == 0:
            ax.set_title("(C)  Instantaneous growth rate", fontsize=9)
        ax.set_ylim(-0.05, max(0.25, 1.3 * np.nanmax(gam_bil[nh:])))
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)

    fig.suptitle(rf"N=3 bilinear vs N=2 Direct β vs kinetic, "
                 rf"$u_b={u_b},\;\varepsilon={eps}$  —  three initializations",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    out = FIG_DIR / "fig_manifold_transient.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    all_figs = "--all" in sys.argv

    print("=== fig_u2_response ===")
    fig_u2_response()

    print("=== fig_u2_time_domain ===")
    fig_u2_time_domain()

    print("=== fig_u2_offmanifold ===")
    fig_u2_offmanifold()

    if all_figs or (RUN_DIR / "sweep_u2_timedomain.npz").exists():
        print("=== fig_u2_sweep ===")
        fig_u2_sweep()
    else:
        print("sweep_u2_timedomain.npz not found — run: python -m bot.sweeps.sweep_u2_timedomain")

    print("=== fig_u2_landau_timedomain ===")
    fig_u2_landau_timedomain()

    print("=== fig_u2_landau_dispref ===")
    fig_u2_landau_dispref()

    print("=== fig_u2_landau_damping ===")
    fig_u2_landau_damping()

    print("=== fig_u2_landau_eigenmode ===")
    fig_u2_landau_eigenmode()

    print("=== fig_u2_landau_strong ===")
    fig_u2_landau_strong()

    print("=== fig_u2_landau_twomode ===")
    fig_u2_landau_twomode()

    print("=== fig_u2_landau_twomode_single ===")
    fig_u2_landau_twomode_single()

    print("=== fig_u2_landau_singlemode ===")
    fig_u2_landau_singlemode()

    if all_figs or (RUN_DIR / "sweep_singlemode.npz").exists():
        print("=== fig_singlemode_sweep ===")
        fig_singlemode_sweep()
    else:
        print("sweep_singlemode.npz not found — run: python -m bot.sweeps.sweep_singlemode")

    print("=== fig_pade_xib_sweep ===")
    fig_pade_xib_sweep()

    print("=== fig_twomode_growing_w1_timedomain ===")
    fig_twomode_growing_w1_timedomain()
