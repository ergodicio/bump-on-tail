"""Driver: fixing the direct closure for slab ITG — shift and manifold variants.

Generates bot/figures/fig_slab_itg_direct_variants.png and
bot/runs/slab_itg_direct_variants.npz.

Variants of the N=2 direct closure U_2 = C(.) U_0 (bot/slab_itg.py):
  * plain:    C = beta(U_1/U_0)                 — gradient-free closure at the
              raw ratio; eta-independent, no ITG (baseline failure).
  * shifted:  C = beta(zeta_hat),  zeta_hat = U_1/U_0 - zeta_*/tau — recovers
              the mode frequency from the moments, but keeps the gradient-free
              beta; its dispersion is exactly the kinetic one frozen at eta=2,
              so it is still eta-independent.
  * itg:      C = beta_itg(zeta_hat; zeta_*, eta) — the ITG-manifold ratio;
              the kinetic root is an exact eigenvalue of the closed system and
              (numerically) its only growing one: the correct generalization.

Also shown: N=3 alpha_itg — exact on the eigenmode too, but possesses a
spurious growing root at Re(zeta) ~ 0.25, gamma ~ 1.3-1.9 that dominates any
IC (exact-but-unstable, cf. BoT paper section 4).
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bot.slab_itg import (D_kinetic, max_growing_root, find_mode,
                          eta_threshold, fluid_hp_modes, kinetic_evolve,
                          direct_beta_evolve, direct_alpha_evolve,
                          density_ic_kinetic, density_ic_fluid, fit_mode)

ZETA_STAR = 1.0
TAU = 1.0
N_W = 600
ETA_TRACE = 4.5
SAVE_PNG = "bot/figures/fig_slab_itg_direct_variants.png"
SAVE_NPZ = "bot/runs/slab_itg_direct_variants.npz"


def kinetic_root_curve(eta_values):
    """Kinetic ITG-branch root vs eta, continued through the damped side."""
    zs = np.empty(len(eta_values), dtype=complex)
    z_prev = None
    for i, e in enumerate(eta_values[::-1]):   # start deep in unstable range
        z = max_growing_root(D_kinetic, ZETA_STAR, e, TAU)
        if np.isnan(z.real) and z_prev is not None:
            z = find_mode(D_kinetic, z_prev, ZETA_STAR, e, TAU)
        zs[len(eta_values) - 1 - i] = z
        if not np.isnan(z.real):
            z_prev = z
    return zs


def scan(eta_curve, eta_fit):
    out = {"eta_curve": eta_curve, "eta_fit": eta_fit}
    print("kinetic ITG-branch curve (incl. damped continuation)...")
    out["z_kin"] = kinetic_root_curve(eta_curve)
    out["z_hp"] = np.array(
        [max(fluid_hp_modes(ZETA_STAR, e, TAU), key=lambda z: z.imag)
         for e in eta_curve])

    print("time-domain fits...")
    t = np.linspace(0.0, 70.0, 701)
    y2 = density_ic_fluid(N=2)
    y3 = density_ic_fluid(N=3)
    runs = {
        "beta_plain": lambda e: direct_beta_evolve(
            ZETA_STAR, e, TAU, t, y2, variant="plain"),
        "beta_shift": lambda e: direct_beta_evolve(
            ZETA_STAR, e, TAU, t, y2, variant="shifted"),
        "beta_itg": lambda e: direct_beta_evolve(
            ZETA_STAR, e, TAU, t, y2, variant="itg"),
        "alpha_itg": lambda e: direct_alpha_evolve(
            ZETA_STAR, e, TAU, t, y3, variant="itg"),
        "kin": lambda e: kinetic_evolve(
            ZETA_STAR, e, TAU, t, density_ic_kinetic(N_w=N_W), N_w=N_W),
    }
    for key in runs:
        out[f"fit_{key}"] = np.empty(len(eta_fit), dtype=complex)
    for i, e in enumerate(eta_fit):
        for key, fn in runs.items():
            r = fn(e)
            out[f"fit_{key}"][i] = fit_mode(r["t"], r["U0"], frac=(0.5, 1.0))
        print(f"  eta={e:4.2f}  kin={out['fit_kin'][i]:.4f}  "
              f"b_itg={out['fit_beta_itg'][i]:.4f}  "
              f"b_shift={out['fit_beta_shift'][i]:.4f}  "
              f"a_itg={out['fit_alpha_itg'][i]:.4f}")
    return out


def run_traces():
    t = np.linspace(0.0, 60.0, 601)
    y2 = density_ic_fluid(N=2)
    out = {"t_trace": t}
    out["tr_kin"] = kinetic_evolve(ZETA_STAR, ETA_TRACE, TAU, t,
                                   density_ic_kinetic(N_w=N_W), N_w=N_W)["U0"]
    for variant, key in [("plain", "tr_plain"), ("shifted", "tr_shift"),
                         ("itg", "tr_itg")]:
        out[key] = direct_beta_evolve(ZETA_STAR, ETA_TRACE, TAU, t, y2,
                                      variant=variant)["U0"]
    out["tr_alpha_itg"] = direct_alpha_evolve(
        ZETA_STAR, ETA_TRACE, TAU, t, density_ic_fluid(N=3), variant="itg")["U0"]
    return out


def make_fig(r):
    eta_c, eta_f = r["eta_curve"], r["eta_fit"]
    eta_th = eta_threshold(ZETA_STAR, TAU)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    ax = axes[0]
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(eta_th, color="gray", lw=0.8, ls="--",
               label=rf"$\eta_{{th}}={eta_th:.3f}$")
    ax.plot(eta_c, r["z_kin"].imag, "k-", lw=2.5,
            label="kinetic ITG branch (dispersion)")
    ax.plot(eta_c, r["z_hp"].imag, "C1-", lw=1.4, alpha=0.8,
            label=r"HP fluid ($\Gamma=3$)")
    ax.plot(eta_f, r["fit_kin"].imag, "ko", ms=7, mfc="none",
            label="kinetic time-domain fit")
    ax.plot(eta_f, r["fit_beta_itg"].imag, "C2*", ms=10,
            label=r"$\beta_{itg}(\hat\zeta)$ N=2 fit")
    ax.plot(eta_f, r["fit_beta_shift"].imag, "C0^", ms=6,
            label=r"$\beta(\hat\zeta)$ shifted-only fit")
    ax.plot(eta_f, r["fit_beta_plain"].imag, "C0+", ms=7,
            label=r"$\beta(U_1/U_0)$ plain fit")
    ax.plot(eta_f, r["fit_alpha_itg"].imag, "C3v", ms=6,
            label=r"$\alpha_{itg}(\hat\zeta)$ N=3 fit (spurious)")
    ax.set_xlabel(r"$\eta_i = L_n/L_T$", fontsize=12)
    ax.set_ylabel(r"$\gamma / (|k_\parallel| v_{ti}\sqrt{2})$", fontsize=11)
    ax.set_title("Growth/damping rate of the fitted mode", fontsize=11)
    ax.legend(fontsize=7.5, loc="upper left")

    ax = axes[1]
    t = r["t_trace"]
    ax.semilogy(t, np.abs(r["tr_kin"]), "k-", lw=2.4, label="kinetic DKE")
    ax.semilogy(t, np.abs(r["tr_itg"]), "C2-", lw=1.6,
                label=r"$\beta_{itg}(\hat\zeta)$ N=2")
    ax.semilogy(t, np.abs(r["tr_shift"]), "C0--", lw=1.6,
                label=r"$\beta(\hat\zeta)$ shifted-only")
    ax.semilogy(t, np.abs(r["tr_plain"]), "C0:", lw=1.6,
                label=r"$\beta(U_1/U_0)$ plain")
    ax.semilogy(t, np.abs(r["tr_alpha_itg"]), "C3-.", lw=1.4,
                label=r"$\alpha_{itg}(\hat\zeta)$ N=3")
    ax.set_xlabel(r"$t \, |k_\parallel| v_{ti}\sqrt{2}$", fontsize=11)
    ax.set_ylabel(r"$|\tilde n_i/n_0|$", fontsize=11)
    ax.set_ylim(1e-7, None)
    ax.set_title(rf"Time traces at $\eta_i = {ETA_TRACE}$, $\delta n$ IC",
                 fontsize=11)
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(r"Repairing the direct closure for slab ITG: "
                 r"$\hat\zeta = U_1/U_0 - \zeta_*/\tau$ shift and "
                 r"ITG-manifold ratio "
                 rf"($\zeta_*={ZETA_STAR}$, $\tau={TAU}$)", fontsize=12)
    plt.tight_layout()
    plt.savefig(SAVE_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {SAVE_PNG}")


if __name__ == "__main__":
    eta_curve = np.linspace(0.5, 6.5, 61)
    eta_fit = np.array([1.5, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0])
    r = {**scan(eta_curve, eta_fit), **run_traces()}
    np.savez_compressed(SAVE_NPZ, **r)
    print(f"Saved {SAVE_NPZ}")
    make_fig(r)
