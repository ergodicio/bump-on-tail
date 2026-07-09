"""Driver: spurious-root comparison — HP closure vs direct manifold closures.

Generates bot/figures/fig_slab_itg_hp_spectrum.png and
bot/runs/slab_itg_hp_spectrum.npz.

Two complementary ways a closure's spectrum can deviate from kinetic:

  * HP (Gamma=3, linear, 3x3): approximates R(zeta) itself by a 3-pole
    rational R_3; its THREE roots are all approximations of true kinetic
    roots (ITG branch, drift branch, one deep pole standing in for the
    Landau continuum) — no extra roots, and (verified here to 1e-9) its
    driven dispersion is EXACTLY R_s with Z -> Z_3 substituted, so the
    frequency-domain substitution test is rigorous for this closure.
    Gamma=5/3 breaks the substitution identity AND is spuriously unstable
    below the kinetic threshold.

  * Direct manifold closures (nonlinear in the state): keep R exact, but the
    characteristic equation picks up prefactor zeros, char_N =
    P_N(zeta)(R+tau)/(tau R)  (notes Appendix A.5).  N=2: P_2 real zero only
    (clean).  N=3: P_3 UHP zero at Re = 1/(4 zeta_*),
    gamma = sqrt(2 zeta_*^2(1+eta) - 1/4)/(2 zeta_*)  (always dominant).

Panel (a): UHP/near-UHP root trajectories vs eta at zeta_* = tau = 1.
Panel (b): stability threshold eta_th(zeta_*): kinetic analytic, HP numeric
(Gamma = 3 and 5/3), and the analytic onset of the N=3 spurious root
eta_sp = max(0, 1/(8 zeta_*^2) - 1).
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bot.slab_itg import (D_kinetic, max_growing_root, find_mode,
                          eta_threshold, fluid_hp_modes)

TAU = 1.0
ZETA_STAR = 1.0
SAVE_PNG = "bot/figures/fig_slab_itg_hp_spectrum.png"
SAVE_NPZ = "bot/runs/slab_itg_hp_spectrum.npz"


def kinetic_branch(eta_values, zeta_star=ZETA_STAR, tau=TAU):
    """ITG-branch kinetic root vs eta, continued through the damped side."""
    zs = np.empty(len(eta_values), dtype=complex)
    z_prev = None
    for i, e in enumerate(eta_values[::-1]):
        z = max_growing_root(D_kinetic, zeta_star, e, tau)
        if np.isnan(z.real) and z_prev is not None:
            z = find_mode(D_kinetic, z_prev, zeta_star, e, tau)
        zs[len(eta_values) - 1 - i] = z
        if not np.isnan(z.real):
            z_prev = z
    return zs


def hp_root_tracks(eta_values, zeta_star=ZETA_STAR, tau=TAU, Gamma=3.0):
    """Track the 3 HP eigenvalues continuously in eta (nearest-neighbour)."""
    tracks = np.empty((3, len(eta_values)), dtype=complex)
    prev = None
    for i, e in enumerate(eta_values):
        eig = fluid_hp_modes(zeta_star, e, tau, Gamma=Gamma)
        if prev is None:
            eig = np.array(sorted(eig, key=lambda z: -z.imag))
        else:
            out = np.empty(3, dtype=complex)
            used = set()
            for j in range(3):
                d = [abs(eig[m] - prev[j]) if m not in used else np.inf
                     for m in range(3)]
                m = int(np.argmin(d))
                used.add(m)
                out[j] = eig[m]
            eig = out
        tracks[:, i] = eig
        prev = eig
    return tracks


def spurious_p3(eta_values, zeta_star=ZETA_STAR):
    """UHP zero of P_3 = zeta_* z^2 - z/2 + zeta_*(1+eta)/2 (Appendix A.5)."""
    disc = 2.0 * zeta_star**2 * (1.0 + np.asarray(eta_values)) - 0.25
    return (0.5 + 1j * np.sqrt(np.maximum(disc, 0.0))) / (2.0 * zeta_star)


def hp_threshold(zeta_star, tau=TAU, Gamma=3.0,
                 eta_lo=0.05, eta_hi=60.0) -> float:
    """Bisect max Im eig(HP) = 0 in eta.  nan if no crossing in range."""
    def g(e):
        return max(fluid_hp_modes(zeta_star, e, tau, Gamma=Gamma),
                   key=lambda z: z.imag).imag
    lo, hi = eta_lo, eta_hi
    if g(lo) > 0:
        return 0.0
    if g(hi) < 0:
        return float("nan")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if g(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def run():
    out = {}
    out["eta_traj"] = np.linspace(1.0, 6.0, 101)
    print("root trajectories at zeta_* = 1 ...")
    out["z_kin_traj"] = kinetic_branch(out["eta_traj"])
    out["z_hp_traj"] = hp_root_tracks(out["eta_traj"], Gamma=3.0)
    out["z_brag_traj"] = hp_root_tracks(out["eta_traj"], Gamma=5.0 / 3.0)
    out["z_p3_traj"] = spurious_p3(out["eta_traj"])

    print("thresholds vs zeta_* ...")
    out["zs_scan"] = np.geomspace(0.3, 4.0, 25)
    out["eta_th_kin"] = np.array([eta_threshold(z, TAU) for z in out["zs_scan"]])
    out["eta_th_hp"] = np.array([hp_threshold(z, Gamma=3.0)
                                 for z in out["zs_scan"]])
    out["eta_th_brag"] = np.array([hp_threshold(z, Gamma=5.0 / 3.0)
                                   for z in out["zs_scan"]])
    out["eta_sp_alpha"] = np.maximum(1.0 / (8.0 * out["zs_scan"]**2) - 1.0, 0.0)
    return out


def make_fig(r):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    # ── (a) root trajectories in the complex plane ──────────────────────────
    ax = axes[0]
    ax.axhline(0, color="k", lw=0.6)
    eta = r["eta_traj"]

    def mark_eta(z_arr, color):
        for e_mark, m in [(1.0, "o"), (3.5, "s"), (6.0, "D")]:
            i = int(np.argmin(np.abs(eta - e_mark)))
            z = z_arr[i]
            if not np.isnan(z.real):
                ax.plot(z.real, z.imag, m, color=color, ms=5, mfc="none")

    z = r["z_kin_traj"]
    ax.plot(z.real, z.imag, "k-", lw=2.6,
            label=r"kinetic ITG branch (= $\beta_{itg}$ N=2 root)")
    mark_eta(z, "k")
    for j in range(3):
        z = r["z_hp_traj"][j]
        ax.plot(z.real, z.imag, "C1-", lw=1.6,
                label=r"HP $\Gamma\!=\!3$ (all 3 roots)" if j == 0 else None)
        mark_eta(z, "C1")
    z = r["z_brag_traj"][0]
    ax.plot(z.real, z.imag, "C2--", lw=1.4,
            label=r"$\Gamma\!=\!5/3$ (most unstable)")
    mark_eta(z, "C2")
    z = r["z_p3_traj"]
    ax.plot(z.real, z.imag, "C3-", lw=1.8,
            label=r"$\alpha_{itg}$ N=3 spurious ($P_3$ zero)")
    mark_eta(z, "C3")

    ax.set_xlabel(r"$\mathrm{Re}\,\zeta$", fontsize=11)
    ax.set_ylabel(r"$\mathrm{Im}\,\zeta$", fontsize=11)
    ax.set_title(r"Root trajectories, $\eta: 1 \to 6$"
                 r"  (markers: $\eta=1,\,3.5,\,6$;  $\zeta_*=\tau=1$)",
                 fontsize=10.5)
    ax.legend(fontsize=7.5, loc="center left")

    # ── (b) stability thresholds vs zeta_* ──────────────────────────────────
    ax = axes[1]
    zs = r["zs_scan"]
    ax.plot(zs, r["eta_th_kin"], "k-", lw=2.6,
            label=r"kinetic $1+\sqrt{1+2\tau(1+\tau)/\zeta_*^2}$"
                  r" (= $\beta_{itg}$ N=2)")
    ax.plot(zs, r["eta_th_hp"], "C1--", lw=1.8, label=r"HP $\Gamma=3$")
    ax.plot(zs, r["eta_th_brag"], "C2:", lw=1.8, label=r"HP $\Gamma=5/3$")
    ax.plot(zs, r["eta_sp_alpha"], "C3-.", lw=1.8,
            label=r"$\alpha_{itg}$ N=3 spurious onset $1/(8\zeta_*^2)-1$")
    ax.axhline(2.0, color="gray", lw=0.8, ls=":",
               label=r"$\eta=2$ ($\zeta_*\to\infty$ limit)")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\zeta_* = \omega_*/(|k_\parallel| v_{ti}\sqrt{2})$",
                  fontsize=11)
    ax.set_ylabel(r"$\eta_{th}$", fontsize=11)
    ax.set_ylim(-0.3, 12)
    ax.set_title(r"Stability threshold $\eta_{th}(\zeta_*)$  ($\tau=1$):"
                 " growth below the kinetic curve is spurious", fontsize=10.5)
    ax.legend(fontsize=7.5, loc="upper right")

    fig.suptitle("Spurious-root anatomy: 3-pole response approximation (HP) "
                 "vs exact-response direct closures", fontsize=12)
    plt.tight_layout()
    plt.savefig(SAVE_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {SAVE_PNG}")


def report(r):
    zs = r["zs_scan"]
    print("\n── threshold table (tau=1) ──")
    print(f"  {'zeta_*':>7} {'kinetic':>9} {'HP G=3':>9} {'G=5/3':>9} "
          f"{'a_itg spurious onset':>21}")
    for i in range(0, len(zs), 4):
        print(f"  {zs[i]:7.3f} {r['eta_th_kin'][i]:9.4f} "
              f"{r['eta_th_hp'][i]:9.4f} {r['eta_th_brag'][i]:9.4f} "
              f"{r['eta_sp_alpha'][i]:21.4f}")


if __name__ == "__main__":
    r = run()
    np.savez_compressed(SAVE_NPZ, **r)
    print(f"Saved {SAVE_NPZ}")
    make_fig(r)
    report(r)
