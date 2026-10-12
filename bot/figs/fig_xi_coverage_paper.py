"""Driver: training rectangle vs kinetic modes in the complex zeta plane.

Generates bot/figures/fig_xi_coverage_paper.png, the manuscript version of
fig_xi_coverage (bot/figures.py), single-column paper style.

  * training rectangle Re zeta in (-3, 3), Im zeta in (-1, 1.5)
  * most unstable kinetic BoT mode on the fig_xi_coverage grid
    (u_b, eps, k), with the Re(omega) <= 1.5 window of bot.sweep.kinetic_peak
  * parameter-scan modes at k_* from bot/runs/sweep_opt_hunana_homog.npz
  * damped two-mode pairs of fig_u2_landau_twomode_2panel
  * standing-wave root pair at k lambda_D = 0.4 (fig_standing_wave)

With --modes, only the rectangle, the scan modes and the test modes are drawn,
written to bot/figures/fig_xi_coverage_modes.png.

With --sim, the growing-mode cloud is replaced by the beam's instantaneous
frequency zeta_hat(t) = U_1/U_0 along the 48 kinetic runs of the parameter
scan (field-kick start, N_v = 96, run length as in sweep_opt_hunana.run_cell),
written to bot/figures/fig_xi_coverage_sim.png.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import newton

from bot.closures.u2 import Z
from bot.figures_u2 import _find_landau_twomode_modes
from bot.kinetic import kinetic_system_sspace, most_unstable_kinetic, s_grid

RE_RANGE, IM_RANGE = (-3.0, 3.0), (-1.0, 1.5)
RECT4X3 = ((-4.0, 4.0), (-1.0, 3.0))   # bot.sweeps.rect_extended
U_B_VALS = np.linspace(3.0, 10.0, 12)
EPS_VALS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20)
KS = np.linspace(0.10, 0.45, 24)
OMEGA_RE_HI = 1.5
NPZ_SCAN = "bot/runs/sweep_opt_hunana_homog.npz"
TWOMODE_CASES = ((1.0, 0.50, 0.05), (1.5, 0.40, 0.05))
K_LAMBDA_D = 0.4

SAVE_PNG = "bot/figures/fig_xi_coverage_paper.png"
SAVE_PNG_SIM = "bot/figures/fig_xi_coverage_sim.png"
SAVE_PNG_MODES = "bot/figures/fig_xi_coverage_modes.png"
RC = {"font.family": "serif", "mathtext.fontset": "cm",
      "font.size": 7, "axes.titlesize": 7}


def zeta_b(omega: complex, k: float, u_b: float) -> complex:
    return (omega - k * u_b) / k


def in_rectangle(z):
    return ((z.real > RE_RANGE[0]) & (z.real < RE_RANGE[1])
            & (z.imag > IM_RANGE[0]) & (z.imag < IM_RANGE[1]))


def scan_outside_mask() -> np.ndarray:
    """(n_u_b, n_eps) mask of parameter-scan modes outside the training rectangle."""
    d = np.load(NPZ_SCAN)
    shape = (len(d["u_b_vals"]), len(d["eps_vals"]))
    return ~in_rectangle(scan_modes()).reshape(shape)


def unstable_branch() -> np.ndarray:
    out = []
    for u_b in U_B_VALS:
        for eps in EPS_VALS:
            for k in KS:
                om = most_unstable_kinetic(k, u_b, eps, omega_re_hi=OMEGA_RE_HI)
                if om.imag > 0.005:
                    out.append(zeta_b(om, k, u_b))
    return np.array(out)


def scan_modes() -> np.ndarray:
    d = np.load(NPZ_SCAN)
    out = []
    for i, u_b in enumerate(d["u_b_vals"]):
        for j, eps in enumerate(d["eps_vals"]):
            k = float(d["k_star"][i, j])
            om = most_unstable_kinetic(k, u_b, eps, N_v=96, V=6.0,
                                       omega_re_hi=OMEGA_RE_HI)
            out.append(zeta_b(om, k, u_b))
    return np.array(out)


def scan_trajectories(t_end_min: float = 60.0, t_fit_min: float = 25.0,
                      n_efold: float = 5.0, dt: float = 0.1,
                      amp: float = 1e-3) -> list[np.ndarray]:
    d = np.load(NPZ_SCAN)
    s, _ = s_grid(96, 6.0)
    out = []
    for i, u_b in enumerate(d["u_b_vals"]):
        for j, eps in enumerate(d["eps_vals"]):
            k = float(d["k_star"][i, j])
            om = most_unstable_kinetic(k, u_b, eps, N_v=96, V=6.0,
                                       omega_re_hi=OMEGA_RE_HI)
            t_end = max(t_end_min, t_fit_min + n_efold / om.imag)
            t = np.arange(1, int(t_end / dt) + 1) * dt
            lam, vec = np.linalg.eig(kinetic_system_sspace(k, u_b, eps, 96, 6.0))
            y0 = np.zeros(len(lam), dtype=complex)
            y0[1] = amp
            F = (vec @ (np.linalg.solve(vec, y0)[:, None]
                        * np.exp(lam[:, None] * t[None, :])))[2:]
            out.append((s[:, None] * F).sum(0) / F.sum(0))
    return out


def twomode_modes() -> np.ndarray:
    out = []
    for u_b, k, eps in TWOMODE_CASES:
        out += [zeta_b(w, k, u_b) for w in _find_landau_twomode_modes(k, u_b, eps)[:2]]
    return np.array(out)


def standing_wave_roots() -> np.ndarray:
    K = K_LAMBDA_D * np.sqrt(2.0)
    z = newton(lambda z: 1.0 + (2.0 / K ** 2) * (1.0 + z * Z(z)), 2.3 - 0.1j, tol=1e-12)
    return np.array([z, -np.conj(z)])


def main(variant: str = "branch") -> None:
    sim = variant == "sim"
    scan = scan_modes()
    pairs, sw = twomode_modes(), standing_wave_roots()
    if sim:
        trajs = scan_trajectories()
        allz = np.concatenate(trajs)
        print(f"kinetic runs: {len(trajs)}, samples {len(allz)}, "
              f"Im<0 fraction {np.mean(allz.imag < 0):.4f}")
    elif variant == "branch":
        branch = unstable_branch()

    inside = in_rectangle

    if variant == "branch":
        print(f"unstable branch: {len(branch)} modes, {np.sum(~inside(branch))} outside")
    print(f"scan modes: {len(scan)}, {np.sum(~inside(scan))} outside")

    with matplotlib.rc_context(RC):
        fig, ax = plt.subplots(figsize=(3.4, 2.8 if sim else 2.45))
        x0, x1 = RE_RANGE
        y0, y1 = IM_RANGE
        ax.fill([x0, x1, x1, x0], [y0, y0, y1, y1], color="C0", alpha=0.12,
                lw=0, zorder=0, label="Training rectangle")
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="C0",
                lw=0.8, zorder=1)
        ax.axhline(0.0, color="0.5", lw=0.4, zorder=1)
        if sim:
            for n, z in enumerate(trajs):
                ax.plot(z.real, z.imag, color="0.55", lw=0.35, alpha=0.7,
                        zorder=2, label=r"Kinetic runs, $\hat{\zeta}(t)$" if n == 0 else None)
        elif variant == "branch":
            ax.scatter(branch.real, branch.imag, s=1.5, color="0.6", lw=0,
                       zorder=2, label=r"Growing modes, $k \in [0.1, 0.45]$")
        ax.scatter(scan.real, scan.imag, s=9, facecolors="none",
                   edgecolors="k", lw=0.6, zorder=3,
                   label={"branch": "Parameter scan"}.get(variant, "Most unstable modes"))
        ax.scatter(pairs.real, pairs.imag, s=14, marker="D", color="C1",
                   lw=0, zorder=4, label="Two-mode tests")
        ax.scatter(sw.real, sw.imag, s=14, marker="s", color="C3",
                   lw=0, zorder=4, label="Standing wave")
        ax.set_xlim(min(-4.0, x0 - 0.3), max(3.4, x1 + 0.3))
        ax.set_ylim(-1.25, max(3.1, y1 + 0.2))
        ax.set_xlabel(r"$\mathrm{Re}\,\zeta$")
        ax.set_ylabel(r"$\mathrm{Im}\,\zeta$")
        if sim:
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
                      fontsize=5.5, handletextpad=0.4, borderpad=0.3,
                      labelspacing=0.25, columnspacing=1.0, frameon=False)
        else:
            ax.legend(loc="upper right", fontsize=5.5, handletextpad=0.4,
                      borderpad=0.3, labelspacing=0.25, framealpha=0.9)
        fig.tight_layout()
        out = {"sim": SAVE_PNG_SIM, "modes": SAVE_PNG_MODES}.get(variant, SAVE_PNG)
        fig.savefig(out, dpi=400, bbox_inches="tight")
        plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    import sys
    if "--rect4x3" in sys.argv[1:]:
        RE_RANGE, IM_RANGE = RECT4X3
        SAVE_PNG_SIM = "bot/figures/fig_xi_coverage_sim_rect4x3.png"
    args = sys.argv[1:]
    main("sim" if "--sim" in args else "modes" if "--modes" in args else "branch")
