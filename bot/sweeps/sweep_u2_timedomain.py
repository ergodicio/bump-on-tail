"""Time-domain 2D sweep for N=2 (U_2) closures.

For each (u_b, eps) grid point:
  1. Find k* where kinetic gamma_max occurs.
  2. Integrate kinetic, Pade N=2, and direct-beta N=2 under a generic delta-E IC.
  3. Fit gamma_eff from late-time slope of log|E(t)|.

This is the stress test: generic IC excites multiple fluid eigenmodes so
the off-manifold behavior of each closure determines the outcome.

Key contrast vs N=3 analysis:
  * The N=2 direct-beta closure uses beta(U_1/U_0) = U_1^2/U_0^2 - 1/Z'(U_1/U_0)
    which IS the exact kinetic formula applied at the effective xi_b = U_1/U_0.
  * There is no bilinear analogue to compare against (M_0 != 0 prevents it).
  * Pade N=2 is a fixed rational approximation, well-understood.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import linregress

from bot.closed_loop import (direct_u2_evolve, kinetic_evolve, pade_u2_evolve)
from bot.kinetic import most_unstable_kinetic
from bot.sweep import kinetic_peak


def fit_gamma(t: np.ndarray, E: np.ndarray, t_fit_min: float = 25.0) -> float:
    """Fit effective growth rate from late-time log|E| slope."""
    mask = t >= t_fit_min
    logE = np.log(np.maximum(np.abs(E[mask]), 1e-30))
    slope, _, _, _, _ = linregress(t[mask], logE)
    return float(slope)


def run_cell_u2(u_b: float, eps: float,
                t_end: float = 60.0, n_t: int = 601,
                amp: float = 1e-3) -> dict:
    """Time-domain gamma_eff for kinetic / Pade N=2 / direct-beta at peak k."""
    ks, ws_kin, i_max = kinetic_peak(u_b, eps)
    k = ks[i_max]
    gamma_kin_disp = ws_kin[i_max].imag

    t_grid = np.linspace(0.0, t_end, n_t)

    # Generic delta-E IC (N=2: 4 components)
    y0_fluid = np.array([0.0, amp, 0.0, 0.0], dtype=complex)
    # Kinetic IC (u=0, E=amp, F=0)
    from bot.kinetic import kinetic_system_sspace
    N_v = 96
    y0_kin = np.zeros(2 + N_v, dtype=complex)
    y0_kin[1] = amp

    E_kin   = kinetic_evolve(k, u_b, eps, t_grid, y0_kin)
    E_pade2 = pade_u2_evolve(k, u_b, eps, t_grid, y0_fluid)
    E_dir   = direct_u2_evolve(k, u_b, eps, t_grid, y0_fluid)

    g_kin   = fit_gamma(t_grid, E_kin)
    g_pade2 = fit_gamma(t_grid, E_pade2)
    g_dir   = fit_gamma(t_grid, E_dir)

    def _over(g):
        return (g - g_kin) / g_kin if g_kin > 0 else np.nan

    return {
        "u_b": u_b, "eps": eps, "k_star": k,
        "gamma_kin_disp": gamma_kin_disp,
        "gamma_kin":      g_kin,
        "gamma_pade2":    g_pade2,
        "gamma_direct":   g_dir,
        "overshoot_pade2":  _over(g_pade2),
        "overshoot_direct": _over(g_dir),
    }


def run_grid_u2(u_b_vals: np.ndarray, eps_vals: np.ndarray) -> dict:
    """Full 2D grid sweep for N=2 time-domain overshoot."""
    n_ub, n_eps = len(u_b_vals), len(eps_vals)
    shape = (n_ub, n_eps)

    keys = ["gamma_kin", "gamma_pade2", "gamma_direct",
            "overshoot_pade2", "overshoot_direct", "k_star"]
    arrs = {k: np.full(shape, np.nan) for k in keys}

    for i, u_b in enumerate(u_b_vals):
        for j, eps in enumerate(eps_vals):
            try:
                r = run_cell_u2(u_b, eps)
                for key in keys:
                    arrs[key][i, j] = r[key]
            except Exception as e:
                print(f"  [u_b={u_b:.2f}, eps={eps:.3f}]  FAIL: {e}")
        print(f"u_b={u_b:.2f}  done")

    arrs["u_b_vals"] = u_b_vals
    arrs["eps_vals"] = eps_vals
    return arrs


if __name__ == "__main__":
    u_b_vals = np.linspace(3.0, 10.0, 8)
    eps_vals = np.array([0.01, 0.02, 0.05, 0.10, 0.15, 0.20])

    print(f"N=2 time-domain sweep: {len(u_b_vals)} x {len(eps_vals)} grid")
    results = run_grid_u2(u_b_vals, eps_vals)

    out = Path(__file__).resolve().parent.parent / "runs" / "sweep_u2_timedomain.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out, **results)
    print(f"\nsaved {out}")

    over_p = results["overshoot_pade2"] * 100
    over_d = results["overshoot_direct"] * 100
    header = "  u_b\\eps  " + "  ".join(f"{e:6.3f}" for e in eps_vals)
    for label, over in [("Pade N=2", over_p), ("direct-beta", over_d)]:
        print(f"\nTime-domain gamma_eff overshoot (%), {label}:")
        print(header)
        for i, ub in enumerate(u_b_vals):
            row = "  ".join(
                f"{over[i, j]:+6.2f}" if np.isfinite(over[i, j]) else "    NaN"
                for j in range(len(eps_vals))
            )
            print(f"  {ub:5.2f}    {row}")
