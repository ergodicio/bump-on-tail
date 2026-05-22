"""Time-domain 2D sweep: γ_eff at peak k for kinetic / Padé / inference.

For each (u_b, eps):
  1. Find k* where kinetic γ_max occurs (via existing sweep machinery).
  2. Generic δE initial condition; integrate all three systems.
  3. Fit γ_eff from late-time slope of log |E(t)|.
  4. Compare overshoots vs kinetic.

This is the stress test: generic IC excites *everything*, so closure quality
shows up. Mode-tracking artefacts (from the eigenvalue-based dispersion
sweep) are absent here — γ_eff is measured directly from the trajectory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import linregress

from bot.closed_loop import (bilinear_evolve, generic_dE_ic,
                              generic_dE_ic_kinetic, inference_evolve,
                              kinetic_evolve, naive_evolve, pade_evolve,
                              sim_evolve)
from bot.closures.naive_nn import load as load_naive
from bot.closures.pade import pade_coefficients
from bot.kinetic import most_unstable_kinetic
from bot.sweep import kinetic_peak
from bot.train_inference import load as load_inference
from bot.train_simdata import load as load_simdata


def fit_gamma(t: np.ndarray, E: np.ndarray, t_fit_min: float = 25.0) -> float:
    """Fit effective growth rate from late-time log|E| slope."""
    mask = t >= t_fit_min
    logE = np.log(np.maximum(np.abs(E[mask]), 1e-30))
    slope, _, _, _, _ = linregress(t[mask], logE)
    return float(slope)


def run_cell(u_b: float, eps: float, model_inf, model_naive, model_sim,
              t_end: float = 60.0, n_t: int = 601,
              amp: float = 1e-3) -> dict:
    """Time-domain γ_eff for kinetic / Padé / inference / naive / sim at peak k."""
    ks, ws_kin, i_max = kinetic_peak(u_b, eps)
    k = ks[i_max]
    gamma_kin_disp = ws_kin[i_max].imag

    t_grid = np.linspace(0.0, t_end, n_t)

    y0_fluid = generic_dE_ic(amp=amp)
    y0_kin = generic_dE_ic_kinetic(amp=amp)
    a = pade_coefficients(3)

    E_kin   = kinetic_evolve(k, u_b, eps, t_grid, y0_kin)
    E_pade  = pade_evolve(k, u_b, eps, a, t_grid, y0_fluid)
    E_inf   = inference_evolve(k, u_b, eps, model_inf, t_grid, y0_fluid)
    E_naive = naive_evolve(k, u_b, eps, model_naive, t_grid, y0_fluid)
    E_sim   = sim_evolve(k, u_b, eps, model_sim, t_grid, y0_fluid)
    E_bilin = bilinear_evolve(k, u_b, eps, t_grid, y0_fluid)

    g_kin   = fit_gamma(t_grid, E_kin)
    g_pade  = fit_gamma(t_grid, E_pade)
    g_inf   = fit_gamma(t_grid, E_inf)
    g_naive = fit_gamma(t_grid, E_naive)
    g_sim   = fit_gamma(t_grid, E_sim)
    g_bilin = fit_gamma(t_grid, E_bilin)

    return {
        "u_b": u_b, "eps": eps, "k_star": k,
        "gamma_kin_disp": gamma_kin_disp,
        "gamma_kin_td":   g_kin,
        "gamma_pade_td":  g_pade,
        "gamma_inf_td":   g_inf,
        "gamma_naive_td": g_naive,
        "gamma_sim_td":   g_sim,
        "gamma_bilin_td": g_bilin,
        "overshoot_pade":  (g_pade  - g_kin) / g_kin if g_kin > 0 else np.nan,
        "overshoot_inf":   (g_inf   - g_kin) / g_kin if g_kin > 0 else np.nan,
        "overshoot_naive": (g_naive - g_kin) / g_kin if g_kin > 0 else np.nan,
        "overshoot_sim":   (g_sim   - g_kin) / g_kin if g_kin > 0 else np.nan,
        "overshoot_bilin": (g_bilin - g_kin) / g_kin if g_kin > 0 else np.nan,
    }


def run_grid(u_b_vals: np.ndarray, eps_vals: np.ndarray) -> dict:
    model_inf = load_inference()
    model_naive = load_naive()
    model_sim = load_simdata(broad=True)
    n_ub, n_eps = len(u_b_vals), len(eps_vals)

    keys = ["gamma_kin_td", "gamma_pade_td", "gamma_inf_td", "gamma_naive_td",
            "gamma_sim_td", "gamma_bilin_td", "overshoot_pade",
            "overshoot_inf", "overshoot_naive", "overshoot_sim",
            "overshoot_bilin", "k_star"]
    arrs = {k: np.full((n_ub, n_eps), np.nan) for k in keys}

    for i, u_b in enumerate(u_b_vals):
        for j, eps in enumerate(eps_vals):
            try:
                r = run_cell(u_b, eps, model_inf=model_inf,
                             model_naive=model_naive, model_sim=model_sim)
                for k in keys:
                    arrs[k][i, j] = r[k]
            except Exception as e:
                print(f"  [u_b={u_b:.2f}, eps={eps:.3f}]  FAIL: {e}")
        print(f"u_b={u_b:.2f}  done")

    arrs["u_b_vals"] = u_b_vals
    arrs["eps_vals"] = eps_vals
    return arrs


if __name__ == "__main__":
    u_b_vals = np.linspace(3.0, 10.0, 8)
    eps_vals = np.array([0.01, 0.02, 0.05, 0.10, 0.15, 0.20])

    print(f"time-domain 2D sweep: {len(u_b_vals)} x {len(eps_vals)} = "
          f"{len(u_b_vals) * len(eps_vals)} cells")

    results = run_grid(u_b_vals, eps_vals)

    out = Path(__file__).resolve().parent / "runs" / "sweep_timedomain.npz"
    np.savez_compressed(out, **results)
    print(f"\nsaved {out}")

    over_p = results["overshoot_pade"] * 100
    over_i = results["overshoot_inf"] * 100
    over_n = results["overshoot_naive"] * 100
    over_b = results["overshoot_bilin"] * 100
    header = "  u_b\\eps  " + "  ".join(f"{e:6.3f}" for e in eps_vals)
    for label, over in [("Padé", over_p), ("inference", over_i),
                          ("naive NN", over_n), ("bilinear", over_b)]:
        print(f"\nTime-domain γ_eff overshoot (%), {label}:")
        print(header)
        for i, ub in enumerate(u_b_vals):
            row = "  ".join(f"{over[i, j]:+6.3f}" if np.isfinite(over[i, j]) else "    NaN"
                            for j in range(len(eps_vals)))
            print(f"  {ub:5.2f}    {row}")
