"""Parametric sweep: 2D grid over (u_b/v_b, eps) for BoT γ_max overshoot.

For each (u_b, eps):
  1. Scan k, find kinetic γ_max and the k* where it peaks.
  2. Build the closed fluid system at that (u_b, eps) with the Padé closure.
  3. Trace the fluid Langmuir-like mode in k and measure its γ_max.
  4. Record overshoot = (γ_fluid - γ_kin) / γ_kin   evaluated at k*.

Output: sweep_results.npz with arrays of shape (n_ub, n_eps).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bot.closures.pade import pade_coefficients
from bot.fluid import trace_fluid
from bot.kinetic import most_unstable_kinetic


def kinetic_peak(u_b: float, eps: float,
                 k_lo: float = 0.10, k_hi: float = 1.20, n_k: int = 80,
                 N_v: int = 96, V: float = 6.0
                 ) -> tuple[np.ndarray, np.ndarray, int]:
    """Scan k via s-space eigenvalue solver; return (ks, omega(k), peak idx).

    For each k pick the most-unstable Langmuir-like mode in Re(omega) in
    [0.3, 3.0]. Robust across the (u_b, eps) grid.
    """
    ks = np.linspace(k_lo, k_hi, n_k)
    ws = np.empty(n_k, dtype=complex)
    for i, k in enumerate(ks):
        ws[i] = most_unstable_kinetic(k, u_b, eps, N_v=N_v, V=V)
    i_max = int(np.argmax(ws.imag))
    return ks, ws, i_max


def measure_overshoot(u_b: float, eps: float, N: int = 3) -> dict:
    """Return dict with γ_kin, γ_fluid, k*, overshoot at the kinetic peak."""
    a = pade_coefficients(N)
    ks, ws_kin, i_max = kinetic_peak(u_b, eps)
    ws_fluid = trace_fluid(ks, u_b, eps, a, ws_kin)
    g_kin = ws_kin[i_max].imag
    g_flu = ws_fluid[i_max].imag
    return {
        "u_b": u_b, "eps": eps, "N": N,
        "k_star": ks[i_max],
        "gamma_kin": g_kin,
        "gamma_fluid": g_flu,
        "overshoot": (g_flu - g_kin) / g_kin if g_kin > 0 else np.nan,
        "ks": ks, "ws_kin": ws_kin, "ws_fluid": ws_fluid,
    }


def run_grid(u_b_vals: np.ndarray, eps_vals: np.ndarray,
             N: int = 3) -> dict:
    """2D sweep over (u_b, eps)."""
    n_ub, n_eps = len(u_b_vals), len(eps_vals)
    gamma_kin = np.full((n_ub, n_eps), np.nan)
    gamma_fluid = np.full((n_ub, n_eps), np.nan)
    k_star = np.full((n_ub, n_eps), np.nan)
    overshoot = np.full((n_ub, n_eps), np.nan)

    for i, u_b in enumerate(u_b_vals):
        for j, eps in enumerate(eps_vals):
            try:
                r = measure_overshoot(u_b, eps, N=N)
                gamma_kin[i, j] = r["gamma_kin"]
                gamma_fluid[i, j] = r["gamma_fluid"]
                k_star[i, j] = r["k_star"]
                overshoot[i, j] = r["overshoot"]
            except Exception as e:
                print(f"  [u_b={u_b:.2f}, eps={eps:.3f}]  FAIL: {e}")
                continue
        print(f"u_b={u_b:.2f}  done  ({n_eps} eps values)")

    return {
        "u_b_vals": u_b_vals, "eps_vals": eps_vals, "N": N,
        "gamma_kin": gamma_kin, "gamma_fluid": gamma_fluid,
        "k_star": k_star, "overshoot": overshoot,
    }


if __name__ == "__main__":
    u_b_vals = np.linspace(3.0, 10.0, 8)
    eps_vals = np.array([0.01, 0.02, 0.05, 0.10, 0.15, 0.20])

    print(f"sweep grid: {len(u_b_vals)} u_b values x {len(eps_vals)} eps values")
    print(f"u_b in {u_b_vals}")
    print(f"eps in {eps_vals}\n")

    results = run_grid(u_b_vals, eps_vals, N=3)

    out = Path(__file__).resolve().parent / "runs" / "sweep_pade_N3.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out, **results)
    print(f"\nsaved {out}")

    # quick summary
    g_kin = results["gamma_kin"]
    g_flu = results["gamma_fluid"]
    over = results["overshoot"] * 100
    print("\nγ_max overshoot (%) of Padé N=3 vs kinetic, rows=u_b, cols=eps:")
    header = "  u_b\\eps  " + "  ".join(f"{e:6.3f}" for e in eps_vals)
    print(header)
    for i, ub in enumerate(u_b_vals):
        row = "  ".join(f"{over[i, j]:+6.1f}" if np.isfinite(over[i, j]) else "    NaN"
                        for j in range(len(eps_vals)))
        print(f"  {ub:5.2f}    {row}")
