"""Time-domain 2D sweep comparing HP (Padé N=3) vs Direct β (N=2) closures.

For each (u_b, eps) grid point:
  1. Find k* where kinetic γ_max occurs.
  2. Integrate kinetic (N_v=96), HP Padé N=3, and direct-β N=2 with a
     generic δE IC (only E≠0 at t=0).
  3. Fit γ_eff from the late-time slope of log|E(t)|.

The HP closure (Hammett-Perkins 1990) uses pade_coefficients(3) applied to
the N=3 (u, E, U_0, U_1, U_2) state with U_3 closed.  HP is linear and
would normally be solved exactly via eigendecomposition, but is run here
with pade_evolve(..., method="rk45") so both fluid closures (HP and direct
beta) are genuinely time-stepped the same way -- direct beta has no choice
(nonlinear), so this keeps the comparison apples-to-apples rather than
giving HP a free exact solve.

The Direct-β closure uses the exact kinetic ratio β(ξ_b) = U_2/U_0 at the
effective ξ_b = U_1/U_0, giving exact dispersion on the eigenmode manifold.

CAVEAT (found while cross-checking against direct eigenvalues elsewhere):
fit_gamma's t_fit_min=25 window is not always long enough for HP's 5
eigenmodes to fully separate from a generic delta-E IC -- e.g. at
u_b=5, eps=0.05 this sweep's fitted overshoot reads ~31%, but the true
eigenvalue overshoot (bot.fluid.fluid_modes at k*) is only ~5%. Treat
gamma_hp/overshoot_hp here as approximate; for a trustworthy per-point
number, compare against the direct eigenvalue instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import linregress

from bot.closed_loop import direct_u2_evolve, kinetic_evolve, pade_evolve
from bot.closures.pade import pade_coefficients
from bot.sweep import kinetic_peak


def fit_gamma(t: np.ndarray, E: np.ndarray, t_fit_min: float = 25.0) -> float:
    """Fit effective growth rate from late-time log|E| slope."""
    mask = t >= t_fit_min
    logE = np.log(np.maximum(np.abs(E[mask]), 1e-30))
    slope, *_ = linregress(t[mask], logE)
    return float(slope)


def run_cell(u_b: float, eps: float,
             t_end_min: float = 60.0, t_fit_min: float = 25.0,
             n_efold: float = 5.0, dt: float = 0.1,
             amp: float = 1e-3) -> dict:
    """Time-domain γ_eff for kinetic / HP Padé N=3 / Direct β at peak k.

    t_end is chosen per-cell so the fit window [t_fit_min, t_end] always
    spans at least n_efold e-foldings of the kinetic growth rate (gamma_kin
    varies ~7x across the (u_b, eps) grid, e.g. 0.035 at u_b=3,eps=0.01 vs
    0.29 at u_b=5,eps=0.20; a fixed t_end=60 gave the fast-growing cells
    ~10 e-foldings to separate the growing mode from the generic-delta-E
    IC's transient, but the slow-growing cells only ~1-2, biasing their
    fitted gamma_eff -- e.g. -3.4% at u_b=3,eps=0.01 with a fixed window,
    vs <0.1% once the window is long enough).  t_end_min keeps the floor at
    the old default for cells where 5 e-foldings fit well inside it.
    """
    ks, ws_kin, i_max = kinetic_peak(u_b, eps)
    k = ks[i_max]
    gamma_kin_peak = ws_kin[i_max].imag

    t_end = max(t_end_min, t_fit_min + n_efold / gamma_kin_peak)
    n_t = int(t_end / dt) + 1
    t_grid = np.linspace(0.0, t_end, n_t)

    # Generic δE IC: only E≠0
    y0_n2 = np.array([0.0, amp, 0.0, 0.0],          dtype=complex)  # N=2
    y0_n3 = np.array([0.0, amp, 0.0, 0.0, 0.0],      dtype=complex)  # N=3
    y0_kin = np.zeros(2 + 96, dtype=complex); y0_kin[1] = amp

    a_hp = pade_coefficients(3)

    E_kin  = kinetic_evolve(k, u_b, eps, t_grid, y0_kin)
    E_hp   = pade_evolve(k, u_b, eps, a_hp, t_grid, y0_n3, method="rk45")
    E_dir  = direct_u2_evolve(k, u_b, eps, t_grid, y0_n2)

    g_kin = fit_gamma(t_grid, E_kin, t_fit_min=t_fit_min)
    g_hp  = fit_gamma(t_grid, E_hp,  t_fit_min=t_fit_min)
    g_dir = fit_gamma(t_grid, E_dir, t_fit_min=t_fit_min)

    def _over(g):
        return (g - g_kin) / g_kin if g_kin > 0 else np.nan

    return {
        "u_b": u_b, "eps": eps, "k_star": k,
        "gamma_kin":    g_kin,
        "gamma_hp":     g_hp,
        "gamma_direct": g_dir,
        "overshoot_hp":     _over(g_hp),
        "overshoot_direct": _over(g_dir),
    }


def run_grid(u_b_vals: np.ndarray, eps_vals: np.ndarray) -> dict:
    """Full 2D grid sweep."""
    n_ub, n_eps = len(u_b_vals), len(eps_vals)
    shape = (n_ub, n_eps)

    keys = ["gamma_kin", "gamma_hp", "gamma_direct",
            "overshoot_hp", "overshoot_direct", "k_star"]
    arrs = {k: np.full(shape, np.nan) for k in keys}

    for i, u_b in enumerate(u_b_vals):
        for j, eps in enumerate(eps_vals):
            try:
                r = run_cell(u_b, eps)
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

    print(f"HP vs Direct-β sweep: {len(u_b_vals)} x {len(eps_vals)} grid")
    results = run_grid(u_b_vals, eps_vals)

    out = Path(__file__).resolve().parent.parent / "runs" / "sweep_hp_beta.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out, **results)
    print(f"\nsaved {out}")

    for label, key in [("HP Padé N=3", "overshoot_hp"),
                       ("Direct β",    "overshoot_direct")]:
        over = results[key] * 100
        print(f"\n{label} overshoot (%), rows=u_b, cols=eps:")
        header = "  u_b\\eps  " + "  ".join(f"{e:6.3f}" for e in eps_vals)
        print(header)
        for i, ub in enumerate(u_b_vals):
            row = "  ".join(
                f"{over[i, j]:+6.2f}" if np.isfinite(over[i, j]) else "    NaN"
                for j in range(len(eps_vals))
            )
            print(f"  {ub:5.2f}    {row}")
