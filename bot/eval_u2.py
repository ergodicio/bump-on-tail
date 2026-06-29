"""Dispersion evaluation and parameter sweep for N=2 (U_2) moment closures.

Compares three closures on the same N=2 state (u, E, U_0, U_1):

  Padé N=2    -- linear closure U_2 = a_0 U_0 + a_1 U_1, coefficients chosen
                 to match the Taylor expansion of -Z'(xi) to O(xi^2).
                 Solved by eigenvalues of the 4x4 linear system.

  Direct beta -- exact per-mode closure U_2 = U_0 * beta(U_1/U_0),
                 where beta(xi) = xi^2 - 1/Z'(xi).  Because r_1=U_1/U_0=xi_b
                 exactly on eigenmode, this closure is exact at the dispersion
                 level and should recover kinetic gamma_max to numerical
                 precision.  Dispersion is found by fixed-point iteration
                 (same strategy as eval_inference.py for the N=3 case).

Key contrast with the N=3 analysis:
  * There is no zero-parameter bilinear identity at N=2 (M_0=1 != 0 breaks it).
  * beta requires Z' explicitly; the exact closure is not algebraically simpler
    than the Pade approximant -- it is just the exact answer vs an approximation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bot.closures.u2 import beta as u2_beta
from bot.closures.pade import pade_coefficients
from bot.fluid import fluid_modes, trace_fluid
from bot.kinetic import most_unstable_kinetic


# ---------------------------------------------------------------------------
# Fixed-point dispersion for the direct-beta N=2 closure
# ---------------------------------------------------------------------------

def _direct_u2_a(xi: complex) -> np.ndarray:
    """Length-2 closure vector [beta(xi), 0] for fluid_system at N=2.

    The closure U_2 = beta(xi_b)*U_0 maps to the coefficient form
    U_2 = a_0*U_0 + a_1*U_1 with a_0=beta(xi_b), a_1=0.
    """
    return np.array([u2_beta(xi), 0.0 + 0.0j], dtype=complex)


def inferred_mode_u2(k: float, u_b: float, eps: float,
                     omega0: complex,
                     n_iter: int = 20,
                     tol: float = 1e-9,
                     re_weight: float = 4.0) -> tuple[complex, complex]:
    """Fixed-point iterate to find the direct-beta N=2 closure eigenmode at k.

    At convergence xi_b = (omega - k*u_b)/k is consistent with the closure
    beta(xi_b) used in the 4x4 fluid eigenvalue problem, so the result is
    the exact kinetic eigenvalue (to numerical precision of Z').

    Returns (omega, xi_b).
    """
    omega = omega0
    for _ in range(n_iter):
        xi_b = (omega - k * u_b) / k
        a_eff = _direct_u2_a(xi_b)
        cands = fluid_modes(k, u_b, eps, a_eff)   # 4 eigenvalues
        dists = re_weight * np.abs(cands.real - omega.real) + \
                np.abs(cands.imag - omega.imag)
        j = int(np.argmin(dists))
        omega_new = cands[j]
        if abs(omega_new - omega) < tol:
            omega = omega_new
            break
        omega = omega_new
    xi_b_final = (omega - k * u_b) / k
    return omega, xi_b_final


def trace_direct_u2(ks: np.ndarray, u_b: float, eps: float,
                    omega_seeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trace the direct-beta N=2 eigenmode over a k array via continuation."""
    omegas = np.empty(len(ks), dtype=complex)
    xis = np.empty(len(ks), dtype=complex)
    for i, (k, w0) in enumerate(zip(ks, omega_seeds)):
        omegas[i], xis[i] = inferred_mode_u2(k, u_b, eps, w0)
    return omegas, xis


# ---------------------------------------------------------------------------
# Single operating-point evaluation
# ---------------------------------------------------------------------------

def evaluate_u2(u_b: float, eps: float,
                k_lo: float = 0.10, k_hi: float = 0.55, n_k: int = 60,
                ) -> dict:
    """Compare kinetic, Padé N=2, and direct-beta N=2 closures at one (u_b, eps).

    Returns dict with keys:
        ks, ws_kin, ws_pade2, ws_direct,
        k_star, gamma_kin, gamma_pade2, gamma_direct,
        overshoot_pade2, overshoot_direct
    """
    ks = np.linspace(k_lo, k_hi, n_k)
    a_pade2 = pade_coefficients(2)

    # kinetic truth
    ws_kin = np.array([most_unstable_kinetic(k, u_b, eps) for k in ks])

    # Padé N=2: linear system, eigenvalue trace
    ws_pade2 = trace_fluid(ks, u_b, eps, a_pade2, ws_kin)

    # direct beta: fixed-point from kinetic seeds
    ws_direct, xis_direct = trace_direct_u2(ks, u_b, eps, ws_kin)

    i_max = int(np.argmax(ws_kin.imag))
    g_kin = ws_kin[i_max].imag

    def _over(g):
        return (g - g_kin) / g_kin if g_kin > 0 else np.nan

    return {
        "ks": ks,
        "ws_kin": ws_kin,
        "ws_pade2": ws_pade2,
        "ws_direct": ws_direct,
        "xis_direct": xis_direct,
        "k_star": ks[i_max],
        "gamma_kin": g_kin,
        "gamma_pade2": ws_pade2[i_max].imag,
        "gamma_direct": ws_direct[i_max].imag,
        "overshoot_pade2": _over(ws_pade2[i_max].imag),
        "overshoot_direct": _over(ws_direct[i_max].imag),
    }


# ---------------------------------------------------------------------------
# 2D parameter sweep
# ---------------------------------------------------------------------------

def _measure_u2_overshoot(u_b: float, eps: float) -> dict:
    """Peak-gamma overshoot for Padé N=2 and direct-beta at one grid point."""
    from bot.sweep import kinetic_peak

    a_pade2 = pade_coefficients(2)
    ks, ws_kin, i_max = kinetic_peak(u_b, eps)
    ws_pade2 = trace_fluid(ks, u_b, eps, a_pade2, ws_kin)
    ws_direct, _ = trace_direct_u2(ks, u_b, eps, ws_kin)

    g_kin = ws_kin[i_max].imag

    def _over(g):
        return (g - g_kin) / g_kin if g_kin > 0 else np.nan

    return {
        "gamma_kin": g_kin,
        "gamma_pade2": ws_pade2[i_max].imag,
        "gamma_direct": ws_direct[i_max].imag,
        "overshoot_pade2": _over(ws_pade2[i_max].imag),
        "overshoot_direct": _over(ws_direct[i_max].imag),
        "k_star": ks[i_max],
    }


def sweep_u2(u_b_vals: np.ndarray, eps_vals: np.ndarray) -> dict:
    """2D sweep: overshoot of Padé N=2 and direct-beta N=2 vs kinetic."""
    n_ub, n_eps = len(u_b_vals), len(eps_vals)
    shape = (n_ub, n_eps)

    gamma_kin    = np.full(shape, np.nan)
    gamma_pade2  = np.full(shape, np.nan)
    gamma_direct = np.full(shape, np.nan)
    over_pade2   = np.full(shape, np.nan)
    over_direct  = np.full(shape, np.nan)
    k_star       = np.full(shape, np.nan)

    for i, u_b in enumerate(u_b_vals):
        for j, eps in enumerate(eps_vals):
            try:
                r = _measure_u2_overshoot(u_b, eps)
                gamma_kin[i, j]    = r["gamma_kin"]
                gamma_pade2[i, j]  = r["gamma_pade2"]
                gamma_direct[i, j] = r["gamma_direct"]
                over_pade2[i, j]   = r["overshoot_pade2"]
                over_direct[i, j]  = r["overshoot_direct"]
                k_star[i, j]       = r["k_star"]
            except Exception as e:
                print(f"  [u_b={u_b:.2f}, eps={eps:.3f}]  FAIL: {e}")
        print(f"u_b={u_b:.2f}  done")

    return {
        "u_b_vals": u_b_vals, "eps_vals": eps_vals,
        "gamma_kin": gamma_kin,
        "gamma_pade2": gamma_pade2,
        "gamma_direct": gamma_direct,
        "overshoot_pade2": over_pade2,
        "overshoot_direct": over_direct,
        "k_star": k_star,
    }


# ---------------------------------------------------------------------------
# Command-line driver
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    cases = [
        (5.0, 0.05), (5.0, 0.20), (8.0, 0.05), (3.0, 0.05),
    ]

    print(f"  {'u_b':>6}  {'eps':>6}  {'γ_kin':>10}  "
          f"{'γ_pade2':>10}  {'over_P2':>9}  "
          f"{'γ_direct':>10}  {'over_D':>9}")
    print("  " + "-" * 78)
    for u_b, eps in cases:
        r = evaluate_u2(u_b, eps)
        print(f"  {u_b:6.2f}  {eps:6.3f}  {r['gamma_kin']:+10.5f}  "
              f"{r['gamma_pade2']:+10.5f}  {r['overshoot_pade2']*100:+8.2f}%  "
              f"{r['gamma_direct']:+10.5f}  {r['overshoot_direct']*100:+8.2f}%")

    if "--sweep" in sys.argv:
        u_b_vals = np.linspace(3.0, 10.0, 8)
        eps_vals = np.array([0.01, 0.02, 0.05, 0.10, 0.15, 0.20])
        print(f"\nRunning 2D sweep ({len(u_b_vals)}x{len(eps_vals)})...")
        results = sweep_u2(u_b_vals, eps_vals)

        out = Path(__file__).resolve().parent / "runs" / "sweep_u2.npz"
        out.parent.mkdir(exist_ok=True)
        np.savez_compressed(out, **results)
        print(f"saved {out}")

        over_p = results["overshoot_pade2"] * 100
        over_d = results["overshoot_direct"] * 100
        print("\nPadé N=2 overshoot (%), rows=u_b, cols=eps:")
        header = "  u_b\\eps  " + "  ".join(f"{e:6.3f}" for e in eps_vals)
        print(header)
        for i, ub in enumerate(u_b_vals):
            row = "  ".join(
                f"{over_p[i, j]:+6.1f}" if np.isfinite(over_p[i, j]) else "    NaN"
                for j in range(len(eps_vals))
            )
            print(f"  {ub:5.2f}    {row}")

        print("\nDirect beta overshoot (%), rows=u_b, cols=eps:")
        print(header)
        for i, ub in enumerate(u_b_vals):
            row = "  ".join(
                f"{over_d[i, j]:+6.2f}" if np.isfinite(over_d[i, j]) else "    NaN"
                for j in range(len(eps_vals))
            )
            print(f"  {ub:5.2f}    {row}")
