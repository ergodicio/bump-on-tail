"""Evaluate the trained inference closure on BoT dispersion.

The inference closure is nonlinear in moments. To get a fluid eigenmode we
fixed-point iterate at each k:
    1. seed with kinetic omega (or last iterate)
    2. compute xi_b = (omega - k u_b) / (k v_b)
    3. compute moments U_0, U_1, U_2 at that xi_b
    4. run NN to get xi_hat_b -> alpha_eff = alpha(xi_hat_b)
    5. solve fluid eigenvalue problem with closure a = (alpha_eff, 0, 0)
    6. pick the eigenmode closest to omega; repeat until converged.

If the NN is exact, this converges to the kinetic answer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bot.closures.inference import alpha, model_xi, moment_ratios
from bot.fluid import fluid_modes
from bot.kinetic import most_unstable_kinetic
from bot.train_inference import load as load_inference


def inference_closure_a(model, xi: complex) -> np.ndarray:
    """Return the (alpha_eff, 0, 0) coefficient vector for use in fluid_system.

    Evaluated at a specific xi_b: computes moments, runs NN to get xi_hat,
    returns alpha(xi_hat) as a single-coefficient closure.
    """
    r1, r2 = moment_ratios(xi)
    xi_hat = complex(model_xi(model, r1, r2))
    a_eff = alpha(xi_hat)
    return np.array([a_eff, 0.0 + 0.0j, 0.0 + 0.0j], dtype=complex)


def inferred_mode(model, k: float, u_b: float, eps: float,
                  omega0: complex, n_iter: int = 12,
                  tol: float = 1e-7,
                  re_weight: float = 4.0) -> tuple[complex, complex, dict]:
    """Fixed-point iterate to converge on a fluid+inference eigenmode at k.

    Mode-tracking strategy: weight the distance metric to prefer matching
    Re(omega) (Langmuir branch is well-defined in real frequency even when
    gamma -> 0). Distance: re_weight * |dRe| + |dIm|.
    """
    omega = omega0
    for it in range(n_iter):
        xi_b = (omega - k * u_b) / k
        a_eff = inference_closure_a(model, xi_b)
        cands = fluid_modes(k, u_b, eps, a_eff)
        dists = re_weight * np.abs(cands.real - omega.real) + \
                np.abs(cands.imag - omega.imag)
        j = int(np.argmin(dists))
        omega_new = cands[j]
        if abs(omega_new - omega) < tol:
            omega = omega_new
            break
        omega = omega_new
    xi_b_final = (omega - k * u_b) / k
    return omega, xi_b_final, {"iter": it + 1}


def trace_inference(model, ks, u_b: float, eps: float,
                    omega_seeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each k, fixed-point solve for the fluid+inference eigenmode."""
    omegas = np.empty(len(ks), dtype=complex)
    xis = np.empty(len(ks), dtype=complex)
    for i, (k, w0) in enumerate(zip(ks, omega_seeds)):
        omega, xi_b, _ = inferred_mode(model, k, u_b, eps, w0)
        omegas[i] = omega
        xis[i] = xi_b
    return omegas, xis


def evaluate_at(u_b: float, eps: float, model=None,
                k_lo: float = 0.10, k_hi: float = 0.55, n_k: int = 60
                ) -> dict:
    """One (u_b, eps) point: compare kinetic, Padé N=3, inference closure."""
    if model is None:
        model = load_inference()

    ks = np.linspace(k_lo, k_hi, n_k)

    # kinetic via s-space eigenvalue solve
    ws_kin = np.array([most_unstable_kinetic(k, u_b, eps) for k in ks])

    # inference: fixed-point iterate from kinetic seed
    ws_inf, xis_inf = trace_inference(model, ks, u_b, eps, ws_kin)

    i_max = int(np.argmax(ws_kin.imag))
    g_kin = ws_kin[i_max].imag
    g_inf = ws_inf[i_max].imag
    return {
        "ks": ks, "ws_kin": ws_kin, "ws_inf": ws_inf, "xis_inf": xis_inf,
        "k_star": ks[i_max], "gamma_kin": g_kin, "gamma_inf": g_inf,
        "overshoot": (g_inf - g_kin) / g_kin if g_kin > 0 else np.nan,
    }


if __name__ == "__main__":
    model = load_inference()

    cases = [
        (5.0, 0.05), (5.0, 0.20), (8.0, 0.05), (3.0, 0.05),
    ]
    print(f"  {'u_b':>6}  {'eps':>6}  {'γ_kin':>10}  {'γ_inf':>10}  {'overshoot':>10}")
    print("  " + "-" * 56)
    for u_b, eps in cases:
        r = evaluate_at(u_b, eps, model=model)
        over = r["overshoot"] * 100
        print(f"  {u_b:6.2f}  {eps:6.3f}  {r['gamma_kin']:+10.5f}  "
              f"{r['gamma_inf']:+10.5f}  {over:+9.2f}%")
