"""Single-mode eigenmode IC sweep: Direct β / N=4 NN super / HP Padé N=3.

Grid: u_b × eps at k*(u_b, eps) (peak kinetic growth rate).

For each cell:
  1. Find k* and the most-unstable kinetic eigenmode omega.
  2. Build eigenmode IC for each closure.
  3. Integrate each system and kinetic reference for t ∈ [0, t_end].
  4. Error metric: rel RMSE  ||E_fluid - E_kin||_2 / ||E_kin||_2.

Saved arrays (rows = u_b, cols = eps):
  err_dir, err_hp, err_nn, u_b_vals, eps_vals, k_star
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bot.closed_loop import (
    direct_u2_evolve,
    fluid_eigenmode_ic_n4,
    fluid_eigenmode_ic_u2,
    fluid_eigenmode_ic_u3,
    nn_n4_evolve,
    pade_evolve,
)
from bot.closures.n4_nn import load_super
from bot.closures.pade import pade_coefficients
from bot.kinetic import most_unstable_kinetic
from bot.sweep import kinetic_peak


def rel_rmse(E_fluid: np.ndarray, E_ref: np.ndarray) -> float:
    rms_ref = np.sqrt(np.mean(np.abs(E_ref) ** 2))
    if rms_ref < 1e-30:
        return np.nan
    return float(np.sqrt(np.mean(np.abs(E_fluid - E_ref) ** 2)) / rms_ref)


def run_cell(u_b: float, eps: float, nn_model,
             amp: float = 1e-3) -> dict:
    """Run one (u_b, eps) cell at k*; return rel RMSE for each closure.

    Reference is the analytic single-mode signal  E_ref(t) = amp * exp(-i*omega*t),
    matching the convention of the two-mode sweeps (w=0 limit).
    """
    ks, _, i_max = kinetic_peak(u_b, eps)
    k = ks[i_max]

    omega = most_unstable_kinetic(k, u_b, eps)
    t_end = min(80.0, 4.0 / max(omega.imag, 0.01))
    t_grid = np.linspace(0.0, t_end, 601)

    E_ref = amp * np.exp(-1j * omega * t_grid)

    ic_u2 = fluid_eigenmode_ic_u2(k, u_b, omega, amp)
    ic_u3 = fluid_eigenmode_ic_u3(k, u_b, omega, amp)
    ic_n4 = fluid_eigenmode_ic_n4(k, u_b, omega, amp)

    a_hp = pade_coefficients(3)

    def _safe(fn, *args):
        try:
            E = fn(*args)
            if len(E) != len(t_grid):
                return np.nan
            return rel_rmse(E, E_ref)
        except Exception:
            return np.nan

    return {
        "err_dir": _safe(direct_u2_evolve, k, u_b, eps, t_grid, ic_u2),
        "err_hp":  _safe(pade_evolve,      k, u_b, eps, a_hp, t_grid, ic_u3),
        "err_nn":  _safe(nn_n4_evolve,     k, u_b, eps, nn_model, t_grid, ic_n4),
        "k_star":  k,
        "omega":   omega,
    }


def run_grid(u_b_vals: np.ndarray, eps_vals: np.ndarray) -> dict:
    nn_model = load_super()
    print("  NN model loaded.")

    n_ub, n_eps = len(u_b_vals), len(eps_vals)
    shape = (n_ub, n_eps)

    err_dir = np.full(shape, np.nan)
    err_hp  = np.full(shape, np.nan)
    err_nn  = np.full(shape, np.nan)
    k_star  = np.full(shape, np.nan)

    for i, u_b in enumerate(u_b_vals):
        for j, eps in enumerate(eps_vals):
            try:
                r = run_cell(u_b, eps, nn_model)
                err_dir[i, j] = r["err_dir"]
                err_hp[i, j]  = r["err_hp"]
                err_nn[i, j]  = r["err_nn"]
                k_star[i, j]  = r["k_star"]
            except Exception as e:
                print(f"  [u_b={u_b:.2f}, eps={eps:.3f}]  FAIL: {e}")
        print(f"u_b={u_b:.2f}  done")

    return {
        "err_dir": err_dir, "err_hp": err_hp, "err_nn": err_nn,
        "k_star": k_star,
        "u_b_vals": u_b_vals, "eps_vals": eps_vals,
    }


if __name__ == "__main__":
    u_b_vals = np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    eps_vals = np.array([0.02, 0.05, 0.10, 0.15, 0.20])

    print(f"Single-mode sweep: {len(u_b_vals)} x {len(eps_vals)} grid")
    results = run_grid(u_b_vals, eps_vals)

    out = Path(__file__).resolve().parent.parent / "runs" / "sweep_singlemode.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out, **results)
    print(f"\nsaved {out}")

    for label, key in [("Direct β", "err_dir"), ("HP Padé N=3", "err_hp"), ("N=4 NN super", "err_nn")]:
        errs = results[key]
        log_errs = np.where(errs > 0, np.log10(errs), np.nan)
        print(f"\n{label}  log₁₀(rel RMSE), rows=u_b, cols=eps:")
        header = "  u_b\\eps  " + "  ".join(f"{e:.2f}" for e in eps_vals)
        print(header)
        for i, ub in enumerate(u_b_vals):
            row = "  ".join(
                f"{log_errs[i, j]:+5.1f}" if np.isfinite(log_errs[i, j]) else "  NaN"
                for j in range(len(eps_vals))
            )
            print(f"  {ub:.1f}    {row}")
