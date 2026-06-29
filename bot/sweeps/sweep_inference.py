"""Same 2D grid sweep but with the inference closure."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bot.eval_inference import evaluate_at
from bot.train_inference import load as load_inference


def run_grid(u_b_vals: np.ndarray, eps_vals: np.ndarray) -> dict:
    model = load_inference()
    n_ub, n_eps = len(u_b_vals), len(eps_vals)
    gamma_kin = np.full((n_ub, n_eps), np.nan)
    gamma_inf = np.full((n_ub, n_eps), np.nan)
    overshoot = np.full((n_ub, n_eps), np.nan)

    for i, u_b in enumerate(u_b_vals):
        for j, eps in enumerate(eps_vals):
            try:
                r = evaluate_at(u_b, eps, model=model)
                gamma_kin[i, j] = r["gamma_kin"]
                gamma_inf[i, j] = r["gamma_inf"]
                overshoot[i, j] = r["overshoot"]
            except Exception as e:
                print(f"  [u_b={u_b:.2f}, eps={eps:.3f}]  FAIL: {e}")
        print(f"u_b={u_b:.2f}  done  ({n_eps} eps values)")

    return {
        "u_b_vals": u_b_vals, "eps_vals": eps_vals,
        "gamma_kin": gamma_kin, "gamma_inf": gamma_inf,
        "overshoot": overshoot,
    }


if __name__ == "__main__":
    u_b_vals = np.linspace(3.0, 10.0, 8)
    eps_vals = np.array([0.01, 0.02, 0.05, 0.10, 0.15, 0.20])

    results = run_grid(u_b_vals, eps_vals)

    out = Path(__file__).resolve().parent.parent / "runs" / "sweep_inference.npz"
    np.savez_compressed(out, **results)
    print(f"\nsaved {out}")

    over = results["overshoot"] * 100
    print("\nγ_max overshoot (%) of inference closure vs kinetic:")
    header = "  u_b\\eps  " + "  ".join(f"{e:6.3f}" for e in eps_vals)
    print(header)
    for i, ub in enumerate(u_b_vals):
        row = "  ".join(f"{over[i, j]:+6.3f}" if np.isfinite(over[i, j]) else "    NaN"
                        for j in range(len(eps_vals)))
        print(f"  {ub:5.2f}    {row}")
