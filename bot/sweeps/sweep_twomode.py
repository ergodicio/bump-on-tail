"""Two-mode mixing IC sweep: Direct β (N=2) vs N=4 NN super vs HP (N=3).

Fixed parameters: u_b=1.0, k=0.5 — two Landau-damped modes exist for eps >= 0.02.
Sweep axes: eps ∈ [0.02, 0.05, 0.10, 0.15, 0.20], w ∈ linspace(0, 1, 11).

At each (eps, w):
  1. Find two kinetic modes ω₁, ω₂ (sorted least-damped first).
  2. Build mixed IC: y0 = (1-w)·eigenmode(ω₁, amp) + w·eigenmode(ω₂, amp).
  3. Integrate Direct β (N=2), N=4 NN super, HP Padé N=3.
  4. Reference: E_ref(t) = (1-w)·amp·e^{-iω₁t} + w·amp·e^{-iω₂t}.
  5. Fit γ_eff from late-time log|E(t)| for closure and for E_ref.
  6. Error: log₁₀|γ_fluid/γ_ref − 1|.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import root as scipy_root

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
from bot.kinetic import find_kinetic_mode, kinetic_dispersion


# Fixed physical parameters
U_B = 1.0
K   = 0.5
AMP = 1e-3


def _find_two_modes(k: float, u_b: float, eps: float) -> list[complex]:
    """Find two Landau-damped kinetic modes; return sorted least-damped first.

    Returns a list of length 0, 1, or 2.
    """
    def res(x):
        D = kinetic_dispersion(x[0] + 1j * x[1], k, u_b, eps)
        return [D.real, D.imag]

    found: list[complex] = []
    for seed_re in np.linspace(0.4, 1.8, 15):
        for seed_im in np.linspace(-0.05, -0.7, 15):
            try:
                sol = scipy_root(res, [seed_re, seed_im], method="hybr",
                                 options={"maxfev": 400})
                if not sol.success:
                    continue
                w = sol.x[0] + 1j * sol.x[1]
                if abs(kinetic_dispersion(w, k, u_b, eps)) > 1e-4:
                    continue
                w = find_kinetic_mode(k, u_b, eps, w)
                if abs(kinetic_dispersion(w, k, u_b, eps)) > 1e-10:
                    continue
                if w.imag > -0.01:
                    continue
                if w.real < 0.3 or w.real > 2.5:
                    continue
                xi_b = (w - k * u_b) / k
                if xi_b.imag >= 0 or xi_b.imag < -1.0:
                    continue
                if all(abs(w - wf) > 0.01 for wf in found):
                    found.append(w)
            except Exception:
                pass

    # Sort least-damped (Im closest to 0) first
    found.sort(key=lambda w: -w.imag)
    return found[:2]


def _ref_E(w: float, amp: float, omega1: complex, omega2: complex,
           t: np.ndarray) -> np.ndarray:
    """Analytic two-mode reference: E_ref(t) = (1-w)·amp·e^{-iω₁t} + w·amp·e^{-iω₂t}."""
    return (1.0 - w) * amp * np.exp(-1j * omega1 * t) + w * amp * np.exp(-1j * omega2 * t)


def rel_rmse(E_fluid: np.ndarray, E_ref: np.ndarray) -> float:
    """Relative RMSE of E_fluid vs analytic E_ref over the full time window."""
    rms_ref = np.sqrt(np.mean(np.abs(E_ref) ** 2))
    if rms_ref < 1e-30:
        return np.nan
    return float(np.sqrt(np.mean(np.abs(E_fluid - E_ref) ** 2)) / rms_ref)


def run_cell(eps: float, w: float,
             t_end: float = 80.0, n_t: int = 801,
             amp: float = AMP) -> "dict | None":
    """Run one (eps, w) cell; return dict of E-field relative errors or None if < 2 modes."""
    _ensure_model()
    modes = _find_two_modes(K, U_B, eps)
    if len(modes) < 2:
        return None

    omega1, omega2 = modes[0], modes[1]
    t_grid = np.linspace(0.0, t_end, n_t)

    E_ref = _ref_E(w, amp, omega1, omega2, t_grid)

    # Mixed ICs
    ic_u2 = (1.0 - w) * fluid_eigenmode_ic_u2(K, U_B, omega1, amp) \
            +       w  * fluid_eigenmode_ic_u2(K, U_B, omega2, amp)
    ic_u3 = (1.0 - w) * fluid_eigenmode_ic_u3(K, U_B, omega1, amp) \
            +       w  * fluid_eigenmode_ic_u3(K, U_B, omega2, amp)
    ic_n4 = (1.0 - w) * fluid_eigenmode_ic_n4(K, U_B, omega1, amp) \
            +       w  * fluid_eigenmode_ic_n4(K, U_B, omega2, amp)

    a_hp = pade_coefficients(3)

    def _safe_rmse(E_fn, *args):
        try:
            E = E_fn(*args)
            if len(E) != n_t:
                return np.nan
            return rel_rmse(E, E_ref)
        except Exception:
            return np.nan

    err_dir = _safe_rmse(direct_u2_evolve, K, U_B, eps, t_grid, ic_u2)
    err_hp  = _safe_rmse(pade_evolve,      K, U_B, eps, a_hp, t_grid, ic_u3)
    err_nn  = _safe_rmse(nn_n4_evolve,     K, U_B, eps, _NN_MODEL, t_grid, ic_n4)

    return {"err_dir": err_dir, "err_hp": err_hp, "err_nn": err_nn,
            "omega1": omega1, "omega2": omega2}


def run_grid(eps_vals: np.ndarray, w_vals: np.ndarray) -> dict:
    """Full 2D sweep over (eps, w)."""
    n_eps, n_w = len(eps_vals), len(w_vals)
    shape = (n_eps, n_w)

    err_dir = np.full(shape, np.nan)
    err_hp  = np.full(shape, np.nan)
    err_nn  = np.full(shape, np.nan)

    for i, eps in enumerate(eps_vals):
        for j, w in enumerate(w_vals):
            try:
                r = run_cell(eps, w)
                if r is None:
                    print(f"  [eps={eps:.3f}, w={w:.2f}]  only 1 mode — skip")
                    continue
                err_dir[i, j] = r["err_dir"]
                err_hp[i, j]  = r["err_hp"]
                err_nn[i, j]  = r["err_nn"]
            except Exception as e:
                print(f"  [eps={eps:.3f}, w={w:.2f}]  FAIL: {e}")
        print(f"eps={eps:.3f}  done")

    return {"err_dir": err_dir, "err_hp": err_hp, "err_nn": err_nn,
            "eps_vals": eps_vals, "w_vals": w_vals}


# Module-level model load (lazy, so importing the file doesn't load JAX models)
_NN_MODEL = None


def _ensure_model() -> None:
    global _NN_MODEL
    if _NN_MODEL is None:
        print("Loading N=4 NN super model...")
        _NN_MODEL = load_super()
        print("  model loaded.")


if __name__ == "__main__":
    _ensure_model()

    eps_vals = np.array([0.02, 0.05, 0.10, 0.15, 0.20])
    w_vals   = np.linspace(0.0, 1.0, 11)

    print(f"Two-mode sweep: {len(eps_vals)} x {len(w_vals)} grid  "
          f"(u_b={U_B}, k={K})")
    results = run_grid(eps_vals, w_vals)

    out = Path(__file__).resolve().parent.parent / "runs" / "sweep_twomode.npz"
    out.parent.mkdir(exist_ok=True)
    # npz can't store complex — store real/imag of omega separately
    save_dict = {k: v for k, v in results.items()
                 if isinstance(v, np.ndarray) and v.dtype != complex}
    np.savez_compressed(out, **save_dict)
    print(f"\nsaved {out}")

    for label, key in [("Direct β", "err_dir"), ("N=4 NN super", "err_nn"), ("HP N=3", "err_hp")]:
        errs = results[key]
        log_errs = np.where(errs > 0, np.log10(errs), np.nan)
        print(f"\n{label}  log₁₀(rel RMSE), rows=eps, cols=w:")
        header = "  eps\\w  " + "  ".join(f"{wv:.1f}" for wv in w_vals)
        print(header)
        for i, ep in enumerate(eps_vals):
            row = "  ".join(
                f"{log_errs[i, j]:+5.1f}" if np.isfinite(log_errs[i, j]) else "  NaN"
                for j in range(len(w_vals))
            )
            print(f"  {ep:.3f}  {row}")
