"""Two-mode mixing IC sweep in the GROWING (beam-unstable) regime.

Same structure as sweep_twomode.py but uses u_b=5.0, k=0.30 where the
kinetic dispersion has:
  mode 1: growing beam mode  (Im(ω) > 0)
  mode 2: damped upper branch (Im(ω) < 0, Re(ω) > 1.5)

Mixing these two modes tests how closures handle a state that is NOT a
pure eigenmode of the kinetic system: Direct β sees a time-varying ξ̂_b
as the two modes beat against each other.

Two error metrics are computed per cell:
  err_*       : full complex-waveform ||E_fluid - E_ref||_2 / ||E_ref||_2.
                Sensitive to BOTH growth-rate and real-frequency (phase)
                mismatch; a closure whose fluid eigenvalue has accurate
                Im(omega) but a few % error in Re(omega) can still show a
                large RMSE here, since the phase error d(Re_omega)*t grows
                linearly over the integration window.
  err_*_gamma : growth-rate-only relative error, |gamma_fluid - gamma_ref|
                / |gamma_ref|, from the late-time log-amplitude slope of
                each signal. Ignores phase entirely; isolates whether the
                closure tracks the correct *instability rate*.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import root as scipy_root
from scipy.stats import linregress

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


U_B = 5.0
K   = 0.30
AMP = 1e-3


def _find_two_modes(k: float, u_b: float, eps: float) -> list[complex]:
    """Find growing beam mode + damped upper branch; return [growing, damped].

    Seeds a grid of (Re, Im) guesses and Newton-refines each accepted root.
    """
    def res(x):
        D = kinetic_dispersion(x[0] + 1j * x[1], k, u_b, eps)
        return [D.real, D.imag]

    found: list[complex] = []
    for seed_re in np.linspace(0.3, 3.5, 25):
        for seed_im in np.linspace(-0.8, 0.8, 25):
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
                if w.real < 0.2:
                    continue
                xi_b = (w - k * u_b) / k
                if abs(xi_b.imag) > 1.4:
                    continue
                if xi_b.real < -3.0 or xi_b.real > 3.0:
                    continue
                if all(abs(w - wf) > 0.02 for wf in found):
                    found.append(w)
            except Exception:
                pass

    found.sort(key=lambda w: -w.imag)
    return found[:2]


def _ref_E(w: float, amp: float, omega1: complex, omega2: complex,
           t: np.ndarray) -> np.ndarray:
    return (1.0 - w) * amp * np.exp(-1j * omega1 * t) + \
               w  * amp * np.exp(-1j * omega2 * t)


def rel_rmse(E_fluid: np.ndarray, E_ref: np.ndarray) -> float:
    rms_ref = np.sqrt(np.mean(np.abs(E_ref) ** 2))
    if rms_ref < 1e-30:
        return np.nan
    return float(np.sqrt(np.mean(np.abs(E_fluid - E_ref) ** 2)) / rms_ref)


def fit_growth_rate(t_grid: np.ndarray, E: np.ndarray, frac: float = 0.75) -> float:
    """Late-time slope of log|E(t)|, fit over the last (1-frac) of t_grid."""
    n = len(t_grid)
    nh = int(frac * n)
    mag = np.abs(E[nh:])
    if not np.all(np.isfinite(mag)) or np.any(mag <= 0):
        return np.nan
    return float(linregress(t_grid[nh:], np.log(mag)).slope)


def rel_gamma_err(g_fluid: float, g_ref: float) -> float:
    if not np.isfinite(g_fluid) or abs(g_ref) < 1e-12:
        return np.nan
    return float(abs(g_fluid - g_ref) / abs(g_ref))


def abs_gamma_err(g_fluid: float, g_ref: float) -> float:
    """Un-normalized |gamma_fluid - gamma_ref|.

    Unlike rel_gamma_err, this stays finite and meaningful even when
    gamma_ref is near zero (e.g. the w->1 damped-upper-branch edge, where
    the tiny |gamma_ref| denominator makes the relative metric blow up even
    for a modest absolute rate error).
    """
    if not np.isfinite(g_fluid) or not np.isfinite(g_ref):
        return np.nan
    return float(abs(g_fluid - g_ref))


def run_cell(eps: float, w: float,
             t_end: float = 40.0, n_t: int = 801,
             amp: float = AMP) -> "dict | None":
    """Run one (eps, w) cell; shorter t_end because growing modes diverge fast."""
    _ensure_model()
    modes = _find_two_modes(K, U_B, eps)
    if len(modes) < 2:
        return None

    omega1, omega2 = modes[0], modes[1]   # growing, damped
    t_grid = np.linspace(0.0, t_end, n_t)

    E_ref = _ref_E(w, amp, omega1, omega2, t_grid)

    ic_u2 = (1.0 - w) * fluid_eigenmode_ic_u2(K, U_B, omega1, amp) \
            +       w  * fluid_eigenmode_ic_u2(K, U_B, omega2, amp)
    ic_u3 = (1.0 - w) * fluid_eigenmode_ic_u3(K, U_B, omega1, amp) \
            +       w  * fluid_eigenmode_ic_u3(K, U_B, omega2, amp)
    ic_n4 = (1.0 - w) * fluid_eigenmode_ic_n4(K, U_B, omega1, amp) \
            +       w  * fluid_eigenmode_ic_n4(K, U_B, omega2, amp)

    a_hp = pade_coefficients(3)
    g_ref = fit_growth_rate(t_grid, E_ref)

    def _safe(E_fn, *args, **kwargs):
        """Return (rel_rmse, rel_gamma_err, abs_gamma_err) or (nan, nan, nan) on failure."""
        try:
            E = E_fn(*args, **kwargs)
            if len(E) != n_t:
                return np.nan, np.nan, np.nan
            g_fluid = fit_growth_rate(t_grid, E)
            return (rel_rmse(E, E_ref), rel_gamma_err(g_fluid, g_ref),
                    abs_gamma_err(g_fluid, g_ref))
        except Exception:
            return np.nan, np.nan, np.nan

    err_dir, err_dir_gamma, err_dir_gamma_abs = _safe(
        direct_u2_evolve, K, U_B, eps, t_grid, ic_u2)
    err_hp,  err_hp_gamma,  err_hp_gamma_abs  = _safe(
        pade_evolve, K, U_B, eps, a_hp, t_grid, ic_u3, method="rk45")
    err_nn,  err_nn_gamma,  err_nn_gamma_abs  = _safe(
        nn_n4_evolve, K, U_B, eps, _NN_MODEL, t_grid, ic_n4)

    return {"err_dir": err_dir, "err_hp": err_hp, "err_nn": err_nn,
            "err_dir_gamma": err_dir_gamma, "err_hp_gamma": err_hp_gamma,
            "err_nn_gamma": err_nn_gamma,
            "err_dir_gamma_abs": err_dir_gamma_abs,
            "err_hp_gamma_abs": err_hp_gamma_abs,
            "err_nn_gamma_abs": err_nn_gamma_abs,
            "omega1": omega1, "omega2": omega2}


def run_grid(eps_vals: np.ndarray, w_vals: np.ndarray) -> dict:
    n_eps, n_w = len(eps_vals), len(w_vals)
    shape = (n_eps, n_w)

    err_dir = np.full(shape, np.nan)
    err_hp  = np.full(shape, np.nan)
    err_nn  = np.full(shape, np.nan)
    err_dir_gamma = np.full(shape, np.nan)
    err_hp_gamma  = np.full(shape, np.nan)
    err_nn_gamma  = np.full(shape, np.nan)
    err_dir_gamma_abs = np.full(shape, np.nan)
    err_hp_gamma_abs  = np.full(shape, np.nan)
    err_nn_gamma_abs  = np.full(shape, np.nan)

    for i, eps in enumerate(eps_vals):
        for j, w in enumerate(w_vals):
            try:
                r = run_cell(eps, w)
                if r is None:
                    print(f"  [eps={eps:.3f}, w={w:.2f}]  < 2 modes — skip")
                    continue
                err_dir[i, j] = r["err_dir"]
                err_hp[i, j]  = r["err_hp"]
                err_nn[i, j]  = r["err_nn"]
                err_dir_gamma[i, j] = r["err_dir_gamma"]
                err_hp_gamma[i, j]  = r["err_hp_gamma"]
                err_nn_gamma[i, j]  = r["err_nn_gamma"]
                err_dir_gamma_abs[i, j] = r["err_dir_gamma_abs"]
                err_hp_gamma_abs[i, j]  = r["err_hp_gamma_abs"]
                err_nn_gamma_abs[i, j]  = r["err_nn_gamma_abs"]
            except Exception as e:
                print(f"  [eps={eps:.3f}, w={w:.2f}]  FAIL: {e}")
        print(f"eps={eps:.3f}  done")

    return {"err_dir": err_dir, "err_hp": err_hp, "err_nn": err_nn,
            "err_dir_gamma": err_dir_gamma, "err_hp_gamma": err_hp_gamma,
            "err_nn_gamma": err_nn_gamma,
            "err_dir_gamma_abs": err_dir_gamma_abs,
            "err_hp_gamma_abs": err_hp_gamma_abs,
            "err_nn_gamma_abs": err_nn_gamma_abs,
            "eps_vals": eps_vals, "w_vals": w_vals}


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

    print(f"Two-mode growing sweep: {len(eps_vals)} x {len(w_vals)} grid  "
          f"(u_b={U_B}, k={K})")

    print("\nModes at each eps:")
    for eps in eps_vals:
        modes = _find_two_modes(K, U_B, eps)
        if len(modes) >= 2:
            xi1 = (modes[0] - K*U_B)/K
            xi2 = (modes[1] - K*U_B)/K
            print(f"  eps={eps:.2f}: growing={modes[0]:.3f} (xi={xi1:.3f})  "
                  f"damped={modes[1]:.3f} (xi={xi2:.3f})")

    print()
    results = run_grid(eps_vals, w_vals)

    out = Path(__file__).resolve().parent.parent / "runs" / "sweep_twomode_growing.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out, **results)
    print(f"\nsaved {out}")

    for label, key in [("Direct β", "err_dir"), ("N=4 NN super", "err_nn"), ("HP N=3", "err_hp"),
                       ("Direct β (gamma-only)", "err_dir_gamma"),
                       ("N=4 NN super (gamma-only)", "err_nn_gamma"),
                       ("HP N=3 (gamma-only)", "err_hp_gamma")]:
        errs = results[key]
        log_errs = np.where(errs > 0, np.log10(errs), np.nan)
        print(f"\n{label}  log₁₀(rel error), rows=eps, cols=w:")
        header = "  eps\\w  " + "  ".join(f"{wv:.1f}" for wv in w_vals)
        print(header)
        for i, ep in enumerate(eps_vals):
            row = "  ".join(
                f"{log_errs[i, j]:+5.1f}" if np.isfinite(log_errs[i, j]) else "  NaN"
                for j in range(len(w_vals))
            )
            print(f"  {ep:.3f}  {row}")
