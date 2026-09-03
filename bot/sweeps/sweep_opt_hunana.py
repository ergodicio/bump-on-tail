"""Time-domain 2D (u_b, eps) sweep for the four Hunana-family linear closures.

Same protocol as bot/sweeps/sweep_hp_beta.py -- identical grid, identical
generic delta-E initial condition, identical gamma_eff fit -- so the output is
directly comparable to fig_hp_beta_sweep{,_abs}.png.  Six closures, arranged so
that the N=4 row isolates the effect of the MATCHING STRATEGY at fixed order:

    hp3       pade_coefficients(3)        HP 1990: 2 conditions at xi=0
                                          (incl. Im Z'(0), the Landau one) plus
                                          the 1/xi^4 asymptotic term
    opt3      opt_coefficients(3)         same family, fit on the NN rectangle

    hunana4   two_point_coefficients(4,2) the direct N=4 analogue of HP's split
                                          -- 2 at xi=0, the rest asymptotic
    hunana4b  two_point_coefficients(4,3) 3 at xi=0, 1 asymptotic
    pade4     pade_coefficients(4)        pure Taylor-at-xi=0 (4 conditions).
                                          NOTE this is NOT an HP-style closure;
                                          pade_coefficients only reproduces HP
                                          at N=3.
    opt4      opt_coefficients(4)         same family, fit on the NN rectangle

For N >= 4 the two leading asymptotic orders (1/xi^2 and (3/2)/xi^4) hold for
ANY coefficients -- the moment hierarchy forces them -- so hunana4 spends its
extra freedom on 1/xi^6 and 1/xi^8, i.e. on |xi| far beyond where any mode
sits, while pade4 gets those two orders free AND four conditions at the origin.

Per (u_b, eps) cell:
  1. k* = argmax of the kinetic gamma(k) scan (bot.sweep.kinetic_peak).
  2. Integrate the kinetic system (N_v=96) and all four closed fluid systems
     from the generic delta-E IC (only E != 0 at t=0), so every fluid mode --
     including each closure's own spurious ones -- is excited.
  3. Fit gamma_eff from the late-time slope of log|E(t)|.

All four closures are linear and are run with pade_evolve(method="rk45"), the
same way sweep_hp_beta.py runs HP, rather than being solved exactly by
eigendecomposition.

Two views are stored per closure:
  gamma_*      time-domain fit, the quantity plotted (comparable to the HP /
               direct-beta sweep, and sensitive to spurious-mode contamination
               of the fit window -- see the CAVEAT in sweep_hp_beta.py).
  gamma_*_eig  most-unstable eigenvalue of the closed fluid system at k*
               (bot.fluid.fluid_modes).  This is what the time-domain fit
               converges to as the window lengthens, so a large
               gamma_* vs gamma_*_eig gap flags an unconverged cell rather
               than a bad closure.

and three kinetic references:
  gamma_kin        time-domain fit of the N_v=96 discretized system (protocol
                   reference -- what the HP / direct-beta sweep uses)
  gamma_kin_eig    most-unstable eigenvalue of that discretized system
  gamma_kin_exact  Newton-polished root of the analytic dispersion relation at
                   k* (bot.kinetic.find_kinetic_mode), seeded from the discrete
                   eigenvalue.  This is the only reference free of both
                   discretization and fit-window error, and it matters at weak
                   growth: at u_b=2.5, eps=0.005 the fitted and discrete
                   kinetic rates differ by 13%, which would otherwise be
                   charged to the closures.  `converged` records
                   |gamma_kin - gamma_kin_exact| / gamma_kin_exact so those
                   cells can be masked.

Run:  python -m bot.sweeps.sweep_opt_hunana          (coarse grid, as above)
      python -m bot.sweeps.sweep_opt_hunana --fine   (refined low-u_b/low-eps
                                                      box -> sweep_opt_hunana_fine.npz)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import linregress

from bot.closed_loop import direct_u2_evolve, kinetic_evolve, pade_evolve
from bot.closures.opt_hunana import opt_coefficients
from bot.closures.pade import pade_coefficients, two_point_coefficients
from bot.fluid import fluid_modes
from bot.kinetic import find_kinetic_mode, kinetic_dispersion
from bot.sweep import kinetic_peak


# Linear closures U_N = sum_i a_i U_i -- these have closed-system eigenvalues,
# so both the time-domain and the eigenvalue view are available for them.
CLOSURES = {
    "hp3":      pade_coefficients(3),
    "opt3":     opt_coefficients(3),
    "hunana4":  two_point_coefficients(4, 2),
    "hunana4b": two_point_coefficients(4, 3),
    "pade4":    pade_coefficients(4),
    "opt4":     opt_coefficients(4),
}

# Nonlinear closures: no fixed system matrix, so time-domain only.  Each entry
# is (state dimension excluding (u, E), evolve callable returning E(t)).
#
# Adding or removing anything in CLOSURES / NONLINEAR changes what gets
# COMPUTED, so it needs a re-run of this script.  Which of the computed
# closures get plotted, and under what titles, is a presentation choice and
# lives in bot/figs/fig_opt_hunana_sweep.py (PANELS, LABELS) -- editing those
# only needs the figure driver re-run, not this.
NONLINEAR = {
    "nn3":   (3, lambda k, u_b, eps, t, y0: _nn3_evolve(k, u_b, eps, t, y0)),
    "nn4":   (4, lambda k, u_b, eps, t, y0: _nn4_evolve(k, u_b, eps, t, y0)),
    "beta2": (2, lambda k, u_b, eps, t, y0: direct_u2_evolve(k, u_b, eps, t, y0)),
}

_NN3_MODEL = None
_NN4_MODEL = None


def _nn3_evolve(k, u_b, eps, t, y0):
    """N=3 NN closure: the naive NN(r_1, r_2) -> alpha (runs/naive_mlp.eqx).

    The naive model, not the inference one (bot/closures/inference.py, which
    routes NN -> xi_hat -> exact Faddeeva): naive is the direct analogue of the
    N=4 NN used here (both regress the closure ratio itself), and the
    single-xi inversion the inference architecture commits to is a liability
    off the eigenmode manifold -- which is exactly where the generic delta-E IC
    starts.
    """
    global _NN3_MODEL
    if _NN3_MODEL is None:
        from bot.closures.naive_nn import load as load_naive
        _NN3_MODEL = load_naive()
    from bot.closed_loop import naive_evolve
    return naive_evolve(k, u_b, eps, _NN3_MODEL, t, y0)


def _nn4_evolve(k, u_b, eps, t, y0):
    """N=4 NN closure, superposition-trained checkpoint (runs/n4_nn_super.eqx).

    The super model is the right one for this sweep: the generic delta-E IC is
    far off the eigenmode manifold, which is exactly the regime the
    superposition training data covers (same choice fig_twomode_sweep makes).
    """
    global _NN4_MODEL
    if _NN4_MODEL is None:
        from bot.closures.n4_nn import load_super
        _NN4_MODEL = load_super()
    from bot.closed_loop import nn_n4_evolve
    return nn_n4_evolve(k, u_b, eps, _NN4_MODEL, t, y0)


def fit_gamma(t: np.ndarray, E: np.ndarray, t_fit_min: float = 25.0) -> float:
    """Fit effective growth rate from late-time log|E| slope."""
    mask = t >= t_fit_min
    logE = np.log(np.maximum(np.abs(E[mask]), 1e-30))
    slope, *_ = linregress(t[mask], logE)
    return float(slope)


def _refine_k_star(ks: np.ndarray, ws_kin: np.ndarray, i_max: int,
                   u_b: float, eps: float) -> tuple[float, complex]:
    """Maximize the ANALYTIC gamma(k) inside the bracket around the discrete peak.

    kinetic_peak scans k on a fixed grid of spacing (1.20-0.10)/79 = 0.0139, so
    k* is quantized.  gamma(k) is flat near its peak but the closures' errors
    are not, so the quantization prints as spurious striping in u_b across a
    refined (u_b, eps) scan -- e.g. u_b=5.50 and 5.75 both land on k*=0.1975,
    and HP's error at u_b=5.75 jumps to 9.7% purely from being evaluated off
    its own peak.  Refining k* removes that artifact.
    """
    lo = ks[max(i_max - 1, 0)]
    hi = ks[min(i_max + 1, len(ks) - 1)]
    seed = ws_kin[i_max]

    def neg_gamma(k: float) -> float:
        w = find_kinetic_mode(float(k), u_b, eps, seed)
        if abs(kinetic_dispersion(w, float(k), u_b, eps)) > 1e-9:
            return 0.0
        return -w.imag

    sol = minimize_scalar(neg_gamma, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-6})
    k_star = float(sol.x) if sol.success else float(ks[i_max])
    return k_star, find_kinetic_mode(k_star, u_b, eps, seed)


def run_cell(u_b: float, eps: float,
             t_end_min: float = 60.0, t_fit_min: float = 25.0,
             n_efold: float = 5.0, dt: float = 0.1,
             amp: float = 1e-3, refine_k: bool = False) -> dict:
    """Time-domain gamma_eff for kinetic + the four closures at peak k.

    t_end is chosen per cell so the fit window spans at least n_efold
    e-foldings of the kinetic rate, for the reason documented in
    sweep_hp_beta.run_cell (gamma_kin varies ~9x across this grid).

    refine_k=False reproduces sweep_hp_beta.py's protocol exactly (k* taken
    from kinetic_peak's discrete scan), which is what keeps the coarse grid
    comparable to fig_hp_beta_sweep.  refine_k=True sub-grid-refines k* and is
    used for the --fine box, where the quantization would otherwise dominate
    the structure (see _refine_k_star).
    """
    ks, ws_kin, i_max = kinetic_peak(u_b, eps)
    k = ks[i_max]
    gamma_kin_peak = ws_kin[i_max].imag
    if refine_k:
        k, w_peak = _refine_k_star(ks, ws_kin, i_max, u_b, eps)
        gamma_kin_peak = w_peak.imag

    t_end = max(t_end_min, t_fit_min + n_efold / gamma_kin_peak)
    t_grid = np.linspace(0.0, t_end, int(t_end / dt) + 1)

    y0_kin = np.zeros(2 + 96, dtype=complex)
    y0_kin[1] = amp
    E_kin = kinetic_evolve(k, u_b, eps, t_grid, y0_kin)
    g_kin = fit_gamma(t_grid, E_kin, t_fit_min=t_fit_min)

    # Analytic root at k*, seeded from the discrete eigenvalue: the reference
    # free of both discretization and fit-window error.
    w_exact = find_kinetic_mode(k, u_b, eps, ws_kin[i_max])
    if abs(kinetic_dispersion(w_exact, k, u_b, eps)) > 1e-9:
        w_exact = complex(np.nan, np.nan)
    g_exact = float(w_exact.imag)

    out = {"u_b": u_b, "eps": eps, "k_star": k,
           "gamma_kin": g_kin, "gamma_kin_eig": gamma_kin_peak,
           "gamma_kin_exact": g_exact,
           "converged": abs(g_kin - g_exact) / abs(g_exact)}
    for name, a in CLOSURES.items():
        y0 = np.zeros(2 + len(a), dtype=complex)
        y0[1] = amp
        E = pade_evolve(k, u_b, eps, a, t_grid, y0, method="rk45")
        out[f"gamma_{name}"] = fit_gamma(t_grid, E, t_fit_min=t_fit_min)
        out[f"gamma_{name}_eig"] = float(fluid_modes(k, u_b, eps, a).imag.max())

    for name, (n_mom, evolve) in NONLINEAR.items():
        y0 = np.zeros(2 + n_mom, dtype=complex)
        y0[1] = amp
        E = evolve(k, u_b, eps, t_grid, y0)
        out[f"gamma_{name}"] = fit_gamma(t_grid, E, t_fit_min=t_fit_min)
    return out


def run_grid(u_b_vals: np.ndarray, eps_vals: np.ndarray,
             refine_k: bool = False) -> dict:
    """Full 2D grid sweep."""
    shape = (len(u_b_vals), len(eps_vals))
    keys = ["gamma_kin", "gamma_kin_eig", "gamma_kin_exact", "converged",
            "k_star"]
    for name in CLOSURES:
        keys += [f"gamma_{name}", f"gamma_{name}_eig"]
    keys += [f"gamma_{name}" for name in NONLINEAR]
    arrs = {k: np.full(shape, np.nan) for k in keys}

    for i, u_b in enumerate(u_b_vals):
        for j, eps in enumerate(eps_vals):
            try:
                r = run_cell(u_b, eps, refine_k=refine_k)
                for key in keys:
                    arrs[key][i, j] = r[key]
            except Exception as e:
                print(f"  [u_b={u_b:.2f}, eps={eps:.3f}]  FAIL: {e}")
        print(f"u_b={u_b:.2f}  done", flush=True)

    arrs["u_b_vals"] = u_b_vals
    arrs["eps_vals"] = eps_vals
    return arrs


if __name__ == "__main__":
    import sys

    if "--fine" in sys.argv:
        # Refined box over the low-u_b / low-eps corner, where the coarse grid
        # shows HP's worst cell (8.3% at u_b=4, eps=0.02 -- the operating point
        # of fig_bot_vs_itg_closures) sitting on an unresolved ridge that the
        # coarse grid truncates at its own edge.  eps is log-spaced.
        u_b_vals = np.linspace(2.5, 6.0, 15)
        eps_vals = np.geomspace(0.005, 0.06, 12)
        fname = "sweep_opt_hunana_fine.npz"
        refine_k = True
    else:
        u_b_vals = np.linspace(3.0, 10.0, 8)
        eps_vals = np.array([0.01, 0.02, 0.05, 0.10, 0.15, 0.20])
        fname = "sweep_opt_hunana.npz"
        refine_k = False        # keep sweep_hp_beta.py's protocol exactly

    print(f"opt-Hunana sweep: {len(u_b_vals)} x {len(eps_vals)} grid, "
          f"{len(CLOSURES)} closures, refine_k={refine_k} -> {fname}")
    results = run_grid(u_b_vals, eps_vals, refine_k=refine_k)

    out = Path(__file__).resolve().parent.parent / "runs" / fname
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out, **results)
    print(f"\nsaved {out}")

    bad = results["converged"] > 0.02
    print(f"\ncells with an unconverged kinetic reference "
          f"(|gamma_kin - gamma_kin_exact|/gamma_kin_exact > 2%): "
          f"{int(np.nansum(bad))}/{results['converged'].size}")

    g_kin = results["gamma_kin"]
    for name in list(CLOSURES) + list(NONLINEAR):
        err = np.abs(results[f"gamma_{name}"] - g_kin)
        rel = err / np.abs(g_kin)
        print(f"\n{name}: |dgamma| median={np.nanmedian(err):.3e} "
              f"max={np.nanmax(err):.3e}   "
              f"relative median={np.nanmedian(rel):.2%} max={np.nanmax(rel):.2%}")
        print("  |dgamma| rows=u_b, cols=eps:")
        print("    u_b\\eps  " + "  ".join(f"{e:8.3f}" for e in eps_vals))
        for i, ub in enumerate(u_b_vals):
            row = "  ".join(f"{err[i, j]:8.2e}" if np.isfinite(err[i, j])
                            else "     NaN" for j in range(len(eps_vals)))
            print(f"    {ub:5.2f}    {row}")
