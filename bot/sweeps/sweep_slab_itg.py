"""Time-domain (zeta_*, eta) sweep for slab ITG: HP N=3 vs EKR (beta_itg N=2).

ITG counterpart of bot/sweeps/sweep_hp_beta.py, same protocol: generic
density IC, RK45 time-stepping for both closures (HP is run with
method="rk45" so the linear closure is integrated the same way as the
nonlinear EKR closure -- see the fairness note in sweep_hp_beta.py), and
gamma_eff fit from the late-time slope of log|phi(t)| via fit_mode.

Reference: the Newton-polished root of the analytic dispersion relation
(max_growing_root of D_kinetic) -- free of discretization and fit-window
error (cf. the gamma_kin_exact discussion in sweep_opt_hunana.py).

The NN and optimized-Pade closures are deliberately absent: both are fit
to the gradient-free response Z'(xi), which bot/closures/opt_hunana.py
documents as nearly uncorrelated with ITG growth-rate accuracy (the drive
terms weight Z-moments the universal fit never sees).

Cells below the kinetic threshold eta_th(zeta_*) have no growing root and
are stored as NaN (masked in the figure).

Saves bot/runs/sweep_slab_itg.npz.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bot.slab_itg import (D_kinetic, density_ic_fluid, direct_beta_evolve,
                          eigenmode_ic_kinetic, eta_threshold, fit_mode,
                          fluid_hp_evolve, kinetic_ic_to_fluid,
                          max_growing_root)

RUN_DIR = Path(__file__).resolve().parent.parent / "runs"

ZETA_STAR_VALS = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
ETA_VALS = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0])
TAU = 1.0

T_FIT_MIN = 25.0
N_EFOLD = 5.0
T_END_MAX = 400.0
DT = 0.1
AMP = 1e-3


def run_cell(zeta_star: float, eta: float) -> dict:
    nan = {"gamma_kin": np.nan, "gamma_hp": np.nan, "gamma_ekr": np.nan,
           "gamma_hp_em": np.nan, "gamma_ekr_em": np.nan}
    root = max_growing_root(D_kinetic, zeta_star, eta, tau=TAU)
    if root is None or root.imag <= 0.0:
        return nan
    gamma_kin = root.imag

    t_end = min(T_END_MAX, max(60.0, T_FIT_MIN + N_EFOLD / gamma_kin))
    t_grid = np.linspace(0.0, t_end, int(t_end / DT) + 1)
    frac = (max(0.5, T_FIT_MIN / t_end), 1.0)

    # generic density IC (off-manifold transient robustness)
    r_hp = fluid_hp_evolve(zeta_star, eta, TAU, t_grid,
                           density_ic_fluid(AMP, N=3), method="rk45")
    r_ekr = direct_beta_evolve(zeta_star, eta, TAU, t_grid,
                               density_ic_fluid(AMP, N=2), variant="itg")

    # eigenmode-projected IC (on-manifold closure exactness)
    g0, _zmode = eigenmode_ic_kinetic(zeta_star, eta, tau=TAU, amp=AMP)
    r_hp_em = fluid_hp_evolve(zeta_star, eta, TAU, t_grid,
                              kinetic_ic_to_fluid(g0, N=3), method="rk45")
    r_ekr_em = direct_beta_evolve(zeta_star, eta, TAU, t_grid,
                                  kinetic_ic_to_fluid(g0, N=2), variant="itg")

    return {"gamma_kin": gamma_kin,
            "gamma_hp": fit_mode(r_hp["t"], r_hp["phi"], frac=frac).imag,
            "gamma_ekr": fit_mode(r_ekr["t"], r_ekr["phi"], frac=frac).imag,
            "gamma_hp_em": fit_mode(r_hp_em["t"], r_hp_em["phi"],
                                    frac=frac).imag,
            "gamma_ekr_em": fit_mode(r_ekr_em["t"], r_ekr_em["phi"],
                                     frac=frac).imag}


def main() -> None:
    shape = (len(ZETA_STAR_VALS), len(ETA_VALS))
    keys = ["gamma_kin", "gamma_hp", "gamma_ekr", "gamma_hp_em",
            "gamma_ekr_em"]
    data = {k: np.full(shape, np.nan) for k in keys}

    for i, zs in enumerate(ZETA_STAR_VALS):
        for j, eta in enumerate(ETA_VALS):
            cell = run_cell(zs, eta)
            for k in keys:
                data[k][i, j] = cell[k]
            status = ("masked (stable)" if np.isnan(cell["gamma_kin"]) else
                      f"kin={cell['gamma_kin']:.4f} "
                      f"hp={cell['gamma_hp']:.4f} ekr={cell['gamma_ekr']:.4f} "
                      f"| em: hp={cell['gamma_hp_em']:.4f} "
                      f"ekr={cell['gamma_ekr_em']:.4f}")
            print(f"zeta_*={zs:g} eta={eta:g}: {status}")

    out = RUN_DIR / "sweep_slab_itg.npz"
    np.savez(out, zeta_star_vals=ZETA_STAR_VALS, eta_vals=ETA_VALS, tau=TAU,
             eta_th=np.array([eta_threshold(z, TAU) for z in ZETA_STAR_VALS]),
             **data)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
