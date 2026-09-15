"""Fig. 3 sweep for the homogeneous NN closures, same protocol as sweep_opt_hunana.

Reuses k*, gamma_kin and the fit window of bot/runs/sweep_opt_hunana.npz, runs
the N=3 and N=4 homogeneous closures on every cell, and writes the merged
bot/runs/sweep_opt_hunana_homog.npz with gamma_nn3h / gamma_nn4h added.
"""
import sys
from pathlib import Path
import numpy as np
from bot.closures import homog_nn as H
from bot.sweeps.sweep_opt_hunana import fit_gamma

RUN = Path(__file__).resolve().parent.parent / "runs"
d = dict(np.load(RUN / "sweep_opt_hunana.npz"))
u_b_vals, eps_vals = d["u_b_vals"], d["eps_vals"]
SUF = sys.argv[1] if len(sys.argv) > 1 else ""          # e.g. "_noimp"
cls = {"nn3h": H.HomogClosure(H.load(3, path=RUN / f"homog_nn_n3{SUF}.eqx")),
       "nn4h": H.HomogClosure(H.load(4, path=RUN / f"homog_nn_n4{SUF}.eqx"))}
for name in cls:
    d[f"gamma_{name}"] = np.full((len(u_b_vals), len(eps_vals)), np.nan)
amp, t_fit_min, n_efold, dt = 1e-3, 25.0, 5.0, 0.1
for i, u_b in enumerate(u_b_vals):
    for j, eps in enumerate(eps_vals):
        k, gp = d["k_star"][i, j], d["gamma_kin_eig"][i, j]
        if not np.isfinite(k):
            continue
        t_end = max(60.0, t_fit_min + n_efold / gp)
        t = np.linspace(0.0, t_end, int(t_end / dt) + 1)
        for name, cl in cls.items():
            y0 = np.zeros(2 + cl.N, dtype=complex); y0[1] = amp
            E = H.bot_evolve(k, u_b, eps, cl, t, y0)
            d[f"gamma_{name}"][i, j] = fit_gamma(t, E, t_fit_min=t_fit_min)
    print(f"u_b={u_b:.2f} done", flush=True)
np.savez_compressed(RUN / f"sweep_opt_hunana_homog{SUF}.npz", **d)
g = d["gamma_kin"]
print(f"\n{'closure':<10}{'median rel err':>16}{'max rel err':>14}")
for name in ["hp3", "hunana4", "opt3", "opt4", "beta2", "nn3", "nn4", "nn3h", "nn4h"]:
    rel = np.abs(d[f"gamma_{name}"] - g) / np.abs(g)
    print(f"{name:<10}{np.nanmedian(rel):>16.2e}{np.nanmax(rel):>14.2e}")

import bot.figs.fig_opt_hunana_sweep as FS
FS.PANELS[:] = ["hp3", "hunana4", "opt3", "opt4", "beta2", "nn3h", "nn4h"]
FS.LABELS.update({"nn3h": r"NN ($N=3$)", "nn4h": r"NN ($N=4$)"})
FS.make_fig(metric="relative", path=RUN / f"sweep_opt_hunana_homog{SUF}.npz",
            out_name=f"homog/fig_opt_hunana_sweep{SUF}.png")
