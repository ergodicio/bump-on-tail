"""Figs. 2 and 3 (and the standing wave) with the modes-only models in the
current configuration, to bot/figures/homog/*_noimp_current.png."""
from pathlib import Path
import sys
import numpy as np
from bot.closures import homog_nn as H

RUN = H.RUN_DIR
ck = {N: RUN / f"homog_nn_n{N}_noimp_current.eqx" for N in (3, 4)}
cl = {N: H.HomogClosure(H.load(N, path=ck[N])) for N in (3, 4)}

# --- Fig. 2 via the manuscript's figure code -------------------------------
import bot.closed_loop as CL, bot.closures.n4_nn as N4, bot.closures.naive_nn as NV, bot.figures_u2 as FU
CL.nn_n4_evolve = lambda k, u_b, eps, model, t, y0, **kw: H.bot_evolve(k, u_b, eps, cl[4], t, y0)
CL.naive_evolve = lambda k, u_b, eps, model, t, y0, **kw: H.bot_evolve(k, u_b, eps, cl[3], t, y0)
N4.load_super = lambda *a, **k: None; NV.load = lambda *a, **k: None
FU.FIG_DIR = Path("bot/figures/homog/noimp_current"); FU.FIG_DIR.mkdir(parents=True, exist_ok=True)
FU.fig_u2_landau_twomode_single()
from bot.closed_loop import fluid_eigenmode_ic_u3, fluid_eigenmode_ic_n4
k, u_b, eps, amp = 0.40, 1.5, 0.05, 1e-3; w1, w2 = 1.205 - 0.119j, 0.717 - 0.106j
t = np.linspace(0, 60, 1201); E_ref = amp * (np.exp(-1j*w1*t) + np.exp(-1j*w2*t))
for N, ic in ((3, fluid_eigenmode_ic_u3), (4, fluid_eigenmode_ic_n4)):
    E = H.bot_evolve(k, u_b, eps, cl[N], t, ic(k,u_b,w1,amp) + ic(k,u_b,w2,amp))
    print(f"Fig. 2 max |E-E_ref|/amp, N={N}: {np.max(np.abs(E-E_ref)/amp):.2e}")

# --- standing wave ----------------------------------------------------------
import bot.figs.fig_standing_wave as F
F.HOMOG_CKPT = ck; F.HOMOG_LABEL = r"NN homog., modes only"
F.SAVE_PNG = "bot/figures/homog/noimp_current/fig_standing_wave.png"
F.main()

# --- Fig. 3 sweep -----------------------------------------------------------
from bot.sweeps.sweep_opt_hunana import fit_gamma
d = dict(np.load(RUN / "sweep_opt_hunana.npz"))
for N in (3, 4):
    gam = np.full_like(d["gamma_kin"], np.nan)
    for i, ub in enumerate(d["u_b_vals"]):
        for j, ep in enumerate(d["eps_vals"]):
            kk, gp = d["k_star"][i, j], d["gamma_kin_eig"][i, j]
            if not np.isfinite(kk): continue
            te = max(60.0, 25.0 + 5.0 / gp); tt = np.linspace(0, te, int(te / 0.1) + 1)
            y0 = np.zeros(2 + N, dtype=complex); y0[1] = amp
            gam[i, j] = fit_gamma(tt, H.bot_evolve(kk, ub, ep, cl[N], tt, y0))
    d[f"gamma_nn{N}h"] = gam
    rel = np.abs(gam - d["gamma_kin"]) / np.abs(d["gamma_kin"])
    print(f"sweep N={N}: median {np.nanmedian(rel):.2e}   max {np.nanmax(rel):.2e}")
np.savez_compressed(RUN / "sweep_opt_hunana_homog_noimp_current.npz", **d)
import bot.figs.fig_opt_hunana_sweep as FS
FS.PANELS[:] = ["hp3", "hunana4", "opt3", "opt4", "beta2", "nn3h", "nn4h"]
FS.LABELS.update({"nn3h": r"NN ($N=3$)", "nn4h": r"NN ($N=4$)"})
FS.make_fig(metric="relative", path=RUN / "sweep_opt_hunana_homog_noimp_current.npz",
            out_name="homog/noimp_current/fig_opt_hunana_sweep.png")
