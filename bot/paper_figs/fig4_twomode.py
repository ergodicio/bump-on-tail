"""Figure 4: two-mode Landau-damped superpositions, seven closures.

Compares the linear HP (N=3), Hunana (N=4) and learned Pade (N=3, N=4)
closures with the nonlinear EKR (N=2) and homogeneous (equivariant) NN
(N=3, N=4) closures, from a superposition initial condition.  The NN
closures are swapped in as in bot/figs/fig_twomode_homog.py.

Needs bot/runs/homog_nn_n3.eqx and bot/runs/homog_nn_n4.eqx.

Writes bot/figures/homog/fig_u2_landau_twomode_single.png (case (b) only)
and bot/figures/homog/fig_u2_landau_twomode_2panel.png (cases (a) and (b)).
"""
from pathlib import Path

import bot.closed_loop as CL
import bot.closures.n4_nn as N4
import bot.closures.naive_nn as NV
import bot.figures_u2 as FU
from bot.closures import homog_nn as H

if __name__ == "__main__":
    cl3, cl4 = H.HomogClosure(H.load(3)), H.HomogClosure(H.load(4))
    CL.nn_n4_evolve = lambda k, u_b, eps, model, t, y0, **kw: H.bot_evolve(k, u_b, eps, cl4, t, y0)
    CL.naive_evolve = lambda k, u_b, eps, model, t, y0, **kw: H.bot_evolve(k, u_b, eps, cl3, t, y0)
    N4.load_super = lambda *a, **k: None
    NV.load = lambda *a, **k: None
    FU.FIG_DIR = Path("bot/figures/homog")
    FU.FIG_DIR.mkdir(exist_ok=True)
    FU.fig_u2_landau_twomode_single()
    FU.fig_u2_landau_twomode_2panel()
