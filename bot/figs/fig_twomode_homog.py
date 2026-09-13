"""Fig. 2 (fig_u2_landau_twomode_single) with the homogeneous NN closures.

Re-runs the manuscript's own figure code, swapping only the two NN evolve
functions (naive N=3, super N=4) for the homogeneous closures of
bot/closures/homog_nn.py.  Output: bot/figures/homog/fig_u2_landau_twomode_single.png
"""
from pathlib import Path
import bot.closed_loop as CL
import bot.closures.n4_nn as N4
import bot.closures.naive_nn as NV
import bot.figures_u2 as FU
from bot.closures import homog_nn as H

cl3, cl4 = H.HomogClosure(H.load(3)), H.HomogClosure(H.load(4))
CL.nn_n4_evolve = lambda k, u_b, eps, model, t, y0, **kw: H.bot_evolve(k, u_b, eps, cl4, t, y0)
CL.naive_evolve = lambda k, u_b, eps, model, t, y0, **kw: H.bot_evolve(k, u_b, eps, cl3, t, y0)
N4.load_super = lambda *a, **k: None
NV.load = lambda *a, **k: None
FU.FIG_DIR = Path("bot/figures/homog"); FU.FIG_DIR.mkdir(exist_ok=True)
FU.fig_u2_landau_twomode_single()
