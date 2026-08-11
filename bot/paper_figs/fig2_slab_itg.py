"""Figure 2: slab ITG instability at zeta_* = 1, tau = 1.

(a) |U_0i|(t) at eta = 10;  (b) growth rate vs eta with the kinetic
dispersion root, HP fluid eigenvalue, and time-domain fits.

Needs bot/runs/bot_vs_itg_closures.npz and bot/runs/slab_itg_growth.npz
(regenerate with bot/figs/fig_bot_vs_itg_closures.py and
bot/figs/fig_slab_itg_growth.py).

Writes bot/figures/fig_slab_itg_combined.png.
"""
from bot.figs.fig_slab_itg_combined import main

if __name__ == "__main__":
    main()
