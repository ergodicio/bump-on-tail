"""Driver: HP Padé N=3 time-series sweep across |ξ_b|.

Generates bot/figures/fig_pade_xib_sweep.png.
No pre-computed data required — time series are run on the fly.
"""

from bot.figures_u2 import fig_pade_xib_sweep

if __name__ == "__main__":
    fig_pade_xib_sweep()
