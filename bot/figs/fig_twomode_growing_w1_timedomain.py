"""Driver: time-domain E(t) traces at the w=1 edge of the growing-regime two-mode sweep.

Generates bot/figures/fig_twomode_growing_w1_timedomain.png.
No npz dependency -- computes kinetic modes and integrates all three closures live.
"""

from bot.figures_u2 import fig_twomode_growing_w1_timedomain

if __name__ == "__main__":
    fig_twomode_growing_w1_timedomain()
