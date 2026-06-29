"""Driver: ITG marginal-stability threshold test.

Generates bot/figures/fig_itg_threshold.png.
Requires bot/runs/nn_u2_r1.eqx (already trained, loaded internally by itg_test.py).
"""

from bot.itg_test import fig_itg_threshold, report

if __name__ == "__main__":
    results, eta_values = fig_itg_threshold(
        zeta_star=1.0, tau=1.0,
        eta_range=(0.1, 5.0), n_eta=50,
    )
    report(results, eta_values)
