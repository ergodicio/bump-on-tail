"""Driver: U2/NN closures vs kinetic Landau damping, time domain.

Generates bot/figures/fig_u2_landau_timedomain.png.
Requires bot/runs/nn_u2_r1.eqx (already trained).
"""

from bot.figures_u2 import fig_u2_landau_timedomain, RUN_DIR

NN_PATH = RUN_DIR / "nn_u2_r1.eqx"

if __name__ == "__main__":
    if not NN_PATH.exists():
        print(f"Model not found: {NN_PATH}")
        raise SystemExit(1)
    fig_u2_landau_timedomain(NN_PATH)
