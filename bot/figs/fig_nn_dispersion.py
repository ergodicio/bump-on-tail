"""Driver: NN closure vs kinetic gamma(k) at representative operating points.

Generates bot/figures/fig_nn_dispersion.png.
Requires bot/runs/naive_mlp.eqx (already trained).
"""

from bot.figures import fig_nn_dispersion, RUN_DIR

NN_PATH = RUN_DIR / "naive_mlp.eqx"

if __name__ == "__main__":
    if not NN_PATH.exists():
        print(f"Model not found: {NN_PATH}")
        raise SystemExit(1)
    fig_nn_dispersion()
