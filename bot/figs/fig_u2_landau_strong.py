"""Driver: strongly-damped Landau cases, eigenmode vs generic delta-E IC.

Generates bot/figures/fig_u2_landau_strong.png.
Requires bot/runs/nn_u2_r1.eqx and bot/runs/naive_mlp.eqx (already trained).
"""

from bot.figures_u2 import fig_u2_landau_strong, RUN_DIR

NN_U2_PATH = RUN_DIR / "nn_u2_r1.eqx"
NN_U3_PATH = RUN_DIR / "naive_mlp.eqx"

if __name__ == "__main__":
    for p in (NN_U2_PATH, NN_U3_PATH):
        if not p.exists():
            print(f"Model not found: {p}")
            raise SystemExit(1)
    fig_u2_landau_strong(NN_U2_PATH, NN_U3_PATH)
