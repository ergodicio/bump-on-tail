"""Driver: |E(t)| Landau-damped cases, kinetic gamma_kin reference line.

Generates bot/figures/fig_u2_landau_dispref.png.
Requires bot/runs/nn_u2_r1.eqx and bot/runs/naive_mlp.eqx (already trained).
"""

from bot.figures_u2 import fig_u2_landau_dispref, RUN_DIR

NN_U2_PATH = RUN_DIR / "nn_u2_r1.eqx"
NN_U3_PATH = RUN_DIR / "naive_mlp.eqx"

if __name__ == "__main__":
    for p in (NN_U2_PATH, NN_U3_PATH):
        if not p.exists():
            print(f"Model not found: {p}")
            raise SystemExit(1)
    fig_u2_landau_dispref(NN_U2_PATH, NN_U3_PATH)
