"""Driver: two-mode (off-eigenmode) Landau IC, exact analytic reference.

Generates bot/figures/fig_u2_landau_twomode.png.
Requires bot/runs/{nn_u2_r1,naive_mlp,n4_nn,n4_nn_super}.eqx (already trained).
"""

from bot.figures_u2 import fig_u2_landau_twomode, RUN_DIR

NN_U2_PATH       = RUN_DIR / "nn_u2_r1.eqx"
NN_U3_PATH       = RUN_DIR / "naive_mlp.eqx"
NN_N4_PATH       = RUN_DIR / "n4_nn.eqx"
NN_N4_SUPER_PATH = RUN_DIR / "n4_nn_super.eqx"

if __name__ == "__main__":
    for p in (NN_U2_PATH, NN_U3_PATH, NN_N4_PATH, NN_N4_SUPER_PATH):
        if not p.exists():
            print(f"Model not found: {p}")
            raise SystemExit(1)
    fig_u2_landau_twomode(NN_U2_PATH, NN_U3_PATH, NN_N4_PATH, NN_N4_SUPER_PATH)
