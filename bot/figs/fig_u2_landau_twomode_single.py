"""Driver: single-panel version of fig_u2_landau_twomode.png (u_b=1.0, k=0.50 case).

Generates bot/figures/fig_u2_landau_twomode_single.png.
Requires bot/runs/n4_nn_super.eqx (already trained).
"""

from bot.figures_u2 import fig_u2_landau_twomode_single, RUN_DIR

NN_N4_SUPER_PATH = RUN_DIR / "n4_nn_super.eqx"

if __name__ == "__main__":
    if not NN_N4_SUPER_PATH.exists():
        print(f"Model not found: {NN_N4_SUPER_PATH}")
        raise SystemExit(1)
    fig_u2_landau_twomode_single(NN_N4_SUPER_PATH)
