"""Figure 4: two-mode Landau-damped superpositions, seven closures.

Compares the linear HP (N=3), Hunana (N=4) and learned Pade (N=3, N=4)
closures with the nonlinear EKR (N=2) and NN (N=3, N=4) closures, from a
superposition initial condition.

Needs the trained NN checkpoints in bot/runs/ (n4_nn_super.eqx, naive_mlp.eqx).

Writes bot/figures/fig_u2_landau_twomode_single.png.
"""
from bot.figures_u2 import fig_u2_landau_twomode_single, RUN_DIR

NN_PATH = RUN_DIR / "n4_nn_super.eqx"

if __name__ == "__main__":
    if not NN_PATH.exists():
        raise SystemExit(f"Model not found: {NN_PATH}")
    fig_u2_landau_twomode_single(NN_PATH)
