"""Driver: sim-trained NN closure vs kinetic gamma(k).

Generates bot/figures/fig_sim_dispersion.png.
Requires bot/runs/sim_mlp_broad.eqx (already trained).
"""

from bot.figures import fig_sim_dispersion, RUN_DIR

NN_PATH = RUN_DIR / "sim_mlp_broad.eqx"

if __name__ == "__main__":
    if not NN_PATH.exists():
        print(f"Model not found: {NN_PATH}")
        raise SystemExit(1)
    fig_sim_dispersion()
