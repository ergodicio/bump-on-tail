"""Driver: HP (Pade N=3) vs Direct beta time-domain 2D sweep.

Generates bot/figures/fig_hp_beta_sweep.png.
Requires bot/runs/sweep_hp_beta.npz — run sweep_hp_beta.py first if missing.
"""

from bot.figures_u2 import fig_hp_beta_sweep, RUN_DIR

NPZ = RUN_DIR / "sweep_hp_beta.npz"

if __name__ == "__main__":
    if not NPZ.exists():
        print(f"Data not found: {NPZ}")
        print("Run:  python -m bot.sweeps.sweep_hp_beta")
        raise SystemExit(1)
    fig_hp_beta_sweep(NPZ)
