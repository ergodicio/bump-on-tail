"""Driver: U2 closure time-domain 2D sweep.

Generates bot/figures/fig_u2_sweep.png.
Requires bot/runs/sweep_u2_timedomain.npz — run sweep_u2_timedomain.py first if missing.
"""

from bot.figures_u2 import fig_u2_sweep, RUN_DIR

NPZ = RUN_DIR / "sweep_u2_timedomain.npz"

if __name__ == "__main__":
    if not NPZ.exists():
        print(f"Data not found: {NPZ}")
        print("Run:  python -m bot.sweeps.sweep_u2_timedomain")
        raise SystemExit(1)
    fig_u2_sweep(NPZ)
