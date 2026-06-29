"""Driver: time-domain 2D sweep, Pade vs xi-sampled NN closure (shared log scale).

Generates bot/figures/fig_sweep_timedomain.png.
Requires bot/runs/sweep_timedomain.npz — run sweep_timedomain.py first if missing.
"""

from bot.figures import fig_sweep_timedomain, RUN_DIR

NPZ = RUN_DIR / "sweep_timedomain.npz"

if __name__ == "__main__":
    if not NPZ.exists():
        print(f"Data not found: {NPZ}")
        print("Run:  python -m bot.sweeps.sweep_timedomain")
        raise SystemExit(1)
    fig_sweep_timedomain(NPZ)
