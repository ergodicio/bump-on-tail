"""Driver: time-domain 2D sweep, sim-trained NN closure overshoot.

Generates bot/figures/fig_sweep_timedomain_sim.png.
Requires bot/runs/sweep_timedomain.npz — run sweep_timedomain.py first if missing.
"""

from bot.figures import fig_sweep_timedomain_sim, RUN_DIR

NPZ = RUN_DIR / "sweep_timedomain.npz"

if __name__ == "__main__":
    if not NPZ.exists():
        print(f"Data not found: {NPZ}")
        print("Run:  python -m bot.sweeps.sweep_timedomain")
        raise SystemExit(1)
    fig_sweep_timedomain_sim(NPZ)
