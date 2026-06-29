"""Driver: |E(t)| at canonical point, sim-trained NN closure.

Generates bot/figures/fig_time_domain_sim.png.
Requires bot/runs/time_domain.npz — run python -m bot.closed_loop first if missing.
"""

from bot.figures import fig_time_domain_sim, RUN_DIR

NPZ = RUN_DIR / "time_domain.npz"

if __name__ == "__main__":
    if not NPZ.exists():
        print(f"Data not found: {NPZ}")
        print("Run:  python -m bot.closed_loop")
        raise SystemExit(1)
    fig_time_domain_sim(NPZ)
