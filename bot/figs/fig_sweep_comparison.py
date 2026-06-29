"""Driver: log10|overshoot| heatmaps, Pade N=3 vs xi-sampled NN closure.

Generates bot/figures/fig_sweep_comparison.png.
Requires bot/runs/sweep_pade_N3.npz and bot/runs/sweep_timedomain.npz.
"""

from bot.figures import fig_sweep_comparison, RUN_DIR

PADE_NPZ = RUN_DIR / "sweep_pade_N3.npz"
TD_NPZ   = RUN_DIR / "sweep_timedomain.npz"

if __name__ == "__main__":
    for p, cmd in [(PADE_NPZ, "python -m bot.sweep"),
                   (TD_NPZ, "python -m bot.sweeps.sweep_timedomain")]:
        if not p.exists():
            print(f"Data not found: {p}")
            print(f"Run:  {cmd}")
            raise SystemExit(1)
    fig_sweep_comparison(PADE_NPZ, TD_NPZ)
