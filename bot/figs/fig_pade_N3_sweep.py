"""Driver: gamma_kin / gamma_fluid / overshoot heatmaps for Pade N=3.

Generates bot/figures/fig_pade_N3_sweep.png.
Requires bot/runs/sweep_pade_N3.npz — run sweep.py first if missing.
"""

from bot.figures import fig_sweep_heatmaps, RUN_DIR

NPZ = RUN_DIR / "sweep_pade_N3.npz"

if __name__ == "__main__":
    if not NPZ.exists():
        print(f"Data not found: {NPZ}")
        print("Run:  python -m bot.sweep")
        raise SystemExit(1)
    fig_sweep_heatmaps(NPZ)
