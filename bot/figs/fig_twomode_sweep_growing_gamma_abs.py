"""Driver: two-mode mixing IC sweep — growing beam regime, ABSOLUTE growth-rate error.

Generates bot/figures/fig_twomode_sweep_growing_gamma_abs.png.
Requires bot/runs/sweep_twomode_growing.npz — run python -m bot.sweeps.sweep_twomode_growing first if missing.
"""

from pathlib import Path
from bot.figures_u2 import fig_twomode_sweep_growing_gamma_abs

NPZ = Path(__file__).resolve().parent.parent / "runs" / "sweep_twomode_growing.npz"

if __name__ == "__main__":
    if not NPZ.exists():
        print(f"Data not found: {NPZ}")
        print("Run:  python -m bot.sweeps.sweep_twomode_growing")
        raise SystemExit(1)
    fig_twomode_sweep_growing_gamma_abs(NPZ)
