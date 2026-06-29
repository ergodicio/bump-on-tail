"""Driver: single-mode eigenmode IC sweep over u_b × eps.

Generates bot/figures/fig_singlemode_sweep.png.
Requires bot/runs/sweep_singlemode.npz — run python -m bot.sweeps.sweep_singlemode first if missing.
"""

from pathlib import Path
from bot.figures_u2 import fig_singlemode_sweep

NPZ = Path(__file__).resolve().parent.parent / "runs" / "sweep_singlemode.npz"

if __name__ == "__main__":
    if not NPZ.exists():
        print(f"Data not found: {NPZ}")
        print("Run:  python -m bot.sweeps.sweep_singlemode")
        raise SystemExit(1)
    fig_singlemode_sweep(NPZ)
