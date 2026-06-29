"""Driver: two-mode growing sweep — NN and HP Padé only (no Direct β).

Generates bot/figures/fig_twomode_sweep_growing_nn_hp.png.
Requires bot/runs/sweep_twomode_growing.npz — run python -m bot.sweeps.sweep_twomode_growing first if missing.
"""

from pathlib import Path
from bot.figures_u2 import fig_twomode_sweep_growing_nn_hp

NPZ = Path(__file__).resolve().parent.parent / "runs" / "sweep_twomode_growing.npz"

if __name__ == "__main__":
    if not NPZ.exists():
        print(f"Data not found: {NPZ}")
        print("Run:  python -m bot.sweeps.sweep_twomode_growing")
        raise SystemExit(1)
    fig_twomode_sweep_growing_nn_hp(NPZ)
