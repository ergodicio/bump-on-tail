"""Driver: two-mode mixing IC sweep — Landau regime, damping-rate-only error.

Generates bot/figures/fig_twomode_sweep_gamma.png.
Requires bot/runs/sweep_twomode.npz — run python -m bot.sweeps.sweep_twomode first if missing.
"""

from pathlib import Path
from bot.figures_u2 import fig_twomode_sweep_gamma

NPZ = Path(__file__).resolve().parent.parent / "runs" / "sweep_twomode.npz"

if __name__ == "__main__":
    if not NPZ.exists():
        print(f"Data not found: {NPZ}")
        print("Run:  python -m bot.sweeps.sweep_twomode")
        raise SystemExit(1)
    fig_twomode_sweep_gamma(NPZ)
