"""Figure 5: bump-on-tail (u_b, eps) sweep for all seven closures.

Relative growth-rate error for HP (N=3), Hunana (N=4), learned Pade
(N=3, N=4), EKR (N=2) and NN (N=3, N=4).  Needs
bot/runs/sweep_opt_hunana.npz
(regenerate with `python -m bot.sweeps.sweep_opt_hunana`).

Writes bot/figures/fig_opt_hunana_sweep.png.
"""
from bot.figs.fig_opt_hunana_sweep import make_fig, NPZ

if __name__ == "__main__":
    if not NPZ.exists():
        raise SystemExit(f"Data not found: {NPZ}\n"
                         "Run:  python -m bot.sweeps.sweep_opt_hunana")
    make_fig("relative")
