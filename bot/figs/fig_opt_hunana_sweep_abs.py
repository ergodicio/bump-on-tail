"""Driver: (u_b, eps) growth-rate sweep for the four Hunana-family closures,
ABSOLUTE rate error.

Generates bot/figures/fig_opt_hunana_sweep_abs.png -- the direct four-closure
counterpart of fig_hp_beta_sweep_abs.png (same grid, same generic delta-E IC,
same gamma_eff fit, same colour scale), so the two figures can be read side by
side.

Requires bot/runs/sweep_opt_hunana.npz -- run
`python -m bot.sweeps.sweep_opt_hunana` first if missing.
"""
from bot.figs.fig_opt_hunana_sweep import NPZ, make_fig

if __name__ == "__main__":
    if not NPZ.exists():
        print(f"Data not found: {NPZ}")
        print("Run:  python -m bot.sweeps.sweep_opt_hunana")
        raise SystemExit(1)
    make_fig("absolute")
