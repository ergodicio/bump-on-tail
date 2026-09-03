"""Figure 3: bump-on-tail (u_b, eps) sweep, HP (N=3) vs EKR (N=2).

Relative growth-rate error |gamma_eff/gamma_kin - 1| over the parameter
space.  Needs bot/runs/sweep_hp_beta.npz
(regenerate with `python -m bot.sweeps.sweep_hp_beta`).

Writes bot/figures/fig_hp_beta_sweep.png.
"""
from bot.figures_u2 import fig_hp_beta_sweep, RUN_DIR

NPZ = RUN_DIR / "sweep_hp_beta.npz"

if __name__ == "__main__":
    if not NPZ.exists():
        raise SystemExit(f"Data not found: {NPZ}\n"
                         "Run:  python -m bot.sweeps.sweep_hp_beta")
    fig_hp_beta_sweep(NPZ)
