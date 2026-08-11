# Paper figures

Driver scripts for the figures in the manuscript, one per figure, in
manuscript order.  Each script is a thin wrapper around the canonical
implementation elsewhere in the repo, so the plotting logic is not
duplicated; these files fix which figure is which.

Run from the repository root, e.g.

    PYTHONPATH=. python3 bot/paper_figs/fig1_kappa_bot.py

| Fig. | Script | Output (`bot/figures/`) | Data (`bot/runs/`) |
|------|--------|-------------------------|--------------------|
| 1 | `fig1_kappa_bot.py`    | `fig_kappa_bot.png`               | computed in-script |
| 2 | `fig2_slab_itg.py`     | `fig_slab_itg_combined.png`       | `bot_vs_itg_closures.npz`, `slab_itg_growth.npz` |
| 3 | `fig3_bot_sweep.py`    | `fig_hp_beta_sweep.png`           | `sweep_hp_beta.npz` |
| 4 | `fig4_twomode.py`      | `fig_u2_landau_twomode_single.png`| NN checkpoints in `bot/runs/` |
| 5 | `fig5_closure_sweep.py`| `fig_opt_hunana_sweep.png`        | `sweep_opt_hunana.npz` |

Shared style (all paper figures): Computer Modern via matplotlib mathtext,
single-column canvas 3.4 in wide, 7 pt axis/title text, 5.5 pt legends.
Figures 4 and 5 are double-column and are drawn on a wider canvas.

If a `.npz` is missing, regenerate it with the corresponding sweep in
`bot/sweeps/` (see each script's docstring).
