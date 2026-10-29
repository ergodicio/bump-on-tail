# Bump-on-tail: PPCF paper figures

This branch contains the numerical methods, two trained neural closures, and
inputs needed for the six figures in the current **BoT Closure Paper** PPCF
manuscript (`main_ppcf.tex`). The research history remains on `prx_figures`;
experimental closures, ablations, slides, old manuscripts, and duplicate plots
are absent from this branch's checkout.

## Redraw the figures

Use Python 3.11 or later in a virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m bot.figures --output build/figures
```

This reads the bundled numerical arrays and needs only NumPy, SciPy, and
Matplotlib. It does not load JAX or train a model. Each numerical figure is
written as a vector PDF and a 400-dpi PNG. The equivariance illustration is
copied from the final manuscript asset. `render_inputs.json` records input hashes.
To draw one figure, add e.g. `--figure twomode`.

| Paper figure | CLI name | Numerical inputs in `bot/data/` | Manuscript filename |
| --- | --- | --- | --- |
| 1. Kappa beam and slab ITG validation | `validation` | `validation_kappa.npz`, `validation_itg_trace.npz`, `validation_itg_scan.npz` | `fig_validation_ppcf_compact.pdf` |
| 2. Equivariance schematic | `equivariance` | static asset in `bot/assets/` | `fig_equivariance_serif.png` |
| 3. Training-domain coverage | `coverage` | `coverage.npz` | `fig_xi_coverage_sim.png` |
| 4. Growth-rate scan | `scan` | `growth_scan.npz`, `coverage.npz` | `fig_opt_hunana_sweep_nomirror.png` |
| 5. Two damped mode pairs | `twomode` | `twomode.npz` | `fig_u2_landau_twomode_2panel_nomirror.png` |
| 6. Standing plasma wave | `standing` | `standing.npz` | `fig_standing_wave_supp_nomirror.png` |

The plot drivers preserve the paper's data, panel order, colors, and line styles.
Figure 1 uses the current PPCF 2-by-2 layout. Figure 4 is sized at manuscript
width with legible serif labels. The branch does not edit the Overleaf project.

## Recompute the numerical data

```sh
python -m pip install -e '.[compute,test]'
# Optional: install the exact direct-dependency versions used for verification.
python -m pip install -r requirements-reproduce.txt
export JAX_ENABLE_X64=0
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -m bot.generate all --output build/recomputed
python -m bot.figures --data build/recomputed --output build/recomputed/figures
python -m bot.verify --data build/recomputed
```

Use a new output directory for each calculation: the generator refuses to
overwrite existing result files. Individual experiments (`validation`, `coverage`,
`scan`, `twomode`, `standing`) can be generated separately. The full calculation
uses dense kinetic eigensolves up to 1000 velocity points, plus 48 growth-rate
comparisons. It is more expensive than redrawing. Model weights are loaded from
`bot/models/homog_nn_n{3,4}_nomirror.eqx` and are never retrained implicitly.

The scan uses the 48 specified peak wavenumbers in `scan_parameters.json`,
selected by the source calculation's 80-point search on k in [0.1, 1.2]. Holding
these input parameters fixed avoids changing the experiment during reproduction.
`bot.modes.kinetic_peak` implements the original search if it is needed again.
The growth errors use late-time kinetic **fits** as their reference, matching
the plotted paper metric; the analytic root rate is also retained separately.

Each generator writes version and input provenance alongside its results.
[docs/reproduction.md](docs/reproduction.md) describes parameters, array fields,
normalizations, tolerances, and the numerical checks performed during cleanup.

## Retrain or refit

The paper networks use 400,000 analytic single/two-mode states, seed 0, three
192-unit tanh layers, batch size 1024, and 150,000 Adam updates. Half of the
proposals are single modes, with loss weight 3; the remaining proposals are
generic pairs. Mirror-pair augmentation and impulse-response data are absent.
The normalized joint-state Gram matrix includes the potential with unit weight.

```sh
python -m bot.train --order 3 --output build/training/nn3.eqx
python -m bot.train --order 4 --output build/training/nn4.eqx
python -m bot.fit_pade --output build/training/learned_pade.npz
```

Training writes a new checkpoint and configuration. To use an alternative model
in an experiment, pass a `HomogClosure(load(order, path=...))` to the experiment
functions; published checkpoint files stay fixed. Learned Padé fits use the
accepted training states' frequencies and parity-constrained relative moment
error. Their published coefficients are explicit in `bot/closures/opt_hunana.py`.

## Read the implementation

Start with `bot/experiments.py`: it names the seven closures and evolves each
from the same initial state at its order. `bot/generate/` contains the figure
experiments, while `bot/figures/` contains plotting only. The physical systems
are in `kinetic.py`, `fluid.py`, `kappa_bot.py`, `slab_itg.py`, and
`standing_wave.py`. `bot/closures/response.py` gives the single-mode formulas;
`homog_nn.py` gives the Gram features, training loss, and neural model.

```sh
python -m pytest
```

Tests check response identities, equilibrium normalization, mode consistency,
complex-scaling equivariance, checkpoint loading, training/save/load, cached
paper diagnostics, and figure rendering. Neural tests skip if the optional
compute dependencies are not installed. Full 150,000-step retraining was not
repeated during cleanup; the supplied checkpoints are the published inputs.
