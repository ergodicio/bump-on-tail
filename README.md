# bump-on-tail

Neural closure models for the bump-on-tail instability.

The bump-on-tail problem is solved with a custom ODE-fluid solver (no
Hermite expansion). The fluid system is closed by a learned heat-flux
closure trained against either inference-time targets, kinetic
trajectory data, or analytical Padé baselines.

## Layout

```
bot/
  fluid.py             fluid system + trace_fluid (closure-augmented ODE solver)
  kinetic.py           Z-function root finder for kinetic dispersion
  closed_loop.py       closed-system matrix + rollout + per-trajectory loss
  landau_damping.py    Landau-damping diagnostics
  closures/
    pade.py            analytical Padé closure (parameter-free baseline)
    inference.py       inference-time MLP closure (alpha, moment ratios, xi)
    naive_nn.py        naive MLP closure (trained on raw trajectories)
  train_inference.py   train inference-time MLP
  train_simdata.py     train MLP against simulation trajectory data
  eval_inference.py    inference-time evaluation
  sweep.py             Padé sweep across (k, vt)
  sweep_inference.py   inference-MLP sweep
  sweep_timedomain.py  time-domain sweep across closures
  figures.py           regenerate all paper/slide figures
  figures/             generated PNGs
  runs/                model checkpoints (.eqx) and cached sweeps (.npz)
  paper/               paper.tex + paper.pdf + paper/figures/
  slides/              slides.tex + slides.pdf + slides/figures/
```

## Setup

```bash
uv venv
uv pip install -e .
```

## Usage

All entry points run as modules from the repo root:

```bash
python -m bot.train_inference        # train inference-time MLP closure
python -m bot.train_simdata          # train against simulation data
python -m bot.eval_inference         # evaluate inference closure

python -m bot.sweep                  # Padé sweep
python -m bot.sweep_inference        # inference-MLP sweep
python -m bot.sweep_timedomain       # time-domain sweep

python -m bot.figures                # regenerate all figures
```

Model checkpoints and cached sweep data live in `bot/runs/`; figures
land in `bot/figures/`.

## Paper & slides

LaTeX sources and built PDFs are checked in under `bot/paper/` and
`bot/slides/`. To rebuild:

```bash
cd bot/paper && latexmk -pdf paper.tex
cd bot/slides && latexmk -pdf slides.tex
```
