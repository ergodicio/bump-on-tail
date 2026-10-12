"""PRX figures: homogeneous NN trained without mirror pairs, learned Pade fit at
its training frequencies (opt_hunana.opt_coefficients_nn_freq).

Checkpoints bot/runs/homog_nn_n{3,4}_nomirror.eqx (homog_nn.train(N,
frac_mirror=0.0)).  Otherwise the settings of the shipped figure drivers:

  bot/figures/homog/fig_u2_landau_twomode_2panel_nomirror.png   two-mode tests
  bot/figures/homog/fig_opt_hunana_sweep_nomirror.png           parameter scan
                                                                (--sweep-only: redraw from the saved npz;
                                                                 --no-scan: all figures, scan from the saved npz)
  bot/figures/fig_standing_wave_supp_nomirror.png               standing wave + table

The scan reuses bot/runs/sweep_opt_hunana_homog.npz and recomputes the NN and
learned Pade cells, written to bot/runs/sweep_opt_hunana_nomirror.npz.
"""
from pathlib import Path

import numpy as np

import bot.closed_loop as CL
import bot.closures.n4_nn as N4
import bot.closures.naive_nn as NV
import bot.closures.opt_hunana as O
import bot.figures_u2 as FU
from bot.closures import homog_nn as H

RUN = Path("bot/runs")
_load = H.load


def load_nomirror(N: int, path=None, **kw):
    return _load(N, path=path or RUN / f"homog_nn_n{N}_nomirror.eqx", **kw)


def scan_npz(cl3, cl4) -> Path:
    from bot.fluid import fluid_modes
    from bot.sweeps.sweep_opt_hunana import fit_gamma
    d = dict(np.load(RUN / "sweep_opt_hunana_homog.npz"))
    pade = {N: O.opt_coefficients_nn_freq(N) for N in (3, 4)}
    keys = ["opt3", "opt4", "nn3h", "nn4h"]
    for key in keys:
        d[f"gamma_{key}"] = np.full(d["k_star"].shape, np.nan)
    for N in (3, 4):
        d[f"gamma_opt{N}_eig"] = np.full(d["k_star"].shape, np.nan)
    for i, u_b in enumerate(d["u_b_vals"]):
        for j, eps in enumerate(d["eps_vals"]):
            k, gp = d["k_star"][i, j], d["gamma_kin_eig"][i, j]
            if not np.isfinite(k):
                continue
            t_end = max(60.0, 25.0 + 5.0 / gp)
            t = np.linspace(0.0, t_end, int(t_end / 0.1) + 1)
            for N in (3, 4):
                y0 = np.zeros(2 + N, dtype=complex)
                y0[1] = 1e-3
                E = CL.pade_evolve(k, u_b, eps, pade[N], t, y0, method="rk45")
                d[f"gamma_opt{N}"][i, j] = fit_gamma(t, E, t_fit_min=25.0)
                d[f"gamma_opt{N}_eig"][i, j] = float(fluid_modes(k, u_b, eps, pade[N]).imag.max())
                cl = cl3 if N == 3 else cl4
                E = H.bot_evolve(k, u_b, eps, cl, t, y0)
                d[f"gamma_nn{N}h"][i, j] = fit_gamma(t, E, t_fit_min=25.0)
    out = RUN / "sweep_opt_hunana_nomirror.npz"
    np.savez_compressed(out, **d)
    return out


def sweep_fig(path=RUN / "sweep_opt_hunana_nomirror.npz") -> None:
    """Parameter-scan figure, with the modes outside the NN training rectangle
    outlined, and the NN median errors inside and outside it."""
    import bot.figs.fig_opt_hunana_sweep as FS
    from bot.figs.fig_xi_coverage_paper import scan_outside_mask
    FS.PANELS[:] = ["hp3", "hunana4", "opt3", "opt4", "beta2", "nn3h", "nn4h"]
    FS.LABELS.update({"nn3h": r"NN ($N=3$)", "nn4h": r"NN ($N=4$)"})
    out = scan_outside_mask()
    maps = FS.error_maps(dict(np.load(path)), "relative")
    print(f"{out.sum()} of {out.size} scan modes outside the training rectangle")
    for name in FS.PANELS:
        e = maps[name]
        print(f"  {name:8s} median inside {np.nanmedian(e[~out]):.2e}, "
              f"outside {np.nanmedian(e[out]):.2e}")
    FS.make_fig(metric="relative", path=path, outline=out,
                out_name="homog/fig_opt_hunana_sweep_nomirror.png")


def main(scan: bool = True) -> None:
    O.opt_coefficients = O.opt_coefficients_nn_freq
    H.load = load_nomirror

    cl3, cl4 = H.HomogClosure(H.load(3)), H.HomogClosure(H.load(4))
    CL.nn_n4_evolve = lambda k, u_b, eps, model, t, y0, **kw: H.bot_evolve(k, u_b, eps, cl4, t, y0)
    CL.naive_evolve = lambda k, u_b, eps, model, t, y0, **kw: H.bot_evolve(k, u_b, eps, cl3, t, y0)
    N4.load_super = lambda *a, **k: None
    NV.load = lambda *a, **k: None
    FU.FIG_DIR = Path("bot/figures/homog")
    FU.fig_u2_landau_twomode_2panel(out_name="fig_u2_landau_twomode_2panel_nomirror.png")

    sweep_fig(scan_npz(cl3, cl4)) if scan else sweep_fig()

    import bot.figs.fig_standing_wave_supp as SWS
    SWS.opt_coefficients = O.opt_coefficients_nn_freq
    SWS.SAVE_PNG = "bot/figures/fig_standing_wave_supp_nomirror.png"
    SWS.main()


if __name__ == "__main__":
    import sys
    if "--sweep-only" in sys.argv[1:]:
        sweep_fig()
    else:
        main(scan="--no-scan" not in sys.argv[1:])
