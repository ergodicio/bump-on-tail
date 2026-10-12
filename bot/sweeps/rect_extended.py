"""Experiment: retrain the learned closures on an extended zeta rectangle.

    python -m bot.sweeps.rect_extended train   # homog NN, N = 3 and 4
    python -m bot.sweeps.rect_extended pade    # learned Pade, N = 3 and 4
    python -m bot.sweeps.rect_extended eval    # old vs new on the paper's tests

Rectangle Re zeta in (-4, 4), Im zeta in (-1, 3), which covers all 48 growing
modes of the parameter scan; sample counts scaled by the area ratio 32/15.
Outputs carry the suffix _rect4x3 in bot/runs/; shipped models are untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RE_EXT, IM_EXT = (-4.0, 4.0), (-1.0, 3.0)
AREA_RATIO = (8.0 * 4.0) / (6.0 * 2.5)
SUF = "_rect4x3"
RUN = Path(__file__).resolve().parent.parent / "runs"
PADE_NPZ = RUN / f"opt_coefficients{SUF}.npz"


def train() -> None:
    from bot.closures import homog_nn as H
    H.XI_RE, H.XI_IM = RE_EXT, IM_EXT
    for N in (3, 4):
        m = H.train(N, n_samples=int(round(400_000 * AREA_RATIO)))
        print(f"saved {H.save(m, N, path=RUN / f'homog_nn_n{N}{SUF}.eqx')}", flush=True)


def pade() -> None:
    from bot.closures import opt_hunana as O
    out = {}
    for N in (3, 4):
        fit = O.fit_coefficients(N, re_range=RE_EXT, im_range=IM_EXT,
                                 n_samples=int(round(20_000 * AREA_RATIO)))
        print(fit)
        print(f"  poles {np.sort_complex(fit.poles)}")
        out[f"a{N}"] = fit.a
    np.savez(PADE_NPZ, **out)
    print(f"saved {PADE_NPZ}")


def _closures():
    from bot.closures import homog_nn as H
    from bot.closures.opt_hunana import opt_coefficients
    p = np.load(PADE_NPZ)
    return {
        "old": {"nn3": H.HomogClosure(H.load(3)), "nn4": H.HomogClosure(H.load(4)),
                "opt3": opt_coefficients(3), "opt4": opt_coefficients(4)},
        "new": {"nn3": H.HomogClosure(H.load(3, path=RUN / f"homog_nn_n3{SUF}.eqx")),
                "nn4": H.HomogClosure(H.load(4, path=RUN / f"homog_nn_n4{SUF}.eqx")),
                "opt3": p["a3"], "opt4": p["a4"]},
    }


def _scan(cl) -> dict:
    from bot.closed_loop import pade_evolve
    from bot.closures import homog_nn as H
    from bot.sweeps.sweep_opt_hunana import fit_gamma
    d = np.load(RUN / "sweep_opt_hunana.npz")
    err = {name: [] for name in cl}
    for i, u_b in enumerate(d["u_b_vals"]):
        for j, eps in enumerate(d["eps_vals"]):
            k, gp, gk = d["k_star"][i, j], d["gamma_kin_eig"][i, j], d["gamma_kin"][i, j]
            t_end = max(60.0, 25.0 + 5.0 / gp)
            t = np.linspace(0.0, t_end, int(t_end / 0.1) + 1)
            for name, c in cl.items():
                n = c.N if name.startswith("nn") else len(c)
                y0 = np.zeros(2 + n, dtype=complex); y0[1] = 1e-3
                E = (H.bot_evolve(k, u_b, eps, c, t, y0) if name.startswith("nn")
                     else pade_evolve(k, u_b, eps, c, t, y0, method="rk45"))
                err[name].append(abs(fit_gamma(t, E, t_fit_min=25.0) - gk) / abs(gk))
    return {name: np.array(v) for name, v in err.items()}


def _twomode(cl) -> dict:
    import bot.figures_u2 as FU
    from bot.closed_loop import fluid_eigenmode_ic_n4, pade_evolve
    from bot.closures import homog_nn as H
    amp, t = 1e-3, np.linspace(0, 60, 1201)
    out = {name: [] for name in cl}
    for u_b, k, eps in ((1.0, 0.50, 0.05), (1.5, 0.40, 0.05)):
        w1, w2 = FU._find_landau_twomode_modes(k, u_b, eps)[:2]
        ref = amp * (np.exp(-1j * w1 * t) + np.exp(-1j * w2 * t))
        y3 = FU.fluid_eigenmode_ic_u3(k, u_b, w1, amp=amp) + FU.fluid_eigenmode_ic_u3(k, u_b, w2, amp=amp)
        y4 = fluid_eigenmode_ic_n4(k, u_b, w1, amp=amp) + fluid_eigenmode_ic_n4(k, u_b, w2, amp=amp)
        for name, c in cl.items():
            y0 = y3 if name.endswith("3") else y4
            E = (H.bot_evolve(k, u_b, eps, c, t, y0) if name.startswith("nn")
                 else pade_evolve(k, u_b, eps, c, t, y0, method="rk45"))
            E = np.asarray(E)[:len(t)]
            out[name].append(np.nanmax(np.abs(E - ref)) / amp)
    return out


def _standing(cl) -> dict:
    import bot.figs.fig_standing_wave as SW
    om = SW.landau_root()
    out = {}
    for name, c in cl.items():
        N = c.N if name.startswith("nn") else len(c)
        f = ((lambda U, c=c: c(U, (2.0 / SW.K ** 2) * U[0])) if name.startswith("nn")
             else SW.linear_closure(c))
        w, g = SW.fit_mode(SW.evolve(N, f))
        out[name] = (w / om.real - 1, g / om.imag - 1)
    return out


def evaluate() -> None:
    cls = _closures()
    res = {tag: (_scan(c), _twomode(c), _standing(c)) for tag, c in cls.items()}
    print(f"\n{'closure':8s} {'':4s} {'scan median':>12s} {'scan max':>10s} "
          f"{'2-mode (a)':>11s} {'2-mode (b)':>11s} {'SW d_omega':>11s} {'SW d_gamma':>11s}")
    for name in ("opt3", "opt4", "nn3", "nn4"):
        for tag in ("old", "new"):
            sc, tm, sw = res[tag]
            print(f"{name:8s} {tag:4s} {np.median(sc[name]):12.2e} {sc[name].max():10.2e} "
                  f"{tm[name][0]:11.2e} {tm[name][1]:11.2e} {100*sw[name][0]:+10.1f}% {100*sw[name][1]:+10.1f}%")


if __name__ == "__main__":
    {"train": train, "pade": pade, "eval": evaluate}[sys.argv[1]]()
