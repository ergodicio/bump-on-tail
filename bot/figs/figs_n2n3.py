"""Figs. 2 and 3 with the homogeneous NN closures at N=2 and N=3 (both trained on
eigenmode superpositions), in place of N=3 and N=4.  Same styling and data as
figures_u2._plot_landau_twomode_case / fig_opt_hunana_sweep.
Outputs: bot/figures/homog/n2n3/{fig_u2_landau_twomode_single,fig_opt_hunana_sweep}.png
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bot.closures import homog_nn as H
from bot.closed_loop import (direct_u2_evolve, pade_evolve, fluid_eigenmode_ic_u2,
                             fluid_eigenmode_ic_u3, fluid_eigenmode_ic_n4)
from bot.closures.opt_hunana import opt_coefficients
from bot.closures.pade import pade_coefficients, two_point_coefficients
from bot.sweeps.sweep_opt_hunana import fit_gamma

OUT = Path("bot/figures/homog/n2n3"); OUT.mkdir(parents=True, exist_ok=True)
RUN = H.RUN_DIR
cl = {N: H.HomogClosure(H.load(N)) for N in (2, 3)}

# ------------------------------------------------------------------ Fig. 2 --
u_b, k, eps, amp, t_end = 1.5, 0.40, 0.05, 1e-3, 60.0
w1, w2 = 1.205 - 0.119j, 0.717 - 0.106j
t = np.linspace(0, t_end, 1201)
E_ref = amp * (np.exp(-1j * w1 * t) + np.exp(-1j * w2 * t))
y0_u2 = fluid_eigenmode_ic_u2(k, u_b, w1, amp) + fluid_eigenmode_ic_u2(k, u_b, w2, amp)
y0_u3 = fluid_eigenmode_ic_u3(k, u_b, w1, amp) + fluid_eigenmode_ic_u3(k, u_b, w2, amp)
y0_n4 = fluid_eigenmode_ic_n4(k, u_b, w1, amp) + fluid_eigenmode_ic_n4(k, u_b, w2, amp)
curves = [
    ("Kinetic", E_ref, dict(color="k", ls="none", marker="o", ms=1.6, markevery=20, zorder=8)),
    (r"HP ($N{=}3$)", pade_evolve(k, u_b, eps, pade_coefficients(3), t, y0_u3, method="rk45"), dict(color="C1", lw=0.7, ls="--", alpha=0.9)),
    (r"Hunana ($N=4$)", pade_evolve(k, u_b, eps, two_point_coefficients(4, 2), t, y0_n4, method="rk45"), dict(color="C4", lw=0.7, ls=":", alpha=0.9)),
    (r"Padé opt ($N=3$)", pade_evolve(k, u_b, eps, opt_coefficients(3), t, y0_u3, method="rk45"), dict(color="C2", lw=0.7, ls="-.", alpha=0.9)),
    (r"Padé opt ($N=4$)", pade_evolve(k, u_b, eps, opt_coefficients(4), t, y0_n4, method="rk45"), dict(color="C5", lw=0.7, ls="-.", alpha=0.9)),
    (r"EKR ($N=2$)", direct_u2_evolve(k, u_b, eps, t, y0_u2), dict(color="C0", lw=0.8, ls="-")),
    (r"NN ($N=2$)", H.bot_evolve(k, u_b, eps, cl[2], t, y0_u2), dict(color="C3", lw=0.7, ls="-", alpha=0.9)),
    (r"NN ($N=3$)", H.bot_evolve(k, u_b, eps, cl[3], t, y0_u3), dict(color="C6", lw=1.0, ls="-")),
]
for lab, E, _ in curves[5:]:
    print(f"Fig. 2 max |E-E_ref|/amp, {lab}: {np.max(np.abs(np.asarray(E)[:len(t)] - E_ref) / amp):.2e}")
with matplotlib.rc_context({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 7, "axes.titlesize": 7}):
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 1.92))
    vals = []
    for lab, E, sty in curves:
        E = np.asarray(E)[:len(t)]; ax.semilogy(t, np.abs(E), label=lab, **sty); vals.append(np.abs(E))
    v = np.concatenate(vals); v = v[np.isfinite(v) & (v > 0)]
    ax.set_ylim(10 ** (np.floor(np.log10(v.min())) - 0.5), 10 ** (np.ceil(np.log10(v.max())) + 0.2))
    ax.set_xlim(0, t_end); ax.set_xlabel(r"$t\,\omega_{pe}$"); ax.set_ylabel(r"$|E|$")
    fig.tight_layout(rect=[0, 0.17, 1, 1])
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=4.8, labelspacing=0.25, handlelength=1.3,
               handletextpad=0.4, columnspacing=1.0, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.savefig(OUT / "fig_u2_landau_twomode_single.png", dpi=400, bbox_inches="tight"); plt.close(fig)
print("saved", OUT / "fig_u2_landau_twomode_single.png")

# ------------------------------------------------------------------ Fig. 3 --
d = dict(np.load(RUN / "sweep_opt_hunana.npz"))
for N in (2, 3):
    gam = np.full_like(d["gamma_kin"], np.nan)
    for i, ub in enumerate(d["u_b_vals"]):
        for j, ep in enumerate(d["eps_vals"]):
            kk, gp = d["k_star"][i, j], d["gamma_kin_eig"][i, j]
            if not np.isfinite(kk): continue
            te = max(60.0, 25.0 + 5.0 / gp); tt = np.linspace(0, te, int(te / 0.1) + 1)
            y0 = np.zeros(2 + N, dtype=complex); y0[1] = amp
            gam[i, j] = fit_gamma(tt, H.bot_evolve(kk, ub, ep, cl[N], tt, y0))
    d[f"gamma_nn{N}h"] = gam
    rel = np.abs(gam - d["gamma_kin"]) / np.abs(d["gamma_kin"])
    print(f"sweep NN N={N}: median {np.nanmedian(rel):.2e}   max {np.nanmax(rel):.2e}")
np.savez_compressed(RUN / "sweep_opt_hunana_homog_n2n3.npz", **d)
import bot.figs.fig_opt_hunana_sweep as FS
FS.PANELS[:] = ["hp3", "hunana4", "opt3", "opt4", "beta2", "nn2h", "nn3h"]
FS.LABELS.update({"nn2h": r"NN ($N=2$)", "nn3h": r"NN ($N=3$)"})
FS.make_fig(metric="relative", path=RUN / "sweep_opt_hunana_homog_n2n3.npz", out_name="homog/n2n3/fig_opt_hunana_sweep.png")
