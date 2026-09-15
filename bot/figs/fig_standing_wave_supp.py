"""Supplement figure: Landau-damped standing Langmuir wave, seven closures.

Same system, initial condition and solvers as fig_standing_wave.py; same
closure set, labels and line styles as Fig. 2 (fig_u2_landau_twomode_single),
with the NN entries being the homogeneous closures of bot/closures/homog_nn.py.
Output: bot/figures/fig_standing_wave_supp.png and a table of fitted
(omega, gamma) per closure.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bot.closures import homog_nn as H
from bot.closures.opt_hunana import opt_coefficients
from bot.closures.pade import pade_coefficients, two_point_coefficients
import bot.figs.fig_standing_wave as S

SAVE_PNG = "bot/figures/fig_standing_wave_supp.png"


def main() -> None:
    om = S.landau_root()
    U_kin = S.kinetic()
    cl3, cl4 = H.HomogClosure(H.load(3)), H.HomogClosure(H.load(4))
    phi = lambda U: (2.0 / S.K ** 2) * U[0]        # Poisson: Phi-hat in terms of U_0

    runs = [  # (label, U0(t), style) -- styles as in bot.figures_u2._plot_landau_twomode_case
        ("Kinetic", U_kin, dict(color="k", ls="none", marker="o", ms=1.6, markevery=25, zorder=8)),
        (r"HP ($N{=}3$)", S.evolve(3, S.linear_closure(pade_coefficients(3))),
         dict(color="C1", lw=0.7, ls="--", alpha=0.9, zorder=4.5)),
        (r"Hunana ($N=4$)", S.evolve(4, S.linear_closure(two_point_coefficients(4, 2))),
         dict(color="C4", lw=0.7, ls=":", alpha=0.9, zorder=3)),
        (r"Padé opt ($N=3$)", S.evolve(3, S.linear_closure(opt_coefficients(3))),
         dict(color="C2", lw=0.7, ls="-.", alpha=0.9, zorder=2)),
        (r"Padé opt ($N=4$)", S.evolve(4, S.linear_closure(opt_coefficients(4))),
         dict(color="C5", lw=0.7, ls="-.", alpha=0.9, zorder=3)),
        (r"EKR ($N=2$)", S.evolve(2, S.ekr_closure),
         dict(color="C0", lw=0.8, ls="-", zorder=4)),
        (r"NN ($N=3$)", S.evolve(3, lambda U: cl3(U, phi(U))),
         dict(color="C3", lw=0.7, ls="-", alpha=0.9, zorder=4)),
        (r"NN ($N=4$)", S.evolve(4, lambda U: cl4(U, phi(U))),
         dict(color="C6", lw=1.0, ls="-", zorder=5)),
    ]

    print(f"k lambda_D = {S.K_LAMBDA_D}: Landau root omega = {om.real:.4f} {om.imag:+.4f}i")
    print(f"{'closure':<18}{'omega':>8}{'err':>8}{'gamma':>9}{'err':>8}")
    i10 = np.searchsorted(S.T, 10.0)
    for lab, U0, _ in runs:
        w, g = S.fit_mode(U0)
        if np.isnan(w) or abs(U0[i10]) / S.EPS < 1e-6:
            print(f"{lab:<18}{'--':>8}{'--':>8}{'--':>9}{'--':>8}   collapsed, |U0|/eps(t=10) = {abs(U0[i10])/S.EPS:.1e}")
            continue
        print(f"{lab:<18}{w:>8.4f}{100*(w/om.real-1):>+7.1f}%{g:>+9.4f}{100*(g/om.imag-1):>+7.1f}%")

    with matplotlib.rc_context({"font.family": "serif", "mathtext.fontset": "cm",
                                "font.size": 7, "axes.titlesize": 7}):
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.8, 2.1))
        for lab, U0, sty in runs:
            ax0.plot(S.T, U0.real / S.EPS, label=lab, **sty)
            ax1.semilogy(S.T, np.abs(U0) / S.EPS, label=lab, **sty)
        m = (S.T > 5.0) & (S.T < 15.0)                 # envelope amplitude of the kinetic trace
        A = np.max(np.abs(U_kin[m]) / S.EPS / np.exp(om.imag * S.T[m]))
        ax1.semilogy(S.T, A * np.exp(om.imag * S.T), color="0.5", lw=0.6, ls="-.",
                     label=r"$\propto e^{\gamma_{\rm L} t}$")
        ax0.set_ylabel(r"Re $U_0/\epsilon$")
        ax1.set_ylabel(r"$|U_0|/\epsilon$")
        ax1.set_ylim(1e-3, 3)
        for ax, tag in ((ax0, "(a)"), (ax1, "(b)")):
            ax.set_xlim(0, S.T_END)
            ax.set_xlabel(r"$t\,\omega_{pe}$")
            ax.text(0.97, 0.95, tag, transform=ax.transAxes, va="top", ha="right")
        fig.tight_layout(rect=[0, 0.09, 1, 1])
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=9, fontsize=5.6,
                   labelspacing=0.25, handlelength=1.4, handletextpad=0.4,
                   columnspacing=0.9, frameon=False, bbox_to_anchor=(0.5, 0.01))
        fig.savefig(SAVE_PNG, dpi=400, bbox_inches="tight")
        plt.close(fig)
    print(f"saved {SAVE_PNG}")


if __name__ == "__main__":
    main()
