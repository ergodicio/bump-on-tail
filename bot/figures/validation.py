"""Figure 1: kappa beam and slab ITG validation, drawn from numerical data."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, MaxNLocator

from bot.paths import DATA

INPUTS = ("validation_kappa.npz", "validation_itg_trace.npz", "validation_itg_scan.npz")


def main(output, data=DATA):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    kd = np.load(data / INPUTS[0], allow_pickle=False)
    r = np.load(data / INPUTS[1], allow_pickle=False)
    s = np.load(data / INPUTS[2], allow_pickle=False)
    kappa_scan = (2, 3, 4, 6, 8, 12, 20, 40)
    rc = {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    legend = dict(
        fontsize=6.5,
        handlelength=1.6,
        labelspacing=0.25,
        borderpad=0.3,
        handletextpad=0.4,
        framealpha=0.9,
    )
    with matplotlib.rc_context(rc):
        fig, axes = plt.subplots(2, 2, figsize=(6.02, 3.35), layout="constrained")
        a, b, c, d = axes.ravel()

        t = kd["t"]
        a.semilogy(t, np.abs(kd["E_kin"]), "ko", ms=1.8, markevery=44, zorder=10, label="Kinetic")
        a.semilogy(t, np.abs(kd["E_hp"]), "C1--", lw=1.0, zorder=5, label=r"HP ($N{=}3$)")
        a.semilogy(t, np.abs(kd["E_ekr2"]), "C0-", lw=1.0, zorder=4, label=r"EKR ($N{=}2$)")
        a.set_title(r"(a) Bump-on-tail, $\kappa=2$")
        a.set_xlabel(r"$t\,\omega_{pe}$")
        a.set_ylabel(r"$|E|$")
        a.set_ylim(1e-5, None)
        a.legend(loc="lower right", **legend)

        zs, tau, eta = r["itg1_params"]
        ti = r["t_itg1"]
        b.semilogy(
            ti,
            np.abs(r["itg1_kin"]),
            color="k",
            ls="none",
            marker="o",
            ms=1.8,
            markevery=20,
            zorder=10,
            label="Kinetic",
        )
        b.semilogy(ti, np.abs(r["itg1_hp"]), "C1--", lw=1.0, zorder=5, label=r"HP ($N{=}3$)")
        b.semilogy(ti, np.abs(r["itg1_beta"]), "C0-", lw=1.0, zorder=4, label=r"EKR ($N{=}2$)")
        b.set_title(
            rf"(b) Slab ITG, $\zeta_*={zs:g}$, "
            rf"$\tau={tau:g}$, $\eta={eta:g}$"
        )
        b.set_xlabel(r"$t\,k_\parallel v_{ti}$")
        b.set_ylabel(r"$|U_{0,i}|$")
        b.set_ylim(1e-6, None)
        b.legend(loc="lower right", **legend)

        c.plot(kappa_scan, kd["g_kin"], "k-o", lw=1.1, ms=3, label="Kinetic")
        c.axhline(kd["g_hp"], color="C1", ls="--", lw=1.0, label=r"HP ($N{=}3$)")
        c.plot(kappa_scan, kd["g_ekr"], "C0^", ms=4, label=r"EKR ($N{=}2$)")
        c.axhline(kd["g_maxw"], color="0.6", ls=":", lw=0.9, label="Maxwellian")
        c.set_xscale("log")
        c.set_xticks(kappa_scan)
        c.set_xticklabels([str(k) for k in kappa_scan])
        c.set_title("(c) Bump-on-tail growth rate")
        c.set_xlabel(r"$\kappa$")
        c.set_ylabel(r"$\gamma/\omega_{pe}$")
        lo = min(kd["g_maxw"], kd["g_kin"].min())
        hi = max(kd["g_kin"].max(), kd["g_hp"])
        c.set_ylim(lo - 0.08 * (hi - lo), hi + 0.75 * (hi - lo))
        c.legend(loc="upper right", ncol=2, **legend)

        eta_c, eta_f = s["eta_curve"], s["eta_fit"]
        eta_th = 1 + np.sqrt(5)

        def gr(z):
            return np.where(np.isnan(z.real), 0.0, np.maximum(z.imag, 0.0))

        d.axhline(0, color="k", lw=0.5)
        d.axvline(eta_th, color="0.6", lw=0.9, ls=":")
        d.plot(eta_c, gr(s["z_kin"]), "k-", lw=1.2, zorder=4, label="Kinetic")
        d.plot(eta_c, gr(s["z_hp"]), "C1--", lw=1.0, zorder=3, label=r"HP ($N{=}3$)")
        d.plot(
            eta_f,
            np.maximum(s["fit_hp"].imag, 0),
            "C1s",
            ms=3.2,
            mew=0.8,
            mfc="none",
            zorder=5,
            label="HP, fit",
        )
        d.plot(eta_f, np.maximum(s["fit_beta"].imag, 0), "C0^", ms=4, zorder=5, label="EKR, fit")
        d.set_title(r"(d) ITG growth rate, $\zeta_*=1$, $\tau=1$")
        d.set_xlabel(r"$\eta=L_n/L_T$")
        d.set_ylabel(r"$\gamma/(|k_\parallel|v_{ti})$")
        d.legend(loc="upper left", ncol=2, **legend)

        # Keep the shorter panels readable without crowding logarithmic ticks.
        a.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
        b.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
        c.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        d.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))

        # Verify every time trace and scan point against its cached input.
        comparisons = [
            (a.lines[0].get_ydata(), np.abs(kd["E_kin"])),
            (a.lines[1].get_ydata(), np.abs(kd["E_hp"])),
            (a.lines[2].get_ydata(), np.abs(kd["E_ekr2"])),
            (b.lines[0].get_ydata(), np.abs(r["itg1_kin"])),
            (b.lines[1].get_ydata(), np.abs(r["itg1_hp"])),
            (b.lines[2].get_ydata(), np.abs(r["itg1_beta"])),
            (c.lines[0].get_ydata(), kd["g_kin"]),
            (c.lines[2].get_ydata(), kd["g_ekr"]),
            (d.lines[2].get_ydata(), gr(s["z_kin"])),
            (d.lines[3].get_ydata(), gr(s["z_hp"])),
            (d.lines[4].get_ydata(), np.maximum(s["fit_hp"].imag, 0)),
            (d.lines[5].get_ydata(), np.maximum(s["fit_beta"].imag, 0)),
        ]
        for plotted, expected in comparisons:
            np.testing.assert_array_equal(plotted, expected)
        for fmt in ("pdf", "png"):
            fig.savefig(output / f"fig_validation_ppcf_compact.{fmt}", dpi=400)
        plt.close(fig)

    print("Validation figure: 12 plotted arrays verified against bundled data.")
