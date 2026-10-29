"""Figures 3--6: plotting only; every curve comes from an NPZ array."""

import matplotlib
import numpy as np

from .style import LABELS, ORDER, RC, STYLES, plt, save


def coverage(data, output):
    d = np.load(data / "coverage.npz", allow_pickle=False)
    with matplotlib.rc_context(RC):
        fig, ax = plt.subplots(figsize=(3.4, 2.8))
        ax.fill(
            [-3, 3, 3, -3],
            [-1, -1, 1.5, 1.5],
            color="C0",
            alpha=0.12,
            lw=0,
            label="Training rectangle",
        )
        ax.plot([-3, 3, 3, -3, -3], [-1, -1, 1.5, 1.5, -1], color="C0", lw=0.8)
        ax.axhline(0, color=".5", lw=0.4)
        offsets = d["offsets"]
        for i, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
            z = d["trajectory"][start:stop]
            ax.plot(
                z.real,
                z.imag,
                color=".55",
                lw=0.35,
                alpha=0.7,
                label=r"Kinetic runs, $\hat{\zeta}(t)$" if i == 0 else None,
            )
        z = d["scan"]
        ax.scatter(
            z.real,
            z.imag,
            s=9,
            facecolors="none",
            edgecolors="k",
            lw=0.6,
            zorder=3,
            label="Most unstable modes",
        )
        for key, marker, color, label in [
            ("pairs", "D", "C1", "Two-mode tests"),
            ("standing", "s", "C3", "Standing wave"),
        ]:
            z = d[key]
            ax.scatter(
                z.real, z.imag, s=14, marker=marker, color=color, lw=0, zorder=4, label=label
            )
        ax.set(
            xlim=(-4, 3.4),
            ylim=(-1.25, 3.1),
            xlabel=r"$\mathrm{Re}\,\zeta$",
            ylabel=r"$\mathrm{Im}\,\zeta$",
        )
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=3,
            fontsize=5.5,
            handletextpad=0.4,
            borderpad=0.3,
            labelspacing=0.25,
            columnspacing=1.0,
            frameon=False,
        )
        fig.tight_layout()
        save(fig, output, "fig_xi_coverage_sim")


def outline_cells(ax, mask):
    """Draw boundaries between grid cells inside/outside the training domain."""
    ny, nx = mask.shape
    style = dict(color="C0", lw=1.1, ls="--", solid_capstyle="butt")
    for i in range(ny):
        for j in range(nx):
            if j + 1 < nx and mask[i, j] != mask[i, j + 1]:
                ax.plot([j + 0.5] * 2, [i - 0.5, i + 0.5], **style)
            if i + 1 < ny and mask[i, j] != mask[i + 1, j]:
                ax.plot([j - 0.5, j + 0.5], [i + 0.5] * 2, **style)


def scan(data, output):
    from bot.coverage import in_rectangle

    d = np.load(data / "growth_scan.npz", allow_pickle=False)
    z = np.load(data / "coverage.npz", allow_pickle=False)["scan"]
    outside = (~in_rectangle(z)).reshape(d["k_star"].shape)
    with matplotlib.rc_context(RC):
        fig, axes = plt.subplots(2, 4, figsize=(6.8, 3.5), sharex=True, sharey=True)
        for ax, name in zip(axes.flat, ORDER[1:]):
            error = abs(d[f"gamma_{name}"] / d["gamma_kin"] - 1.0)
            image = ax.imshow(
                np.log10(np.maximum(error, 1e-4)),
                origin="lower",
                aspect="auto",
                cmap="magma_r",
                vmin=-4,
                vmax=1,
                interpolation="nearest",
            )
            outline_cells(ax, outside)
            ax.set_title(LABELS[name])
            ax.set_xticks(
                range(len(d["eps_vals"])), [f"{e:g}" for e in d["eps_vals"]], fontsize=5.5
            )
            ax.set_yticks(range(len(d["u_b_vals"])), [f"{u:g}" for u in d["u_b_vals"]])
            ax.text(
                0.025,
                0.955,
                f"median {np.median(error):.2e}\nmax {np.max(error):.2e}",
                transform=ax.transAxes,
                fontsize=4.8,
                va="top",
                bbox=dict(fc="w", ec="none", alpha=0.6, pad=1.5),
            )
        axes[1, 3].set_visible(False)
        for j in range(4):
            ax = axes[1, j] if j < 3 else axes[0, j]
            ax.set_xlabel(r"$\varepsilon=n_b/n_0$")
            ax.tick_params(labelbottom=True)
        for ax in axes[:, 0]:
            ax.set_ylabel(r"$u_b/v_{tb}$")
        fig.tight_layout(rect=(0, 0, 0.92, 0.98))
        cax = fig.add_axes((0.94, 0.14, 0.014, 0.72))
        cb = fig.colorbar(image, cax=cax, label=r"$|\gamma_{\rm eff}/\gamma_{\rm kin}-1|$")
        cb.set_ticks([-4, -3, -2, -1, 0, 1])
        cb.set_ticklabels([r"$\leq10^{-4}$"] + [rf"$10^{{{n}}}$" for n in (-3, -2, -1, 0, 1)])
        save(fig, output, "fig_opt_hunana_sweep_nomirror")


def twomode(data, output):
    d = np.load(data / "twomode.npz", allow_pickle=False)
    time = d["t"]
    with matplotlib.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
        for i, ax in enumerate(axes):
            values = []
            for name in ORDER:
                trace = abs(d[f"case{i}_{name}"])
                values.append(trace)
                ax.semilogy(time, trace, label=LABELS[name], **STYLES[name])
            finite = np.concatenate(values)
            finite = finite[np.isfinite(finite) & (finite > 0)]
            ax.set(
                xlim=(0, 60),
                ylim=(
                    10 ** (np.floor(np.log10(finite.min())) - 0.5),
                    10 ** (np.ceil(np.log10(finite.max())) + 0.2),
                ),
                title=f"({chr(97 + i)})",
                xlabel=r"$t\,\omega_{pe}$",
                ylabel=r"$|E|$",
            )
        axes[0].legend(
            fontsize=5.5,
            loc="lower left",
            ncol=2,
            labelspacing=0.25,
            handlelength=1.6,
            borderpad=0.3,
            columnspacing=1.0,
        )
        fig.tight_layout(w_pad=1.5)
        save(fig, output, "fig_u2_landau_twomode_2panel_nomirror")


def standing(data, output):
    d = np.load(data / "standing.npz", allow_pickle=False)
    time = d["t"]
    amp = d["epsilon"]
    with matplotlib.rc_context(RC):
        fig, (real, envelope) = plt.subplots(1, 2, figsize=(6.8, 2.1))
        for name in ORDER:
            style = STYLES[name].copy()
            if name == "kinetic":
                style["markevery"] = 25
            if name == "nn3h":
                style["markevery"] = (len(time) // 24, len(time) // 12)
            real.plot(time, d[name].real / amp, label=LABELS[name], **style)
            envelope.semilogy(time, abs(d[name]) / amp, label=LABELS[name], **style)
        window = (time > 5) & (time < 15)
        gamma = d["root"].imag
        scale = np.max(abs(d["kinetic"][window]) / amp / np.exp(gamma * time[window]))
        envelope.semilogy(
            time,
            scale * np.exp(gamma * time),
            color=".5",
            lw=0.6,
            ls="-.",
            label=r"$\propto e^{\gamma_{\rm L} t}$",
        )
        real.set_ylabel(r"Re $U_0/\epsilon$")
        envelope.set_ylabel(r"$|U_0|/\epsilon$")
        envelope.set_ylim(1e-3, 3)
        for i, ax in enumerate((real, envelope)):
            ax.set(xlim=(0, 40), xlabel=r"$t\,\omega_{pe}$")
            ax.text(0.97, 0.95, f"({chr(97 + i)})", transform=ax.transAxes, va="top", ha="right")
        fig.tight_layout(rect=[0, 0.09, 1, 1])
        handles, labels = envelope.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=9,
            fontsize=5.6,
            labelspacing=0.25,
            handlelength=2.6,
            handletextpad=0.4,
            columnspacing=0.6,
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
        )
        save(fig, output, "fig_standing_wave_supp_nomirror")
