"""Shared serif typography, colors and line styles for the manuscript."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RC = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 7,
    "axes.titlesize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
ORDER = ("kinetic", "hp3", "hunana4", "opt3", "opt4", "beta2", "nn3h", "nn4h")
LABELS = {
    "kinetic": "Kinetic",
    "hp3": r"HP ($N=3$)",
    "hunana4": r"Hunana ($N=4$)",
    "opt3": r"Learned Padé ($N=3$)",
    "opt4": r"Learned Padé ($N=4$)",
    "beta2": r"EKR ($N=2$)",
    "nn3h": r"NN ($N=3$)",
    "nn4h": r"NN ($N=4$)",
}
STYLES = {
    "kinetic": dict(color="k", ls="none", marker="o", ms=1.6, markevery=20, zorder=8),
    "hp3": dict(color="C1", lw=0.7, ls="--", alpha=0.9, zorder=4.5),
    "hunana4": dict(color="C4", lw=0.7, ls=":", alpha=0.9, zorder=3),
    "opt3": dict(color="C2", lw=0.7, ls="-.", alpha=0.9, zorder=2),
    "opt4": dict(color="C5", lw=0.7, ls=(0, (3, 1, 1, 1, 1, 1)), alpha=0.9, zorder=3),
    "beta2": dict(color="C0", lw=0.9, ls=(0, (8, 1.5)), zorder=4),
    "nn3h": dict(
        color="C3",
        lw=0.8,
        ls="-",
        alpha=0.9,
        zorder=4,
        marker="^",
        ms=3,
        mfc="none",
        mew=0.7,
        markevery=(0.05, 0.1),
    ),
    "nn4h": dict(color="C6", lw=1.3, ls="-", zorder=5),
}


def save(fig, output, stem):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(output / f"{stem}.{extension}", dpi=400, bbox_inches="tight")
    plt.close(fig)
