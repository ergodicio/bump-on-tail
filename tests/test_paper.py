"""Paper diagnostics and an end-to-end plot-only run."""

import os
import subprocess
import sys

import numpy as np

from bot.coverage import in_rectangle
from bot.paths import DATA, ROOT


def test_paper_diagnostics():
    with np.load(DATA / "coverage.npz") as data:
        assert len(data["scan"]) == 48
        assert (~in_rectangle(data["scan"])).sum() == 13
        assert len(data["trajectory"]) == 32394
        assert np.mean(data["trajectory"].imag < 0) < 0.001
    with np.load(DATA / "growth_scan.npz") as data:
        assert data["gamma_nn4h"].shape == (8, 6)
        assert np.all(data["gamma_kin"] > 0)
        nn_error = abs(data["gamma_nn4h"] / data["gamma_kin"] - 1)
        hp_error = abs(data["gamma_hp3"] / data["gamma_kin"] - 1)
        assert np.median(nn_error) < 1e-4
        assert np.median(hp_error) > 1e-2


def test_all_figures_render_without_importing_jax(tmp_path):
    # A plotting environment must not need optional neural dependencies.
    script = """
import builtins,sys
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in ('jax','equinox','optax'):
        raise AssertionError('Plotting imported '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from bot.figures.__main__ import main
main()
"""
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "mpl")
    env["PYTHONPATH"] = str(ROOT)
    subprocess.run(
        [sys.executable, "-c", script, "--output", str(tmp_path / "figures")],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    files = list((tmp_path / "figures").glob("*"))
    assert len([p for p in files if p.suffix == ".png"]) == 6
    assert len([p for p in files if p.suffix == ".pdf"]) == 5
    assert all(p.stat().st_size > 1000 for p in files if p.suffix in (".png", ".pdf"))
