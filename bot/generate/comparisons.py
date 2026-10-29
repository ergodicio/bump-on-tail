"""Generate Figures 3--6 from explicit physical and closure parameters."""

import json

import numpy as np

from bot import standing_wave as sw
from bot.closed_loop import fluid_eigenmode_ic, kinetic_evolve
from bot.coverage import scan_modes, scan_trajectories, standing_wave_roots, twomode_modes
from bot.experiments import ORDERS, beam_traces, fit_gamma, linear_coefficients, neural_closures
from bot.kinetic import most_unstable_kinetic
from bot.modes import damped_pair_candidates
from bot.paths import DATA


def coverage(output):
    trajectories = scan_trajectories()
    # Flat array plus offsets avoids object arrays/pickle for variable-length runs.
    offsets = np.cumsum([0] + [len(z) for z in trajectories])
    np.savez_compressed(
        output / "coverage.npz",
        scan=scan_modes(),
        pairs=twomode_modes(),
        standing=standing_wave_roots(),
        trajectory=np.concatenate(trajectories),
        offsets=offsets,
    )


def twomode(output):
    neural = neural_closures()
    time = np.linspace(0.0, 60.0, 1201)
    data = {"t": time}
    cases = ((1.0, 0.50, 0.05), (1.5, 0.40, 0.05))
    for i, (u_b, k, epsilon) in enumerate(cases):
        roots = np.array(damped_pair_candidates(k, u_b, epsilon)[:2])
        if len(roots) != 2:
            raise RuntimeError("Could not find both damped kinetic roots")
        initial = {n: sum((fluid_eigenmode_ic(k, u_b, w, n) for w in roots)) for n in (2, 3, 4)}
        traces = beam_traces(k, u_b, epsilon, time, initial, neural)
        traces["kinetic"] = 1e-3 * np.exp(-1j * roots[:, None] * time).sum(axis=0)
        data.update({f"case{i}_{name}": value for name, value in traces.items()})
        data[f"case{i}_roots"] = roots
        data[f"case{i}_params"] = np.array([u_b, k, epsilon])
    np.savez_compressed(output / "twomode.npz", **data)


def standing(output):
    neural = neural_closures()
    traces = {
        name: sw.evolve(len(coeff), sw.linear_closure(coeff))
        for name, coeff in linear_coefficients().items()
    }
    traces["beta2"] = sw.evolve(2, sw.ekr_closure)
    for name, closure in neural.items():
        traces[name] = sw.evolve(closure.N, lambda U, cl=closure: cl(U, 2.0 * U[0] / sw.K**2))
    traces["kinetic"] = sw.kinetic()
    np.savez_compressed(
        output / "standing.npz", t=sw.T, root=sw.landau_root(), epsilon=sw.EPS, **traces
    )


def scan(output):
    """Recompute all seven growth fits at the paper's specified k_* values.

    The 48 fixed wavenumbers are input parameters, not refitted here. To
    repeat their original 80-point search use bot.modes.kinetic_peak.
    """
    params = json.loads((DATA / "scan_parameters.json").read_text())
    ub = np.array(params["u_b_vals"])
    eps = np.array(params["eps_vals"])
    ks = np.array(params["k_star"])
    data = dict(u_b_vals=ub, eps_vals=eps, k_star=ks)
    for key in ["gamma_kin", "gamma_kin_eig", "gamma_kin_exact"] + [f"gamma_{n}" for n in ORDERS]:
        data[key] = np.empty(ks.shape)
    neural = neural_closures()
    for i, u_b in enumerate(ub):
        for j, epsilon in enumerate(eps):
            k = ks[i, j]
            root = most_unstable_kinetic(k, u_b, epsilon, omega_re_hi=1.5)
            end = max(60.0, 25.0 + 5.0 / root.imag)
            time = np.linspace(0.0, end, int(end / 0.1) + 1)
            initial = {n: np.zeros(n + 2, dtype=complex) for n in (2, 3, 4)}
            for y in initial.values():
                y[1] = 1e-3
            traces = beam_traces(k, u_b, epsilon, time, initial, neural)
            y0 = np.zeros(98, dtype=complex)
            y0[1] = 1e-3
            traces["kin"] = kinetic_evolve(k, u_b, epsilon, time, y0)
            for name, trace in traces.items():
                data[f"gamma_{name}"][i, j] = fit_gamma(time, trace)
            data["gamma_kin_eig"][i, j] = root.imag
            data["gamma_kin_exact"][i, j] = root.imag
            print(
                f"scan {i * len(eps) + j + 1}/{ks.size}: u_b={u_b:g}, epsilon={epsilon:g}",
                flush=True,
            )
    np.savez_compressed(output / "growth_scan.npz", **data)
