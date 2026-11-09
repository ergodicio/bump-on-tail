"""Kinetic root selection for the growing scan and two damped test cases."""

import numpy as np
from scipy.optimize import root as _scipy_root

from bot.kinetic import kinetic_dispersion, most_unstable_kinetic


def damped_pair_candidates(k, u_b, eps):
    """Return all distinct Landau-damped roots within the training rectangle.

    Training-rectangle constraints (empirical from NN training range):
      xi_b.imag ∈ (-1.0, 0)    — damped but not extreme
      xi_b.real ∈ (0, 3.5)     — positive thermal speed
      omega.imag < -0.03       — clearly damped, not noise
    """

    def res(x):
        D = kinetic_dispersion(x[0] + 1j * x[1], k, u_b, eps)
        return [D.real, D.imag]

    found = []
    for seed_re in np.linspace(0.3, 2.5, 15):
        for seed_im in np.linspace(-0.05, -0.80, 15):
            try:
                sol = _scipy_root(res, [seed_re, seed_im], method="hybr")
                if not sol.success:
                    continue
                w = sol.x[0] + 1j * sol.x[1]
                if abs(kinetic_dispersion(w, k, u_b, eps)) > 1e-6:
                    continue
                if w.imag > -0.03:
                    continue
                xi_b = (w - k * u_b) / k
                if xi_b.imag < -1.0 or xi_b.imag >= 0:
                    continue
                if xi_b.real < 0 or xi_b.real > 3.5:
                    continue
                # Deduplicate: keep if not already in list (|Δω| > 0.01)
                if all(abs(w - wf) > 0.01 for wf in found):
                    found.append(w)
            except Exception:
                pass
    return sorted(found, key=lambda w: w.imag)  # most-damped first


def kinetic_peak(
    u_b: float,
    eps: float,
    k_lo: float = 0.10,
    k_hi: float = 1.20,
    n_k: int = 80,
    N_v: int = 96,
    V: float = 6.0,
    omega_re_hi: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Scan k via s-space eigenvalue solver; return (ks, omega(k), peak idx).

    For each k pick the most-unstable Langmuir-like mode in Re(omega) in
    [0.3, omega_re_hi].  omega_re_hi defaults to 1.5, not the wider 3.0 used
    prior to this fix: at some (u_b, eps, k) -- verified at u_b=7,
    k~0.34-0.38 -- a window that wide lets in a numerically-spurious
    high-frequency branch (Re(omega)~3.0-3.05) that outgrows the true
    fundamental Langmuir mode (Re(omega) stays ~0.7-1.05 across every
    (u_b, eps) checked in this module), corrupting the reported k* and
    omega there (e.g. the ~38%/~400% outlier cells this produced in
    fig_hp_beta_sweep.png before this fix, and propagating into any other
    sweep that calls kinetic_peak).  A Newton-continuation alternative
    (track one branch in k from a single seed, rather than independently
    re-selecting "most unstable in window" at each k) was tried and
    rejected: it fixed u_b=7 but broke other points whose true growing
    peak does not connect smoothly back to the low-k seed (e.g.
    u_b=3,eps=0.05 and u_b=4,eps=0.02 landed on k*=k_lo with zero growth).
    """
    ks = np.linspace(k_lo, k_hi, n_k)
    ws = np.empty(n_k, dtype=complex)
    for i, k in enumerate(ks):
        ws[i] = most_unstable_kinetic(k, u_b, eps, N_v=N_v, V=V, omega_re_hi=omega_re_hi)
    i_max = int(np.argmax(ws.imag))
    return ks, ws, i_max
