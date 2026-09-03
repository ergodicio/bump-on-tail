"""M4' fixed rational closure: AAA-fit multi-pole "absorbing pad" for slab ITG.

Per dressed_pole_closure_handoff.md, ADDENDUM 2 (M4'): instead of a
state-dependent (fitted) pole location with linear weights [the M3'
approach, found impractical], use a FIXED support (poles from an offline
AAA fit) with the state entering only linearly. The closure becomes a
constant linear map -- superposition holds exactly, K-K/UHP analyticity
holds by construction (checked once, offline), and the whole system is
expm-compatible (no adaptive integrator, no runtime NN, no gates).

Scope: this fits AAA directly to the ITG-SPECIFIC response
U_0/phi = -R_kinetic(zeta; zeta_star, eta) (option B in the handoff doc's
Step 1 note), rather than the universal gradient-free Z'(zeta) with drive
kept in explicit sigma_n source terms (option A, the doc's stated
preference). Option A requires appending the pad to a truncated physical
moment ladder ("coupled to the chain end") -- a genuinely different, more
involved construction not implemented here. Option B is exact for the
tested (zeta_star, eta) but the fit is specific to that drive (not
reusable across parameter scans without refitting) -- acceptable since
AAA fitting is milliseconds (confirmed below), matching the doc's own
fallback note ("re-running AAA per background -- test that first").

Because quasineutrality for ITG is purely algebraic (phi = U_0/tau, no
separate field ODE), the pad alone is a complete, self-contained closed
system: no separate U_1, U_2, U_3 moments are needed at all.

State representation
---------------------
U_0(zeta)/phi(zeta) = sum_j c_j/(zeta - p_j)  (AAA's partial-fraction fit)
Each pole contributes an auxiliary state w_j with
    dw_j/dt = -i p_j w_j - i c_j phi,      phi = (sum_j w_j) / tau
so that, for a pure exp(-i*zeta*t) drive, w_j -> c_j*phi/(zeta-p_j) exactly
and U_0 = sum_j w_j reproduces the AAA rational function.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import AAA

from bot.slab_itg import R_kinetic, manifold_Un_over_phi
from bot.closures.pade import maxwellian_moment


@dataclass
class AAAPad:
    """Fixed pole/residue support for the ITG U_0/phi response at (zeta_star, eta)."""
    zeta_star: float
    eta: float
    poles: np.ndarray     # (r,) complex
    residues: np.ndarray  # (r,) complex
    n_pruned: int          # number of UHP/Froissart poles discarded (Step 2)
    max_rel_err: float     # validation error on the real-axis grid used to fit


def fit_aaa_pad_itg(zeta_star: float, eta: float,
                    xi_range: tuple[float, float] = (-8.0, 8.0),
                    n_train: int = 4001, max_terms: int = 21,
                    rtol: float = 1e-12) -> AAAPad:
    """Step 1+2: AAA-fit U_0/phi = -R_kinetic(zeta; zeta_star, eta) on the
    real axis, then prune any pole that lands outside the lower half-plane
    (non-causal / Froissart artifact) and re-solve residues by linear
    least-squares against the retained poles.
    """
    xi = np.linspace(*xi_range, n_train).astype(complex)
    target = -R_kinetic(xi, zeta_star, eta)

    aaa = AAA(xi, target, max_terms=max_terms, rtol=rtol)
    poles = aaa.poles()
    residues = aaa.residues()

    bad = poles.imag >= 0.0
    n_pruned = int(np.sum(bad))
    if n_pruned:
        poles = poles[~bad]
        # Re-solve residues by linear least squares on the retained poles
        # (see module docstring / handoff Step 2: "re-solve the residues by
        # linear least squares against the sample values on the retained
        # pole set").
        basis = 1.0 / (xi[:, None] - poles[None, :])
        residues, *_ = np.linalg.lstsq(basis, target, rcond=None)

    # Validation
    test = np.linspace(xi_range[0] * 0.9, xi_range[1] * 0.9, 4 * n_train // 5).astype(complex)
    test_target = -R_kinetic(test, zeta_star, eta)
    approx = sum(residues[j] / (test - poles[j]) for j in range(len(poles)))
    rel_err = np.abs(approx - test_target) / np.maximum(np.abs(test_target), 1e-12)
    max_rel_err = float(np.nanmax(rel_err))

    return AAAPad(zeta_star=zeta_star, eta=eta, poles=poles, residues=residues,
                  n_pruned=n_pruned, max_rel_err=max_rel_err)


def aaa_pad_system(pad: AAAPad, tau: float) -> np.ndarray:
    """r x r matrix A for d/dt w = A w  (phi = sum(w)/tau folded in)."""
    r = len(pad.poles)
    A = np.diag(-1j * pad.poles).astype(complex)
    A += -1j * np.outer(pad.residues, np.ones(r)) / tau
    return A


def aaa_pad_ic(pad: AAAPad, zeta0: complex, amp: float = 1e-3) -> np.ndarray:
    """Steady-state single-frequency IC: w_j(0) = c_j*amp/(zeta0 - p_j).

    This is the pad's own best representation of "a pure mode at zeta0":
    sum_j w_j(0) = amp * sum_j c_j/(zeta0-p_j) = amp * (AAA fit of U_0/phi
    at zeta0), matching U_0(0) = amp to the fit's accuracy at zeta0.
    """
    return pad.residues * amp / (zeta0 - pad.poles)


def _linear_rhs(t, y_real, A):
    n = A.shape[0]
    y = y_real[:n] + 1j * y_real[n:]
    dy = A @ y
    return np.concatenate([dy.real, dy.imag])


def aaa_pad_evolve(pad: AAAPad, tau: float, t_grid: np.ndarray,
                   w0: np.ndarray, method: str = "eigen",
                   rtol: float = 1e-9, atol: float = 1e-12) -> dict:
    """Evolve the fixed-pole pad system; returns dict with t, U0, w, phi."""
    A = aaa_pad_system(pad, tau)
    n = A.shape[0]
    if method == "eigen":
        lam, V = np.linalg.eig(A)
        c = np.linalg.solve(V, w0)
        W = np.empty((n, len(t_grid)), dtype=complex)
        for i, t in enumerate(t_grid):
            W[:, i] = V @ (c * np.exp(lam * t))
    elif method == "rk45":
        y0_real = np.concatenate([w0.real, w0.imag])
        sol = solve_ivp(_linear_rhs, (t_grid[0], t_grid[-1]), y0_real,
                        t_eval=t_grid, args=(A,), method="RK45",
                        rtol=rtol, atol=atol)
        if not sol.success:
            print(f"  aaa_pad_evolve WARNING: {sol.message}")
        m = len(sol.t)
        W = sol.y[:n, :m] + 1j * sol.y[n:, :m]
    else:
        raise ValueError(f"unknown method {method!r}")
    U0 = W.sum(axis=0)
    return {"t": t_grid, "U0": U0, "w": W, "phi": U0 / tau}


# ---------------------------------------------------------------------------
# "Ladder" variant: retains the physical U_0..U_{M-1} moments (exact
# sigma_n-driven dynamics, unchanged from HP/direct-beta) and uses the AAA
# poles only to CLOSE U_M, via a per-step fixed-pole least-squares
# projection -- not a re-fit of pole locations (that's M3', found
# impractical), just a linear solve against a FIXED basis every step.  This
# is what makes generic (non-eigenmode) ICs representable at all: the
# U_0-only pad above has no U_1, U_2 state and so cannot encode an IC where
# those are set independently of U_0 (e.g. the density-only IC used in
# fig_bot_vs_itg_closures.py).  With r poles >> M retained moments the
# per-step weight solve is underdetermined; np.linalg.lstsq returns the
# minimum-norm solution.
# ---------------------------------------------------------------------------

@dataclass
class AAAPadLadder:
    """Fixed-pole closure basis for retained moments U_0..U_{M-1} -> U_M."""
    zeta_star: float
    eta: float
    poles: np.ndarray   # (r,)
    Bm: np.ndarray      # (M, r): rows n=0..M-1 of manifold_Un_over_phi(pole)
    mM: np.ndarray      # (r,): row n=M


def fit_aaa_pad_ladder(zeta_star: float, eta: float, M: int = 3,
                       **fit_kwargs) -> AAAPadLadder:
    """Fit the AAA pad (same poles as fit_aaa_pad_itg) and build the
    ITG-driven manifold basis (manifold_Un_over_phi, NOT the gradient-free
    basis_moments -- the poles are meant to stand in for candidate ITG
    eigenmodes of THIS (zeta_star, eta), same convention as itg_evolve's
    r=1 dressed-pole fit) for closing U_0..U_{M-1} -> U_M.
    """
    pad = fit_aaa_pad_itg(zeta_star, eta, **fit_kwargs)
    basis = np.array([manifold_Un_over_phi(p, zeta_star, eta, n_max=M)
                      for p in pad.poles])   # (r, M+1)
    Bm = basis[:, :M].T   # (M, r)
    mM = basis[:, M]      # (r,)
    return AAAPadLadder(zeta_star=zeta_star, eta=eta, poles=pad.poles,
                        Bm=Bm, mM=mM)


def _sigma(n: int, zeta_star: float, eta: float) -> float:
    M = maxwellian_moment
    return zeta_star * (M(n) + eta * (M(n + 2) - 0.5 * M(n))) - M(n + 1)


def _aaa_ladder_rhs(t, y_real, zeta_star, eta, tau, ladder):
    M = ladder.Bm.shape[0]
    U = y_real[:M] + 1j * y_real[M:]
    phi = U[0] / tau
    w, *_ = np.linalg.lstsq(ladder.Bm, U, rcond=None)
    U_M = ladder.mM @ w
    dU = np.empty(M, dtype=complex)
    for n in range(M - 1):
        dU[n] = -1j * U[n + 1] + 1j * phi * _sigma(n, zeta_star, eta)
    dU[M - 1] = -1j * U_M + 1j * phi * _sigma(M - 1, zeta_star, eta)
    return np.concatenate([dU.real, dU.imag])


def aaa_pad_ladder_evolve(zeta_star: float, eta: float, tau: float,
                          t_grid: np.ndarray, y0: np.ndarray,
                          ladder: AAAPadLadder,
                          rtol: float = 1e-9, atol: float = 1e-12) -> dict:
    """Evolve the ladder system (retained U_0..U_{M-1}, AAA-pad-closed U_M).

    RK45 only -- the closure is state-dependent (a per-step linear solve),
    so unlike aaa_pad_evolve this is not a fixed matrix and has no
    matrix-exponential shortcut.
    """
    M = ladder.Bm.shape[0]
    y0_real = np.concatenate([y0.real, y0.imag])
    sol = solve_ivp(_aaa_ladder_rhs, (t_grid[0], t_grid[-1]), y0_real,
                    t_eval=t_grid, args=(zeta_star, eta, tau, ladder),
                    method="RK45", rtol=rtol, atol=atol)
    if not sol.success:
        print(f"  aaa_pad_ladder_evolve WARNING: {sol.message}")
    m = len(sol.t)
    Y = sol.y[:M, :m] + 1j * sol.y[M:, :m]
    out = {"t": sol.t, "phi": Y[0] / tau}
    for n in range(M):
        out[f"U{n}"] = Y[n]
    return out
