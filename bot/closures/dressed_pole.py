"""Dressed-pole (Case-van Kampen) multimode closure — M1/M2.

See dressed_pole_closure_handoff.md.  Instead of estimating a single
frequency zeta = U_1/U_0 and evaluating an exact response function there
(direct-beta/alpha), represent the perturbed distribution as a sum of
"dressed pole" velocity-space basis functions

    f_1(s) = sum_j w_j f0'(s)/(s - zeta_j) + g(s)

Each basis function's moment sequence is exactly analytic:

    m_n(zeta) = (v_ts/n_s) int ds s^n f0'(s)/(s - zeta)

and satisfies the two-term recursion (M_k = int s^k F0 ds, M_k = 0 for k<0)

    m_0(zeta) = Z'(zeta)
    m_n(zeta) = zeta*m_{n-1}(zeta) - (n-1)*M_{n-2}

m_0, m_2/m_0, m_3/m_0 reproduce bot.closures.u2.Zprime/beta and
bot.closures.inference.alpha exactly (verified to 1e-16) -- these basis
moments ARE the U_n(xi_b=zeta, Phi=1) sequence already used throughout the
repo; the new content here is fitting (w_j, zeta_j) to the RETAINED
moments of the actual (possibly multi-mode / off-manifold) state, rather
than reading zeta off a single ratio.

r=1 (this module): closes the hierarchy with a single dressed pole, fit
by *overdetermined* nonlinear least squares when N >= 2 moments are
retained (VARPRO-separable: linear in w, nonlinear only in zeta). On a
single true eigenmode this recovers direct-beta/alpha's (w, zeta) exactly
with zero residual; off-manifold, the fit residual is a built-in
settledness/contamination diagnostic that direct-beta has no equivalent
of. r>=2 (multi-pole, the M3 "payoff") is not implemented in this module.

For the driven slab-ITG problem, the basis moments are
bot.slab_itg.manifold_Un_over_phi(zeta, zeta_star, eta) (same object,
generalized to the ITG drive) -- reused directly here, not reimplemented.

VERIFIED (M1/M2), both against the paper's own benchmark cases:
  - M1 unit test: synthetic single eigenmode's moment sequence recovered
    to machine precision (|dzeta|, |dw|, residual ~ 1e-14..1e-16).
  - M2, BoT (u_b=4, eps=0.02, k=0.295) and slab ITG (zeta_*=1, tau=1,
    eta=10), each started from an eigenmode-projected (single-mode) IC
    with continuation warm-starting (see below): matches the kinetic
    growth rate to ~1e-6-1e-10 % -- effectively exact, consistent with
    direct-beta/alpha's own exactness on a settled single mode.

IMPORTANT LIMITATION FOUND (not a bug -- exactly what motivates M3):
  fit_r1's zeta is found by local nonlinear least squares (Levenberg-
  Marquardt), which can converge to a DIFFERENT local minimum for two
  nearby states if the state's dominant content is genuinely shifting
  between two comparable underlying roots (verified: a generic density-
  only IC for the zeta_*=1,tau=1,eta=10 ITG case produces a fitted
  zeta(t) that jumps discontinuously, e.g. -2.24-1.11j -> +0.39+2.10j
  over a single 0.1-time-unit step, even though the physical state
  U_0,U_1,U_2 evolves smoothly).  This is r=1 fundamentally unable to
  represent a state with significant weight on >1 root -- exactly the
  failure mode M3 (r>=2) is meant to fix, not a fitting bug.  Practical
  consequences for anyone using this module before M3 exists:
    1. A naive adaptive ODE integrator (scipy solve_ivp/RK45) treats
       that jump as a near-singularity and pathologically collapses its
       step size -- observed hangs of hours of CPU with zero progress
       on the raw generic-IC problem.  Always warm-start fit_r1/
       dressed_pole_r1_next's zeta0 with the PREVIOUS step's converged
       zeta (continuation) rather than recomputing zeta0=U_1/U_0 fresh
       every call -- this delays but does NOT eliminate the jump when
       the state truly is multi-root (confirmed: continuation moved the
       jump from t~0.55 to t~0.85 in the same test, not away).
    2. r=1 (this module) should only be trusted on states that are
       genuinely close to a single eigenmode (e.g. eigenmode-projected
       ICs, or late-time settled trajectories) -- use the fit residual
       as the gate, and expect trouble (or implement r>=2) for anything
       with real multi-root content, including generic/naive ICs.

M3-pre (implemented): dressed_pole_r1_gated_next blends continuously toward
the linear HP closure (hp_U3_fallback) with a smooth (never hard-switched)
weight built from the relative fit residual -- see its docstring.  This
does NOT remove the underlying fit's jump (confirmed still present, just
relocated in time under blending), but it stops that jump from reaching
the closed system's right-hand side: verified on the exact case that hung
above (itg_evolve, zeta_*=1, tau=1, eta=10, generic density IC) -- runs to
completion (RK45, no hang), blend peaks ~0.78 during the multi-root
transient and decays to ~0 once the state settles onto the true root,
final growth rate matches kinetic to 4e-10 % (same as the eigenmode-IC
M2 result, now reached from the previously-hanging generic IC).  A
"promote-warnings-to-errors" sweep of this run found no reproducible
Z(zeta) conditioning failure along the fitted zeta(t) trajectory (which
stays at modest |zeta| <~ 3.3 even through the jump); an intermittent
"invalid value" RuntimeWarning traces to the LM optimizer's exploratory
finite-difference Jacobian probing far-out trial zeta, not the converged
fit, and does not affect the result.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from bot.closures.u2 import Zprime

SQRT_PI = float(np.sqrt(np.pi))

# Gaussian F0(s) = exp(-s^2)/sqrt(pi) moments M_0..M_8 (M_odd = 0)
_MAXW_M = (1.0, 0.0, 0.5, 0.0, 0.75, 0.0, 1.875, 0.0, 6.5625)


# ---------------------------------------------------------------------------
# M1: basis moments (gradient-free / bump-on-tail)
# ---------------------------------------------------------------------------

def basis_moments(zeta: complex, N: int) -> np.ndarray:
    """m_0(zeta)..m_N(zeta) for the gradient-free (BoT) Maxwellian.

    m_0(zeta) = Z'(zeta);  m_n(zeta) = zeta*m_{n-1}(zeta) - (n-1)*M_{n-2}.
    """
    if N + 2 > len(_MAXW_M):
        raise ValueError(f"N too large for tabulated Maxwellian moments "
                         f"(N <= {len(_MAXW_M) - 2})")
    m = np.empty(N + 1, dtype=complex)
    m[0] = Zprime(zeta)
    for n in range(1, N + 1):
        M_nm2 = _MAXW_M[n - 2] if n - 2 >= 0 else 0.0
        m[n] = zeta * m[n - 1] - (n - 1) * M_nm2
    return m


# ---------------------------------------------------------------------------
# M2: r=1 (single dressed pole), separable / VARPRO nonlinear least squares
# ---------------------------------------------------------------------------

def _w_of_zeta(U: np.ndarray, zeta: complex, N: int, basis_fn):
    """Linear least-squares weight given zeta (VARPRO inner solve)."""
    m = np.asarray(basis_fn(zeta, N), dtype=complex)
    denom = np.vdot(m, m)
    w = np.vdot(m, U) / denom if abs(denom) > 1e-300 else 0.0 + 0.0j
    return w, m


def fit_r1(U: np.ndarray, zeta0: complex | None = None, basis_fn=basis_moments):
    """Overdetermined single-pole fit: U_n ~= w*m_n(zeta), n=0..len(U)-1.

    Separable nonlinear least squares: w solved linearly given zeta (see
    _w_of_zeta); only zeta (2 real unknowns) is optimized nonlinearly.
    len(U)=2 (N=1) is exactly-determined and recovers direct-beta's
    zeta=U_1/U_0 identically; len(U)>=3 is overdetermined.

    basis_fn(zeta, N) -> array of m_0..m_N defaults to the gradient-free
    (BoT) basis_moments above; pass a driven basis (e.g. a closure over
    bot.slab_itg.manifold_Un_over_phi(zeta, zeta_star, eta, N)) for ITG.

    Returns (w, zeta, residual) with residual = ||U - w*m(zeta)||_2.
    """
    N = len(U) - 1
    U = np.asarray(U, dtype=complex)
    if zeta0 is None:
        zeta0 = complex(U[1] / U[0]) if abs(U[0]) > 1e-14 else 1.0 + 0.1j

    def resid(x):
        zeta = complex(x[0], x[1])
        w, m = _w_of_zeta(U, zeta, N, basis_fn)
        r = U - w * m
        return np.concatenate([r.real, r.imag])

    sol = least_squares(resid, [zeta0.real, zeta0.imag], method="lm",
                        xtol=1e-14, ftol=1e-14, gtol=1e-14)
    zeta = complex(sol.x[0], sol.x[1])
    w, m = _w_of_zeta(U, zeta, N, basis_fn)
    residual = float(np.linalg.norm(U - w * m))
    return w, zeta, residual


# ---------------------------------------------------------------------------
# Closure interface: predict the next moment given retained moments U_0..U_N
# ---------------------------------------------------------------------------

def dressed_pole_r1_next(U: np.ndarray, zeta0: complex | None = None,
                         basis_fn=basis_moments):
    """Close the hierarchy: fit r=1 to U_0..U_N, predict U_{N+1}.

    Returns (U_next, zeta, residual). Use `residual` as a settledness gate
    (large residual = state is not well-described by a single pole, e.g.
    a genuine superposition or continuum content -- direct-beta has no
    equivalent diagnostic and will silently return a compromise value).
    """
    N = len(U) - 1
    w, zeta, residual = fit_r1(U, zeta0, basis_fn=basis_fn)
    m_next = np.asarray(basis_fn(zeta, N + 1))[-1]
    return w * m_next, zeta, residual


# ---------------------------------------------------------------------------
# M3-pre: residual gate + continuous HP-blend fallback for the N=2 -> U_3 case
# ---------------------------------------------------------------------------
# Makes r=1 fail gracefully instead of hanging an adaptive integrator (see
# module docstring "KNOWN FAILURE"): when the overdetermined fit's residual
# is large (state has real weight on >1 root -- r=1 cannot represent it),
# blend continuously toward the linear HP closure rather than trusting a
# fit whose zeta(t) may be discontinuously jumping between local minima.
# Blend is a smooth (never a hard-switch) function of the RELATIVE residual
# ||U - w*m(zeta)|| / ||U||, so the closed system's RHS stays continuous in
# time even while the underlying fit is not.

def hp_U3_fallback(U0: complex, U1: complex, U2: complex,
                   Gamma: float = 3.0, chi1: float = 2.0 / SQRT_PI) -> complex:
    """HP's linear N=3 closure U_3 = (Gamma/2) U_1 - i(chi1/2)(2 U_2 - U_0).

    Same formula for BoT and slab ITG (the drive only enters the lower
    moment equations, not this closure map) -- see bot.slab_itg.CHI1_HP.
    """
    return (Gamma / 2.0) * U1 - 1j * (chi1 / 2.0) * (2.0 * U2 - U0)


def dressed_pole_r1_gated_next(U: np.ndarray, zeta0: complex | None = None,
                               basis_fn=basis_moments, rel_tol: float = 0.3):
    """r=1 fit for N=2 (U_0,U_1,U_2 -> U_3) with a continuous HP-blend gate.

    blend = rr^2/(rr^2 + rel_tol^2) in [0, 1), rr = residual/||U|| -- smooth
    in rr, blend(0)=0 (trust the fit), blend->1 as rr grows past rel_tol
    (trust HP).  U_next = (1-blend)*U_dressed_pole + blend*U_HP.

    Returns (U_next, zeta, residual, blend).
    """
    if len(U) != 3:
        raise ValueError("dressed_pole_r1_gated_next is specialized to "
                         "N=2 (U_0, U_1, U_2) -> U_3")
    U = np.asarray(U, dtype=complex)
    w, zeta, residual = fit_r1(U, zeta0, basis_fn=basis_fn)
    m3 = np.asarray(basis_fn(zeta, 3))[-1]
    U_dp = w * m3
    U_hp = hp_U3_fallback(U[0], U[1], U[2])
    rr = residual / (np.linalg.norm(U) + 1e-300)
    blend = rr**2 / (rr**2 + rel_tol**2)
    U_next = (1.0 - blend) * U_dp + blend * U_hp
    return U_next, zeta, residual, blend


# ---------------------------------------------------------------------------
# Time-domain evolve helpers (N=3 state, gated r=1 closes U_3), both problems
# ---------------------------------------------------------------------------
# Continuation (warm-starting zeta0 from the previous accepted step) plus the
# gate above is what makes this usable inside an adaptive integrator at all
# -- see module docstring "KNOWN FAILURE" / M3-pre.

def itg_evolve(zeta_star: float, eta: float, tau: float,
               t_grid: np.ndarray, y0: np.ndarray,
               rel_tol: float = 0.3, zeta0: complex = 1.0 + 0.1j,
               rtol: float = 1e-9, atol: float = 1e-12) -> dict:
    """Slab-ITG N=3 system (U_0,U_1,U_2), U_3 closed by the gated r=1 fit.

    Field equations identical to bot.slab_itg._direct_alpha_rhs; only the
    U_3 closure differs (gated dressed-pole instead of alpha_itg(zeta_hat)).

    Returns dict with t, U0, U1, U2, zeta (fitted, per accepted step),
    residual, blend.
    """
    from scipy.integrate import solve_ivp
    from bot.slab_itg import manifold_Un_over_phi

    def basis_fn(zeta, N):
        return manifold_Un_over_phi(zeta, zeta_star, eta, n_max=N)

    zeta_holder = [complex(zeta0)]
    log = {"t": [], "zeta": [], "residual": [], "blend": []}

    def rhs(t, y_real):
        y = y_real[:3] + 1j * y_real[3:]
        U0, U1, U2 = y
        phi = U0 / tau
        if abs(U0) < 1e-13:
            U3 = 0.0 + 0.0j
        else:
            U3, zeta, resid, blend = dressed_pole_r1_gated_next(
                np.array([U0, U1, U2]), zeta0=zeta_holder[0],
                basis_fn=basis_fn, rel_tol=rel_tol)
            zeta_holder[0] = zeta
            log["t"].append(t); log["zeta"].append(zeta)
            log["residual"].append(resid); log["blend"].append(blend)
        dU0 = -1j * U1 + 1j * zeta_star * phi
        dU1 = -1j * U2 - 0.5j * phi
        dU2 = -1j * U3 + 1j * zeta_star * (1.0 + eta) * 0.5 * phi
        dy = np.array([dU0, dU1, dU2])
        return np.concatenate([dy.real, dy.imag])

    y0_real = np.concatenate([y0.real, y0.imag])
    sol = solve_ivp(rhs, (t_grid[0], t_grid[-1]), y0_real, t_eval=t_grid,
                    method="RK45", rtol=rtol, atol=atol)
    if not sol.success:
        print(f"  itg_evolve WARNING: {sol.message}")
    n = len(sol.t)
    Y = sol.y[:3, :n] + 1j * sol.y[3:, :n]
    return {"t": sol.t, "U0": Y[0], "U1": Y[1], "U2": Y[2],
            "log_t": np.array(log["t"]), "log_zeta": np.array(log["zeta"]),
            "log_residual": np.array(log["residual"]),
            "log_blend": np.array(log["blend"])}


def bot_evolve(k: float, u_b: float, eps: float,
              t_grid: np.ndarray, y0: np.ndarray,
              rel_tol: float = 0.3, zeta0: complex = 1.0 + 0.1j,
              rtol: float = 1e-9, atol: float = 1e-12) -> dict:
    """BoT N=3 system (u,E,U_0,U_1,U_2), U_3 closed by the gated r=1 fit.

    Field equations identical to bot.closed_loop._direct_alpha_rhs; only the
    U_3 closure differs.  Returns dict with t, E, U0, U1, U2, and the same
    fit-diagnostic logs as itg_evolve.
    """
    from scipy.integrate import solve_ivp

    zeta_holder = [complex(zeta0)]
    log = {"t": [], "zeta": [], "residual": [], "blend": []}

    def rhs(t, y_real):
        y = y_real[:5] + 1j * y_real[5:]
        u, E, U0, U1, U2 = y
        if abs(U0) < 1e-13:
            U3 = 0.0 + 0.0j
        else:
            U3, zeta, resid, blend = dressed_pole_r1_gated_next(
                np.array([U0, U1, U2]), zeta0=zeta_holder[0],
                rel_tol=rel_tol)
            zeta_holder[0] = zeta
            log["t"].append(t); log["zeta"].append(zeta)
            log["residual"].append(resid); log["blend"].append(blend)
        du = -E
        dE = u - eps * (u_b * U0 + U1)
        dU0 = -1j * k * u_b * U0 - 1j * k * U1
        dU1 = -1j * k * u_b * U1 - 1j * k * U2 + 1.0 * E
        dU2 = -1j * k * u_b * U2 - 1j * k * U3
        dy = np.array([du, dE, dU0, dU1, dU2])
        return np.concatenate([dy.real, dy.imag])

    y0_real = np.concatenate([y0.real, y0.imag])
    sol = solve_ivp(rhs, (t_grid[0], t_grid[-1]), y0_real, t_eval=t_grid,
                    method="RK45", rtol=rtol, atol=atol)
    if not sol.success:
        print(f"  bot_evolve WARNING: {sol.message}")
    n = len(sol.t)
    Y = sol.y[:5, :n] + 1j * sol.y[5:, :n]
    return {"t": sol.t, "u": Y[0], "E": Y[1], "U0": Y[2], "U1": Y[3],
            "U2": Y[4],
            "log_t": np.array(log["t"]), "log_zeta": np.array(log["zeta"]),
            "log_residual": np.array(log["residual"]),
            "log_blend": np.array(log["blend"])}
