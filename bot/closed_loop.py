"""Time-domain comparison of kinetic / Padé / inference closures.

For a fixed Fourier mode k:
  * kinetic truth: integrate the s-space discretized Vlasov-Poisson
    (state = (u, E, F[s_0], ..., F[s_{N_v-1}])) via matrix exponential.
  * Padé closure: integrate the linear 3-moment fluid system
    (state = (u, E, U_0, U_1, U_2)) via matrix exponential.
  * Inference closure: integrate the nonlinear fluid system (closure
    provides U_3 = alpha(xi_hat) U_0 at each timestep) via RK45.

Two initial conditions:
  * Langmuir-projected: aligned with the unstable eigenvector so spurious
    closure modes are not excited.
  * Generic delta-E: u = U_n = 0, E = small. Excites everything in the
    fluid system; this is where the BoT paper §4 NN closure blew up.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from bot.closures.inference import alpha, model_xi
from bot.closures.naive_nn import (load as load_naive,
                                    naive_U3_over_U0, naive_predict_alpha)
from bot.train_simdata import load as load_simdata
from bot.closures.pade import maxwellian_moment, pade_coefficients
from bot.closures.u2 import beta as _u2_beta, model_beta as _u2_model_beta, MLP_u2
from bot.fluid import fluid_system
from bot.kinetic import (kinetic_system_sspace, most_unstable_kinetic, s_grid,
                          SQRT_PI)
from bot.train_inference import load as load_inference


# ----- kinetic time evolution (matrix exponential) ------------------------

def kinetic_evolve(k: float, u_b: float, eps: float,
                    t_grid: np.ndarray, y0: np.ndarray,
                    N_v: int = 96, V: float = 6.0) -> np.ndarray:
    """Return E(t) for the s-space kinetic system."""
    A = kinetic_system_sspace(k, u_b, eps, N_v, V)
    eigvals, evecs = np.linalg.eig(A)
    coefs = np.linalg.solve(evecs, y0)
    E_t = np.empty(len(t_grid), dtype=complex)
    for i, ti in enumerate(t_grid):
        y = evecs @ (coefs * np.exp(eigvals * ti))
        E_t[i] = y[1]
    return E_t


# ----- Padé time evolution (linear ODE, matrix exponential) --------------

def pade_evolve(k: float, u_b: float, eps: float, a: np.ndarray,
                 t_grid: np.ndarray, y0: np.ndarray) -> np.ndarray:
    """Return E(t) for the closed fluid system with Padé closure a."""
    A = fluid_system(k, u_b, eps, a)
    eigvals, evecs = np.linalg.eig(A)
    coefs = np.linalg.solve(evecs, y0)
    E_t = np.empty(len(t_grid), dtype=complex)
    for i, ti in enumerate(t_grid):
        y = evecs @ (coefs * np.exp(eigvals * ti))
        E_t[i] = y[1]
    return E_t


# ----- inference time evolution (nonlinear ODE, RK45) --------------------

def _inference_rhs(t, y_real, k: float, u_b: float, eps: float, model):
    """Real-stacked RHS for the inference-closure fluid system.

    y_real = [Re u, Im u, Re E, Im E, Re U_0, Im U_0, Re U_1, Im U_1, Re U_2, Im U_2]
    """
    y = y_real[:5] + 1j * y_real[5:]
    u, E, U0, U1, U2 = y
    # closure
    if abs(U0) < 1e-15:
        U3 = 0.0 + 0.0j
    else:
        r1 = U1 / U0
        r2 = U2 / U0
        xi_hat = complex(model_xi(model, r1, r2))
        U3 = alpha(xi_hat) * U0
    # RHS
    du  = -E
    dE  = u - eps * (u_b * U0 + U1)
    dU0 = -1j * k * u_b * U0 - 1j * k * U1
    dU1 = -1j * k * u_b * U1 - 1j * k * U2 + 1.0 * E     # M_0 = 1
    dU2 = -1j * k * u_b * U2 - 1j * k * U3               # M_1 = 0
    dy = np.array([du, dE, dU0, dU1, dU2])
    return np.concatenate([dy.real, dy.imag])


def inference_evolve(k: float, u_b: float, eps: float, model,
                      t_grid: np.ndarray, y0: np.ndarray,
                      rtol: float = 1e-8, atol: float = 1e-10) -> np.ndarray:
    """Return E(t) for the fluid system with the inference closure."""
    y0_real = np.concatenate([y0.real, y0.imag])
    sol = solve_ivp(
        _inference_rhs, (t_grid[0], t_grid[-1]), y0_real,
        t_eval=t_grid, args=(k, u_b, eps, model),
        method="RK45", rtol=rtol, atol=atol,
    )
    if not sol.success:
        print(f"  inference_evolve WARNING: {sol.message}")
    y_complex = sol.y[:5] + 1j * sol.y[5:]
    return y_complex[1]   # E(t)


# ----- NAIVE closure time evolution ---------------------------------------

def _naive_rhs(t, y_real, k: float, u_b: float, eps: float, model):
    """RHS for naive closure (NN outputs alpha directly)."""
    y = y_real[:5] + 1j * y_real[5:]
    u, E, U0, U1, U2 = y
    if abs(U0) < 1e-15:
        U3 = 0.0 + 0.0j
    else:
        r1 = U1 / U0
        r2 = U2 / U0
        a_eff = complex(naive_predict_alpha(model, r1, r2))
        U3 = a_eff * U0
    du  = -E
    dE  = u - eps * (u_b * U0 + U1)
    dU0 = -1j * k * u_b * U0 - 1j * k * U1
    dU1 = -1j * k * u_b * U1 - 1j * k * U2 + 1.0 * E
    dU2 = -1j * k * u_b * U2 - 1j * k * U3
    dy = np.array([du, dE, dU0, dU1, dU2])
    return np.concatenate([dy.real, dy.imag])


def naive_evolve(k: float, u_b: float, eps: float, model,
                  t_grid: np.ndarray, y0: np.ndarray,
                  rtol: float = 1e-8, atol: float = 1e-10) -> np.ndarray:
    """Return E(t) for the fluid system with the naive closure."""
    y0_real = np.concatenate([y0.real, y0.imag])
    sol = solve_ivp(
        _naive_rhs, (t_grid[0], t_grid[-1]), y0_real,
        t_eval=t_grid, args=(k, u_b, eps, model),
        method="RK45", rtol=rtol, atol=atol,
    )
    if not sol.success:
        print(f"  naive_evolve WARNING: {sol.message}")
    y_complex = sol.y[:5] + 1j * sol.y[5:]
    return y_complex[1]


# Sim-trained NN closure: uses the same architecture as naive but the model
# was trained on simulation trajectory data instead of closed-form xi-sampled
# data. Otherwise identical to naive.

def sim_evolve(k: float, u_b: float, eps: float, model,
                t_grid: np.ndarray, y0: np.ndarray,
                rtol: float = 1e-8, atol: float = 1e-10) -> np.ndarray:
    """Return E(t) using the simulation-trained NN closure."""
    return naive_evolve(k, u_b, eps, model, t_grid, y0, rtol, atol)


# ----- safeguard for Z'-based closures (beta, alpha, NN-of-xi) ------------
# Off the single-eigenmode manifold, U_0(t) can pass through a near-zero
# "node" from destructive interference between superposed modes with
# different complex frequencies (e.g. two-mode mixing ICs).  At that node
# U_1/U_0 races to large |Im|, where Z'(xi) grows like exp(Im(xi)^2) and
# overflows within a handful of RK45 steps, collapsing the adaptive step
# size ("Required step size is less than spacing between numbers").  The
# closure value itself stays finite (beta(xi) -> xi^2 as Z'(xi) -> inf), but
# the speed at which xi sweeps through huge values makes the ODE locally
# stiff.  Clipping |xi_hat| bounds this without affecting normal eigenmode
# dynamics (|xi_b| is O(1) for every physically relevant case in this repo).

_XI_RATIO_MAX = 12.0


def _safe_xi_ratio(U1: complex, U0: complex, U0_floor: float = 1e-15,
                    max_mag: float = _XI_RATIO_MAX) -> complex:
    """Return U_1/U_0, clipped in magnitude to avoid feeding a Z'-based
    closure an argument that blows up near a U_0 node."""
    if abs(U0) < U0_floor:
        return 0.0 + 0.0j
    xi = U1 / U0
    mag = abs(xi)
    if mag > max_mag:
        xi = xi * (max_mag / mag)
    return xi


# ----- DIRECT ALPHA closure (analytic, zero-parameter) --------------------
# Apply the exact per-mode formula alpha(xi) = xi^3 - xi/Z'(xi) at xi=U1/U0.
# This uses only r_1 = U1/U0 (ignoring U2 as a closure input), exactly as the
# N=2 direct-beta closure uses beta(U1/U0).  On a single eigenmode r_1 = xi_b
# so the closure is exact; off-manifold it commits a Jensen-type error.
# This is DISTINCT from the bilinear closure below, which uses r_1 * r_2.

def _direct_alpha_rhs(t, y_real, k: float, u_b: float, eps: float):
    y = y_real[:5] + 1j * y_real[5:]
    u, E, U0, U1, U2 = y
    U3 = U0 * alpha(_safe_xi_ratio(U1, U0))   # only r1; no r2 used
    du  = -E
    dE  = u - eps * (u_b * U0 + U1)
    dU0 = -1j * k * u_b * U0 - 1j * k * U1
    dU1 = -1j * k * u_b * U1 - 1j * k * U2 + 1.0 * E
    dU2 = -1j * k * u_b * U2 - 1j * k * U3
    dy = np.array([du, dE, dU0, dU1, dU2])
    return np.concatenate([dy.real, dy.imag])


def direct_alpha_evolve(k: float, u_b: float, eps: float,
                         t_grid: np.ndarray, y0: np.ndarray,
                         rtol: float = 1e-8, atol: float = 1e-10) -> np.ndarray:
    """Return E(t) with the direct alpha(U1/U0) closure — no NN, no r2."""
    y0_real = np.concatenate([y0.real, y0.imag])
    sol = solve_ivp(
        _direct_alpha_rhs, (t_grid[0], t_grid[-1]), y0_real,
        t_eval=t_grid, args=(k, u_b, eps),
        method="RK45", rtol=rtol, atol=atol,
    )
    if not sol.success:
        print(f"  direct_alpha_evolve WARNING: {sol.message}")
    y_complex = sol.y[:5] + 1j * sol.y[5:]
    return y_complex[1]


# ----- BILINEAR closure (analytic, zero-parameter) ------------------------
# Exact per-eigenmode closure from the moment recurrence with M_1 = 0:
#     U_3 = xi U_2 = (U_1/U_0) U_2  =>  U_3/U_0 = (U_1/U_0)(U_2/U_0)
# Equivalent to U_3 = U_1 U_2 / U_0. No NN, no Pade coefficients.

def _bilinear_rhs(t, y_real, k: float, u_b: float, eps: float):
    y = y_real[:5] + 1j * y_real[5:]
    u, E, U0, U1, U2 = y
    if abs(U0) < 1e-15:
        U3 = 0.0 + 0.0j
    else:
        U3 = U1 * U2 / U0
    du  = -E
    dE  = u - eps * (u_b * U0 + U1)
    dU0 = -1j * k * u_b * U0 - 1j * k * U1
    dU1 = -1j * k * u_b * U1 - 1j * k * U2 + 1.0 * E
    dU2 = -1j * k * u_b * U2 - 1j * k * U3
    dy = np.array([du, dE, dU0, dU1, dU2])
    return np.concatenate([dy.real, dy.imag])


def bilinear_evolve(k: float, u_b: float, eps: float,
                     t_grid: np.ndarray, y0: np.ndarray,
                     rtol: float = 1e-8, atol: float = 1e-10) -> np.ndarray:
    """Return E(t) with the hardcoded bilinear closure U_3 = U_1 U_2 / U_0."""
    y0_real = np.concatenate([y0.real, y0.imag])
    sol = solve_ivp(
        _bilinear_rhs, (t_grid[0], t_grid[-1]), y0_real,
        t_eval=t_grid, args=(k, u_b, eps),
        method="RK45", rtol=rtol, atol=atol,
    )
    if not sol.success:
        print(f"  bilinear_evolve WARNING: {sol.message}")
    y_complex = sol.y[:5] + 1j * sol.y[5:]
    return y_complex[1]


# ----- initial conditions -------------------------------------------------

def langmuir_ic_kinetic(k: float, u_b: float, eps: float, amp: float = 1e-3,
                         N_v: int = 96, V: float = 6.0) -> tuple[np.ndarray, complex]:
    """Most-unstable eigenvector of s-space kinetic system, normalised by E.

    Returns (y0_full_state, omega_eigval).
    """
    A = kinetic_system_sspace(k, u_b, eps, N_v, V)
    eigvals_lam, evecs = np.linalg.eig(A)
    omegas = 1j * eigvals_lam
    # filter for Langmuir-like
    mask = (omegas.real > 0.3) & (omegas.real < 3.0)
    cand_idx = np.where(mask)[0]
    j = cand_idx[int(np.argmax(omegas[mask].imag))]
    v = evecs[:, j]
    y0 = v * (amp / abs(v[1]))   # normalize so |E_0| = amp
    return y0, omegas[j]


def kinetic_ic_to_fluid(y0_kinetic: np.ndarray, k: float, u_b: float,
                         N_v: int = 96, V: float = 6.0) -> np.ndarray:
    """Project a kinetic state onto the fluid 5-d state by computing moments."""
    s, ds = s_grid(N_v, V)
    F = y0_kinetic[2:]
    U0 = ds * np.sum(F)
    U1 = ds * np.sum(s * F)
    U2 = ds * np.sum(s**2 * F)
    return np.array([y0_kinetic[0], y0_kinetic[1], U0, U1, U2], dtype=complex)


def generic_dE_ic(amp: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Generic δE perturbation: u = U_n = 0, E = amp."""
    y0_fluid = np.array([0.0, amp, 0.0, 0.0, 0.0], dtype=complex)
    return y0_fluid


def generic_dE_ic_kinetic(amp: float = 1e-3, N_v: int = 96) -> np.ndarray:
    y0 = np.zeros(2 + N_v, dtype=complex)
    y0[1] = amp
    return y0


# ----- main: comparison at canonical (5, 0.05, k=0.24) -------------------

def run_comparison(u_b: float = 5.0, eps: float = 0.05, k: float = 0.24,
                    t_end: float = 80.0, n_t: int = 801,
                    save: bool = True) -> dict:
    print(f"=== Time-domain comparison at u_b={u_b}, eps={eps}, k={k} ===")
    t_grid = np.linspace(0.0, t_end, n_t)
    model_inf = load_inference()
    model_naive = load_naive()
    model_sim = load_simdata(broad=True)

    # kinetic Langmuir mode + IC
    y0_kin_lan, omega_kin = langmuir_ic_kinetic(k, u_b, eps)
    print(f"  kinetic Langmuir-like ω = {omega_kin:.4f}")

    y0_fluid_lan = kinetic_ic_to_fluid(y0_kin_lan, k, u_b)

    # generic δE IC
    y0_fluid_gen = generic_dE_ic(amp=1e-3)
    y0_kin_gen = generic_dE_ic_kinetic(amp=1e-3)

    a_pade = pade_coefficients(3)

    print("  integrating kinetic (Langmuir IC)...")
    E_kin_lan = kinetic_evolve(k, u_b, eps, t_grid, y0_kin_lan)
    print("  integrating kinetic (generic IC)...")
    E_kin_gen = kinetic_evolve(k, u_b, eps, t_grid, y0_kin_gen)
    print("  integrating Padé (Langmuir IC)...")
    E_pade_lan = pade_evolve(k, u_b, eps, a_pade, t_grid, y0_fluid_lan)
    print("  integrating Padé (generic IC)...")
    E_pade_gen = pade_evolve(k, u_b, eps, a_pade, t_grid, y0_fluid_gen)
    print("  integrating inference (Langmuir IC)...")
    E_inf_lan = inference_evolve(k, u_b, eps, model_inf, t_grid, y0_fluid_lan)
    print("  integrating inference (generic IC)...")
    E_inf_gen = inference_evolve(k, u_b, eps, model_inf, t_grid, y0_fluid_gen)
    print("  integrating naive (Langmuir IC)...")
    E_nai_lan = naive_evolve(k, u_b, eps, model_naive, t_grid, y0_fluid_lan)
    print("  integrating naive (generic IC)...")
    E_nai_gen = naive_evolve(k, u_b, eps, model_naive, t_grid, y0_fluid_gen)
    print("  integrating sim-trained (Langmuir IC)...")
    E_sim_lan = sim_evolve(k, u_b, eps, model_sim, t_grid, y0_fluid_lan)
    print("  integrating sim-trained (generic IC)...")
    E_sim_gen = sim_evolve(k, u_b, eps, model_sim, t_grid, y0_fluid_gen)

    results = {
        "t": t_grid, "u_b": u_b, "eps": eps, "k": k,
        "omega_kin": omega_kin,
        "E_kin_lan": E_kin_lan, "E_kin_gen": E_kin_gen,
        "E_pade_lan": E_pade_lan, "E_pade_gen": E_pade_gen,
        "E_inf_lan":  E_inf_lan,  "E_inf_gen":  E_inf_gen,
        "E_nai_lan":  E_nai_lan,  "E_nai_gen":  E_nai_gen,
        "E_sim_lan":  E_sim_lan,  "E_sim_gen":  E_sim_gen,
    }
    if save:
        out = Path(__file__).resolve().parent / "runs" / "time_domain.npz"
        np.savez_compressed(out, **{k_: v for k_, v in results.items()
                                     if not isinstance(v, (float, complex, str))})
        print(f"\n  saved {out}")
    return results


# =============================================================================
# N=2 system: (u, E, U_0, U_1) with closure U_2 = U_0 * beta(U_1/U_0)
# =============================================================================

def _direct_u2_rhs(t, y_real, k: float, u_b: float, eps: float):
    """RHS for N=2 fluid system with exact per-mode beta(U_1/U_0) closure.

    State y = (u, E, U_0, U_1),  stacked as real-imaginary pairs.
    Closure: U_2 = U_0 * beta(U_1/U_0)  where beta(xi) = xi^2 - 1/Z'(xi).

    Off the single-eigenmode manifold (e.g. generic delta-E IC), U_1/U_0 is
    not equal to xi_b for any single mode; beta is evaluated at the effective
    ratio and acts as the off-manifold extension.
    """
    y = y_real[:4] + 1j * y_real[4:]
    u, E, U0, U1 = y
    U2 = U0 * _u2_beta(_safe_xi_ratio(U1, U0))
    du  = -E
    dE  = u - eps * (u_b * U0 + U1)
    dU0 = -1j * k * u_b * U0 - 1j * k * U1
    dU1 = -1j * k * u_b * U1 - 1j * k * U2 + 1.0 * E   # M_0 = 1
    dy = np.array([du, dE, dU0, dU1])
    return np.concatenate([dy.real, dy.imag])


def direct_u2_evolve(k: float, u_b: float, eps: float,
                     t_grid: np.ndarray, y0: np.ndarray,
                     rtol: float = 1e-8, atol: float = 1e-10) -> np.ndarray:
    """Return E(t) for the N=2 fluid system with the exact beta closure."""
    y0_real = np.concatenate([y0.real, y0.imag])
    sol = solve_ivp(
        _direct_u2_rhs, (t_grid[0], t_grid[-1]), y0_real,
        t_eval=t_grid, args=(k, u_b, eps),
        method="RK45", rtol=rtol, atol=atol,
    )
    if not sol.success:
        print(f"  direct_u2_evolve WARNING: {sol.message}")
    y_complex = sol.y[:4] + 1j * sol.y[4:]
    return y_complex[1]   # E(t)


def pade_u2_evolve(k: float, u_b: float, eps: float,
                   t_grid: np.ndarray, y0: np.ndarray) -> np.ndarray:
    """Return E(t) for the linear N=2 Padé-closed fluid system."""
    a_pade2 = pade_coefficients(2)
    return pade_evolve(k, u_b, eps, a_pade2, t_grid, y0)


def _nn_u2_rhs(t, y_real, k: float, u_b: float, eps: float, model):
    """RHS for N=2 fluid system with NN beta(U_1/U_0) closure."""
    y = y_real[:4] + 1j * y_real[4:]
    u, E, U0, U1 = y
    r1 = _safe_xi_ratio(U1, U0)
    U2 = U0 * complex(_u2_model_beta(model, r1))
    du  = -E
    dE  = u - eps * (u_b * U0 + U1)
    dU0 = -1j * k * u_b * U0 - 1j * k * U1
    dU1 = -1j * k * u_b * U1 - 1j * k * U2 + 1.0 * E
    dy = np.array([du, dE, dU0, dU1])
    return np.concatenate([dy.real, dy.imag])


def nn_u2_evolve(k: float, u_b: float, eps: float, model: MLP_u2,
                 t_grid: np.ndarray, y0: np.ndarray,
                 rtol: float = 1e-8, atol: float = 1e-10) -> np.ndarray:
    """Return E(t) for the N=2 fluid system with the NN beta closure."""
    y0_real = np.concatenate([y0.real, y0.imag])
    sol = solve_ivp(
        _nn_u2_rhs, (t_grid[0], t_grid[-1]), y0_real,
        t_eval=t_grid, args=(k, u_b, eps, model),
        method="RK45", rtol=rtol, atol=atol,
    )
    if not sol.success:
        print(f"  nn_u2_evolve WARNING: {sol.message}")
    y_complex = sol.y[:4] + 1j * sol.y[4:]
    return y_complex[1]


def kinetic_ic_to_fluid_u2(y0_kinetic: np.ndarray, k: float, u_b: float,
                            N_v: int = 96, V: float = 6.0) -> np.ndarray:
    """Project a kinetic state onto the N=2 fluid 4-d state (u, E, U_0, U_1)."""
    s, ds = s_grid(N_v, V)
    F = y0_kinetic[2:]
    U0 = ds * np.sum(F)
    U1 = ds * np.sum(s * F)
    return np.array([y0_kinetic[0], y0_kinetic[1], U0, U1], dtype=complex)


def kinetic_eigenmode_ic(k: float, u_b: float, eps: float,
                          omega_target: complex, amp: float = 1e-3,
                          N_v: int = 96, V: float = 6.0) -> np.ndarray:
    """Kinetic eigenvector closest to omega_target, normalised so |E| = amp.

    omega_target is a complex frequency (e.g. from kinetic_dispersion root).
    The kinetic matrix eigenvalue is lambda = -i*omega, so we search over
    omega = i*lambda for the closest match.
    """
    A = kinetic_system_sspace(k, u_b, eps, N_v, V)
    eigvals_lam, evecs = np.linalg.eig(A)
    omegas = 1j * eigvals_lam
    j = int(np.argmin(np.abs(omegas - omega_target)))
    v = evecs[:, j]
    y0 = v * (amp / abs(v[1]))
    return y0


def fluid_eigenmode_ic_u2(k: float, u_b: float, omega: complex,
                           amp: float = 1e-3) -> np.ndarray:
    """Analytical N=2 fluid eigenmode IC at kinetic eigenfrequency omega.

    Derived from the moment recurrence on a single eigenmode at xi_b:
        U1/U0 = xi_b                    (from the dU0/dt equation)
        U2    = U0 * beta(xi_b)         (direct-beta closure, exact on eigenmode)
        U0    = amp * i * Z'(xi_b) / k  (from the dU1/dt equation)
        u     = amp / (i*omega)

    This is the exact eigenmode of the direct-beta fluid system and is also
    a very accurate IC for the NN closure (to the extent the NN approximates
    beta accurately at xi_b).  The Pade closure does NOT support this eigenmode
    structure, so Pade will immediately project onto its own (spuriously growing)
    eigenmodes when started from this IC.

    Returns: complex array [u, E, U0, U1].
    """
    from bot.closures.u2 import Zprime
    xi_b = (omega - k * u_b) / k
    Zp = Zprime(xi_b)
    E0 = float(amp)
    u0 = E0 / (1j * omega)
    U0 = 1j * Zp * E0 / k
    U1 = 1j * xi_b * Zp * E0 / k
    return np.array([u0, E0, U0, U1], dtype=complex)


def fluid_eigenmode_ic_u3(k: float, u_b: float, omega: complex,
                           amp: float = 1e-3) -> np.ndarray:
    """Analytical N=3 fluid eigenmode IC at kinetic eigenfrequency omega.

    Extends fluid_eigenmode_ic_u2 by appending U2 = beta(xi_b) * U0:
        U2 = amp * i * beta(xi_b) * Z'(xi_b) / k

    This is the exact eigenmode structure for both the direct-alpha and
    bilinear N=3 closures (which are exact on eigenmode via alpha = xi * beta).
    It is also a good IC for the N=3 NN closure to the extent the NN
    approximates alpha accurately at xi_b.

    Returns: complex array [u, E, U0, U1, U2].
    """
    from bot.closures.u2 import Zprime, beta as _beta
    xi_b = (omega - k * u_b) / k
    Zp = Zprime(xi_b)
    beta_val = _beta(xi_b)
    E0 = float(amp)
    u0 = E0 / (1j * omega)
    U0 = 1j * Zp * E0 / k
    U1 = 1j * xi_b * Zp * E0 / k
    U2 = 1j * beta_val * Zp * E0 / k   # = beta_val * U0
    return np.array([u0, E0, U0, U1, U2], dtype=complex)


def kinetic_ic_from_fluid_moments(
        k: float, u_b: float, omega: complex,
        amp: float = 1e-3,
        N_v: int = 96, V: float = 6.0) -> np.ndarray:
    """Kinetic IC constructed from the N=2 fluid eigenmode moments.

    Projects the fluid eigenmode onto a smooth beam-frame velocity distribution:

        delta_F(s) = U0 * G(s) + U1 * s * G(s)

    where G(s) = exp(-s^2/2)/sqrt(2*pi) is the normalised beam Maxwellian and
    s = v - u_b is the beam-frame velocity.  This satisfies

        int delta_F(s) ds  = U0
        int s * delta_F(s) ds = U1

    exactly by Gaussian moment identities (int G ds = 1, int s G ds = 0,
    int s^2 G ds = 1).

    Advantages over kinetic_eigenmode_ic for Landau-damped cases:
      - Does not rely on finding a discrete eigenvalue (Landau poles are not
        discrete eigenvalues of the finite velocity-grid kinetic matrix).
      - Smooth in velocity space: suppresses van Kampen quasi-continuum modes
        relative to the target beam mode.
      - Moments (u, E, U0, U1) are exactly those of the fluid eigenmode, so
        the kinetic IC is consistent with the fluid IC used for direct-beta/NN.

    Returns: complex array of length 2 + N_v  (u, E, F[s_0], ..., F[s_{N_v-1}]).
    """
    from bot.closures.u2 import Zprime
    s, _ds = s_grid(N_v, V)

    xi_b = (omega - k * u_b) / k
    Zp   = Zprime(xi_b)
    E0   = float(amp)
    u0   = E0 / (1j * omega)
    U0   = 1j * Zp * E0 / k
    U1   = 1j * xi_b * Zp * E0 / k

    G       = np.exp(-0.5 * s**2) / np.sqrt(2.0 * np.pi)
    dF      = U0 * G + U1 * s * G          # delta_F on velocity grid

    y0 = np.zeros(2 + N_v, dtype=complex)
    y0[0]  = u0
    y0[1]  = E0
    y0[2:] = dF
    return y0


def kinetic_landau_ic(
        k: float, u_b: float, omega: complex,
        amp: float = 1e-3,
        N_v: int = 96, V: float = 6.0) -> np.ndarray:
    """Exact Landau-mode IC for the finite velocity-grid kinetic system.

    Derived from the linearized beam Vlasov equation in the kinetic matrix:

        d(delta_F)/dt = -ik(u_b+s) delta_F  +  E * src(s)

    For a single eigenmode e^{-i omega t} the exact solution is:

        delta_F(s) = -i E_0 * src(s) / (k * (s - xi_b))

    where  src(s) = (2s/sqrt(pi)) * exp(-s^2)  (from the code),
    xi_b = (omega - k*u_b)/k  is complex with Im(xi_b) < 0 for damped modes.

    Since Im(xi_b) != 0, the denominator (s - xi_b) never vanishes on the
    real s-axis, so delta_F(s) is a smooth complex function — no van Kampen
    quasi-continuum contamination in the continuous limit.

    The moments of this IC satisfy exactly:
        U0 = i * Z'(xi_b) * E0 / k    (matches fluid_eigenmode_ic_u2)
        U1 = i * xi_b * Z'(xi_b) * E0 / k

    This is the correct IC to see Landau damping in the kinetic simulation.
    kinetic_eigenmode_ic (argmin strategy) always picks a van Kampen mode
    instead because the Landau pole is not a discrete eigenvalue of the
    finite-grid kinetic matrix.

    Returns: complex array of length 2 + N_v  (u, E, F[s_0], ..., F[s_{N_v-1}]).
    """
    s, _ds = s_grid(N_v, V)

    xi_b = (omega - k * u_b) / k
    E0   = float(amp)
    u0   = E0 / (1j * omega)

    src  = (2.0 * s / np.sqrt(np.pi)) * np.exp(-s**2)   # same as in kinetic matrix
    dF   = -1j * E0 * src / (k * (s - xi_b))

    y0 = np.zeros(2 + N_v, dtype=complex)
    y0[0]  = u0
    y0[1]  = E0
    y0[2:] = dF
    return y0


# =============================================================================
# N=4 system: (u, E, U_0, U_1, U_2, U_3) with closure U_4 = U_0 * alpha4(...)
# =============================================================================

def fluid_eigenmode_ic_n4(k: float, u_b: float, omega: complex,
                           amp: float = 1e-3) -> np.ndarray:
    """Analytical N=4 fluid eigenmode IC at kinetic eigenfrequency omega.

    Extends fluid_eigenmode_ic_u3 by appending U3 = alpha(xi_b) * U0:
        U3 = amp * i * alpha(xi_b) * Z'(xi_b) / k

    Returns: complex array [u, E, U0, U1, U2, U3].
    """
    from bot.closures.u2 import Zprime, beta as _beta
    from bot.closures.inference import alpha as _alpha
    xi_b = (omega - k * u_b) / k
    Zp = Zprime(xi_b)
    beta_val  = _beta(xi_b)
    alpha_val = _alpha(xi_b)
    E0 = float(amp)
    u0 = E0 / (1j * omega)
    U0 = 1j * Zp * E0 / k
    U1 = 1j * xi_b * Zp * E0 / k
    U2 = 1j * beta_val * Zp * E0 / k
    U3 = 1j * alpha_val * Zp * E0 / k
    return np.array([u0, E0, U0, U1, U2, U3], dtype=complex)


def _direct_n4_rhs(t, y_real, k: float, u_b: float, eps: float):
    """RHS for N=4 fluid system with direct alpha4(U1/U0) closure.

    Extends the N=3 system by one beam moment: state is
        y = (u, E, U_0, U_1, U_2, U_3).
    Closure: U_4 = U_0 * alpha4(U_1/U_0)  (uses only r_1, Jensen error off-manifold).
    New dU_3 equation picks up an E source from M_2 = 1/2:
        dU_3/dt = -ik u_b U_3 - ik U_4 + 3*M_2 * E  =  ... + (3/2) E
    """
    from bot.closures.inference import alpha4 as _alpha4
    y = y_real[:6] + 1j * y_real[6:]
    u, E, U0, U1, U2, U3 = y
    if abs(U0) < 1e-15:
        U4 = 0.0 + 0.0j
    else:
        U4 = U0 * _alpha4(U1 / U0)
    du  = -E
    dE  = u - eps * (u_b * U0 + U1)
    dU0 = -1j * k * u_b * U0 - 1j * k * U1
    dU1 = -1j * k * u_b * U1 - 1j * k * U2 + 1.0 * E   # M_0 = 1
    dU2 = -1j * k * u_b * U2 - 1j * k * U3              # M_1 = 0
    dU3 = -1j * k * u_b * U3 - 1j * k * U4 + 1.5 * E   # 3*M_2 = 3*0.5 = 1.5
    dy  = np.array([du, dE, dU0, dU1, dU2, dU3])
    return np.concatenate([dy.real, dy.imag])


def direct_n4_evolve(k: float, u_b: float, eps: float,
                     t_grid: np.ndarray, y0: np.ndarray,
                     rtol: float = 1e-8, atol: float = 1e-10) -> np.ndarray:
    """Return E(t) for the N=4 fluid system with the direct alpha4(U1/U0) closure."""
    y0_real = np.concatenate([y0.real, y0.imag])
    sol = solve_ivp(
        _direct_n4_rhs, (t_grid[0], t_grid[-1]), y0_real,
        t_eval=t_grid, args=(k, u_b, eps),
        method="RK45", rtol=rtol, atol=atol,
    )
    if not sol.success:
        print(f"  direct_n4_evolve WARNING: {sol.message}")
    y_complex = sol.y[:6] + 1j * sol.y[6:]
    return y_complex[1]   # E(t)


def _nn_n4_rhs(t, y_real, k: float, u_b: float, eps: float, model):
    """RHS for N=4 fluid system with NN alpha4(r1, r2, r3) closure."""
    from bot.closures.n4_nn import n4_predict_alpha4
    y = y_real[:6] + 1j * y_real[6:]
    u, E, U0, U1, U2, U3 = y
    r1 = _safe_xi_ratio(U1, U0)
    r2 = _safe_xi_ratio(U2, U0)
    r3 = _safe_xi_ratio(U3, U0)
    U4 = U0 * complex(n4_predict_alpha4(model, r1, r2, r3))
    du  = -E
    dE  = u - eps * (u_b * U0 + U1)
    dU0 = -1j * k * u_b * U0 - 1j * k * U1
    dU1 = -1j * k * u_b * U1 - 1j * k * U2 + 1.0 * E
    dU2 = -1j * k * u_b * U2 - 1j * k * U3
    dU3 = -1j * k * u_b * U3 - 1j * k * U4 + 1.5 * E
    dy  = np.array([du, dE, dU0, dU1, dU2, dU3])
    return np.concatenate([dy.real, dy.imag])


def nn_n4_evolve(k: float, u_b: float, eps: float, model,
                 t_grid: np.ndarray, y0: np.ndarray,
                 rtol: float = 1e-8, atol: float = 1e-10) -> np.ndarray:
    """Return E(t) for the N=4 fluid system with the NN alpha4 closure."""
    y0_real = np.concatenate([y0.real, y0.imag])
    sol = solve_ivp(
        _nn_n4_rhs, (t_grid[0], t_grid[-1]), y0_real,
        t_eval=t_grid, args=(k, u_b, eps, model),
        method="RK45", rtol=rtol, atol=atol,
    )
    if not sol.success:
        print(f"  nn_n4_evolve WARNING: {sol.message}")
    y_complex = sol.y[:6] + 1j * sol.y[6:]
    return y_complex[1]


def pade_n4_evolve(k: float, u_b: float, eps: float,
                   t_grid: np.ndarray, y0: np.ndarray) -> np.ndarray:
    """Return E(t) for the linear N=4 Padé-closed fluid system."""
    a_pade4 = pade_coefficients(4)
    return pade_evolve(k, u_b, eps, a_pade4, t_grid, y0)


if __name__ == "__main__":
    r = run_comparison()
    # quick summary
    t = r["t"]
    gam_kin = r["omega_kin"].imag
    print(f"\n  γ_kin = {gam_kin:.4f}")
    print(f"\n  |E(t_end)| / |E(0)|:")
    for name in ["kin_lan", "kin_gen", "pade_lan", "pade_gen",
                  "inf_lan", "inf_gen", "nai_lan", "nai_gen"]:
        E = r[f"E_{name}"]
        ratio = abs(E[-1]) / abs(E[0])
        # fit effective growth rate from last half
        n_half = len(t) // 2
        E_late = np.abs(E[n_half:])
        if E_late.max() > 0:
            slope = np.polyfit(t[n_half:], np.log(np.maximum(E_late, 1e-30)), 1)[0]
        else:
            slope = np.nan
        print(f"    {name:>8s}:  {ratio:.3e}    γ_eff ≈ {slope:+.4f}")
