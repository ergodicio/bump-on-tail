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
