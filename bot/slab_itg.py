"""Slab ITG: time-domain kinetic (DKE) vs fluid closures, per Fourier mode.

Model (Slab_ITG_Closure_Relations.pdf, following Hammett-Perkins PRL 64 3019):
the linearized slab DKE for ions with a Maxwellian f0 whose density and
temperature vary along x, perturbations ~ exp(i k_y y + i k_par z - i omega t),
E_par = -i k_par Phi, v_E = -i k_y Phi/B:

    df/dt + i k_par v_par f = i (e Phi/T0) *
        ( omega_* [1 + eta (m v^2/(2 T0) - 3/2)] - k_par v_par ) f0

with omega_* = (T0/eB) k_y/L_n and eta = L_n/L_T.  Since v_perp enters only
through the drive, integrating over a Maxwellian v_perp (<m v_perp^2/(2T0)> = 1)
reduces the problem exactly to 1D:

    dg/dt = -i w g + i phi ( zeta_* [1 + eta (w^2 - 1/2)] - w ) F0(w)   (*)

Normalization
-------------
    w      = v_par / vbar,  vbar = sqrt(2) v_ti,  F0(w) = exp(-w^2)/sqrt(pi)
    t      in units of (k_par vbar)^{-1}   (k_par > 0)
    zeta   = omega  / (k_par vbar),   zeta_* = omega_* / (k_par vbar)
    phi    = e Phi / T0i
    tau    = T0i / T0e
    g(w,t) with  U_n = int w^n g dw,  U_0 = n_i,tilde / n0

Quasineutrality with Boltzmann electrons (n_e,tilde = +n0 e Phi/T0e) closes
the loop:  phi = U_0 / tau.

Kinetic response (integrate (*) over w with the eigenmode ansatz):

    R(zeta) = -U_0/phi
            = 1 - eta zeta zeta_*
              + [zeta - zeta_*(1 - eta/2 + eta zeta^2)] Z(zeta)

and the dispersion relation is D(zeta) = R(zeta) + tau = 0.

NOTE: bot/itg_test.py uses 1 - eta*zeta_* for the first gradient term
(no zeta), citing HP Eq. 12.  Direct integration of the DKE gives
1 - eta*zeta*zeta_* (the -eta*zeta*zeta_* term is the polynomial part of
zeta_* eta int w^2 F0/(w - zeta) dw = zeta_* eta (zeta + zeta^2 Z)).
Both variants are provided here; the discretized-DKE eigenvalues and the
time-domain solve adjudicate (see bot/paper/notes_slab_itg.md).

Marginal stability of the DKE-derived form (real zeta, both the Z-coefficient
bracket and the polynomial part must vanish):

    eta_th = 1 + sqrt(1 + 2 tau (1 + tau) / zeta_*^2)

which -> 2 only as zeta_* -> inf.  The legacy form gives eta_th = (1+tau)/zeta_*.

Fluid hierarchy (exact moments of (*)):

    dU_n/dt = -i U_{n+1} + i phi sigma_n
    sigma_n = zeta_* [ M_n + eta (M_{n+2} - M_n/2) ] - M_{n+1}
    sigma_0 = zeta_*,   sigma_1 = -1/2,   sigma_2 = zeta_* (1 + eta)/2

(M_n = int w^n F0 dw: 1, 0, 1/2, 0, 3/4, ...).  The eta drive first enters at
n = 2: an N=2 fluid model (any closure for U_2, including the direct beta
closure) is structurally eta-independent and cannot contain ITG physics.

Closures implemented:
  * direct beta  (N=2): U_2 = beta(U_1/U_0) U_0,  beta(z) = z^2 - 1/Z'(z)
  * direct alpha (N=3): U_3 = alpha(U_1/U_0) U_0, alpha(z) = z^3 - z/Z'(z)
  * Hammett-Perkins (N=3, linear):  q = -n0 chi1 vbar i sgn(k) T~, i.e.
        U_3 = (Gamma/2) U_1 - i (chi1/2)(2 U_2 - U_0)
    with chi1 = 2/sqrt(pi), Gamma = 3 (exact 1D hierarchy).  In the
    gradient-free limit this reproduces HP's 3-pole R_3 exactly
    (checked against bot/itg_test.py R3_hp).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.special import wofz

from bot.closures.pade import maxwellian_moment

SQRT_PI = float(np.sqrt(np.pi))
CHI1_HP = 2.0 / SQRT_PI


# ── plasma dispersion function ───────────────────────────────────────────────

def Z(zeta):
    """Z(zeta) = i sqrt(pi) w(zeta) via Faddeeva (valid in both half-planes)."""
    return 1j * SQRT_PI * wofz(zeta)


def Zprime(zeta):
    return -2.0 * (1.0 + zeta * Z(zeta))


def beta_closure(z):
    """Exact per-mode U_2/U_0 (gradient-free manifold): z^2 - 1/Z'(z)."""
    return z**2 - 1.0 / Zprime(z)


def alpha_closure(z):
    """Exact per-mode U_3/U_0 (gradient-free manifold): z^3 - z/Z'(z)."""
    return z**3 - z / Zprime(z)


# ── ITG-manifold moments and closures ────────────────────────────────────────
# On the kinetic ITG eigenmode g = phi S(w)/(w - zeta), so with
# I_n(zeta) = int w^n F0/(w - zeta) dw   (I_0 = Z, I_{n+1} = zeta I_n + M_n):
#     U_n / phi = zeta_*(1 - eta/2) I_n + zeta_* eta I_{n+2} - I_{n+1}
# The manifold moment ratios therefore depend on (zeta_*, eta) — the drive
# lives inside the closure, unlike the gradient-free beta/alpha.

def manifold_Un_over_phi(z, zeta_star: float, eta: float, n_max: int = 3):
    """U_n/phi on the kinetic ITG manifold, n = 0..n_max (needs I_{n_max+2})."""
    I = [Z(z)]
    for n in range(n_max + 2):
        I.append(z * I[-1] + maxwellian_moment(n))
    return [zeta_star * (1.0 - 0.5 * eta) * I[n]
            + zeta_star * eta * I[n + 2] - I[n + 1]
            for n in range(n_max + 1)]


def beta_itg(z, zeta_star: float, eta: float):
    """Exact U_2/U_0 on the ITG manifold at frequency z (cf. beta_closure)."""
    U = manifold_Un_over_phi(z, zeta_star, eta, n_max=2)
    den = U[0]
    if abs(den) < 1e-12:
        den = 1e-12 * (den / abs(den) if den != 0 else 1.0)
    return U[2] / den


def alpha_itg(z, zeta_star: float, eta: float):
    """Exact U_3/U_0 on the ITG manifold at frequency z (cf. alpha_closure)."""
    U = manifold_Un_over_phi(z, zeta_star, eta, n_max=3)
    den = U[0]
    if abs(den) < 1e-12:
        den = 1e-12 * (den / abs(den) if den != 0 else 1.0)
    return U[3] / den


# ── kinetic response / dispersion relation ───────────────────────────────────

def R_kinetic(zeta, zeta_star: float, eta: float):
    """DKE-derived ion response R = -U_0/phi (has the eta*zeta*zeta_* term)."""
    return (1.0 - eta * zeta * zeta_star
            + (zeta - zeta_star * (1.0 - 0.5 * eta + eta * zeta**2)) * Z(zeta))


def R_legacy(zeta, zeta_star: float, eta: float):
    """Variant used in bot/itg_test.py: 1 - eta*zeta_* (no zeta factor)."""
    return (1.0 - eta * zeta_star
            + (zeta - zeta_star * (1.0 - 0.5 * eta + eta * zeta**2)) * Z(zeta))


def D_kinetic(zeta, zeta_star: float, eta: float, tau: float = 1.0):
    return R_kinetic(zeta, zeta_star, eta) + tau


def D_legacy(zeta, zeta_star: float, eta: float, tau: float = 1.0):
    return R_legacy(zeta, zeta_star, eta) + tau


def eta_threshold(zeta_star: float, tau: float = 1.0) -> float:
    """Marginal eta of the DKE-derived dispersion (analytic, real-zeta)."""
    return 1.0 + np.sqrt(1.0 + 2.0 * tau * (1.0 + tau) / zeta_star**2)


def find_mode(D_func, zeta0: complex, zeta_star: float, eta: float,
              tau: float = 1.0) -> complex:
    """Newton-solve D(zeta) = 0 near zeta0; nan on failure."""
    def res(x):
        v = D_func(complex(x[0], x[1]), zeta_star, eta, tau)
        return [v.real, v.imag]
    sol = root(res, [zeta0.real, zeta0.imag], method="hybr", tol=1e-13)
    if not sol.success:
        return complex(float("nan"), float("nan"))
    return complex(sol.x[0], sol.x[1])


def max_growing_root(D_func, zeta_star: float, eta: float, tau: float = 1.0,
                     re_lo: float = -3.0, re_hi: float = 3.0,
                     im_lo: float = 0.001, im_hi: float = 3.0,
                     n_re: int = 25, n_im: int = 15) -> complex:
    """Most-unstable root of D in the upper half plane (grid + Newton).

    Returns complex(nan, 0) if no growing root is found.
    """
    best = complex(float("nan"), 0.0)
    for re_v in np.linspace(re_lo, re_hi, n_re):
        for im_v in np.geomspace(im_lo, im_hi, n_im):
            zt = find_mode(D_func, complex(re_v, im_v), zeta_star, eta, tau)
            if np.isnan(zt.real) or zt.imag <= best.imag:
                continue
            if abs(D_func(zt, zeta_star, eta, tau)) < 1e-9:
                best = zt
    return best


def min_damped_root(D_func, zeta_star: float, eta: float, tau: float = 1.0,
                    re_lo: float = -3.0, re_hi: float = 3.0,
                    im_lo: float = -3.0, im_hi: float = -0.001,
                    n_re: int = 25, n_im: int = 15) -> complex:
    """Least-damped root of D in the lower half plane (grid + Newton).

    Mirror image of max_growing_root: below threshold, D has no growing
    root at all, so the physically relevant mode is the slowest-decaying
    (least negative Im(zeta)) Landau-damped root instead.

    Returns complex(nan, -inf) if no damped root is found.
    """
    best = complex(float("nan"), -np.inf)
    for re_v in np.linspace(re_lo, re_hi, n_re):
        for im_v in -np.geomspace(-im_hi, -im_lo, n_im):
            zt = find_mode(D_func, complex(re_v, im_v), zeta_star, eta, tau)
            if np.isnan(zt.real) or zt.imag <= best.imag:
                continue
            if abs(D_func(zt, zeta_star, eta, tau)) < 1e-9:
                best = zt
    return best


# ── discretized kinetic system (velocity grid) ───────────────────────────────

def w_grid(N_w: int, W: float = 8.0):
    w = np.linspace(-W, W, N_w)
    return w, w[1] - w[0]


def drive_S(w: np.ndarray, zeta_star: float, eta: float) -> np.ndarray:
    """S(w) = (zeta_*[1 + eta(w^2 - 1/2)] - w) F0(w)."""
    F0 = np.exp(-w**2) / SQRT_PI
    return (zeta_star * (1.0 + eta * (w**2 - 0.5)) - w) * F0


def kinetic_system(zeta_star: float, eta: float, tau: float = 1.0,
                   N_w: int = 600, W: float = 8.0) -> np.ndarray:
    """Linear ODE matrix A for dg/dt = A g on the w-grid.

    dg_j/dt = -i w_j g_j + i S(w_j) phi,   phi = U_0/tau = (dw sum_l g_l)/tau
    """
    w, dw = w_grid(N_w, W)
    A = np.diag(-1j * w).astype(complex)
    A += np.outer(1j * drive_S(w, zeta_star, eta), np.full(N_w, dw / tau))
    return A


def kinetic_modes(zeta_star: float, eta: float, tau: float = 1.0,
                  N_w: int = 600, W: float = 8.0) -> np.ndarray:
    """All zeta = i * eigval(A).  Growing modes have Im(zeta) > 0."""
    return 1j * np.linalg.eigvals(kinetic_system(zeta_star, eta, tau, N_w, W))


def most_unstable_kinetic(zeta_star: float, eta: float, tau: float = 1.0,
                          N_w: int = 600, W: float = 8.0,
                          refine: bool = True) -> complex:
    """Most-unstable eigenvalue of the discretized DKE.

    If refine=True and the mode is growing, Newton-polish it on the analytic
    D_kinetic (the discrete eigenvalue converges to the analytic root as
    N_w -> inf; the polish removes the O(dw) discretization error).
    """
    zs = kinetic_modes(zeta_star, eta, tau, N_w, W)
    z0 = zs[int(np.argmax(zs.imag))]
    if refine and z0.imag > 1e-6:
        zr = find_mode(D_kinetic, z0, zeta_star, eta, tau)
        if not np.isnan(zr.real) and abs(zr - z0) < 0.2:
            return zr
    return z0


def kinetic_evolve(zeta_star: float, eta: float, tau: float,
                   t_grid: np.ndarray, g0: np.ndarray,
                   N_w: int = 600, W: float = 8.0) -> dict:
    """Evolve the discretized DKE from g0 via eigendecomposition.

    Returns dict with t, U0(t), U1(t), U2(t), phi(t).
    """
    w, dw = w_grid(N_w, W)
    A = kinetic_system(zeta_star, eta, tau, N_w, W)
    lam, V = np.linalg.eig(A)
    c = np.linalg.solve(V, g0)
    U = np.empty((3, len(t_grid)), dtype=complex)
    for i, t in enumerate(t_grid):
        g = V @ (c * np.exp(lam * t))
        U[0, i] = dw * np.sum(g)
        U[1, i] = dw * np.sum(w * g)
        U[2, i] = dw * np.sum(w**2 * g)
    return {"t": t_grid, "U0": U[0], "U1": U[1], "U2": U[2],
            "phi": U[0] / tau}


# ── fluid: Hammett-Perkins 3-moment (linear) ─────────────────────────────────

def fluid_hp_system(zeta_star: float, eta: float, tau: float = 1.0,
                    Gamma: float = 3.0, chi1: float = CHI1_HP) -> np.ndarray:
    """3x3 matrix for d/dt (U_0, U_1, U_2) with the HP heat-flux closure.

        dU_0 = -i U_1 + i zeta_* phi
        dU_1 = -i U_2 - (i/2) phi
        dU_2 = -i U_3 + i zeta_*(1+eta)/2 phi
        U_3  = (Gamma/2) U_1 - i (chi1/2) (2 U_2 - U_0),   phi = U_0/tau

    Gamma = 3 is the exact 1D hierarchy coefficient (HP recommended); the
    Braginskii-like variant uses Gamma = 5/3.
    """
    A = np.zeros((3, 3), dtype=complex)
    # dU_0
    A[0, 1] = -1j
    A[0, 0] = 1j * zeta_star / tau
    # dU_1
    A[1, 2] = -1j
    A[1, 0] = -0.5j / tau
    # dU_2 = -i U_3 + source
    A[2, 1] = -1j * (Gamma / 2.0)
    A[2, 2] = -1j * (-1j * chi1)          # -i * (-i chi1/2 * 2) U_2
    A[2, 0] = -1j * (1j * chi1 / 2.0)     # -i * (+i chi1/2) U_0
    A[2, 0] += 1j * zeta_star * (1.0 + eta) / (2.0 * tau)
    return A


def fluid_hp_modes(zeta_star: float, eta: float, tau: float = 1.0,
                   Gamma: float = 3.0, chi1: float = CHI1_HP) -> np.ndarray:
    """All zeta = i * eigval (3 modes)."""
    return 1j * np.linalg.eigvals(fluid_hp_system(zeta_star, eta, tau,
                                                  Gamma, chi1))


def _linear_rhs(t, y_real, A):
    n = A.shape[0]
    y = y_real[:n] + 1j * y_real[n:]
    dy = A @ y
    return np.concatenate([dy.real, dy.imag])


def linear_evolve_rk45(A: np.ndarray, t_grid: np.ndarray, y0: np.ndarray,
                       rtol: float = 1e-9, atol: float = 1e-12) -> np.ndarray:
    """RK45 time-stepping of dy/dt = A y (same physics as eigendecomposition,
    but genuinely integrated -- for apples-to-apples comparison against a
    nonlinear closure that has no choice but to be time-stepped)."""
    n = A.shape[0]
    y0_real = np.concatenate([y0.real, y0.imag])
    sol = solve_ivp(_linear_rhs, (t_grid[0], t_grid[-1]), y0_real,
                    t_eval=t_grid, args=(A,), method="RK45",
                    rtol=rtol, atol=atol)
    if not sol.success:
        print(f"  linear_evolve_rk45 WARNING: {sol.message}")
    m = len(sol.t)
    return sol.y[:n, :m] + 1j * sol.y[n:, :m]


def fluid_hp_evolve(zeta_star: float, eta: float, tau: float,
                    t_grid: np.ndarray, y0: np.ndarray,
                    Gamma: float = 3.0, chi1: float = CHI1_HP,
                    method: str = "eigen",
                    rtol: float = 1e-9, atol: float = 1e-12) -> dict:
    """Evolve the HP 3-moment system from y0 = (U_0, U_1, U_2).

    method="eigen" (default): exact via eigendecomposition (matrix
    exponential), no time-stepping error.
    method="rk45": genuinely time-stepped, matching the numerical scheme
    used for the nonlinear direct-beta/alpha closures (fair side-by-side
    comparison of the transient, not just the asymptotic rate).
    """
    A = fluid_hp_system(zeta_star, eta, tau, Gamma, chi1)
    if method == "eigen":
        lam, V = np.linalg.eig(A)
        c = np.linalg.solve(V, y0)
        Y = np.empty((3, len(t_grid)), dtype=complex)
        for i, t in enumerate(t_grid):
            Y[:, i] = V @ (c * np.exp(lam * t))
    elif method == "rk45":
        Y = linear_evolve_rk45(A, t_grid, y0, rtol=rtol, atol=atol)
    else:
        raise ValueError(f"unknown method {method!r}")
    return {"t": t_grid, "U0": Y[0], "U1": Y[1], "U2": Y[2],
            "phi": Y[0] / tau}


# ── fluid: general N-moment linear closure (e.g. AAA-fit, generalizes HP) ───
# HP's fluid_hp_system is the N=3 special case of this with a hardcoded
# (Gamma, chi1) closure; here `a` is an arbitrary length-N closure-coefficient
# array U_N = sum_i a_i U_i (same convention as bot.closures.pade / bot.fluid,
# and the SAME `a` -- fit purely from the universal Zprime(zeta) -- works for
# both BoT's fluid_system and this ITG system, since the drive only enters
# the lower-moment sigma_n source terms, not the closure map itself.

def fluid_pade_system(zeta_star: float, eta: float, tau: float,
                      a: np.ndarray) -> np.ndarray:
    """N x N matrix for d/dt (U_0,...,U_{N-1}) with closure U_N = sum a_i U_i.

        dU_n = -i U_{n+1} + i phi sigma_n     n = 0..N-2
        dU_{N-1} = -i (sum_i a_i U_i) + i phi sigma_{N-1}
        sigma_n = zeta_* [M_n + eta (M_{n+2} - M_n/2)] - M_{n+1},   phi = U_0/tau
    """
    a = np.asarray(a, dtype=complex)
    N = len(a)

    def sigma(n):
        M = maxwellian_moment
        return (zeta_star * (M(n) + eta * (M(n + 2) - 0.5 * M(n)))
                - M(n + 1))

    A = np.zeros((N, N), dtype=complex)
    for n in range(N - 1):
        A[n, n + 1] = -1j
        A[n, 0] += 1j * sigma(n) / tau
    n = N - 1
    A[n, :] += -1j * a
    A[n, 0] += 1j * sigma(n) / tau
    return A


def fluid_pade_evolve(zeta_star: float, eta: float, tau: float,
                      t_grid: np.ndarray, y0: np.ndarray, a: np.ndarray,
                      method: str = "rk45",
                      rtol: float = 1e-9, atol: float = 1e-12) -> dict:
    """Evolve the general N-moment linear closure system from y0.

    method="eigen": exact via eigendecomposition.  method="rk45": genuinely
    time-stepped (see fluid_hp_evolve).
    """
    A = fluid_pade_system(zeta_star, eta, tau, a)
    N = len(a)
    if method == "eigen":
        lam, V = np.linalg.eig(A)
        c = np.linalg.solve(V, y0)
        Y = np.empty((N, len(t_grid)), dtype=complex)
        for i, t in enumerate(t_grid):
            Y[:, i] = V @ (c * np.exp(lam * t))
    elif method == "rk45":
        Y = linear_evolve_rk45(A, t_grid, y0, rtol=rtol, atol=atol)
    else:
        raise ValueError(f"unknown method {method!r}")
    out = {"t": t_grid, "phi": Y[0] / tau}
    for n in range(N):
        out[f"U{n}"] = Y[n]
    return out


# ── fluid: direct beta (N=2) and direct alpha (N=3) closures ────────────────
# Same safeguard rationale as bot/closed_loop.py: off-manifold U_0 nodes send
# U_1/U_0 to large |Im| where Z' overflows; clip the ratio magnitude.

_XI_RATIO_MAX = 12.0


def _safe_ratio(U1, U0, floor: float = 1e-15,
                max_mag: float = _XI_RATIO_MAX) -> complex:
    if abs(U0) < floor:
        return 0.0 + 0.0j
    r = U1 / U0
    m = abs(r)
    return r * (max_mag / m) if m > max_mag else r


def _closure_arg(U1, U0, zeta_star: float, tau: float, variant: str):
    """Closure argument: r_1 = U_1/U_0, drive-shifted for non-plain variants.

    On the ITG eigenmode the exact n=0 moment equation with quasineutrality
    gives U_1/U_0 = zeta + zeta_*/tau, so zeta_hat = U_1/U_0 - zeta_*/tau
    recovers the mode frequency ('shifted' and 'itg' variants).
    """
    r1 = _safe_ratio(U1, U0)
    if variant == "plain":
        return r1
    return r1 - zeta_star / tau


def _beta_of(variant: str, z, zeta_star: float, eta: float):
    """U_2/U_0 closure value: gradient-free beta or ITG-manifold beta."""
    if variant == "itg":
        return beta_itg(z, zeta_star, eta)
    return beta_closure(z)


def _alpha_of(variant: str, z, zeta_star: float, eta: float):
    """U_3/U_0 closure value: gradient-free alpha or ITG-manifold alpha."""
    if variant == "itg":
        return alpha_itg(z, zeta_star, eta)
    return alpha_closure(z)


def _direct_beta_rhs(t, y_real, zeta_star: float, eta: float, tau: float,
                     variant: str):
    """N=2 system (U_0, U_1) with U_2 closed per variant:

    'plain'   U_2 = beta(U_1/U_0) U_0            (eta-independent; no ITG)
    'shifted' U_2 = beta(zeta_hat) U_0           (still eta-independent: its
              dispersion is exactly the kinetic one frozen at eta = 2)
    'itg'     U_2 = beta_itg(zeta_hat; zeta_*, eta) U_0   (kinetic root is an
              exact eigenvalue of the closed system)
    """
    y = y_real[:2] + 1j * y_real[2:]
    U0, U1 = y
    phi = U0 / tau
    z = _closure_arg(U1, U0, zeta_star, tau, variant)
    U2 = U0 * _beta_of(variant, z, zeta_star, eta)
    dU0 = -1j * U1 + 1j * zeta_star * phi
    dU1 = -1j * U2 - 0.5j * phi
    dy = np.array([dU0, dU1])
    return np.concatenate([dy.real, dy.imag])


def direct_beta_evolve(zeta_star: float, eta: float, tau: float,
                       t_grid: np.ndarray, y0: np.ndarray,
                       rtol: float = 1e-9, atol: float = 1e-12,
                       variant: str = "plain") -> dict:
    """Evolve the N=2 direct-beta system from y0 = (U_0, U_1)."""
    y0_real = np.concatenate([y0.real, y0.imag])
    sol = solve_ivp(_direct_beta_rhs, (t_grid[0], t_grid[-1]), y0_real,
                    t_eval=t_grid, args=(zeta_star, eta, tau, variant),
                    method="RK45", rtol=rtol, atol=atol)
    if not sol.success:
        print(f"  direct_beta_evolve[{variant}] WARNING: {sol.message}")
    n = len(sol.t)
    Y = sol.y[:2, :n] + 1j * sol.y[2:, :n]
    return {"t": sol.t, "U0": Y[0], "U1": Y[1], "phi": Y[0] / tau}


def _direct_alpha_rhs(t, y_real, zeta_star: float, eta: float, tau: float,
                      variant: str):
    """N=3 system (U_0, U_1, U_2) with U_3 closed per variant (see beta)."""
    y = y_real[:3] + 1j * y_real[3:]
    U0, U1, U2 = y
    phi = U0 / tau
    z = _closure_arg(U1, U0, zeta_star, tau, variant)
    U3 = U0 * _alpha_of(variant, z, zeta_star, eta)
    dU0 = -1j * U1 + 1j * zeta_star * phi
    dU1 = -1j * U2 - 0.5j * phi
    dU2 = -1j * U3 + 1j * zeta_star * (1.0 + eta) * 0.5 * phi
    dy = np.array([dU0, dU1, dU2])
    return np.concatenate([dy.real, dy.imag])


def direct_alpha_evolve(zeta_star: float, eta: float, tau: float,
                        t_grid: np.ndarray, y0: np.ndarray,
                        rtol: float = 1e-9, atol: float = 1e-12,
                        variant: str = "plain") -> dict:
    """Evolve the N=3 direct-alpha system from y0 = (U_0, U_1, U_2)."""
    y0_real = np.concatenate([y0.real, y0.imag])
    sol = solve_ivp(_direct_alpha_rhs, (t_grid[0], t_grid[-1]), y0_real,
                    t_eval=t_grid, args=(zeta_star, eta, tau, variant),
                    method="RK45", rtol=rtol, atol=atol)
    if not sol.success:
        print(f"  direct_alpha_evolve[{variant}] WARNING: {sol.message}")
    n = len(sol.t)
    Y = sol.y[:3, :n] + 1j * sol.y[3:, :n]
    return {"t": sol.t, "U0": Y[0], "U1": Y[1], "U2": Y[2], "phi": Y[0] / tau}


# ── initial conditions ───────────────────────────────────────────────────────

def density_ic_kinetic(amp: float = 1e-3, N_w: int = 600,
                       W: float = 8.0) -> np.ndarray:
    """Generic density perturbation: g0 = amp F0(w)  (U_0 = amp, U_1 = 0...)."""
    w, _dw = w_grid(N_w, W)
    return amp * np.exp(-w**2) / SQRT_PI


def density_ic_fluid(amp: float = 1e-3, N: int = 3) -> np.ndarray:
    """Matching fluid IC: U_n = amp * M_n for n = 0..N-1 (moments of amp*F0(w)).

    Exact for any N (was previously hardcoded/incomplete for N > 3, silently
    zeroing M_4, M_6, ... instead of their true nonzero Maxwellian values).
    """
    return np.array([amp * maxwellian_moment(n) for n in range(N)],
                    dtype=complex)


def eigenmode_ic_kinetic(zeta_star: float, eta: float, tau: float = 1.0,
                         amp: float = 1e-3, N_w: int = 600, W: float = 8.0
                         ) -> tuple[np.ndarray, complex]:
    """Most-unstable eigenvector of the discretized DKE, |U_0| = amp.

    Returns (g0, zeta_mode).
    """
    _w, dw = w_grid(N_w, W)
    A = kinetic_system(zeta_star, eta, tau, N_w, W)
    lam, V = np.linalg.eig(A)
    zs = 1j * lam
    j = int(np.argmax(zs.imag))
    g = V[:, j]
    U0 = dw * np.sum(g)
    return g * (amp / U0), zs[j]


def kinetic_ic_to_fluid(g0: np.ndarray, N: int = 3, N_w: int = 600,
                        W: float = 8.0) -> np.ndarray:
    """Project a kinetic state onto the first N fluid moments."""
    w, dw = w_grid(N_w, W)
    return np.array([dw * np.sum(w**n * g0) for n in range(N)], dtype=complex)


# ── growth-rate / frequency extraction ───────────────────────────────────────

def fit_mode(t: np.ndarray, x: np.ndarray,
             frac: tuple[float, float] = (0.5, 1.0)) -> complex:
    """Fit zeta = omega_r + i gamma from a complex signal x(t) ~ e^{-i zeta t}.

    gamma from the log-|x| slope, omega_r from the unwrapped-phase slope,
    over the window frac of the record.
    """
    i0, i1 = int(frac[0] * len(t)), int(frac[1] * len(t))
    tt, xx = t[i0:i1], x[i0:i1]
    mag = np.abs(xx)
    if mag.max() <= 0 or np.any(mag < 1e-300):
        return complex(float("nan"), float("nan"))
    gamma = np.polyfit(tt, np.log(mag), 1)[0]
    omega_r = -np.polyfit(tt, np.unwrap(np.angle(xx)), 1)[0]
    return complex(omega_r, gamma)
