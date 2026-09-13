"""Driver: standing Langmuir wave from a density perturbation, all seven closures.

Generates bot/figures/fig_standing_wave.png.

The stress test from the "Standing plasma wave example for EKR" note: a single
Maxwellian electron species (no beam), linear 1D-1V Vlasov-Poisson, one
harmonic e^{ikx}, initialized with f_1 = eps f_0 (pure density perturbation,
so U_n(0) = eps M_n: U_0 = eps, U_1 = 0, U_2 = eps/2, U_3 = 0).  The kinetic
answer is a Langmuir oscillation Landau-damped at the least-damped root; on
the e^{ikx} harmonic this is the equal-weight superposition of the root pair
(zeta_L, -zeta_L^*), so U_0(t) is real and crosses zero every half period.

Repo units: v_t = 1, omega_p = 1, so k lambda_D = k / sqrt 2.  With Poisson
slaving the field to U_0, the N-moment hierarchy is

    dU_n/dt = -i k U_{n+1} + n M_{n-1} E,     E = -(i/k) U_0,

with M_0 = 1, M_1 = 0, M_2 = 1/2, M_3 = 0, closed at U_N by each closure.  The
closures are the same seven objects used by fig_u2_landau_twomode_single
(same coefficient sets and NN checkpoints), applied to a different field
equation -- the closure map U_N/U_0 = f(U_j/U_0) is species-intrinsic.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import newton

from bot.closed_loop import _safe_xi_ratio
from bot.closures.n4_nn import load_super as load_n4_super, n4_predict_alpha4
from bot.closures.naive_nn import load as load_naive, naive_predict_alpha
from bot.closures.opt_hunana import opt_coefficients
from bot.closures.pade import pade_coefficients, two_point_coefficients
from bot.closures.u2 import Z, beta

K_LAMBDA_D = 0.4
K = K_LAMBDA_D * np.sqrt(2.0)
EPS = 1e-3
T_END = 40.0
T = np.linspace(0.0, T_END, 2001)
M = np.array([1.0, 0.0, 0.5, 0.0])          # Maxwellian moments M_0..M_3

SAVE_PNG = "bot/figures/fig_standing_wave.png"
HOMOG_CKPT: dict = {}          # {N: path} overrides for the homogeneous checkpoints
HOMOG_KW: dict = {}            # extra load() kwargs (pin, ext) for non-default checkpoints
HOMOG_LAM = None               # normalization lam override (None = module default)
HOMOG_LABEL = r"NN homog."

RC = {"font.family": "serif", "mathtext.fontset": "cm",
      "font.size": 7, "axes.titlesize": 7}


def R(z):
    return 1.0 + z * Z(z)


def landau_root() -> complex:
    disp = lambda z: 1.0 + (2.0 / K ** 2) * R(z)
    zeta = newton(disp, 2.3 - 0.1j, tol=1e-12)
    return zeta * K                              # omega, in omega_p units


# --- kinetic reference (velocity-grid eigen-decomposition) -------------------
def kinetic(N_v: int = 1000, V: float = 8.0) -> np.ndarray:
    v = np.linspace(-V, V, N_v)
    dv = v[1] - v[0]
    w = np.full(N_v, dv); w[0] = w[-1] = dv / 2
    f0 = np.exp(-v ** 2) / np.sqrt(np.pi)
    df0 = -2.0 * v * f0
    A = np.diag(-1j * K * v) + (1j / K) * np.outer(df0, w)
    lam, Vec = np.linalg.eig(A)
    c = np.linalg.solve(Vec, EPS * f0)
    return np.array([w @ (Vec @ (c * np.exp(lam * tt))) for tt in T])


# --- generic N-moment Vlasov-Poisson system -----------------------------------
def evolve(N: int, closure, rtol=1e-9, atol=1e-13,
           U_init: np.ndarray | None = None) -> np.ndarray:
    """Return U_0(t) for the N-moment system closed by closure(U) -> U_N.

    Default initial condition is the density perturbation U_n(0) = eps M_n;
    pass U_init (length N, complex) to start elsewhere.
    """
    def rhs(t, y):
        U = y[:N] + 1j * y[N:]
        UN = closure(U)
        E = -(1j / K) * U[0]
        dU = np.empty(N, dtype=complex)
        for n in range(N):
            nxt = U[n + 1] if n + 1 < N else UN
            dU[n] = -1j * K * nxt + n * M[n - 1] * E if n > 0 else -1j * K * nxt
        return np.concatenate([dU.real, dU.imag])

    U0 = EPS * M[:N] if U_init is None else np.asarray(U_init, dtype=complex)
    sol = solve_ivp(rhs, (0.0, T_END), np.concatenate([U0.real, U0.imag]),
                    t_eval=T, method="RK45", rtol=rtol, atol=atol)
    if not sol.success:
        print(f"  WARNING ({N}-moment): {sol.message}")
    out = np.full(len(T), np.nan, dtype=complex)
    out[:sol.y.shape[1]] = sol.y[0] + 1j * sol.y[N]
    return out


def linear_closure(a: np.ndarray):
    a = np.asarray(a, dtype=complex)
    return lambda U: a @ U


def linear_roots(a: np.ndarray) -> np.ndarray:
    """Eigen-frequencies omega of the linear closed system (for the table)."""
    N = len(a)
    A = np.zeros((N, N), dtype=complex)
    for n in range(N):
        if n + 1 < N:
            A[n, n + 1] += -1j * K
        else:
            A[n, :] += -1j * K * a
        if n > 0:
            A[n, 0] += n * M[n - 1] * (-1j / K)
    lam = np.linalg.eigvals(A)            # dU/dt = A U,  U ~ e^{lam t} = e^{-i omega t}
    return 1j * lam


def ekr_closure(U):
    return U[0] * beta(_safe_xi_ratio(U[1], U[0]))


def nn3_closure(model):
    def f(U):
        if abs(U[0]) < 1e-15:
            return 0j
        return U[0] * complex(naive_predict_alpha(model, U[1] / U[0], U[2] / U[0]))
    return f


def nn4_closure(model):
    def f(U):
        r = [_safe_xi_ratio(U[j], U[0]) for j in (1, 2, 3)]
        return U[0] * complex(n4_predict_alpha4(model, *r))
    return f


def fit_mode(U0: np.ndarray, t0: float = 5.0):
    """(omega, gamma) from zero-crossing spacing and the log-envelope slope."""
    m = (T >= t0) & np.isfinite(U0)
    t, x = T[m], U0.real[m]
    s = np.sign(x)
    ic = np.where(s[1:] * s[:-1] < 0)[0]
    if len(ic) < 3:
        return np.nan, np.nan
    tc = t[ic] - x[ic] * (t[ic + 1] - t[ic]) / (x[ic + 1] - x[ic])
    omega = np.pi / np.mean(np.diff(tc))
    env_t, env_v = [], []
    for i0, i1 in zip(ic[:-1], ic[1:]):
        j = i0 + np.argmax(np.abs(x[i0:i1 + 1]))
        env_t.append(t[j]); env_v.append(np.abs(x[j]))
    gamma = np.polyfit(env_t, np.log(env_v), 1)[0]
    return omega, gamma


def main() -> None:
    om = landau_root()
    print(f"k lambda_D = {K_LAMBDA_D} (k = {K:.4f}): Landau root omega = "
          f"{om.real:.4f} {om.imag:+.4f}i")

    U_kin = kinetic()
    a_hp, a_hun4 = pade_coefficients(3), two_point_coefficients(4, 2)
    a_opt3, a_opt4 = opt_coefficients(3), opt_coefficients(4)
    model_nn3, model_nn4 = load_naive(), load_n4_super()
    from bot.closures import homog_nn as H
    lam = H.LAM if HOMOG_LAM is None else HOMOG_LAM
    cl3 = H.HomogClosure(H.load(3, path=HOMOG_CKPT.get(3), **HOMOG_KW), lam=lam)
    cl4 = H.HomogClosure(H.load(4, path=HOMOG_CKPT.get(4), **HOMOG_KW), lam=lam)

    runs = [  # (label, U0(t), style, linear roots or None)
        ("Kinetic", U_kin, dict(color="k", ls="none", marker="o", ms=1.3,
                                markevery=25, zorder=10), None),
        (r"HP ($N{=}3$)", evolve(3, linear_closure(a_hp)),
         dict(color="C1", ls="--", lw=0.8), linear_roots(a_hp)),
        (r"Hunana ($N{=}4$)", evolve(4, linear_closure(a_hun4)),
         dict(color="C4", ls=":", lw=0.9), linear_roots(a_hun4)),
        ("Padé opt ($N{=}3$)", evolve(3, linear_closure(a_opt3)),
         dict(color="C2", ls="-.", lw=0.8), linear_roots(a_opt3)),
        ("Padé opt ($N{=}4$)", evolve(4, linear_closure(a_opt4)),
         dict(color="C5", ls="-.", lw=0.8), linear_roots(a_opt4)),
        (r"EKR ($N{=}2$)", evolve(2, ekr_closure),
         dict(color="C0", ls="-", lw=0.9), None),
        (r"NN ($N{=}3$)", evolve(3, nn3_closure(model_nn3)),
         dict(color="C3", ls="-", lw=0.8), None),
        (r"NN ($N{=}4$)", evolve(4, nn4_closure(model_nn4)),
         dict(color="C6", ls="-", lw=1.0), None),
        (HOMOG_LABEL + r" ($N{=}3$)", evolve(3, lambda U: cl3(U, (2.0 / K ** 2) * U[0])),
         dict(color="C8", ls="--", lw=0.9), None),
        (HOMOG_LABEL + r" ($N{=}4$)", evolve(4, lambda U: cl4(U, (2.0 / K ** 2) * U[0])),
         dict(color="C9", ls="--", lw=1.1), None),
    ]

    i10 = np.searchsorted(T, 10.0)
    print(f"\n{'closure':<18}{'omega':>8}{'err':>8}{'gamma':>9}{'err':>8}"
          f"   |U0|/eps @t=10   (linear closure: exact least-damped root)")
    for lab, U0, _, roots in runs:
        w_fit, g_fit = fit_mode(U0)
        extra = ""
        if roots is not None:
            r = roots[np.argmax(roots.imag)]
            extra = f"   ({abs(r.real):.4f}, {r.imag:+.4f})"
        if np.abs(U0[i10]) / EPS < 1e-6:
            print(f"{lab:<18}{'--':>8}{'--':>8}{'--':>9}{'--':>8}   {np.abs(U0[i10])/EPS:.1e}  (collapsed)")
            continue
        print(f"{lab:<18}{w_fit:>8.4f}{100*(w_fit/om.real-1):>+7.1f}%{g_fit:>+9.4f}"
              f"{100*(g_fit/om.imag-1):>+7.1f}%   {np.abs(U0[i10])/EPS:.2e}{extra}")

    with matplotlib.rc_context(RC):
        fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.6), sharex=True)
        for lab, U0, sty, _ in runs:
            axes[0].plot(T, U0.real / EPS, label=lab, **sty)
            axes[1].semilogy(T, np.abs(U0) / EPS, label=lab, **sty)
        axes[1].semilogy(T, np.exp(om.imag * T), color="0.6", lw=0.6, ls="-.",
                         label=r"$e^{\gamma_L t}$")
        axes[0].set_ylabel(r"Re $U_0/\epsilon$")
        axes[0].set_title(rf"Standing Langmuir wave, $k\lambda_D={K_LAMBDA_D}$, "
                          rf"$f_1(0)=\epsilon f_0$")
        axes[1].set_ylabel(r"$|U_0|/\epsilon$")
        axes[1].set_xlabel(r"$t\,\omega_{p}$")
        axes[1].set_ylim(1e-3, 3)
        axes[1].legend(ncol=3, loc="lower left", fontsize=4.6, handlelength=1.6,
                       labelspacing=0.2, borderpad=0.25, handletextpad=0.4,
                       framealpha=0.85)
        fig.tight_layout(h_pad=0.4)
        fig.savefig(SAVE_PNG, dpi=400, bbox_inches="tight")
        plt.close(fig)
    print(f"\nSaved {SAVE_PNG}")


if __name__ == "__main__":
    main()
