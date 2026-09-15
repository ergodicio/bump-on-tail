"""Validate the homogeneous NN closures (bot/closures/homog_nn.py).

  A. offline error by data type (single modes / two-mode / impulse states)
  B. standing Langmuir wave, k lambda_D = 0.4 (Poisson), vs kinetic
  C. bump-on-tail, Fig. 2 two-mode damped case, vs analytic reference
  D. bump-on-tail, Fig. 1 E-kick growing mode, growth-rate error vs kinetic
Old ratio-based NN closures (naive N=3, super N=4), HP and Pade opt are run
alongside for comparison.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

import bot.figs.fig_standing_wave as F
from bot.closures import homog_nn as H
from bot.closures.homog_nn import HomogClosure, load
from bot.closed_loop import (kinetic_evolve, pade_evolve, naive_evolve, nn_n4_evolve,
                             fluid_eigenmode_ic_n4, fluid_eigenmode_ic_u3)
from bot.closures.n4_nn import load_super
from bot.closures.naive_nn import load as load_naive
from bot.closures.opt_hunana import opt_coefficients
from bot.closures.pade import pade_coefficients, two_point_coefficients
from bot.sweeps.sweep_opt_hunana import fit_gamma

MAXW = H.MAXW
cl3, cl4 = HomogClosure(load(3)), HomogClosure(load(4))
old3, old4 = load_naive(), load_super()

# ------------------------------------------------------------------ A ------
print("A. offline relative error |c.U - U_N| / |U_N| (median / 90th pct), fresh samples")
rng = np.random.default_rng(123)
for N, cl in ((3, cl3), (4, cl4)):
    rows = {}
    for kind in ("single", "two-mode", "mirror pair", "impulse kap<1", "impulse 1<kap<5", "impulse kap>5"):
        errs = []
        while len(errs) < 300:
            if kind == "single":
                U, phi = H.mode_state([H._draw_zeta(rng)], [1.0], N)
            elif kind in ("two-mode", "mirror pair"):
                if N == 3:
                    break
                z1 = H._draw_zeta(rng); z2 = -np.conj(z1) if kind == "mirror pair" else H._draw_zeta(rng)
                if abs(z1 - z2) < 0.1:
                    continue
                U, phi = H.mode_state([z1, z2], H._sphere_weights(rng, 2), N)
            else:
                lo, hi = {"impulse kap<1": (0.02, 1), "impulse 1<kap<5": (1, 5), "impulse kap>5": (5, 40)}[kind]
                m = 1 if N == 3 or rng.random() < 0.5 else 2
                zs = [H._draw_zeta(rng) for _ in range(m)]
                kap = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
                U, phi = H.impulse_state(zs, H._sphere_weights(rng, m), kap, N)
            if not np.all(np.isfinite(U)) or abs(U[N]) < 1e-12:
                continue
            errs.append(abs(cl(U[:N], phi) - U[N]) / abs(U[N]))
        if errs:
            rows[kind] = (np.median(errs), np.percentile(errs, 90))
    print(f"  N={N}: " + "   ".join(f"{k}: {a:.1e}/{b:.1e}" for k, (a, b) in rows.items()))

# ------------------------------------------------------------------ B ------
print("\nB. standing Langmuir wave, k lambda_D = 0.4, f1(0) = eps f0   (kinetic: omega 1.2851, gamma -0.0661)")
om = F.landau_root()
U_kin = F.kinetic()


def poisson_closure(cl):
    return lambda U: cl(U, (2.0 / F.K ** 2) * U[0])          # Phi = 2 wp^2/(k^2 vt^2) U0


def report(name, U0):
    w, g = F.fit_mode(U0)
    i10 = np.searchsorted(F.T, 10.0)
    if np.abs(U0[i10]) / F.EPS < 1e-6:
        print(f"  {name:<30} collapsed"); return
    print(f"  {name:<30} omega {w:.4f} ({100*(w/om.real-1):+.1f}%)   gamma {g:+.4f} ({100*(g/om.imag-1):+.1f}%)")


report("Pade opt (N=4)", F.evolve(4, F.linear_closure(opt_coefficients(4))))
report("old NN (N=3), ratios", F.evolve(3, F.nn3_closure(old3)))
report("old NN (N=4), ratios", F.evolve(4, F.nn4_closure(old4)))
report("homog NN (N=3)", F.evolve(3, poisson_closure(cl3)))
report("homog NN (N=4)", F.evolve(4, poisson_closure(cl4)))

# ------------------------------------------------------------------ BoT ----
def bot_evolve(k, u_b, eps, cl, t_grid, y0, rtol=1e-8, atol=1e-10):
    """Repo beam system (u, E, U_0..U_{N-1}) closed by cl(U, Phi), Phi = 2iE/k."""
    N = cl.N

    def rhs(t, y):
        z = y[:N + 2] + 1j * y[N + 2:]
        u, E, U = z[0], z[1], z[2:]
        UN = cl(U, 2j * E / k)
        dU = np.empty(N, dtype=complex)
        for n in range(N):
            nxt = U[n + 1] if n + 1 < N else UN
            dU[n] = -1j * k * u_b * U[n] - 1j * k * nxt + (n * MAXW[n - 1] * E if n > 0 else 0.0)
        dz = np.concatenate([[-E, u - eps * (u_b * U[0] + U[1])], dU])
        return np.concatenate([dz.real, dz.imag])

    sol = solve_ivp(rhs, (t_grid[0], t_grid[-1]), np.concatenate([y0.real, y0.imag]),
                    t_eval=t_grid, method="RK45", rtol=rtol, atol=atol)
    if not sol.success:
        print("   WARNING:", sol.message)
    return sol.y[1] + 1j * sol.y[N + 2 + 1]


# ------------------------------------------------------------------ C ------
print("\nC. bump-on-tail two-mode damped case (Fig. 2): u_b=1.5, k=0.40, eps=0.05; max |E - E_ref|/amp over t<60, and t<20")
k, u_b, eps, amp = 0.40, 1.5, 0.05, 1e-3
w1, w2 = 1.205 - 0.119j, 0.717 - 0.106j
t = np.linspace(0, 60, 1201)
E_ref = amp * np.exp(-1j * w1 * t) + amp * np.exp(-1j * w2 * t)
y0_3 = fluid_eigenmode_ic_u3(k, u_b, w1, amp) + fluid_eigenmode_ic_u3(k, u_b, w2, amp)
y0_4 = fluid_eigenmode_ic_n4(k, u_b, w1, amp) + fluid_eigenmode_ic_n4(k, u_b, w2, amp)
i20 = np.searchsorted(t, 20.0)
runs = [("HP (N=3)", pade_evolve(k, u_b, eps, pade_coefficients(3), t, y0_3, method="rk45")),
        ("Pade opt (N=4)", pade_evolve(k, u_b, eps, opt_coefficients(4), t, y0_4, method="rk45")),
        ("old NN (N=3), ratios", naive_evolve(k, u_b, eps, old3, t, y0_3)),
        ("old NN (N=4), ratios", nn_n4_evolve(k, u_b, eps, old4, t, y0_4)),
        ("homog NN (N=3)", bot_evolve(k, u_b, eps, cl3, t, y0_3)),
        ("homog NN (N=4)", bot_evolve(k, u_b, eps, cl4, t, y0_4))]
for name, E in runs:
    E = np.asarray(E)[:len(t)]
    d = np.abs(E - E_ref) / amp
    print(f"  {name:<30} max dev {np.nanmax(d):.2e}   (t<20: {np.nanmax(d[:i20]):.2e})")

# ------------------------------------------------------------------ D ------
print("\nD. bump-on-tail E-kick growing mode (Fig. 1 case): u_b=4, eps=0.02, k=0.295; |gamma_eff/gamma_kin - 1|")
k, u_b, eps = 0.295, 4.0, 0.02
from bot.kinetic import most_unstable_kinetic
g_exact = most_unstable_kinetic(k, u_b, eps).imag
t_end = max(60.0, 25.0 + 5.0 / g_exact)
t = np.linspace(0, t_end, int(t_end / 0.1) + 1)
y0k = np.zeros(2 + 96, dtype=complex); y0k[1] = amp
g_kin = fit_gamma(t, kinetic_evolve(k, u_b, eps, t, y0k))
print(f"  kinetic fit gamma = {g_kin:.6f} (exact root {g_exact:.6f})")


def kick(n):
    y = np.zeros(2 + n, dtype=complex); y[1] = amp; return y


for name, E in [("HP (N=3)", pade_evolve(k, u_b, eps, pade_coefficients(3), t, kick(3), method="rk45")),
                ("Hunana (N=4)", pade_evolve(k, u_b, eps, two_point_coefficients(4, 2), t, kick(4), method="rk45")),
                ("Pade opt (N=4)", pade_evolve(k, u_b, eps, opt_coefficients(4), t, kick(4), method="rk45")),
                ("old NN (N=3), ratios", naive_evolve(k, u_b, eps, old3, t, kick(3))),
                ("old NN (N=4), ratios", nn_n4_evolve(k, u_b, eps, old4, t, kick(4))),
                ("homog NN (N=3)", bot_evolve(k, u_b, eps, cl3, t, kick(3))),
                ("homog NN (N=4)", bot_evolve(k, u_b, eps, cl4, t, kick(4)))]:
    g = fit_gamma(t, np.asarray(E)[:len(t)])
    print(f"  {name:<30} gamma {g:.6f}   rel err {abs(g/g_kin-1):.2e}")
