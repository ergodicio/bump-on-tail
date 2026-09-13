"""Diagnostics: why does the N=4 NN closure degrade on the standing wave?

Three tests on the kλ_D = 0.4 Maxwellian standing wave (see fig_standing_wave):

  T1  offline: NN alpha4 error along the EXACT two-mode standing-wave
      trajectory, vs. phase in the period.  If the error is concentrated
      where |r_j| is large (near the nodes of U_0), the ratio-input
      singularity is the culprit; if it is uniform, the NN is simply
      inaccurate on this slice of the two-mode manifold.
  T2  closed loop from the EXACT two-mode IC (no ballistic continuum), and
      from the exact SINGLE-mode IC (travelling wave) with and without a
      1e-3 perturbation.  Separates "continuum kicks it off" from
      "manifold is not attracting".
  T3  closed loop, standing wave, with the ratio clip at 3, 12 (repo), 30, 100.
      Sensitivity to the node handling.
"""
from __future__ import annotations

import numpy as np

import bot.figs.fig_standing_wave as F
from bot.closures.inference import alpha4 as alpha4_exact, moment_ratios_n4
from bot.closures.n4_nn import load_super, n4_predict_alpha4
from bot.closed_loop import _safe_xi_ratio

K = F.K
om = F.landau_root()
zL = om / K
zP = -np.conj(zL)                      # the mirror root on e^{ikx}
model = load_super()


def mode_moments(z):
    """(U_0..U_4)/U_0 on a single eigenmode: (1, r1, r2, r3, alpha4)."""
    r1, r2, r3 = moment_ratios_n4(z)
    return np.array([1.0, r1, r2, r3, alpha4_exact(z)], dtype=complex)


def two_mode(t, g1=0.5, g2=0.5):
    """Exact two-mode moments U_0..U_4 (unit total density) at time t."""
    return (g1 * mode_moments(zL) * np.exp(-1j * zL * K * t)
            + g2 * mode_moments(zP) * np.exp(-1j * zP * K * t))


# ---------------------------------------------------------------- T1 ------
print(f"Landau root zeta_L = {zL:.4f}; mirror root {zP:.4f}; period = {2*np.pi/om.real:.3f}")
print("\nT1: NN alpha4 error along the exact standing-wave trajectory (one period)")
period = 2 * np.pi / om.real
ts = np.linspace(0, period, 41)
rows = []
for t in ts:
    U = two_mode(t)
    r = [_safe_xi_ratio(U[j], U[0]) for j in (1, 2, 3)]
    a_nn = complex(n4_predict_alpha4(model, *r))
    a_ex = U[4] / U[0]
    clipped = any(abs(U[j] / U[0]) > 12 for j in (1, 2, 3)) if abs(U[0]) > 1e-15 else True
    rows.append((t / period, abs(U[0]), abs(U[1] / U[0]) if abs(U[0]) > 1e-15 else np.inf,
                 abs(a_nn - a_ex), abs(a_ex), clipped))
print(f"{'t/T':>6}{'|U0|':>8}{'|r1|':>8}{'|dα4|':>10}{'|α4|':>9}  clipped?")
for row in rows[::2]:
    print(f"{row[0]:>6.3f}{row[1]:>8.3f}{row[2]:>8.2f}{row[3]:>10.3f}{row[4]:>9.2f}  {row[5]}")
err = np.array([r[3] for r in rows]); r1 = np.array([r[2] for r in rows])
inn = r1 < 3
print(f"  median |dα4| where |r1|<3: {np.median(err[inn]):.3f}   where |r1|>=3: {np.median(err[~inn]):.3f}")

# also: single-mode (travelling) accuracy for reference
a_nn = complex(n4_predict_alpha4(model, *moment_ratios_n4(zL)))
print(f"  single-mode zeta_L: |NN - exact| = {abs(a_nn - alpha4_exact(zL)):.4f}  (|alpha4| = {abs(alpha4_exact(zL)):.2f})")

# ---------------------------------------------------------------- T2 ------
print("\nT2: closed-loop NN N=4 from exact ICs")
EPS = F.EPS


def run(U0_init, clip=12.0):
    def closure(U):
        r = [_safe_xi_ratio(U[j], U[0], max_mag=clip) for j in (1, 2, 3)]
        return U[0] * complex(n4_predict_alpha4(model, *r))
    return F.evolve(4, closure, U_init=np.asarray(U0_init[:4], dtype=complex))


def summarize(name, U0, ref=None):
    w, g = F.fit_mode(U0)
    i10, i40 = np.searchsorted(F.T, 10.0), np.searchsorted(F.T, 39.9)
    line = (f"  {name:<44} omega {w:.4f} ({100*(w/om.real-1):+.1f}%)  gamma {g:+.4f} "
            f"({100*(g/om.imag-1):+.1f}%)")
    if ref is not None:
        line += f"  max|U0-exact|/eps over t<10: {np.max(np.abs(U0[:i10]-ref[:i10]))/EPS:.2e}"
    print(line)


Uexact = np.array([two_mode(t)[0] for t in F.T]) * EPS
summarize("exact two-mode IC (no continuum)", run(EPS * two_mode(0.0)), Uexact)
# single travelling mode, exact and perturbed
U1m = EPS * mode_moments(zL); ref1 = EPS * np.exp(-1j * om * F.T)
summarize("exact single-mode IC (travelling)", run(U1m), ref1)
U1p = U1m.copy(); U1p[1] *= (1 + 1e-3)
summarize("single-mode IC, r1 perturbed by 1e-3", run(U1p), ref1)

# ---------------------------------------------------------------- T3 ------
print("\nT3: standing wave (density IC), clip sensitivity")
for clip in (3.0, 12.0, 30.0, 100.0):
    summarize(f"density IC, clip {clip:g}", run(EPS * np.array([1, 0, 0.5, 0]), clip=clip))
