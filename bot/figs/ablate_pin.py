"""Ablation of the origin pinning c(G) = c_pin + N(G) - N(0).

Variants (same data, architecture, schedule):
  asym  pinned to the asymptotically matched Pade closure (HP / Hunana)  [shipped]
  opt   pinned to the least-squares learned Pade closure
  none  no pinning, free output bias
Compared on: offline error near the kick (impulse states, small kappa), the
E-kick transient and growth rate, the standing wave, the Fig. 2 two-mode
case, and the Fig. 3 sweep medians.
"""
import numpy as np
from bot.closures import homog_nn as H
import bot.figs.fig_standing_wave as F
from bot.closed_loop import kinetic_evolve, fluid_eigenmode_ic_n4, fluid_eigenmode_ic_u3
from bot.kinetic import most_unstable_kinetic
from bot.sweeps.sweep_opt_hunana import fit_gamma

RUN = H.RUN_DIR
PINS = ("asym", "opt", "none", "ext")
models = {}
for N in (3, 4):
    # all plain-Gram, lam = 0.3 models (this ablation predates the ext/lam=1 default)
    models[(N, "none")] = H.HomogClosure(H.load(N, path=RUN / f"homog_nn_n{N}_plain_lam0.3.eqx", ext=False), lam=0.3)
    for pin in ("asym", "opt"):
        models[(N, pin)] = H.HomogClosure(H.load(N, path=RUN / f"homog_nn_n{N}_pin{pin}.eqx", pin=pin, ext=False), lam=0.3)
    pe = RUN / f"homog_nn_n{N}_ext_lam0.3_pinasym.eqx"
    if pe.exists():
        models[(N, "ext")] = H.HomogClosure(H.load(N, path=pe, pin="asym", ext=True), lam=0.3)
PINS = tuple(p for p in PINS if (4, p) in models)

print("origin coefficients c(0):")
for (N, pin), cl in sorted(models.items()):
    print(f"  N={N} {pin:<5} {np.round(cl.coeffs(np.zeros(N), 1.0), 3)}")   # (ext: 25 features, same pin)

# --- A: offline, impulse states near the kick --------------------------------
print("\nA. offline relative error on impulse states, median (90th pct)")
for N in (3, 4):
    for pin in PINS:
        cl = models[(N, pin)]; rng = np.random.default_rng(11); out = []
        for lo, hi in ((0.02, 0.1), (0.1, 0.3), (0.3, 1.0), (1.0, 5.0)):
            errs = []
            while len(errs) < 300:
                m = 1 if N == 3 or rng.random() < 0.5 else 2
                kap = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
                U, phi = H.impulse_state([H._draw_zeta(rng) for _ in range(m)], H._sphere_weights(rng, m), kap, N)
                if not np.all(np.isfinite(U)) or abs(U[N]) < 1e-12: continue
                errs.append(abs(cl(U[:N], phi) - U[N]) / abs(U[N]))
            out.append(f"k<{hi:g}: {np.median(errs):.1e} ({np.percentile(errs, 90):.1e})")
        print(f"  N={N} {pin:<5} " + "  ".join(out))

# --- B: E-kick transient + growth rate ---------------------------------------
print("\nB. E-kick growing mode (u_b=4, eps=0.02, k=0.295): max|E-E_kin|/|E_kin| for t<15; gamma error")
k, u_b, eps, amp = 0.295, 4.0, 0.02, 1e-3
g_exact = most_unstable_kinetic(k, u_b, eps).imag
t_end = max(60.0, 25.0 + 5.0 / g_exact); t = np.linspace(0, t_end, int(t_end / 0.1) + 1)
y0k = np.zeros(2 + 96, dtype=complex); y0k[1] = amp
E_kin = kinetic_evolve(k, u_b, eps, t, y0k); g_kin = fit_gamma(t, E_kin)
i15 = np.searchsorted(t, 15.0); i3 = np.searchsorted(t, 3.0)
for (N, pin), cl in sorted(models.items()):
    y0 = np.zeros(2 + N, dtype=complex); y0[1] = amp
    E = H.bot_evolve(k, u_b, eps, cl, t, y0)
    tr15 = np.max(np.abs(E[:i15] - E_kin[:i15]) / np.abs(E_kin[:i15]))
    tr3 = np.max(np.abs(E[1:i3] - E_kin[1:i3]) / np.abs(E_kin[1:i3]))
    print(f"  N={N} {pin:<5} transient t<3: {tr3:.2e}   t<15: {tr15:.2e}   gamma rel err {abs(fit_gamma(t, E)/g_kin - 1):.2e}")

# --- C: standing wave --------------------------------------------------------
print("\nC. standing wave k lambda_D=0.4 (kinetic omega 1.2851, gamma -0.0661)")
om = F.landau_root()
for (N, pin), cl in sorted(models.items()):
    U0 = F.evolve(N, lambda U, cl=cl: cl(U, (2.0 / F.K ** 2) * U[0]))
    w, g = F.fit_mode(U0)
    print(f"  N={N} {pin:<5} omega {w:.4f} ({100*(w/om.real-1):+.1f}%)  gamma {g:+.4f} ({100*(g/om.imag-1):+.1f}%)")

# --- D: Fig. 2 ---------------------------------------------------------------
print("\nD. Fig. 2 two-mode damped case: max |E-E_ref|/amp, t<60")
k2, ub2, eps2 = 0.40, 1.5, 0.05
w1, w2 = 1.205 - 0.119j, 0.717 - 0.106j
t2 = np.linspace(0, 60, 1201); E_ref = amp * (np.exp(-1j * w1 * t2) + np.exp(-1j * w2 * t2))
for (N, pin), cl in sorted(models.items()):
    ic = fluid_eigenmode_ic_u3 if N == 3 else fluid_eigenmode_ic_n4
    E = H.bot_evolve(k2, ub2, eps2, cl, t2, ic(k2, ub2, w1, amp) + ic(k2, ub2, w2, amp))
    print(f"  N={N} {pin:<5} {np.max(np.abs(E - E_ref) / amp):.2e}")

# --- E: sweep ----------------------------------------------------------------
print("\nE. Fig. 3 sweep, relative gamma error median / max")
d = dict(np.load(RUN / "sweep_opt_hunana.npz"))
dh = dict(np.load(RUN / "sweep_opt_hunana_homog.npz"))
for (N, pin), cl in sorted(models.items()):
    if False:
        rel = None
    else:
        gam = np.full_like(d["gamma_kin"], np.nan)
        for i, ub in enumerate(d["u_b_vals"]):
            for j, ep in enumerate(d["eps_vals"]):
                kk, gp = d["k_star"][i, j], d["gamma_kin_eig"][i, j]
                if not np.isfinite(kk): continue
                te = max(60.0, 25.0 + 5.0 / gp); tt = np.linspace(0, te, int(te / 0.1) + 1)
                y0 = np.zeros(2 + N, dtype=complex); y0[1] = amp
                gam[i, j] = fit_gamma(tt, H.bot_evolve(kk, ub, ep, cl, tt, y0))
        rel = np.abs(gam - d["gamma_kin"]) / np.abs(d["gamma_kin"])
    print(f"  N={N} {pin:<5} median {np.nanmedian(rel):.2e}   max {np.nanmax(rel):.2e}")
