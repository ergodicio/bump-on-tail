"""Ablation: homogeneous closure trained WITHOUT impulse-response data.

Same architecture, data size and schedule as the shipped checkpoints, but
frac_impulse = 0 (eigenmode superpositions only).  Compares against the
shipped (with-impulse) models on: offline error by state type, the E-kick
transient and growth rate (Fig. 1 case), the standing wave, the Fig. 2
two-mode case, and the Fig. 3 sweep medians.
"""
from pathlib import Path
import numpy as np
from bot.closures import homog_nn as H
import bot.figs.fig_standing_wave as F
from bot.closed_loop import kinetic_evolve, fluid_eigenmode_ic_n4, fluid_eigenmode_ic_u3
from bot.kinetic import most_unstable_kinetic
from bot.sweeps.sweep_opt_hunana import fit_gamma

RUN = H.RUN_DIR
models = {}
for N in (3, 4):
    p = RUN / f"homog_nn_n{N}_noimp.eqx"
    if not p.exists():
        print(f"=== training N={N} without impulse data")
        H.save(H.train(N, frac_impulse=0.0, pin="asym", ext=False, lam=0.3), N, path=p)
    # plain-Gram, lam = 0.3, asym-pinned models (this ablation predates the ext/lam=1 default)
    models[(N, "no impulse")] = H.HomogClosure(H.load(N, path=p, pin="asym", ext=False), lam=0.3)
    models[(N, "with impulse")] = H.HomogClosure(H.load(N, path=RUN / f"homog_nn_n{N}_pinasym.eqx", pin="asym", ext=False), lam=0.3)

# --- offline ---------------------------------------------------------------
print("\nA. offline relative error, median (90th pct)")
rng = np.random.default_rng(7)
kinds = ["single", "two-mode", "impulse k<0.3", "impulse 0.3<k<1", "impulse 1<k<5", "impulse k>5"]
for N in (3, 4):
    for tag in ("no impulse", "with impulse"):
        cl = models[(N, tag)]; rng = np.random.default_rng(7); row = []
        for kind in kinds:
            errs = []
            while len(errs) < 300:
                if kind == "single":
                    U, phi = H.mode_state([H._draw_zeta(rng)], [1.0], N)
                elif kind == "two-mode":
                    if N == 3: break
                    z1, z2 = H._draw_zeta(rng), H._draw_zeta(rng)
                    if abs(z1 - z2) < 0.1: continue
                    U, phi = H.mode_state([z1, z2], H._sphere_weights(rng, 2), N)
                else:
                    lo, hi = {"impulse k<0.3": (0.02, 0.3), "impulse 0.3<k<1": (0.3, 1), "impulse 1<k<5": (1, 5), "impulse k>5": (5, 40)}[kind]
                    m = 1 if N == 3 or rng.random() < 0.5 else 2
                    kap = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
                    U, phi = H.impulse_state([H._draw_zeta(rng) for _ in range(m)], H._sphere_weights(rng, m), kap, N)
                if not np.all(np.isfinite(U)) or abs(U[N]) < 1e-12: continue
                errs.append(abs(cl(U[:N], phi) - U[N]) / abs(U[N]))
            row.append(f"{kind}: {np.median(errs):.1e} ({np.percentile(errs, 90):.1e})" if errs else "")
        print(f"  N={N} {tag:<13} " + "  ".join(r for r in row if r))

# --- E-kick: transient + growth rate ---------------------------------------
print("\nB. E-kick growing mode (u_b=4, eps=0.02, k=0.295): transient error max|E-E_kin|/|E_kin| for t<15, and gamma error")
k, u_b, eps, amp = 0.295, 4.0, 0.02, 1e-3
g_exact = most_unstable_kinetic(k, u_b, eps).imag
t_end = max(60.0, 25.0 + 5.0 / g_exact)
t = np.linspace(0, t_end, int(t_end / 0.1) + 1)
y0k = np.zeros(2 + 96, dtype=complex); y0k[1] = amp
E_kin = kinetic_evolve(k, u_b, eps, t, y0k); g_kin = fit_gamma(t, E_kin)
i15 = np.searchsorted(t, 15.0)
for (N, tag), cl in sorted(models.items()):
    y0 = np.zeros(2 + N, dtype=complex); y0[1] = amp
    E = H.bot_evolve(k, u_b, eps, cl, t, y0)
    tr = np.max(np.abs(E[:i15] - E_kin[:i15]) / np.abs(E_kin[:i15]))
    print(f"  N={N} {tag:<13} transient {tr:.2e}   gamma rel err {abs(fit_gamma(t, E)/g_kin - 1):.2e}")

# --- standing wave -----------------------------------------------------------
print("\nC. standing wave k lambda_D=0.4 (kinetic omega 1.2851, gamma -0.0661)")
om = F.landau_root()
for (N, tag), cl in sorted(models.items()):
    U0 = F.evolve(N, lambda U, cl=cl: cl(U, (2.0 / F.K ** 2) * U[0]))
    w, g = F.fit_mode(U0)
    print(f"  N={N} {tag:<13} omega {w:.4f} ({100*(w/om.real-1):+.1f}%)  gamma {g:+.4f} ({100*(g/om.imag-1):+.1f}%)")

# --- Fig. 2 two-mode ---------------------------------------------------------
print("\nD. Fig. 2 two-mode damped case: max |E-E_ref|/amp, t<60")
k, u_b, eps = 0.40, 1.5, 0.05
w1, w2 = 1.205 - 0.119j, 0.717 - 0.106j
t2 = np.linspace(0, 60, 1201)
E_ref = amp * (np.exp(-1j * w1 * t2) + np.exp(-1j * w2 * t2))
for (N, tag), cl in sorted(models.items()):
    ic = fluid_eigenmode_ic_u3 if N == 3 else fluid_eigenmode_ic_n4
    E = H.bot_evolve(k, u_b, eps, cl, t2, ic(k, u_b, w1, amp) + ic(k, u_b, w2, amp))
    print(f"  N={N} {tag:<13} {np.max(np.abs(E - E_ref) / amp):.2e}")

# --- sweep -------------------------------------------------------------------
print("\nE. Fig. 3 sweep, relative gamma error median / max")
d = dict(np.load(RUN / "sweep_opt_hunana.npz"))
for (N, tag), cl in sorted(models.items()):
    if tag == "with impulse":
        key = f"gamma_nn{N}h"
        dh = dict(np.load(RUN / "sweep_opt_hunana_homog.npz"))
        rel = np.abs(dh[key] - dh["gamma_kin"]) / np.abs(dh["gamma_kin"])
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
    print(f"  N={N} {tag:<13} median {np.nanmedian(rel):.2e}   max {np.nanmax(rel):.2e}")
