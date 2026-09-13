"""lambda = 0.3 (current) vs lambda = 1, plain and joint-state Gram, unpinned.
Tests: offline error by state type, E-kick transient/gamma, standing wave,
Fig. 2, Fig. 3 sweep medians."""
import numpy as np
from bot.closures import homog_nn as H
import bot.figs.fig_standing_wave as F
from bot.closed_loop import kinetic_evolve, fluid_eigenmode_ic_n4, fluid_eigenmode_ic_u3
from bot.kinetic import most_unstable_kinetic
from bot.sweeps.sweep_opt_hunana import fit_gamma

RUN = H.RUN_DIR
models = {}
for N in (3, 4):
    models[(N, "lam0.3")] = H.HomogClosure(H.load(N, path=RUN / f"homog_nn_n{N}_plain_lam0.3.eqx", ext=False), lam=0.3)
    models[(N, "lam0.3 ext")] = H.HomogClosure(H.load(N, path=RUN / f"homog_nn_n{N}_ext_lam0.3_pinasym.eqx", pin="asym", ext=True), lam=0.3)
    models[(N, "lam1")] = H.HomogClosure(H.load(N, path=RUN / f"homog_nn_n{N}_plain_lam1.eqx", ext=False), lam=1.0)
    models[(N, "lam1 ext")] = H.HomogClosure(H.load(N))                                   # shipped default
TAGS = ("lam0.3", "lam1", "lam0.3 ext", "lam1 ext")

print("A. offline relative error, median (90th pct)  [ext lam0.3 was trained with the asym pin]")
for N in (3, 4):
    for tag in TAGS:
        cl = models[(N, tag)]; rng = np.random.default_rng(5); out = []
        for kind in ("single", "two-mode", "impulse k<1", "impulse 1<k<5"):
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
                    lo, hi = (0.02, 1.0) if kind == "impulse k<1" else (1.0, 5.0)
                    m = 1 if N == 3 or rng.random() < 0.5 else 2
                    U, phi = H.impulse_state([H._draw_zeta(rng) for _ in range(m)], H._sphere_weights(rng, m),
                                             float(np.exp(rng.uniform(np.log(lo), np.log(hi)))), N)
                if not np.all(np.isfinite(U)) or abs(U[N]) < 1e-12: continue
                errs.append(abs(cl(U[:N], phi) - U[N]) / abs(U[N]))
            if errs: out.append(f"{kind}: {np.median(errs):.1e} ({np.percentile(errs, 90):.1e})")
        print(f"  N={N} {tag:<11} " + "  ".join(out))

print("\nB. E-kick growing mode (u_b=4, eps=0.02, k=0.295): transient t<15; gamma error")
k, u_b, eps, amp = 0.295, 4.0, 0.02, 1e-3
g_exact = most_unstable_kinetic(k, u_b, eps).imag
t_end = max(60.0, 25.0 + 5.0 / g_exact); t = np.linspace(0, t_end, int(t_end / 0.1) + 1)
y0k = np.zeros(2 + 96, dtype=complex); y0k[1] = amp
E_kin = kinetic_evolve(k, u_b, eps, t, y0k); g_kin = fit_gamma(t, E_kin); i15 = np.searchsorted(t, 15.0)
for N in (3, 4):
    for tag in TAGS:
        y0 = np.zeros(2 + N, dtype=complex); y0[1] = amp
        E = H.bot_evolve(k, u_b, eps, models[(N, tag)], t, y0)
        print(f"  N={N} {tag:<11} transient {np.max(np.abs(E[:i15]-E_kin[:i15])/np.abs(E_kin[:i15])):.2e}   gamma rel err {abs(fit_gamma(t, E)/g_kin-1):.2e}")

print("\nC. standing wave k lambda_D=0.4 (kinetic omega 1.2851, gamma -0.0661)")
om = F.landau_root()
for N in (3, 4):
    for tag in TAGS:
        cl = models[(N, tag)]
        w, g = F.fit_mode(F.evolve(N, lambda U, cl=cl: cl(U, (2.0 / F.K ** 2) * U[0])))
        print(f"  N={N} {tag:<11} omega {w:.4f} ({100*(w/om.real-1):+.1f}%)  gamma {g:+.4f} ({100*(g/om.imag-1):+.1f}%)")

print("\nD. Fig. 2 two-mode damped case: max |E-E_ref|/amp, t<60")
k2, ub2, eps2 = 0.40, 1.5, 0.05; w1, w2 = 1.205 - 0.119j, 0.717 - 0.106j
t2 = np.linspace(0, 60, 1201); E_ref = amp * (np.exp(-1j * w1 * t2) + np.exp(-1j * w2 * t2))
for N in (3, 4):
    ic = fluid_eigenmode_ic_u3 if N == 3 else fluid_eigenmode_ic_n4
    for tag in TAGS:
        E = H.bot_evolve(k2, ub2, eps2, models[(N, tag)], t2, ic(k2, ub2, w1, amp) + ic(k2, ub2, w2, amp))
        print(f"  N={N} {tag:<11} {np.max(np.abs(E - E_ref) / amp):.2e}")

print("\nE. Fig. 3 sweep, relative gamma error median / max")
d = dict(np.load(RUN / "sweep_opt_hunana.npz"))
for N in (3, 4):
    for tag in TAGS:
        cl = models[(N, tag)]; gam = np.full_like(d["gamma_kin"], np.nan)
        for i, ub in enumerate(d["u_b_vals"]):
            for j, ep in enumerate(d["eps_vals"]):
                kk, gp = d["k_star"][i, j], d["gamma_kin_eig"][i, j]
                if not np.isfinite(kk): continue
                te = max(60.0, 25.0 + 5.0 / gp); tt = np.linspace(0, te, int(te / 0.1) + 1)
                y0 = np.zeros(2 + N, dtype=complex); y0[1] = amp
                gam[i, j] = fit_gamma(tt, H.bot_evolve(kk, ub, ep, cl, tt, y0))
        rel = np.abs(gam - d["gamma_kin"]) / np.abs(d["gamma_kin"])
        print(f"  N={N} {tag:<11} median {np.nanmedian(rel):.2e}   max {np.nanmax(rel):.2e}")
