# Notes: time-domain tests of the slab-ITG dispersion relation (Hammett–Perkins)

Branch: `slab_itg_timedomain`.  Code: `bot/slab_itg.py`; drivers
`bot/figs/fig_slab_itg_growth.py`, `bot/figs/fig_slab_itg_timedomain.py`.

Goal: rigorous initial-value tests of the slab-ITG problem from
`Slab_ITG_Closure_Relations.pdf` (following HP PRL 64, 3019 (1990)): solve the
parallel slab DKE in Fourier space **in the time domain**, and compare against
the fluid moment hierarchy solved in the time domain with the direct beta
closure (plus HP heat-flux and direct-alpha closures for context).  This goes
beyond `bot/itg_test.py`, which only substitutes Z-function approximations
into the frequency-domain response R_s.

## 1. Model and normalization

Slab DKE (source notes eq. 1–3), perturbations ~ exp(i k_y y + i k_par z − iωt),
Maxwellian f0 with ∂f0/∂x = f0 [1/L_n + (mv²/2T0 − 3/2)/L_T]:

    ∂f/∂t + i k_par v_par f = i (eΦ/T0) ( ω_*[1 + η(mv²/2T0 − 3/2)] − k_par v_par ) f0

with ω_* = (T0/eB) k_y/L_n, η = L_n/L_T.  v_perp enters only through the drive;
integrating over the perpendicular Maxwellian (⟨mv⊥²/2T0⟩ = 1) reduces the
problem **exactly** to 1D with drive η(w² − 1/2), w = v_par/(√2 v_ti):

    ∂g/∂t = −i w g + i φ ( ζ_*[1 + η(w² − 1/2)] − w ) F0(w),   F0 = e^{−w²}/√π

Units: t in (k_par √2 v_ti)^{-1}, ζ = ω/(k_par √2 v_ti), ζ_* likewise,
φ = eΦ/T0i, τ = T0i/T0e.  Boltzmann electrons ñ_e = +n0 eΦ/T0e (standard sign;
the source PDF writes a minus sign in the electron relation, which is
inconsistent with its own dispersion relation — see §5) give quasineutrality
φ = U_0/τ with U_0 = ∫ g dw = ñ_i/n0.

## 2. Kinetic response and dispersion relation

Eigenmode ansatz in the reduced DKE, using ∫F0/(w−ζ) = Z, ∫wF0/(w−ζ) = 1+ζZ,
∫w²F0/(w−ζ) = ζ + ζ²Z:

    R(ζ) = −U_0/φ = 1 − η ζ ζ_* + [ζ − ζ_*(1 − η/2 + η ζ²)] Z(ζ)      (†)

Dispersion: D(ζ) = R(ζ) + τ = 0.  This matches the source PDF eq. (4).
Sanity limits: ζ_* = 0 gives R = 1 + ζZ (Vlasov slab); ζ → 0 gives R → 1
(adiabatic); τ ≪ 1 recovers the ion-acoustic wave ζ² = 1/(2τ).

**Marginal stability (analytic).**  On the real-ζ axis Im Z = √π e^{−ζ²} ≠ 0,
so a marginal root needs both the Z-coefficient bracket and the polynomial
part to vanish:  ζ = (1+τ)/(η ζ_*) and ζ = ζ_*(1 − η/2 + ηζ²).  Eliminating ζ:

    η_th = 1 + sqrt( 1 + 2 τ (1 + τ) / ζ_*² )

At ζ_* = 1, τ = 1: **η_th = 1 + √5 ≈ 3.236**.  The HP value η_th = 2 is the
ζ_* → ∞ (ω ≪ ω_*) limit of this formula.

**Discrepancy with `bot/itg_test.py`.**  `itg_test.py` (`_D_slab`) uses
`R_s = 1 − η ζ_*  + [...]Z` — the first gradient term is missing a factor ζ
relative to (†).  The −ηζζ_* term is forced by the polynomial part of
ζ_* η ∫ w² F0/(w−ζ) dw = ζ_* η (ζ + ζ²Z); it cannot be absorbed by any
convention choice.  With the legacy form the threshold at ζ_* = τ = 1 comes
out as (1+τ)/ζ_* = 2, which is where the "kinetic threshold = 2" claim in
`itg_test.py` came from.  The discretized DKE (independent of any Z algebra)
adjudicates — see §4.1.

## 3. Fluid hierarchy, sources, and closures

Taking w-moments U_n = ∫ wⁿ g dw of the reduced DKE (M_n = ∫wⁿF0: 1, 0, 1/2, 0, 3/4):

    dU_n/dt = −i U_{n+1} + i φ σ_n,   σ_n = ζ_*[M_n + η(M_{n+2} − M_n/2)] − M_{n+1}
    σ_0 = ζ_*        (η exactly cancels: M_2 − M_0/2 = 0)
    σ_1 = −1/2
    σ_2 = ζ_*(1+η)/2  (η first enters here: M_4 − M_2/2 = 1/2)

**Structural observation:** the temperature-gradient drive first enters the
hierarchy at the n = 2 (pressure) equation.  Any N=2 fluid model — including
the direct beta closure U_2 = β(U_1/U_0)U_0, β(z) = z² − 1/Z′(z) — is
therefore *exactly η-independent*: it cannot contain ITG physics no matter
how accurate β is.  This is invisible in the frequency-domain Z-substitution
test of `itg_test.py` (where "U2 exact" trivially reproduces the kinetic
result because it substitutes the exact Z into R_s), and is the central
reason the time-domain test is more rigorous: the Z-substitution is **not**
equivalent to actually closing the driven moment hierarchy.

Closures implemented (`bot/slab_itg.py`):
  * **direct beta** (N=2): U_2 = β(U_1/U_0) U_0 — as in `bot/closed_loop.py`.
  * **direct alpha** (N=3): U_3 = α(U_1/U_0) U_0, α(z) = z³ − z/Z′(z).
  * **Hammett–Perkins** (N=3, linear): q̃ = −n0 χ1 √2 v_t (ik/|k|) T̃ with
    χ1 = 2/√π.  In moment variables (q̃ = n0T0v̄(2U_3 − 3U_1), T̃/T0 = 2U_2 − U_0):
    U_3 = (Γ/2)U_1 − i(χ1/2)(2U_2 − U_0), Γ = 3.
    Verified analytically and numerically that in the gradient-free limit this
    hierarchy reproduces HP's 3-pole response R_3 (`itg_test.R3_hp`) exactly
    (agreement to 6+ digits at spot-checked complex ζ).

Note that on the ITG eigenmode U_1/U_0 = ζ + ζ_*/τ ≠ ζ (the ζ_* source shifts
the moment ratio), so the direct closures β(r1), α(r1) — exact on the
*gradient-free* eigenmode manifold — are being evaluated off their manifold.

## 4. Numerical methods

* **Kinetic:** velocity grid w ∈ [−W, W], N_w points (default 600, W=8);
  dg_j/dt = −i w_j g_j + i S(w_j)(Δw Σg)/τ.  Modes via eigendecomposition
  (ζ = i·eigval, growing ⇔ Im ζ > 0); time evolution via eigendecomposition
  (matrix exponential), same approach as `bot/kinetic.py` /
  `bot/closed_loop.py`.  Recurrence time 2π/Δw ≈ 235 (600 pts) ≫ fit windows.
* **HP fluid:** 3×3 linear system, eigendecomposition.
* **Direct beta / alpha:** nonlinear (homogeneous degree-1) RHS, RK45
  (rtol 1e-9), with the same |U_1/U_0| ≤ 12 safeguard as `bot/closed_loop.py`.
* Growth/frequency extracted from U_0(t) by log-magnitude and unwrapped-phase
  slopes over the second half of the record (`fit_mode`).
* Generic IC: density perturbation g0 = amp·F0(w) (fluid: U_0 = amp,
  U_2 = amp/2 from the same F0 shape, U_1 = 0).

### 4.1 Verification results (ζ_* = 1, τ = 1)

Discretized-DKE most-unstable eigenvalue vs roots of the two dispersion
variants:

| η | D_kinetic (†) root | legacy (itg_test) root | DKE matrix (N_w=400) |
|---|---|---|---|
| 3.0 | no growing root | 0.804 + 0.151i | no growing mode |
| 3.5 | 0.62366 + 0.05175i | 0.871 + 0.170i | 0.62368 + 0.05172i |
| 4.0 | 0.63554 + 0.14041i | 0.919 + 0.181i | 0.63554 + 0.14041i |
| 5.0 | 0.66137 + 0.29021i | 0.982 + 0.194i | 0.66137 + 0.29021i |

The matrix eigenvalues (no Z-function algebra involved) confirm (†) to 4–5
digits and confirm the threshold sits between η = 3.0 and 3.5 (analytic
1+√5 ≈ 3.236), **not** at 2.  The legacy form is unstable at η = 2.5–3 where
the actual DKE has no growing mode.

Time-domain fits reproduce the eigenvalues to 5 digits (η = 4.5):
kinetic fit 0.64828 + 0.21920i vs root 0.64828 + 0.21920i; HP fluid fit
0.61192 + 0.20395i vs eigenvalue 0.61192 + 0.20395i.

## 5. Findings

1. **`itg_test.py` dispersion bug:** the first gradient term of R_s must be
   1 − ηζζ_*, not 1 − ηζ_*.  Consequently the kinetic slab-ITG threshold at
   ζ_* = 1, τ = 1 is η_th = 1 + √5 ≈ 3.236, not 2.  η_th → 2 only as
   ζ_* → ∞.  (The source PDF eq. (4) is correct; its eq. (5) and the
   Boltzmann-electron sign stated above it are internally inconsistent —
   quasineutrality with ñ_e = +n0eΦ/T0e gives D = R + τ, which is the form
   whose roots the discretized DKE reproduces.)
2. **HP 3-moment closure works in the time domain:** with the exact-hierarchy
   sources and U_3 = (3/2)U_1 − i(χ1/2)(2U_2 − U_0), the fluid growth rate
   tracks the kinetic one to ~5–10% above threshold (e.g. η = 5:
   γ = 0.268 vs 0.290) and its threshold is close to the kinetic 3.236.
3. **Direct beta closure (N=2) fails structurally for ITG:** the η drive
   cannot enter the (U_0, U_1) system; it converges to an η-independent
   damped drift-type mode (fit ζ ≈ −1.62 − 0.31i at ζ_* = 1, τ = 1) at every
   η.  The frequency-domain claim "U2 exact = kinetic" in `itg_test.py` is an
   artifact of Z-substitution, not a property of the closed moment system.
4. **Direct alpha closure (N=3) is qualitatively wrong for ITG:** it grows
   far too fast (η = 4.5: γ ≈ 0.60 vs kinetic 0.22) because α(U_1/U_0) is
   exact only on the gradient-free manifold, and the ITG sources shift
   U_1/U_0 off that manifold (U_1/U_0 = ζ + ζ_*/τ on the true eigenmode).
   Closures that are provably exact for the bump-on-tail/Landau problem do
   not transfer to the driven ITG hierarchy; the HP closure — matched to the
   response function rather than to per-mode moment ratios — does.

## 6. Figure results

### `bot/figures/fig_slab_itg_growth.png` (driver `bot/figs/fig_slab_itg_growth.py`)

γ(η) and ω_r(η) at ζ_* = 1, τ = 1 (curves: dispersion roots / eigenvalues;
markers: time-domain fits of U_0(t) from the density-perturbation IC).
Data: `bot/runs/slab_itg_growth.npz`.

* Kinetic time-domain fits sit exactly on the DKE-derived dispersion curve
  (†) on **both** sides of threshold — the damped fits at η = 2.5, 3.0
  (γ = −0.170, −0.050) continue the ITG branch smoothly through marginality
  at η_th = 3.236.  The legacy (itg_test) curve is nowhere consistent with
  the initial-value kinetic solution: it predicts growth from η ≈ 2.1 and
  γ ≈ 0.18–0.19 saturating at large η, while the actual mode reaches
  γ = 0.41 by η = 6.
* HP fluid (Γ=3) tracks the kinetic ITG branch closely on both sides of
  threshold (η = 4: 0.132 vs 0.140; η = 6: 0.379 vs 0.414; threshold within
  ~1% of 3.236).  Γ = 5/3 destroys this: spurious growth from η ≈ 2.4 and a
  much steeper γ(η) — consistent with HP's argument for Γ = 3.
* Direct beta (N=2): γ ≡ −0.267, ω_r ≡ −1.655 at every η (η-independent, as
  required by the hierarchy structure).  No ITG.
* Direct alpha (N=3): spuriously unstable at *all* η scanned, including
  η = 1 (γ = +0.28) where the kinetic system is strongly damped; γ grows to
  ~0.74 at η = 6, nearly double kinetic.  Also its ω_r branch is wrong
  (≈ 0.3–0.5 with the wrong trend).

At η ≤ 2 the kinetic fit picks up a fast-damped branch (ζ ≈ −2.1 − 0.20i)
rather than the ITG branch — with the density IC the ITG-branch residue is
tiny there; the fitted point is a real feature of the initial-value evolution,
not a solver artifact.

### `bot/figures/fig_slab_itg_timedomain.png` (driver `bot/figs/fig_slab_itg_timedomain.py`)

|ñ_i/n0|(t) at η = 2.5, 3.5, 4.5, 6.0, common density IC, with exp(γ_kin t)
guides.  Data: `bot/runs/slab_itg_timedomain.npz`.

* Kinetic and HP-fluid traces overlay through the initial transient and the
  asymptotic phase; the guide slope matches the kinetic trace exactly.
* η = 2.5 (stable): kinetic and HP both show the damped oscillatory ITG
  branch; direct beta decays monotonically on its own drift branch; direct
  alpha grows by 11 decades over t = 60 — a spurious instability of the
  closure, not present in the kinetic system.
* Above threshold the direct-alpha trace grows at roughly twice the kinetic
  rate everywhere; direct beta decays identically in every panel
  (η-independent).

## 7. Takeaway

The frequency-domain Z-substitution test (`itg_test.py`) is not a faithful
proxy for closure performance on slab ITG: (i) its R_s had a typo that moved
the kinetic threshold from 1+√5 to 2, and (ii) substituting a closure's Z
approximation into R_s implicitly assumes the closure commutes with the
gradient drive, which the driven moment hierarchy shows is false — the exact
gradient-free closures (beta, alpha) fail qualitatively in the initial-value
problem, while the HP heat-flux closure (built as a response-function
approximation with the drive entering every moment equation exactly) performs
as advertised.  Rigorous closure tests for ITG must be run as initial-value
problems on the driven hierarchy.

## 8. Comparison with the bump-on-tail testing strategy

The BoT problem was *already* tested in the time domain
(`bot/closed_loop.py`, `bot/sweeps/*`): s-space kinetic matrix vs the closed
moment hierarchy with Padé/HP, direct-beta, direct-alpha, and NN closures,
generic-δE and eigenmode/two-mode ICs, fitted γ_eff.  The slab-ITG machinery
here deliberately mirrors that strategy component-for-component:

| ingredient | BoT (`kinetic.py`, `closed_loop.py`) | slab ITG (`slab_itg.py`) |
|---|---|---|
| kinetic ground truth | s-grid Vlasov–Poisson matrix, eigendecomposition | w-grid DKE matrix, eigendecomposition |
| field response | dynamical bulk (u, E) oscillator (Langmuir) | quasineutral: φ = U_0/τ, slaved instantaneously |
| hierarchy drive | E enters rows n ≥ 1 (n M_{n−1} E); n = 0 row undriven | gradient sources at **every** row (σ_0 = ζ_*, σ_1 = −1/2, σ_2 = ζ_*(1+η)/2) |
| on-manifold ratio | U_1/U_0 = ξ_b exactly | U_1/U_0 = ζ + ζ_*/τ ≠ ζ |
| closure status | direct β/α **exact** per eigenmode | direct β/α off-manifold even for a pure eigenmode |
| dispersion check | D_kin root ↔ matrix eig ↔ time fit | same triple check (§4.1) |

Note `pade_coefficients(3)` = [iχ1/2, 3/2, −iχ1] — the BoT "Padé N=3"
closure IS the HP 3-pole heat-flux closure, so the identical closure can be
run on both problems.

**Side-by-side result** (`bot/figures/fig_bot_vs_itg_closures.png`, driver
`bot/figs/fig_bot_vs_itg_closures.py`, data `bot/runs/bot_vs_itg_closures.npz`;
BoT at u_b = 5, ε = 0.05, k = 0.24, δE IC; ITG at ζ_* = τ = 1, η = 4.5, δn IC):

| closure | BoT fitted mode (kin: 0.91910 + 0.17001i) | ITG fitted mode (kin: 0.64828 + 0.21920i) |
|---|---|---|
| HP 3-pole | 0.92025 + 0.17861i (γ +5.1%) | 0.61192 + 0.20395i (γ −7.0%) |
| direct β (N=2) | 0.91910 + 0.17001i (exact) | −1.615 − 0.314i (decays; no ITG) |
| direct α (N=3) | 0.91910 + 0.17001i (exact) | 0.350 + 0.598i (spurious, ~2.7× γ) |

The closure ranking **flips** between problems.  On BoT the per-mode direct
closures are exact by construction and HP is the ~5% approximation; on ITG
the direct closures inherit an uncontrolled off-manifold error from the
gradient drive and fail qualitatively, while HP degrades only mildly
(5% → 7%).  This is the faithful, like-for-like comparison: the same
closures, the same initial-value strategy, and the difference in outcome is
attributable purely to how the drive enters the hierarchy — i.e., it is a
property of the closures, not of the testing methodology.  Implication for
the BoT program: closures learned/validated on the undriven (BoT/Landau)
manifold should be expected to transfer to driven problems only if the
closure inputs are augmented to carry the drive (e.g. features beyond
U_1/U_0, or response-function-matched forms like HP's).
