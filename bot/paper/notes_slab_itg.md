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

## 9. Repairing the direct closure: the ζ̂-shift and the ITG-manifold ratio

The failure mode identified in §5 suggests a fix: on the ITG eigenmode the
exact n = 0 moment equation plus quasineutrality give U_1/U_0 = ζ + ζ_*/τ,
so the mode frequency is recoverable from the moments as

    ζ̂ = U_1/U_0 − ζ_*/τ.

Two repair levels were implemented (`bot/slab_itg.py`, `variant=` argument of
`direct_beta_evolve` / `direct_alpha_evolve`; driver
`bot/figs/fig_slab_itg_direct_variants.py`, data
`bot/runs/slab_itg_direct_variants.npz`):

**(a) Shift only** — U_2 = β(ζ̂)U_0 with the gradient-free β.  Analytically,
the closed N=2 dispersion becomes ζ(ζ+ζ_*/τ) − 1/(2τ) = β(ζ), i.e.
1 − 2ζζ_* + (ζ − 2ζ_*ζ²)Z(ζ) + τ = 0 — exactly the kinetic dispersion
**frozen at η = 2**.  The shift fixes the frequency argument but η still
never enters the system, so it remains η-independent: time-domain fits sit
at ζ ≈ 0.61–0.64 − 0.32…0.35i at every η (the η=2 kinetic root is
0.6086 − 0.3132i), always damped.  Not a usable ITG closure.

**(b) Shift + ITG-manifold closure function** — replace β with the ratio the
kinetic eigenmode actually has.  With I_0 = Z, I_{n+1} = ζI_n + M_n, the
manifold moments are U_n/φ = ζ_*(1−η/2)I_n + ζ_*ηI_{n+2} − I_{n+1}, and

    β_itg(ζ; ζ_*, η) = (U_2/U_0)|_manifold ,   α_itg = (U_3/U_0)|_manifold

evaluated at ζ̂.  The drive now lives inside the closure — resolving the §3
structural objection (η enters the N=2 system through β_itg).

Exactness: the closed-system characteristic equations
ζ(ζ+ζ_*/τ) − 1/(2τ) = β_itg(ζ)  (N=2)  and
ζ[ζ(ζ+ζ_*/τ) − 1/(2τ)] + ζ_*(1+η)/(2τ) = α_itg(ζ)  (N=3)
are satisfied by the kinetic root to machine precision (checked at η = 4.5:
|char| ~ 4e-15).

Spectral structure (Newton root sweep over the UHP, ζ_* = τ = 1):

| η | kinetic root | N=2 β_itg UHP roots | N=3 α_itg UHP roots |
|---|---|---|---|
| 2.5 | none (stable) | none | 0.250 + 1.299i |
| 4.5 | 0.648 + 0.219i | 0.648 + 0.219i only | 0.648+0.219i AND 0.250+1.639i |
| 6.0 | 0.688 + 0.414i | 0.688 + 0.414i only | 0.688+0.414i AND 0.250+1.854i |

* **N=2 β_itg is exact AND spectrally clean**: its only growing root is the
  kinetic one, and none below threshold.  Time-domain fits from a generic
  density IC reproduce the kinetic mode to 4–5 digits at every η ∈ [2.5, 6]
  including the damped branch (η = 2.5: −0.169 vs kinetic −0.170).  This is
  the correct generalization of the BoT direct closure to slab ITG.
* **N=3 α_itg is exact but unstable**: a spurious root at Re ζ = 0.25 with
  γ ≈ 1.1–1.9 grows even where the kinetic system is stable and dominates
  every IC tried (including the eigenmode-projected one) — the same
  exact-but-unstable pathology as the BoT paper §4 NN blow-up.  The spurious
  root is fully analytic: it is a zero of the prefactor polynomial
  P_3 = ζ_*ζ² − ζ/2 + ζ_*(1+η)/2 in the factorized characteristic equation
  (Appendix A.5), giving Re ζ = 1/(4ζ_*) exactly (hence the observed 0.250)
  and γ = √(2ζ_*²(1+η) − 1/4)/(2ζ_*), matching the observed values to all
  digits.
* Caveat: at η ≲ 2 the kinetic initial-value evolution is dominated by a
  fast-damped branch (fit ζ ≈ −2.1 − 0.20i) that the N=2 closed system does
  not contain; β_itg instead sits on its own near-marginal root there
  (ζ ≈ 0.126 − 0.006i at η = 1.5).  Deep in the stable regime the closure
  reproduces the ITG branch, not the full multi-branch transient.

Implication for the NN-closure program: the function an ITG-capable NN
closure must learn is β_itg(ζ̂; ζ_*, η) — i.e. the closure inputs must
include the drive parameters (or equivalently features that resolve them),
and the N=2 level is preferable to N=3, which is spectrally poisoned even
with the exact manifold closure.

## Appendix A: derivation of the ITG-manifold closure β_itg

### A.1 Eigenmode solution of the reduced DKE

On a single eigenmode g, φ ~ e^{−iζt} the reduced DKE (§1)

    ∂g/∂t = −i w g + i φ S(w),   S(w) = ( ζ_*[1 + η(w² − 1/2)] − w ) F0(w)

becomes algebraic: −ζ g = −w g + φ S(w), i.e.

    g(w) = φ S(w) / (w − ζ).                                        (A1)

This is the exact velocity-space shape of the perturbation on the mode; all
manifold quantities below are moments of (A1).

### A.2 Manifold moments, two equivalent ways

**(i) Via Hilbert-transform integrals.**  Define
I_n(ζ) = ∫ wⁿ F0/(w − ζ) dw.  Writing wⁿ⁺¹ = wⁿ(w − ζ) + ζwⁿ gives the
recurrence

    I_0 = Z(ζ),    I_{n+1} = ζ I_n + M_n        (M_n = ∫wⁿF0: 1, 0, 1/2, 0, 3/4, …)

(Z is the Faddeeva-continued plasma dispersion function, valid in both
half-planes).  Expanding S(w) in (A1):

    U_n/φ = ζ_*(1 − η/2) I_n + ζ_* η I_{n+2} − I_{n+1}.             (A2)

This is what `manifold_Un_over_phi` implements.

**(ii) Via the moment equations.**  Equivalently, the eigenmode form of the
exact hierarchy dU_n/dt = −iU_{n+1} + iφσ_n (§3) gives the *manifold
recurrence*

    U_{n+1} = ζ U_n + σ_n φ,     σ_0 = ζ_*,  σ_1 = −1/2,  σ_2 = ζ_*(1+η)/2.  (A3)

Both routes agree identically ((A2) satisfies (A3) by the I_n recurrence).

### A.3 Closed forms in terms of the response function

Since U_0/φ = −R(ζ) by definition of the response (†), iterating (A3) gives
compact closed forms:

    U_1/U_0 = ζ + ζ_* φ/U_0                = ζ − ζ_*/R
    β_itg   ≡ U_2/U_0 = ζ(U_1/U_0) − φ/(2U_0)   = ζ² − (ζζ_* − 1/2)/R(ζ)     (A4)
    α_itg   ≡ U_3/U_0 = ζ β_itg + ζ_*(1+η) φ/(2U_0)
                                            = ζ β_itg − ζ_*(1+η)/(2R(ζ))     (A5)

(verified against (A2) to ~1e-15).  **Gradient-free limit:** at ζ_* = 0,
R = 1 + ζZ = −Z′/2, so (A4) → ζ² − 1/Z′ = β(ζ) and (A5) → ζ³ − ζ/Z′ = α(ζ):
the BoT closures are recovered exactly.  All of the drive dependence of the
manifold closure enters through R(ζ; ζ_*, η) and the explicit ζ_* terms.

### A.4 The frequency estimate ζ̂

The first line of (A4) evaluated at the *eigenmode of the coupled system*
(where quasineutrality fixes R = −τ, i.e. φ/U_0 = 1/τ) gives

    U_1/U_0 = ζ + ζ_*/τ    ⇒    ζ̂ = U_1/U_0 − ζ_*/τ = ζ,

which is the shift used as the closure argument.  (Off the coupled
eigenmode, U_1/U_0 = ζ − ζ_*/R(ζ) ≠ ζ + ζ_*/τ, so ζ̂ is exact only where
the dispersion relation holds — transients incur a controlled ansatz error.)

### A.5 Exactness and spectral structure of the closed systems

Substituting the closures into the closed-system characteristic equations
and using (A4)–(A5), both factor through the kinetic dispersion function:

    N=2:  ζ(ζ + ζ_*/τ) − 1/(2τ) − β_itg(ζ)
          = (ζζ_* − 1/2) (R + τ) / (τ R)                            (A6)

    N=3:  ζ[ζ(ζ + ζ_*/τ) − 1/(2τ)] + ζ_*(1+η)/(2τ) − α_itg(ζ)
          = P_3(ζ) (R + τ) / (τ R),   P_3 = ζ_*ζ² − ζ/2 + ζ_*(1+η)/2  (A7)

So the eigenvalues of the closed fluid systems are exactly the kinetic
dispersion roots (R + τ = 0), **plus the zeros of the prefactor polynomial**
(minus poles at R = 0).  This explains the §9 numerics completely:

* **N=2:** P_2 = ζζ_* − 1/2 has the single real zero ζ = 1/(2ζ_*) — no
  growing spurious mode ever.  The β_itg N=2 closure is not merely "exact on
  the eigenmode": its dispersion relation is *equivalent* to the kinetic one
  (up to an isolated marginal point on the real axis).
* **N=3:** P_3 has the complex-conjugate pair
  ζ = [1/2 ± i√(2ζ_*²(1+η) − 1/4)] / (2ζ_*), one of which is always in the
  UHP.  At ζ_* = 1 this is Re ζ = 1/4 exactly — the "curious" 0.250 observed
  numerically — with γ = √(2(1+η) − 1/4)/2: γ(η=2.5, 4.5, 6) =
  1.29904, 1.63936, 1.85405, matching the observed spurious roots to all
  digits.  The spurious instability is thus an analytic artifact of where
  the closure identity degenerates (the drive part of the n = 2 row), not a
  numerical issue and not kinetic physics; it grows like √η and always
  outruns the ITG mode.

The same factorization viewpoint applied at general N says: the direct
manifold closure at level N is spectrally safe iff P_N(ζ) has no UHP zeros —
a checkable a-priori criterion.  For slab ITG only N=2 passes.

## 10. Spurious-root anatomy: HP vs the direct manifold closures

(Driver `bot/figs/fig_slab_itg_hp_spectrum.py`, figure
`bot/figures/fig_slab_itg_hp_spectrum.png`, data
`bot/runs/slab_itg_hp_spectrum.npz`.)

The Appendix A factorization suggests a taxonomy: a closure's spectrum can
deviate from kinetic either by *approximating the response* R(ζ) or by
*adding prefactor zeros* to an exact response.  HP and the direct manifold
closures sit at opposite corners:

**HP (Γ=3) — approximate response, no extra roots.**  The closed system is
linear and 3×3, so it has exactly three roots, all of them approximations of
true kinetic roots: one tracks the ITG/drift branch (γ error 5–10% above
threshold), one tracks the fast-damped drift branch (ζ ≈ −2), and one deeply
damped root (ζ ≈ 0.3 − 0.8i…1.3i) stands in for the Landau continuum.  There
is no spurious growth anywhere in the (ζ_*, η) plane scanned.

Two sharper statements, both verified numerically:

1. **The Z-substitution identity is exact for Γ = 3.**  The eigenvalues of
   the driven HP fluid matrix coincide with the roots of R_s(Z → Z₃) + τ
   (the corrected `itg_test.py` methodology) to 10⁻⁹ at all (η, root) pairs
   checked.  So the frequency-domain substitution test, invalid for the
   nonlinear direct closures (§7), is rigorous for the HP closure.
2. **HP Γ=3 reproduces the kinetic threshold exactly at every ζ_*** —
   η_th agrees with 1 + √(1 + 2τ(1+τ)/ζ_*²) to 4+ decimals across
   ζ_* ∈ [0.3, 4] (bisection on the 3×3 eigenvalues).  This is structural,
   not a numerical coincidence: given the substitution identity, a marginal
   root is real, and since Im Z₃ ≠ 0 on the real axis (the χ₁ term — HP's
   Landau-damping surrogate), Im D = 0 forces the same Z-independent bracket
   condition ζ = ζ_*(1 − η/2 + ηζ²) as kinetically, and Re D = 0 then gives
   the same polynomial condition 1 + τ − ηζζ_* = 0.  **Any closure that (i)
   satisfies the substitution identity and (ii) retains Im Z_closure ≠ 0 on
   the real axis inherits the exact kinetic marginal boundary**, while its
   growth rates away from marginality carry the response-approximation
   error.  (This generalizes HP's η_th observation to all ζ_*, and is the
   same mechanism behind the closure-independent thresholds seen in the
   fixed `itg_test.py` scan.)

**Γ = 5/3 — breaks the substitution identity, spuriously unstable.**  With
Γ ≠ 3 the closure row no longer respects the exact Maxwellian recurrence,
the substitution identity fails (matrix eigenvalues ≠ substitution roots by
O(1)), and requirement (i) above is lost: the Γ=5/3 system's threshold at
ζ_* = 1 is η_th ≈ 1.03 (growth is weak at first, γ = 0.07 at η = 1.5, which
is why §6 visually reported onset ≈ 2.4) — spuriously unstable across
1 ≲ η ≲ 3.24 where the kinetic system is stable, and η_th drops to ~0.7 at
large ζ_*.  Also note the corrected-`itg_test` Brag curve (Z → Z₃^{Γ=5/3}
substitution) does *not* represent the actual Γ=5/3 fluid system — the real
system is substantially worse than the substitution suggests.

**Direct manifold closures — exact response, prefactor zeros.**  Mirror
image of HP: the tracked root is exact (not 5–10% off), but char_N =
P_N(ζ)(R+τ)/(τR) adds the zeros of P_N.  N=2 is clean (P_2 real zero only;
its threshold curve coincides with kinetic identically); N=3 adds the
always-dominant UHP zero of P_3 with onset η_sp = max(0, 1/(8ζ_*²) − 1) —
i.e. unstable at essentially all η for ζ_* ≳ 0.35.

Summary table (τ = 1):

| closure | tracked-root error | extra/spurious growth | threshold η_th(ζ_*) |
|---|---|---|---|
| HP Γ=3 | γ 5–10% above threshold | none found | exact (structural) |
| HP Γ=5/3 | O(1) | one of its 3 roots grows from η ≈ 1 (ζ_*=1) | badly low |
| direct β_itg N=2 | exact | none (P_2 zero is real) | exact (identical dispersion) |
| direct α_itg N=3 | exact | P_3 zero, γ ~ √η, always dominant | ≈ 0 |
| direct β/α (gradient-free), shifted or not | n/a | η-independent / spurious | none / wrong |

Practical reading: for a *linear* closure the thing to certify is the
substitution identity (then thresholds are free and the only cost is
response error in γ); for a *direct/nonlinear* closure the thing to certify
is the prefactor polynomial P_N (then the response is free and the only risk
is spurious prefactor zeros).  β_itg at N=2 and HP Γ=3 each pass their
respective certificate; α_itg at N=3 and Γ=5/3 each fail theirs.

## 11. AAA rational approximation: a learned/fit alternative to HP's closure

(Single-eigenmode test driver `bot/figs/fig_slab_itg_landau_singlemode.py`,
figure `bot/figures/fig_slab_itg_landau_singlemode.png`; closure code
`bot.closures.pade.aaa_coefficients_n4` and `bot.closures.aaa_pad`.)

### 11.1 Motivation

§10 explains *why* HP gets a genuinely wrong damping rate away from
marginality: its 3-pole closure is derived by matching only two points —
the large-|ζ| asymptotic expansion of Z′(ζ) and the small-ζ Taylor value/
slope at ζ = 0 — with nothing constraining the fit in between.  Checked
directly on a below-threshold Landau-damped root (ζ_* = 1, τ = 1, η = 2,
ζ_root = −2.190 − 0.192i, found via the new `min_damped_root` — the
mirror image of `max_growing_root`, searching the lower half-plane): HP's
rational R(ζ) is 23–68% off from the exact Z′(ζ) at the ζ values relevant
to this session's benchmark modes, only dropping to a few percent once
|ζ| ≳ 4–6.  `scipy.interpolate.AAA` (SciPy ≥ 1.15) replaces the two-point
match with a genuine data fit over a domain, so the natural question is
whether a higher-order, AAA-derived closure closes that gap.

### 11.2 First attempt: AAA folded into the existing N-moment closure family

The direct approach is to stay inside the same closure family HP already
uses — `U_N = Σ_i a_i U_i`, generalized from N=3 to N=4 — and choose the
four complex coefficients `a` by fitting, rather than by two-point
matching.  `scipy.interpolate.AAA` fit to Z′(ξ) along the real axis
(ξ ∈ [−6, 6]; a 2D complex-plane fit was tried first but fails badly —
Z′(ζ) grows ~exponentially as Im(ζ) → −∞, e.g. |Z′| ~ 4.5×10⁻² at ζ=4 vs
~9×10³ at ζ=−2.5i, so AAA's greedy absolute-error point selection is
completely dominated by the large-magnitude corner and ignores the O(1)
region entirely) gives an excellent fit: ~1–5% relative error at the
session's benchmark ζ values, vs HP's 23–68% there.

That accuracy does **not** carry over when forced into the `a_i` family,
and the reason is structural, not a fitting failure: `fluid_U0(ξ; a)` (the
closure's own implied response) has only N complex degrees of freedom,
while a general rational function with N poles needs 2N−1 (poles *and*
residues both free — the residues are *not* independently choosable here,
they're fixed once the poles are fixed by the physical moment recursion).
Two independent checks confirm the cap:

- Setting the closure's own characteristic-polynomial roots equal to
  AAA's fitted poles (elementary symmetric functions of the poles) gives
  ~20–25% error at the benchmark ζ values — no better than HP.
- Building the physically-implied weights from those same poles via the
  dressed-pole basis (`bot.closures.dressed_pole.basis_moments`) gives the
  same ~20–25%, confirming it's the family, not the matching method.

A direct nonlinear least-squares fit of `a_0..a_3` (not constrained to
interpolate any particular points) against Z′(ξ) on the real axis —
`scipy.optimize.least_squares`, `'lm'`, HP-extended initial guess with
random restarts — gets much closer to what the family can actually
achieve: median 1.6% error on-axis (matching AAA there), 6–21% at the
session's off-axis complex test points (vs HP's 23–68% at the same
points).  This is `aaa_coefficients_n4()` in `bot/closures/pade.py`,
usable as a drop-in `a` for both `bot.fluid.fluid_system` (BoT) and the
new `bot.slab_itg.fluid_pade_system`/`fluid_pade_evolve` (the N-moment
generalization of `fluid_hp_system`, verified byte-identical to it when
fed HP's own coefficients).  On the ζ_root = −2.190−0.192i single-mode
benchmark it visibly tracks the true decay far more closely than HP
throughout a 30-time-unit window, but with a real, bounded residual bias
(amplitude ratio to truth settles in [0.28, 0.86], vs HP's drifting from
1.4 to 2.9) — a genuine improvement, not a solved problem.

### 11.3 M4′: fixed multi-pole state-space closure (the real fix)

The cap in §11.2 is the moment-ladder closure *form*, not AAA.  Getting
AAA's actual accuracy requires leaving that form: represent the response
as an explicit sum of poles with independently-fit residues, realized as
auxiliary linear ODE state — one "pad" variable per pole — rather than as
a static `Σ a_i U_i` relation.  (This is "M4′" in
`dressed_pole_closure_handoff.md`'s post-M3′ addendum; M3′, the amortized-
NN pole-inversion scheme from the same document, was tried earlier this
session and found impractical — RHS roughness from the fit/gate/fallback
machinery defeated the adaptive integrator, 1.7M RHS evaluations for a
short window and still not converged.  M4′ replaces "fit poles every
step" with "fit poles once, offline," which sidesteps that failure mode
entirely: no runtime optimizer, no gates, no adaptive integrator.)

**Construction** (`bot/closures/aaa_pad.py`).  Because ITG quasineutrality
is algebraic (φ = U₀/τ, no separate field equation), the pad alone is a
complete closed system — no separate U₁, U₂, U₃ needed.  For a set of
poles {p_j} and residues {c_j} with U₀/φ ≈ Σ_j c_j/(ζ−p_j):

    dw_j/dt = −i p_j w_j − i c_j φ,      U_0 = Σ_j w_j,      φ = U_0/τ

which is linear and time-invariant, so it propagates by matrix
exponential — no RK45, no per-step evaluation at all.

Two design choices, both following the handoff doc:

1. **What to fit.**  The doc's *preferred* option is to fit the universal,
   drive-independent Z′(ζ) and append the pad only to the *tail* of a
   truncated physical moment ladder ("coupled to the chain end"), keeping
   the ζ_*/η drive in the explicit lower-moment source terms exactly as
   HP does.  That requires deriving how the pad couples back into a
   partially-retained ladder and was not completed this session.  Instead
   the *fallback* option was used: fit AAA directly to this problem's own
   U_0/φ = −R_kinetic(ζ; ζ_*, η) (drive baked into the fit).  The pad is
   then a complete standalone closure for that one (ζ_*, η), not reusable
   across a parameter scan without refitting — acceptable since the fit
   itself is cheap (~0.18 s for 15 poles), matching the doc's own note
   that per-background refitting is "milliseconds."
2. **Pole check.**  AAA does not know about causality — a pole with
   Im(p_j) ≥ 0 would make an undriven pad mode spontaneously grow, which
   is unphysical for a linear response of a stable Maxwellian background.
   `fit_aaa_pad_itg` checks this after every fit and, if needed, discards
   offending poles and re-solves the residues by linear least squares on
   the retained set.  For ζ_* = 1, τ = 1, η = 2 (`max_terms=21`): AAA
   converges to 15 poles, all with Im(p_j) ∈ [−2.47, −2.36] — comfortably
   inside the lower half-plane, nothing pruned.

**Accuracy.**  Static fit quality: ~machine precision (relative error
~10⁻¹³–10⁻¹² over the real-axis validation grid *and* at ζ_root itself —
unsurprising given 15 poles fit a function whose magnitude only varies
over roughly a factor of 10 along the real axis for this (ζ_*, η), unlike
the earlier universal-Z′ 2D-domain attempt).  Time-domain, same single-
eigenmode benchmark as §11.2: the amplitude ratio to the true kinetic
decay stays in [1.0000, 1.0002] for the entire 30-time-unit window —
three to four orders of magnitude closer to truth than either HP or the
N=4 moment-ladder fit.

**IC construction.**  `aaa_pad_ic` sets `w_j(0) = c_j·amp/(ζ_0 − p_j)`,
i.e. the pad's own steady-state response to a hypothetical continuous
forcing at frequency ζ_0 — the natural, well-defined way to ask "what
does this pad think a pure mode at ζ_0 looks like," giving
`U_0(0) = amp · (AAA fit of U_0/φ at ζ_0)`, accurate to the fit's own
error at that point (here, machine precision).

### 11.4 Summary

| closure | dof vs. target | fit domain | error at ζ_root = −2.190−0.192i (t=0→30) |
|---|---|---|---|
| HP (Γ=3) | 3 complex, 2-point matched | ζ=0 Taylor + ζ→∞ asymptotic | ratio to truth drifts 1.4 → 2.9 |
| AAA N=4 (least-squares, §11.2) | 4 complex, moment-ladder-constrained | real axis, Z′(ξ) | ratio settles in [0.28, 0.86] |
| AAA pad (M4′, §11.3) | 15 complex poles, unconstrained (state-space) | real axis, −R_kinetic(ζ;ζ_*,η) | ratio in [1.0000, 1.0002] |

The moment-ladder closure family (HP's own form, and its N=4 extension)
is fundamentally dof-limited regardless of how well its coefficients are
fit; the fixed-pole state-space realization removes that ceiling entirely
and — for a single benchmark eigenmode — reproduces the kinetic decay to
the AAA fit's own tolerance.  Not yet run: the fuller M4′ "gauntlet" from
the handoff doc (two-mode beat, the generic-IC case that previously hung
M3-pre, wall-clock/RHS-eval accounting vs. M3′), and the doc's preferred
option A (universal-Z′ closure with the pad appended only to a truncated
ladder, reusable across (ζ_*, η) without refitting).

### 11.5 Limitation: generic (non-eigenmode) ICs are not yet representable

(Driver `bot/figs/fig_slab_itg_aaa_pad_eigenmode.py`, figure
`bot/figures/fig_slab_itg_aaa_pad_eigenmode.png`.)

§11.3's pad tracks only U₀ — there is no separate U₁, U₂ state — because
ITG quasineutrality only ever needs U₀.  That is fine for a single-
eigenmode IC (§11.3), but it means the pad has no way to encode an IC
where U₁, U₂ are specified independently of U₀, e.g. the generic
density-only IC (`density_ic_fluid`/`density_ic_kinetic`: U₀=amp, U₁=0,
U₂=amp/2) used throughout `fig_bot_vs_itg_closures.py` — including its
ITG-1 case (ζ_*=1, τ=1, η=10), the same case redone here on an eigenmode
IC instead, since that is what the current pad can actually do.

A first attempt at fixing this — retain the physical U₀, U₁, U₂ ladder
with its exact σ_n-driven dynamics unchanged, and close U₃ at every step
by a fixed-pole least-squares projection onto the ITG-driven manifold
basis (`manifold_Un_over_phi` at the same 14 AAA poles, not the
gradient-free `basis_moments`) — **fails badly** on the density IC: with
14 poles and only 3 retained moments the per-step weight solve is
underdetermined, `np.linalg.lstsq`'s minimum-norm solution is not
physically meaningful, and the resulting U₃ instantly overdamps the state
(fitted γ comes out **negative** — wrong sign — vs. the true γ=+0.775;
`bot.closures.aaa_pad.aaa_pad_ladder_evolve`, kept for the record but not
used in any figure).  This is not simply a bug to fix with better
regularization: the density IC has genuine continuum/non-pole content (a
smooth F₀-shaped velocity perturbation is not a finite sum of simple
poles), which is exactly the "remainder channel" the original handoff
doc flags as a separate, harder, stretch-goal problem (M5) — no finite-
pole closure, fit however well, represents it exactly.  Properly closing
this gap needs either the doc's preferred option A done correctly (pad
coupled to the chain end, not a per-step re-projection of the whole
state) or an explicit remainder/continuum channel; neither is done here.

On the eigenmode IC (ζ_root = 0.7839+0.7747i, same ζ_*, τ, η as ITG-1):
AAA pad and direct-β both track the true growing rate to within
plotting/fit precision throughout t=0→25 (25 e-foldings, amplitude
growing five orders of magnitude), while HP's ~9.7% low growth-rate error
compounds into roughly two orders of magnitude of amplitude error by the
end of the window — visually obvious on the semilog plot.
