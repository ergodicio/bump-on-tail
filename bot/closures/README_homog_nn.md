# Homogeneous NN closure (`bot/closures/homog_nn.py`)

State-dependent linear closure for the N-moment beam hierarchy, replacing the
ratio-based NN closures (`naive_nn.py`, N=3; `n4_nn.py`, N=4).

## Why

The ratio closures `U_N = U_0 · NN(U_j/U_0)` are exactly homogeneous but use a
chart of moment-direction space that is singular on U_0 = 0. A real standing
wave crosses U_0 = 0 twice per period, so every such closure is evaluated at
clipped, off-distribution inputs at each node (`_safe_xi_ratio`, |r| ≤ 12);
the training sampler `w = a/(1+a)` never sees node states and discards
|α4| > 50. Measured on a Maxwellian standing wave (kλ_D = 0.4, f₁ = εf₀):
EKR collapses (Gaussian, no oscillation), N=4 ratio NN gives ω +4.1 %,
γ −40 %, and at kλ_D = 0.3 stops damping. The physics is not the limit: a
two-mode fit in homogeneous coordinates reproduces the same wave to 0.1 %
(`bot/figs/poc_two_mode_homog.py`). The failure is the chart.

## Closure

    U_N = c(G) · U,      c(G) = NN(G)
    V   = (U_0, …, U_{N−1}, Φ̂),      G = V V† / ‖V‖²

* **Inputs**: the (N+1)² real entries of the rank-1 Hermitian matrix G
  (diagonal, Re/Im of the upper triangle). Under (U, Φ̂) → (cU, cΦ̂) the
  products V_iV_j* scale by |c|² and so does ‖V‖², so G is scale- and
  phase-invariant; it is bounded in [−1, 1] and smooth everywhere except the
  trivial state V = 0. No ratios, no clip, no input-normalisation statistics.
* **Output**: N complex coefficients contracted with the moments, so U_N
  transforms like the state (degree-1 homogeneous, phase-equivariant) by
  construction; constant c is a Padé closure. The map U ↦ U_N is nonlinear.
* **Potential as a state component**: Φ̂ is the potential of the hierarchy
  (beam system Φ̂ = 2iE/k; Poisson Φ̂ = 2ω_p²/(k²v_t²)·U₀; quasineutral
  Φ̂ = U₀/τ). On an eigenmode U₀ = −R(ζ)Φ̂, so Φ̂ carries the response
  function directly. Counting: the direction of V has N complex degrees of
  freedom versus 2M−1 for an M-mode superposition, so M ≤ (N+1)/2 — two modes
  at N = 3 (verified: N=3 resolves the standing wave, γ +3.7 %).
* **Origin**: no pinning; the zero-moment state (the E-kick initial
  condition) is regular because G is defined by the potential there.

Conventions (beam frame, v_t = 1): R = 1 + ζZ = −U₀/Φ̂,
ζU_n − U_{n+1} = (n/2)Φ̂M_{n−1}, M = (1, 0, ½, 0, ¾, 0, 15/8).

## Training data (analytic, no simulations)

**Eigenmode superpositions only.** On a mode of frequency ζ, U₀ = −R(ζ)Φ̂
and the recurrence U_{n+1} = ζU_n − (n/2)Φ̂M_{n−1} give every moment; a
state is a sum of such modes, ζ ∈ (−3,3)×(−1,1.5), modal densities on the
unit sphere with random phases (so node states are sampled). Both N=3 and
N=4: 50 % single, 25 % generic pairs, 25 % mirror pairs ζ₂ = −ζ₁* (two modes
are resolvable for N ≥ 3 once Φ̂ is a state component).

Exact impulse-response states (f₁ = 0 driven by a switched-on potential,
U_n(κ) = iⁿ∫₀^κ g_{n+1}(κ−κ′)Φ̂(κ′)dκ′, g_m = d^m e^{−κ²/4}/dκ^m; the
ballistic continuum and the kick→manifold transient) are implemented
(`frac_impulse > 0`) but **off by default**: with the potential in the state
and two-mode data they no longer improve closed-loop results, and cost a
little manifold accuracy (`bot/figs/figs_noimp_current.py`). In the old
moment-only/single-mode-data configuration they were what kept N=3 from
growing spuriously on the standing wave.

400k samples; loss mean w·|c·U − U_N|²/‖V‖² (scale-free), single-mode
samples weighted ×3. MLP 192-192-192 tanh, Adam 2e-3 cosine, 150k steps ×
1024, ~7 min/model on CPU.

## Results (vs ratio NN)

| test | N=4 old → new | N=3 old → new |
|---|---|---|
| sweep (Fig. 3) median / max rel. γ error | 7.2e-5 / 1.4e-3 → **1.3e-5 / 1.3e-4** | 2.8e-4 / 3.9e-3 → **1.0e-4 / 6.4e-4** |
| two-mode damped case (Fig. 2) max \|E−E_ref\|/amp | 3.6e-2 → **6.8e-3** | 0.46 → **0.019** |
| standing wave kλ_D=0.4, ω / γ error | +4.1 % / −40 % → **0.0 % / −0.1 %** | −6.7 % / −18 % → **+1.1 % / +0.5 %** |

For reference EKR (N=2) is 5.5e-5 / 2.1e-3 on the sweep and collapses on the
standing wave. Intermediate configurations (96³ net, impulse data,
single-mode N=3 data) are kept as `*_ext_lam1_imp_96.eqx`,
`*_singlemode_data.eqx`, `*_noimp_current.eqx`, `homog_nn_n3_{wide,long,
widelong}.eqx`; the capacity scan (`bot/figs/train_n3_capacity.py`) showed
the 96³/60k N=3 model was under-trained, not at a limit.

Ablations (same data and schedule): origin pinning to HP/Hunana
(`bot/figs/ablate_pin.py`) and λ = 0.3 vs 1 in S = ‖U‖² + λ²|Φ̂|²
(`bot/figs/ablate_lam.py`) both change results by ≲ 2× in either direction,
within training noise; the moment-only Gram (ext=False) loses the extra
mode at N = 3 (standing wave γ −12 % to −22 %); training without
impulse-response states (`bot/figs/ablate_impulse.py`) makes N = 3 grow
spuriously on the standing wave.

## Files

* `bot/closures/homog_nn.py` — kernels, data, model, `train`, `load`,
  `HomogClosure(model)(U, Φ̂)`, `bot_evolve` (repo beam system).
* `bot/runs/homog_nn_n3.eqx`, `homog_nn_n4.eqx` — shipped checkpoints
  (joint-state Gram, λ = 1, unpinned, modes-only data, 192³/150k;
  `python -m bot.closures.homog_nn 3 4` retrains). Ablation variants: `*_plain_lam0.3.eqx`, `*_plain_lam1.eqx`
  (moment-only Gram), `*_pinasym.eqx`, `*_pinopt.eqx`,
  `*_ext_lam0.3_pinasym.eqx`, `*_noimp.eqx`; load with the matching
  `pin`/`ext`/`lam` arguments.
* `bot/figs/validate_homog_nn.py` — offline errors + tests B–D above.
* `bot/figs/fig_twomode_homog.py` — Fig. 2 via the manuscript's figure code
  (monkeypatches the two NN evolve functions).
* `bot/sweeps/sweep_homog.py` — Fig. 3 on the stored k*, γ_kin grid →
  `bot/runs/sweep_opt_hunana_homog.npz`, `bot/figures/homog/`.
* `bot/figs/fig_standing_wave.py`, `diag_standing_wave_nn.py`,
  `poc_two_mode_homog.py` — the standing-wave tests that motivated this;
  `fig_standing_wave_supp.py` — the supplement figure (Fig. 2 closure set).

## Open items

* Causality: c(G) implies a rational response at each state; pole locations
  are computable and could be monitored or projected during training.
* Seed variance has not been measured; 2× differences between variants on
  the sweep median are not yet distinguishable from noise.
* Two-mode / mirror-pair offline error (1–5 %) is the weakest region; more
  capacity or a per-region loss weight would help before pushing beyond N=4.
