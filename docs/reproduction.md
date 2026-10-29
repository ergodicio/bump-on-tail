# Numerical conventions and reproduction details

## Units and closure definitions

Fourier modes vary as `exp(i*k*x - i*omega*t)`; positive Im(omega) means growth.
The beam calculations use omega_pe=v_tb=1, epsilon=n_b/n_0, and beam-frame
velocity s=v-u_b. Maxwellian moments are M_0=1, M_1=0, M_2=1/2, M_3=0.
The fluid state is `(u,E,U_0,...,U_{N-1})`; the kinetic state replaces the
moments by the sampled perturbed distribution F(s)=f_b1/n_b.

The Maxwellian response is R(zeta)=1+zeta*Z(zeta)=-Z'(zeta)/2, with
zeta=(omega-k*u_b)/k and Phi=2iE/k. EKR closes U_2=U_0*beta(U_1/U_0),
where beta=zeta^2+1/(2R). The ratio guard returns zero for |U_0|<1e-15 and
caps |zeta| at 12, as in the source simulations. Slab ITG additionally shifts
the inferred frequency by zeta_star/tau and uses the driven ITG response.
This guard is a numerical regularization. No global complex-analytic extension
through density nodes is asserted. In the standing-wave test EKR collapses at
those nodes; this failure is part of the plotted comparison.

HP (N=3), Hunana (N=4), and learned Padé closures use U_N=a dot U.
The neural closure is U_N=c(G) dot U, where G is the normalized Hermitian Gram
matrix of `(U_0,...,U_{N-1},Phi)`. It is equivariant under common complex
scaling. Its features are real invariants, not holomorphic coordinates.

## Figure parameters

- **Validation:** kappa beam u_b=4, epsilon=.02, k=.295; field kick 1e-3;
  t in [0,110], 1101 samples, kinetic grid 500 points on [-25,25]. Kappa
  values 2,3,4,6,8,12,20,40; all equilibria have M_2=1/2. Growth fits use
  the last half of the record. ITG trace zeta_star=tau=1, eta=10, t in
  [0,25], 501 samples; density kick 1e-3; kinetic grid 600 points on [-8,8].
  ITG growth curves and fit locations are stored explicitly in their NPZ.
  Time-domain ITG growth fits use t in [0,70], 701 samples, last half.
- **Coverage:** training rectangle Re(zeta) in (-3,3), Im(zeta) in (-1,1.5).
  Kinetic field-kick trajectories use 96 points on [-6,6], spacing .1 in
  time, omit t=0, and end at max(60,25+5/gamma_kin). The plot uses U_1/U_0
  of the distribution, not a closure prediction. Scan roots are selected in
  .3<Re(omega)<1.5 and Newton polished. Two-mode and standing roots use the
  same definitions as their own experiments.
- **Growth scan:** u_b=3,4,...,10; epsilon=.01,.02,.05,.10,.15,.20.
  Wavenumbers are the published 80-point scan maxima, fixed in
  `scan_parameters.json`. The IC has only E=1e-3. Growth is fitted to
  log(max(|E|,1e-30)) for t>=25 through max(60,25+5/gamma_kin).
  The rate error is abs(gamma_closure/gamma_kin_fit-1). The 13 cells whose
  kinetic zeta lies outside the rectangle are outlined. These fits carry
  transient and velocity-discretization error; they are not analytic-root
  accuracy measurements.
- **Two-mode tests:** (u_b,k,epsilon)=(1,.5,.05) and (1.5,.4,.05).
  The two selected damped roots have Im(zeta) in (-1,0), Re(zeta) in [0,3.5],
  Im(omega)<-.03, sorted most damped first. Each has field amplitude 1e-3;
  initial moments are sums of the exact single-mode moments. Reference E(t)
  is the sum of the analytic exponentials. Time spans [0,60], 1201 samples.
- **Standing wave:** k*lambda_D=.4, k=sqrt(2)*.4, density perturbation 1e-3,
  U_n(0)=1e-3*M_n. Poisson gives E=-iU_0/k and Phi=2U_0/k^2. Time spans
  [0,40], 2001 samples; kinetic grid 1000 points on [-8,8], trapezoid weights.
  The least-damped root is approximately 1.2851-.0661i.

Beam comparison RK45 tolerances are (rtol,atol)=(1e-9,1e-12) for linear
closures, (1e-8,1e-10) for Maxwellian EKR and NN; kappa EKR uses
(1e-9,1e-12), ITG uses (1e-9,1e-12), standing-wave fluid uses (1e-9,1e-13).
An unsuccessful integration raises an error rather than plotting a truncated
trace. The source sampling, architecture, checkpoint leaves and default JAX
32-bit arithmetic are retained; hardware/library changes can affect the last
few digits of nonlinear trajectories.

## Data schema and provenance

All NPZ files contain numeric arrays readable with `allow_pickle=False`.
`validation_kappa.npz` has t, E_kin, E_hp, E_ekr2, g_hp, g_maxw, g_kin,
g_ekr. `validation_itg_trace.npz` has t_itg1, itg1_params=(zeta_star,tau,eta),
itg1_kin, itg1_hp, itg1_beta, itg1_zeta_kin. `validation_itg_scan.npz` has
eta_curve, eta_fit, z_kin, z_hp, and fit_kin/fit_hp/fit_beta complex rates.

`growth_scan.npz` arrays have shape (8,6), indexed by (u_b,epsilon), with
axis arrays u_b_vals and eps_vals, k_star, gamma_kin (fit), gamma_kin_eig
(analytic root seeded by the kinetic matrix), gamma_kin_exact, and seven
`gamma_<closure>` arrays. Closure keys are hp3, hunana4, opt3, opt4, beta2,
nn3h, nn4h. The `h` denotes the homogeneous neural architecture.

`coverage.npz` stores scan (48 roots in row-major grid order), pairs (4 roots),
standing (2 roots), trajectory (flat complex array), and offsets (49 integer
boundaries into trajectory). `twomode.npz` stores t, and case0_/case1_ prefixes
for roots, params=(u_b,k,epsilon), and eight complex field traces (kinetic plus
the seven closure keys). `standing.npz` stores t, epsilon, root, and the same
eight complex density traces without case prefixes.

`bot/data/source_inputs.json` and `manifest.json` record source revisions,
input hashes and the manuscript mapping. Published caches and checkpoints came
from `prx_figures` at 1fc7b41 (the later 4ddeb05 commit contains an unused
extended-domain experiment). The validation layout and equivariance asset came
from the PPCF manuscript. The latter is a static schematic edited from the
slide-deck illustration, not a simulation result or an executable plot.

`verification.json` records comparisons against the original figure drivers
and regenerated numerical data. Ordinary floating-point tolerances apply; this
is not a claim of byte-identical rendering across Matplotlib versions. The
bundled arrays remain the fixed reference for the paper.
