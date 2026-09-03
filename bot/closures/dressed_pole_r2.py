"""Dressed-pole r=2 multi-pole closure via amortized (NN) inversion -- M3'.

See dressed_pole_closure_handoff.md addendum ("M3': amortized inversion").
Classical VARPRO for r=2 basin-hops the same way r=1's did (worse, in a
higher-dimensional parameter space) -- see bot/closures/dressed_pole.py's
documented r=1 jump.  The fix here is architectural, not a better optimizer:

    moment features -> [NN] -> monic poly coeffs (c1, c0)
                     -> companion-matrix eigenproblem -> poles (zeta_1, zeta_2)
                     -> analytic linear weight solve -> (w_1, w_2), residual

The NN predicts ONLY permutation-invariant coefficients (elementary
symmetric functions of the poles) -- smooth through root collisions by
construction, unlike labeled/ordered roots, which must jump when two poles
cross. Weights are NEVER learned: always a closed-form r x r linear
least-squares solve against the basis moments, so the residual is an
honest, differentiable-if-needed diagnostic, and homogeneity of the
closure in the state is automatic.

This module is specialized to the slab-ITG basis at a FIXED (zeta_star,
tau, eta) -- the benchmark point zeta_*=1, tau=1, eta=10 used throughout
this project (bot.slab_itg.manifold_Un_over_phi) -- not yet generalized
to a network conditioned on the drive parameters (a natural extension,
not attempted here). The classical r=1 gated closure this extends lives
in bot/closures/dressed_pole.py.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from bot.slab_itg import manifold_Un_over_phi

SQRT_PI = float(np.sqrt(np.pi))


# ---------------------------------------------------------------------------
# Classical r=2 machinery (weight solve, companion roots, VARPRO) -- all
# analytic; used for training-data ground truth, M3'b comparison, and the
# optional post-NN polish step.
# ---------------------------------------------------------------------------

def weights_given_poles(U: np.ndarray, zetas: np.ndarray, basis_fn):
    """Linear least-squares weights given r pole locations.

    M[:, j] = m_n(zeta_j) for n=0..N; w = argmin_w ||U - M w||_2.
    Returns (w, residual).
    """
    N = len(U) - 1
    M = np.column_stack([np.asarray(basis_fn(z, N), dtype=complex)
                         for z in zetas])
    w, _, _, _ = np.linalg.lstsq(M, U, rcond=None)
    residual = float(np.linalg.norm(U - M @ w))
    return w, residual


# Same rationale/value as bot.closed_loop._safe_xi_ratio's _XI_RATIO_MAX and
# bot.slab_itg's off-manifold clipping: Z(zeta)/the basis moments overflow
# for |zeta| much beyond where any physically relevant root of this problem
# lives (observed roots stay <~ 3.3 even mid-transient); clip proposed poles
# there before ever evaluating the basis, rather than discovering the
# overflow deep inside a linear-algebra call (confirmed failure mode: an
# uncaught companion-matrix/LAPACK NaN crashed or stalled an integration
# for many CPU-minutes after otherwise-fine early progress).
_ZETA_MAG_MAX = 15.0


def _clip_zeta(z: complex, max_mag: float = _ZETA_MAG_MAX) -> complex:
    mag = abs(z)
    return z * (max_mag / mag) if mag > max_mag else z


def companion_roots_r2(c1: complex, c0: complex) -> np.ndarray:
    """Roots of the monic quadratic z^2 + c1 z + c0 via companion matrix.

    (For r=2 the quadratic formula would be simpler/faster, but the
    companion-matrix route is what generalizes to r>2 without change, per
    the handoff's prescribed architecture.)
    """
    C = np.array([[0.0, -c0], [1.0, -c1]], dtype=complex)
    return np.linalg.eigvals(C)


def poly_coeffs_from_roots(zetas: np.ndarray) -> tuple:
    """Monic quadratic coefficients (c1, c0) from roots: z^2+c1 z+c0, r=2."""
    z1, z2 = zetas[0], zetas[1]
    return -(z1 + z2), z1 * z2


def fit_r2_varpro(U: np.ndarray, basis_fn, zetas0: np.ndarray):
    """Classical two-pole VARPRO fit (separable nonlinear least squares).

    zetas0: length-2 initial guess. Returns (w, zetas, residual). This is
    the "classical pencil/VARPRO" M3'b baseline -- prone to the same
    basin-hopping pathology as r=1's fit_r1 when used inside a time
    integrator (that's the whole reason M3' exists), but perfectly fine as
    a one-shot fit for comparison/ground truth generation.
    """
    N = len(U) - 1

    def resid(x):
        zetas = np.array([complex(x[0], x[1]), complex(x[2], x[3])])
        w, _ = weights_given_poles(U, zetas, basis_fn)
        M = np.column_stack([np.asarray(basis_fn(z, N), dtype=complex)
                             for z in zetas])
        r = U - M @ w
        return np.concatenate([r.real, r.imag])

    x0 = [zetas0[0].real, zetas0[0].imag, zetas0[1].real, zetas0[1].imag]
    sol = least_squares(resid, x0, method="lm", xtol=1e-14, ftol=1e-14,
                        gtol=1e-14, max_nfev=2000)
    zetas = np.array([complex(sol.x[0], sol.x[1]),
                      complex(sol.x[2], sol.x[3])])
    w, residual = weights_given_poles(U, zetas, basis_fn)
    return w, zetas, residual


def polish_r2(U: np.ndarray, basis_fn, zetas0: np.ndarray, n_iter: int = 2,
             lam: float = 1e-3, max_mag: float = _ZETA_MAG_MAX):
    """FIXED-count (n_iter, never "until converged") Gauss-Newton polish.

    Per the handoff: a variable iteration count reintroduces the RHS
    discontinuity M3' exists to remove (an optimizer that sometimes takes
    3 steps and sometimes 30 near degeneracy is itself a discontinuous
    function of the state). Always takes exactly n_iter damped
    Gauss-Newton steps from zetas0, using a small fixed Levenberg damping
    lam (not adapted, for the same reason). Returns (w, zetas, residual).

    CRITICAL: resid() clips its zeta argument to max_mag before ever
    evaluating basis_fn (which calls Z(zeta) -- overflows for large
    |zeta|).  Earlier versions only clipped the function's final output,
    not the values visited mid-iteration; an unclipped Gauss-Newton step
    (or an unclipped finite-difference Jacobian probe) could send x to a
    huge or NaN value that corrupted every subsequent basis-function
    evaluation and, downstream, np.linalg.eigvals/lstsq (confirmed
    failure mode: floods of LAPACK "illegal value" errors from
    DLASCL/ZLASCL, and either a hard crash or many-minutes-long adaptive-
    integrator stalls). Clipping inside resid() means every quantity this
    function ever touches -- Jacobian probes included -- stays bounded.
    """
    N = len(U) - 1

    def clipped_zetas(xv):
        z1 = _clip_zeta(complex(xv[0], xv[1]), max_mag)
        z2 = _clip_zeta(complex(xv[2], xv[3]), max_mag)
        return np.array([z1, z2])

    def resid(xv):
        zetas = clipped_zetas(xv)
        w, _ = weights_given_poles(U, zetas, basis_fn)
        M = np.column_stack([np.asarray(basis_fn(z, N), dtype=complex)
                             for z in zetas])
        r = U - M @ w
        out = np.concatenate([r.real, r.imag])
        if not np.all(np.isfinite(out)):
            return np.full_like(out, 1e6)  # bounded "very bad" signal, never NaN
        return out

    x = np.array([zetas0[0].real, zetas0[0].imag,
                  zetas0[1].real, zetas0[1].imag])
    h = 1e-6
    for _ in range(n_iter):
        r0 = resid(x)
        J = np.empty((len(r0), 4))
        for i in range(4):
            xp = x.copy(); xp[i] += h
            J[:, i] = (resid(xp) - r0) / h
        JTJ = J.T @ J + lam * np.eye(4)
        JTr = J.T @ r0
        try:
            step = np.linalg.solve(JTJ, JTr)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break
        x = x + step

    zetas = clipped_zetas(x)
    w, residual = weights_given_poles(U, zetas, basis_fn)
    return w, zetas, residual


# ---------------------------------------------------------------------------
# ITG basis at the fixed benchmark point (zeta_* = 1, tau = 1, eta = 10)
# ---------------------------------------------------------------------------

ZETA_STAR, TAU, ETA = 1.0, 1.0, 10.0


def itg_basis(zeta: complex, N: int) -> np.ndarray:
    return np.asarray(manifold_Un_over_phi(zeta, ZETA_STAR, ETA, n_max=N))


# ---------------------------------------------------------------------------
# M3'a: synthetic training data (exact by construction, no simulation)
# ---------------------------------------------------------------------------
# Retained moments U_0..U_4 (N=4, 5 moments) -> predict monic quadratic
# coefficients (c1, c0). N=4 is the largest n_max bot.slab_itg's tabulated
# Maxwellian moments support without extension beyond M_7 (see there);
# 5 moments is a comfortable overdetermination for r=2 (4 real unknowns
# w_1,w_2 given zeta_1,zeta_2; c1,c0 themselves are 4 more reals -> total
# problem is well-posed with margin).

N_RETAIN = 4  # U_0..U_4

# Anchors for oversampling the physically relevant root loci: the true
# growing kinetic root at (zeta_*=1,tau=1,eta=10), and damped/transient
# branches observed in the ITG generic-IC transient trace (§ dressed_pole
# module docstring / fig_dressed_pole_itg_gated.png bottom-right panel).
_PHYSICAL_ANCHORS = [0.7839 + 0.7747j, -0.55 - 0.30j, -2.0 - 0.70j,
                     -2.35 - 0.60j, 1.0 + 0.0j]


def sample_training_point(rng: np.random.Generator):
    """Draw (w1, zeta1, w2, zeta2) from a mixture of three regimes.

    40% generic complex-plane region, 30% near the physical ITG root
    loci (oversampled per the handoff's "confirmed choice"), 30% near-
    degenerate pairs (the beat regime that broke classical VARPRO).
    """
    mode = rng.choice(3, p=[0.4, 0.3, 0.3])
    if mode == 0:  # generic
        z1 = complex(rng.uniform(-3, 2), rng.uniform(-1.5, 3.5))
        z2 = complex(rng.uniform(-3, 2), rng.uniform(-1.5, 3.5))
    elif mode == 1:  # physical anchors
        a1 = _PHYSICAL_ANCHORS[rng.integers(len(_PHYSICAL_ANCHORS))]
        a2 = _PHYSICAL_ANCHORS[rng.integers(len(_PHYSICAL_ANCHORS))]
        z1 = a1 + complex(rng.normal(0, 0.25), rng.normal(0, 0.25))
        z2 = a2 + complex(rng.normal(0, 0.25), rng.normal(0, 0.25))
    else:  # near-degenerate (beat regime)
        center = complex(rng.uniform(-2.5, 1.5), rng.uniform(-1, 2.5))
        sep_mag = 10 ** rng.uniform(-3, -0.3)
        sep_phase = rng.uniform(0, 2 * np.pi)
        sep = sep_mag * np.exp(1j * sep_phase)
        z1 = center + sep / 2
        z2 = center - sep / 2

    mag1 = 10 ** rng.uniform(-1.0, 0.5)
    mag2 = 10 ** rng.uniform(-1.0, 0.5)
    w1 = mag1 * np.exp(1j * rng.uniform(0, 2 * np.pi))
    w2 = mag2 * np.exp(1j * rng.uniform(0, 2 * np.pi))
    return w1, z1, w2, z2


def make_dataset(n: int, seed: int = 0, N: int = N_RETAIN):
    """Synthesize n (feature, target) pairs.

    feature: U_0..U_N normalized by U_0 (canonical gauge -- removes both
    magnitude and phase redundancy from the target's true invariance),
    split into 2(N+1) reals.
    target: (c1, c0) monic quadratic coefficients, split into 4 reals.
    Also returns the raw (w1,zeta1,w2,zeta2) for diagnostics.
    """
    rng = np.random.default_rng(seed)
    feats = np.empty((n, 2 * (N + 1)), dtype=np.float32)
    targs = np.empty((n, 4), dtype=np.float32)
    meta = []
    i = 0
    while i < n:
        w1, z1, w2, z2 = sample_training_point(rng)
        U = w1 * itg_basis(z1, N) + w2 * itg_basis(z2, N)
        if abs(U[0]) < 1e-6:
            continue  # skip near-singular gauge (rare; resample)
        Un = U / U[0]
        c1, c0 = poly_coeffs_from_roots(np.array([z1, z2]))
        feats[i] = np.concatenate([Un.real, Un.imag]).astype(np.float32)
        targs[i] = np.array([c1.real, c1.imag, c0.real, c0.imag],
                            dtype=np.float32)
        meta.append((w1, z1, w2, z2))
        i += 1
    return feats, targs, meta


# ---------------------------------------------------------------------------
# M3'a: small MLP, coefficient regression (moment features -> monic coeffs)
# ---------------------------------------------------------------------------

import equinox as eqx
import jax
import jax.numpy as jnp
import optax


class MLP_r2(eqx.Module):
    """10 real inputs (normalized U_0..U_4, re/im) -> 4 real outputs (c1,c0).

    x_stats/y_stats (mean, std) hold the training-set feature/target
    normalization (z-score); baked into the module so predict_coeffs never
    has to be called with a mismatched normalization.
    """
    layers: list
    x_mean: jax.Array
    x_std: jax.Array
    y_mean: jax.Array
    y_std: jax.Array

    def __init__(self, hidden: tuple = (64, 64, 64), *, key,
                x_mean=None, x_std=None, y_mean=None, y_std=None):
        keys = jax.random.split(key, len(hidden) + 1)
        sizes = (2 * (N_RETAIN + 1),) + hidden + (4,)
        self.layers = [eqx.nn.Linear(i, o, key=k)
                      for i, o, k in zip(sizes[:-1], sizes[1:], keys)]
        nx, ny = sizes[0], sizes[-1]
        self.x_mean = jnp.zeros(nx) if x_mean is None else jnp.asarray(x_mean)
        self.x_std = jnp.ones(nx) if x_std is None else jnp.asarray(x_std)
        self.y_mean = jnp.zeros(ny) if y_mean is None else jnp.asarray(y_mean)
        self.y_std = jnp.ones(ny) if y_std is None else jnp.asarray(y_std)

    def __call__(self, x):
        # stop_gradient: these are fixed normalization statistics, not
        # learnable parameters -- without this, eqx.filter_value_and_grad
        # differentiates through them (they're plain jax.Array leaves) and
        # the optimizer "learns" to collapse the prediction toward y_mean
        # instead of training the actual layers (confirmed bug: predicted
        # coefficients were nearly constant across wildly different inputs
        # until this fix).
        x_mean = jax.lax.stop_gradient(self.x_mean)
        x_std = jax.lax.stop_gradient(self.x_std)
        y_mean = jax.lax.stop_gradient(self.y_mean)
        y_std = jax.lax.stop_gradient(self.y_std)
        x = (x - x_mean) / x_std
        for layer in self.layers[:-1]:
            x = jnp.tanh(layer(x))
        out = self.layers[-1](x)
        return out * y_std + y_mean


def predict_coeffs(model: MLP_r2, U: np.ndarray) -> tuple:
    """Run the NN: normalized moments -> (c1, c0) complex coefficients."""
    Un = np.asarray(U, dtype=complex) / U[0]
    x = jnp.array(np.concatenate([Un.real, Un.imag]).astype(np.float32))
    out = model(x)
    c1 = complex(out[0], out[1])
    c0 = complex(out[2], out[3])
    return c1, c0


def loss_fn(model, bx, by):
    pred = jax.vmap(model)(bx)
    # normalized-target loss: divide by y_std so all 4 outputs are weighted
    # comparably regardless of their raw scale
    return jnp.mean(((pred - by) / model.y_std) ** 2)


@eqx.filter_jit
def train_step(model, opt_state, bx, by, optim):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, bx, by)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def make_model_with_stats(feats: np.ndarray, targs: np.ndarray,
                          hidden: tuple = (64, 64, 64), *, key) -> MLP_r2:
    """Build an MLP_r2 with x/y normalization baked in from this dataset.

    Uses robust (median/MAD-based) scale rather than plain std -- the
    training distribution is heavy-tailed (near-degenerate pairs and
    large-|zeta| corners of the generic region produce moment/coefficient
    magnitudes orders of magnitude above the typical point), and a plain
    std is dominated by those rare outliers, under-normalizing the bulk of
    the (physically relevant) data.
    """
    def robust_scale(a):
        med = np.median(a, axis=0)
        mad = np.median(np.abs(a - med), axis=0) * 1.4826  # ~= std for Gaussian
        return med, np.maximum(mad, 1e-6)

    x_mean, x_std = robust_scale(feats)
    y_mean, y_std = robust_scale(targs)
    return MLP_r2(hidden, key=key, x_mean=x_mean, x_std=x_std,
                 y_mean=y_mean, y_std=y_std)


def train_r2(model: MLP_r2, feats: np.ndarray, targs: np.ndarray,
            n_steps: int = 8000, lr: float = 2e-3, batch_size: int = 256,
            seed: int = 0, log_every: int = 1000):
    X = jnp.array(feats)
    Y = jnp.array(targs)
    schedule = optax.cosine_decay_schedule(lr, n_steps, alpha=1e-5 / lr)
    optim = optax.adam(schedule)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    key = jax.random.PRNGKey(seed)
    n = X.shape[0]
    losses = []
    for step in range(n_steps):
        key, sub = jax.random.split(key)
        idx = jax.random.randint(sub, (batch_size,), 0, n)
        model, opt_state, loss = train_step(model, opt_state, X[idx], Y[idx],
                                            optim)
        losses.append(float(loss))
        if step % log_every == 0 or step == n_steps - 1:
            print(f"  step {step:5d}  loss = {float(loss):.4e}")
    return model, losses


# ---------------------------------------------------------------------------
# M3'c: runtime pipeline (NN propose -> fixed polish -> weights -> gate) and
# a genuine time-domain closure with a cascading r=2 -> r=1(gated) -> HP
# fallback hierarchy.
# ---------------------------------------------------------------------------

def dressed_pole_r2_next(U: np.ndarray, model: MLP_r2, n_polish: int = 2):
    """NN proposes poles -> fixed-count polish -> analytic weights.

    Returns (w, zetas, residual). No optimizer basin-hopping: the NN is a
    fixed continuous function, and n_polish is FIXED (not "until
    converged"), so this whole map is deterministic and (empirically)
    continuous in U -- see module docstring.
    """
    c1, c0 = predict_coeffs(model, U)
    zetas_nn = companion_roots_r2(c1, c0)
    if not np.all(np.isfinite(zetas_nn)):
        raise FloatingPointError("companion_roots_r2 produced non-finite "
                                 "poles (NN proposal out of range)")
    zetas_nn = np.array([_clip_zeta(z) for z in zetas_nn])
    w, zetas, residual = polish_r2(U, itg_basis, zetas_nn, n_iter=n_polish)
    zetas = np.array([_clip_zeta(z) for z in zetas])
    if not (np.all(np.isfinite(w)) and np.isfinite(residual)):
        raise FloatingPointError("polish_r2 produced non-finite weights/"
                                 "residual")
    return w, zetas, residual


def dressed_pole_r2_gated_next(U5: np.ndarray, model: MLP_r2,
                               zeta0_r1: complex | None = None,
                               n_polish: int = 2, rel_tol: float = 0.3):
    """Close U_5 given U_0..U_4, with a cascading fallback hierarchy:

        r=2 (this NN pipeline) --[residual gate]--> r=1 gated (dressed_pole.py)
                                                     --[residual gate]--> HP

    blend = rr_r2^2/(rr_r2^2+rel_tol^2) in [0,1): 0 = trust r=2 fully,
    ->1 = trust the r=1-gated fallback (itself already smoothly blended
    with HP).  Smooth in rr_r2, no hard switch -- same principle as
    dressed_pole.dressed_pole_r1_gated_next's HP blend.

    Returns (U5_next, zetas_r2, residual_r2, blend, zeta_r1) -- zeta_r1 is
    the r=1 fallback's own fitted frequency (for logging/continuation).
    """
    from bot.closures.dressed_pole import dressed_pole_r1_gated_next

    U5 = np.asarray(U5, dtype=complex)

    # r=1 fallback acts on the last 3 retained moments (U_2,U_3,U_4), same
    # "closure predicts the next moment" interface as dressed_pole.py.
    # Computed unconditionally (cheap, analytic) so it's always available
    # as the fallback -- including the hard-failure path below.
    U3 = U5[2:5]
    U5_r1, zeta_r1, resid_r1, blend_r1 = dressed_pole_r1_gated_next(
        U3, zeta0=zeta0_r1, basis_fn=itg_basis, rel_tol=rel_tol)

    try:
        w2, zetas2, resid2 = dressed_pole_r2_next(U5, model, n_polish=n_polish)
        m5_2 = np.array([itg_basis(z, 5)[-1] for z in zetas2])
        U5_r2 = w2 @ m5_2
        if not (np.isfinite(U5_r2.real) and np.isfinite(U5_r2.imag)):
            raise FloatingPointError("r=2 closure produced a non-finite U_5")
    except (FloatingPointError, np.linalg.LinAlgError, ValueError):
        # r=2 pipeline failed outright (should be rare post-clipping; this
        # is the last-resort safety net, not the normal gate path) -- fall
        # back to r=1 completely rather than propagate a crash.
        return U5_r1, np.array([zeta_r1, zeta_r1]), np.inf, 1.0, zeta_r1

    rr2 = resid2 / (np.linalg.norm(U5) + 1e-300)
    blend = rr2**2 / (rr2**2 + rel_tol**2)
    U5_next = (1.0 - blend) * U5_r2 + blend * U5_r1
    return U5_next, zetas2, resid2, blend, zeta_r1


def itg_evolve_r2(zeta_star: float, eta: float, tau: float,
                  t_grid: np.ndarray, y0: np.ndarray, model: MLP_r2,
                  n_polish: int = 2, rel_tol: float = 0.3,
                  rtol: float = 1e-9, atol: float = 1e-12) -> dict:
    """Slab-ITG N=5 system (U_0..U_4), U_5 closed by the gated r=2 pipeline.

    y0 = [U_0, U_1, U_2, U_3, U_4]. Only valid at the fixed benchmark point
    this module (and the trained model) are specialized to:
    zeta_star=1, tau=1, eta=10 (see ZETA_STAR/TAU/ETA module constants).
    """
    from scipy.integrate import solve_ivp

    assert abs(zeta_star - ZETA_STAR) < 1e-9 and abs(eta - ETA) < 1e-9, (
        "itg_evolve_r2 is specialized to the trained model's benchmark "
        "point; retrain for other (zeta_star, eta)")

    zeta_r1_holder = [1.0 + 0.1j]
    log = {"t": [], "zetas2": [], "residual2": [], "blend": [], "zeta1": []}

    def rhs(t, y_real):
        y = y_real[:5] + 1j * y_real[5:]
        U0, U1, U2, U3, U4 = y
        phi = U0 / tau
        if abs(U0) < 1e-13:
            U5 = 0.0 + 0.0j
        else:
            U5, zetas2, resid2, blend, zeta_r1 = dressed_pole_r2_gated_next(
                y, model, zeta0_r1=zeta_r1_holder[0], n_polish=n_polish,
                rel_tol=rel_tol)
            zeta_r1_holder[0] = zeta_r1
            log["t"].append(t); log["zetas2"].append(zetas2)
            log["residual2"].append(resid2); log["blend"].append(blend)
            log["zeta1"].append(zeta_r1)
        dU0 = -1j * U1 + 1j * zeta_star * phi
        dU1 = -1j * U2 - 0.5j * phi
        dU2 = -1j * U3 + 1j * zeta_star * (1.0 + eta) * 0.5 * phi
        dU3 = -1j * U4 - 0.75j * phi
        dU4 = -1j * U5 + 1j * zeta_star * (0.75 + 1.5 * eta) * phi
        dy = np.array([dU0, dU1, dU2, dU3, dU4])
        return np.concatenate([dy.real, dy.imag])

    y0_real = np.concatenate([y0.real, y0.imag])
    sol = solve_ivp(rhs, (t_grid[0], t_grid[-1]), y0_real, t_eval=t_grid,
                    method="RK45", rtol=rtol, atol=atol)
    if not sol.success:
        print(f"  itg_evolve_r2 WARNING: {sol.message}")
    n = len(sol.t)
    Y = sol.y[:5, :n] + 1j * sol.y[5:, :n]
    return {"t": sol.t, "U0": Y[0], "U1": Y[1], "U2": Y[2], "U3": Y[3],
            "U4": Y[4], "log_t": np.array(log["t"]),
            "log_zetas2": np.array(log["zetas2"]),
            "log_residual2": np.array(log["residual2"]),
            "log_blend": np.array(log["blend"]),
            "log_zeta1": np.array(log["zeta1"])}


def density_ic_r2(amp: float = 1e-3) -> np.ndarray:
    """Generic density-only IC for the N=5 (U_0..U_4) system."""
    y0 = np.zeros(5, dtype=complex)
    y0[0] = amp
    y0[2] = amp * 0.5  # M_2 of the equilibrium F0 (matches density_ic_fluid)
    return y0
