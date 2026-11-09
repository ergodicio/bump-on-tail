"""Equivariant neural closure and its analytic training data.

U_N=c(G) dot U, G=VV^dagger/(V^dagger V), V=(U_0,...,U_{N-1},Phi).
The normalized Gram features make the closure equivariant under common
complex scaling. The paper uses three 192-unit tanh layers, free bias,
unit potential weight, and single/generic-pair data without mirror pairs.
JAX/Equinox/Optax are optional: only regeneration and training import this module.
"""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from bot.closures.response import R, alpha4
from bot.closures.response import alpha as alpha3
from bot.closures.response import beta as alpha2
from bot.paths import MODELS

RUN_DIR = MODELS
LAM = 1.0
HIDDEN = (192, 192, 192)
XI_RE, XI_IM = (-3.0, 3.0), (-1.0, 1.5)
MAXW = np.array([1.0, 0.0, 0.5, 0.0, 0.75, 0.0, 1.875])


def mode_ratios(z, N: int) -> np.ndarray:
    """alpha_0..alpha_N on a single eigenmode (alpha_j = U_j/U_0)."""
    a = [1.0, z, alpha2(z), alpha3(z), alpha4(z)]
    return np.array(a[: N + 1], dtype=complex)


def mode_state(zetas, weights, N: int):
    """Superposition of eigenmodes with moment weights g_m (U_0 = sum g_m)."""
    U = np.zeros(N + 1, dtype=complex)
    phi = 0j
    for z, g in zip(zetas, weights):
        U += g * mode_ratios(z, N)
        phi += g / (-R(z))
    return U, phi


def gram_features(U: np.ndarray, phi: complex) -> np.ndarray:
    """Pack normalized joint-state Gram matrix: diagonal, Re upper, Im upper."""
    U = np.asarray(U, dtype=complex)
    S = float(np.vdot(U, U).real) + abs(phi) ** 2
    V = np.append(U, phi)
    if S <= 0.0:
        return np.zeros(len(V) ** 2)
    G = np.outer(V, V.conj()) / S
    M = len(V)
    iu = np.triu_indices(M, k=1)
    return np.concatenate([G.diagonal().real, G[iu].real, G[iu].imag])


def n_features(N: int) -> int:
    """Number of real entries in an (N+1)-dimensional Hermitian Gram matrix."""
    return (N + 1) ** 2


def _draw_zeta(rng):
    return complex(rng.uniform(*XI_RE), rng.uniform(*XI_IM))


def _sphere_weights(rng, m):
    g = rng.normal(size=m) + 1j * rng.normal(size=m)
    return g / np.linalg.norm(g)


def make_dataset(N: int, n: int, seed: int = 0, return_zetas: bool = False):
    """Draw the paper's 50/50 single-mode and generic two-mode training states.

    Return (Gram features, input moments, potential, target moment, is_single).
    If requested, append frequencies in accepted-sample order. Pair separation
    must exceed 0.1; reject target energy / joint-state energy > 1e4.
    The extra RNG draw preserves the published sampler's exact seed sequence.
    """
    if N not in (3, 4) or n <= 0:
        raise ValueError("Use order 3 or 4 and a positive sample count")
    rng = np.random.default_rng(seed)
    # with the potential in the state the direction carries N complex numbers,
    # so two modes are resolvable for N >= 3 (N >= 2M - 1)
    feats = np.empty((n, n_features(N)))
    Ustate = np.empty((n, N), dtype=complex)
    Phi = np.empty(n, dtype=complex)
    UN = np.empty(n, dtype=complex)
    single = np.zeros(n, dtype=bool)
    zetas = []
    i = 0
    while i < n:
        rng.random()  # Preserve the source sampler's (disabled) impulse draw.
        r = rng.random()
        if r < 0.5:
            zs = [_draw_zeta(rng)]
        else:
            zs = [_draw_zeta(rng), _draw_zeta(rng)]
            if abs(zs[0] - zs[1]) < 0.1:
                continue
        U, phi = mode_state(zs, _sphere_weights(rng, len(zs)), N)
        is_single = len(zs) == 1
        if not np.all(np.isfinite(U)) or not np.isfinite(phi):
            continue
        S = np.vdot(U[:N], U[:N]).real + abs(phi) ** 2
        if S <= 0 or abs(U[N]) ** 2 / S > 1e4:
            continue
        feats[i] = gram_features(U[:N], phi)
        Ustate[i] = U[:N]
        Phi[i] = phi
        UN[i] = U[N]
        single[i] = is_single
        zetas += list(zs)
        i += 1
    if return_zetas:
        return feats, Ustate, Phi, UN, single, np.array(zetas)
    return feats, Ustate, Phi, UN, single


class HomogMLP(eqx.Module):
    """(N+1)^2 Gram features -> N complex coefficients, with free output bias.

    Zero reference arrays and static flags retain the original Equinox
    checkpoint structure. They do not pin the paper model at zero features.
    """

    layers: list
    c_asym_re: jax.Array
    c_asym_im: jax.Array

    subtract0: bool = eqx.field(static=True)
    ext: bool = eqx.field(static=True)

    def __init__(self, N: int, hidden=HIDDEN, *, key):
        keys = jax.random.split(key, len(hidden) + 1)
        self.ext = True
        sizes = (n_features(N),) + tuple(hidden) + (2 * N,)
        self.layers = [eqx.nn.Linear(a, b, key=k) for a, b, k in zip(sizes[:-1], sizes[1:], keys)]
        ca = np.zeros(N, dtype=complex)
        self.subtract0 = False
        self.c_asym_re = jnp.asarray(ca.real)
        self.c_asym_im = jnp.asarray(ca.imag)

    def _raw(self, x):
        for layer in self.layers[:-1]:
            x = jnp.tanh(layer(x))
        return self.layers[-1](x)

    def coeffs(self, x):
        """Complex coefficient vector c(G), preserving checkpoint arithmetic."""
        d = self._raw(x)
        N = self.c_asym_re.shape[0]
        ca_re = jax.lax.stop_gradient(self.c_asym_re)
        ca_im = jax.lax.stop_gradient(self.c_asym_im)
        return (ca_re + d[:N]) + 1j * (ca_im + d[N:])

    def __call__(self, x):
        return self.coeffs(x)


def _loss(model, fx, U, S, UN, w):
    c = jax.vmap(model)(fx)  # (B, N) complex
    pred = jnp.sum(c * U, axis=1)
    return jnp.mean(w * jnp.abs(pred - UN) ** 2 / S)


@eqx.filter_jit
def _step(model, opt_state, fx, U, S, UN, w, optim):
    loss, grads = eqx.filter_value_and_grad(_loss)(model, fx, U, S, UN, w)
    updates, opt_state = optim.update(grads, opt_state, model)
    return eqx.apply_updates(model, updates), opt_state, loss


def train(
    N: int,
    n_samples: int = 400_000,
    n_steps: int = 150_000,
    lr: float = 2e-3,
    hidden=None,
    batch: int = 1024,
    seed: int = 0,
    log_every: int = 5000,
    w_single: float = 3.0,
):
    """Adam with cosine LR decay; relative joint-state MSE, singles weighted 3x."""
    feats, U, Phi, UN, single = make_dataset(N, n_samples, seed=seed)
    S = (np.abs(U) ** 2).sum(1) + np.abs(Phi) ** 2
    w = np.where(single, w_single, 1.0)
    fx, Uj, Sj, UNj, wj = (
        jnp.asarray(feats),
        jnp.asarray(U),
        jnp.asarray(S),
        jnp.asarray(UN),
        jnp.asarray(w),
    )
    model = HomogMLP(N, hidden=hidden or HIDDEN, key=jax.random.PRNGKey(seed))
    sched = optax.cosine_decay_schedule(lr, n_steps, alpha=1e-3)
    optim = optax.adam(sched)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    key = jax.random.PRNGKey(seed + 1)
    n = fx.shape[0]
    for step in range(n_steps):
        key, sub = jax.random.split(key)
        idx = jax.random.randint(sub, (batch,), 0, n)
        model, opt_state, loss = _step(
            model, opt_state, fx[idx], Uj[idx], Sj[idx], UNj[idx], wj[idx], optim
        )
        if step % log_every == 0 or step == n_steps - 1:
            print(f"  N={N} step {step:6d}  loss {float(loss):.3e}", flush=True)
    full = float(_loss(model, fx, Uj, Sj, UNj, jnp.ones_like(wj)))
    print(f"  N={N} final training loss {full:.3e}")
    return model


def save(model: HomogMLP, N: int, path: Path | None = None):
    path = path or RUN_DIR / f"homog_nn_n{N}_nomirror.eqx"
    eqx.tree_serialise_leaves(path, model)
    return path


def load(N: int, path: Path | None = None, hidden=None) -> HomogMLP:
    path = path or RUN_DIR / f"homog_nn_n{N}_nomirror.eqx"
    skel = HomogMLP(N, hidden=hidden or HIDDEN, key=jax.random.PRNGKey(0))
    return eqx.tree_deserialise_leaves(path, skel)


class HomogClosure:
    """closure(U, Phi) -> U_N, jitted forward pass; N inferred from the model."""

    def __init__(self, model: HomogMLP):
        self.model = model
        self.N = int(model.c_asym_re.shape[0])
        self._f = eqx.filter_jit(lambda m, x: m.coeffs(x))

    def coeffs(self, U, phi) -> np.ndarray:
        x = gram_features(U, phi)
        return np.asarray(self._f(self.model, jnp.asarray(x)))

    def __call__(self, U, phi) -> complex:
        U = np.asarray(U, dtype=complex)
        if np.vdot(U, U).real + abs(phi) ** 2 <= 0.0:
            return 0j
        return complex(self.coeffs(U, phi) @ U)


def bot_evolve(
    k: float,
    u_b: float,
    eps: float,
    cl: HomogClosure,
    t_grid: np.ndarray,
    y0: np.ndarray,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> np.ndarray:
    """E(t) for the beam fluid system (u, E, U_0..U_{N-1}) closed by cl(U, Phi).

    Phi = 2 i E / k. RK45 uses rtol=1e-8 and atol=1e-10.
    """
    from scipy.integrate import solve_ivp

    N = cl.N

    def rhs(t, y):
        z = y[: N + 2] + 1j * y[N + 2 :]
        u, E, U = z[0], z[1], z[2:]
        UN = cl(U, 2j * E / k)
        dU = np.empty(N, dtype=complex)
        for n in range(N):
            nxt = U[n + 1] if n + 1 < N else UN
            dU[n] = -1j * k * u_b * U[n] - 1j * k * nxt + (n * MAXW[n - 1] * E if n > 0 else 0.0)
        dz = np.concatenate([[-E, u - eps * (u_b * U[0] + U[1])], dU])
        return np.concatenate([dz.real, dz.imag])

    sol = solve_ivp(
        rhs,
        (t_grid[0], t_grid[-1]),
        np.concatenate([y0.real, y0.imag]),
        t_eval=t_grid,
        method="RK45",
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[1] + 1j * sol.y[N + 2 + 1]
