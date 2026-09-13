"""Homogeneous NN closure: state-dependent linear closure on invariant features.

    U_N = c(G) . U,      c(G) = NN(G),

    V = (U_0, ..., U_{N-1}, Phi),   G = V V^dagger / |V|^2.

* homogeneous of degree 1 and phase-equivariant in the joint state (U, Phi)
  by construction (c depends only on the invariants G, the moments carry the
  scale and phase);
* regular everywhere on the nontrivial state space (|V| = 0 only for the
  trivial solution U = Phi = 0), no ratios U_j/U_0 and no clipping;
* the potential enters as one more component of the state, so the direction
  of V carries N complex numbers and the closure can resolve M <= (N+1)/2
  superposed modes (two at N = 3);
* reduces to a learned-Pade closure when c is constant.

Options kept for the ablations (bot/figs/ablate_*.py): ext=False uses the
Gram matrix of U alone with S = |U|^2 + lam^2 |Phi|^2; pin='asym'/'opt'
pins c(0) to a Pade closure via c = c_pin + NN(G) - NN(0).  Neither is used
by default: the pin changes closed-loop results only marginally, and lam is
a pure reparametrization (lam = 1 by default).

Conventions (paper / repo, beam frame, v_t = 1): R(z) = 1 + z Z(z) = -U_0/Phi,
hierarchy  z U_n - U_{n+1} = (n/2) Phi M_{n-1},  M = (1, 0, 1/2, 0, 3/4, 0, 15/8).
In the repo's time-domain beam system  dU_1/dt = ... + E  so  Phi = 2 i E / k.

Training data (all analytic, no simulation): eigenmode superpositions
  U_j = sum_m g_m alpha_j(z_m),  Phi = sum_m g_m / (-R(z_m))  -- single modes,
generic pairs and mirror pairs z_2 = -z_1^*, with weights g_m on the unit
sphere so that nodes U_0 = 0 are sampled.  Exact impulse-response states
(frac_impulse > 0; kernels below) are available for ablations but off by
default: with the potential in the state and two-mode data they no longer
improve closed-loop results (bot/figs/figs_noimp_current.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.integrate import simpson

from bot.closures.inference import alpha as alpha3, alpha4
from bot.closures.pade import pade_coefficients, two_point_coefficients
from bot.closures.u2 import Z, beta as alpha2

RUN_DIR = Path(__file__).resolve().parent.parent / "runs"
LAM = 1.0
HIDDEN = (192, 192, 192)
XI_RE, XI_IM = (-3.0, 3.0), (-1.0, 1.5)
MAXW = np.array([1.0, 0.0, 0.5, 0.0, 0.75, 0.0, 1.875])   # M_0..M_6


def R(z):
    return 1.0 + z * Z(z)


def mode_ratios(z, N: int) -> np.ndarray:
    """alpha_0..alpha_N on a single eigenmode (alpha_j = U_j/U_0)."""
    a = [1.0, z, alpha2(z), alpha3(z), alpha4(z)]
    return np.array(a[:N + 1], dtype=complex)


def c_asym(N: int) -> np.ndarray:
    return pade_coefficients(3) if N == 3 else two_point_coefficients(4, 2)


# --------------------------------------------------------------------------
# exact impulse response
# --------------------------------------------------------------------------
def gauss_derivs(kap: np.ndarray, m_max: int) -> np.ndarray:
    """g_m(kap) = d^m/dkap^m exp(-kap^2/4), m = 0..m_max; shape (m_max+1, len(kap))."""
    kap = np.asarray(kap, dtype=float)
    g = np.empty((m_max + 1,) + kap.shape)
    g[0] = np.exp(-kap ** 2 / 4)
    if m_max >= 1:
        g[1] = -(kap / 2) * g[0]
    for m in range(1, m_max):
        g[m + 1] = -(kap / 2) * g[m] - (m / 2) * g[m - 1]
    return g


def impulse_state(zetas, amps, kap: float, N: int, n_grid: int = 401):
    """Moments U_0..U_N and Phi at kap for the drive Phi(k') = sum a_m e^{i z_m (kap-k')}.

    Returns (U[0..N], Phi(kap)).  Simpson quadrature on n_grid points.
    """
    kp = np.linspace(0.0, kap, n_grid)
    phi = np.zeros(n_grid, dtype=complex)
    for z, a in zip(zetas, amps):
        phi += a * np.exp(1j * z * (kap - kp))
    g = gauss_derivs(kap - kp, N + 1)                      # g_m(kap - k')
    U = np.empty(N + 1, dtype=complex)
    for n in range(N + 1):
        U[n] = (1j ** n) * simpson(g[n + 1] * phi, x=kp)
    return U, phi[-1]


def mode_state(zetas, weights, N: int):
    """Superposition of eigenmodes with moment weights g_m (U_0 = sum g_m)."""
    U = np.zeros(N + 1, dtype=complex)
    phi = 0j
    for z, g in zip(zetas, weights):
        U += g * mode_ratios(z, N)
        phi += g / (-R(z))
    return U, phi


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------
def gram_features(U: np.ndarray, phi: complex, lam: float = LAM, ext: bool = True) -> np.ndarray:
    """Real feature vector of G = V V^dagger / S (diag, then re/im of upper triangle),
    V = U (ext=False) or V = (U, lam*Phi) (ext=True, the joint-state Gram matrix)."""
    U = np.asarray(U, dtype=complex)
    S = float(np.vdot(U, U).real) + lam ** 2 * abs(phi) ** 2
    V = np.append(U, lam * phi) if ext else U
    if S <= 0.0:
        return np.zeros(len(V) ** 2)
    G = np.outer(V, V.conj()) / S
    M = len(V)
    iu = np.triu_indices(M, k=1)
    return np.concatenate([G.diagonal().real, G[iu].real, G[iu].imag])


def n_features(N: int, ext: bool = True) -> int:
    return (N + 1) ** 2 if ext else N * N


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------
def _draw_zeta(rng):
    return complex(rng.uniform(*XI_RE), rng.uniform(*XI_IM))


def _sphere_weights(rng, m):
    g = rng.normal(size=m) + 1j * rng.normal(size=m)
    return g / np.linalg.norm(g)


def make_dataset(N: int, n: int, seed: int = 0, frac_impulse: float = 0.0,
                 kap_range=(0.02, 40.0), lam: float = LAM, frac_single: float = 0.5,
                 ext: bool = True):
    """(features, U_state, Phi, U_N target, is_single) for n samples."""
    rng = np.random.default_rng(seed)
    # with the potential in the state the direction carries N complex numbers,
    # so two modes are resolvable for N >= 3 (N >= 2M - 1)
    n_modes_max = 2 if (ext or N == 4) else 1
    feats = np.empty((n, n_features(N, ext)))
    Ustate = np.empty((n, N), dtype=complex)
    Phi = np.empty(n, dtype=complex)
    UN = np.empty(n, dtype=complex)
    single = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if rng.random() < frac_impulse:
            m = 1 if (n_modes_max == 1 or rng.random() < 0.7) else 2
            zs = [_draw_zeta(rng) for _ in range(m)]
            if m == 2 and rng.random() < 0.3:
                zs[1] = -np.conj(zs[0])
            amps = _sphere_weights(rng, m)
            kap = float(np.exp(rng.uniform(np.log(kap_range[0]), np.log(kap_range[1]))))
            U, phi = impulse_state(zs, amps, kap, N)
            is_single = False
        else:
            r = rng.random()
            if n_modes_max == 1 or r < frac_single:
                zs = [_draw_zeta(rng)]
            else:
                zs = [_draw_zeta(rng), _draw_zeta(rng)]
                if r < frac_single + 0.5 * (1 - frac_single):
                    zs[1] = -np.conj(zs[0])
                if abs(zs[0] - zs[1]) < 0.1:
                    continue
            U, phi = mode_state(zs, _sphere_weights(rng, len(zs)), N)
            is_single = len(zs) == 1
        if not np.all(np.isfinite(U)) or not np.isfinite(phi):
            continue
        S = np.vdot(U[:N], U[:N]).real + lam ** 2 * abs(phi) ** 2
        if S <= 0 or abs(U[N]) ** 2 / S > 1e4:
            continue
        feats[i] = gram_features(U[:N], phi, lam, ext)
        Ustate[i] = U[:N]
        Phi[i] = phi
        UN[i] = U[N]
        single[i] = is_single
        i += 1
    return feats, Ustate, Phi, UN, single


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
import equinox as eqx
import jax
import jax.numpy as jnp
import optax


class HomogMLP(eqx.Module):
    """features (N^2) -> coefficient correction (2N reals), zero at zero features."""
    layers: list
    c_asym_re: jax.Array
    c_asym_im: jax.Array

    subtract0: bool = eqx.field(static=True)
    ext: bool = eqx.field(static=True)

    def __init__(self, N: int, hidden=HIDDEN, *, key, pin: str = "none", ext: bool = True):
        """pin: 'none' (free bias, default), 'asym' (HP / Hunana), 'opt' (learned Pade).
        ext: use the Gram matrix of the joint state (U, lam*Phi) as input."""
        keys = jax.random.split(key, len(hidden) + 1)
        self.ext = ext
        sizes = (n_features(N, ext),) + tuple(hidden) + (2 * N,)
        self.layers = [eqx.nn.Linear(a, b, key=k) for a, b, k in zip(sizes[:-1], sizes[1:], keys)]
        if pin == "asym":
            ca = c_asym(N)
        elif pin == "opt":
            from bot.closures.opt_hunana import opt_coefficients
            ca = opt_coefficients(N)
        elif pin == "none":
            ca = np.zeros(N, dtype=complex)
        else:
            raise ValueError(pin)
        self.subtract0 = pin != "none"
        self.c_asym_re = jnp.asarray(ca.real)
        self.c_asym_im = jnp.asarray(ca.imag)

    def _raw(self, x):
        for layer in self.layers[:-1]:
            x = jnp.tanh(layer(x))
        return self.layers[-1](x)

    def coeffs(self, x):
        """Complex coefficient vector c(G) = c_asym + raw(x) - raw(0)."""
        d = self._raw(x) - self._raw(jnp.zeros_like(x)) if self.subtract0 else self._raw(x)
        N = self.c_asym_re.shape[0]
        ca_re = jax.lax.stop_gradient(self.c_asym_re)
        ca_im = jax.lax.stop_gradient(self.c_asym_im)
        return (ca_re + d[:N]) + 1j * (ca_im + d[N:])

    def __call__(self, x):
        return self.coeffs(x)


def _loss(model, fx, U, S, UN, w):
    c = jax.vmap(model)(fx)                               # (B, N) complex
    pred = jnp.sum(c * U, axis=1)
    return jnp.mean(w * jnp.abs(pred - UN) ** 2 / S)


@eqx.filter_jit
def _step(model, opt_state, fx, U, S, UN, w, optim):
    loss, grads = eqx.filter_value_and_grad(_loss)(model, fx, U, S, UN, w)
    updates, opt_state = optim.update(grads, opt_state, model)
    return eqx.apply_updates(model, updates), opt_state, loss


def train(N: int, n_samples: int = 400_000, n_steps: int = 150_000, lr: float = 2e-3,
          hidden=None,
          batch: int = 1024, seed: int = 0, lam: float = LAM, log_every: int = 5000,
          w_single: float = 3.0, frac_impulse: float = 0.0, pin: str = "none",
          ext: bool = True):
    feats, U, Phi, UN, single = make_dataset(N, n_samples, seed=seed, lam=lam,
                                             frac_impulse=frac_impulse, ext=ext)
    S = (np.abs(U) ** 2).sum(1) + lam ** 2 * np.abs(Phi) ** 2
    w = np.where(single, w_single, 1.0)
    fx, Uj, Sj, UNj, wj = (jnp.asarray(feats), jnp.asarray(U), jnp.asarray(S), jnp.asarray(UN), jnp.asarray(w))
    model = HomogMLP(N, hidden=hidden or HIDDEN, key=jax.random.PRNGKey(seed), pin=pin, ext=ext)
    sched = optax.cosine_decay_schedule(lr, n_steps, alpha=1e-3)
    optim = optax.adam(sched)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    key = jax.random.PRNGKey(seed + 1)
    n = fx.shape[0]
    for step in range(n_steps):
        key, sub = jax.random.split(key)
        idx = jax.random.randint(sub, (batch,), 0, n)
        model, opt_state, loss = _step(model, opt_state, fx[idx], Uj[idx], Sj[idx], UNj[idx], wj[idx], optim)
        if step % log_every == 0 or step == n_steps - 1:
            print(f"  N={N} step {step:6d}  loss {float(loss):.3e}", flush=True)
    full = float(_loss(model, fx, Uj, Sj, UNj, jnp.ones_like(wj)))
    print(f"  N={N} final training loss {full:.3e}")
    return model


def save(model: HomogMLP, N: int, path: Path | None = None):
    path = path or RUN_DIR / f"homog_nn_n{N}.eqx"
    eqx.tree_serialise_leaves(path, model)
    return path


def load(N: int, path: Path | None = None, pin: str = "none", ext: bool = True,
         hidden=None) -> HomogMLP:
    path = path or RUN_DIR / f"homog_nn_n{N}.eqx"
    skel = HomogMLP(N, hidden=hidden or HIDDEN, key=jax.random.PRNGKey(0), pin=pin, ext=ext)
    return eqx.tree_deserialise_leaves(path, skel)


# --------------------------------------------------------------------------
# runtime closure
# --------------------------------------------------------------------------
class HomogClosure:
    """closure(U, Phi) -> U_N, jitted forward pass; N inferred from the model."""

    def __init__(self, model: HomogMLP, lam: float = LAM):
        self.model = model
        self.lam = lam
        self.N = int(model.c_asym_re.shape[0])
        self._f = eqx.filter_jit(lambda m, x: m.coeffs(x))

    def coeffs(self, U, phi) -> np.ndarray:
        x = gram_features(U, phi, self.lam, self.model.ext)
        return np.asarray(self._f(self.model, jnp.asarray(x)))

    def __call__(self, U, phi) -> complex:
        U = np.asarray(U, dtype=complex)
        if np.vdot(U, U).real + self.lam ** 2 * abs(phi) ** 2 <= 0.0:
            return 0j
        return complex(self.coeffs(U, phi) @ U)


if __name__ == "__main__":
    import sys
    Ns = [int(a) for a in sys.argv[1:]] or [3, 4]
    for N in Ns:
        print(f"=== training homogeneous closure, N={N}")
        m = train(N)
        p = save(m, N)
        print(f"  saved {p}")


# --------------------------------------------------------------------------
# closed-loop beam system (repo conventions), for figure drivers
# --------------------------------------------------------------------------
def bot_evolve(k: float, u_b: float, eps: float, cl: HomogClosure,
               t_grid: np.ndarray, y0: np.ndarray, rtol: float = 1e-8,
               atol: float = 1e-10) -> np.ndarray:
    """E(t) for the beam fluid system (u, E, U_0..U_{N-1}) closed by cl(U, Phi).

    Same state layout, sources and tolerances as bot.closed_loop.nn_n4_evolve /
    naive_evolve; Phi = 2 i E / k.
    """
    from scipy.integrate import solve_ivp
    N = cl.N

    def rhs(t, y):
        z = y[:N + 2] + 1j * y[N + 2:]
        u, E, U = z[0], z[1], z[2:]
        UN = cl(U, 2j * E / k)
        dU = np.empty(N, dtype=complex)
        for n in range(N):
            nxt = U[n + 1] if n + 1 < N else UN
            dU[n] = -1j * k * u_b * U[n] - 1j * k * nxt + (n * MAXW[n - 1] * E if n > 0 else 0.0)
        dz = np.concatenate([[-E, u - eps * (u_b * U[0] + U[1])], dU])
        return np.concatenate([dz.real, dz.imag])

    sol = solve_ivp(rhs, (t_grid[0], t_grid[-1]), np.concatenate([y0.real, y0.imag]),
                    t_eval=t_grid, method="RK45", rtol=rtol, atol=atol)
    if not sol.success:
        print(f"  bot_evolve WARNING: {sol.message}")
    return sol.y[1] + 1j * sol.y[N + 2 + 1]
