"""Exact per-mode U_2 closure for the N=2 (U_0, U_1) beam moment system.

For a Maxwellian beam the moment recurrence with Phi=1 reads:
    xi_b U_n - U_{n+1} = n M_{n-1}    (M_0=1, M_1=0, M_2=1/2, ...)

This gives:
    U_1 / U_0 = xi_b                  (n=0 row, M_{-1}=0 => RHS=0)
    U_2 / U_0 = beta(xi_b)            where beta(xi) = xi^2 - 1/Z'(xi)

Unlike the N=3 case, beta has no bilinear-product structure because the n=1
row has nonzero RHS (M_0=1), so U_2 cannot be written as a product of moment
ratios without invoking Z'.

Since r_1 = U_1/U_0 = xi_b exactly on a single eigenmode, the "inference"
map is trivial at this level.  The exact on-manifold closure is:
    U_2 = U_0 * beta(U_1/U_0)
applied directly, with no NN inversion step needed.

This file provides:
  * beta(xi)             -- exact closure ratio U_2/U_0
  * moments_at_xi_u2(xi) -- (U_0, U_1) at xi with Phi=1
  * make_dataset_u2      -- training data for a naive NN at this level
  * MLP_u2               -- small MLP: 2 real inputs -> 2 real outputs
  * Training utilities
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.special import wofz as _np_wofz


SQRT_PI = float(np.sqrt(np.pi))


# ---------------------------------------------------------------------------
# Z, Z', beta  (numpy / scalar)
# ---------------------------------------------------------------------------

def Z(xi):
    """Plasma dispersion function via Faddeeva."""
    return 1j * SQRT_PI * _np_wofz(xi)


def Zprime(xi):
    return -2.0 * (1.0 + xi * Z(xi))


def beta(xi):
    """Exact U_2/U_0 on a single eigenmode at xi_b = xi.

    Derivation (Phi=1):
        U_0 = Z'(xi)
        U_1 = xi * U_0           (from n=0 recurrence, M_{-1}=0)
        U_2 = xi*U_1 - M_0       (from n=1 recurrence, M_0=1)
           = xi^2 * U_0 - 1
    So  U_2/U_0 = xi^2 - 1/Z'(xi).
    """
    return xi**2 - 1.0 / Zprime(xi)


# ---------------------------------------------------------------------------
# Closed-form moments at a single eigenmode
# ---------------------------------------------------------------------------

def moments_at_xi_u2(xi):
    """Return (U_0, U_1) at xi_b with Phi=1.

    U_0 = Z'(xi),   U_1 = xi * Z'(xi)
    """
    U0 = Zprime(xi)
    U1 = xi * U0
    return U0, U1


def moment_ratio_u2(xi):
    """r_1 = U_1/U_0 at xi_b -- the closure input at N=2.

    By construction r_1 = xi_b, so this is the identity map.
    Provided for symmetry with the N=3 inference interface.
    """
    return xi  # trivially


# ---------------------------------------------------------------------------
# Equinox MLP: 2 real inputs -> 2 real outputs
# ---------------------------------------------------------------------------

class MLP_u2(eqx.Module):
    """Small MLP for naive beta learning: 2 real inputs -> 2 real outputs."""
    layers: list

    def __init__(self, hidden: tuple[int, ...] = (32, 32, 32),
                 *, key: jax.Array):
        keys = jax.random.split(key, len(hidden) + 1)
        sizes = (2,) + hidden + (2,)
        self.layers = [
            eqx.nn.Linear(in_, out_, key=k)
            for in_, out_, k in zip(sizes[:-1], sizes[1:], keys)
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jnp.tanh(layer(x))
        return self.layers[-1](x)


def _ratio_to_features_u2(r1: complex) -> jnp.ndarray:
    """Stack re/im of U_1/U_0 into a 2-real vector."""
    return jnp.stack([jnp.array(r1.real), jnp.array(r1.imag)])


def model_beta(model: MLP_u2, r1) -> complex:
    """Run MLP and return beta_hat = re + 1j*im."""
    feats = jnp.stack([jnp.array(float(r1.real)), jnp.array(float(r1.imag))])
    out = model(feats)
    return out[0] + 1j * out[1]


# ---------------------------------------------------------------------------
# Dataset and training
# ---------------------------------------------------------------------------

@dataclass
class Dataset_u2:
    features: jnp.ndarray   # (N, 2) -- re/im of r_1 = xi_b
    targets: jnp.ndarray    # (N, 2) -- re/im of beta(xi_b)


def make_dataset_u2(N: int = 8192,
                    xi_re_range: tuple[float, float] = (-3.0, 3.0),
                    xi_im_range: tuple[float, float] = (-1.0, 1.5),
                    seed: int = 0) -> Dataset_u2:
    """Sample xi_b uniformly in a complex rectangle, compute r_1 and beta.

    Since r_1 = xi_b exactly on eigenmode, the feature vector is simply
    (re xi_b, im xi_b).  Training this NN teaches it to approximate beta
    over the full xi_b rectangle.
    """
    rng = np.random.default_rng(seed)
    re = rng.uniform(*xi_re_range, size=N)
    im = rng.uniform(*xi_im_range, size=N)
    xi = re + 1j * im

    beta_vals = np.array([beta(x) for x in xi], dtype=complex)
    feats = np.stack([xi.real, xi.imag], axis=1).astype(np.float32)
    targs = np.stack([beta_vals.real, beta_vals.imag], axis=1).astype(np.float32)
    return Dataset_u2(jnp.asarray(feats), jnp.asarray(targs))


def loss_fn_u2(model: MLP_u2, bx: jnp.ndarray, by: jnp.ndarray) -> jnp.ndarray:
    preds = jax.vmap(model)(bx)
    return jnp.mean((preds - by) ** 2)


@eqx.filter_jit
def train_step_u2(model: MLP_u2, opt_state, bx, by, optim):
    loss, grads = eqx.filter_value_and_grad(loss_fn_u2)(model, bx, by)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def train_u2(model: MLP_u2, ds: Dataset_u2,
             n_steps: int = 5000, lr: float = 3e-3, lr_end: float = 1e-5,
             batch_size: int = 512, seed: int = 0,
             log_every: int = 500) -> tuple[MLP_u2, list[float]]:
    schedule = optax.cosine_decay_schedule(lr, n_steps, alpha=lr_end / lr)
    optim = optax.adam(schedule)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    key = jax.random.PRNGKey(seed)

    losses = []
    N = ds.features.shape[0]
    for step in range(n_steps):
        key, sub = jax.random.split(key)
        idx = jax.random.randint(sub, (batch_size,), 0, N)
        bx, by = ds.features[idx], ds.targets[idx]
        model, opt_state, loss = train_step_u2(model, opt_state, bx, by, optim)
        losses.append(float(loss))
        if step % log_every == 0 or step == n_steps - 1:
            print(f"  step {step:5d}   loss = {float(loss):.4e}")
    return model, losses


# ---------------------------------------------------------------------------
# Closure interface
# ---------------------------------------------------------------------------

def naive_u2_beta(model: MLP_u2, U0: complex, U1: complex) -> complex:
    """Naive NN U_2 closure: return beta_hat * U_0."""
    if abs(U0) < 1e-15:
        return 0.0 + 0.0j
    r1 = U1 / U0
    return complex(model_beta(model, r1))


def exact_u2_beta(U0: complex, U1: complex) -> complex:
    """Exact per-mode U_2 closure: beta(U_1/U_0).

    Exact on a single eigenmode; off the eigenmode manifold this applies
    beta at the effective ratio r_1 = U_1/U_0.
    """
    if abs(U0) < 1e-15:
        return 0.0 + 0.0j
    return complex(beta(U1 / U0))
