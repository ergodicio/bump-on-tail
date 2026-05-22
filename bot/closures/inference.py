"""Inference-based closure: NN(moment ratios) -> xi_hat -> alpha(xi_hat) U_0.

Architecture
------------
The exact linear-physics closure for the 3-moment beam is
    U_3 = alpha(xi_b) U_0,      alpha(xi) = xi^3 + xi / (sqrt(pi) Z'(xi))
where xi_b = (omega - k u_b) / (k v_b). The function alpha(xi) is fully
known analytically; on a single linear eigenmode at xi_b, U_3 is exactly
alpha(xi_b) U_0 and any closure that delivers this is exact at the linear
level.

The challenge: in a time-domain fluid solver xi_b is not directly available;
the closure sees only the instantaneous moments (U_0, U_1, U_2). On a pure
eigenmode the moment ratios U_n/U_0 = g_n(xi_b)/g_0(xi_b) determine xi_b
uniquely. So a NN can learn the inverse map (moment ratios) -> xi_b and the
closure becomes
    closure(U_0, U_1, U_2) = alpha(NN(U_1/U_0, U_2/U_0)) * U_0
with alpha computed via exact Faddeeva.

This file:
  * `Z`, `Zprime`, `alpha`: numerical (numpy/scipy) evaluators
  * `moments_at_xi`: closed-form (U_0, U_1, U_2) at xi_b from the recurrence
  * `MLP`: equinox MLP with 4 real inputs (re/im of two complex ratios) and
    2 real outputs (re/im of xi_hat_b)
  * `make_dataset`, `train_step`, `train`: training utilities
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


# ----- numerical Z, Z', alpha (numpy) --------------------------------------

def Z(xi):
    """Plasma dispersion function via Faddeeva. numpy / complex input."""
    return 1j * SQRT_PI * _np_wofz(xi)


def Zprime(xi):
    return -2.0 * (1.0 + xi * Z(xi))


def alpha(xi):
    """Exact closure ratio U_3 / U_0 on a single linear eigenmode at xi_b.

    Derivation: from the recurrence U_{n+1} = xi U_n - n M_{n-1} Phi (M_0=1, M_1=0)
        U_1 = xi U_0
        U_2 = xi^2 U_0 - Phi
        U_3 = xi U_2 = xi^3 U_0 - xi Phi
    So U_3/U_0 = xi^3 - xi (Phi/U_0). With U_0/Phi = Z'(xi):
        alpha(xi) = xi^3 - xi / Z'(xi)
    """
    return xi**3 - xi / Zprime(xi)


# ----- closed-form moments at a single eigenmode ---------------------------

def moments_at_xi(xi):
    """Return (U_0, U_1, U_2) at xi_b with Phi = 1.

    Recurrence:  U_{n+1} = xi U_n - n M_{n-1} Phi   (M_0 = 1, M_1 = 0)
    With U_0/Phi = Z'(xi):
        U_0 = Z'(xi)
        U_1 = xi Z'(xi)
        U_2 = xi^2 Z'(xi) - 1
    """
    U0 = Zprime(xi)
    U1 = xi * U0
    U2 = xi * U1 - 1.0
    return U0, U1, U2


def moment_ratios(xi):
    """(U_1/U_0, U_2/U_0) at xi_b -- the NN input. Both complex."""
    U0, U1, U2 = moments_at_xi(xi)
    return U1 / U0, U2 / U0


# ----- equinox MLP ----------------------------------------------------------

class MLP(eqx.Module):
    """Small MLP: 4 real inputs -> 2 real outputs (re/im of xi_hat_b)."""
    layers: list

    def __init__(self, hidden: tuple[int, ...] = (32, 32, 32),
                 *, key: jax.Array):
        keys = jax.random.split(key, len(hidden) + 1)
        sizes = (4,) + hidden + (2,)
        self.layers = [
            eqx.nn.Linear(in_, out_, key=k)
            for in_, out_, k in zip(sizes[:-1], sizes[1:], keys)
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jnp.tanh(layer(x))
        return self.layers[-1](x)


def _ratios_to_features(r1: complex, r2: complex) -> jnp.ndarray:
    """Stack (re/im of U_1/U_0, re/im of U_2/U_0) into 4-real vector."""
    return jnp.stack([r1.real, r1.imag, r2.real, r2.imag])


def model_xi(model: MLP, r1, r2):
    """Run MLP and return xi_hat = re + 1j * im."""
    feats = _ratios_to_features(r1, r2)
    out = model(feats)
    return out[0] + 1j * out[1]


# ----- training -------------------------------------------------------------

@dataclass
class Dataset:
    features: jnp.ndarray   # (N, 4)
    targets: jnp.ndarray    # (N, 2)  -- (re xi_b, im xi_b)


def make_dataset(N: int = 8192, xi_re_range: tuple[float, float] = (-3.0, 3.0),
                 xi_im_range: tuple[float, float] = (-1.0, 1.5),
                 seed: int = 0) -> Dataset:
    """Sample xi_b uniformly in a complex rectangle, compute moment ratios.

    Default xi_im_range covers slightly-damped and slightly-growing modes
    which is the regime relevant for BoT and Landau damping.
    """
    rng = np.random.default_rng(seed)
    re = rng.uniform(*xi_re_range, size=N)
    im = rng.uniform(*xi_im_range, size=N)
    xi = re + 1j * im

    r1 = np.empty(N, dtype=complex)
    r2 = np.empty(N, dtype=complex)
    for i, x in enumerate(xi):
        r1[i], r2[i] = moment_ratios(x)
    feats = np.stack([r1.real, r1.imag, r2.real, r2.imag], axis=1)
    targs = np.stack([xi.real, xi.imag], axis=1)
    return Dataset(jnp.asarray(feats), jnp.asarray(targs))


def loss_fn(model: MLP, batch_x: jnp.ndarray, batch_y: jnp.ndarray) -> jnp.ndarray:
    """MSE on (re xi, im xi)."""
    preds = jax.vmap(model)(batch_x)
    return jnp.mean((preds - batch_y) ** 2)


@eqx.filter_jit
def train_step(model: MLP, opt_state, batch_x, batch_y, optim):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch_x, batch_y)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def train(model: MLP, ds: Dataset,
          n_steps: int = 5000, lr: float = 3e-3, lr_end: float = 1e-5,
          batch_size: int = 512, seed: int = 0,
          log_every: int = 500
          ) -> tuple[MLP, list[float]]:
    schedule = optax.cosine_decay_schedule(lr, n_steps, alpha=lr_end / lr)
    optim = optax.adam(schedule)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    key = jax.random.PRNGKey(seed)

    losses = []
    N = ds.features.shape[0]
    for step in range(n_steps):
        key, sub = jax.random.split(key)
        idx = jax.random.randint(sub, (batch_size,), 0, N)
        bx = ds.features[idx]
        by = ds.targets[idx]
        model, opt_state, loss = train_step(model, opt_state, bx, by, optim)
        losses.append(float(loss))
        if step % log_every == 0 or step == n_steps - 1:
            print(f"  step {step:5d}   loss = {float(loss):.4e}")
    return model, losses


# ----- closure interface ---------------------------------------------------

def inference_alpha(model: MLP, U0: complex, U1: complex, U2: complex) -> complex:
    """Closure ratio U_3 / U_0 from current moments via NN + exact Faddeeva.

    Used at inference / dispersion-test time. Faddeeva via scipy (numpy);
    NN forward via the trained equinox model.
    """
    r1 = U1 / U0
    r2 = U2 / U0
    xi_hat = complex(model_xi(model, r1, r2))
    return alpha(xi_hat)
