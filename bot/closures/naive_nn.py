"""Naive NN closure: learn U_3/U_0 = alpha directly from moment ratios.

Contrast with the inference closure (closures/inference.py): there the NN
learns only xi_hat, and alpha(xi_hat) is computed via exact Faddeeva. Here
the NN learns the *entire* closure response surface alpha as a function of
moment ratios, with no closed-form structure.

Both pass per-eigenmode tests because both fit the training manifold
(eigenmode-induced map ratios -> alpha) well. They differ off-eigenmode
during time evolution: the inference closure inherits alpha's known
structure (Faddeeva); the naive closure relies on the NN to learn that
structure implicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from bot.closures.inference import (MLP, alpha as exact_alpha,
                                     moment_ratios as _moment_ratios)


CKPT = Path(__file__).resolve().parent.parent / "runs" / "naive_mlp.eqx"


# Use the same MLP architecture; only the training target differs.

def naive_predict_alpha(model: MLP, r1: complex, r2: complex) -> complex:
    """Run naive MLP on moment ratios; output is alpha directly."""
    feats = jnp.stack([r1.real, r1.imag, r2.real, r2.imag])
    out = model(feats)
    return out[0] + 1j * out[1]


@dataclass
class NaiveDataset:
    features: jnp.ndarray   # (N, 4) -- re/im of moment ratios
    targets: jnp.ndarray    # (N, 2) -- re/im of alpha(xi_b)


def make_naive_dataset(N: int = 16384,
                       xi_re_range: tuple[float, float] = (-3.0, 3.0),
                       xi_im_range: tuple[float, float] = (-1.0, 1.5),
                       seed: int = 0) -> NaiveDataset:
    """Same xi_b sampling as inference closure; target is alpha(xi_b)."""
    rng = np.random.default_rng(seed)
    re = rng.uniform(*xi_re_range, size=N)
    im = rng.uniform(*xi_im_range, size=N)
    xi = re + 1j * im

    r1 = np.empty(N, dtype=complex)
    r2 = np.empty(N, dtype=complex)
    a = np.empty(N, dtype=complex)
    for i, x in enumerate(xi):
        r1[i], r2[i] = _moment_ratios(x)
        a[i] = exact_alpha(x)
    feats = np.stack([r1.real, r1.imag, r2.real, r2.imag], axis=1)
    targs = np.stack([a.real, a.imag], axis=1)
    return NaiveDataset(jnp.asarray(feats), jnp.asarray(targs))


def naive_loss(model: MLP, bx: jnp.ndarray, by: jnp.ndarray) -> jnp.ndarray:
    preds = jax.vmap(model)(bx)
    return jnp.mean((preds - by) ** 2)


@eqx.filter_jit
def _naive_step(model, opt_state, bx, by, optim):
    loss, grads = eqx.filter_value_and_grad(naive_loss)(model, bx, by)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def train_naive(model: MLP, ds: NaiveDataset,
                 n_steps: int = 8000, lr: float = 3e-3, lr_end: float = 1e-5,
                 batch_size: int = 512, seed: int = 0,
                 log_every: int = 1000):
    schedule = optax.cosine_decay_schedule(lr, n_steps, alpha=lr_end / lr)
    optim = optax.adam(schedule)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    key = jax.random.PRNGKey(seed)
    N = ds.features.shape[0]
    losses = []
    for step in range(n_steps):
        key, sub = jax.random.split(key)
        idx = jax.random.randint(sub, (batch_size,), 0, N)
        bx, by = ds.features[idx], ds.targets[idx]
        model, opt_state, loss = _naive_step(model, opt_state, bx, by, optim)
        losses.append(float(loss))
        if step % log_every == 0 or step == n_steps - 1:
            print(f"  step {step:5d}   loss = {float(loss):.4e}")
    return model, losses


def save(model: MLP, path: Path = CKPT) -> None:
    path.parent.mkdir(exist_ok=True)
    eqx.tree_serialise_leaves(path, model)


def load(key: jax.Array = jax.random.PRNGKey(0),
         hidden: tuple[int, ...] = (64, 64, 64, 64)) -> MLP:
    skeleton = MLP(hidden=hidden, key=key)
    return eqx.tree_deserialise_leaves(CKPT, skeleton)


# ----- closure interface (drop-in alternative to inference) ----------------

def naive_U3_over_U0(model: MLP, U0: complex, U1: complex, U2: complex
                      ) -> complex:
    """Naive closure: return alpha directly from NN output."""
    if abs(U0) < 1e-15:
        return 0.0 + 0.0j
    r1 = U1 / U0
    r2 = U2 / U0
    return complex(naive_predict_alpha(model, r1, r2))


# ----- training driver -----------------------------------------------------

def main(hidden: tuple[int, ...] = (64, 64, 64, 64),
          n_steps: int = 8000, lr: float = 3e-3,
          N_train: int = 16384, seed: int = 42) -> None:
    print("=== Training NAIVE closure (NN outputs alpha directly) ===")
    print(f"  hidden:        {hidden}")
    print(f"  steps:         {n_steps}")
    print(f"  train samples: {N_train}\n")

    ds_train = make_naive_dataset(N=N_train, seed=seed)
    ds_val = make_naive_dataset(N=2048, seed=seed + 1)

    key = jax.random.PRNGKey(seed)
    model = MLP(hidden=hidden, key=key)
    n_params = sum(int(p.size) for p in jax.tree.leaves(model) if hasattr(p, "size"))
    print(f"  params: {n_params}\n")

    model, losses = train_naive(model, ds_train, n_steps=n_steps, lr=lr,
                                  batch_size=512, seed=seed)

    val_loss = float(jnp.mean(
        (jax.vmap(model)(ds_val.features) - ds_val.targets) ** 2
    ))
    print(f"\n  final val MSE: {val_loss:.4e}")

    # spot checks
    print(f"\n=== alpha recovery on representative points ===")
    test_xis = [0.5 + 0.0j, -0.83 + 0.05j, -1.5 - 0.2j, 1.2 + 0.1j]
    print(f"  {'xi_true':>16}  {'alpha_naive':>32}  {'alpha_true':>32}  {'rel err':>10}")
    for xi in test_xis:
        r1, r2 = _moment_ratios(xi)
        a_naive = complex(naive_predict_alpha(model, r1, r2))
        a_true = exact_alpha(xi)
        err = abs(a_naive - a_true) / abs(a_true)
        print(f"  {str(xi):>16}  {str(a_naive):>32}  {str(a_true):>32}  {err:>10.2e}")

    save(model)
    print(f"\nsaved {CKPT}")


if __name__ == "__main__":
    main()
