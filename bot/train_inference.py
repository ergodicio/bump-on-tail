"""Train the inference closure on closed-form linear eigenmode data."""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from bot.closures.inference import (
    MLP, alpha, make_dataset, model_xi, moment_ratios, train,
)


CKPT = Path(__file__).resolve().parent / "runs" / "inference_mlp.eqx"


def main(n_steps: int = 8000, hidden: tuple[int, ...] = (64, 64, 64, 64),
          lr: float = 3e-3, N_train: int = 16384, seed: int = 42) -> None:
    print(f"=== Training inference closure ===")
    print(f"  hidden:        {hidden}")
    print(f"  steps:         {n_steps}")
    print(f"  lr:            {lr} → cosine decay")
    print(f"  train samples: {N_train}")

    ds_train = make_dataset(N=N_train, seed=seed)
    ds_val = make_dataset(N=2048, seed=seed + 1)

    key = jax.random.PRNGKey(seed)
    model = MLP(hidden=hidden, key=key)
    n_params = sum(int(p.size) for p in jax.tree.leaves(model) if hasattr(p, "size"))
    print(f"  params:        {n_params}")

    model, losses = train(model, ds_train, n_steps=n_steps, lr=lr,
                          batch_size=512, seed=seed)

    val_loss = float(jnp.mean(
        (jax.vmap(model)(ds_val.features) - ds_val.targets) ** 2
    ))
    print(f"\n  final val MSE: {val_loss:.4e}")

    # specific point tests
    print(f"\n=== ξ_b recovery on representative points ===")
    test_xis = [
        0.5 + 0.0j,
        -0.83 + 0.05j,    # canonical BoT (u_b=5, eps=0.05)
        -1.5 - 0.2j,
        1.2 + 0.1j,
        -2.0 + 0.0j,
        0.0 + 0.0j,
    ]
    header = f"  {'ξ_true':>22}  {'ξ_hat':>32}  {'|err|':>10}  {'α err':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for xi in test_xis:
        r1, r2 = moment_ratios(xi)
        xi_hat = complex(model_xi(model, r1, r2))
        err = abs(xi_hat - xi)
        a_err = abs(alpha(xi_hat) - alpha(xi)) / max(abs(alpha(xi)), 1e-10)
        print(f"  {str(xi):>22}  {str(xi_hat):>32}  {err:>10.2e}  {a_err:>10.2e}")

    # save
    CKPT.parent.mkdir(exist_ok=True)
    eqx.tree_serialise_leaves(CKPT, model)
    print(f"\nsaved {CKPT}")


def load(key: jax.Array = jax.random.PRNGKey(0),
         hidden: tuple[int, ...] = (64, 64, 64, 64)) -> MLP:
    """Reload trained inference closure."""
    skeleton = MLP(hidden=hidden, key=key)
    return eqx.tree_deserialise_leaves(CKPT, skeleton)


if __name__ == "__main__":
    main()
