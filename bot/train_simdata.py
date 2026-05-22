"""Train a NN closure on data generated from kinetic simulations.

Comparison point for the closed-form-trained NNs. The hypothesis: training on
simulation trajectories biases the dataset toward specific eigenmodes (the
unstable Langmuir-like one dominates after the transient), giving worse
coverage of the (moment_ratios -> alpha) map than uniform xi sampling.

We deliberately train at ONE operating point only -- (u_b=5, eps=0.05) at
a few k values -- to expose the generalization gap. The closed-form NNs
are problem-agnostic in xi.
"""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from bot.closures.inference import MLP, alpha as exact_alpha
from bot.closures.naive_nn import NaiveDataset, train_naive, _naive_step
from bot.kinetic import kinetic_system_sspace, s_grid


SQRT_PI = float(np.sqrt(np.pi))
CKPT = Path(__file__).resolve().parent / "runs" / "sim_mlp.eqx"
CKPT_BROAD = Path(__file__).resolve().parent / "runs" / "sim_mlp_broad.eqx"


def make_sim_dataset(operating_points: list[tuple[float, float]] | None = None,
                     u_b: float = 5.0, eps: float = 0.05,
                     ks: tuple[float, ...] = (0.18, 0.22, 0.26, 0.30, 0.34),
                     n_ics: int = 40, t_end: float = 60.0, n_t: int = 300,
                     N_v: int = 96, V: float = 6.0,
                     amp: float = 1e-3, seed: int = 0
                     ) -> NaiveDataset:
    """Generate (moment_ratios, alpha) pairs from kinetic s-space simulations.

    If `operating_points` is supplied, iterate over all (u_b, eps) pairs and
    use the same `ks` for each. Otherwise use the single (u_b, eps) pair.
    """
    s, ds = s_grid(N_v, V)
    rng = np.random.default_rng(seed)
    feats_list, targs_list = [], []

    if operating_points is None:
        operating_points = [(u_b, eps)]

    for ub_i, ep_i in operating_points:
     for k in ks:
        A = kinetic_system_sspace(k, ub_i, ep_i, N_v, V)
        eigvals, evecs = np.linalg.eig(A)

        for ic_idx in range(n_ics):
            # random IC: small random bulk + random low-order beam perturbation
            y0 = np.zeros(2 + N_v, dtype=complex)
            y0[0] = amp * (rng.normal() + 1j * rng.normal()) / np.sqrt(2)
            y0[1] = amp * (rng.normal() + 1j * rng.normal()) / np.sqrt(2)
            # beam perturbation: a few low-order Hermite-Gauss modes
            envelope = np.exp(-s**2 / 2)
            n_modes = 6
            coefs = (rng.normal(size=n_modes) + 1j * rng.normal(size=n_modes)) / np.sqrt(n_modes)
            for n, c in enumerate(coefs):
                Hn = np.polynomial.hermite_e.hermeval(s, [0]*n + [1])
                y0[2:] += amp * c * envelope * Hn

            # evolve via matrix exp
            coefs_lam = np.linalg.solve(evecs, y0)
            t_grid = np.linspace(0, t_end, n_t)
            for ti in t_grid:
                y = evecs @ (coefs_lam * np.exp(eigvals * ti))
                F = y[2:]
                U0 = ds * np.sum(F)
                U1 = ds * np.sum(s * F)
                U2 = ds * np.sum(s**2 * F)
                U3 = ds * np.sum(s**3 * F)
                if abs(U0) < 1e-12:
                    continue
                r1 = U1 / U0
                r2 = U2 / U0
                alpha = U3 / U0
                feats_list.append([r1.real, r1.imag, r2.real, r2.imag])
                targs_list.append([alpha.real, alpha.imag])

    feats = np.array(feats_list, dtype=np.float32)
    targs = np.array(targs_list, dtype=np.float32)
    # filter out wildly off-distribution samples (huge moments at the very
    # beginning of trajectories before mode projection)
    mag = np.linalg.norm(feats, axis=1)
    mask = mag < 50.0    # generous bound
    feats = feats[mask]
    targs = targs[mask]
    print(f"  generated {len(feats)} samples from {len(ks)} k values × "
          f"{n_ics} ICs at (u_b={u_b}, eps={eps})")
    return NaiveDataset(jnp.asarray(feats), jnp.asarray(targs))


def main(n_steps: int = 8000, seed: int = 42, broad: bool = False) -> None:
    if broad:
        # Broad sampling: many (u_b, eps) pairs to cover xi-space more uniformly
        u_b_vals = [3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        eps_vals = [0.02, 0.05, 0.10, 0.20]
        op_pts = [(u, e) for u in u_b_vals for e in eps_vals]
        # also use a wider k range to catch more xi values
        ks = (0.12, 0.18, 0.24, 0.30, 0.36)
        n_ics = 12   # fewer ICs per op point to keep dataset reasonable
        print(f"=== Generating BROAD sim training data ({len(op_pts)} op pts, {len(ks)} k each) ===")
        ds_train = make_sim_dataset(operating_points=op_pts, ks=ks, n_ics=n_ics, seed=seed)
        ds_val = make_sim_dataset(operating_points=[(5.0, 0.05), (7.0, 0.10)],
                                   ks=(0.22, 0.30), n_ics=6, seed=seed + 100)
        out_path = CKPT_BROAD
    else:
        print("=== Generating sim training data ===")
        ds_train = make_sim_dataset(seed=seed)
        ds_val = make_sim_dataset(seed=seed + 100, ks=(0.20, 0.28), n_ics=20)
        out_path = CKPT

    key = jax.random.PRNGKey(seed)
    model = MLP(hidden=(64, 64, 64, 64), key=key)
    print(f"\n  training MLP on simulation data ({n_steps} steps)...\n")

    model, _ = train_naive(model, ds_train, n_steps=n_steps,
                             lr=3e-3, batch_size=512, seed=seed)

    # eval on val set (same operating point)
    pred = jax.vmap(model)(ds_val.features)
    val_mse = float(jnp.mean((pred - ds_val.targets) ** 2))
    print(f"\n  val MSE (same operating point): {val_mse:.4e}")

    # eval on canonical xi sweep (problem-agnostic test)
    print(f"\n  Testing on canonical xi sweep (closed-form alpha):")
    from bot.closures.inference import moment_ratios
    test_xis = [(-1.17 + 0.71j), (-0.83 + 0.05j), (0.5 + 0.0j),
                (-1.5 - 0.2j), (1.2 + 0.1j), (-2.0 + 0.0j)]
    for xi in test_xis:
        r1, r2 = moment_ratios(xi)
        feats = jnp.asarray([r1.real, r1.imag, r2.real, r2.imag],
                             dtype=jnp.float32)
        pred = model(feats)
        a_pred = complex(pred[0]) + 1j * complex(pred[1])
        a_true = exact_alpha(xi)
        err = abs(a_pred - a_true) / abs(a_true) if abs(a_true) > 1e-9 else abs(a_pred - a_true)
        print(f"    xi={xi}:  alpha_sim={a_pred:+.4f},  "
              f"alpha_true={a_true:+.4f},  rel err = {err:.2e}")

    out_path.parent.mkdir(exist_ok=True)
    eqx.tree_serialise_leaves(out_path, model)
    print(f"\nsaved {out_path}")


def load(key: jax.Array = jax.random.PRNGKey(0),
         hidden: tuple[int, ...] = (64, 64, 64, 64),
         broad: bool = False) -> MLP:
    path = CKPT_BROAD if broad else CKPT
    skeleton = MLP(hidden=hidden, key=key)
    return eqx.tree_deserialise_leaves(path, skeleton)


if __name__ == "__main__":
    import sys
    broad = "--broad" in sys.argv
    main(broad=broad)
