"""N=4 moment hierarchy closure: NN(r_1, r_2, r_3) -> U_4 / U_0 = alpha4.

This is the "N=3 closure" in the sense that it uses THREE moment ratios
(r_1 = U_1/U_0, r_2 = U_2/U_0, r_3 = U_3/U_0) as inputs to close the
4-moment beam fluid system at U_4.

Information-counting argument
------------------------------
For a two-mode superposition w_1 delta_1 + w_2 delta_2:

  r_j = (w_1 U_j^(1) + w_2 U_j^(2)) / (w_1 U_0^(1) + w_2 U_0^(2)),  j=1,2,3

On the eigenmode manifold U_j^(m) = alpha_{j}(xi_m) * U_0^(m).  Writing
g_m = w_m U_0^(m) (a single complex weight for each mode), the ratios
r_1, r_2, r_3 satisfy:

  r_j = (g_1 xi_m^{f_j}(...) + g_2 xi_2^{f_j}(...)) / (g_1 + g_2)

After absorbing the equal denominator this gives 6 real equations for the
6 real unknowns (Re g_1, Im g_1, Re xi_1, Im xi_1, Re xi_2, Im xi_2)
(the overall amplitude is fixed).  So for a GENERIC two-mode superposition
the map (r_1, r_2, r_3) -> (xi_1, xi_2, g_1/g_2) is generically uniquely
invertible, meaning the correct alpha4 value IS a well-defined function
of (r_1, r_2, r_3).  This is in contrast to the N=3 closure which uses only
(r_1, r_2) -- 4 real equations for 6 unknowns -- which is underdetermined.

Architecture
------------
MLP_n4: 6 real inputs [Re(r1), Im(r1), Re(r2), Im(r2), Re(r3), Im(r3)]
        -> 2 real outputs [Re(alpha4), Im(alpha4)]

Training: sample xi_b from Re in (-3, 3), Im in (-1, 1.5), compute exact
alpha4(xi_b) from the plasma dispersion function.  Features are
moment_ratios_n4(xi_b) = (r1=xi_b, r2=beta(xi_b), r3=alpha(xi_b)).

On the single-eigenmode manifold r_1 = xi_b exactly, so the NN is
effectively learning a 6->2 regression of a scalar holomorphic function.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from bot.closures.inference import (alpha4 as exact_alpha4,
                                     alpha as exact_alpha,
                                     moment_ratios_n4 as _moment_ratios_n4,
                                     Zprime)
from bot.closures.u2 import beta as exact_beta


CKPT       = Path(__file__).resolve().parent.parent / "runs" / "n4_nn.eqx"
CKPT_SUPER = Path(__file__).resolve().parent.parent / "runs" / "n4_nn_super.eqx"


# ---------------------------------------------------------------------------
# MLP architecture: 6 real inputs -> 2 real outputs
# ---------------------------------------------------------------------------

class MLP_n4(eqx.Module):
    """MLP for the N=4 hierarchy closure: 6 real inputs -> 2 real outputs."""
    layers: list

    def __init__(self, hidden: tuple[int, ...] = (64, 64, 64, 64),
                 *, key: jax.Array):
        keys = jax.random.split(key, len(hidden) + 1)
        sizes = (6,) + hidden + (2,)
        self.layers = [
            eqx.nn.Linear(in_, out_, key=k)
            for in_, out_, k in zip(sizes[:-1], sizes[1:], keys)
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jnp.tanh(layer(x))
        return self.layers[-1](x)


def _ratios_to_features_n4(r1: complex, r2: complex, r3: complex) -> jnp.ndarray:
    """Stack (re/im of r1, r2, r3) into a 6-real vector."""
    return jnp.stack([
        jnp.array(float(r1.real)), jnp.array(float(r1.imag)),
        jnp.array(float(r2.real)), jnp.array(float(r2.imag)),
        jnp.array(float(r3.real)), jnp.array(float(r3.imag)),
    ])


def n4_predict_alpha4(model: MLP_n4, r1, r2, r3) -> complex:
    """Run N=4 NN; return alpha4_hat = re + 1j * im."""
    feats = _ratios_to_features_n4(r1, r2, r3)
    out = model(feats)
    return out[0] + 1j * out[1]


# ---------------------------------------------------------------------------
# Dataset and training
# ---------------------------------------------------------------------------

@dataclass
class Dataset_n4:
    features: jnp.ndarray   # (N, 6)  -- re/im of (r1, r2, r3)
    targets: jnp.ndarray     # (N, 2)  -- re/im of alpha4(xi_b)


def make_dataset_n4(N: int = 16384,
                    xi_re_range: tuple[float, float] = (-3.0, 3.0),
                    xi_im_range: tuple[float, float] = (-1.0, 1.5),
                    seed: int = 0) -> Dataset_n4:
    """Sample xi_b in a complex rectangle; compute (r1,r2,r3) and alpha4."""
    rng = np.random.default_rng(seed)
    re = rng.uniform(*xi_re_range, size=N)
    im = rng.uniform(*xi_im_range, size=N)
    xi = re + 1j * im

    r1 = np.empty(N, dtype=complex)
    r2 = np.empty(N, dtype=complex)
    r3 = np.empty(N, dtype=complex)
    a4 = np.empty(N, dtype=complex)
    for i, x in enumerate(xi):
        r1[i], r2[i], r3[i] = _moment_ratios_n4(x)
        a4[i] = exact_alpha4(x)

    feats = np.stack([r1.real, r1.imag, r2.real, r2.imag,
                      r3.real, r3.imag], axis=1).astype(np.float32)
    targs = np.stack([a4.real, a4.imag], axis=1).astype(np.float32)
    return Dataset_n4(jnp.asarray(feats), jnp.asarray(targs))


def make_dataset_n4_superpositions(
        N: int = 16384,
        xi_re_range: tuple[float, float] = (-3.0, 3.0),
        xi_im_range: tuple[float, float] = (-1.0, 1.5),
        seed: int = 100) -> Dataset_n4:
    """Two-mode superposition training data for the N=4 closure.

    Sampling strategy
    -----------------
    Draw two independent eigenmode parameters xi_1, xi_2 from the training
    rectangle and a complex mixing weight w.  The effective feature vector and
    target are:

        r_j = (1-w)*alpha_j(xi_1) + w*alpha_j(xi_2)   j = 0,1,2,3
        target = r_4 = (1-w)*alpha4(xi_1) + w*alpha4(xi_2)

    where alpha_0 = 1, alpha_1 = xi, alpha_2 = beta(xi), alpha_3 = alpha(xi).

    The complex weight w is parametrised via the amplitude ratio
        a = g_2/g_1,   log|a| ~ Uniform(-1.5, 1.5),   arg(a) ~ Uniform(0, 2pi)
        w = a / (1 + a)

    This covers:
    * Real w near 0.5 -- the equal-amplitude chord test
    * Complex w with varying phase -- the physical time-evolution case, where
      the two-mode amplitudes acquire complex relative phases as they propagate
    * A broad range of amplitude ratios (1/4.5 to 4.5)

    The map (r_1, r_2, r_3) -> alpha4 is well-defined (unique) for this data
    because we have 6 real equations for 6 real unknowns at N=4.  Training
    on this data teaches the NN to correctly interpolate between modes rather
    than commit the Jensen error by evaluating alpha4 at the "mean" xi.

    Degenerate pairs (|xi_1 - xi_2| < 0.15) are excluded; targets with
    |alpha4| > 50 are excluded to avoid numerical outliers.
    """
    rng = np.random.default_rng(seed)
    feats_list = []
    targs_list = []

    # Generate more than N to account for rejections
    batch = max(N * 3, 32768)

    re1 = rng.uniform(*xi_re_range, size=batch)
    im1 = rng.uniform(*xi_im_range, size=batch)
    re2 = rng.uniform(*xi_re_range, size=batch)
    im2 = rng.uniform(*xi_im_range, size=batch)
    xi1 = re1 + 1j * im1
    xi2 = re2 + 1j * im2

    # Complex mixing weight via amplitude ratio a = g2/g1
    log_abs_a = rng.uniform(-1.5, 1.5, size=batch)
    phi       = rng.uniform(0, 2 * np.pi, size=batch)
    a = np.exp(log_abs_a) * np.exp(1j * phi)
    w = a / (1 + a)   # complex weight: both |w| and |1-w| are bounded

    for i in range(batch):
        # Skip near-degenerate pairs
        if abs(xi1[i] - xi2[i]) < 0.15:
            continue
        try:
            # Moment ratios on each eigenmode
            b1, b2 = exact_beta(xi1[i]),  exact_beta(xi2[i])
            al1, al2 = exact_alpha(xi1[i]), exact_alpha(xi2[i])
            a41, a42 = exact_alpha4(xi1[i]), exact_alpha4(xi2[i])
        except Exception:
            continue

        # Mixed inputs
        wi = w[i]
        r1 = (1 - wi) * xi1[i]  + wi * xi2[i]
        r2 = (1 - wi) * b1      + wi * b2
        r3 = (1 - wi) * al1     + wi * al2
        a4 = (1 - wi) * a41     + wi * a42   # correct "chord" target

        # Skip numerical outliers
        if not (np.isfinite(a4) and abs(a4) < 50.0):
            continue

        feats_list.append([r1.real, r1.imag, r2.real, r2.imag, r3.real, r3.imag])
        targs_list.append([a4.real, a4.imag])

        if len(feats_list) >= N:
            break

    if len(feats_list) < N:
        raise RuntimeError(
            f"Only {len(feats_list)} valid superposition samples generated "
            f"(needed {N}).  Try a larger batch or looser thresholds."
        )

    feats = np.array(feats_list[:N], dtype=np.float32)
    targs = np.array(targs_list[:N], dtype=np.float32)
    return Dataset_n4(jnp.asarray(feats), jnp.asarray(targs))


def make_dataset_n4_real_chord(
        N: int = 16384,
        xi_re_range: tuple[float, float] = (-3.0, 3.0),
        xi_im_range: tuple[float, float] = (-1.0, 1.5),
        seed: int = 200) -> Dataset_n4:
    """Two-mode superposition data with REAL mixing weight w ∈ (0, 1).

    This targets the equal-amplitude chord test: given two modes with the
    same initial electric field amplitude, the physically relevant mixing
    weight w is real.  Complex w arises during time evolution as modes
    accumulate relative phase; real w covers the t=0 state and the
    static chord-test figure.

    The real-chord data is combined with complex-w data in
    make_dataset_n4_combined to ensure broad coverage.
    """
    rng = np.random.default_rng(seed)
    feats_list = []
    targs_list = []

    batch = max(N * 2, 16384)
    re1 = rng.uniform(*xi_re_range, size=batch)
    im1 = rng.uniform(*xi_im_range, size=batch)
    re2 = rng.uniform(*xi_re_range, size=batch)
    im2 = rng.uniform(*xi_im_range, size=batch)
    xi1 = re1 + 1j * im1
    xi2 = re2 + 1j * im2
    # Real w uniformly in (0.02, 0.98) — avoid single-mode limits
    w_vals = rng.uniform(0.02, 0.98, size=batch)

    for i in range(batch):
        if abs(xi1[i] - xi2[i]) < 0.15:
            continue
        try:
            b1, b2   = exact_beta(xi1[i]),  exact_beta(xi2[i])
            al1, al2 = exact_alpha(xi1[i]), exact_alpha(xi2[i])
            a41, a42 = exact_alpha4(xi1[i]), exact_alpha4(xi2[i])
        except Exception:
            continue

        wi = w_vals[i]
        r1 = (1 - wi) * xi1[i]  + wi * xi2[i]
        r2 = (1 - wi) * b1      + wi * b2
        r3 = (1 - wi) * al1     + wi * al2
        a4 = (1 - wi) * a41     + wi * a42

        if not (np.isfinite(a4) and abs(a4) < 50.0):
            continue

        feats_list.append([r1.real, r1.imag, r2.real, r2.imag, r3.real, r3.imag])
        targs_list.append([a4.real, a4.imag])
        if len(feats_list) >= N:
            break

    if len(feats_list) < N:
        raise RuntimeError(f"Only {len(feats_list)} valid chord samples generated.")

    feats = np.array(feats_list[:N], dtype=np.float32)
    targs = np.array(targs_list[:N], dtype=np.float32)
    return Dataset_n4(jnp.asarray(feats), jnp.asarray(targs))


def make_dataset_n4_combined(
        N_single: int = 16384,
        N_super:  int = 16384,
        N_chord:  int = 16384,
        xi_re_range: tuple[float, float] = (-3.0, 3.0),
        xi_im_range: tuple[float, float] = (-1.0, 1.5),
        seed: int = 0) -> Dataset_n4:
    """Combined single-eigenmode + complex-w superposition + real-chord dataset.

    Three equal-size components:
      * Single-mode (eigenmode manifold): exact by construction, N=N_single
      * Complex-w superposition: covers the time-evolution off-manifold regime
      * Real-chord: covers the equal-amplitude chord test and w ∈ (0,1) real

    Single-mode data teaches the NN to be exact on the eigenmode manifold.
    Complex superposition data covers the general time-evolution case.
    Real-chord data specifically targets the equal-amplitude mixing test
    (which is measure zero in the complex-w distribution).
    """
    ds_single = make_dataset_n4(N=N_single, xi_re_range=xi_re_range,
                                  xi_im_range=xi_im_range, seed=seed)
    ds_super  = make_dataset_n4_superpositions(N=N_super,
                                                xi_re_range=xi_re_range,
                                                xi_im_range=xi_im_range,
                                                seed=seed + 999)
    ds_chord  = make_dataset_n4_real_chord(N=N_chord,
                                            xi_re_range=xi_re_range,
                                            xi_im_range=xi_im_range,
                                            seed=seed + 777)

    rng = np.random.default_rng(seed + 7)
    feats = np.concatenate([np.array(ds_single.features),
                             np.array(ds_super.features),
                             np.array(ds_chord.features)], axis=0)
    targs = np.concatenate([np.array(ds_single.targets),
                             np.array(ds_super.targets),
                             np.array(ds_chord.targets)], axis=0)
    perm  = rng.permutation(len(feats))
    return Dataset_n4(jnp.asarray(feats[perm].astype(np.float32)),
                      jnp.asarray(targs[perm].astype(np.float32)))


def _n4_loss(model: MLP_n4, bx: jnp.ndarray, by: jnp.ndarray) -> jnp.ndarray:
    preds = jax.vmap(model)(bx)
    return jnp.mean((preds - by) ** 2)


@eqx.filter_jit
def _n4_step(model, opt_state, bx, by, optim):
    loss, grads = eqx.filter_value_and_grad(_n4_loss)(model, bx, by)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def train_n4(model: MLP_n4, ds: Dataset_n4,
             n_steps: int = 10000, lr: float = 3e-3, lr_end: float = 1e-5,
             batch_size: int = 512, seed: int = 0,
             log_every: int = 1000) -> tuple[MLP_n4, list[float]]:
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
        model, opt_state, loss = _n4_step(model, opt_state, bx, by, optim)
        losses.append(float(loss))
        if step % log_every == 0 or step == n_steps - 1:
            print(f"  step {step:5d}   loss = {float(loss):.4e}")
    return model, losses


def save(model: MLP_n4, path: Path = CKPT) -> None:
    path.parent.mkdir(exist_ok=True)
    eqx.tree_serialise_leaves(str(path), model)


def load(key: jax.Array = jax.random.PRNGKey(0),
         hidden: tuple[int, ...] = (64, 64, 64, 64)) -> MLP_n4:
    skeleton = MLP_n4(hidden=hidden, key=key)
    return eqx.tree_deserialise_leaves(str(CKPT), skeleton)


def load_super(key: jax.Array = jax.random.PRNGKey(0),
               hidden: tuple[int, ...] = (64, 64, 64, 64)) -> MLP_n4:
    """Load the superposition-trained N=4 model."""
    skeleton = MLP_n4(hidden=hidden, key=key)
    return eqx.tree_deserialise_leaves(str(CKPT_SUPER), skeleton)


# ---------------------------------------------------------------------------
# Closure interface
# ---------------------------------------------------------------------------

def n4_U4_over_U0(model: MLP_n4,
                   U0: complex, U1: complex, U2: complex, U3: complex
                   ) -> complex:
    """N=4 NN closure: return alpha4_hat from (U0, U1, U2, U3)."""
    if abs(U0) < 1e-15:
        return 0.0 + 0.0j
    r1 = U1 / U0
    r2 = U2 / U0
    r3 = U3 / U0
    return complex(n4_predict_alpha4(model, r1, r2, r3))


# ---------------------------------------------------------------------------
# Direct (analytic) N=4 closure for comparison
# ---------------------------------------------------------------------------

def direct_n4_alpha4(U0: complex, U1: complex, U2: complex, U3: complex
                     ) -> complex:
    """Exact per-mode alpha4 closure: alpha4(U1/U0).

    Uses only r1 = U1/U0, evaluating the exact formula at the 'mean' xi.
    Equivalent to the direct-alpha closure at the N=4 level -- exact on a
    single eigenmode, commits Jensen error off-manifold.
    """
    if abs(U0) < 1e-15:
        return 0.0 + 0.0j
    return complex(exact_alpha4(U1 / U0))


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------

def main(hidden: tuple[int, ...] = (64, 64, 64, 64),
          n_steps: int = 10000, lr: float = 3e-3,
          N_train: int = 16384, seed: int = 42) -> None:
    print("=== Training N=4 closure (NN outputs alpha4 from r1,r2,r3) ===")
    print(f"  hidden:        {hidden}")
    print(f"  steps:         {n_steps}")
    print(f"  train samples: {N_train}\n")

    ds_train = make_dataset_n4(N=N_train, seed=seed)
    ds_val   = make_dataset_n4(N=2048, seed=seed + 1)

    key   = jax.random.PRNGKey(seed)
    model = MLP_n4(hidden=hidden, key=key)
    n_params = sum(int(p.size) for p in jax.tree.leaves(model)
                   if hasattr(p, "size"))
    print(f"  params: {n_params}\n")

    model, losses = train_n4(model, ds_train, n_steps=n_steps, lr=lr,
                              batch_size=512, seed=seed, log_every=1000)

    val_loss = float(jnp.mean(
        (jax.vmap(model)(ds_val.features) - ds_val.targets) ** 2
    ))
    print(f"\n  final val MSE: {val_loss:.4e}")

    print(f"\n=== alpha4 recovery on representative points ===")
    test_xis = [0.5 + 0.0j, -0.83 + 0.05j, -1.5 - 0.2j, 1.2 + 0.1j]
    print(f"  {'xi_true':>16}  {'alpha4_nn':>36}  {'alpha4_true':>36}  {'rel err':>10}")
    for xi in test_xis:
        r1, r2, r3 = _moment_ratios_n4(xi)
        a4_nn   = complex(n4_predict_alpha4(model, r1, r2, r3))
        a4_true = exact_alpha4(xi)
        err = abs(a4_nn - a4_true) / (abs(a4_true) + 1e-30)
        print(f"  {str(xi):>16}  {str(a4_nn):>36}  {str(a4_true):>36}  {err:>10.2e}")

    save(model)
    print(f"\nsaved {CKPT}")


def main_super(hidden: tuple[int, ...] = (64, 64, 64, 64),
               n_steps: int = 30000, lr: float = 3e-3,
               N_single: int = 16384, N_super: int = 16384, N_chord: int = 16384,
               seed: int = 42) -> None:
    """Train N=4 NN on combined single-mode + superposition data.

    The superposition data provides correct (r1,r2,r3)->alpha4 targets for
    two-mode states, which eliminates the Jensen error that the single-mode-
    trained model accumulates off the eigenmode manifold.

    Key difference from N=3: the N=4 superposition targets are CONSISTENT
    (the map (r1,r2,r3)->alpha4 is well-defined for two-mode states because
    we have 6 real equations for 6 unknowns).  Training converges to a unique
    solution.  For N=3, training on superpositions fails because the same
    input (r1,r2) can map to different correct alpha values depending on which
    two-mode decomposition is meant -- the targets are contradictory.
    """
    print("=== Training N=4 NN with superposition data ===")
    print(f"  hidden:    {hidden}")
    print(f"  steps:     {n_steps}")
    print(f"  N_single:  {N_single}  (eigenmode manifold)")
    print(f"  N_super:   {N_super}   (complex-w superpositions)")
    print(f"  N_chord:   {N_chord}   (real-w chord data)")
    print()

    print("Building combined dataset...")
    ds_train = make_dataset_n4_combined(N_single=N_single, N_super=N_super,
                                         N_chord=N_chord, seed=seed)
    ds_val_s = make_dataset_n4(N=1024, seed=seed + 1)
    ds_val_2 = make_dataset_n4_superpositions(N=1024, seed=seed + 2)
    ds_val_c = make_dataset_n4_real_chord(N=1024, seed=seed + 3)
    print(f"  combined train size: {ds_train.features.shape[0]}\n")

    key   = jax.random.PRNGKey(seed)
    model = MLP_n4(hidden=hidden, key=key)
    n_params = sum(int(p.size) for p in jax.tree.leaves(model)
                   if hasattr(p, "size"))
    print(f"  params: {n_params}\n")

    model, losses = train_n4(model, ds_train, n_steps=n_steps, lr=lr,
                              batch_size=512, seed=seed, log_every=1000)

    val_s = float(jnp.mean((jax.vmap(model)(ds_val_s.features) - ds_val_s.targets)**2))
    val_2 = float(jnp.mean((jax.vmap(model)(ds_val_2.features) - ds_val_2.targets)**2))
    val_c = float(jnp.mean((jax.vmap(model)(ds_val_c.features) - ds_val_c.targets)**2))
    print(f"\n  val MSE single-mode:    {val_s:.4e}")
    print(f"  val MSE complex-super:  {val_2:.4e}")
    print(f"  val MSE real-chord:     {val_c:.4e}")

    # Spot checks on eigenmode manifold
    print(f"\n=== Eigenmode manifold recovery ===")
    test_xis = [0.5 + 0.0j, -0.83 + 0.05j, -1.5 - 0.2j, 1.2 + 0.1j]
    for xi in test_xis:
        r1, r2, r3 = _moment_ratios_n4(xi)
        a4_nn   = complex(n4_predict_alpha4(model, r1, r2, r3))
        a4_true = exact_alpha4(xi)
        err = abs(a4_nn - a4_true) / (abs(a4_true) + 1e-30)
        print(f"  xi={str(xi):>16}  rel_err={err:.2e}")

    # Spot checks on chord (real w, equal-amplitude superposition)
    print(f"\n=== Chord test (real w=0.5, xi1=1.22-0.61i, xi2=0.16-0.08i) ===")
    xi1, xi2 = 1.225 - 0.612j, 0.164 - 0.078j
    b1, b2   = exact_beta(xi1), exact_beta(xi2)
    al1, al2 = exact_alpha(xi1), exact_alpha(xi2)
    a41, a42 = exact_alpha4(xi1), exact_alpha4(xi2)
    for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
        r1_in = (1-w)*xi1 + w*xi2
        r2_in = (1-w)*b1  + w*b2
        r3_in = (1-w)*al1 + w*al2
        a4_chord = (1-w)*a41 + w*a42
        a4_nn    = complex(n4_predict_alpha4(model, r1_in, r2_in, r3_in))
        err = abs(a4_nn - a4_chord) / (abs(a4_chord) + 1e-30)
        print(f"  w={w:.2f}  chord={a4_chord:.4f}  nn={a4_nn:.4f}  rel_err={err:.3f}")

    save(model, CKPT_SUPER)
    print(f"\nsaved {CKPT_SUPER}")


if __name__ == "__main__":
    import sys
    if "--super" in sys.argv:
        main_super()
    else:
        main()
