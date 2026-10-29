"""Checkpoint, equivariance, and optimizer checks; optional compute dependencies."""

import numpy as np
import pytest

pytest.importorskip("equinox")
pytest.importorskip("optax")
from bot.closures.homog_nn import HomogClosure, gram_features, load, make_dataset, save, train


@pytest.mark.parametrize("order", [3, 4])
def test_complex_scaling_equivariance(order):
    rng = np.random.default_rng(5)
    moments = rng.normal(size=order) + 1j * rng.normal(size=order)
    phi = 0.3 - 0.7j
    factor = 1.7 * np.exp(1.3j)
    closure = HomogClosure(load(order))
    np.testing.assert_allclose(
        gram_features(factor * moments, factor * phi), gram_features(moments, phi), atol=1e-15
    )
    np.testing.assert_allclose(
        closure(factor * moments, factor * phi),
        factor * closure(moments, phi),
        rtol=2e-6,
        atol=1e-7,
    )
    assert closure(np.zeros(order), 0j) == 0j


def test_sampler_and_training_round_trip(tmp_path):
    features, moments, phi, target, single, zetas = make_dataset(3, 64, return_zetas=True)
    assert features.shape == (64, 16)
    assert 64 <= len(zetas) <= 128
    assert single.any() and (~single).any()
    assert np.all(np.isfinite(features))
    model = train(3, n_samples=64, n_steps=2, hidden=(8,), batch=16, log_every=1)
    path = tmp_path / "smoke.eqx"
    save(model, 3, path)
    restored = load(3, path, hidden=(8,))
    np.testing.assert_array_equal(
        HomogClosure(model).coeffs(moments[0], phi[0]),
        HomogClosure(restored).coeffs(moments[0], phi[0]),
    )
