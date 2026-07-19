"""Tests for the gradient-free random-walk Metropolis-Hastings sampler."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from jaxptpolypol.sampler import run_rwmh


def test_rwmh_recovers_correlated_gaussian():
    """RWMH should recover the mean and covariance of a correlated Gaussian."""
    # 3D target with a known non-diagonal precision (corr 0.6).
    corr = 0.6
    cov_true = jnp.array(
        [
            [1.0, corr, corr],
            [corr, 1.0, corr],
            [corr, corr, 1.0],
        ]
    )
    precision = jnp.linalg.inv(cov_true)

    def log_posterior(x):
        return -0.5 * x @ precision @ x

    key = jr.PRNGKey(0)
    samples, diag = run_rwmh(
        key,
        log_posterior,
        jnp.zeros(3),
        num_samples=20_000,
        num_chains=2,
    )

    assert samples.shape == (2, 20_000, 3)

    # Discard burn-in and pool chains.
    pooled = samples[:, 5_000:, :].reshape(-1, 3)

    sample_mean = np.asarray(pooled.mean(axis=0))
    sample_cov = np.asarray(jnp.cov(pooled.T))
    cov_np = np.asarray(cov_true)

    # Mean within 0.15 of 0 per component.
    assert np.all(np.abs(sample_mean) < 0.15)

    # Diagonal covariance within 25% (relative); off-diagonal within 0.15 (abs).
    for i in range(3):
        assert abs(sample_cov[i, i] - cov_np[i, i]) < 0.25 * cov_np[i, i]
        for j in range(3):
            if i != j:
                assert abs(sample_cov[i, j] - cov_np[i, j]) < 0.15

    acc = np.asarray(diag["acceptance_rate"])
    assert acc.shape == (2,)
    assert np.all(acc > 0.1)
    assert np.all(acc < 0.6)


def test_rwmh_shapes_and_determinism():
    """Same key -> identical samples; shapes correct; scalar sigma broadcasts."""
    def log_posterior(x):
        return -0.5 * jnp.sum(x ** 2)

    key = jr.PRNGKey(42)
    init = jnp.zeros(4)

    samples_a, diag_a = run_rwmh(
        key, log_posterior, init, num_samples=500, num_chains=3,
        proposal_sigma=0.8,
    )
    samples_b, diag_b = run_rwmh(
        key, log_posterior, init, num_samples=500, num_chains=3,
        proposal_sigma=0.8,
    )

    # Shapes.
    assert samples_a.shape == (3, 500, 4)
    assert diag_a["acceptance_rate"].shape == (3,)

    # Scalar proposal_sigma broadcast to a d-vector.
    assert diag_a["proposal_sigma"].shape == (4,)
    assert np.allclose(np.asarray(diag_a["proposal_sigma"]), 0.8)

    # Determinism.
    assert np.array_equal(np.asarray(samples_a), np.asarray(samples_b))
    assert np.array_equal(
        np.asarray(diag_a["acceptance_rate"]),
        np.asarray(diag_b["acceptance_rate"]),
    )


def test_rwmh_default_proposal_sigma():
    """Default proposal_sigma is (2.38 / sqrt(d)) * ones(d)."""
    def log_posterior(x):
        return -0.5 * jnp.sum(x ** 2)

    d = 5
    _, diag = run_rwmh(
        jr.PRNGKey(1), log_posterior, jnp.zeros(d),
        num_samples=200, num_chains=1,
    )
    expected = (2.38 / np.sqrt(d)) * np.ones(d)
    assert np.allclose(np.asarray(diag["proposal_sigma"]), expected)
