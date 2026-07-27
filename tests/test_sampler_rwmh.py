"""Tests for the gradient-free random-walk Metropolis-Hastings sampler."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from jaxptpolypol.sampler import run_rwmh, run_rwmh_python


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


# ---------------------------------------------------------------------------
# Python-driven RWMH (run_rwmh_python): lax.scan-free driver
# ---------------------------------------------------------------------------


def test_rwmh_python_recovers_correlated_gaussian():
    """Python-driven RWMH recovers the mean and covariance of a Gaussian."""
    # Same 3D target and thresholds as test_rwmh_recovers_correlated_gaussian.
    corr = 0.6
    cov_true = jnp.array(
        [
            [1.0, corr, corr],
            [corr, 1.0, corr],
            [corr, corr, 1.0],
        ]
    )
    precision = jnp.linalg.inv(cov_true)

    # The driver expects an already-jit-compiled log-posterior.
    @jax.jit
    def log_posterior(x):
        return -0.5 * x @ precision @ x

    key = jr.PRNGKey(0)
    samples, diag = run_rwmh_python(
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

    assert np.all(np.abs(sample_mean) < 0.15)

    for i in range(3):
        assert abs(sample_cov[i, i] - cov_np[i, i]) < 0.25 * cov_np[i, i]
        for j in range(3):
            if i != j:
                assert abs(sample_cov[i, j] - cov_np[i, j]) < 0.15

    acc = np.asarray(diag["acceptance_rate"])
    assert acc.shape == (2,)
    assert np.all(acc > 0.1)
    assert np.all(acc < 0.6)


def test_rwmh_python_determinism_and_shapes():
    """Same key -> identical samples; shapes; scalar sigma; thin subsamples."""
    @jax.jit
    def log_posterior(x):
        return -0.5 * jnp.sum(x ** 2)

    key = jr.PRNGKey(42)
    init = jnp.zeros(4)

    samples_a, diag_a = run_rwmh_python(
        key, log_posterior, init, num_samples=200, num_chains=3,
        proposal_sigma=0.8,
    )
    samples_b, diag_b = run_rwmh_python(
        key, log_posterior, init, num_samples=200, num_chains=3,
        proposal_sigma=0.8,
    )

    # Shapes.
    assert samples_a.shape == (3, 200, 4)
    assert diag_a["acceptance_rate"].shape == (3,)

    # Scalar proposal_sigma broadcast to a d-vector.
    assert diag_a["proposal_sigma"].shape == (4,)
    assert np.allclose(np.asarray(diag_a["proposal_sigma"]), 0.8)

    # Determinism: identical arrays for the same key.
    assert np.array_equal(np.asarray(samples_a), np.asarray(samples_b))
    assert np.array_equal(
        np.asarray(diag_a["acceptance_rate"]),
        np.asarray(diag_b["acceptance_rate"]),
    )

    # Thinning: num_samples is the number of KEPT draws, total steps =
    # num_samples * thin.  A thin=2 run over N kept draws walks the SAME
    # 2N-step chain (same key -> same jitter, same RNG stream) as a thin=1
    # run of 2N kept draws, subsampled to every 2nd state.
    N = 300
    key2 = jr.PRNGKey(7)
    full, _ = run_rwmh_python(
        key2, log_posterior, init, num_samples=2 * N, num_chains=1,
        proposal_sigma=0.8, thin=1,
    )
    thinned, _ = run_rwmh_python(
        key2, log_posterior, init, num_samples=N, num_chains=1,
        proposal_sigma=0.8, thin=2,
    )
    assert thinned.shape == (1, N, 4)
    assert np.array_equal(
        np.asarray(thinned[0]), np.asarray(full[0, 1::2])
    )


def test_rwmh_python_matches_contract_of_run_rwmh():
    """Diagnostics keys match run_rwmh; outputs are float64."""
    @jax.jit
    def log_posterior(x):
        return -0.5 * jnp.sum(x ** 2)

    key = jr.PRNGKey(3)
    init = jnp.zeros(4)

    samples_scan, diag_scan = run_rwmh(
        key, log_posterior, init, num_samples=100, num_chains=2,
    )
    samples_py, diag_py = run_rwmh_python(
        key, log_posterior, init, num_samples=100, num_chains=2,
    )

    # Identical diagnostics contract.
    assert set(diag_py.keys()) == set(diag_scan.keys())
    assert set(diag_py.keys()) == {"acceptance_rate", "proposal_sigma"}

    # float64 outputs.
    assert samples_py.dtype == jnp.float64
    assert diag_py["acceptance_rate"].dtype == jnp.float64
    assert diag_py["proposal_sigma"].dtype == jnp.float64


def test_cholesky_transform_roundtrip_and_isotropizes():
    """Cholesky whitening: exact round-trip, batch support, and the whitened
    image of cov-distributed samples is the isotropic unit Gaussian."""
    from jaxptpolypol.sampler import make_cholesky_transform

    rng = np.random.default_rng(7)
    d = 4
    A = rng.normal(size=(d, d))
    cov = A @ A.T + d * np.eye(d)          # SPD, non-trivial correlations
    center = rng.normal(size=d)

    to_w, to_p = make_cholesky_transform(center, cov)

    # exact round-trip, single vector
    theta = jnp.asarray(rng.normal(size=d))
    np.testing.assert_allclose(np.asarray(to_p(to_w(theta))), np.asarray(theta),
                               rtol=1e-12)

    # batched to_physical: (n, d) and (chains, n, d)
    xb = jnp.asarray(rng.normal(size=(50, d)))
    assert to_p(xb).shape == (50, d)
    assert to_p(xb[None]).shape == (1, 50, d)

    # to_physical maps unit-isotropic draws to cov-distributed draws
    z = rng.normal(size=(200_000, d))
    phys = np.asarray(to_p(jnp.asarray(z)))
    emp = np.cov(phys - center, rowvar=False)
    np.testing.assert_allclose(emp, cov, rtol=0.05, atol=0.05 * np.abs(cov).max())

    # and to_whitened isotropizes them again
    w = np.asarray(to_w(jnp.asarray(phys[0])))
    assert w.shape == (d,)
