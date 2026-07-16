# tests/test_marginal_likelihood.py
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


def test_marginal_loglike_matches_woodbury_direct():
    from jaxptpolypol.marginal_likelihood import gaussian_marginal_loglike

    rng = np.random.default_rng(0)
    n_d, n_l = 8, 3
    M = rng.normal(size=(n_d, n_l))
    C = np.diag(rng.uniform(0.5, 2.0, size=n_d))
    mu_p = rng.normal(size=n_l)
    sigma_p = rng.uniform(0.5, 2.0, size=n_l)
    data = rng.normal(size=n_d)
    m0 = rng.normal(size=n_d)

    # Direct: Gaussian in data with covariance C + M Sigma_p M^T
    Cm = C + M @ np.diag(sigma_p**2) @ M.T
    r = data - m0 - M @ mu_p
    direct = -0.5 * (r @ np.linalg.solve(Cm, r) + np.linalg.slogdet(Cm)[1])
    # gaussian_marginal_loglike = direct + 0.5*logdet(C)  (constant offset)
    expected = direct + 0.5 * np.linalg.slogdet(C)[1]

    got = float(gaussian_marginal_loglike(
        jnp.asarray(data), jnp.asarray(m0), jnp.asarray(M),
        jnp.asarray(np.linalg.inv(C)), jnp.asarray(mu_p), jnp.asarray(sigma_p),
    ))
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_marginal_loglike_no_logdet_is_pure_quadratic():
    from jaxptpolypol.marginal_likelihood import gaussian_marginal_loglike

    rng = np.random.default_rng(1)
    n_d, n_l = 6, 2
    M = rng.normal(size=(n_d, n_l))
    C = np.eye(n_d)
    sigma_p = np.ones(n_l)
    mu_p = np.zeros(n_l)
    data = rng.normal(size=n_d)

    # With data == m0 and mu_p = 0: residual = 0, quadratic = 0
    got = float(gaussian_marginal_loglike(
        jnp.asarray(data), jnp.asarray(data), jnp.asarray(M),
        jnp.asarray(C), jnp.asarray(mu_p), jnp.asarray(sigma_p),
        include_logdet=False,
    ))
    np.testing.assert_allclose(got, 0.0, atol=1e-13)
