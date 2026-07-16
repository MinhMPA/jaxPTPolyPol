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


SURVEY_KEYS_16 = (
    ('shared', 'bias', 'b1'), ('shared', 'bias', 'b2'), ('shared', 'bias', 'bG2'),
    ('shared', 'bias', 'bGamma3'), ('shared', 'stoch', 'P_shot'),
    ('shared', 'k_nl', None), ('shared', 'ndens', None),
    ('pk', 'ctr', 'c0'), ('pk', 'ctr', 'c2'), ('pk', 'ctr', 'c4'), ('pk', 'ctr', 'cfog'),
    ('pk', 'stoch', 'a0'), ('pk', 'stoch', 'a2'),
    ('bk', 'ctr', 'c1'), ('bk', 'stoch', 'B_shot'), ('bk', 'stoch', 'A_shot'),
)


def test_split_marginal_indices_layout():
    from jaxptpolypol.marginal_likelihood import LIN_SURVEY_KEYS, split_marginal_indices

    n_cosmo, n_bins = 9, 2
    split = split_marginal_indices(
        n_cosmo_params=n_cosmo, survey_keys=SURVEY_KEYS_16, n_bins=n_bins,
        fixed_cosmo=(5, 6, 7, 8),
        fixed_survey_keys={('shared', 'k_nl', None), ('shared', 'ndens', None)},
    )
    assert len(LIN_SURVEY_KEYS) == 11
    assert split.n_lin == 11 * n_bins
    # NL = 5 varied cosmo + (b1, b2, bG2) per bin
    assert split.n_nl == 5 + 3 * n_bins
    assert split.nl_idx[:5] == (0, 1, 2, 3, 4)
    # bin 0 survey starts at full index 9: b1,b2,bG2 at 9,10,11
    assert split.nl_idx[5:8] == (9, 10, 11)
    # bin 1 at 9+16=25: 25,26,27
    assert split.nl_idx[8:11] == (25, 26, 27)
    # lin bin 0: offsets {3,4,7..15} shifted by 9, in LIN_SURVEY_KEYS order
    assert split.lin_idx[:11] == (12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24)
    assert split.lin_idx[11:] == tuple(i + 16 for i in split.lin_idx[:11])
    # b1 positions inside the NL vector
    assert split.nl_b1_pos == (5, 8)
    # no overlap, no fixed keys anywhere
    assert set(split.nl_idx).isdisjoint(split.lin_idx)
    assert split.lin_keys[0] == (0, ('shared', 'bias', 'bGamma3'))


def test_split_rejects_unknown_lin_key():
    from jaxptpolypol.marginal_likelihood import split_marginal_indices
    import pytest
    bad_keys = tuple(k for k in SURVEY_KEYS_16 if k != ('bk', 'ctr', 'c1'))
    with pytest.raises(ValueError, match="c1"):
        split_marginal_indices(
            n_cosmo_params=9, survey_keys=bad_keys, n_bins=1,
            fixed_survey_keys={('shared', 'k_nl', None), ('shared', 'ndens', None)},
        )
