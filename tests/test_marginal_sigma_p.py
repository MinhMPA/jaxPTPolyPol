"""Full prior-covariance (Sigma_p) support in ``gaussian_marginal_loglike``.

Branch stream-b-sigmap (Task 5sigma part B). ``sigma_p`` accepts either the
historical 1-d diagonal widths or a 2-d full prior **covariance** ``Sigma_p``
(variances on the diagonal). These tests pin the ndim==2 convention:

  * loglike(sigma_p=widths) == loglike(sigma_p=jnp.diag(widths**2))  (equivalence);
  * a dense off-diagonal Sigma_p matches an independent numpy implementation of
    -2 ln L = r^T C^-1 r - b^T A^-1 b + ln det(A Sigma_p).
"""
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from jaxptpolypol.marginal_likelihood import gaussian_marginal_loglike


def _synthetic(seed=20260731, n_data=6, n_lin=3):
    """Fixed-seed synthetic ``(data, m0, M, cov_inv, mu_p, widths)``."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(n_data)
    m0 = rng.standard_normal(n_data)
    M = rng.standard_normal((n_data, n_lin))
    araw = rng.standard_normal((n_data, n_data))
    cov = araw @ araw.T + n_data * np.eye(n_data)      # SPD data covariance
    cov_inv = np.linalg.inv(cov)
    mu_p = rng.standard_normal(n_lin)
    widths = 0.5 + rng.random(n_lin)                   # strictly positive widths
    return (jnp.asarray(data), jnp.asarray(m0), jnp.asarray(M),
            jnp.asarray(cov_inv), jnp.asarray(mu_p), np.asarray(widths))


@pytest.mark.parametrize("include_logdet", [True, False])
def test_diag_cov_equivalence(include_logdet):
    """sigma_p=widths (ndim 1) == sigma_p=diag(widths**2) (ndim 2, covariance)."""
    data, m0, M, cov_inv, mu_p, widths = _synthetic()
    diag_ll = gaussian_marginal_loglike(
        data, m0, M, cov_inv, mu_p, jnp.asarray(widths),
        include_logdet=include_logdet)
    cov_ll = gaussian_marginal_loglike(
        data, m0, M, cov_inv, mu_p, jnp.diag(jnp.asarray(widths) ** 2),
        include_logdet=include_logdet)
    assert float(cov_ll) == pytest.approx(float(diag_ll), abs=1e-10)


def _ref_m2lnL(data, m0, M, cov_inv, mu_p, Sigma_p, include_logdet):
    """Independent numpy -2 ln L (no reuse of the production function)."""
    data = np.asarray(data); m0 = np.asarray(m0); M = np.asarray(M)
    cov_inv = np.asarray(cov_inv); mu_p = np.asarray(mu_p)
    Sigma_p = np.asarray(Sigma_p)
    resid = data - m0 - M @ mu_p
    A = M.T @ cov_inv @ M + np.linalg.inv(Sigma_p)
    b = M.T @ cov_inv @ resid
    quad = resid @ cov_inv @ resid - b @ np.linalg.solve(A, b)
    if not include_logdet:
        return quad
    _, logdet_A = np.linalg.slogdet(A)
    _, logdet_S = np.linalg.slogdet(Sigma_p)
    return quad + logdet_A + logdet_S


@pytest.mark.parametrize("include_logdet", [True, False])
def test_full_sigma_p_vs_numpy_reference(include_logdet):
    """Dense off-diagonal Sigma_p vs an independent numpy -2 ln L reference."""
    data, m0, M, cov_inv, mu_p, _ = _synthetic()
    rng = np.random.default_rng(99)
    n_lin = int(np.asarray(M).shape[1])
    fac = np.tril(rng.standard_normal((n_lin, n_lin)))
    np.fill_diagonal(fac, 0.7 + np.abs(np.diag(fac)))
    Sigma_p = fac @ fac.T                               # SPD, genuine off-diagonals
    assert np.any(np.abs(Sigma_p - np.diag(np.diag(Sigma_p))) > 1e-6)
    ll = gaussian_marginal_loglike(
        data, m0, M, cov_inv, mu_p, jnp.asarray(Sigma_p),
        include_logdet=include_logdet)
    ref = _ref_m2lnL(data, m0, M, cov_inv, mu_p, Sigma_p, include_logdet)
    assert float(ll) == pytest.approx(-0.5 * float(ref), rel=1e-9, abs=1e-9)


# --- per-bin / Taylor consumption of a stacked (n_bins, n_per, n_per) block ---


def _perbin_common():
    def bin0(p):
        return jnp.array([p[0] + 2.0 * p[1] + 3.0 * p[2]])

    def bin1(p):
        return jnp.array([p[0] - 1.5 * p[3] + 0.5 * p[4]])

    fid = jnp.array([1.0, 0.1, 0.2, 0.3, 0.4])
    common = dict(
        bin_theory_fns=[bin0, bin1],
        bin_data=[jnp.array([1.2]), jnp.array([0.8])],
        bin_cov_invs=[jnp.array([[2.0]]), jnp.array([[1.5]])],
        bin_lin_idx=[(1, 2), (3, 4)],           # 2 lin params/bin -> n_lin = 4
        prior_mean_fn=lambda t: jnp.array([0.1, -0.2, 0.3, -0.1]),
        log_prior_nl_fn=lambda t: 0.0,
        to_physical=lambda x: x,
        full_params_fn=lambda t: fid.at[jnp.array([0])].set(t),
    )
    return common, fid


def test_perbin_stacked_cov_matches_diagonal_widths():
    """prior_sigma_fn returning stacked diagonal blocks == returning widths."""
    from jaxptpolypol.marginal_likelihood import make_marginal_log_posterior_perbin

    common, fid = _perbin_common()
    widths = jnp.array([0.7, 1.3, 0.9, 1.1])    # bin0 (0.7,1.3), bin1 (0.9,1.1)
    diag = make_marginal_log_posterior_perbin(
        prior_sigma_fn=lambda t: widths, **common)
    stacked = jnp.stack([jnp.diag(widths[:2] ** 2), jnp.diag(widths[2:] ** 2)])
    cov = make_marginal_log_posterior_perbin(
        prior_sigma_fn=lambda t: stacked, **common)
    x = fid[:1]
    assert float(cov(x)) == pytest.approx(float(diag(x)), abs=1e-10)


def test_perbin_stacked_cov_wrong_nbins_raises():
    from jaxptpolypol.marginal_likelihood import make_marginal_log_posterior_perbin

    common, fid = _perbin_common()
    bad = make_marginal_log_posterior_perbin(
        prior_sigma_fn=lambda t: jnp.ones((3, 2, 2)), **common)  # 3 != 2 bins
    with pytest.raises(ValueError, match="prior_sigma_fn"):
        bad(fid[:1])


def test_taylor_stacked_cov_matches_diagonal_widths():
    """Same equivalence through the Taylor surrogate consumption path."""
    from jaxptpolypol.sampler import make_full_params_fn
    from jaxptpolypol.marginal_taylor import (
        build_taylor_templates, make_marginal_log_posterior_taylor)

    def theory(p):
        t0, t1, l0, l1 = p[0], p[1], p[2], p[3]
        m0 = jnp.array([t0**2 + t1, 3.0 * t1**2, t0 * t1])
        M = jnp.array([[1.0 + t0, 2.0], [t1, -1.0], [0.5, t0 + t1]])
        return m0 + M @ jnp.array([l0, l1])

    packed = jnp.array([0.3, -0.2, 9.9, -9.9])
    fpf = make_full_params_fn(packed, (0, 1))
    theta0 = jnp.array([0.3, -0.2])
    data = theory(jnp.array([0.3, -0.2, 0.4, -0.3]))
    cov_inv = jnp.linalg.inv(jnp.diag(jnp.array([0.9, 1.4, 0.6])))
    tt = build_taylor_templates(
        bin_theory_fns=[theory], bin_lin_idx=[(2, 3)], full_params_fn=fpf,
        theta0=theta0, order2_m0=True)
    widths = jnp.array([0.7, 1.3])
    common = dict(
        tt=tt, bin_data=[data], bin_cov_invs=[cov_inv],
        prior_mean_fn=lambda t: jnp.array([0.2, -0.1]),
        log_prior_nl_fn=lambda t: 0.0, to_physical=lambda x: x,
        full_params_fn=fpf, include_logdet=True)
    diag = make_marginal_log_posterior_taylor(
        prior_sigma_fn=lambda t: widths, **common)
    stacked = jnp.stack([jnp.diag(widths ** 2)])   # (1, 2, 2)
    cov = make_marginal_log_posterior_taylor(
        prior_sigma_fn=lambda t: stacked, **common)
    for dx in (jnp.zeros(2), jnp.array([0.15, -0.2])):
        x = theta0 + dx
        assert float(cov(x)) == pytest.approx(float(diag(x)), abs=1e-10)


# --- permanent two-representation invariance (rotation vs sigmap) -------------


@pytest.mark.parametrize("seed", [11, 202607, 999983])
def test_rotation_vs_sigmap_marginal_invariance(seed):
    """Exact linear-reparameterization invariance of the marginal likelihood.

    The two production representations of the DESI counterterm prior give the
    SAME ``gaussian_marginal_loglike``, including the ``ln det`` term:

      * **Rotation** (A): rotate the templates, ``M_rot = M @ T``, and keep the
        paper's *diagonal* prior widths ``sigma_paper`` with mean ``mu_paper``.
      * **Sigma_p** (B): keep the templates and pass the *full* rotated prior
        covariance ``Sigma = T diag(sigma_paper**2) T^T`` with mean
        ``mu_ours = T @ mu_paper``.

    ``T`` is the 11x11 identity with the counterterm ``(2:5, 2:5)`` block set to
    ``L = ctr_rotation_matrices((0.8155,))[0]`` -- an invertible upper-triangular
    map with ``det T = 1``. Because the marginal likelihood fully integrates the
    linear block, it is invariant under this invertible linear reparameterization
    of ``theta_lin`` -- the logdet included (the ``+2 ln det T`` from ``A`` and
    from ``Sigma_p^{-1}`` cancel). This synthetic test locks in the equivalence
    the end-to-end production Task-E check verified on the real P+B pipeline to
    ``max|dlogpost| = 4.2e-12`` (``example/mcmc/cache/task_e_equivalence.json``).
    """
    from jaxptpolypol.desi_priors import ctr_rotation_matrices

    rng = np.random.default_rng(seed)
    n_data, n_lin = 8, 11
    data = jnp.asarray(rng.standard_normal(n_data))
    m0 = jnp.asarray(rng.standard_normal(n_data))
    M = jnp.asarray(rng.standard_normal((n_data, n_lin)))
    araw = rng.standard_normal((n_data, n_data))
    cov_inv = jnp.asarray(np.linalg.inv(araw @ araw.T + n_data * np.eye(n_data)))

    mu_paper = np.zeros(n_lin)
    mu_paper[2:5] = [0.0, 30.0, 0.0]                 # ctr slots (c0, c2, c4)
    mu_paper = jnp.asarray(mu_paper)
    sigma_paper = jnp.asarray(0.5 + rng.random(n_lin))   # strictly positive widths

    L = ctr_rotation_matrices(jnp.asarray([0.8155]))[0]  # (3, 3)
    T = np.eye(n_lin)
    T[2:5, 2:5] = np.asarray(L)                          # embed L at the ctr block
    T = jnp.asarray(T)

    M_rot = M @ T                                        # representation A
    Sigma_full = T @ jnp.diag(sigma_paper ** 2) @ T.T    # representation B
    mu_ours = T @ mu_paper

    for include_logdet in (True, False):
        ll_A = gaussian_marginal_loglike(
            data, m0, M_rot, cov_inv, mu_paper, sigma_paper,
            include_logdet=include_logdet)
        ll_B = gaussian_marginal_loglike(
            data, m0, M, cov_inv, mu_ours, Sigma_full,
            include_logdet=include_logdet)
        assert float(ll_A) == pytest.approx(float(ll_B), abs=1e-9)
