import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest


def _toy_full_setup():
    """Full vector: [t0, t1 | l0, l1]  (2 theta_NL, one 'bin' with 2 lin params).

    theory(p) = m0(t) + M(t) @ l  with
      m0(t) = [t0**2 + t1, 3*t1**2, t0*t1]          (quadratic -> H exact)
      M(t)  = [[1+t0, 2], [t1, -1], [0.5, t0+t1]]   (linear -> dM exact)
    """
    from jaxptpolypol.sampler import make_full_params_fn

    def theory(p):
        t0, t1, l0, l1 = p[0], p[1], p[2], p[3]
        m0 = jnp.array([t0**2 + t1, 3.0 * t1**2, t0 * t1])
        M = jnp.array([[1.0 + t0, 2.0], [t1, -1.0], [0.5, t0 + t1]])
        return m0 + M @ jnp.array([l0, l1])

    packed = jnp.array([0.3, -0.2, 9.9, -9.9])   # junk lin values must not matter
    fpf = make_full_params_fn(packed, (0, 1))
    return theory, fpf, packed


def test_builder_exact_on_representable_toy():
    from jaxptpolypol.marginal_taylor import build_taylor_templates

    theory, fpf, packed = _toy_full_setup()
    theta0 = jnp.array([0.3, -0.2])
    tt = build_taylor_templates(
        bin_theory_fns=[theory], bin_lin_idx=[(2, 3)], full_params_fn=fpf,
        theta0=theta0, order2_m0=True, chunk_J=1, chunk_H=1)

    t0, t1 = float(theta0[0]), float(theta0[1])
    np.testing.assert_allclose(np.asarray(tt.bin_m00[0]),
                               [t0**2 + t1, 3 * t1**2, t0 * t1], atol=1e-14)
    np.testing.assert_allclose(np.asarray(tt.bin_M0[0]),
                               [[1 + t0, 2], [t1, -1], [0.5, t0 + t1]], atol=1e-14)
    np.testing.assert_allclose(np.asarray(tt.bin_J[0]),
                               [[2 * t0, 1], [0, 6 * t1], [t1, t0]], atol=1e-12)
    H = np.zeros((3, 2, 2)); H[0, 0, 0] = 2.0; H[1, 1, 1] = 6.0
    H[2, 0, 1] = H[2, 1, 0] = 1.0
    np.testing.assert_allclose(np.asarray(tt.bin_H[0]), H, atol=1e-12)
    dM = np.zeros((3, 2, 2))
    dM[0, 0, 0] = 1.0; dM[1, 0, 1] = 1.0; dM[2, 1, 0] = 1.0; dM[2, 1, 1] = 1.0
    np.testing.assert_allclose(np.asarray(tt.bin_dM[0]), dM, atol=1e-12)


def test_builder_chunking_invariance():
    """Different chunk sizes must give bit-comparable tensors."""
    from jaxptpolypol.marginal_taylor import build_taylor_templates
    theory, fpf, packed = _toy_full_setup()
    theta0 = jnp.array([0.3, -0.2])
    a = build_taylor_templates(bin_theory_fns=[theory], bin_lin_idx=[(2, 3)],
                               full_params_fn=fpf, theta0=theta0, chunk_J=1, chunk_H=1)
    b = build_taylor_templates(bin_theory_fns=[theory], bin_lin_idx=[(2, 3)],
                               full_params_fn=fpf, theta0=theta0, chunk_J=2, chunk_H=2)
    for x, y in ((a.bin_J[0], b.bin_J[0]), (a.bin_H[0], b.bin_H[0]),
                 (a.bin_dM[0], b.bin_dM[0])):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=1e-13)


def test_builder_nondivisor_chunk_pasting():
    """Partial final chunk (chunk does not divide d) must paste identically.

    Production runs d=26 with chunk_J=4 (final chunk of 2); this pins that
    path with a d=3 toy and chunk 2 vs full-width chunk 3.
    """
    from jaxptpolypol.marginal_taylor import build_taylor_templates
    from jaxptpolypol.sampler import make_full_params_fn

    def theory(p):
        t0, t1, t2, l0 = p[0], p[1], p[2], p[3]
        m0 = jnp.array([t0 * t1 + t2**2, t0**2 - 2.0 * t2, t1 * t2])
        M = jnp.array([[1.0 + t0 + t2], [t1 - t2], [0.5 * t0]])
        return m0 + M @ jnp.array([l0])

    packed = jnp.array([0.4, -0.1, 0.25, 7.7])
    fpf = make_full_params_fn(packed, (0, 1, 2))
    theta0 = jnp.array([0.4, -0.1, 0.25])

    a = build_taylor_templates(bin_theory_fns=[theory], bin_lin_idx=[(3,)],
                               full_params_fn=fpf, theta0=theta0,
                               chunk_J=2, chunk_H=2)     # non-divisor of d=3
    b = build_taylor_templates(bin_theory_fns=[theory], bin_lin_idx=[(3,)],
                               full_params_fn=fpf, theta0=theta0,
                               chunk_J=3, chunk_H=3)     # full width
    for x, y in ((a.bin_J[0], b.bin_J[0]), (a.bin_H[0], b.bin_H[0]),
                 (a.bin_dM[0], b.bin_dM[0])):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=1e-13)
    # and against hand-computed J for direct (not merely mutual) correctness
    t0, t1, t2 = 0.4, -0.1, 0.25
    J = [[t1, t0, 2 * t2], [2 * t0, 0, -2.0], [0, t2, t1]]
    np.testing.assert_allclose(np.asarray(a.bin_J[0]), J, atol=1e-12)


def test_surrogate_equals_perbin_on_representable_toy():
    """The surrogate reproduces the exact per-bin posterior on the toy.

    The toy's ``m0`` is quadratic and ``M`` is linear in theta_NL, so the carried
    Taylor expansion (H for m0, dM for M) reconstructs ``(m0, M)`` *exactly* at
    every theta_NL. The surrogate marginal log-posterior must therefore equal the
    exact :func:`make_marginal_log_posterior_perbin` value -- at the expansion
    centre AND at displaced points -- to the float64 floor, logdet included.
    """
    from jaxptpolypol.marginal_likelihood import make_marginal_log_posterior_perbin
    from jaxptpolypol.marginal_taylor import (
        build_taylor_templates, make_marginal_log_posterior_taylor)

    theory, fpf, packed = _toy_full_setup()
    theta0 = jnp.array([0.3, -0.2])
    mu_p = jnp.array([0.4, -0.3])
    sigma_p = jnp.array([0.7, 1.3])
    # Data = theory at theta_NL = theta0 with the lin values the priors center on.
    data = theory(jnp.array([0.3, -0.2, 0.4, -0.3]))
    cov_inv = jnp.linalg.inv(jnp.diag(jnp.array([0.9, 1.4, 0.6])))  # non-identity

    prior_mean_fn = lambda _t: mu_p
    prior_sigma_fn = lambda _t: sigma_p

    per = make_marginal_log_posterior_perbin(
        bin_theory_fns=[theory], bin_data=[data], bin_cov_invs=[cov_inv],
        bin_lin_idx=[(2, 3)],
        prior_mean_fn=prior_mean_fn, prior_sigma_fn=prior_sigma_fn,
        log_prior_nl_fn=lambda _t: 0.0, to_physical=lambda x: x,
        full_params_fn=fpf, include_logdet=True)

    tt = build_taylor_templates(
        bin_theory_fns=[theory], bin_lin_idx=[(2, 3)], full_params_fn=fpf,
        theta0=theta0, order2_m0=True)
    sur = make_marginal_log_posterior_taylor(
        tt, bin_data=[data], bin_cov_invs=[cov_inv],
        prior_mean_fn=prior_mean_fn, prior_sigma_fn=prior_sigma_fn,
        log_prior_nl_fn=lambda _t: 0.0, to_physical=lambda x: x,
        full_params_fn=fpf, include_logdet=True)

    for dx in (jnp.zeros(2), jnp.array([0.15, -0.2]), jnp.array([-0.3, 0.1])):
        x = theta0 + dx
        np.testing.assert_allclose(float(sur(x)), float(per(x)), rtol=1e-12)
    # The displaced points must actually move the posterior (else the check is vacuous).
    assert abs(float(sur(theta0 + jnp.array([0.15, -0.2]))) - float(sur(theta0))) > 1e-6


def test_surrogate_with_extra_term():
    """A cosmology-only linear extra (BAO stand-in) block, added once and exactly.

    The extra term carries no theta_lin dependence and is evaluated exactly in
    both posteriors, so the surrogate still equals the per-bin form to the
    float64 floor at the centre and at displaced points.

    Contract-binding: ``extra_theory_fn`` receives the FULL packed vector via
    ``full_params_fn`` in BOTH builders (drop-in symmetry). The extra block here
    deliberately reads ``p[3]`` -- a non-varied slot that exists only in the
    full vector (theta_NL is length 2) -- so an implementation that wrongly fed
    theta_NL to ``extra_theory_fn`` would fail on shape, not pass by luck.
    """
    from jaxptpolypol.marginal_likelihood import make_marginal_log_posterior_perbin
    from jaxptpolypol.marginal_taylor import (
        build_taylor_templates, make_marginal_log_posterior_taylor)

    theory, fpf, packed = _toy_full_setup()
    theta0 = jnp.array([0.3, -0.2])
    mu_p = jnp.array([0.4, -0.3])
    sigma_p = jnp.array([0.7, 1.3])
    data = theory(jnp.array([0.3, -0.2, 0.4, -0.3]))
    cov_inv = jnp.linalg.inv(jnp.diag(jnp.array([0.9, 1.4, 0.6])))

    rng = np.random.default_rng(0)
    A_extra = jnp.asarray(rng.normal(size=(4, 3)))

    def extra_theory_fn(p):
        # Full-vector slots [0, 1, 3]: theta_NL plus a slot full_params_fn keeps
        # at its packed value. theta_NL (length 2) has no slot 3 -> a wrong
        # argument convention gives an empty p[3:4] and a (4,3)@(2,) shape error.
        return A_extra @ jnp.concatenate([p[:2], p[3:4]])

    extra_data = extra_theory_fn(fpf(theta0)) + 0.1   # non-zero residual at theta0
    extra_cov_inv = jnp.diag(jnp.array([1.0, 0.5, 2.0, 1.5]))

    prior_mean_fn = lambda _t: mu_p
    prior_sigma_fn = lambda _t: sigma_p

    per = make_marginal_log_posterior_perbin(
        bin_theory_fns=[theory], bin_data=[data], bin_cov_invs=[cov_inv],
        bin_lin_idx=[(2, 3)],
        extra_theory_fn=extra_theory_fn, extra_data=extra_data,
        extra_cov_inv=extra_cov_inv,
        prior_mean_fn=prior_mean_fn, prior_sigma_fn=prior_sigma_fn,
        log_prior_nl_fn=lambda _t: 0.0, to_physical=lambda x: x,
        full_params_fn=fpf, include_logdet=True)

    tt = build_taylor_templates(
        bin_theory_fns=[theory], bin_lin_idx=[(2, 3)], full_params_fn=fpf,
        theta0=theta0, order2_m0=True)
    sur = make_marginal_log_posterior_taylor(
        tt, bin_data=[data], bin_cov_invs=[cov_inv],
        extra_theory_fn=extra_theory_fn, extra_data=extra_data,
        extra_cov_inv=extra_cov_inv,
        prior_mean_fn=prior_mean_fn, prior_sigma_fn=prior_sigma_fn,
        log_prior_nl_fn=lambda _t: 0.0, to_physical=lambda x: x,
        full_params_fn=fpf, include_logdet=True)

    for dx in (jnp.zeros(2), jnp.array([0.15, -0.2]), jnp.array([-0.3, 0.1])):
        x = theta0 + dx
        np.testing.assert_allclose(float(sur(x)), float(per(x)), rtol=1e-12)
    # The extra term must actually contribute (otherwise the check above is vacuous).
    r0 = np.asarray(extra_data - extra_theory_fn(fpf(theta0)))
    assert abs(float(-0.5 * r0 @ np.asarray(extra_cov_inv) @ r0)) > 1e-6


def test_surrogate_prior_guard_fires():
    """The prior-slice tiling guard must fire, naming the offending function.

    Mirrors tests/test_marginal_perbin.py::
    test_perbin_rejects_prior_vector_that_does_not_tile_bins: a prior function
    returning the wrong number of entries is a mis-configured split, and must
    raise rather than silently mis-slice.
    """
    from jaxptpolypol.marginal_taylor import (
        build_taylor_templates, make_marginal_log_posterior_taylor)

    theory, fpf, packed = _toy_full_setup()
    theta0 = jnp.array([0.3, -0.2])
    data = theory(jnp.array([0.3, -0.2, 0.4, -0.3]))
    cov_inv = jnp.linalg.inv(jnp.diag(jnp.array([0.9, 1.4, 0.6])))
    tt = build_taylor_templates(
        bin_theory_fns=[theory], bin_lin_idx=[(2, 3)], full_params_fn=fpf,
        theta0=theta0, order2_m0=True)                 # 2 lin params -> n_lin = 2
    common = dict(
        bin_data=[data], bin_cov_invs=[cov_inv],
        log_prior_nl_fn=lambda _t: 0.0, to_physical=lambda x: x,
        full_params_fn=fpf, include_logdet=True)

    # Correct width (2) works.
    ok = make_marginal_log_posterior_taylor(
        tt, prior_mean_fn=lambda _t: jnp.zeros(2),
        prior_sigma_fn=lambda _t: jnp.ones(2), **common)
    assert np.isfinite(float(ok(theta0)))

    # Wrong mean width (3) must raise, naming the offending function.
    bad = make_marginal_log_posterior_taylor(
        tt, prior_mean_fn=lambda _t: jnp.zeros(3),
        prior_sigma_fn=lambda _t: jnp.ones(2), **common)
    with pytest.raises(ValueError, match="prior_mean_fn"):
        bad(theta0)

    # Wrong sigma width (3) must raise, naming the offending function.
    bad_sigma = make_marginal_log_posterior_taylor(
        tt, prior_mean_fn=lambda _t: jnp.zeros(2),
        prior_sigma_fn=lambda _t: jnp.ones(3), **common)
    with pytest.raises(ValueError, match="prior_sigma_fn"):
        bad_sigma(theta0)
