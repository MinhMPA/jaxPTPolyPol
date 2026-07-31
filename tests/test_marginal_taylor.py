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


def _toy_surrogate_common(theory, fpf):
    """The test-(a) surrogate ingredients: data, cov_inv, and constant priors."""
    mu_p = jnp.array([0.4, -0.3])
    sigma_p = jnp.array([0.7, 1.3])
    data = theory(jnp.array([0.3, -0.2, 0.4, -0.3]))
    cov_inv = jnp.linalg.inv(jnp.diag(jnp.array([0.9, 1.4, 0.6])))
    return dict(
        bin_data=[data], bin_cov_invs=[cov_inv],
        prior_mean_fn=lambda _t: mu_p, prior_sigma_fn=lambda _t: sigma_p,
        log_prior_nl_fn=lambda _t: 0.0, to_physical=lambda x: x,
        full_params_fn=fpf, include_logdet=True)


def test_save_load_roundtrip_bitwise(tmp_path):
    """save -> load reproduces every tensor bitwise and both scalars, and a
    surrogate built from the loaded templates matches the original exactly."""
    from jaxptpolypol.marginal_taylor import (
        build_taylor_templates, save_taylor_templates, load_taylor_templates,
        make_marginal_log_posterior_taylor)

    theory, fpf, packed = _toy_full_setup()
    theta0 = jnp.array([0.3, -0.2])
    tt = build_taylor_templates(
        bin_theory_fns=[theory], bin_lin_idx=[(2, 3)], full_params_fn=fpf,
        theta0=theta0, order2_m0=True)

    path = tmp_path / "tt.npz"
    meta = dict(n_bins=1, n_k=8, tag="toy")
    save_taylor_templates(tt, path, meta=meta)
    loaded = load_taylor_templates(path, expect_meta=meta)

    # Both scalars, bitwise.
    assert loaded.order2_m0 == tt.order2_m0
    assert np.array_equal(np.asarray(loaded.theta0), np.asarray(tt.theta0))
    # Every per-bin tensor, bitwise (atol=0, not allclose).
    for b in range(len(tt.bin_m00)):
        for name in ("bin_m00", "bin_J", "bin_H", "bin_M0", "bin_dM"):
            lo = getattr(loaded, name)[b]
            og = getattr(tt, name)[b]
            assert np.array_equal(np.asarray(lo), np.asarray(og)), name

    # The loaded meta is carried in build_diagnostics.
    assert loaded.build_diagnostics["meta"] == meta

    # Surrogate from the loaded templates == surrogate from the original.
    common = _toy_surrogate_common(theory, fpf)
    sur_orig = make_marginal_log_posterior_taylor(tt, **common)
    sur_load = make_marginal_log_posterior_taylor(loaded, **common)
    for dx in (jnp.zeros(2), jnp.array([0.15, -0.2])):
        x = theta0 + dx
        np.testing.assert_allclose(
            float(sur_load(x)), float(sur_orig(x)), rtol=1e-15)


def test_load_meta_mismatch_raises_all_keys(tmp_path):
    """expect_meta mismatch must name EVERY offending key (value diff + missing),
    and must NOT name matching keys."""
    from jaxptpolypol.marginal_taylor import (
        build_taylor_templates, save_taylor_templates, load_taylor_templates)

    theory, fpf, packed = _toy_full_setup()
    theta0 = jnp.array([0.3, -0.2])
    tt = build_taylor_templates(
        bin_theory_fns=[theory], bin_lin_idx=[(2, 3)], full_params_fn=fpf,
        theta0=theta0)

    path = tmp_path / "tt.npz"
    save_taylor_templates(tt, path, meta=dict(n_bins=1, n_k=8))
    with pytest.raises(ValueError) as exc:
        load_taylor_templates(path, expect_meta=dict(n_bins=2, n_k=8, num_mu=65))
    msg = str(exc.value)
    assert "n_bins" in msg          # value differs (1 vs 2)
    assert "num_mu" in msg          # present only in expect_meta
    assert "n_k" not in msg         # matches (8 == 8): must not be listed


def test_load_meta_mismatch_new_keys_raises(tmp_path):
    """A VALUE mismatch on a new theory-config identifier (present in BOTH metas)
    still raises -- backward compatibility tolerates only its ABSENCE from a
    pre-dating cache, never a disagreement."""
    from jaxptpolypol.marginal_taylor import (
        build_taylor_templates, save_taylor_templates, load_taylor_templates)

    theory, fpf, packed = _toy_full_setup()
    theta0 = jnp.array([0.3, -0.2])
    tt = build_taylor_templates(
        bin_theory_fns=[theory], bin_lin_idx=[(2, 3)], full_params_fn=fpf,
        theta0=theta0)

    path = tmp_path / "tt.npz"
    # Stored carries the new keys, so absence-tolerance does not apply.
    save_taylor_templates(tt, path, meta=dict(
        n_bins=1, theory_config_hash="aaa", z_bins="(0.7,)"))
    with pytest.raises(ValueError) as exc:
        load_taylor_templates(path, expect_meta=dict(
            n_bins=1, theory_config_hash="bbb", z_bins="(0.7,)"))
    msg = str(exc.value)
    assert "theory_config_hash" in msg    # value differs (aaa vs bbb)
    assert "z_bins" not in msg            # matches: must not be listed


def test_load_missing_new_keys_warns_not_raises(tmp_path):
    """A cache predating the theory-config keys (absent from its stored meta)
    loads with a UserWarning instead of a stale-template error; a genuine
    mismatch on a legacy key alongside the missing new keys still raises."""
    from jaxptpolypol.marginal_taylor import (
        build_taylor_templates, save_taylor_templates, load_taylor_templates)

    theory, fpf, packed = _toy_full_setup()
    theta0 = jnp.array([0.3, -0.2])
    tt = build_taylor_templates(
        bin_theory_fns=[theory], bin_lin_idx=[(2, 3)], full_params_fn=fpf,
        theta0=theta0)

    path = tmp_path / "tt.npz"
    # Legacy stored meta -- no theory_config_hash / z_bins / knl_bins yet.
    save_taylor_templates(tt, path, meta=dict(n_bins=1, n_k=8))

    # Expect the new keys; their absence from the old cache warns, still loads.
    expect = dict(n_bins=1, n_k=8, theory_config_hash="abc",
                  z_bins="(0.7, 0.9)", knl_bins="(0.5,)")
    with pytest.warns(UserWarning, match="theory_config_hash"):
        loaded = load_taylor_templates(path, expect_meta=expect)
    assert np.array_equal(np.asarray(loaded.theta0), np.asarray(tt.theta0))

    # A real mismatch on a LEGACY key still raises, even with new keys missing.
    with pytest.raises(ValueError, match="n_bins"):
        load_taylor_templates(path, expect_meta=dict(
            n_bins=2, theory_config_hash="abc"))


def test_save_rejects_non_flat_meta(tmp_path):
    """A nested meta value is not a config identifier -> TypeError at save."""
    from jaxptpolypol.marginal_taylor import (
        build_taylor_templates, save_taylor_templates)

    theory, fpf, packed = _toy_full_setup()
    theta0 = jnp.array([0.3, -0.2])
    tt = build_taylor_templates(
        bin_theory_fns=[theory], bin_lin_idx=[(2, 3)], full_params_fn=fpf,
        theta0=theta0)
    with pytest.raises(TypeError):
        save_taylor_templates(tt, tmp_path / "tt.npz", meta=dict(cfg={"nested": 1}))


def test_roundtrip_order1_no_H(tmp_path):
    """order2_m0=False: H entries survive as None through save/load and the
    loaded surrogate still evaluates."""
    from jaxptpolypol.marginal_taylor import (
        build_taylor_templates, save_taylor_templates, load_taylor_templates,
        make_marginal_log_posterior_taylor)

    theory, fpf, packed = _toy_full_setup()
    theta0 = jnp.array([0.3, -0.2])
    tt = build_taylor_templates(
        bin_theory_fns=[theory], bin_lin_idx=[(2, 3)], full_params_fn=fpf,
        theta0=theta0, order2_m0=False)

    path = tmp_path / "tt.npz"
    save_taylor_templates(tt, path, meta=dict(n_bins=1))
    loaded = load_taylor_templates(path)

    assert loaded.order2_m0 is False
    assert len(loaded.bin_H) == len(tt.bin_H)
    for h in loaded.bin_H:
        assert h is None

    common = _toy_surrogate_common(theory, fpf)
    sur = make_marginal_log_posterior_taylor(loaded, **common)
    assert np.isfinite(float(sur(theta0)))


# --- Importance reweighting (Task 5) ------------------------------------------
#
# The surrogate posterior is fast but only asymptotically exact. These tests
# pin the post-hoc importance-reweighting utility that restores exactness on a
# surrogate chain, plus the ESS / max-weight diagnostics that reveal when the
# surrogate under-covers the tails (in which case the reweighted answer must be
# reported as untrustworthy, not silently trusted). All toys are analytic 1-D/
# N-D Gaussians (no emulator, no jax): the exact and surrogate "posteriors" are
# closed-form Gaussian log-pdfs, so the reweighting algebra is checked against
# values known to the float64 floor and against a numerically-computed ESS.

_TRAPZ = getattr(np, "trapezoid", np.trapz)   # np>=2 renamed trapz -> trapezoid


def _normal_logpdf(x, mu, var):
    """Log N(x; mu, var) -- SECOND ARG IS THE VARIANCE (so N(0.3, 1.2**2) below
    is variance 1.44). Vectorized over x."""
    return -0.5 * np.log(2.0 * np.pi * var) - 0.5 * (x - mu) ** 2 / var


def _analytic_ess_frac(mu_p, var_p, mu_q, var_q):
    """Limiting ESS/n for weights w = p/q with x drawn from q = N(mu_q, var_q),
    target p = N(mu_p, var_p).

    E_q[w] = int q * (p/q) = 1, so ESS/n -> (E_q[w])**2 / E_q[w**2]
    = 1 / int p(x)**2 / q(x) dx.  The integral is done by fine-grid quadrature
    (deterministic, independent of the Monte-Carlo draw the test asserts on).
    """
    span = 12.0 * np.sqrt(max(var_p, var_q))
    xg = np.linspace(min(mu_p, mu_q) - span, max(mu_p, mu_q) + span, 2_000_001)
    integrand = np.exp(2 * _normal_logpdf(xg, mu_p, var_p)
                       - _normal_logpdf(xg, mu_q, var_q))
    e_w2 = _TRAPZ(integrand, xg)
    return 1.0 / e_w2


def test_is_reweight_corrects_offset_gaussian():
    """Reweighting a surrogate draw to an offset, wider exact target recovers
    the exact mean/std, and the ESS diagnostic matches its analytic value.

    exact    p = N(0.3, 1.2**2 = 1.44)
    surrogate q = N(0, 1)              -- 200k samples drawn FROM q.

    The naive (unweighted) surrogate sample has mean ~0 and std ~1; reweighting
    by w = p/q must move them to 0.3 and 1.2. ESS/n is asserted against the
    numerically-integrated expectation (not a hardcoded band), and max_weight
    must be tiny (q covers p well, so no single sample dominates).
    """
    from jaxptpolypol.marginal_taylor import (
        importance_reweight, reweighted_moments)

    rng = np.random.default_rng(12345)
    n = 200_000
    samples = rng.standard_normal(n).reshape(-1, 1)      # (n, 1), drawn from q

    log_p_exact = lambda x: _normal_logpdf(x[0], 0.3, 1.44)
    log_p_surrogate = lambda x: _normal_logpdf(x[0], 0.0, 1.0)

    res = importance_reweight(samples, log_p_exact, log_p_surrogate)

    # Keys and shapes.
    assert set(res) == {"weights", "log_w_raw", "ess", "ess_frac",
                        "max_weight", "idx"}
    assert res["weights"].shape == (n,)
    assert res["log_w_raw"].shape == (n,)
    assert res["idx"].shape == (n,)
    np.testing.assert_allclose(res["weights"].sum(), 1.0, rtol=1e-12)

    mean, std = reweighted_moments(samples, res["weights"], idx=res["idx"])
    assert abs(mean[0] - 0.3) < 0.02, mean[0]
    assert abs(std[0] - 1.2) < 0.02, std[0]

    # ESS/n within +-0.1 of the numerically-computed expectation.
    ess_frac_analytic = _analytic_ess_frac(0.3, 1.44, 0.0, 1.0)   # ~0.765
    assert abs(res["ess_frac"] - ess_frac_analytic) < 0.1, (
        res["ess_frac"], ess_frac_analytic)
    assert 0.5 < res["ess_frac"] < 1.0            # well-covered regime
    assert res["max_weight"] < 0.01               # no single sample dominates


def test_is_reweight_subsample_path():
    """subsample=k evaluates the exact/surrogate fns on only k uniformly-drawn
    (without replacement) indices -- the laptop path where each exact eval is
    ~5 s. Returns k weights over sorted, unique, in-range indices, and its
    moments agree with the full-set answer at a looser tolerance."""
    from jaxptpolypol.marginal_taylor import (
        importance_reweight, reweighted_moments)

    rng = np.random.default_rng(2024)
    n = 200_000
    samples = rng.standard_normal(n).reshape(-1, 1)

    log_p_exact = lambda x: _normal_logpdf(x[0], 0.3, 1.44)
    log_p_surrogate = lambda x: _normal_logpdf(x[0], 0.0, 1.0)

    full = importance_reweight(samples, log_p_exact, log_p_surrogate)
    sub = importance_reweight(samples, log_p_exact, log_p_surrogate,
                              subsample=10_000, seed=0)

    assert sub["weights"].shape == (10_000,)
    assert sub["idx"].shape == (10_000,)
    assert len(np.unique(sub["idx"])) == 10_000          # unique
    assert np.all(np.diff(sub["idx"]) > 0)               # sorted (strictly)
    assert sub["idx"].min() >= 0 and sub["idx"].max() < n  # in range
    np.testing.assert_allclose(sub["weights"].sum(), 1.0, rtol=1e-12)

    m_full, s_full = reweighted_moments(samples, full["weights"],
                                        idx=full["idx"])
    m_sub, s_sub = reweighted_moments(samples, sub["weights"], idx=sub["idx"])
    assert abs(m_sub[0] - m_full[0]) < 0.05, (m_sub[0], m_full[0])
    assert abs(s_sub[0] - s_full[0]) < 0.05, (s_sub[0], s_full[0])


def test_is_reweight_flags_undercoverage():
    """A surrogate much NARROWER than the exact target under-covers the tails;
    the diagnostics must fire so the reweighted answer is flagged untrustworthy.

    exact    p = N(0.3, 1.44)
    narrow   q = N(0, 0.4**2 = 0.16)   -- std 0.4, much narrower than std 1.2.

    Because var_p (1.44) > 2*var_q (0.32) the importance-weight variance
    diverges: ESS/n -> 0 and the weight mass concentrates on the few samples
    that stray into the tail.

    Threshold note (Gumbel ceiling; the brief originally said max_weight > 0.2):
    at n=200_000 a *single* normalized weight cannot approach 1 for a smooth 1-D
    Gaussian mismatch -- there is always a shell of comparably-extreme samples
    sharing the peak weight. The extreme log-weight follows a Gumbel law, so the
    top normalized weight ~ exp(lw_max - log n) sits at an O(0.1) *ceiling*:
    across a 12-seed sweep of this narrow proposal it spans ~0.013-0.12, always
    below ~0.2 and never near 1. A lower bound like `> 0.01` is therefore thin
    and seed-fragile (the 12-seed minimum is 0.0127), so the robust, seed-
    independent assertions are the ess_frac tripwire and the Gumbel ceiling
    `max_weight < 0.2`. max_weight's *elevation* over the well-covered baseline
    (~2e-4 in test 1) is still pinned by the `> 20x` ratio check. Both are
    checked against a healthy N(0,1) surrogate reweighted to the same target.
    """
    from jaxptpolypol.marginal_taylor import importance_reweight

    n = 200_000
    log_p_exact = lambda x: _normal_logpdf(x[0], 0.3, 1.44)

    # Healthy proposal: N(0, 1) -- covers the target.
    rng_ok = np.random.default_rng(12345)
    s_ok = rng_ok.standard_normal(n).reshape(-1, 1)
    ok = importance_reweight(
        s_ok, log_p_exact, lambda x: _normal_logpdf(x[0], 0.0, 1.0))

    # Under-covering proposal: N(0, 0.4**2) -- far too narrow.
    rng_bad = np.random.default_rng(7)
    s_bad = (rng_bad.standard_normal(n) * 0.4).reshape(-1, 1)
    bad = importance_reweight(
        s_bad, log_p_exact, lambda x: _normal_logpdf(x[0], 0.0, 0.16))

    # Diagnostics fire on the narrow proposal.
    assert bad["ess_frac"] < 0.05, bad["ess_frac"]
    # Gumbel ceiling: for a smooth Gaussian mismatch the top normalized weight
    # ~ exp(lw_max - log n) sits at an O(0.1) ceiling and cannot reach 1 (a shell
    # of comparably-extreme draws shares the peak) -- a robust, seed-independent
    # bound replacing the thin `> 0.01` lower (12-seed min 0.0127); elevation
    # over the healthy baseline is pinned by the `> 20x` ratio check below.
    assert bad["max_weight"] < 0.2, bad["max_weight"]
    # ... and clearly separate the bad proposal from the healthy one.
    assert bad["ess_frac"] < ok["ess_frac"] / 5.0
    assert bad["max_weight"] > 20.0 * ok["max_weight"]


def test_is_reweight_accepts_chain_axis():
    """A (chains, n, d) sample array is reshaped to (chains*n, d) and run;
    idx indexes into the flattened stack."""
    from jaxptpolypol.marginal_taylor import (
        importance_reweight, reweighted_moments)

    rng = np.random.default_rng(3)
    samples = rng.standard_normal((2, 500, 3))            # (chains, n, d)

    mu_p = np.array([0.2, -0.1, 0.4])
    log_p_exact = lambda x: float(np.sum(_normal_logpdf(x, mu_p, 1.5)))
    log_p_surrogate = lambda x: float(np.sum(_normal_logpdf(x, 0.0, 1.0)))

    res = importance_reweight(samples, log_p_exact, log_p_surrogate)

    assert res["weights"].shape == (1000,)
    assert res["idx"].shape == (1000,)
    assert res["idx"].min() >= 0 and res["idx"].max() < 1000
    np.testing.assert_allclose(res["weights"].sum(), 1.0, rtol=1e-12)
    assert np.isfinite(res["ess"]) and 0.0 < res["ess_frac"] <= 1.0

    mean, std = reweighted_moments(samples, res["weights"], idx=res["idx"])
    assert mean.shape == (3,) and std.shape == (3,)


def test_reweighted_moments_warns_on_degenerate_weights():
    """Near-degenerate weights (one sample carries the mass -> ess_frac << 0.01)
    make reweighted_moments warn, naming ess_frac and advising the exact-target
    path; a healthy uniform weight set (ess_frac == 1) does not warn."""
    import warnings as _warnings

    from jaxptpolypol.marginal_taylor import reweighted_moments

    m = 300
    samples = np.random.default_rng(0).standard_normal((m, 2))
    w = np.full(m, 1e-12)
    w[0] = 1.0
    w = w / w.sum()                     # ess ~ 1, ess_frac ~ 1/300 << 0.01
    with pytest.warns(UserWarning, match="ess_frac"):
        mean, std = reweighted_moments(samples, w)
    assert mean.shape == (2,) and std.shape == (2,)

    # Healthy uniform weights (ess_frac == 1.0): our degenerate warning must NOT
    # fire (record all warnings, assert none mention ess_frac).
    w_ok = np.full(m, 1.0 / m)
    with _warnings.catch_warnings(record=True) as rec:
        _warnings.simplefilter("always")
        reweighted_moments(samples, w_ok)
    assert not any("ess_frac" in str(r.message) for r in rec)
