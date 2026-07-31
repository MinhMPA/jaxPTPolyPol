"""Toy-scale validation of the Tier-3 c1-SAMPLED machinery.

Exercises, entirely at toy scale (no emulator, < 1 s), the three pillars the
production sampled-c1 vs marginalized-c1 comparison rests on:

(a) ``split_marginal_indices`` with the reduced lin-keys (``LIN_SURVEY_KEYS``
    minus c1) yields the expected index bookkeeping in PRODUCTION survey-key
    order -- c1 leaves the marginalized block (10/bin) and joins theta_NL at
    ``nl_b1_pos[b] + 3`` (order b1, b2, bG2, c1), n_nl gains one c1 per bin, and
    the counterterm trio stays at slots 2,3,4.

(b) ``make_desi_prior_fns`` with the 10-key variant returns cov-mode
    ``(n_bins, 10, 10)`` blocks with NO c1 row, and the sampled c1 prior term
    evaluates with the same ``A_AP*A_amp`` R-division the marginalized c1 row
    would have carried. (The base R-division / block oracles live in
    ``tests/test_desi_priors.py``; here we pin the PRODUCTION-order wiring.)

(c) On a SYNTHETIC theory with an explicit c1^2 term, the ORDER-2 Taylor
    surrogate reproduces the exact marginal posterior in the c1 direction to the
    float64 floor (m0 is quadratic in c1 -> H captures it exactly), while an
    ORDER-1 surrogate misses the c1^2 by the analytically-predicted amount
    ``0.5 * d^2 m0/dc1^2 * dc1^2`` -- the non-vacuity tripwire proving the
    order-2 exactness claim is doing real work.

This is the "runs NOW" deliverable of the Tier-3 part-1 task; the production
7-bin build + chains are part 2.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from jaxptpolypol.marginal_likelihood import (
    LIN_SURVEY_KEYS, split_marginal_indices,
)

_C1_KEY = ("bk", "ctr", "c1")
LIN_KEYS_NO_C1 = tuple(k for k in LIN_SURVEY_KEYS if k != _C1_KEY)
FIXED_SURVEY_KEYS = {("shared", "k_nl", None), ("shared", "ndens", None)}


def _production_survey_keys():
    """Production-order joint survey keys (b1/b2/bG2 first, c1 late in bk.ctr)."""
    from jaxptpolypol.params import FullShapeSurveyParams
    s = FullShapeSurveyParams(
        shared={"bias": {"b1": 1.2, "b2": -0.5, "bG2": -0.1, "bGamma3": 0.2},
                "stoch": {"P_shot": 1.0}, "k_nl": 0.52, "ndens": 3e-4},
        pk={"ctr": {"c0": 10.0, "c2": 10.0, "c4": 1.0, "cfog": 0.52 ** -4},
            "stoch": {"a0": 0.0, "a2": 0.0}},
        bk={"ctr": {"c1": 0.0}, "stoch": {"B_shot": 1.0, "A_shot": 1.0}})
    return s.joint_param_keys


# =============================================================================
# (a) split bookkeeping in PRODUCTION survey-key order
# =============================================================================

def test_a_c1_sampled_split_bookkeeping():
    survey_keys = _production_survey_keys()
    n_cosmo, n_bins = 9, 7                 # 9 cosmo params, fixed [5,6,7,8]
    fixed_cosmo = [5, 6, 7, 8]             # -> 5 varied cosmo (LCDM core)

    marg = split_marginal_indices(
        n_cosmo_params=n_cosmo, survey_keys=survey_keys, n_bins=n_bins,
        fixed_cosmo=fixed_cosmo, fixed_survey_keys=FIXED_SURVEY_KEYS)
    samp = split_marginal_indices(
        n_cosmo_params=n_cosmo, survey_keys=survey_keys, n_bins=n_bins,
        fixed_cosmo=fixed_cosmo, fixed_survey_keys=FIXED_SURVEY_KEYS,
        lin_survey_keys=LIN_KEYS_NO_C1)

    # Marginalized reference: 5 cosmo + 3 bias/bin, 11 marginalized/bin.
    assert marg.n_nl == 5 + 3 * n_bins             # 26
    assert marg.n_lin == 11 * n_bins               # 77
    # Sampled: c1 moves nl<-lin -> +1 nl per bin, -1 lin per bin.
    assert samp.n_nl == 5 + 4 * n_bins             # 33 (== 26 + 7)
    assert samp.n_lin == 10 * n_bins               # 70
    assert _C1_KEY not in [k for _, k in samp.lin_keys]

    # c1 lands at nl_b1_pos[b] + 3 (b1, b2, bG2, c1) in production order.
    n_survey = len(survey_keys)
    c1_off = survey_keys.index(_C1_KEY)
    nl_pos = {full: pos for pos, full in enumerate(samp.nl_idx)}
    for b in range(n_bins):
        c1_full = n_cosmo + b * n_survey + c1_off
        assert nl_pos[c1_full] == samp.nl_b1_pos[b] + 3

    # b1/b2/bG2 relative offsets unchanged; ctr trio still at lin slots 2,3,4.
    assert LIN_KEYS_NO_C1.index(("pk", "ctr", "c0")) == 2
    assert LIN_KEYS_NO_C1.index(("pk", "ctr", "c2")) == 3
    assert LIN_KEYS_NO_C1.index(("pk", "ctr", "c4")) == 4


# =============================================================================
# (b) make_desi_prior_fns 10-key wiring in PRODUCTION order
# =============================================================================

def test_b_prior_fns_10key_blocks_and_sampled_c1_prod_order():
    from jaxptpolypol.desi_priors import (
        load_desi_prior_spec, make_desi_prior_fns,
    )
    survey_keys = _production_survey_keys()
    n_cosmo, n_bins = 9, 7
    fixed_cosmo = [5, 6, 7, 8]
    knl_bins = (0.52, 0.65, 0.82, 1.02, 1.29, 1.82, 2.88)
    split = split_marginal_indices(
        n_cosmo_params=n_cosmo, survey_keys=survey_keys, n_bins=n_bins,
        fixed_cosmo=fixed_cosmo, fixed_survey_keys=FIXED_SURVEY_KEYS,
        lin_survey_keys=LIN_KEYS_NO_C1)
    spec = load_desi_prior_spec()          # real spec: cov-mode ctr trio

    sigma8_ref = jnp.full(n_bins, 0.6)

    def sigma8_bins_fn(theta):
        return sigma8_ref * (1.0 + 0.05 * theta[2])   # logA proxy at cosmo idx 2

    def a_ap_bins_fn(theta):
        return jnp.ones(n_bins) * (1.0 + 0.1 * theta[4])   # h proxy at cosmo idx 4

    # c1 lives at nl_b1_pos + 3 (verified in test (a)).
    c1_pos = [int(p) + 3 for p in split.nl_b1_pos]
    mean_fn, sigma_fn, log_prior_nl = make_desi_prior_fns(
        spec, split=split, knl_bins=knl_bins, sigma8_bins_fn=sigma8_bins_fn,
        a_ap_bins_fn=a_ap_bins_fn, sigma8_ref_bins=sigma8_ref,
        lin_keys=LIN_KEYS_NO_C1,
        sampled_marginal_priors=[(_C1_KEY, c1_pos)])

    theta0 = jnp.zeros(split.n_nl)
    mu, sig = mean_fn(theta0), sigma_fn(theta0)
    assert mu.shape == (n_bins * 10,)              # 10 marginalized/bin, c1 gone
    assert sig.shape == (n_bins, 10, 10)           # cov-mode, no c1 row
    # Sampled c1 prior at fiducial (R=1): delta == -0.5*(w / sigma_c1)^2.
    sig_c1 = spec.marginalized[_C1_KEY].sigma      # 1.0125
    w = 0.4
    theta_c1 = theta0.at[c1_pos[0]].set(w)
    d = float(log_prior_nl(theta_c1)) - float(log_prior_nl(theta0))
    assert d == pytest.approx(-0.5 * (w / sig_c1) ** 2)
    # Off-fiducial: bump the a_ap proxy (cosmo idx 4) -> width divides by R.
    theta_off = theta0.at[4].set(1.0)              # a_ap = 1.1, a_amp = 1
    R = 1.1
    d_off = (float(log_prior_nl(theta_off.at[c1_pos[0]].set(w)))
             - float(log_prior_nl(theta_off)))
    assert d_off == pytest.approx(-0.5 * (w / (sig_c1 / R)) ** 2)


# =============================================================================
# (c) synthetic quadratic-in-c1: order-2 exact, order-1 misses c1^2
# =============================================================================

def _quadratic_c1_theory():
    """Full vector [t0, c1 | l0, l1]; theta_NL=[t0, c1], theta_lin=[l0, l1].

    m0(theta_NL) carries an explicit c1^2 term (coeffs Q per data component);
    M(theta_NL) is linear. So the order-2 m0 expansion is EXACT and the order-1
    one misses exactly 0.5 * Q * dc1^2 along the c1 direction.
    """
    from jaxptpolypol.sampler import make_full_params_fn
    Q = jnp.array([3.0, 0.0, -1.0])        # c1^2 coefficient per component

    def theory(p):
        t0, c1, l0, l1 = p[0], p[1], p[2], p[3]
        m0 = jnp.array([t0 + 2.0 * c1 + Q[0] * c1 ** 2,
                        t0 * c1 + Q[1] * c1 ** 2,
                        5.0 - t0 + Q[2] * c1 ** 2])
        M = jnp.array([[1.0 + t0, 0.5], [c1, 2.0], [0.0, 1.0]])
        return m0 + M @ jnp.array([l0, l1])

    packed = jnp.array([0.3, 0.0, 9.9, -9.9])   # c1 fiducial 0; junk lin values
    fpf = make_full_params_fn(packed, (0, 1))
    return theory, fpf, Q


def test_c_order2_exact_order1_misses_c1_squared():
    from jaxptpolypol.marginal_likelihood import (
        make_marginal_log_posterior_perbin,
    )
    from jaxptpolypol.marginal_taylor import (
        build_taylor_templates, make_marginal_log_posterior_taylor,
    )

    theory, fpf, Q = _quadratic_c1_theory()
    theta0 = jnp.array([0.3, 0.0])              # expansion centre (c1 = 0)
    lin_idx = (2, 3)
    mu_p, sigma_p = jnp.array([0.4, -0.3]), jnp.array([0.7, 1.3])
    data = theory(jnp.array([0.3, 0.0, 0.4, -0.3]))
    cov_inv = jnp.linalg.inv(jnp.diag(jnp.array([0.9, 1.4, 0.6])))
    common = dict(prior_mean_fn=lambda _t: mu_p,
                  prior_sigma_fn=lambda _t: sigma_p,
                  log_prior_nl_fn=lambda _t: 0.0, to_physical=lambda x: x,
                  full_params_fn=fpf, include_logdet=True)

    per = make_marginal_log_posterior_perbin(
        bin_theory_fns=[theory], bin_data=[data], bin_cov_invs=[cov_inv],
        bin_lin_idx=[lin_idx], **common)

    tt2 = build_taylor_templates(
        bin_theory_fns=[theory], bin_lin_idx=[lin_idx], full_params_fn=fpf,
        theta0=theta0, order2_m0=True)
    tt1 = build_taylor_templates(
        bin_theory_fns=[theory], bin_lin_idx=[lin_idx], full_params_fn=fpf,
        theta0=theta0, order2_m0=False)
    sur2 = make_marginal_log_posterior_taylor(
        tt2, bin_data=[data], bin_cov_invs=[cov_inv], **common)
    sur1 = make_marginal_log_posterior_taylor(
        tt1, bin_data=[data], bin_cov_invs=[cov_inv], **common)

    # Displace PURELY along c1 (theta_NL index 1).
    for dc1 in (0.2, -0.35, 0.5):
        x = theta0 + jnp.array([0.0, dc1])
        # order-2 reproduces the exact per-bin posterior (m0 quadratic -> H exact).
        np.testing.assert_allclose(float(sur2(x)), float(per(x)), rtol=1e-11)
        # order-1 (no H) misses the c1^2: non-vacuously different.
        assert abs(float(sur1(x)) - float(per(x))) > 1e-4

    # Direct m0 reconstruction: order-2 == exact; order-1 misses 0.5*Q*dc1^2.
    dc1 = 0.5
    delta = jnp.array([0.0, dc1])
    lin_arr = jnp.asarray(lin_idx)
    full_at = fpf(theta0 + delta).at[lin_arr].set(0.0)      # theta_lin = 0
    m0_exact = np.asarray(theory(full_at))
    m0_2 = np.asarray(tt2.bin_m00[0] + tt2.bin_J[0] @ delta
                      + 0.5 * jnp.einsum("ijk,j,k->i", tt2.bin_H[0], delta, delta))
    m0_1 = np.asarray(tt1.bin_m00[0] + tt1.bin_J[0] @ delta)
    np.testing.assert_allclose(m0_2, m0_exact, atol=1e-12)
    # The order-1 miss equals 0.5 * (d^2 m0/dc1^2) * dc1^2 = 0.5 * (2*Q) * dc1^2.
    expected_miss = 0.5 * (2.0 * np.asarray(Q)) * dc1 ** 2
    np.testing.assert_allclose(m0_exact - m0_1, expected_miss, atol=1e-12)
    # And that miss is non-zero where Q != 0 (test is not vacuous).
    assert abs(expected_miss[0]) > 1e-6 and abs(expected_miss[2]) > 1e-6
