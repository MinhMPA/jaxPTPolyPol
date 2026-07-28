"""Tier-2 validation: the Taylor surrogate against the REAL P+B per-bin pipeline.

Tasks 1-3 validated ``build_taylor_templates`` /
``make_marginal_log_posterior_taylor`` on analytic toys where the expansion is
exact. This module is first contact with the actual ps_1loop_jax theory at the
2-bin small config (the shared ``cfg`` fixture): the surrogate is only an
approximation of the true ``(m0, M)(theta_NL)`` there, so we pin the three
properties the downstream IS / delayed-acceptance layers rely on:

1. ``test_center_exactness`` -- at the expansion centre the carried tensors are
   the exact ``m0(theta0), J, H, M(theta0), dM``, so the surrogate posterior must
   equal the exact per-bin posterior to the float64 floor (this also catches any
   wiring / prior / logdet mismatch between the two builders).
2. ``test_gradient_matches_at_center`` -- the logdet-tilt fidelity check. The
   ``ln det(A Sigma_p)`` term genuinely tilts the posterior; because ``J`` and
   ``dM`` are exact at the centre, the surrogate and the exact posterior share
   the same analytic gradient ``g`` there, so ``jax.grad`` of the two must
   agree.
3. ``test_radius_error_profile`` -- the calibration record: how fast the
   surrogate degrades as we step away from the centre along random directions.
   Printed for the Task-7 IS / DA expectations.
"""
from functools import partial

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from conftest import FIXED_COSMO, FIXED_SURVEY_KEYS, _random_block_cov, needs_emulator

import pytest


@pytest.fixture(scope="module")
def taylor_pipeline(cfg):
    """Build the surrogate and the exact per-bin posterior ONCE from identical
    inputs, so the three tests share the (~1-2 min) template build.

    Returns ``(sur, per, theta0, d)`` where ``sur`` is the Taylor surrogate,
    ``per`` the exact per-bin posterior, ``theta0`` the physical theta_NL
    fiducial (expansion centre), and ``d`` its dimension.
    """
    from jaxptpolypol.marginal_likelihood import (
        bin_lin_slices, make_constant_prior_fns,
        make_marginal_log_posterior_perbin, split_marginal_indices)
    from jaxptpolypol.marginal_taylor import (
        build_taylor_templates, make_marginal_log_posterior_taylor)
    from jaxptpolypol.sampler import make_full_params_fn
    from jaxptpolypol.theory import make_joint_pk_bk_bin_fn, make_joint_pk_bk_fn

    kwargs, packed, k, triangles, n_bins, survey_keys, n_cosmo = cfg

    split = split_marginal_indices(
        n_cosmo_params=n_cosmo, survey_keys=survey_keys, n_bins=n_bins,
        fixed_cosmo=FIXED_COSMO, fixed_survey_keys=FIXED_SURVEY_KEYS)

    # Noiseless mock data = jitted joint theory at the packed fiducial, sliced per bin.
    joint = make_joint_pk_bk_fn(**kwargs)
    theory_fn = partial(joint, k=k, triangles=triangles)
    data = jax.jit(theory_fn)(packed)
    block = data.shape[0] // n_bins
    bin_data = [data[b * block:(b + 1) * block] for b in range(n_bins)]

    # Per-bin distinct, non-identity diagonal covariances (as in test_marginal_perbin).
    rng = np.random.default_rng(0)
    blocks = [_random_block_cov(bin_data[b], rng) for b in range(n_bins)]
    bin_cov_invs = [jnp.asarray(np.linalg.inv(blocks[b])) for b in range(n_bins)]

    # Fiducial-centered constant priors, Fisher-consistent widths.
    mu_p = packed[jnp.array(split.lin_idx)]
    sigma_p = jnp.asarray(rng.uniform(0.5, 2.0, size=split.n_lin))
    prior_mean_fn, prior_sigma_fn = make_constant_prior_fns(mu_p, sigma_p)

    fpf = make_full_params_fn(packed, split.nl_idx)
    bin_lin_idx = [split.lin_idx[sl] for sl in bin_lin_slices(split, n_bins)]
    bin_fns = [partial(make_joint_pk_bk_bin_fn(bin_index=b, **kwargs),
                       k=k, triangles=triangles) for b in range(n_bins)]

    theta0 = packed[jnp.array(split.nl_idx)]

    common = dict(
        prior_mean_fn=prior_mean_fn, prior_sigma_fn=prior_sigma_fn,
        log_prior_nl_fn=lambda t: 0.0, to_physical=lambda x: x,
        full_params_fn=fpf, include_logdet=True)

    per = make_marginal_log_posterior_perbin(
        bin_theory_fns=bin_fns, bin_data=bin_data, bin_cov_invs=bin_cov_invs,
        bin_lin_idx=bin_lin_idx, **common)

    tt = build_taylor_templates(
        bin_theory_fns=bin_fns, bin_lin_idx=bin_lin_idx, full_params_fn=fpf,
        theta0=theta0, order2_m0=True, chunk_J=4, chunk_H=2)
    sur = make_marginal_log_posterior_taylor(
        tt, bin_data=bin_data, bin_cov_invs=bin_cov_invs, **common)

    return sur, per, theta0, int(split.n_nl)


@needs_emulator
def test_center_exactness(taylor_pipeline):
    """At the expansion centre the surrogate reproduces the exact per-bin
    posterior to the float64 floor (center tensors are exact by construction)."""
    sur, per, theta0, _d = taylor_pipeline
    np.testing.assert_allclose(float(sur(theta0)), float(per(theta0)), rtol=1e-10)


@needs_emulator
def test_gradient_matches_at_center(taylor_pipeline):
    """Logdet-tilt fidelity: the gradients of the surrogate and the exact
    per-bin posterior agree at the centre.

    ``J`` and ``dM`` are exact at ``theta0``, so both posteriors carry the same
    analytic gradient ``g`` there -- the ``ln det(A Sigma_p)`` tilt included.
    Compared via ``jax.grad`` of each posterior (reverse mode is fine on this
    small config; ``test_marginal_pipeline`` already runs ``jax.hessian`` on
    the monolith).

    Central differences at a fixed absolute ``h=0.02`` were tried first and
    measured 5e-4 .. 1.7e-1 relative discrepancies on precisely the five
    cosmology dims, ordered by each dim's RELATIVE step (h=0.02 is an 89%
    displacement of ombh2, 17% of omch2, 3% of h, 2% of ns) while every
    survey-bias dim matched to all printed digits -- the O(h^2 f''') truncation
    of the finite difference beating the surrogate's deliberate higher-order
    truncation, not a gradient defect. ``jax.grad`` compares the true
    gradients with no truncation term.
    """
    sur, per, theta0, _d = taylor_pipeline
    g_sur = np.asarray(jax.grad(sur)(theta0))
    g_per = np.asarray(jax.grad(per)(theta0))
    # The tilt must be real, else the comparison is vacuous.
    assert np.max(np.abs(g_per)) > 0.0
    # atol is the float-noise floor for the one sub-1e-3 gradient component;
    # it is ~3e-12 relative to the largest entry (~3.7e4).
    np.testing.assert_allclose(g_sur, g_per, rtol=1e-4, atol=1e-7)


@needs_emulator
def test_radius_error_profile(taylor_pipeline):
    """Calibration record: |surrogate - exact| along 6 fixed random directions,
    at whitened radii r in {1, 2, 3} (each param moves r x 5% of its fiducial
    magnitude). The median error must increase monotonically in r and be < 0.5
    log-units at r=1. The full profile is printed for the Task-7 IS/DA layer.
    """
    sur, per, theta0, d = taylor_pipeline

    scale = 0.05 * jnp.maximum(jnp.abs(theta0), 1e-3)
    rng = np.random.default_rng(20240724)
    dirs = []
    for _ in range(6):
        v = rng.normal(size=d)
        dirs.append(jnp.asarray(v / np.linalg.norm(v)))

    radii = (1, 2, 3)
    profile = {}
    for r in radii:
        errs = []
        for v in dirs:
            x = theta0 + r * v * scale
            errs.append(abs(float(sur(x)) - float(per(x))))
        profile[r] = (float(np.median(errs)), float(np.max(errs)))

    print("\nradius error profile |surrogate - exact| (log-units):")
    print("  r -> median / max")
    for r in radii:
        print(f"  r={r}: median={profile[r][0]:.6e}  max={profile[r][1]:.6e}")

    meds = [profile[r][0] for r in radii]
    # Monotonically increasing median error with radius (surrogate degrades).
    assert meds[0] < meds[1] < meds[2], f"non-monotone medians: {meds}"
    # Faithful near the centre.
    assert meds[0] < 0.5, f"median error at r=1 too large: {meds[0]}"
