"""The per-bin factorized marginal log-posterior must equal the dense monolith.

The data covariance is block-diagonal across redshift bins, each bin owns its
own 11 theta_lin, and BAO is an independent cosmology-only block -- so the
dense (n_bins*11)^2 marginalization equals the sum of per-bin 11x11
marginalizations term by term, ln det included.
"""
import os
import pathlib
from functools import partial

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import block_diag as _bd

EMULATOR_PATH = os.environ.get(
    "PFS_EMULATOR_PATH",
    "/Users/nguyenmn/cosmopower-jax-for-pfs/cosmology/jense2024/jense_2023_camb_lcdm/networks/jense_2023_camb_lcdm_Pk_lin.npz",
)
needs_emulator = pytest.mark.skipif(
    not pathlib.Path(str(EMULATOR_PATH)).exists(),
    reason="PFS emulator not available (set PFS_EMULATOR_PATH)",
)

FIXED_COSMO = (5, 6, 7, 8)
FIXED_SURVEY_KEYS = {('shared', 'k_nl', None), ('shared', 'ndens', None)}


def _random_block_cov(data_block, rng):
    """Random, non-identity, diagonal covariance for one bin's data block.

    Random (an identity covariance would hide block-pairing bugs) but scaled
    to the data: P_ell ~ 1e4 and B0 ~ 1e8 in these units, so an *absolute*
    O(1) covariance weights the bispectrum by ~1e16, giving chi^2 ~ 1e15 and a
    numerically singular A (cond ~ 8e16). The per-bin and monolithic algebra
    are still identical there, but the comparison would then be measuring
    float64 round-off in a ~1e5-sigma-displaced regime rather than the block
    factorization. Fractional errors keep chi^2 = O(1e2-1e3) and cond(A) ~ 5e4.
    """
    scale = np.abs(np.asarray(data_block))
    scale = np.maximum(scale, 1e-3 * scale.max())
    return np.diag((rng.uniform(0.005, 0.02, size=scale.shape) * scale) ** 2)


@pytest.fixture(scope="module")
def cfg():
    """Small 2-bin config; mirrors tests/test_theory_perbin.py's fixture."""
    from jaxptpolypol.model import CosmoEmulator, PS1LoopModel, BispectrumTreeModel
    from jaxptpolypol.params import CosmoParams, FullShapeSurveyParams, pack_joint_params
    from jaxptpolypol.theory import (
        build_bispectrum_triangles_from_k_grid, compute_fiducial_distances)
    from ps_1loop_jax import background as bg

    MNU_FIXED, K_NL_RSD = 0.06, 0.45
    z_bins, knl_bins, n_bar = (0.7, 0.9), (0.52, 0.65), (3.06e-4, 9.61e-4)
    cosmo_dict = {'ombh2': 0.02237, 'omch2': 0.1200, 'logA': 3.044,
                  'ns': 0.9649, 'h': 0.6736,
                  'z': 0.7, 'A_b': 3.13, 'eta_b': 0.603, 'logT_AGN': 7.8}
    cosmo = CosmoParams(cosmo_dict)

    def b1z(z): return 0.9 + 0.4 * z
    def b2z(z): return -0.704 - 0.208 * z + 0.183 * z**2 - 0.00771 * z**3
    def bG2z(z): return -(2. / 7.) * (b1z(z) - 1.)
    def bGamma3z(z): return (23. / 42.) * (b1z(z) - 1.)
    def Dplusz(z):
        return float(bg.growth_factor(cosmo_dict['ombh2'], cosmo_dict['omch2'],
                                      cosmo_dict['h'], z, mnu=MNU_FIXED))

    surveys = []
    for z, knl, nd in zip(z_bins, knl_bins, n_bar):
        surveys.append(FullShapeSurveyParams(
            shared={'bias': {'b1': b1z(z), 'b2': b2z(z), 'bG2': bG2z(z),
                             'bGamma3': bGamma3z(z)},
                    'stoch': {'P_shot': 1.0}, 'k_nl': knl, 'ndens': nd},
            pk={'ctr': {'c0': 25. * Dplusz(z)**2, 'c2': 25. * Dplusz(z)**2,
                        'c4': Dplusz(z)**2, 'cfog': knl**(-4)},
                'stoch': {'a0': 0., 'a2': 0.}},
            bk={'ctr': {'c1': 0.0}, 'stoch': {'B_shot': 1.0, 'A_shot': 1.0}},
        ))
    survey_keys = surveys[0].joint_param_keys
    packed = pack_joint_params(cosmo, surveys)
    n_cosmo = int(sum(cosmo.param_sizes))
    Hz_fid, DAz_fid = compute_fiducial_distances(cosmo, z_bins)
    emulator = CosmoEmulator(probe='custom_log', emulator_path=EMULATOR_PATH)
    kwargs = dict(
        pklin_emulator=emulator, ps1loop_model=PS1LoopModel(do_irres=True),
        bispectrum_model=BispectrumTreeModel(do_AP=True, k_nl_rsd=K_NL_RSD),
        cosmo_keys=cosmo.param_keys, cosmo_sizes=cosmo.param_sizes,
        survey_keys=survey_keys, ap=True, z_bins=z_bins,
        Hz_fid=Hz_fid, DAz_fid=DAz_fid, n_gl=8, num_mu=8, num_phi=8,
        background_mode="direct")
    k = jnp.linspace(0.02, 0.18, 8)
    triangles, _ = build_bispectrum_triangles_from_k_grid(
        k, k_min=0.02, k_max=0.10, dk=float(k[1] - k[0]))
    return kwargs, packed, k, triangles, len(z_bins), survey_keys, n_cosmo


@needs_emulator
def test_perbin_logpost_equals_monolith(cfg):
    """Same joint posterior, computed two ways: sum-of-bins == dense 22x22."""
    from jaxptpolypol.marginal_likelihood import (
        make_constant_prior_fns, make_marginal_log_posterior,
        bin_lin_slices, make_marginal_log_posterior_perbin,
        split_marginal_indices)
    from jaxptpolypol.sampler import make_full_params_fn
    from jaxptpolypol.theory import make_joint_pk_bk_fn, make_joint_pk_bk_bin_fn
    kwargs, packed, k, triangles, n_bins, survey_keys, n_cosmo = cfg

    joint = make_joint_pk_bk_fn(**kwargs)
    theory_fn = partial(joint, k=k, triangles=triangles)
    split = split_marginal_indices(
        n_cosmo_params=n_cosmo, survey_keys=survey_keys, n_bins=n_bins,
        fixed_cosmo=FIXED_COSMO, fixed_survey_keys=FIXED_SURVEY_KEYS)

    data = theory_fn(packed)                       # noiseless mock
    n_data = data.shape[0]
    block = n_data // n_bins
    rng = np.random.default_rng(0)
    # Block-diagonal, non-identity covariance (identity would hide block-pairing bugs)
    blocks = [_random_block_cov(data[b * block:(b + 1) * block], rng)
              for b in range(n_bins)]
    cov = _bd(*blocks)
    cov_inv = jnp.asarray(np.linalg.inv(cov))
    mu_p = packed[jnp.array(split.lin_idx)]
    sigma_p = jnp.asarray(rng.uniform(0.5, 2.0, size=split.n_lin))
    prior_mean_fn, prior_sigma_fn = make_constant_prior_fns(mu_p, sigma_p)
    fpf = make_full_params_fn(packed, split.nl_idx)

    mono = make_marginal_log_posterior(
        theory_fn=theory_fn, data=data, cov_inv=cov_inv, lin_idx=split.lin_idx,
        prior_mean_fn=prior_mean_fn, prior_sigma_fn=prior_sigma_fn,
        log_prior_nl_fn=lambda t: 0.0, to_physical=lambda x: x,
        full_params_fn=fpf, include_logdet=True)

    bin_fns = [partial(make_joint_pk_bk_bin_fn(bin_index=b, **kwargs),
                       k=k, triangles=triangles) for b in range(n_bins)]
    per = make_marginal_log_posterior_perbin(
        bin_theory_fns=bin_fns,
        bin_data=[data[b * block:(b + 1) * block] for b in range(n_bins)],
        bin_cov_invs=[jnp.asarray(np.linalg.inv(blocks[b])) for b in range(n_bins)],
        bin_lin_idx=[split.lin_idx[sl] for sl in bin_lin_slices(split, n_bins)],
        prior_mean_fn=prior_mean_fn, prior_sigma_fn=prior_sigma_fn,
        log_prior_nl_fn=lambda t: 0.0, to_physical=lambda x: x,
        full_params_fn=fpf, include_logdet=True)

    x0 = packed[jnp.array(split.nl_idx)]
    np.testing.assert_allclose(float(per(x0)), float(mono(x0)), rtol=1e-10)
    # displaced point -- catches wrong theta_NL wiring that a fiducial-only test misses
    x1 = x0 * (1.0 + 0.01 * jnp.arange(x0.shape[0]) / max(x0.shape[0], 1))
    np.testing.assert_allclose(float(per(x1)), float(mono(x1)), rtol=1e-10)


@needs_emulator
def test_perbin_logpost_with_extra_bao_term_equals_monolith(cfg):
    """BAO as a separate cosmology-only chi^2 == BAO concatenated into the monolith."""
    from jaxptpolypol.marginal_likelihood import (
        make_constant_prior_fns, make_marginal_log_posterior,
        bin_lin_slices, make_marginal_log_posterior_perbin,
        split_marginal_indices)
    from jaxptpolypol.sampler import make_full_params_fn
    from jaxptpolypol.theory import make_joint_pk_bk_fn, make_joint_pk_bk_bin_fn
    kwargs, packed, k, triangles, n_bins, survey_keys, n_cosmo = cfg

    joint = make_joint_pk_bk_fn(**kwargs)
    theory_fn = partial(joint, k=k, triangles=triangles)
    split = split_marginal_indices(
        n_cosmo_params=n_cosmo, survey_keys=survey_keys, n_bins=n_bins,
        fixed_cosmo=FIXED_COSMO, fixed_survey_keys=FIXED_SURVEY_KEYS)

    data = theory_fn(packed)
    n_data = data.shape[0]
    block = n_data // n_bins
    rng = np.random.default_rng(1)
    blocks = [_random_block_cov(data[b * block:(b + 1) * block], rng)
              for b in range(n_bins)]

    # Stand-in cosmology-only 'BAO' block: linear map of the cosmology sub-vector,
    # carrying no theta_lin dependence, with its own dense 13x13 covariance.
    n_bao = 13
    A_bao = jnp.asarray(rng.normal(size=(n_bao, n_cosmo)))

    def bao_fn(full_params):
        return A_bao @ full_params[:n_cosmo]

    L = rng.normal(size=(n_bao, n_bao))
    cov_bao = L @ L.T + n_bao * np.eye(n_bao)
    cov_bao_inv = jnp.asarray(np.linalg.inv(cov_bao))
    # Offset the BAO data so its residual is non-zero already at the fiducial.
    data_bao = jnp.asarray(np.asarray(bao_fn(packed)) + rng.normal(size=n_bao))

    def combined_fn(full_params):
        return jnp.concatenate([theory_fn(full_params), bao_fn(full_params)])

    data_all = jnp.concatenate([data, data_bao])
    # Exact inverse of block_diag(*blocks, cov_bao), built block-wise so the
    # monolith's cov_inv is bit-for-bit the block-diagonal of the per-block ones.
    cov_inv_all = jnp.asarray(_bd(*[np.linalg.inv(b) for b in blocks],
                                  np.asarray(cov_bao_inv)))

    mu_p = packed[jnp.array(split.lin_idx)]
    sigma_p = jnp.asarray(rng.uniform(0.5, 2.0, size=split.n_lin))
    prior_mean_fn, prior_sigma_fn = make_constant_prior_fns(mu_p, sigma_p)
    fpf = make_full_params_fn(packed, split.nl_idx)

    mono = make_marginal_log_posterior(
        theory_fn=combined_fn, data=data_all, cov_inv=cov_inv_all,
        lin_idx=split.lin_idx,
        prior_mean_fn=prior_mean_fn, prior_sigma_fn=prior_sigma_fn,
        log_prior_nl_fn=lambda t: 0.0, to_physical=lambda x: x,
        full_params_fn=fpf, include_logdet=True)

    bin_fns = [partial(make_joint_pk_bk_bin_fn(bin_index=b, **kwargs),
                       k=k, triangles=triangles) for b in range(n_bins)]
    per = make_marginal_log_posterior_perbin(
        bin_theory_fns=bin_fns,
        bin_data=[data[b * block:(b + 1) * block] for b in range(n_bins)],
        bin_cov_invs=[jnp.asarray(np.linalg.inv(blocks[b])) for b in range(n_bins)],
        bin_lin_idx=[split.lin_idx[sl] for sl in bin_lin_slices(split, n_bins)],
        extra_theory_fn=bao_fn, extra_data=data_bao, extra_cov_inv=cov_bao_inv,
        prior_mean_fn=prior_mean_fn, prior_sigma_fn=prior_sigma_fn,
        log_prior_nl_fn=lambda t: 0.0, to_physical=lambda x: x,
        full_params_fn=fpf, include_logdet=True)

    x0 = packed[jnp.array(split.nl_idx)]
    np.testing.assert_allclose(float(per(x0)), float(mono(x0)), rtol=1e-10)
    x1 = x0 * (1.0 + 0.01 * jnp.arange(x0.shape[0]) / max(x0.shape[0], 1))
    np.testing.assert_allclose(float(per(x1)), float(mono(x1)), rtol=1e-10)
    # The BAO term must actually contribute (otherwise the check above is vacuous).
    r0 = np.asarray(data_bao - bao_fn(packed))
    assert abs(float(-0.5 * r0 @ np.asarray(cov_bao_inv) @ r0)) > 1e-6


@needs_emulator
def test_scan_logpost_equals_perbin_and_monolith(cfg):
    """lax.scan over stacked per-bin blocks == unrolled per-bin == dense monolith."""
    from jaxptpolypol.marginal_likelihood import (
        make_constant_prior_fns, make_marginal_log_posterior,
        bin_lin_slices, make_marginal_log_posterior_perbin,
        make_marginal_log_posterior_scan, split_marginal_indices)
    from jaxptpolypol.sampler import make_full_params_fn
    from jaxptpolypol.theory import make_joint_pk_bk_fn, make_joint_pk_bk_bin_fn
    kwargs, packed, k, triangles, n_bins, survey_keys, n_cosmo = cfg

    joint = make_joint_pk_bk_fn(**kwargs)
    theory_fn = partial(joint, k=k, triangles=triangles)
    split = split_marginal_indices(
        n_cosmo_params=n_cosmo, survey_keys=survey_keys, n_bins=n_bins,
        fixed_cosmo=FIXED_COSMO, fixed_survey_keys=FIXED_SURVEY_KEYS)

    data = theory_fn(packed)
    n_data = data.shape[0]
    block = n_data // n_bins
    rng = np.random.default_rng(0)
    blocks = [_random_block_cov(data[b * block:(b + 1) * block], rng)
              for b in range(n_bins)]
    cov = _bd(*blocks)
    cov_inv = jnp.asarray(np.linalg.inv(cov))
    mu_p = packed[jnp.array(split.lin_idx)]
    sigma_p = jnp.asarray(rng.uniform(0.5, 2.0, size=split.n_lin))
    prior_mean_fn, prior_sigma_fn = make_constant_prior_fns(mu_p, sigma_p)
    fpf = make_full_params_fn(packed, split.nl_idx)

    mono = make_marginal_log_posterior(
        theory_fn=theory_fn, data=data, cov_inv=cov_inv, lin_idx=split.lin_idx,
        prior_mean_fn=prior_mean_fn, prior_sigma_fn=prior_sigma_fn,
        log_prior_nl_fn=lambda t: 0.0, to_physical=lambda x: x,
        full_params_fn=fpf, include_logdet=True)

    bin_fns = [partial(make_joint_pk_bk_bin_fn(bin_index=b, **kwargs),
                       k=k, triangles=triangles) for b in range(n_bins)]
    common = dict(
        bin_theory_fns=bin_fns,
        bin_data=[data[b * block:(b + 1) * block] for b in range(n_bins)],
        bin_cov_invs=[jnp.asarray(np.linalg.inv(blocks[b])) for b in range(n_bins)],
        bin_lin_idx=[split.lin_idx[sl] for sl in bin_lin_slices(split, n_bins)],
        prior_mean_fn=prior_mean_fn, prior_sigma_fn=prior_sigma_fn,
        log_prior_nl_fn=lambda t: 0.0, to_physical=lambda x: x,
        full_params_fn=fpf, include_logdet=True)
    per = make_marginal_log_posterior_perbin(**common)
    scan = make_marginal_log_posterior_scan(**common)

    x0 = packed[jnp.array(split.nl_idx)]
    np.testing.assert_allclose(float(scan(x0)), float(per(x0)), rtol=1e-10)
    np.testing.assert_allclose(float(scan(x0)), float(mono(x0)), rtol=1e-10)
    # displaced point -- catches wrong theta_NL / wrong per-bin xs wiring
    x1 = x0 * (1.0 + 0.01 * jnp.arange(x0.shape[0]) / max(x0.shape[0], 1))
    np.testing.assert_allclose(float(scan(x1)), float(per(x1)), rtol=1e-10)
    np.testing.assert_allclose(float(scan(x1)), float(mono(x1)), rtol=1e-10)
    # The two points must differ, otherwise the displaced check is vacuous.
    assert abs(float(scan(x1)) - float(scan(x0))) > 1e-6


@needs_emulator
def test_scan_logpost_with_extra_bao_term_equals_perbin(cfg):
    """The optional extra (BAO) block must be added once, outside the scan."""
    from jaxptpolypol.marginal_likelihood import (
        make_constant_prior_fns, bin_lin_slices,
        make_marginal_log_posterior_perbin, make_marginal_log_posterior_scan,
        split_marginal_indices)
    from jaxptpolypol.sampler import make_full_params_fn
    from jaxptpolypol.theory import make_joint_pk_bk_fn, make_joint_pk_bk_bin_fn
    kwargs, packed, k, triangles, n_bins, survey_keys, n_cosmo = cfg

    joint = make_joint_pk_bk_fn(**kwargs)
    theory_fn = partial(joint, k=k, triangles=triangles)
    split = split_marginal_indices(
        n_cosmo_params=n_cosmo, survey_keys=survey_keys, n_bins=n_bins,
        fixed_cosmo=FIXED_COSMO, fixed_survey_keys=FIXED_SURVEY_KEYS)

    data = theory_fn(packed)
    block = data.shape[0] // n_bins
    rng = np.random.default_rng(1)
    blocks = [_random_block_cov(data[b * block:(b + 1) * block], rng)
              for b in range(n_bins)]

    n_bao = 13
    A_bao = jnp.asarray(rng.normal(size=(n_bao, n_cosmo)))

    def bao_fn(full_params):
        return A_bao @ full_params[:n_cosmo]

    L = rng.normal(size=(n_bao, n_bao))
    cov_bao = L @ L.T + n_bao * np.eye(n_bao)
    cov_bao_inv = jnp.asarray(np.linalg.inv(cov_bao))
    data_bao = jnp.asarray(np.asarray(bao_fn(packed)) + rng.normal(size=n_bao))

    mu_p = packed[jnp.array(split.lin_idx)]
    sigma_p = jnp.asarray(rng.uniform(0.5, 2.0, size=split.n_lin))
    prior_mean_fn, prior_sigma_fn = make_constant_prior_fns(mu_p, sigma_p)
    fpf = make_full_params_fn(packed, split.nl_idx)

    bin_fns = [partial(make_joint_pk_bk_bin_fn(bin_index=b, **kwargs),
                       k=k, triangles=triangles) for b in range(n_bins)]
    common = dict(
        bin_theory_fns=bin_fns,
        bin_data=[data[b * block:(b + 1) * block] for b in range(n_bins)],
        bin_cov_invs=[jnp.asarray(np.linalg.inv(blocks[b])) for b in range(n_bins)],
        bin_lin_idx=[split.lin_idx[sl] for sl in bin_lin_slices(split, n_bins)],
        extra_theory_fn=bao_fn, extra_data=data_bao, extra_cov_inv=cov_bao_inv,
        prior_mean_fn=prior_mean_fn, prior_sigma_fn=prior_sigma_fn,
        log_prior_nl_fn=lambda t: 0.0, to_physical=lambda x: x,
        full_params_fn=fpf, include_logdet=True)
    per = make_marginal_log_posterior_perbin(**common)
    scan = make_marginal_log_posterior_scan(**common)

    x0 = packed[jnp.array(split.nl_idx)]
    np.testing.assert_allclose(float(scan(x0)), float(per(x0)), rtol=1e-10)
    x1 = x0 * (1.0 + 0.01 * jnp.arange(x0.shape[0]) / max(x0.shape[0], 1))
    np.testing.assert_allclose(float(scan(x1)), float(per(x1)), rtol=1e-10)
    r0 = np.asarray(data_bao - bao_fn(packed))
    assert abs(float(-0.5 * r0 @ np.asarray(cov_bao_inv) @ r0)) > 1e-6


def test_scan_requires_uniform_bin_shapes():
    """Stacking is only defined for equal-sized blocks; ragged input must raise.

    Emulator-free: the same 2-bin toy as the per-bin tiling guard test.
    """
    from jaxptpolypol.marginal_likelihood import (
        make_marginal_log_posterior_perbin, make_marginal_log_posterior_scan)

    def bin0(p):
        return jnp.array([p[0] + 2.0 * p[1] + 3.0 * p[2]])

    def bin1(p):
        return jnp.array([p[0] - 1.5 * p[3] + 0.5 * p[4]])

    def bin1_wide(p):
        return jnp.array([p[0] - 1.5 * p[3], 0.5 * p[4]])

    fid = jnp.array([1.0, 0.1, 0.2, 0.3, 0.4])
    common = dict(
        log_prior_nl_fn=lambda t: 0.0,
        to_physical=lambda x: x,
        full_params_fn=lambda t: fid.at[jnp.array([0])].set(t),
        prior_mean_fn=lambda t: jnp.zeros(4),
        prior_sigma_fn=lambda t: jnp.ones(4),
    )
    uniform = dict(
        bin_theory_fns=[bin0, bin1],
        bin_data=[jnp.array([1.0]), jnp.array([1.0])],
        bin_cov_invs=[jnp.eye(1), jnp.eye(1)],
        bin_lin_idx=[(1, 2), (3, 4)],
        **common)
    # Uniform toy: scan matches the unrolled per-bin form exactly.
    ok_scan = make_marginal_log_posterior_scan(**uniform)
    ok_per = make_marginal_log_posterior_perbin(**uniform)
    np.testing.assert_allclose(float(ok_scan(fid[:1])), float(ok_per(fid[:1])),
                               rtol=1e-10)

    # Ragged data blocks cannot be stacked.
    ragged = dict(uniform)
    ragged.update(bin_theory_fns=[bin0, bin1_wide],
                  bin_data=[jnp.array([1.0]), jnp.array([1.0, 2.0])],
                  bin_cov_invs=[jnp.eye(1), jnp.eye(2)])
    with pytest.raises(ValueError, match="same length"):
        make_marginal_log_posterior_scan(**ragged)

    # Ragged marginalized blocks cannot be stacked either.
    ragged_lin = dict(uniform)
    ragged_lin.update(bin_lin_idx=[(1,), (2, 3, 4)])
    with pytest.raises(ValueError, match="same number of marginalized"):
        make_marginal_log_posterior_scan(**ragged_lin)


def test_bin_lin_slices_and_validation():
    """Pure-python: slices are 11-wide bin-major; non-bin-major input raises."""
    from jaxptpolypol.marginal_likelihood import (
        LIN_SURVEY_KEYS, MarginalSplit, bin_lin_slices, split_marginal_indices)
    from jaxptpolypol.params import FullShapeSurveyParams

    n_per = len(LIN_SURVEY_KEYS)
    assert n_per == 11

    # 1. Against the real producer: split_marginal_indices is bin-major with
    #    exactly 11 contiguous lin entries per bin.
    survey = FullShapeSurveyParams(
        shared={'bias': {'b1': 1.0, 'b2': 0.0, 'bG2': 0.0, 'bGamma3': 0.0},
                'stoch': {'P_shot': 1.0}, 'k_nl': 0.5, 'ndens': 1e-4},
        pk={'ctr': {'c0': 0.0, 'c2': 0.0, 'c4': 0.0, 'cfog': 0.0},
            'stoch': {'a0': 0.0, 'a2': 0.0}},
        bk={'ctr': {'c1': 0.0}, 'stoch': {'B_shot': 1.0, 'A_shot': 1.0}},
    )
    n_bins = 3
    real = split_marginal_indices(
        n_cosmo_params=9, survey_keys=survey.joint_param_keys, n_bins=n_bins,
        fixed_cosmo=FIXED_COSMO, fixed_survey_keys=FIXED_SURVEY_KEYS)
    slices = bin_lin_slices(real, n_bins)
    assert slices == (slice(0, 11), slice(11, 22), slice(22, 33))
    for b, sl in enumerate(slices):
        assert [key[0] for key in real.lin_keys[sl]] == [b] * n_per
        assert [key[1] for key in real.lin_keys[sl]] == list(LIN_SURVEY_KEYS)

    def _split(lin_keys):
        n_lin = len(lin_keys)
        return MarginalSplit(nl_idx=(), lin_idx=tuple(range(n_lin)),
                             lin_keys=tuple(lin_keys), nl_b1_pos=(),
                             n_lin=n_lin, n_nl=0)

    # 2. Bin-minor (interleaved) ordering must raise, not silently mis-slice.
    bad = _split([(b, key) for key in LIN_SURVEY_KEYS for b in range(n_bins)])
    with pytest.raises(ValueError, match="bin-major"):
        bin_lin_slices(bad, n_bins)

    # 3. Unequal counts per bin (n_lin not a multiple of n_bins) must raise.
    with pytest.raises(ValueError, match="multiple"):
        bin_lin_slices(real, 2)

    # 4. Inconsistent lin_keys / n_lin bookkeeping must raise.
    broken = MarginalSplit(nl_idx=(), lin_idx=(0, 1), lin_keys=((0, 'a'),),
                           nl_b1_pos=(), n_lin=2, n_nl=0)
    with pytest.raises(ValueError, match="lin_keys"):
        bin_lin_slices(broken, 1)

    with pytest.raises(ValueError, match="positive"):
        bin_lin_slices(real, 0)


def test_perbin_rejects_prior_vector_that_does_not_tile_bins():
    """The prior-slice tiling guard must fire, not silently mis-slice.

    Emulator-free: a 2-bin toy whose theory is linear in its lin params.
    """
    from jaxptpolypol.marginal_likelihood import make_marginal_log_posterior_perbin

    # full vector: [nl | bin0 lin (2) | bin1 lin (2)]
    def bin0(p):
        return jnp.array([p[0] + 2.0 * p[1] + 3.0 * p[2]])

    def bin1(p):
        return jnp.array([p[0] - 1.5 * p[3] + 0.5 * p[4]])

    fid = jnp.array([1.0, 0.1, 0.2, 0.3, 0.4])
    common = dict(
        bin_theory_fns=[bin0, bin1],
        bin_data=[jnp.array([1.0]), jnp.array([1.0])],
        bin_cov_invs=[jnp.eye(1), jnp.eye(1)],
        bin_lin_idx=[(1, 2), (3, 4)],          # 2 lin params per bin -> n_lin = 4
        log_prior_nl_fn=lambda t: 0.0,
        to_physical=lambda x: x,
        full_params_fn=lambda t: fid.at[jnp.array([0])].set(t),
    )

    # Correct width (4) works.
    ok = make_marginal_log_posterior_perbin(
        prior_mean_fn=lambda t: jnp.zeros(4),
        prior_sigma_fn=lambda t: jnp.ones(4), **common)
    assert np.isfinite(float(ok(fid[:1])))

    # Wrong width (3) must raise, naming the offending function.
    bad = make_marginal_log_posterior_perbin(
        prior_mean_fn=lambda t: jnp.zeros(3),
        prior_sigma_fn=lambda t: jnp.ones(4), **common)
    with pytest.raises(ValueError, match="prior_mean_fn"):
        bad(fid[:1])

    bad_sigma = make_marginal_log_posterior_perbin(
        prior_mean_fn=lambda t: jnp.zeros(4),
        prior_sigma_fn=lambda t: jnp.ones(5), **common)
    with pytest.raises(ValueError, match="prior_sigma_fn"):
        bad_sigma(fid[:1])
