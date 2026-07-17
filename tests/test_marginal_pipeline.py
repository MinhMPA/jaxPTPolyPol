"""Tier-1 integration tests: exact template reconstruction on the real P+B pipeline."""
import os
import pathlib

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

EMULATOR_PATH = os.environ.get(
    "PFS_EMULATOR_PATH",
    "/Users/nguyenmn/cosmopower-jax-for-pfs/cosmology/jense2024/jense_2023_camb_lcdm/networks/jense_2023_camb_lcdm_Pk_lin.npz",
)
needs_emulator = pytest.mark.skipif(
    not pathlib.Path(str(EMULATOR_PATH)).exists(),
    reason="PFS emulator not available (set PFS_EMULATOR_PATH)",
)


@pytest.fixture(scope="module")
def pipeline():
    from jaxptpolypol.model import CosmoEmulator, PS1LoopModel, BispectrumTreeModel
    from jaxptpolypol.params import CosmoParams, FullShapeSurveyParams, pack_joint_params
    from jaxptpolypol.theory import (
        build_bispectrum_triangles_from_k_grid, compute_fiducial_distances,
        make_joint_pk_bk_fn)
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
    joint_survey_keys = surveys[0].joint_param_keys
    packed = pack_joint_params(cosmo, surveys)
    Hz_fid, DAz_fid = compute_fiducial_distances(cosmo, z_bins)

    emulator = CosmoEmulator(probe='custom_log', emulator_path=EMULATOR_PATH)
    joint_fn = make_joint_pk_bk_fn(
        pklin_emulator=emulator, ps1loop_model=PS1LoopModel(do_irres=True),
        bispectrum_model=BispectrumTreeModel(do_AP=True, k_nl_rsd=K_NL_RSD),
        cosmo_keys=cosmo.param_keys, cosmo_sizes=cosmo.param_sizes,
        survey_keys=joint_survey_keys, ap=True, z_bins=z_bins,
        Hz_fid=Hz_fid, DAz_fid=DAz_fid, n_gl=8, num_mu=8, num_phi=8,
        background_mode="direct")
    k = jnp.linspace(0.02, 0.18, 8)
    triangles, _ = build_bispectrum_triangles_from_k_grid(
        k, k_min=0.02, k_max=0.10, dk=float(k[1] - k[0]))
    from functools import partial
    theory_fn = partial(joint_fn, k=k, triangles=triangles)
    n_cosmo = sum(cosmo.param_sizes)
    return theory_fn, packed, joint_survey_keys, n_cosmo, len(z_bins)


@needs_emulator
def test_exact_reconstruction_non_c1_columns(pipeline):
    from jaxptpolypol.marginal_likelihood import (
        LIN_SURVEY_KEYS, make_marginal_templates, split_marginal_indices)
    theory_fn, packed, survey_keys, n_cosmo, n_bins = pipeline
    split = split_marginal_indices(
        n_cosmo_params=n_cosmo, survey_keys=survey_keys, n_bins=n_bins,
        fixed_cosmo=(5, 6, 7, 8),
        fixed_survey_keys={('shared', 'k_nl', None), ('shared', 'ndens', None)})
    templates = make_marginal_templates(theory_fn, split.lin_idx)
    m0, M = templates(packed)

    # Non-trivial theta_lin: fiducial values plus an aggressive perturbation,
    # with every c1 slot held at 0 (c1 columns tested separately).
    lin_fid = packed[jnp.array(split.lin_idx)]
    bump = jnp.array([1.0 + 0.37 * (i % 5) for i in range(split.n_lin)])
    theta_lin = lin_fid * bump + 0.61
    c1_mask = jnp.array([key == ('bk', 'ctr', 'c1') for (_b, key) in split.lin_keys])
    theta_lin = jnp.where(c1_mask, 0.0, theta_lin)

    t_full = theory_fn(packed.at[jnp.array(split.lin_idx)].set(theta_lin))
    recon = m0 + M @ theta_lin
    scale = jnp.maximum(jnp.abs(t_full), 1.0)
    np.testing.assert_allclose(
        np.asarray((t_full - recon) / scale), 0.0, atol=1e-10)


@needs_emulator
def test_c1_residual_is_exactly_quadratic(pipeline):
    from jaxptpolypol.marginal_likelihood import (
        make_marginal_templates, split_marginal_indices)
    theory_fn, packed, survey_keys, n_cosmo, n_bins = pipeline
    split = split_marginal_indices(
        n_cosmo_params=n_cosmo, survey_keys=survey_keys, n_bins=n_bins,
        fixed_cosmo=(5, 6, 7, 8),
        fixed_survey_keys={('shared', 'k_nl', None), ('shared', 'ndens', None)})
    templates = make_marginal_templates(theory_fn, split.lin_idx)
    m0, M = templates(packed)
    lin_idx_arr = jnp.array(split.lin_idx)
    c1_pos = [i for i, (_b, key) in enumerate(split.lin_keys)
              if key == ('bk', 'ctr', 'c1')]

    def residual_norm(c1_value):
        theta_lin = jnp.zeros(split.n_lin).at[jnp.array(c1_pos)].set(c1_value)
        t_full = theory_fn(packed.at[lin_idx_arr].set(theta_lin))
        return float(jnp.linalg.norm(t_full - (m0 + M @ theta_lin)))

    r1, r2 = residual_norm(0.4), residual_norm(0.8)
    assert r1 > 0.0                                    # curvature is real
    np.testing.assert_allclose(r2 / r1, 4.0, rtol=1e-6)  # exact quadratic law


@needs_emulator
def test_marginal_curvature_equals_fisher_schur(pipeline):
    from jaxptpolypol.marginal_likelihood import (
        make_constant_prior_fns, make_marginal_log_posterior,
        split_marginal_indices)
    from jaxptpolypol.sampler import make_full_params_fn

    theory_fn, packed, survey_keys, n_cosmo, n_bins = pipeline
    split = split_marginal_indices(
        n_cosmo_params=n_cosmo, survey_keys=survey_keys, n_bins=n_bins,
        fixed_cosmo=(5, 6, 7, 8),
        fixed_survey_keys={('shared', 'k_nl', None), ('shared', 'ndens', None)})

    data = theory_fn(packed)                      # noiseless mock
    n_data = data.shape[0]
    cov_inv = jnp.eye(n_data)                     # identity covariance suffices for the identity
    sigma_p = jnp.full(split.n_lin, 1.0)
    mu_p = packed[jnp.array(split.lin_idx)]

    prior_mean_fn, prior_sigma_fn = make_constant_prior_fns(mu_p, sigma_p)
    log_post = make_marginal_log_posterior(
        theory_fn=theory_fn, data=data, cov_inv=cov_inv, lin_idx=split.lin_idx,
        prior_mean_fn=prior_mean_fn, prior_sigma_fn=prior_sigma_fn,
        log_prior_nl_fn=lambda t: 0.0, to_physical=lambda x: x,
        full_params_fn=make_full_params_fn(packed, split.nl_idx),
        include_logdet=False)
    H = jax.hessian(log_post)(packed[jnp.array(split.nl_idx)])

    # Full-space GN Fisher over [nl | lin], prior on lin slots, Schur out lin.
    jac = jax.jacfwd(theory_fn)(packed)
    varied = list(split.nl_idx) + list(split.lin_idx)
    J = jac[:, jnp.array(varied)]
    F = np.array(J.T @ cov_inv @ J)   # writable copy (np.asarray of a jax array is read-only)
    n_nl = split.n_nl
    F[n_nl:, n_nl:] += np.diag(1.0 / np.asarray(sigma_p) ** 2)
    schur = F[:n_nl, :n_nl] - F[:n_nl, n_nl:] @ np.linalg.solve(
        F[n_nl:, n_nl:], F[n_nl:, :n_nl])
    scale = np.abs(schur).max()
    np.testing.assert_allclose(np.asarray(-H) / scale, schur / scale, atol=1e-8)
