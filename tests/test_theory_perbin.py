"""Per-bin theory evaluation must reproduce the monolithic joint theory exactly."""
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
def cfg():
    """Small 2-bin config; mirrors tests/test_marginal_pipeline.py's fixture."""
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
    return kwargs, packed, k, triangles, len(z_bins)


@needs_emulator
def test_perbin_blocks_concatenate_to_joint(cfg):
    from jaxptpolypol.theory import make_joint_pk_bk_fn, make_joint_pk_bk_bin_fn
    kwargs, packed, k, triangles, n_bins = cfg

    joint = make_joint_pk_bk_fn(**kwargs)
    full = joint(packed, k=k, triangles=triangles)

    blocks = []
    for b in range(n_bins):
        bin_fn = make_joint_pk_bk_bin_fn(bin_index=b, **kwargs)
        blk = bin_fn(packed, k=k, triangles=triangles)
        assert blk.shape == (3 * k.shape[0] + triangles.shape[0],)
        blocks.append(blk)
    recon = jnp.concatenate(blocks)
    assert recon.shape == full.shape
    np.testing.assert_allclose(np.asarray(recon), np.asarray(full), rtol=1e-12)


@needs_emulator
def test_perbin_block_is_bin_specific(cfg):
    """Guards the slot-vs-global index pairing: bin 1 must NOT equal bin 0."""
    from jaxptpolypol.theory import make_joint_pk_bk_bin_fn
    kwargs, packed, k, triangles, _ = cfg
    b0 = make_joint_pk_bk_bin_fn(bin_index=0, **kwargs)(packed, k=k, triangles=triangles)
    b1 = make_joint_pk_bk_bin_fn(bin_index=1, **kwargs)(packed, k=k, triangles=triangles)
    assert not np.allclose(np.asarray(b0), np.asarray(b1), rtol=1e-6)
