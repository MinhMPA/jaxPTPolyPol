"""Shared pytest fixtures for the marginal-likelihood test suites.

The 2-bin small config below is copied verbatim from
``tests/test_marginal_perbin.py`` (its third+ consumer). pytest resolves this
conftest ``cfg`` fixture for new test files while the older suites keep their own
shadowing local copies, so moving it here changes no existing behaviour.

Heavy-test gating
-----------------
A bare ``pytest tests/`` used to stack the fixtures of the memory-heavy modules
(``test_marginal_pipeline``, ``test_marginal_perbin``, ``test_theory_perbin``,
``test_marginal_taylor_pipeline``) to ~85 GB in a single process — the per-bin
theory/marginal graphs and their compiled evals are not freed between modules,
so their peaks accumulate. To keep the default run safe, every item in those
modules is marked ``heavy`` and **deselected unless ``--run-heavy`` is passed**.
Run the heavy files individually (``pytest tests/test_theory_perbin.py``) or opt
in deliberately with ``pytest tests/ --run-heavy``.
"""
import os
import pathlib

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

# Modules whose fixtures/compiled evals accumulate to ~85 GB when collected in
# one process. Gated behind --run-heavy (see the module docstring).
HEAVY_MODULES = {
    "test_marginal_pipeline",
    "test_marginal_perbin",
    "test_theory_perbin",
    "test_marginal_taylor_pipeline",
}


def pytest_addoption(parser):
    parser.addoption(
        "--run-heavy",
        action="store_true",
        default=False,
        help="Run the memory-heavy marginal/theory per-bin suites "
             "(~85 GB stacked); otherwise they are deselected.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "heavy: memory-heavy suite; deselected unless --run-heavy is passed.",
    )


def pytest_collection_modifyitems(config, items):
    run_heavy = config.getoption("--run-heavy")
    heavy_marker = pytest.mark.heavy
    selected, deselected = [], []
    for item in items:
        if item.path.stem in HEAVY_MODULES:
            item.add_marker(heavy_marker)
            if not run_heavy:
                deselected.append(item)
                continue
        selected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


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
