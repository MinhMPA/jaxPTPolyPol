"""Single source of truth for the production Stream config/assembly.

Used by the three Stream validation/chain drivers -- ``taylor_surrogate_
validation.py``, ``damh_exact_chain_lcdm.py`` and ``desi_prior_validation.py`` --
which previously carried byte-for-byte copies of the same production
configuration and theory/BAO assembly (each a verbatim mirror of
``mcmc_joint_PFS_BAO_BBN_ns_LCDM.ipynb`` and of ``build_taylor_templates_lcdm.py``).

This module collects ONLY the clearly-duplicated, byte-identical pieces:

- the production config constants (fiducial cosmology, redshift bins, k-grid,
  quadrature, emulator path, the ``META`` config stamp, ``SHARED_KEYS``);
- template/whitening loading (:func:`load_templates_and_whitening`);
- theory/BAO assembly (:func:`build_fiducial_surveys`, :func:`build_split`,
  :func:`build_kgrid_and_blocks`, :func:`build_bao`).

Each helper reproduces the original inline block exactly -- same JAX ops in the
same order -- so every assembled array is bit-identical to the pre-extraction
value; this is behavior-preserving refactoring, not a logic change. Per-script
unique parts (RSS watchdogs, loaded-artifact subsets, exact/surrogate posterior
construction, gate logic, output filenames, seeds, config-drift tripwires) stay
in the individual scripts.

``jax`` is only used at CALL time (no array ops at import), so a caller that has
already enabled float64 gets the production dtype; import order is irrelevant.
"""

import json
import sys

import numpy as np

import jax.numpy as jnp

from ps_1loop_jax import background as bg

from jaxptpolypol import (
    load_taylor_templates,
    split_marginal_indices,
)
from jaxptpolypol.bao import load_desi_dr2, make_bao_theory_fn
from jaxptpolypol.params import CosmoParams, FullShapeSurveyParams
from jaxptpolypol.theory import build_bispectrum_triangles_from_k_grid

# ---------------------------------------------------------------------------
# Production config constants -- copied VERBATIM from
# mcmc_joint_PFS_BAO_BBN_ns_LCDM.ipynb (cell "Configuration (mirrors the Fisher
# notebook)"), in lockstep with build_taylor_templates_lcdm.py. The META guard +
# the packed/split/BAO tripwires in each script enforce that lockstep.
# ---------------------------------------------------------------------------

FIDUCIAL = {
    'ombh2': 0.02242, 'omch2': 0.11933, 'logA': 3.047,
    'ns': 0.9665, 'h': 0.6766, 'tau': 0.0561,
}
MNU_FIXED = 0.06  # eV, not varied in LCDM

z_bins   = (0.7,  0.9,  1.1,  1.3,  1.5,  1.8,  2.2)
knl_bins = (0.52, 0.65, 0.82, 1.02, 1.29, 1.82, 2.88)
n_bar    = (3.06e-4, 9.61e-4, 9.75e-4, 6.54e-4, 3.40e-4, 2.02e-4, 3.51e-4)
n_zbins  = len(z_bins)

K_PK_MIN, K_PK_MAX, N_K = 0.02, 0.20, 37
K_BK_MIN, K_BK_MAX = 0.02, 0.08
K_NL_RSD = 0.45
NUM_MU = NUM_PHI = 65
N_GL = 16
BACKGROUND_MODE = 'direct'

PFS_EMULATOR = '/Users/nguyenmn/cosmopower-jax-for-pfs/cosmology/jense2024/jense_2023_camb_lcdm/networks/jense_2023_camb_lcdm_Pk_lin.npz'
DEFAULT_BAO_DATA_DIR = "../../ext_data/bao_data/desi_bao_dr2"  # chdir-sensitive

SHARED_KEYS = ('ombh2', 'omch2', 'logA', 'ns', 'h')   # sampled cosmo (nl block)

META = {
    "n_bins": 7, "n_k": 37, "n_tri": 264, "n_gl": 16,
    "num_mu": 65, "num_phi": 65,
    "k_min": 0.02, "k_max": 0.20, "k_bk_max": 0.08, "k_nl_rsd": 0.45,
    "order2_m0": True,
}


# ---------------------------------------------------------------------------
# Template/whitening loading (strict meta guards).
# ---------------------------------------------------------------------------

def load_templates_and_whitening(templates_path, whitening_path,
                                 expect_meta=META):
    """Load Taylor templates + the whitening npz with the strict meta guards.

    Returns ``(tt, wz)`` where ``tt`` is the :class:`TaylorTemplates` (loaded
    with ``expect_meta``, the stale-config guard) and ``wz`` is the open
    ``np.load`` handle on the whitening file. Exits (``sys.exit``) if the
    whitening stamp does not match ``expect_meta`` -- identical behaviour and
    message to the pre-extraction inline block.
    """
    tt = load_taylor_templates(templates_path, expect_meta=expect_meta)
    wz = np.load(whitening_path)
    stored_meta = json.loads(str(wz["meta"].item()))
    if stored_meta != expect_meta:
        sys.exit(f"Whitening npz meta mismatch:\nstored   {stored_meta}\n"
                 f"expected {expect_meta}")
    return tt, wz


# ---------------------------------------------------------------------------
# Theory/BAO assembly.
# ---------------------------------------------------------------------------

def build_fiducial_surveys():
    """Rebuild the fiducial cosmology + per-bin survey pytrees.

    Returns ``(cosmo_dict, cosmo, surveys, joint_survey_keys)`` -- verbatim from
    the notebook cell "Emulator, models, fiducial parameters" (bias/counterterm
    fiducials arXiv:1907.06666). The theory statics themselves (emulator, models)
    cannot be serialized and are rebuilt by each caller.
    """
    cosmo_dict = {
        'ombh2': FIDUCIAL['ombh2'], 'omch2': FIDUCIAL['omch2'],
        'logA':  FIDUCIAL['logA'],  'ns':    FIDUCIAL['ns'], 'h': FIDUCIAL['h'],
        'z': 0.7, 'A_b': 3.13, 'eta_b': 0.603, 'logT_AGN': 7.8,
    }
    cosmo = CosmoParams(cosmo_dict)

    def b1z(z): return 0.9 + 0.4 * z
    def b2z(z): return -0.704 - 0.208 * z + 0.183 * z**2 - 0.00771 * z**3
    def bG2z(z): return -(2. / 7.) * (b1z(z) - 1.)
    def bGamma3z(z): return (23. / 42.) * (b1z(z) - 1.)
    def Dplusz(z):
        return float(bg.growth_factor(
            cosmo_dict['ombh2'], cosmo_dict['omch2'], cosmo_dict['h'], z,
            mnu=MNU_FIXED))
    def c0z(z): return 25. * Dplusz(z)**2
    def c2z(z): return 25. * Dplusz(z)**2
    def c4z(z): return Dplusz(z)**2

    surveys = []
    for z, knl, nd in zip(z_bins, knl_bins, n_bar):
        surveys.append(FullShapeSurveyParams(
            shared={'bias': {'b1': b1z(z), 'b2': b2z(z), 'bG2': bG2z(z),
                             'bGamma3': bGamma3z(z)},
                    'stoch': {'P_shot': 1.0}, 'k_nl': knl, 'ndens': nd},
            pk={'ctr': {'c0': c0z(z), 'c2': c2z(z), 'c4': c4z(z),
                        'cfog': knl**(-4)},
                'stoch': {'a0': 0., 'a2': 0.}},
            bk={'ctr': {'c1': 0.0}, 'stoch': {'B_shot': 1.0, 'A_shot': 1.0}},
        ))
    joint_survey_keys = surveys[0].joint_param_keys
    return cosmo_dict, cosmo, surveys, joint_survey_keys


def build_split(n_cosmo_params, joint_survey_keys):
    """The production sampled/marginalized index split (fixed_cosmo [5,6,7,8],
    k_nl + ndens fixed) -- notebook cell "Varied block ...", verbatim."""
    return split_marginal_indices(
        n_cosmo_params=n_cosmo_params, survey_keys=joint_survey_keys,
        n_bins=n_zbins, fixed_cosmo=[5, 6, 7, 8],
        fixed_survey_keys={('shared', 'k_nl', None), ('shared', 'ndens', None)})


def build_kgrid_and_blocks():
    """Build the P k-grid + B triangles and the per-bin data blocks.

    Returns ``(k, dk, triangles, block_len, bin_blocks)`` -- verbatim from the
    notebook, identical across the three scripts (``triangle_dk`` is discarded by
    all of them).
    """
    k = jnp.linspace(K_PK_MIN, K_PK_MAX, N_K)
    dk = float(k[1] - k[0])
    triangles, _triangle_dk = build_bispectrum_triangles_from_k_grid(
        k, k_min=K_BK_MIN, k_max=K_BK_MAX, dk=dk)
    block_len = 3 * int(k.shape[0]) + int(triangles.shape[0])
    bin_blocks = [slice(b * block_len, (b + 1) * block_len)
                  for b in range(n_zbins)]
    return k, dk, triangles, block_len, bin_blocks


def build_bao(bao_data_dir, cosmo, fiducial_cosmo, bao_fid):
    """Assemble the DESI DR2 BAO theory fn and check the fiducial tripwire.

    Returns ``(bao_dr2, bao_theory_fn)``. ``fiducial_cosmo`` is the packed
    cosmology sub-vector ``packed_params[:n_cosmo_params]`` and ``bao_fid`` the
    stored fiducial BAO vector; a recomputed-vs-stored disagreement beyond 1e-10
    exits (``sys.exit``), identical to the pre-extraction inline block.
    """
    bao_dr2 = load_desi_dr2("all", data_dir=bao_data_dir)
    bao_theory_fn = make_bao_theory_fn(
        bao_dr2, cosmo_keys=cosmo.param_keys, cosmo_sizes=cosmo.param_sizes,
        mnu_fixed=MNU_FIXED)
    if not np.allclose(np.asarray(bao_theory_fn(fiducial_cosmo)),
                       np.asarray(bao_fid), rtol=1e-10, atol=0.0):
        sys.exit("ABORT: recomputed BAO fiducial vector differs from the stored "
                 "one beyond 1e-10.")
    return bao_dr2, bao_theory_fn
