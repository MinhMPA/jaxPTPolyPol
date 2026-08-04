"""Single source of truth for the production Stream config/assembly.

Used by the four Stream validation/chain drivers -- ``taylor_surrogate_
validation.py``, ``damh_exact_chain_lcdm.py``, ``desi_prior_validation.py`` and
``tier3_c1_validation.py`` -- which previously carried byte-for-byte copies of
the same production
configuration and theory/BAO assembly (each a verbatim mirror of
``mcmc_joint_PFS_BAO_BBN_ns_LCDM.ipynb`` and of ``build_taylor_templates_lcdm.py``).

This module collects ONLY the clearly-duplicated, byte-identical pieces:

- the production config constants (fiducial cosmology, redshift/volume bins,
  k-grid, quadrature, emulator path, the ``META`` config stamp, ``SHARED_KEYS``)
  and the derived ``THEORY_CONFIG_HASH`` -- ``build_taylor_templates_lcdm.py``
  imports these too, so the stamped and the expected config cannot diverge;
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

import hashlib
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
from jaxptpolypol.marginal_taylor import check_meta
from jaxptpolypol.params import CosmoParams, FullShapeSurveyParams
from jaxptpolypol.theory import build_bispectrum_triangles_from_k_grid

# ---------------------------------------------------------------------------
# Production config constants -- copied VERBATIM from
# mcmc_joint_PFS_BAO_BBN_ns_LCDM.ipynb (cell "Configuration (mirrors the Fisher
# notebook)"). build_taylor_templates_lcdm.py IMPORTS them from here rather than
# keeping its own copies, so producer and consumer cannot drift; the META guard +
# the packed/split/BAO tripwires in each script enforce the rest of the lockstep.
# ---------------------------------------------------------------------------

FIDUCIAL = {
    'ombh2': 0.02242, 'omch2': 0.11933, 'logA': 3.047,
    'ns': 0.9665, 'h': 0.6766, 'tau': 0.0561,
}
MNU_FIXED = 0.06  # eV, not varied in LCDM

z_bins   = (0.7,  0.9,  1.1,  1.3,  1.5,  1.8,  2.2)
V_bins   = tuple(v * 1000.**3 for v in (0.59, 0.79, 0.96, 1.09, 1.19, 2.58, 2.71))
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
FIXED_COSMO = (5, 6, 7, 8)  # packed-cosmo indices held fixed (z, A_b, eta_b, logT_AGN)

META = {
    "n_bins": 7, "n_k": 37, "n_tri": 264, "n_gl": 16,
    "num_mu": 65, "num_phi": 65,
    "k_min": 0.02, "k_max": 0.20, "k_bk_max": 0.08, "k_nl_rsd": 0.45,
    "order2_m0": True,
}

#: The theory/grid/basis configuration the Taylor tensors are a function of, and
#: its sha256. Stored as a NAMED dict (self-documenting) and hashed
#: order-insensitively via ``repr(sorted(...items()))``, so the serialization is
#: canonical regardless of insertion order. ``build_taylor_templates_lcdm.py``
#: reaches :data:`THEORY_CONFIG_HASH` (through :func:`template_meta_for`) to STAMP
#: its templates npz, and the loaders below use it to EXPECT, so the two hashes
#: are byte-identical by construction and ANY change to a hashed entry invalidates
#: every cache built before it.
#:
#: Coverage -- everything that determines TEMPLATE validity:
#:   * survey/grid geometry -- ``V_bins``, ``n_bar``, ``knl_bins``, ``z_bins``, the
#:     P/B k-grid ``(K_PK_MIN, K_PK_MAX, N_K, K_BK_MIN, K_BK_MAX)`` and
#:     ``K_NL_RSD``;
#:   * the COSMOLOGY BASIS -- ``(SHARED_KEYS, FIXED_COSMO, MNU_FIXED)``: which cosmo
#:     params are sampled vs fixed, plus the fixed neutrino mass. This is the
#:     primary LCDM-vs-nuLCDM discriminator -- a nuLCDM run adds ``'mnu'`` to the
#:     sampled basis, so its hash necessarily differs and its templates can never
#:     be loaded as LCDM (or vice versa) even at byte-identical survey config;
#:   * the EMULATOR -- the FULL ``PFS_EMULATOR`` path (the LCDM linear-Pk network).
#:     An mnu emulator lives at a different path and produces different templates;
#:     the full path (not the basename) is hashed to preclude any basename
#:     collision between sibling networks;
#:   * the FIDUCIAL cosmology values -- ``sorted(FIDUCIAL.items())``: theta0 (the
#:     Taylor expansion centre) and the mock data vector both depend on them;
#:   * the model/discretization flags DEFINED IN this module that feed the theory
#:     -- ``N_GL``, ``NUM_MU``, ``NUM_PHI``, ``BACKGROUND_MODE``.
#:
#: ``V_bins`` feeds only the covariance, not the templates, but it is KEPT here
#: deliberately: the whitening npz's Gaussian covariance depends on it, and the
#: whitening and templates stamps share this one hash, so hashing ``V_bins`` also
#: guards the whitening. Dropping it (as an early review suggested) would weaken
#: that guard.
#:
#: Deliberately NOT hashed -- the invariant model flags that live in
#: ``build_taylor_templates_lcdm.py`` rather than here: ``do_irres=True``,
#: ``do_AP=True``, ``ap=True`` (genuine template determinants, but hard-coded
#: constants that nobody varies) and ``BB_POWER_MODEL='kaiser'`` (a COVARIANCE
#: choice -- it feeds the whitening, not the templates). None of them discriminate
#: LCDM from nuLCDM, and the scope here is the single-source-of-truth constants
#: defined in THIS module; pulling a build-script literal in would force a
#: non-surgical move or duplicate a magic value that could silently drift.
#: Templates are prior-independent, so no prior identifier belongs here either.
_THEORY_CONFIG = {
    "V_bins": V_bins,
    "n_bar": n_bar,
    "knl_bins": knl_bins,
    "z_bins": z_bins,
    "k_grid": (K_PK_MIN, K_PK_MAX, N_K, K_BK_MIN, K_BK_MAX),
    "k_nl_rsd": K_NL_RSD,
    "cosmo_basis": (SHARED_KEYS, FIXED_COSMO, MNU_FIXED),
    "emulator": PFS_EMULATOR,
    "fiducial": tuple(sorted(FIDUCIAL.items())),
    "n_gl": N_GL,
    "num_mu": NUM_MU,
    "num_phi": NUM_PHI,
    "background_mode": BACKGROUND_MODE,
}
THEORY_CONFIG_HASH = hashlib.sha256(
    repr(sorted(_THEORY_CONFIG.items())).encode()).hexdigest()

#: c1 treatments: 'marginalized' (the base LCDM split, c1 in theta_lin) and
#: 'sampled' (the Tier-3 split, c1 moved into theta_NL). See CONTEXT.md's c1
#: section and build_taylor_templates_lcdm.py --c1-sampled.
C1_TREATMENTS = ("marginalized", "sampled")


def meta_for(treatment):
    """Config-stamp META for a c1 treatment, ``{**META, 'c1_treatment': treatment}``.

    ``build_taylor_templates_lcdm.py`` stamps this on the WHITENING npz (plus
    the prior identifiers in marginalized mode); the templates npz gets the
    richer :func:`template_meta_for` stamp.

    This is likewise the WHITENING-side expectation; the templates side also
    expects the theory identifiers (:func:`template_meta_for`).
    :func:`load_templates_and_whitening` builds both from its ``treatment``
    argument, so ``c1_treatment`` is now VERIFIED by default -- under the guard's
    semantics (:func:`jaxptpolypol.marginal_taylor.compare_meta`) an identifier
    the caller does not name is not checked, and a marginalized/sampled mix-up
    is precisely what must not slip through. Both stamps that exist on disk load
    either way -- an old cache lacking ``c1_treatment`` warns (backward compat)
    and a newer cache carrying it matches -- so naming it costs no
    compatibility.
    """
    if treatment not in C1_TREATMENTS:
        raise ValueError(
            f"unknown c1 treatment {treatment!r}; expected one of {C1_TREATMENTS}")
    return {**META, "c1_treatment": treatment}


def template_meta_for(treatment):
    """Full expected TEMPLATES-npz stamp: :func:`meta_for` + the theory config.

    This is exactly what ``build_taylor_templates_lcdm.py`` stamps on the
    templates npz (that script imports this function), so a template cache built
    from the current constants matches it key-for-key with no warnings, and a
    cache built from ANY other theory config fails on ``theory_config_hash``.

    Whitening stamps deliberately do NOT carry the theory identifiers (they
    carry the PRIOR ones instead), so the whitening side keeps expecting the
    plain :func:`meta_for` stamp -- see :func:`load_templates_and_whitening`.
    """
    return {**meta_for(treatment),
            "theory_config_hash": THEORY_CONFIG_HASH,
            "z_bins": str(z_bins), "knl_bins": str(knl_bins),
            "n_bar": str(n_bar), "V_bins": str(V_bins)}


# ---------------------------------------------------------------------------
# Template/whitening loading (strict meta guards).
# ---------------------------------------------------------------------------

def load_templates_and_whitening(templates_path, whitening_path, *,
                                 treatment="marginalized",
                                 expect_meta=None, expect_template_meta=None):
    """Load Taylor templates + the whitening npz with the LIVE config guards.

    Returns ``(tt, wz)``: the :class:`TaylorTemplates` and the open ``np.load``
    handle on the whitening file. A stale templates npz raises ``ValueError``; a
    stale whitening npz exits (``sys.exit``).

    Expectations
    ------------
    By DEFAULT the two files are held to different (correct) standards, both
    derived from ``treatment`` (``'marginalized'`` | ``'sampled'``):

    * templates -> :func:`template_meta_for` -- the 11 grid keys, ``c1_treatment``
      AND the theory identifiers (``theory_config_hash``, ``z_bins``,
      ``knl_bins``, ``n_bar``, ``V_bins``);
    * whitening -> :func:`meta_for` -- the 11 grid keys and ``c1_treatment``;
      theory identifiers are never stamped there, prior ones are (and those are
      informational, since no consumer expects them).

    Naming ``theory_config_hash`` and ``c1_treatment`` by default is the whole
    point: they are the two identifiers that distinguish "templates for THIS
    theory config / THIS split" from silently wrong ones, and an expectation
    that does not name a key does not check it.

    Chosen rule: ENFORCE-IF-PRESENT
    -------------------------------
    Per :func:`jaxptpolypol.marginal_taylor.compare_meta`, the newer identifiers
    are all in ``_BACKWARD_COMPAT_META_KEYS``, so the three cases are:

    1. stored value differs from expected -> **hard failure** (genuine config
       drift, or a marginalized/sampled cache mix-up);
    2. stored stamp LACKS the key -> warning, loads anyway (the on-disk caches
       predate these keys; rebuilding stamps them);
    3. stored stamp carries a key the expectation does not name (e.g. the
       whitening npz's ``prior_spec``/``cosmo_priors``) -> warning, not
       staleness. A rebuilt cache with extra keys must still load.

    So the guard cannot fire on an old cache and cannot be silent on a real
    drift. Pass ``expect_meta`` / ``expect_template_meta`` to override either
    side explicitly.
    """
    if expect_template_meta is None:
        expect_template_meta = template_meta_for(treatment)
    if expect_meta is None:
        expect_meta = meta_for(treatment)
    tt = load_taylor_templates(templates_path, expect_meta=expect_template_meta)
    wz = np.load(whitening_path)
    stored_meta = json.loads(str(wz["meta"].item()))
    try:
        check_meta(stored_meta, expect_meta, what="whitening npz")
    except ValueError as err:
        sys.exit(str(err))
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
        n_bins=n_zbins, fixed_cosmo=list(FIXED_COSMO),
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
