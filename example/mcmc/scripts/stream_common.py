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
import pathlib
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
N_TRI = 264                # closed triangles on the [K_BK_MIN, K_BK_MAX] grid
BACKGROUND_MODE = 'direct'
#: Bispectrum IR resummation (flipped False -> True 2026-08-23, jaxptpolypol
#: commit 7304e6a). THE single source of truth for the flag: it is hashed into
#: _THEORY_CONFIG below AND passed explicitly (do_irres=BK_DO_IRRES) by every
#: template/chain producer, so the stamped value IS the built value by
#: construction -- never rely on the library default to keep them in sync.
BK_DO_IRRES = True

PFS_EMULATOR = '/Users/nguyenmn/cosmopower-jax-for-pfs/cosmology/jense2024/jense_2023_camb_lcdm/networks/jense_2023_camb_lcdm_Pk_lin.npz'
DEFAULT_BAO_DATA_DIR = "../../ext_data/bao_data/desi_bao_dr2"  # chdir-sensitive

SHARED_KEYS = ('ombh2', 'omch2', 'logA', 'ns', 'h')   # sampled cosmo (nl block)
FIXED_COSMO = (5, 6, 7, 8)  # packed-cosmo indices held fixed (z, A_b, eta_b, logT_AGN)

META = {
    "n_bins": n_zbins, "n_k": N_K, "n_tri": N_TRI, "n_gl": N_GL,
    "num_mu": NUM_MU, "num_phi": NUM_PHI,
    "k_min": K_PK_MIN, "k_max": K_PK_MAX, "k_bk_max": K_BK_MAX,
    "k_nl_rsd": K_NL_RSD,
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
#: ``bk_do_irres`` IS hashed (added 2026-08-23): the bispectrum IR-resummation
#: flag was flipped False -> True, which falsified the "nobody varies it"
#: premise that justified excluding it. It moves the B entries of the theory
#: vector, so a cache built under the old value must not load. PRECISION on how
#: that rejection actually happens (audited 2026-08-23): the library guard is
#: ENFORCE-IF-PRESENT (``theory_config_hash`` is in
#: ``marginal_taylor._BACKWARD_COMPAT_META_KEYS``), so hashing alone hard-fails
#: only caches that ALREADY STAMP the hash (the nuLCDM pair). Caches predating
#: the key -- the committed LCDM/_c1s pairs -- would warn-and-load with stale
#: non-resummed B templates. :func:`load_templates_and_whitening` therefore
#: ESCALATES a missing templates-side hash to a hard exit; the notebooks carry
#: the same explicit check. Rebuild stale caches with
#: build_taylor_templates_lcdm.py.
#:
#: Deliberately NOT hashed -- the invariant model flags that live in
#: ``build_taylor_templates_lcdm.py`` rather than here: ``do_irres=True`` (the
#: POWER-SPECTRUM flag), ``do_AP=True``, ``ap=True`` (genuine template
#: determinants, but hard-coded constants that nobody varies) and ``BB_POWER_MODEL='kaiser'`` (a COVARIANCE
#: choice -- it feeds the whitening, not the templates). None of them discriminate
#: LCDM from nuLCDM, and the scope here is the single-source-of-truth constants
#: defined in THIS module; pulling a build-script literal in would force a
#: non-surgical move or duplicate a magic value that could silently drift.
#: Templates are prior-independent, so no prior identifier belongs here either.
_THEORY_CONFIG = {
    "bk_do_irres": BK_DO_IRRES,
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

# ---------------------------------------------------------------------------
# nuLCDM config block (ADDITIVE -- every LCDM constant above is byte-unchanged).
# The nuLCDM run reuses the whole LCDM production config with mnu ADDED to the
# sampled cosmology basis; only the cosmology-dependent hash inputs (cosmo basis,
# linear-Pk emulator, fiducial) differ, so a nuLCDM template cache can never be
# confused with the LCDM one (NULCDM_THEORY_CONFIG_HASH != THEORY_CONFIG_HASH).
# ---------------------------------------------------------------------------

#: The mnu linear-Pk network (jense_2023_camb_mnu), verified from the nuLCDM
#: Fisher notebooks (example/fisher/fisher_joint_PFS_BAO_BBN_ns_nuLCDM.ipynb cell
#: "Emulator and models (mnu variant)": ``PFS_EMULATOR = '.../jense_2023_camb_mnu/
#: networks/jense_2023_camb_mnu_Pk_lin.npz'``).
NULCDM_EMULATOR = '/Users/nguyenmn/cosmopower-jax-for-pfs/cosmology/jense2024/jense_2023_camb_mnu/networks/jense_2023_camb_mnu_Pk_lin.npz'

#: nuLCDM sampled cosmo basis: the LCDM :data:`SHARED_KEYS` plus ``'mnu'``.
#: Per fisher_joint_PFS_BAO_BBN_ns_nuLCDM.ipynb cell "Emulator and models (mnu
#: variant)", the packed cosmo dict places mnu LAST so the fixed-cosmo indices
#: are UNCHANGED from LCDM (quoting the notebook construction verbatim)::
#:
#:     cosmo_dict = {
#:         'ombh2': ..., 'omch2': ..., 'logA': ..., 'ns': ..., 'h': ...,  # 0-4 sampled core
#:         'z': 0.7, 'A_b': 3.13, 'eta_b': 0.603, 'logT_AGN': 7.8,        # 5-8 FIXED
#:         'mnu': FIDUCIAL['mnu'],  # keep LAST: fixed_cosmo=[5,6,7,8] indexes z/A_b/eta_b/logT_AGN
#:     }
#:
#: so the packed order is (ombh2, omch2, logA, ns, h, z, A_b, eta_b, logT_AGN, mnu):
#: a 10-key packed basis in which mnu is SAMPLED at index 9 and :data:`FIXED_COSMO`
#: == (5, 6, 7, 8) is REUSED byte-for-byte (notebook cell "Remove truly fixed
#: parameters": ``fixed_cosmo = [5, 6, 7, 8]``; varied cosmo = ombh2(0), omch2(1),
#: logA(2), ns(3), h(4), mnu(9) -> 6 params).
SHARED_KEYS_NU = ('ombh2', 'omch2', 'logA', 'ns', 'h', 'mnu')

#: nuLCDM fiducial = the LCDM :data:`FIDUCIAL` plus ``mnu`` at :data:`MNU_FIXED`
#: (0.06 eV), matching the notebook FIDUCIAL. Numerically the same mnu as the LCDM
#: fixed value, but here it is the Taylor-expansion centre of a SAMPLED parameter.
NULCDM_FIDUCIAL = {**FIDUCIAL, 'mnu': MNU_FIXED}

#: nuLCDM theory-config: the LCDM :data:`_THEORY_CONFIG` with ONLY the three
#: cosmology-dependent entries swapped -- ``cosmo_basis`` gains mnu, ``emulator``
#: is the mnu network, ``fiducial`` gains mnu. Every survey/grid/model entry is
#: SHARED by reference (``**_THEORY_CONFIG``), so the two dicts carry an identical
#: key set (locked in tests/test_stream_common_meta.py) and cannot silently drop a
#: hash input on one side only.
NULCDM_THEORY_CONFIG = {
    **_THEORY_CONFIG,
    "cosmo_basis": (SHARED_KEYS_NU, FIXED_COSMO, MNU_FIXED),
    "emulator": NULCDM_EMULATOR,
    "fiducial": tuple(sorted(NULCDM_FIDUCIAL.items())),
}
NULCDM_THEORY_CONFIG_HASH = hashlib.sha256(
    repr(sorted(NULCDM_THEORY_CONFIG.items())).encode()).hexdigest()

#: The cosmologies :func:`meta_for` / :func:`template_meta_for` can stamp.
_COSMOLOGIES = ("lcdm", "nulcdm")

#: c1 treatments: 'marginalized' (the base LCDM split, c1 in theta_lin) and
#: 'sampled' (the Tier-3 split, c1 moved into theta_NL). See theory.md: Why c1
#: sits in the linear block, and build_taylor_templates_lcdm.py --c1-sampled.
C1_TREATMENTS = ("marginalized", "sampled")


def meta_for(treatment, *, cosmology="lcdm"):
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

    ``cosmology`` (default ``"lcdm"``) selects the run family. The default is
    byte-identical to the pre-nuLCDM stamp -- it adds NO ``cosmology`` key, so the
    committed LCDM caches and the production tripwire are untouched. ``"nulcdm"``
    adds ``cosmology: "nulcdm"`` (and, in :func:`template_meta_for`, swaps in the
    nuLCDM theory-config hash) so the two families' caches are guard-distinct.
    """
    if treatment not in C1_TREATMENTS:
        raise ValueError(
            f"unknown c1 treatment {treatment!r}; expected one of {C1_TREATMENTS}")
    if cosmology not in _COSMOLOGIES:
        raise ValueError(
            f"unknown cosmology {cosmology!r}; expected one of {_COSMOLOGIES}")
    meta = {**META, "c1_treatment": treatment}
    if cosmology == "nulcdm":
        meta["cosmology"] = "nulcdm"   # lcdm adds nothing -> byte-identical default
    return meta


def template_meta_for(treatment, *, cosmology="lcdm"):
    """Full expected TEMPLATES-npz stamp: :func:`meta_for` + the theory config.

    This is exactly what ``build_taylor_templates_lcdm.py`` stamps on the
    templates npz (that script imports this function), so a template cache built
    from the current constants matches it key-for-key with no warnings, and a
    cache built from ANY other theory config fails on ``theory_config_hash``.

    Whitening stamps deliberately do NOT carry the theory identifiers (they
    carry the PRIOR ones instead), so the whitening side keeps expecting the
    plain :func:`meta_for` stamp -- see :func:`load_templates_and_whitening`.

    ``cosmology`` (default ``"lcdm"``) selects which theory-config hash is
    stamped: the LCDM default is byte-identical to the pre-nuLCDM stamp (no
    ``cosmology`` key, :data:`THEORY_CONFIG_HASH`), while ``"nulcdm"`` stamps
    :data:`NULCDM_THEORY_CONFIG_HASH` plus ``cosmology: "nulcdm"``.
    """
    hashes = {"lcdm": THEORY_CONFIG_HASH, "nulcdm": NULCDM_THEORY_CONFIG_HASH}
    if cosmology not in hashes:
        raise ValueError(
            f"unknown cosmology {cosmology!r}; expected one of {tuple(hashes)}")
    return {**meta_for(treatment, cosmology=cosmology),
            "theory_config_hash": hashes[cosmology],
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
    2. stored stamp LACKS the key -> warning at the library layer -- but THIS
       loader escalates a missing templates-side ``theory_config_hash`` to a
       hard exit (2026-08-23: pre-flip caches predate the key and carry
       non-IR-resummed B templates; other missing keys still warn-and-load);
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
    # ESCALATION (2026-08-23): the library rule above tolerates a stored stamp
    # that LACKS ``theory_config_hash`` (backward compat). After the bispectrum
    # IR-resummation flip that grace is exactly the silent-wrong-physics hole --
    # every pre-flip cache predates the key and carries non-resummed B
    # templates. If the expectation names the hash, the stored stamp must too.
    if "theory_config_hash" in expect_template_meta:
        _stored_t = tt.build_diagnostics["meta"]
        if "theory_config_hash" not in _stored_t:
            sys.exit(
                f"Stale Taylor templates: {templates_path} carries no "
                "theory_config_hash stamp, so it predates the current theory "
                "config (bispectrum IR resummation flipped ON 2026-08-23) and "
                "its B templates are NOT IR-resummed. Rebuild with "
                "build_taylor_templates_lcdm.py.")
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

def build_fiducial_surveys(cosmology="lcdm"):
    """Rebuild the fiducial cosmology + per-bin survey pytrees.

    Returns ``(cosmo_dict, cosmo, surveys, joint_survey_keys)`` -- verbatim from
    the notebook cell "Emulator, models, fiducial parameters" (bias/counterterm
    fiducials arXiv:1907.06666). The theory statics themselves (emulator, models)
    cannot be serialized and are rebuilt by each caller.

    ``cosmology`` (default ``"lcdm"``) selects the run family. The LCDM default is
    byte-identical to the pre-nuLCDM behaviour. ``"nulcdm"`` appends ``'mnu'`` LAST
    to ``cosmo_dict`` (at :data:`NULCDM_FIDUCIAL`'s value), so the packed cosmo
    order becomes (ombh2, omch2, logA, ns, h, z, A_b, eta_b, logT_AGN, mnu): mnu is
    SAMPLED at packed index 9 and :data:`FIXED_COSMO` == (5, 6, 7, 8) is unchanged
    (mirrors ``build_taylor_templates_lcdm.py`` and the nuLCDM notebook). The
    fiducial mnu equals :data:`MNU_FIXED` (0.06), so the bias/counterterm
    fiducials (which use ``mnu=MNU_FIXED``) are numerically identical.
    """
    if cosmology not in _COSMOLOGIES:
        raise ValueError(
            f"unknown cosmology {cosmology!r}; expected one of {_COSMOLOGIES}")
    cosmo_dict = {
        'ombh2': FIDUCIAL['ombh2'], 'omch2': FIDUCIAL['omch2'],
        'logA':  FIDUCIAL['logA'],  'ns':    FIDUCIAL['ns'], 'h': FIDUCIAL['h'],
        'z': 0.7, 'A_b': 3.13, 'eta_b': 0.603, 'logT_AGN': 7.8,
    }
    if cosmology == "nulcdm":
        cosmo_dict['mnu'] = NULCDM_FIDUCIAL['mnu']   # keep LAST (packed idx 9)
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


# ---------------------------------------------------------------------------
# CMB Fisher block (joint PFS+BAO+CMB+BBN forecasts, 2026-08-06 decisions).
# ADDITIVE -- every constant/function above is byte-unchanged.
#
# The CMB block is a fiducial-centered GAUSSIAN summary of the candl/clipy stack
# (Planck highl TTTEEE + lowl TT + lowl EE simall + Planck lensing + ACT DR6
# lensing), built once by ``build_cmb_fisher_block.py`` and loaded here. The
# shared basis and its ORDER are copied from the Fisher notebooks
# example/fisher/fisher_joint_PFS_BAO_CMB_{LCDM,nuLCDM}.ipynb cell 3
# (``SHARED_KEYS``); tau is the CMB-only direction that PFS/BAO cannot constrain,
# hence it sits LAST in the LCDM basis (and mnu after it in nuLCDM).
# ---------------------------------------------------------------------------

#: Cache directory for the mcmc artifacts, resolved from THIS file rather than
#: the cwd (the scripts' ``CACHE = pathlib.Path("cache")`` convention is
#: chdir-sensitive; ``b1sigma8_measure_report.py``'s ``HERE = parents[1]`` /
#: "cache" is the chdir-proof form reused here). parents[1] == example/mcmc.
CACHE_DIR = pathlib.Path(__file__).resolve().parents[1] / "cache"

#: Shared cosmology basis of the CMB Fisher block, LCDM. Verbatim from
#: fisher_joint_PFS_BAO_CMB_LCDM.ipynb cell 3 (``SHARED_KEYS``).
SHARED_KEYS_CMB_LCDM = ('ombh2', 'omch2', 'logA', 'ns', 'h', 'tau')

#: Shared cosmology basis, nuLCDM -- the LCDM basis plus ``mnu`` LAST. Verbatim
#: from fisher_joint_PFS_BAO_CMB_nuLCDM.ipynb cell 3 (``SHARED_KEYS``).
SHARED_KEYS_CMB_NULCDM = ('ombh2', 'omch2', 'logA', 'ns', 'h', 'tau', 'mnu')

#: Fiducial optical depth (Planck 2018 TT,TE,EE+lowE+lensing+BAO), the CMB-only
#: parameter absent from :data:`FIDUCIAL`'s sampled PFS basis but present in the
#: notebooks' unified ``FIDUCIAL`` dict -- same value, quoted from there.
TAU_FID = FIDUCIAL['tau']

#: Mossa et al. 2020 BBN WIDTH on ombh2. This is a sigma, NOT a center: the
#: forecast prior is ALWAYS centered on the fiducial ombh2 (fiducial-centered
#: policy), so no measured central value enters.
BBN_SIGMA_MOSSA = 0.00036

#: CONTENT-DERIVED fingerprint of the CMB Fisher block artifacts, pinned here
#: and HARD-REQUIRED by :func:`load_cmb_fisher_block`.
#:
#: ``theory_config_hash`` above fingerprints the PFS/BAO production config; it
#: says nothing about the CMB side. Everything that actually determines the CMB
#: block -- which Cl emulator networks, which Planck .clik data, which candl /
#: clipy / jax versions, which per-term method, which Gauss-Newton algorithm
#: revision, which shared-prior policy, which fiducial and basis -- was
#: previously unfingerprinted, so a block built against a different emulator
#: generation or a re-downloaded likelihood would load silently.
#:
#: The value is computed by ``build_cmb_fisher_block.py`` from FILE CONTENT
#: (sha256 of every emulator .npz and of every file under each .clik directory),
#: not from paths or mtimes, and stamped into the artifact META.
#:
#: A ``None`` pin means "no artifact of this cosmology may be loaded" -- the
#: loader refuses rather than falling back, so an unpinned build cannot be
#: consumed by accident. Re-pin from the build's ``[fingerprint]`` line whenever
#: the CMB inputs legitimately change.
CMB_CONFIG_HASH_LCDM = '97f8695acb8a05435fe5e7dc3ec9f923b5a453de78d74b27a3b137190f2d8417'
CMB_CONFIG_HASH_NULCDM = 'e89efa399fe355907dcd3f85cec47e72c799c694dce6e9d8f35adc93bb94d421'


def cmb_fisher_path(cosmology, cache_dir=None):
    """Path of the CMB Fisher block artifact for ``cosmology``.

    THE single place the artifact filename is constructed -- producer
    (``build_cmb_fisher_block.py``) and consumer (:func:`load_cmb_fisher_block`)
    both route through it, so a diagnostic/variant mode cannot tag one path and
    miss another (the 2026-08-04 output-path lesson).
    """
    if cosmology not in _COSMOLOGIES:
        raise ValueError(
            f"unknown cosmology {cosmology!r}; expected one of {_COSMOLOGIES}")
    base = pathlib.Path(cache_dir) if cache_dir is not None else CACHE_DIR
    return base / f"cmb_fisher_{cosmology}.npz"


#: Theory-config ERAS accepted by :func:`load_cmb_fisher_block` IN ADDITION to
#: the live hash. The CMB Fisher block depends only on the cosmo basis,
#: fiducial and CMB-side inputs (all separately guarded by CMB_CONFIG_HASH) --
#: NOT on the PFS-side bispectrum flag. Adding ``bk_do_irres`` to
#: _THEORY_CONFIG (2026-08-23) therefore changed the live hash without
#: invalidating the committed CMB artifacts, which stamp the pre-flip era
#: recorded here. Each entry must say WHY the era is CMB-equivalent; rebuild
#: the artifacts (build_cmb_fisher_block.py) to retire an entry.
_CMB_EQUIVALENT_THEORY_HASHES = {
    # pre bk_do_irres (2026-08-23): differs from live only by the PFS-side
    # bispectrum IR-resummation flag, which never enters the CMB physics.
    "lcdm": frozenset({"903aeb06e1cca1c17c7bd6f5166f4fa7039f6907371b53788d45cb71083917c3"}),
    "nulcdm": frozenset({"8f0f2e74332a4a80ffcede7edc471921fd6e8fcd0ed827717880ef03995adec6"}),
}


def load_cmb_fisher_block(cosmology, cache_dir=None):
    """Load the precomputed fiducial-centered CMB Fisher block with META guards.

    Returns ``{"F_shared", "fid_shared", "shared_keys", "sigma_tau", "meta"}``:
    the shared-basis Fisher matrix and its expansion centre (both jnp arrays),
    the basis key tuple, the marginalized sigma(tau) recorded at build time, and
    the full META dict.

    Four HARD guards, all ``ValueError`` (no backward-compat leniency -- unlike
    the template stamps there is no legacy CMB cache to stay compatible with,
    and each of these mismatches silently mis-assigns Fisher rows):

    * ``cosmology`` -- an lcdm artifact must never load as nulcdm or vice versa
      (different native basis, different emulators);
    * ``shared_keys`` -- ORDER-sensitive: the consumer indexes this block by
      position when embedding it into the joint packed vector;
    * ``theory_config_hash`` -- :data:`THEORY_CONFIG_HASH` (lcdm) /
      :data:`NULCDM_THEORY_CONFIG_HASH` (nulcdm), so a CMB block built against a
      different production config cannot be summed with the PFS/BAO blocks.
      Enforce-if-present, matching the template-guard convention.
    * ``cmb_config_hash`` -- :data:`CMB_CONFIG_HASH_LCDM` /
      :data:`CMB_CONFIG_HASH_NULCDM`. HARD-REQUIRED, with NO enforce-if-present
      grace: an artifact that carries no fingerprint is refused, and so is one
      built while the pin was ``None``. The other three guards check what the
      block CLAIMS to be; this one checks what actually went into it (emulator
      and .clik file content, library versions, per-term method, Gauss-Newton
      algorithm revision, shared-prior policy, fiducial and basis).
    """
    if cosmology not in _COSMOLOGIES:
        raise ValueError(
            f"unknown cosmology {cosmology!r}; expected one of {_COSMOLOGIES}")
    path = cmb_fisher_path(cosmology, cache_dir)
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"]))
        expected_keys = (SHARED_KEYS_CMB_LCDM if cosmology == "lcdm"
                         else SHARED_KEYS_CMB_NULCDM)
        if meta.get("cosmology") != cosmology:
            raise ValueError(
                f"artifact cosmology {meta.get('cosmology')!r} != requested "
                f"{cosmology!r} ({path})")
        if tuple(meta.get("shared_keys", ())) != expected_keys:
            raise ValueError(
                f"artifact shared_keys {meta.get('shared_keys')} != expected "
                f"{expected_keys} ({path})")
        expected_hash = (THEORY_CONFIG_HASH if cosmology == "lcdm"
                         else NULCDM_THEORY_CONFIG_HASH)
        got = meta.get("theory_config_hash")
        accepted = {expected_hash} | _CMB_EQUIVALENT_THEORY_HASHES[cosmology]
        if got is not None and got not in accepted:
            raise ValueError(
                f"artifact theory_config_hash {got} not in accepted set "
                f"{sorted(accepted)} ({path})")
        expected_cmb_hash = (CMB_CONFIG_HASH_LCDM if cosmology == "lcdm"
                             else CMB_CONFIG_HASH_NULCDM)
        got_cmb = meta.get("cmb_config_hash")
        if expected_cmb_hash is None:
            raise ValueError(
                f"no CMB_CONFIG_HASH is pinned for {cosmology!r} in "
                "stream_common, so no artifact of that cosmology may be loaded."
                f" Build it and pin the fingerprint it reports ({path})")
        if got_cmb is None:
            raise ValueError(
                f"artifact carries no cmb_config_hash ({path}); it predates the "
                "CMB provenance fingerprint. Rebuild it with "
                "build_cmb_fisher_block.py")
        if got_cmb != expected_cmb_hash:
            raise ValueError(
                f"artifact cmb_config_hash {got_cmb} != pinned "
                f"{expected_cmb_hash} ({path}) -- the CMB inputs (emulators, "
                ".clik data, candl/clipy/jax versions, per-term method, "
                "Gauss-Newton algorithm version, shared-prior policy, fiducial "
                "or basis) changed since the pin")
        return {"F_shared": jnp.asarray(z["F_cmb_shared"]),
                "fid_shared": jnp.asarray(z["fid_shared"]),
                "shared_keys": expected_keys,
                "sigma_tau": float(z["sigma_tau"]),
                "meta": meta}
