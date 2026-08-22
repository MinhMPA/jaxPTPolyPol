#!/usr/bin/env python3
"""Build the fiducial-centered Gaussian CMB Fisher block artifact.

Ports cells 3/13/14/16 of ``example/fisher/fisher_joint_PFS_BAO_CMB_{LCDM,
nuLCDM}.ipynb``:

  1. load the 5 candl/clipy likelihood terms (Planck highl TTTEEE, lowl TT,
     lowl EE simall, Planck lensing, ACT DR6 lensing) with internal priors ON;
  2. per-term loglike over (COSMO_KEYS_CMB + sampled CMB nuisances) at the
     fiducial;
  3. ``F_full = sum_terms F_term`` with a HYBRID per-term rule (see below);
  4. Schur-marginalize the nuisances -> cosmo-native block;
  5. project H0 -> h into the shared basis (ombh2, omch2, logA, ns, h, tau
     [, mnu]);
  6. HARD GATES (abort loudly, no fallback):
       G1 (E1): |grad of the lowl_EE term wrt tau| > 0 at the fiducial, AND
                sigma_tau = sqrt(inv(F_shared)[tau, tau]) in [0.004, 0.02];
       G2 (E2): min(eigvals(F_shared)) > 0, with NO eigenvalue clipping;
       G3 (E3): the d(shared)/d(native) Jacobian has J[h_row, H0_col] == 0.01;
  7. save the npz + META (incl. gate results, the per-term method and the exact
     build command).

Hybrid Gauss-Newton per-term rule
---------------------------------
The three terms whose data model is Gaussian in band powers (``planck_highl``,
``planck_lensing``, ``act_dr6_lensing``) contribute the EXPECTED (Gauss-Newton)
Fisher ``J^T C^-1 J`` plus their internal nuisance-prior curvature. The two
non-Gaussian low-ell terms (``planck_lowl_tt`` Gibbs/Blackwell-Rao,
``planck_lowl_ee`` simall spline) have no ``J^T C^-1 J`` and contribute the
observed Hessian ``-0.5 (H + H^T)`` exactly as before. See
``cmb_gn_fisher.py`` for the derivation, the clipy/candl introspection results
and the machine-precision validation of every reconstructed Gaussian form.

Why: the observed Hessian of a real-data likelihood evaluated AWAY from its own
maximum carries a residual-curvature term ``-sum_a (C^-1 delta)_a d2 m_a`` of
indefinite sign, which dominates near-null directions. In nuLCDM that tipped the
CMB geometric-degeneracy mode (99.7% H0, 7.7% mnu) negative and aborted G2
(raw -0.250293, projected -46.2436). Gauss-Newton drops that term and is PSD by
construction; per-term attribution (report section on the diagnostic) shows
``planck_highl`` sources 93% of the negative mode and ``planck_lensing`` the
rest -- both Gauss-Newton-able -- while the two low-ell terms contribute NET
POSITIVE curvature there, so the hybrid cures it.

Shared internal priors
----------------------
All four Planck ``.clik`` terms are loaded with ``all_priors=True``, so each one
folds the SAME Gaussian ``A_planck`` calibration prior into its own log-like and
the summed block counts it four times. Step 3 therefore ends with a shared-prior
INVENTORY (widths read programmatically from the likelihood objects) and a
duplicate-curvature subtraction applied AFTER summation -- which leaves every
per-term log-likelihood, and hence every Gauss-Newton validation reference,
untouched. The build aborts if any shared prior's curvature cannot be located
analytically.

Dependencies of THIS script (not of the notebooks)
--------------------------------------------------
Three distributions beyond ``pip install -e ".[full]"``'s core set, all imported
lazily so the rest of the repo never pays for them:

==============  ==============  =============================================
import name     distribution    where it comes from
==============  ==============  =============================================
``candl``       ``candl-like``  PyPI (also github.com/Lbalkenhol/candl)
``clipy``       ``clipy-like``  PyPI (also github.com/benabed/clipy)
``candl_data``  ``candl_data``  **NOT on PyPI** -- ships inside the candl
                                source tree, ``pip install -e candl/candl_data``
==============  ==============  =============================================

``candl-like`` and ``clipy-like`` are declared in ``pyproject.toml``'s ``full``
extra; ``candl_data`` cannot be, and is documented there instead. On top of
those, the four Planck ``.clik`` likelihood trees (~2 GB of DATA, paths pinned
in the constants below) and the CMB emulator networks must be present; their
CONTENT is hashed into ``CMB_CONFIG_HASH``, as are the ``candl`` / ``clipy`` /
``jax`` versions -- so a library upgrade forces a rebuild and a repin rather
than a silent reuse of a stale artifact.

The consuming notebooks need NONE of this: they load
``example/mcmc/cache/cmb_fisher_{lcdm,nulcdm}.npz``.

Usage: ``python3 build_cmb_fisher_block.py --cosmology {lcdm,nulcdm}
[--dry-run] [--diagnose-negative-mode]``, or ``--summary`` to regenerate the
comparison JSON from the two existing artifacts. ``--dry-run`` runs steps 1-2
shapes-only (no Fisher), prints the layout, exits 0.

The output artifact is a GAUSSIAN summary of the CMB posterior centered on the
FIDUCIAL cosmology -- a forecast object, not a fit to the real Planck/ACT data
vector. Tasks 3-4 consume it through ``stream_common.load_cmb_fisher_block``.

Every path/key constant below is copied VERBATIM from the notebook cells named
above, and :func:`verify_notebook_provenance` re-reads those notebooks at run
time and refuses to build unless each literal is still found in the source cell
-- so this script cannot silently drift from the Fisher notebooks it mirrors.
"""

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cmb_gn_fisher  # noqa: E402
import stream_common  # noqa: E402
from stream_common import (  # noqa: E402
    FIDUCIAL,
    NULCDM_FIDUCIAL,
    NULCDM_THEORY_CONFIG_HASH,
    SHARED_KEYS_CMB_LCDM,
    SHARED_KEYS_CMB_NULCDM,
    TAU_FID,
    THEORY_CONFIG_HASH,
    cmb_fisher_path,
)

from jaxptpolypol.cmb import (  # noqa: E402
    CandlParameterLayout,
    get_candl_default_parameters,
    get_candl_parameter_names,
    load_candl_likelihood,
    make_candl_loglike_fn,
    make_candl_pars_to_theory_specs_fn,
)
from jaxptpolypol.inference import (  # noqa: E402
    marginalized_fisher_block,
    project_fisher_to_derived,
)
from jaxptpolypol.params import CosmoParams  # noqa: E402

# ---------------------------------------------------------------------------
# Config constants -- VERBATIM from fisher_joint_PFS_BAO_CMB_{LCDM,nuLCDM}.ipynb
# cell 3 ("Unified fiducial cosmology ... / CMB config"). The likelihood paths
# and INCLUDE_INTERNAL_PRIORS are byte-identical between the two notebooks; only
# the emulator networks and the native cosmology basis differ.
# ---------------------------------------------------------------------------

PLANCK_ROOT = pathlib.Path('/Users/nguyenmn/candl/clipy/Planck_likelihoods/baseline/plc_3.0')
PLANCK_HIGHL   = PLANCK_ROOT / 'hi_l/plik/plik_rd12_HM_v22b_TTTEEE.clik'
PLANCK_LOWL_TT = PLANCK_ROOT / 'low_l/commander/commander_dx12_v3_2_29.clik'
PLANCK_LOWL_EE = PLANCK_ROOT / 'low_l/simall/simall_100x143_offlike5_EE_Aplanck_B.clik'
PLANCK_LENSING = PLANCK_ROOT / 'lensing/smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8_CMBmarged.clik_lensing'


def act_dr6_lens():
    """ACT DR6 lensing dataset path -- resolved LAZILY, at call time.

    ``candl_data`` is a DATA package: importing it at module scope would make
    importing THIS module fail wherever the datasets are not installed, which
    breaks collection of the data-free ``tests/test_cmb_gn_fisher.py`` (it
    imports this module for its constants). Only the build path needs the
    dataset, so the import lives here.
    """
    import candl_data
    return candl_data.ACT_DR6_Lens_only


INCLUDE_INTERNAL_PRIORS = True

CMB_EMULATOR_DIR_LCDM = pathlib.Path('/Users/nguyenmn/cosmopower-jax-for-pfs/cosmology/jense2024/jense_2023_camb_lcdm/networks')
CMB_EMULATOR_FILENAMES_LCDM = {
    'TT': str(CMB_EMULATOR_DIR_LCDM / 'jense_2023_camb_lcdm_Cl_tt.npz'),
    'TE': str(CMB_EMULATOR_DIR_LCDM / 'jense_2023_camb_lcdm_Cl_te.npz'),
    'EE': str(CMB_EMULATOR_DIR_LCDM / 'jense_2023_camb_lcdm_Cl_ee.npz'),
    'pp': str(CMB_EMULATOR_DIR_LCDM / 'jense_2023_camb_lcdm_Cl_pp.npz'),
}

CMB_EMULATOR_DIR_NULCDM = pathlib.Path('/Users/nguyenmn/cosmopower-jax-for-pfs/cosmology/jense2024/jense_2023_camb_mnu/networks')
CMB_EMULATOR_FILENAMES_NULCDM = {
    'TT': str(CMB_EMULATOR_DIR_NULCDM / 'jense_2023_camb_mnu_Cl_tt.npz'),
    'TE': str(CMB_EMULATOR_DIR_NULCDM / 'jense_2023_camb_mnu_Cl_te.npz'),
    'EE': str(CMB_EMULATOR_DIR_NULCDM / 'jense_2023_camb_mnu_Cl_ee.npz'),
    'pp': str(CMB_EMULATOR_DIR_NULCDM / 'jense_2023_camb_mnu_Cl_pp.npz'),
}

COSMO_KEYS_CMB_LCDM = ('H0', 'ombh2', 'omch2', 'logA', 'ns', 'tau')
COSMO_KEYS_CMB_NULCDM = ('H0', 'ombh2', 'omch2', 'logA', 'ns', 'tau', 'mnu')

#: Per-cosmology bundle: the native (candl) basis, the shared basis, the CMB
#: emulator networks, the fiducial source dict in stream_common, the expected
#: theory-config hash and the notebook this was ported from.
CONFIG = {
    "lcdm": {
        "cosmo_keys": COSMO_KEYS_CMB_LCDM,
        "shared_keys": SHARED_KEYS_CMB_LCDM,
        "emulator_filenames": CMB_EMULATOR_FILENAMES_LCDM,
        "fiducial": FIDUCIAL,
        "theory_config_hash": THEORY_CONFIG_HASH,
        "notebook": "fisher_joint_PFS_BAO_CMB_LCDM.ipynb",
    },
    "nulcdm": {
        "cosmo_keys": COSMO_KEYS_CMB_NULCDM,
        "shared_keys": SHARED_KEYS_CMB_NULCDM,
        "emulator_filenames": CMB_EMULATOR_FILENAMES_NULCDM,
        "fiducial": NULCDM_FIDUCIAL,
        "theory_config_hash": NULCDM_THEORY_CONFIG_HASH,
        "notebook": "fisher_joint_PFS_BAO_CMB_nuLCDM.ipynb",
    },
}

#: G1's sigma(tau) acceptance window. Planck lowE gives sigma(tau) ~ 0.007; the
#: window is wide enough to admit any sane likelihood combination and narrow
#: enough to catch a dead simall spline (which drives sigma(tau) -> unbounded)
#: or a mis-scaled tau direction.
SIGMA_TAU_RANGE = (0.004, 0.02)

#: G1's gradient floor. A dead clipy simall spline returns EXACTLY 0.0 here.
TAU_GRAD_FLOOR = 1e-3

#: Marginalized 1-sigma widths from the PFS-ONLY production runs, in the shared
#: basis. Used ONLY by ``--summary`` to build a joint-proxy regularizer
#: ``F_reg = diag(1/sigma^2)`` so the CMB block's marginal widths can be quoted
#: in a setting where every direction is constrained -- the CMB block alone does
#: not constrain the PFS-facing directions. ``sigma = 0`` means "PFS carries no
#: information on this parameter" (tau) and maps to a ZERO regularizer entry,
#: i.e. infinite prior width. These are inputs to a diagnostic summary only;
#: nothing in the artifact depends on them. Source: the committed production
#: MCMC ``sig`` columns of ``mcmc_joint_PFS_BAO_BBN_ns_{LCDM,nuLCDM}.ipynb``.
PFS_ONLY_SIGMAS = {
    "lcdm": {"ombh2": 0.00047985, "omch2": 0.0032185, "logA": 0.060783,
             "ns": 0.027632, "h": 0.0035686, "tau": 0.0},
    "nulcdm": {"ombh2": 0.00048188, "omch2": 0.0032276, "logA": 0.078534,
               "ns": 0.033284, "h": 0.0036311, "mnu": 0.095725, "tau": 0.0},
}

#: Where ``--summary`` writes. The ONLY output path of that mode.
SUMMARY_PATH = stream_common.CACHE_DIR / "cmb_block_branchB_summary.json"

NOTEBOOK_DIR = pathlib.Path(__file__).resolve().parents[2] / "fisher"


# ---------------------------------------------------------------------------
# Provenance guard -- the script refuses to run if the source cells moved.
# ---------------------------------------------------------------------------

def verify_notebook_provenance(cosmology):
    """Re-read the source notebook and assert every ported literal is still there.

    The constants above are a COPY of notebook state; a copy with no guard is a
    silent-drift hazard (the notebook could be re-run with a different Planck
    root, emulator generation or native basis while this script keeps the stale
    literals). So the build locates the two source cells BY CONTENT -- the
    config cell contains ``CMB_EMULATOR_FILENAMES = {``, the loader cell contains
    ``load_likelihood_terms`` -- and requires every ported literal to appear
    verbatim in them. Any miss aborts the build (``SystemExit``).
    """
    cfg = CONFIG[cosmology]
    path = NOTEBOOK_DIR / cfg["notebook"]
    if not path.is_file():
        sys.exit(f"ABORT: source notebook not found: {path}")
    cells = [
        "".join(cell["source"])
        for cell in json.loads(path.read_text())["cells"]
        if cell["cell_type"] == "code"
    ]

    def _one(marker, what):
        hits = [src for src in cells if marker in src]
        if len(hits) != 1:
            sys.exit(f"ABORT: expected exactly 1 {what} cell containing "
                     f"{marker!r} in {path.name}, found {len(hits)}. The "
                     "notebook layout changed -- re-port the constants before "
                     "building.")
        return hits[0]

    config_cell = _one("CMB_EMULATOR_FILENAMES = {", "CMB-config")
    loader_cell = _one("def load_likelihood_terms", "likelihood-loader")

    required_in_config = [
        str(PLANCK_ROOT),
        "hi_l/plik/plik_rd12_HM_v22b_TTTEEE.clik",
        "low_l/commander/commander_dx12_v3_2_29.clik",
        "low_l/simall/simall_100x143_offlike5_EE_Aplanck_B.clik",
        "lensing/smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8_CMBmarged.clik_lensing",
        "candl_data.ACT_DR6_Lens_only",
        "INCLUDE_INTERNAL_PRIORS = True",
        str(CMB_EMULATOR_DIR_LCDM if cosmology == "lcdm" else CMB_EMULATOR_DIR_NULCDM),
        f"COSMO_KEYS_CMB = {cfg['cosmo_keys']!r}",
        f"SHARED_KEYS = {cfg['shared_keys']!r}",
    ]
    required_in_config += [
        pathlib.Path(fname).name for fname in cfg["emulator_filenames"].values()
    ]
    required_in_loader = [
        "'planck_highl'", "'planck_lowl_tt'", "'planck_lowl_ee'",
        "'planck_lensing'", "'act_dr6_lensing'",
        "{'all_priors': True} if INCLUDE_INTERNAL_PRIORS else {}",
    ]
    missing = ([f"config cell: {t!r}" for t in required_in_config
                if t not in config_cell]
               + [f"loader cell: {t!r}" for t in required_in_loader
                  if t not in loader_cell])
    if missing:
        sys.exit("ABORT: ported constants no longer match "
                 f"{path.name}:\n  " + "\n  ".join(missing))
    print(f"[provenance] all ported constants verified against {path.name} "
          f"({len(required_in_config)} config + {len(required_in_loader)} "
          "loader literals)", flush=True)


# ---------------------------------------------------------------------------
# Helpers -- VERBATIM from cell 13 of both notebooks (``cosmo_keys`` and
# ``include_internal_priors``, globals in the notebook, become arguments here).
# ---------------------------------------------------------------------------

def scalar_value(value):
    return float(np.asarray(value).reshape(()))


def ordered_union(name_lists):
    ordered, seen = [], set()
    for names in name_lists:
        for name in names:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
    return tuple(ordered)


def load_likelihood_terms():
    clipy_args = {'all_priors': True} if INCLUDE_INTERNAL_PRIORS else {}
    return {
        'planck_highl': load_candl_likelihood(str(PLANCK_HIGHL), wrapper='clipy', additional_args=clipy_args),
        'planck_lowl_tt': load_candl_likelihood(str(PLANCK_LOWL_TT), wrapper='clipy', additional_args=clipy_args),
        'planck_lowl_ee': load_candl_likelihood(str(PLANCK_LOWL_EE), wrapper='clipy', additional_args=clipy_args),
        'planck_lensing': load_candl_likelihood(str(PLANCK_LENSING), wrapper='clipy', additional_args=clipy_args),
        'act_dr6_lensing': load_candl_likelihood(
            act_dr6_lens(), lensing=True, feedback=False,
            clear_internal_priors=not INCLUDE_INTERNAL_PRIORS,
        ),
    }


def collect_cmb_nuisance_defaults(likelihoods, cosmo_keys):
    nuisance_names_by_term, nuisance_defaults = {}, {}
    for term_name, like in likelihoods.items():
        names = get_candl_parameter_names(like, cosmo_keys=cosmo_keys, include_prior_params=True)
        nuisance_names_by_term[term_name] = names
        for key, value in get_candl_default_parameters(like).items():
            nuisance_defaults.setdefault(key, scalar_value(value))
    nuisance_names = ordered_union(nuisance_names_by_term.values())
    missing_defaults = [name for name in nuisance_names if name not in nuisance_defaults]
    if missing_defaults:
        raise ValueError(
            'Missing default values for CMB nuisance parameters: '
            + ', '.join(missing_defaults)
        )
    return nuisance_names, nuisance_defaults, nuisance_names_by_term


# ---------------------------------------------------------------------------
# Assembly (cell 14) and projection (cell 16).
# ---------------------------------------------------------------------------

def build_fiducial_cosmo_cmb(cosmology):
    """``FIDUCIAL_COSMO_CMB`` in COSMO_KEYS_CMB insertion order (cell 3).

    Values come from ``stream_common`` (FIDUCIAL / NULCDM_FIDUCIAL / TAU_FID),
    never retyped; insertion order matters because
    ``CandlParameterLayout.pack`` fills the cosmology block by it.
    """
    fid = CONFIG[cosmology]["fiducial"]
    out = {
        'H0': fid['h'] * 100,
        'ombh2': fid['ombh2'], 'omch2': fid['omch2'],
        'logA': fid['logA'], 'ns': fid['ns'], 'tau': TAU_FID,
    }
    if cosmology == "nulcdm":
        out['mnu'] = fid['mnu']
    if tuple(out) != CONFIG[cosmology]["cosmo_keys"]:
        sys.exit(f"ABORT: fiducial key order {tuple(out)} != COSMO_KEYS_CMB "
                 f"{CONFIG[cosmology]['cosmo_keys']}")
    return out


def assemble(cosmology):
    """Cell 14: likelihoods, layout, theta_fid, per-term and joint loglikes."""
    cfg = CONFIG[cosmology]
    cosmo_keys = cfg["cosmo_keys"]
    pars_to_theory_specs = make_candl_pars_to_theory_specs_fn(
        emulator_filenames=cfg["emulator_filenames"])
    likelihoods = load_likelihood_terms()
    nuisance_names, nuisance_defaults, _by_term = collect_cmb_nuisance_defaults(
        likelihoods, cosmo_keys)

    sampled_nuisance = tuple(nuisance_names)
    layout = CandlParameterLayout(
        cosmo_keys=tuple(cosmo_keys),
        cosmo_sizes=tuple(1 for _ in cosmo_keys),
        cmb_nuisance_names=sampled_nuisance,
    )
    fiducial_cosmo_cmb = build_fiducial_cosmo_cmb(cosmology)
    theta_fid_cmb = layout.pack(
        CosmoParams(fiducial_cosmo_cmb),
        {name: nuisance_defaults[name] for name in sampled_nuisance},
    )
    fixed_cmb_params = {}  # All nuisance sampled

    term_loglikes = {
        term_name: make_candl_loglike_fn(
            like, pars_to_theory_specs=pars_to_theory_specs,
            layout=layout, fixed_cmb_params=fixed_cmb_params,
        )
        for term_name, like in likelihoods.items()
    }
    return {
        "layout": layout, "theta_fid": theta_fid_cmb,
        "sampled_nuisance": sampled_nuisance,
        "term_loglikes": term_loglikes,
        "fiducial_cosmo_cmb": fiducial_cosmo_cmb,
        "likelihoods": likelihoods,
        "pars_to_theory_specs": pars_to_theory_specs,
        "fixed_cmb_params": fixed_cmb_params,
    }


def build_cmb_fisher_full(pieces):
    """Per-term hybrid CMB Fisher over the packed (cosmo + nuisance) vector.

    Gauss-Newton ``J^T C^-1 J`` for every term with a Gaussian band-power data
    model; observed Hessian ``-0.5 (H + H^T)`` for the non-Gaussian low-ell
    terms. Every Gauss-Newton term is validated first: its reconstructed
    ``(data, model, covariance, prior)`` form must reproduce the untouched
    candl/clipy log-likelihood in BOTH value and full Hessian, otherwise the
    build aborts.

    Returns ``{"per_term", "method", "sources", "reports"}`` -- the per-term
    blocks are kept so the caller can deduplicate shared priors and attribute
    the negative mode without recomputing anything.
    """
    theta = pieces["theta_fid"]
    method, sources, reports, per_term = {}, {}, [], {}
    for term_name, likelihood in pieces["likelihoods"].items():
        gn = cmb_gn_fisher.make_gn_pieces(
            term_name, likelihood,
            pars_to_theory_specs=pieces["pars_to_theory_specs"],
            layout=pieces["layout"],
            fixed_cmb_params=pieces["fixed_cmb_params"],
        )
        if gn is None:
            per_term[term_name] = observed_hessian_fisher(pieces, term_name)
            method[term_name] = "hessian"
            sources[term_name] = (
                f"{type(likelihood._internal).__module__}."
                f"{type(likelihood._internal).__name__}: non-Gaussian "
                "likelihood, observed Hessian -0.5 (H + H^T)")
            print(f"[method] {term_name:16s} hessian  (non-Gaussian "
                  "likelihood: no J^T C^-1 J exists)", flush=True)
            continue
        try:
            report = cmb_gn_fisher.validate_gn_term(
                term_name, gn, pieces["term_loglikes"][term_name], theta)
        except cmb_gn_fisher.GNValidationError as exc:
            sys.exit(f"ABORT: Gauss-Newton validation failed -- {exc}")
        reports.append(report)
        per_term[term_name] = cmb_gn_fisher.gn_fisher(gn, theta)
        method[term_name] = "GN"
        sources[term_name] = gn["source"]
        print(f"[method] {term_name:16s} GN       n_data={report['n_data']:5d}  "
              f"logL rel err {report['value_rel_err']:.3g}  "
              f"Hessian abs err {report['hessian_max_abs_err']:.3g}  "
              f"dir err {report['directional_abs_err']:.3g} < "
              f"{report['directional_budget']:.3g}  "
              f"chi2_resid(fid) {report['chi2_residual_at_fiducial']:.6g}",
              flush=True)
    return {"per_term": per_term, "method": method, "sources": sources,
            "reports": reports}


def observed_hessian_fisher(pieces, term_name):
    """``-0.5 (H + H^T)`` of one term's log-likelihood at the fiducial."""
    hess = np.asarray(jax.jit(jax.hessian(
        pieces["term_loglikes"][term_name]))(pieces["theta_fid"]))
    return -0.5 * (hess + hess.T)


def apply_shared_prior_dedupe(pieces, F_full):
    """Remove the duplicate curvature of priors shared across likelihood terms.

    All four Planck ``.clik`` likelihoods are loaded with ``all_priors=True``,
    so each one folds the SAME Gaussian ``A_planck`` calibration prior into its
    own ``log_like``; summing the five per-term blocks counts it four times.
    The widths are read from the likelihood objects themselves (never
    hardcoded), and the subtraction happens AFTER summation so no per-term
    log-likelihood -- and therefore no Gauss-Newton validation reference -- is
    disturbed.

    Returns ``(F_deduped, prior_policy)``.
    """
    inventory = cmb_gn_fisher.inventory_shared_priors(
        pieces["likelihoods"], layout=pieces["layout"],
        theta_fid=pieces["theta_fid"],
        fixed_cmb_params=pieces["fixed_cmb_params"])
    correction = cmb_gn_fisher.duplicate_prior_curvature(
        inventory, F_full.shape[0])
    if not inventory:
        print("[priors] no prior is shared across terms; nothing to "
              "deduplicate", flush=True)
    for name, entry in inventory.items():
        print(f"[priors] {name!r} prior appears in {entry['count']} terms "
              f"{entry['terms']}: sigma = {entry['sigma']:.6g}, curvature = "
              f"{entry['curvature']:.10g} each; subtracting "
              f"{entry['count'] - 1} x {entry['curvature']:.10g} = "
              f"{(entry['count'] - 1) * entry['curvature']:.10g} at packed "
              f"index {entry['packed_index']}", flush=True)
    before = float(np.linalg.eigvalsh(F_full).min())
    F_dedup = F_full - correction
    after = float(np.linalg.eigvalsh(F_dedup).min())
    print(f"[priors] full-block min eig {before:.6g} -> {after:.6g}",
          flush=True)
    policy = {
        "shared_prior_inventory": {
            name: {k: v for k, v in entry.items()}
            for name, entry in inventory.items()},
        "total_curvature_subtracted": {
            name: (entry["count"] - 1) * entry["curvature"]
            for name, entry in inventory.items()},
        "applied": "after summation of the per-term blocks",
        "exactness": (
            "Exact, not approximate: a Gaussian prior's Hessian is a constant "
            "matrix, so each duplicate copy contributes exactly "
            "curvature * e_p e_p^T to the summed block -- explicitly for the "
            "Gauss-Newton terms (which add the prior Hessian by hand) and "
            "identically for the observed-Hessian terms (whose Hessian "
            "contains that same constant). Removing count-1 copies restores "
            "single counting with no residual."),
        "widths_source": (
            "read programmatically by differentiating each likelihood "
            "object's own prior callable; never hardcoded"),
        "full_block_min_eig_before": before,
        "full_block_min_eig_after": after,
    }
    return F_dedup, policy


def diagnose_negative_mode(per_term_observed, n_cosmo):
    """Per-term attribution of the observed-Hessian near-null eigenvalue.

    This recomputes -- rather than quotes -- the numbers that motivated the
    hybrid Gauss-Newton method. The summed Fisher is exactly additive over the
    five terms in the packed basis, so for the nuisance-profiled minimum
    eigenvector ``u`` of the nuisance-marginalized cosmology block,
    ``sum_t u^T F_t u`` reproduces that eigenvalue exactly and splits it by
    term with no bookkeeping slack. Deliberately computed on the PRE-dedupe
    observed-Hessian sum: that is the object whose eigenvalue is being
    explained, and per-term additivity is exact there.

    Returns a structured dict (no prose, no retyped numbers).
    """
    F_obs = sum(per_term_observed.values())
    A_cc = F_obs[:n_cosmo, :n_cosmo]
    A_cn = F_obs[:n_cosmo, n_cosmo:]
    A_nn = F_obs[n_cosmo:, n_cosmo:]
    F_marg = A_cc - A_cn @ np.linalg.solve(A_nn, A_cn.T)
    eigvals, eigvecs = np.linalg.eigh(F_marg)
    v_cosmo = eigvecs[:, 0]
    u = np.concatenate([v_cosmo, -np.linalg.solve(A_nn, A_cn.T @ v_cosmo)])
    attribution = {t: float(u @ F @ u) for t, F in per_term_observed.items()}
    return {
        "basis": "observed Hessian, pre-dedupe, packed (cosmo + nuisance)",
        "marginalized_min_eig": float(eigvals[0]),
        "attribution_sums_to": float(sum(attribution.values())),
        "direction_cosmo_components": [float(x) for x in v_cosmo],
        "per_term": attribution,
    }


def make_cmb_to_shared(cosmology):
    """Cell 16's ``cmb_to_shared``: native (H0, ...) -> shared (..., h, ...)."""
    cosmo_keys = CONFIG[cosmology]["cosmo_keys"]

    def cmb_to_shared(params):
        H0     = params[cosmo_keys.index('H0')]
        ombh2  = params[cosmo_keys.index('ombh2')]
        omch2  = params[cosmo_keys.index('omch2')]
        logA   = params[cosmo_keys.index('logA')]
        ns     = params[cosmo_keys.index('ns')]
        tau    = params[cosmo_keys.index('tau')]
        if cosmology == "lcdm":
            return jnp.array([ombh2, omch2, logA, ns, H0 / 100.0, tau])
        mnu = params[cosmo_keys.index('mnu')]
        return jnp.array([ombh2, omch2, logA, ns, H0 / 100.0, tau, mnu])

    return cmb_to_shared


def shared_basis_marginal_sigmas(F_full, cosmology, fiducial_cosmo_cmb):
    """Shared-basis 1-sigma marginal widths of a packed (cosmo + nuisance) block.

    Runs the SAME chain as the main build path -- Schur-marginalize the
    nuisances away, project native -> shared -- and returns
    ``sqrt(diag(inv(F_shared)))`` in ``shared_keys`` order.

    Called TWICE per build, on the pre- and post-dedupe full blocks. The
    duplication is deliberate: routing both through one function makes the two
    vectors traverse byte-identical code, so their ratio isolates the
    shared-prior deduplication and nothing else. (Reusing the main path's
    already-projected ``F_cmb_shared`` for the post vector would have been
    cheaper but would compare two differently-computed quantities.) That turns
    the dedupe's effect on the widths into a measured, artifact-stored number
    instead of a prose claim. Cost is one extra marginalize + invert;
    milliseconds.
    """
    cfg = CONFIG[cosmology]
    F_cosmo = marginalized_fisher_block(
        F_full, list(range(len(cfg["cosmo_keys"]))))
    fid = jnp.array([fiducial_cosmo_cmb[k] for k in cfg["cosmo_keys"]],
                    dtype=jnp.float64)
    F_shared, _fid, _jac, _cov = project_fisher_to_derived(
        F_cosmo, fid, make_cmb_to_shared(cosmology))
    return np.sqrt(np.diag(np.linalg.inv(np.asarray(F_shared))))


# ---------------------------------------------------------------------------
# Gates -- each aborts loudly; there is no fallback path.
# ---------------------------------------------------------------------------

def gate_g1_tau_gradient(pieces, cosmology):
    """G1a (E1): the lowl-EE (simall) term must have a LIVE tau gradient.

    Run BEFORE the Hessian: a dead clipy simall spline returns exactly 0.0 and
    would otherwise be discovered only after minutes of Hessian work, and would
    silently produce a CMB block with no tau information at all.
    """
    tau_idx = CONFIG[cosmology]["cosmo_keys"].index('tau')
    grad = np.asarray(jax.grad(pieces["term_loglikes"]["planck_lowl_ee"])(
        pieces["theta_fid"]))
    g_tau = float(grad[tau_idx])
    if not abs(g_tau) > TAU_GRAD_FLOOR:
        sys.exit(
            "ABORT G1 (E1): d(planck_lowl_ee loglike)/d(tau) = "
            f"{g_tau!r} at the fiducial (|.| <= {TAU_GRAD_FLOOR}) -- the "
            "installed clipy simall has no tau gradient — the 2026-07-14 "
            "cubic-spline fix is missing from this environment.")
    print(f"[G1a] PASS  d(lowl_EE)/d(tau) = {g_tau:.6g}  "
          f"(|.| > {TAU_GRAD_FLOOR})", flush=True)
    return g_tau


def gate_g1_sigma_tau(F_shared, shared_keys):
    """G1b (E1): sigma_tau from the shared-basis block must be Planck-like."""
    tau_idx = shared_keys.index('tau')
    cov = np.linalg.inv(np.asarray(F_shared))
    sigma_tau = float(np.sqrt(cov[tau_idx, tau_idx]))
    lo, hi = SIGMA_TAU_RANGE
    if not (lo <= sigma_tau <= hi):
        sys.exit(f"ABORT G1 (E1): sigma_tau = {sigma_tau:.6g} outside "
                 f"[{lo}, {hi}] -- the CMB block does not carry a Planck-like "
                 "optical-depth constraint.")
    print(f"[G1b] PASS  sigma_tau = {sigma_tau:.6g}  in [{lo}, {hi}]",
          flush=True)
    return sigma_tau


def gate_g2_positive_definite(F_shared):
    """G2 (E2): the shared-basis CMB Fisher must be positive DEFINITE.

    Strict ``> 0``, as in the pre-branch baseline. The hybrid Gauss-Newton block
    is PSD by construction wherever it applies and clears this comfortably
    (+52.2 nuLCDM, +4332 LCDM); relaxing the comparison to ``>= 0`` would have
    weakened the gate relative to the baseline for no benefit.

    NO clipping, no regularization: a failure here means the two low-ell
    observed-Hessian terms have overwhelmed the Gauss-Newton part and the method
    itself has to be revisited -- not patched over.
    """
    eigvals = np.linalg.eigvalsh(np.asarray(F_shared))
    min_eig = float(eigvals.min())
    max_eig = float(eigvals.max())
    if not min_eig > 0.0:
        sys.exit(f"ABORT G2 (E2): min eigenvalue of F_shared = {min_eig:.6g} "
                 "<= 0 -- the projected CMB Fisher is not positive definite. "
                 f"Full spectrum: {eigvals.tolist()!r}")
    cond = f"{max_eig / min_eig:.6g}" if min_eig > 0.0 else "inf"
    print(f"[G2]  PASS  min eig(F_shared) = {min_eig:.6g}  "
          f"(max {max_eig:.6g}, cond {cond})", flush=True)
    return min_eig, max_eig


def gate_g3_jacobian(jacobian, cosmology):
    """G3 (E3): the H0 -> h projection entry must be exactly 1/100."""
    cosmo_keys = CONFIG[cosmology]["cosmo_keys"]
    shared_keys = CONFIG[cosmology]["shared_keys"]
    h_row = shared_keys.index('h')
    h0_col = cosmo_keys.index('H0')
    entry = float(np.asarray(jacobian)[h_row, h0_col])
    if entry != 0.01:
        sys.exit(f"ABORT G3 (E3): d(h)/d(H0) = {entry!r} != 0.01 -- the "
                 "shared-basis projection is not the expected H0/100 map.")
    print(f"[G3]  PASS  J[h={h_row}, H0={h0_col}] = {entry!r} == 0.01",
          flush=True)
    return entry


# ---------------------------------------------------------------------------
# Provenance fingerprint.
# ---------------------------------------------------------------------------

def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(root):
    """Content digest of a directory: sorted (relpath, file-sha256) listing.

    Paths are relative, so relocating a likelihood directory does not change the
    fingerprint, but editing or re-downloading ANY file inside it does.
    """
    root = pathlib.Path(root)
    if root.is_file():
        return {"kind": "file", "sha256": _sha256_file(root)}
    entries = sorted(p for p in root.rglob("*") if p.is_file())
    listing = [[str(p.relative_to(root)), _sha256_file(p)] for p in entries]
    digest = hashlib.sha256(
        json.dumps(listing, sort_keys=True).encode()).hexdigest()
    return {"kind": "dir", "n_files": len(listing), "sha256": digest}


def compute_cmb_config_hash(cosmology, *, method_per_term, prior_policy,
                            fiducial_native, shared_keys, native_keys):
    """Content-derived fingerprint of everything that determines the block.

    Deliberately hashes FILE CONTENT, not paths or mtimes: a re-downloaded
    Planck likelihood or a regenerated emulator network changes the block and
    must change the fingerprint, while moving the data tree must not.

    Returns ``(hash, components)``; the components go into META verbatim so a
    mismatch can be diagnosed without re-running anything.
    """
    cfg = CONFIG[cosmology]
    import candl
    import clipy
    components = {
        "cosmology": cosmology,
        "emulator_files": {
            spec: _sha256_file(path)
            for spec, path in sorted(cfg["emulator_filenames"].items())},
        "likelihood_data": {
            "planck_highl": _sha256_tree(PLANCK_HIGHL),
            "planck_lowl_tt": _sha256_tree(PLANCK_LOWL_TT),
            "planck_lowl_ee": _sha256_tree(PLANCK_LOWL_EE),
            "planck_lensing": _sha256_tree(PLANCK_LENSING),
        },
        "act_dr6_dataset": str(act_dr6_lens()),
        "library_versions": {
            "candl": getattr(candl, "__version__", "unknown"),
            "clipy": getattr(clipy, "__version__", "unknown"),
            "jax": jax.__version__,
        },
        "method_per_term": dict(method_per_term),
        "gn_algorithm_version": cmb_gn_fisher.GN_ALGORITHM_VERSION,
        "include_internal_priors": INCLUDE_INTERNAL_PRIORS,
        "shared_prior_inventory": {
            name: {"sigma": entry["sigma"], "count": entry["count"],
                   "terms": entry["terms"],
                   "packed_index": entry["packed_index"]}
            for name, entry in
            prior_policy["shared_prior_inventory"].items()},
        "fiducial_native": {k: float(v) for k, v in fiducial_native.items()},
        "shared_keys": list(shared_keys),
        "native_keys": list(native_keys),
    }
    payload = json.dumps(components, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest(), components


def enforce_cmb_config_hash_pin(cosmology, cmb_config_hash, *, pinned):
    """Compare the freshly computed fingerprint against the pin. Abort on drift.

    MUST be called BEFORE the artifact is written. Three outcomes:

    * ``pinned is None`` -> returns ``True`` (BOOTSTRAP). A content-derived hash
      cannot be pinned before the build that computes it, so the first build --
      and the first after the CMB inputs legitimately change -- has nothing to
      compare against. The artifact is written and the post-build loader
      self-check is allowed to report the refusal instead of failing.
    * ``pinned == cmb_config_hash`` -> returns ``False``. Normal build.
    * otherwise -> ``sys.exit(1)``, BEFORE anything is written.

    The mismatch case used to print a warning and carry on: the artifact was
    overwritten, the post-build loader refused it, and a too-broad bootstrap
    escape swallowed that refusal and returned 0. A build in a drifted
    environment therefore replaced a good production artifact with an unloadable
    one and reported success -- this repo's known silent-overwrite failure mode
    (the 2026-08-04 output-path lesson). Aborting before the write leaves the previous
    artifact intact, which is the only outcome that cannot lose work.
    """
    if pinned is None:
        print(f"[fingerprint] stream_common pins None for {cosmology!r}: pin "
              "this value before any consumer can load the artifact",
              flush=True)
        return True
    if pinned == cmb_config_hash:
        return False
    print(f"[fingerprint] MISMATCH: stream_common pins {pinned!r} for "
          f"{cosmology!r}, this build computes {cmb_config_hash!r}. The CMB "
          "inputs (emulators, .clik data, candl/clipy/jax versions, per-term "
          "method, Gauss-Newton algorithm version, shared-prior policy, "
          "fiducial or basis) have drifted since the pin.", flush=True)
    print(f"[fingerprint] ACTION: if this drift is INTENDED, set "
          f"CMB_CONFIG_HASH_{cosmology.upper()} = {cmb_config_hash!r} in "
          "example/mcmc/scripts/stream_common.py and re-run. Otherwise restore "
          "the pinned environment.", flush=True)
    sys.exit(
        f"ABORT: CMB_CONFIG_HASH_{cosmology.upper()} mismatch -- refusing to "
        f"overwrite {cmb_fisher_path(cosmology)} with an artifact the loader "
        "would reject. The existing artifact is UNCHANGED.")


#: Substring identifying the ONE loader refusal a bootstrap build may survive:
#: ``stream_common.load_cmb_fisher_block`` raises "no CMB_CONFIG_HASH is pinned
#: for ..." when the pin is ``None``. Pinned by a covering test, because the
#: escape below silently stops working if that message is ever reworded.
_MISSING_PIN_MARKER = "CMB_CONFIG_HASH"


def verify_artifact_round_trip(cosmology, *, bootstrap_pin, cmb_config_hash):
    """Read the artifact back through the production loader.

    The artifact must satisfy the very guards its consumers apply. Exactly ONE
    refusal is survivable, and only under BOTH conditions:

    * ``bootstrap_pin`` -- the pin is ``None``, so a content-derived fingerprint
      could not have been pinned before the build that computes it; AND
    * the loader's complaint is specifically the missing-pin one.

    The conjunction matters in both directions:

    * keying on the message ALONE also matched the pin-MISMATCH refusal, which
      is how a drifted build used to overwrite a good artifact and exit 0
      (fixed in the previous round; a mismatch now exits before the write, so it
      cannot reach here at all);
    * keying on ``bootstrap_pin`` ALONE swallows every OTHER loader failure
      during a bootstrap build. The loader checks cosmology -> shared_keys ->
      theory_config_hash -> CMB hash IN THAT ORDER, so a STRUCTURAL defect fires
      FIRST and would have been reported as "written but NOT yet loadable" and
      exited 0. That is precisely the bootstrap scenario -- changing the shared
      basis is a prime reason to ``None`` the pin and rebuild -- so the widened
      guard would have hidden a genuinely broken artifact exactly when it was
      most likely to occur.

    Returns ``True`` if the artifact loaded, ``False`` on the tolerated
    bootstrap refusal. Any other loader complaint is a structural defect and
    exits non-zero.
    """
    try:
        loaded = stream_common.load_cmb_fisher_block(cosmology)
    except ValueError as exc:
        if bootstrap_pin and _MISSING_PIN_MARKER in str(exc):
            print(f"[loader] artifact written but NOT yet loadable: {exc}",
                  flush=True)
            print(f"[loader] ACTION: set CMB_CONFIG_HASH_"
                  f"{cosmology.upper()} = {cmb_config_hash!r} in "
                  "example/mcmc/scripts/stream_common.py, then re-run this "
                  "build to confirm the round-trip.", flush=True)
            return False
        sys.exit(f"ABORT: the artifact just written fails the production "
                 f"loader's guards -- {exc}")
    print(f"[loader] round-trip OK: F_shared {tuple(loaded['F_shared'].shape)}, "
          f"sigma_tau {loaded['sigma_tau']:.6g}", flush=True)
    return True


# ---------------------------------------------------------------------------
# Summary mode.
# ---------------------------------------------------------------------------

def write_summary():
    """Regenerate the branch-B comparison JSON from the two built artifacts.

    Reads only; the ONLY thing it writes is :data:`SUMMARY_PATH`. Previously
    this JSON was produced by an uncommitted scratch script, so the numbers in
    it had no reproducible provenance.
    """
    out = {}
    for cosmology in ("lcdm", "nulcdm"):
        path = cmb_fisher_path(cosmology)
        if not path.is_file():
            sys.exit(f"ABORT --summary: missing artifact {path}; build both "
                     "cosmologies first.")
        with np.load(path, allow_pickle=False) as z:
            fisher = np.asarray(z["F_cmb_shared"], dtype=np.float64)
            keys = [str(k) for k in z["shared_keys"]]
            sigma_tau = float(z["sigma_tau"])
            meta = json.loads(str(z["meta_json"]))
        sigmas = PFS_ONLY_SIGMAS[cosmology]
        missing = [k for k in keys if k not in sigmas]
        if missing:
            sys.exit(f"ABORT --summary: no PFS-only sigma for {missing} "
                     f"({cosmology}); refusing to guess a regularizer entry.")
        reg = np.array([1.0 / sigmas[k] ** 2 if sigmas[k] > 0 else 0.0
                        for k in keys])
        cov = np.linalg.inv(fisher + np.diag(reg))
        out[cosmology] = {
            "shared_keys": keys,
            "sigma_tau": sigma_tau,
            "eigenvalues": sorted(
                float(v) for v in np.linalg.eigvalsh(fisher)),
            "method_per_term": meta["method"]["per_term"],
            "negative_mode_attribution": meta["method"][
                "negative_mode_attribution"],
            "prior_policy": meta["prior_policy"],
            "cmb_config_hash": meta.get("cmb_config_hash"),
            "marginal_sigmas_when_regularized": {
                k: float(np.sqrt(cov[i, i])) for i, k in enumerate(keys)},
            "regularizer_sigmas_pfs_only": {k: sigmas[k] for k in keys},
            "gates": meta["gates"],
        }
        # sigma(tau) with mnu held fixed, to quantify how much of nuLCDM's
        # weaker tau sharpening is mnu absorbing the degeneracy-breaking.
        if cosmology == "nulcdm":
            keep = [i for i, k in enumerate(keys) if k != "mnu"]
            F_fixed = np.asarray(fisher)[np.ix_(keep, keep)]
            i_tau = [k for k in keys if k != "mnu"].index("tau")
            out[cosmology]["sigma_tau_mnu_fixed_refit"] = float(
                np.sqrt(np.linalg.inv(F_fixed)[i_tau, i_tau]))
    out["_note"] = (
        "Branch B (expt/cmb-expected-fisher): hybrid Gauss-Newton expected CMB "
        "Fisher, with the shared-prior duplicate curvature removed. "
        "marginal_sigmas_when_regularized = sqrt(diag(inv(F_shared + F_reg))) "
        "with F_reg = diag(1/sigma^2) from the PFS-only production sigmas; "
        "sigma = 0 (tau) means PFS carries no constraint -> zero regularizer "
        "entry. sigma_tau_mnu_fixed_refit (nuLCDM only) = sqrt(inv(F_shared "
        "with the mnu row/column DELETED)[tau, tau]), i.e. sigma(tau) with mnu "
        "held fixed rather than marginalized. It is a CMB-BLOCK-ALONE number: "
        "the mnu row/column is dropped and NOTHING else is changed, no external "
        "information of any kind is added, and it is recomputable to full "
        "precision from the committed artifact by exactly the four lines above "
        "this note's own source. A ratio of ~0.927 for an mnu-fixed refit is "
        "quoted elsewhere in this analysis; its numerator (0.006579) is an "
        "UNCOMMITTED review-time quantity that does NOT reproduce from this "
        "artifact combined with any documented regularizer set (the "
        "mnu-dropped block plus the regularizer_sigmas_pfs_only proxy gives "
        "0.0062329, ratio 0.8781), so treat that 0.927 as an unverified "
        "historical estimate rather than a citable result, and do not infer "
        "any construction for it from this key. "
        "No eigenvalue clipping was applied to F_shared. Regenerate "
        "with: python3 example/mcmc/scripts/build_cmb_fisher_block.py --summary")
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[summary] wrote {SUMMARY_PATH}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cosmology", choices=("lcdm", "nulcdm"),
                        help="required unless --summary")
    parser.add_argument("--dry-run", action="store_true",
                        help="steps 1-2 only (no Fisher); print the layout")
    parser.add_argument("--diagnose-negative-mode", action="store_true",
                        help="additionally dump the full per-term negative-mode "
                             "attribution as JSON (the attribution is computed "
                             "and stored in META on EVERY build regardless)")
    parser.add_argument("--summary", action="store_true",
                        help="build no Fisher; regenerate the branch-B "
                             "comparison JSON from the two existing artifacts")
    args = parser.parse_args()
    if args.summary:
        if args.cosmology:
            sys.exit("ABORT: --summary reads BOTH artifacts; do not pass "
                     "--cosmology with it.")
        return write_summary()
    if not args.cosmology:
        parser.error("--cosmology is required unless --summary is given")
    cosmology = args.cosmology
    cfg = CONFIG[cosmology]
    build_command = ("python3 example/mcmc/scripts/build_cmb_fisher_block.py "
                     f"--cosmology {cosmology}")

    print(f"=== CMB Fisher block build: {cosmology} ===", flush=True)
    verify_notebook_provenance(cosmology)

    t0 = time.time()
    pieces = assemble(cosmology)
    theta_fid = pieces["theta_fid"]
    terms = list(pieces["term_loglikes"])
    shared_keys = cfg["shared_keys"]
    print(f"CMB terms ({len(terms)}): {terms}", flush=True)
    print(f"Native cosmo basis ({len(cfg['cosmo_keys'])}): "
          f"{cfg['cosmo_keys']}", flush=True)
    print(f"Shared basis ({len(shared_keys)}): {shared_keys}", flush=True)
    print(f"Fiducial (native): {pieces['fiducial_cosmo_cmb']}", flush=True)
    print(f"Nuisance union ({len(pieces['sampled_nuisance'])}): "
          f"{list(pieces['sampled_nuisance'])}", flush=True)
    print(f"CMB parameter vector length: {int(theta_fid.shape[0])} "
          f"(cosmo {len(cfg['cosmo_keys'])} + nuisance "
          f"{len(pieces['sampled_nuisance'])})", flush=True)
    print(f"Emulators: {cfg['emulator_filenames']}", flush=True)
    print(f"[assembly] {time.time() - t0:.1f} s", flush=True)

    if args.dry_run:
        print("[dry-run] layout OK; exiting WITHOUT building the Fisher.",
              flush=True)
        return 0

    # G1a before the (minutes-scale) Hessian -- fail fast on a dead spline.
    g_tau = gate_g1_tau_gradient(pieces, cosmology)

    print("Computing per-term CMB Fisher blocks (hybrid GN + low-ell "
          "Hessian)...", flush=True)
    t_fisher = time.time()
    built = build_cmb_fisher_full(pieces)
    F_cmb_full = sum(built["per_term"].values())
    fisher_seconds = time.time() - t_fisher
    print(f"[fisher] {fisher_seconds:.1f} s  shape {F_cmb_full.shape}  "
          f"min eig {float(np.linalg.eigvalsh(F_cmb_full).min()):.6g}",
          flush=True)

    # Keep the PRE-dedupe block alive so the dedupe's effect on the shared-basis
    # widths can be MEASURED below rather than asserted in prose.
    F_cmb_full_pre_dedupe = F_cmb_full
    F_cmb_full, prior_policy = apply_shared_prior_dedupe(pieces, F_cmb_full)

    print("Diagnosing the observed-Hessian negative mode (per-term "
          "attribution)...", flush=True)
    per_term_observed = {
        t: (built["per_term"][t] if built["method"][t] == "hessian"
            else observed_hessian_fisher(pieces, t))
        for t in built["per_term"]}
    negative_mode = diagnose_negative_mode(
        per_term_observed, len(cfg["cosmo_keys"]))
    print(f"[diagnose] observed-Hessian marginalized min eig = "
          f"{negative_mode['marginalized_min_eig']:.6g}", flush=True)
    for term_name, value in negative_mode["per_term"].items():
        print(f"[diagnose]   {term_name:16s} {value:+.6g}", flush=True)
    print(f"[diagnose]   {'SUM':16s} "
          f"{negative_mode['attribution_sums_to']:+.6g}", flush=True)
    if args.diagnose_negative_mode:
        print(json.dumps(negative_mode, indent=2), flush=True)

    cmb_cosmo_idx = list(range(len(cfg["cosmo_keys"])))
    F_cmb_cosmo = marginalized_fisher_block(F_cmb_full, cmb_cosmo_idx)
    print(f"CMB cosmo-only Fisher shape: {F_cmb_cosmo.shape}", flush=True)

    fid_cmb_cosmo = jnp.array(
        [pieces["fiducial_cosmo_cmb"][k] for k in cfg["cosmo_keys"]],
        dtype=jnp.float64)
    F_cmb_shared, fid_cmb_derived, jacobian, _cov = project_fisher_to_derived(
        F_cmb_cosmo, fid_cmb_cosmo, make_cmb_to_shared(cosmology))
    print(f"Fiducial (shared): {fid_cmb_derived}", flush=True)

    sigma_tau = gate_g1_sigma_tau(F_cmb_shared, shared_keys)
    min_eig, max_eig = gate_g2_positive_definite(F_cmb_shared)
    jac_entry = gate_g3_jacobian(jacobian, cosmology)

    # How much of the shared-basis widths the shared-prior dedupe accounts for.
    # Removing count-1 duplicate copies of a Gaussian prior REMOVES curvature,
    # so every post-dedupe width is >= its pre-dedupe value and shift_pct >= 0.
    sig_pre = shared_basis_marginal_sigmas(
        F_cmb_full_pre_dedupe, cosmology, pieces["fiducial_cosmo_cmb"])
    sig_post = shared_basis_marginal_sigmas(
        F_cmb_full, cosmology, pieces["fiducial_cosmo_cmb"])
    dedupe_width_effect = {
        "basis": list(shared_keys),
        "pre_dedupe_marginal_sigmas": [float(v) for v in sig_pre],
        "post_dedupe_marginal_sigmas": [float(v) for v in sig_post],
        "shift_pct": [float(100.0 * (post / pre - 1.0))
                      for pre, post in zip(sig_pre, sig_post)],
        "definition": (
            "sqrt(diag(inv(F_shared))) of the nuisance-marginalized, "
            "shared-basis block built from the summed per-term Fisher BEFORE "
            "and AFTER the duplicate shared-prior curvature is subtracted, in "
            "`basis` order; shift_pct = 100 * (post/pre - 1), i.e. how much "
            "WIDER each marginal gets once the A_planck prior is counted once "
            "instead of four times."),
    }
    for key, pre, post, shift in zip(shared_keys, sig_pre, sig_post,
                                     dedupe_width_effect["shift_pct"]):
        print(f"[dedupe-width] {key:6s} sigma {pre:.6g} -> {post:.6g}  "
              f"({shift:+.3f}%)", flush=True)

    cmb_config_hash, hash_components = compute_cmb_config_hash(
        cosmology, method_per_term=built["method"], prior_policy=prior_policy,
        fiducial_native=pieces["fiducial_cosmo_cmb"], shared_keys=shared_keys,
        native_keys=cfg["cosmo_keys"])
    print(f"[fingerprint] cmb_config_hash = {cmb_config_hash}", flush=True)
    pinned = (stream_common.CMB_CONFIG_HASH_LCDM if cosmology == "lcdm"
              else stream_common.CMB_CONFIG_HASH_NULCDM)
    # BEFORE the write: a mismatched pin exits non-zero here, so a drifted
    # environment cannot replace a good artifact with an unloadable one.
    bootstrap_pin = enforce_cmb_config_hash_pin(
        cosmology, cmb_config_hash, pinned=pinned)

    # Attached AFTER the fingerprint is computed, deliberately: these widths are
    # an OUTPUT of the block, never an input that determines it, and must not
    # perturb cmb_config_hash (which reads only prior_policy's inventory
    # fields). Ordering it this way makes that impossible by construction.
    prior_policy["dedupe_width_effect"] = dedupe_width_effect

    meta = {
        "cosmology": cosmology,
        "shared_keys": list(shared_keys),
        "native_keys": list(cfg["cosmo_keys"]),
        "terms": terms,
        "emulator_files": cfg["emulator_filenames"],
        "theory_config_hash": cfg["theory_config_hash"],
        "build_command": build_command,
        "include_internal_priors": INCLUDE_INTERNAL_PRIORS,
        "n_nuisance": len(pieces["sampled_nuisance"]),
        "nuisance_names": list(pieces["sampled_nuisance"]),
        "fisher_seconds": round(fisher_seconds, 1),
        "cmb_config_hash": cmb_config_hash,
        "cmb_config_hash_components": hash_components,
        "prior_policy": prior_policy,
        "method": {
            "per_term": built["method"],
            "sources": built["sources"],
            "gn_algorithm_version": cmb_gn_fisher.GN_ALGORITHM_VERSION,
            "rationale": (
                "Hybrid expected-Fisher build. Terms whose data model is "
                "Gaussian in band powers contribute the Gauss-Newton expected "
                "Fisher J^T C^-1 J plus their internal nuisance-prior "
                "curvature; each reconstructed Gaussian form is validated "
                "against the untouched candl/clipy log_like in value, full "
                "Hessian and along the reference minimum-eigenvalue direction "
                "before use (see method.gn_validation for the measured "
                "errors). The two low-ell terms are non-Gaussian likelihoods "
                "(planck_lowl_tt = Gibbs / Blackwell-Rao cl2x spline, "
                "planck_lowl_ee = simall tabulated probability spline) for "
                "which J^T C^-1 J does not exist, so they keep the observed "
                "Hessian -0.5 (H + H^T). Motivation: the observed Hessian of a "
                "real-data likelihood evaluated away from its own maximum "
                "carries an indefinite residual-curvature term that dominates "
                "near-null directions; see method.negative_mode_attribution "
                "for the per-term split MEASURED during this build. No "
                "eigenvalue clipping or regularization is applied anywhere."
            ),
            "negative_mode_attribution": negative_mode,
            "gn_validation": built["reports"],
        },
        "gates": {
            "G1a_lowl_ee_dtau": g_tau,
            "G1b_sigma_tau": sigma_tau,
            "G1b_sigma_tau_range": list(SIGMA_TAU_RANGE),
            "G2_min_eig": min_eig,
            "G2_max_eig": max_eig,
            "G3_dh_dH0": jac_entry,
        },
    }
    out_path = cmb_fisher_path(cosmology)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             F_cmb_shared=np.asarray(F_cmb_shared),
             fid_shared=np.asarray(fid_cmb_derived),
             shared_keys=np.array(list(shared_keys)),
             F_cmb_native=np.asarray(F_cmb_cosmo),
             fid_native=np.asarray(fid_cmb_cosmo),
             native_keys=np.array(list(cfg["cosmo_keys"])),
             sigma_tau=np.float64(sigma_tau),
             meta_json=json.dumps(meta))
    print(f"[saved] {out_path}", flush=True)

    verify_artifact_round_trip(cosmology, bootstrap_pin=bootstrap_pin,
                               cmb_config_hash=cmb_config_hash)
    print(f"[total] {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
