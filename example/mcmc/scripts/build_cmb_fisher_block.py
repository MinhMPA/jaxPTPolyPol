#!/usr/bin/env python3
"""Build the fiducial-centered Gaussian CMB Fisher block artifact.

Ports cells 3/13/14/16 of ``example/fisher/fisher_joint_PFS_BAO_CMB_{LCDM,
nuLCDM}.ipynb``:

  1. load the 5 candl/clipy likelihood terms (Planck highl TTTEEE, lowl TT,
     lowl EE simall, Planck lensing, ACT DR6 lensing) with internal priors ON;
  2. joint loglike over (COSMO_KEYS_CMB + sampled CMB nuisances) at the
     fiducial;
  3. ``F_full = -0.5 * (H + H.T)`` with ``H = jax.hessian(joint_loglike)
     (theta_fid)``;
  4. Schur-marginalize the nuisances -> cosmo-native block;
  5. project H0 -> h into the shared basis (ombh2, omch2, logA, ns, h, tau
     [, mnu]);
  6. HARD GATES (abort loudly, no fallback):
       G1 (E1): |grad of the lowl_EE term wrt tau| > 0 at the fiducial, AND
                sigma_tau = sqrt(inv(F_shared)[tau, tau]) in [0.004, 0.02];
       G2 (E2): min(eigvals(F_shared)) > 0;
       G3 (E3): the d(shared)/d(native) Jacobian has J[h_row, H0_col] == 0.01;
  7. save the npz + META (incl. gate results and the exact build command).

Usage: ``python3 build_cmb_fisher_block.py --cosmology {lcdm,nulcdm}
[--dry-run]``. ``--dry-run`` runs steps 1-2 shapes-only (no Hessian), prints the
layout, exits 0.

The output artifact is a GAUSSIAN summary of the CMB posterior centered on the
FIDUCIAL cosmology -- a forecast object, not a fit to the real Planck/ACT data
vector. Tasks 3-4 consume it through ``stream_common.load_cmb_fisher_block``.

Every path/key constant below is copied VERBATIM from the notebook cells named
above, and :func:`verify_notebook_provenance` re-reads those notebooks at run
time and refuses to build unless each literal is still found in the source cell
-- so this script cannot silently drift from the Fisher notebooks it mirrors.
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import candl_data

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
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
    make_joint_loglike_fn,
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
ACT_DR6_LENS   = candl_data.ACT_DR6_Lens_only

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
            ACT_DR6_LENS, lensing=True, feedback=False,
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
    joint_cmb_loglike = make_joint_loglike_fn(
        extra_loglike_fns=[term_loglikes[name] for name in term_loglikes],
    )
    return {
        "layout": layout, "theta_fid": theta_fid_cmb,
        "sampled_nuisance": sampled_nuisance,
        "term_loglikes": term_loglikes,
        "joint_loglike": joint_cmb_loglike,
        "fiducial_cosmo_cmb": fiducial_cosmo_cmb,
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
    """G2 (E2): the shared-basis CMB Fisher must be positive definite."""
    eigvals = np.linalg.eigvalsh(np.asarray(F_shared))
    min_eig = float(eigvals.min())
    if not min_eig > 0.0:
        sys.exit(f"ABORT G2 (E2): min eigenvalue of F_shared = {min_eig:.6g} "
                 "<= 0 -- the projected CMB Fisher is not positive definite.")
    print(f"[G2]  PASS  min eig(F_shared) = {min_eig:.6g}  "
          f"(max {float(eigvals.max()):.6g}, cond "
          f"{float(eigvals.max() / min_eig):.6g})", flush=True)
    return min_eig, float(eigvals.max())


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
# Main.
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cosmology", required=True, choices=("lcdm", "nulcdm"))
    parser.add_argument("--dry-run", action="store_true",
                        help="steps 1-2 only (no Hessian); print the layout")
    args = parser.parse_args()
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
        print("[dry-run] layout OK; exiting WITHOUT building the Hessian.",
              flush=True)
        return 0

    # G1a before the (minutes-scale) Hessian -- fail fast on a dead spline.
    g_tau = gate_g1_tau_gradient(pieces, cosmology)

    print("Computing CMB Hessian (this may take a few minutes)...", flush=True)
    t_hess = time.time()
    hess = np.asarray(jax.jit(jax.hessian(pieces["joint_loglike"]))(theta_fid))
    hess_seconds = time.time() - t_hess
    print(f"[hessian] {hess_seconds:.1f} s  shape {hess.shape}", flush=True)

    F_cmb_full = -0.5 * (hess + hess.T)
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
        "hessian_seconds": round(hess_seconds, 1),
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

    # Read-back through the production loader: the artifact must satisfy the
    # very guards its consumers apply (cosmology / shared_keys / hash).
    loaded = stream_common.load_cmb_fisher_block(cosmology)
    print(f"[loader] round-trip OK: F_shared {tuple(loaded['F_shared'].shape)}, "
          f"sigma_tau {loaded['sigma_tau']:.6g}", flush=True)
    print(f"[total] {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
