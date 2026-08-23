"""Production 7-bin Taylor-template build for the LCDM P+B + BAO marginal posterior.

Replicates the theory / data / covariance / prior configuration of
``example/mcmc/mcmc_joint_PFS_BAO_BBN_ns_LCDM.ipynb`` EXACTLY (every literal
below is copied from that notebook, with a comment naming the source cell),
then:

1. builds the noiseless mock data vector (via the JITTED monolith joint_fn --
   jit-vs-eager matters: the eager ``_ap_is_identity`` shortcut differs by
   ~1e-3 at the fiducial), per-bin covariance inverses, BAO arrays, priors and
   the sampled/marginalized split;
2. computes the full Fisher (monolith jacfwd, measured 77.6 GB peak) + BAO +
   priors, Schur-reduces to the sampled theta_NL block ``F_nl_prior``, and
   saves the whitening + every posterior input the validation script needs to
   ``cache/taylor_whitening_lcdm.npz``;
3. runs ``build_taylor_templates`` (chunk_J=4, chunk_H=2; predicted 18-25 GB)
   at theta0 = fid_nl and saves ``cache/taylor_templates_lcdm.npz`` with a
   native-scalar config meta stamp, plus ``cache/taylor_build_summary.json``
   with stage wall times, peak RSS and the Hessian symmetry errors.

Run from ``example/mcmc`` (the BAO loader path is relative to that directory),
ideally under ``caffeinate`` so the laptop cannot sleep mid-build::

    cd example/mcmc
    caffeinate -i python3 scripts/build_taylor_templates_lcdm.py

An RSS watchdog thread (samples ``ps -o rss=`` every 2 s) hard-aborts the
process if resident memory exceeds the per-stage limit: 105 GB for both the
Fisher stage and the template stage (the machine has 128 GB). Both limits are
measured, not guessed: the planned 90 GB Fisher limit aborted at 91.5 GB
during the fresh-process jacfwd compile transient (the 77.6 GB reference was
taken inside an already-warm notebook kernel), and the planned 70 GB template
limit aborted at 83.8 GB (chunk_H=2) and again at 80.8 GB (chunk_H=1) during
the first second-order H chunk -- the H build's inner full-d Jacobian pass
sets an ~82 GB floor that no chunk knob reduces. To redo only the template
stage (e.g. after a template-stage abort), re-run in a fresh process with
``--templates-only``; the whitening npz from the earlier run is kept.
"""

from functools import partial
import gc
import json
import os
import pathlib
import subprocess
import sys
import threading
import time

N_THREADS = os.cpu_count() or 1
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(N_THREADS)

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
# Persistent XLA compilation cache, as in the notebook (cell 1): repeated runs
# reuse compiled graphs.
jax.config.update("jax_compilation_cache_dir",
                  str(pathlib.Path.home() / ".jax_xla_cache"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 10)
import jax.numpy as jnp
from jax.scipy.linalg import inv

from ps_1loop_jax import background as bg

from stream_common import (
    BACKGROUND_MODE, BK_DO_IRRES, DEFAULT_BAO_DATA_DIR, FIDUCIAL, FIXED_COSMO, K_BK_MAX,
    K_BK_MIN, K_NL_RSD, K_PK_MAX, K_PK_MIN, META, MNU_FIXED, N_GL, N_K,
    NULCDM_EMULATOR, NULCDM_FIDUCIAL, NUM_MU, NUM_PHI, PFS_EMULATOR, V_bins,
    knl_bins, meta_for, n_bar, n_zbins, template_meta_for, z_bins,
)

from jaxptpolypol import (
    LIN_SURVEY_KEYS,
    bin_lin_slices,
    build_taylor_templates,
    save_taylor_templates,
    split_marginal_indices,
)
from jaxptpolypol.bao import (
    add_bao_to_fullshape_fisher,
    bao_fisher_matrix,
    load_desi_dr2,
    make_bao_theory_fn,
)
from jaxptpolypol.inference import (
    fisher_matrix,
    gaussian_prior_fisher,
    marginalize_fisher,
    marginalized_fisher_block,
)
from jaxptpolypol.model import BispectrumTreeModel, CosmoEmulator, PS1LoopModel
from jaxptpolypol.params import (
    CosmoParams,
    FullShapeSurveyParams,
    pack_joint_params,
)
from jaxptpolypol.priors import build_prior_sigmas_from_spec, load_prior_spec
from jaxptpolypol.sampler import make_full_params_fn
from jaxptpolypol.theory import (
    build_bispectrum_triangles_from_k_grid,
    compute_fiducial_distances,
    make_gaussian_joint_covariance_fn,
    make_joint_pk_bk_bin_fn,
    make_joint_pk_bk_fn,
)

TEMPLATES_ONLY = "--templates-only" in sys.argv[1:]
# --c1-sampled: build the Tier-3 sampled-c1 templates. c1 leaves the
# marginalized block (10 lin/bin) and joins theta_NL (n_nl 26 -> 33 at 7 bins);
# the theory is exactly quadratic in c1, so the order-2 m0 surrogate carries the
# c1^2 the marginalized (linearized) path drops. Distinct output filenames +
# c1_treatment meta stamp; everything else (theory/grid/whitening machinery) is
# identical to the marginalized build. See theory.md: Why c1 sits in the
# linear block.
C1_SAMPLED = "--c1-sampled" in sys.argv[1:]
C1_TREATMENT = "sampled" if C1_SAMPLED else "marginalized"
_C1_KEY = ('bk', 'ctr', 'c1')
LIN_SURVEY_KEYS_BUILD = (tuple(k for k in LIN_SURVEY_KEYS if k != _C1_KEY)
                         if C1_SAMPLED else LIN_SURVEY_KEYS)


def _arg_value(flag, default):
    """Read the value after ``flag`` in argv, or ``default`` if absent."""
    args = sys.argv[1:]
    if flag in args:
        i = args.index(flag)
        if i + 1 >= len(args):
            sys.exit(f"{flag} requires a value")
        return args[i + 1]
    return default


# --cosmology {lcdm,nulcdm}: default lcdm keeps the existing behavior
# byte-identical. nulcdm swaps in the mnu linear-Pk emulator and adds mnu to the
# sampled cosmology basis (packed index 9, theta_NL position 5; n_nl 26 -> 27).
# The theory/covariance/BAO closures auto-trace mnu once it is in cosmo.param_keys
# ("mnu" in cosmo_keys -> has_mnu path), so no mnu_fixed pathway feeds the
# varied-mnu basis. See stream_common's SHARED_KEYS_NU / NULCDM_FIDUCIAL block.
COSMOLOGY = _arg_value("--cosmology", "lcdm")
if COSMOLOGY not in ("lcdm", "nulcdm"):
    sys.exit(f"--cosmology must be 'lcdm' or 'nulcdm', got {COSMOLOGY!r}.")
# --dry-run: build every closure + the split, run the construction-time wiring
# checks and ONE bin's forward theory eval, print the would-be META, then exit
# WITHOUT the (heavy) Fisher/whitening or Taylor-template stages.
DRY_RUN = "--dry-run" in sys.argv[1:]

# The sampled-c1 nuLCDM split (n_nl 27 -> 34) is untested and outside the
# replication plan (marginalized c1 only); refuse it rather than silently emit a
# wrong cache.
if C1_SAMPLED and COSMOLOGY == "nulcdm":
    sys.exit("--c1-sampled is not supported with --cosmology nulcdm: the "
             "sampled-c1 nuLCDM split is untested and outside the replication "
             "plan (marginalized c1 only). Drop --c1-sampled.")

EMULATOR_PATH = NULCDM_EMULATOR if COSMOLOGY == "nulcdm" else PFS_EMULATOR

# ---------------------------------------------------------------------------
# RSS watchdog: sample `ps -o rss=` in a thread, hard-abort above the stage
# limit, and print a heartbeat every ~60 s so the log shows liveness.
# ---------------------------------------------------------------------------

_T0 = time.perf_counter()
_watch = {"stage": "init", "limit_gb": 90.0, "peak_gb": 0.0,
          "stage_peak_gb": 0.0}
_stage_peaks = {}


def _rss_gb():
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                         capture_output=True, text=True).stdout.strip()
    return int(out) / 1048576 if out else 0.0


def _watchdog():
    n = 0
    while True:
        gb = _rss_gb()
        _watch["peak_gb"] = max(_watch["peak_gb"], gb)
        _watch["stage_peak_gb"] = max(_watch["stage_peak_gb"], gb)
        _stage_peaks[_watch["stage"]] = max(
            _stage_peaks.get(_watch["stage"], 0.0), gb)
        if gb > _watch["limit_gb"]:
            print(f"!!! RSS WATCHDOG ABORT: {gb:.1f} GB > "
                  f"{_watch['limit_gb']:.0f} GB limit during stage "
                  f"'{_watch['stage']}' ({time.perf_counter() - _T0:.0f}s in). "
                  "Aborting to protect the machine. If this fired in the "
                  "template stage of a combined run, residual heap from the "
                  "Fisher stage may be the cause: re-run with "
                  "--templates-only in a fresh process.", flush=True)
            os._exit(17)
        n += 1
        if n % 30 == 0:
            print(f"  [watchdog] stage='{_watch['stage']}' rss={gb:.1f} GB "
                  f"(stage peak {_watch['stage_peak_gb']:.1f}, limit "
                  f"{_watch['limit_gb']:.0f}) t={time.perf_counter() - _T0:.0f}s",
                  flush=True)
        time.sleep(2.0)


threading.Thread(target=_watchdog, daemon=True).start()


def set_stage(name, limit_gb):
    _watch["stage"] = name
    _watch["limit_gb"] = limit_gb
    _watch["stage_peak_gb"] = _rss_gb()
    print(f"\n===== STAGE: {name} (RSS abort {limit_gb:.0f} GB; now "
          f"{_watch['stage_peak_gb']:.1f} GB; t={time.perf_counter() - _T0:.0f}s) "
          "=====", flush=True)


# ---------------------------------------------------------------------------
# Configuration. The production constants (fiducial cosmology, redshift/volume/
# knl/nbar bins, k-grid, quadrature, emulator path, META) originate in
# mcmc_joint_PFS_BAO_BBN_ns_LCDM.ipynb, cell "Configuration (mirrors the Fisher
# notebook)"; they are IMPORTED from stream_common (the single source of truth)
# rather than re-declared here. They used to be a byte-identical copy, which
# made the theory_config_hash stamped below a hash of a config that could
# silently differ from the one every consumer expects -- i.e. a guard comparing
# two different worlds. Only the build-only knobs stay local.
# ---------------------------------------------------------------------------

BB_POWER_MODEL = 'kaiser'
BAO_DATA_DIR = DEFAULT_BAO_DATA_DIR   # chdir-sensitive: run from example/mcmc

COSMO_PRIORS = {'ombh2': 0.00055, 'ns': 0.042}  # BBN + ns10 (arXiv:2411.12022)

# Taylor-build knobs (the config stamp itself, META, is imported above).
# MEASURED (2026-07-29): the first H (jacfwd-of-jacfwd) chunk spiked to
# 83.8 GB at chunk_H=2 and 80.8 GB at chunk_H=1 -- the planned 18-25 GB
# prediction did not hold, and lowering chunk_H does NOT reduce the peak,
# because the builder's inner m0 Jacobian is a full-d (26-tangent) pass
# regardless of chunk_H. chunk_H=2 is kept (halves the outer-chunk count,
# ~2x faster H build, same memory); the stage watchdog is set to 105 GB
# below instead of the planned 70 GB. J chunks (4-wide) stayed under 34 GB.
CHUNK_J, CHUNK_H = 4, 2

# --- Template vs whitening meta stamps -------------------------------------
# Both stamps carry the c1 treatment ("marginalized" | "sampled") so a loaded
# cache is SELF-DESCRIBING, and the TEMPLATES stamp additionally carries the
# theory-config identifiers that determine template validity: a sha256 binding
# the full theory/grid config plus the short per-bin tuples verbatim (as
# strings). Both come from stream_common, which is also what
# load_templates_and_whitening EXPECTS by default -- producer and consumer read
# the same constants, so the hash guard compares one world with itself and a
# genuine config change hard-fails every stale cache.
#
# Templates are PRIOR-INDEPENDENT, so prior identifiers never go on the template
# stamp; they live on the WHITENING stamp (marginalized mode), where they are
# informational to consumers (marginal_taylor.compare_meta case 3).
TEMPLATE_META = template_meta_for(C1_TREATMENT, cosmology=COSMOLOGY)
# Since 2026-08-23 the whitening stamp ALSO carries theory_config_hash: the
# whitening's covariance depends on the same theory config as the templates,
# and stamping both is what lets the --templates-only gate below detect a
# whitening npz left over from a DIFFERENT theory era (e.g. the bispectrum
# IR-resummation flip). Consumers that expect only meta_for() treat the extra
# key as informational (compare_meta case 3), so old consumers keep working.
_THEORY_HASH_STAMP = TEMPLATE_META["theory_config_hash"]
if C1_SAMPLED:
    WHITENING_META = {**meta_for(C1_TREATMENT, cosmology=COSMOLOGY),
                      "theory_config_hash": _THEORY_HASH_STAMP}
else:
    WHITENING_META = {
        **meta_for(C1_TREATMENT, cosmology=COSMOLOGY),
        "theory_config_hash": _THEORY_HASH_STAMP,
        "prior_spec": "eft_eq12_2405_02252",
        "cosmo_priors": COSMO_PRIORS,
    }

#: Post-flip (2026-08-23, do_irres=True) LCDM cosmo Fisher sigmas -- the live
#: config-drift gate. Pinned 2026-08-23 from the bootstrap rebuild (s1 log;
#: tier2 pre-flip delta was max rel 2.03e-4, the do_irres physics change).
SIG_FISHER_LCDM_REF = (
    0.00047593905640853234, 0.003147992566163979, 0.04680952791742042,
    0.025500167552351714, 0.003625186448285401)

CACHE = pathlib.Path("cache")
_SUFFIX = "_c1s" if C1_SAMPLED else ""
# COSMOLOGY tags templates/whitening (lcdm -> the legacy names byte-for-byte).
# The summary keeps its untagged legacy name for lcdm; nulcdm gets a distinct
# one so a nuLCDM build never overwrites an LCDM summary.
WHITENING_PATH = CACHE / f"taylor_whitening_{COSMOLOGY}{_SUFFIX}.npz"
TEMPLATES_PATH = CACHE / f"taylor_templates_{COSMOLOGY}{_SUFFIX}.npz"
_SUMMARY_TAG = "" if COSMOLOGY == "lcdm" else f"_{COSMOLOGY}"
SUMMARY_PATH = CACHE / f"taylor_build_summary{_SUMMARY_TAG}{_SUFFIX}.json"

if not pathlib.Path(BAO_DATA_DIR).is_dir():
    sys.exit(f"BAO data dir not found at {BAO_DATA_DIR!r} -- this script must "
             "be run from example/mcmc (the notebook's relative path).")
if not pathlib.Path(EMULATOR_PATH).is_file():
    sys.exit(f"{COSMOLOGY} emulator not found at {EMULATOR_PATH!r}.")
CACHE.mkdir(exist_ok=True)
print(f"===== c1_treatment = {C1_TREATMENT} "
      f"(lin keys/bin = {len(LIN_SURVEY_KEYS_BUILD)}); outputs -> "
      f"{TEMPLATES_PATH.name}, {WHITENING_PATH.name} =====", flush=True)
if TEMPLATES_ONLY and not WHITENING_PATH.exists():
    print(f"WARNING: --templates-only but {WHITENING_PATH} does not exist; "
          "the validation script will need it -- run the full build later.",
          flush=True)
if TEMPLATES_ONLY and WHITENING_PATH.exists():
    # --templates-only reuses the existing whitening npz. That is only sound if
    # it was built under the CURRENT theory config -- after a physics flip
    # (e.g. bispectrum IR resummation, 2026-08-23) an old whitening carries a
    # covariance/pb_fid from the other era and the fresh templates would pair
    # with stale whitening SILENTLY. Hard-require the era stamp to match.
    with np.load(WHITENING_PATH) as _wz_gate:
        _stored_w = (json.loads(str(_wz_gate["meta"].item()))
                     if "meta" in _wz_gate.files else {})
    _got_w = _stored_w.get("theory_config_hash")
    if _got_w != _THEORY_HASH_STAMP:
        sys.exit(
            f"--templates-only refused: {WHITENING_PATH} stamps "
            f"theory_config_hash {_got_w!r} but the current config is "
            f"{_THEORY_HASH_STAMP!r}. The whitening npz predates (or"
            " postdates) the current theory config; run a FULL build so the"
            " templates and whitening come from the same era.")

# ---------------------------------------------------------------------------
# Stage 1: emulator, models, fiducial parameters -- copied VERBATIM from the
# notebook cell "Emulator, models, fiducial parameters".
# ---------------------------------------------------------------------------

set_stage("setup: models + fiducials", 90.0)
t_stage1 = time.perf_counter()

pklin_emulator = CosmoEmulator(probe='custom_log', emulator_path=EMULATOR_PATH)
ps1loop_model = PS1LoopModel(do_irres=True)
bispectrum_model = BispectrumTreeModel(
    do_irres=BK_DO_IRRES, do_AP=True, k_nl_rsd=K_NL_RSD)

cosmo_dict = {
    'ombh2': FIDUCIAL['ombh2'], 'omch2': FIDUCIAL['omch2'],
    'logA':  FIDUCIAL['logA'],  'ns':    FIDUCIAL['ns'], 'h': FIDUCIAL['h'],
    'z': 0.7, 'A_b': 3.13, 'eta_b': 0.603, 'logT_AGN': 7.8,
}
if COSMOLOGY == "nulcdm":
    # mnu LAST so FIXED_COSMO=(5,6,7,8) still indexes z/A_b/eta_b/logT_AGN and
    # mnu is SAMPLED at packed cosmo index 9 (mirrors the nuLCDM Fisher notebook
    # cell "Emulator and models (mnu variant)").
    cosmo_dict['mnu'] = NULCDM_FIDUCIAL['mnu']
cosmo = CosmoParams(cosmo_dict)


# -- Bias / counterterm fiducials (arXiv:1907.06666), notebook cell 5 --
def b1z(z): return 0.9 + 0.4 * z
def b2z(z): return -0.704 - 0.208 * z + 0.183 * z**2 - 0.00771 * z**3
def bG2z(z): return -(2. / 7.) * (b1z(z) - 1.)
def bGamma3z(z): return (23. / 42.) * (b1z(z) - 1.)
def Dplusz(z):
    return float(bg.growth_factor(
        cosmo_dict['ombh2'], cosmo_dict['omch2'], cosmo_dict['h'], z,
        mnu=cosmo_dict.get('mnu', MNU_FIXED)))
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
packed_params = pack_joint_params(cosmo, surveys)
n_cosmo_params = sum(cosmo.param_sizes)
n_survey_params = len(joint_survey_keys)
Hz_fid, DAz_fid = compute_fiducial_distances(cosmo, z_bins)
print(f"Packed params: {packed_params.shape[0]} ({n_cosmo_params} cosmo + "
      f"{n_survey_params}x{n_zbins} survey)", flush=True)

# Theory config -- single source of truth, notebook cell "Joint P+B theory".
joint_theory_kwargs = dict(
    pklin_emulator=pklin_emulator, ps1loop_model=ps1loop_model,
    bispectrum_model=bispectrum_model,
    cosmo_keys=cosmo.param_keys, cosmo_sizes=cosmo.param_sizes,
    survey_keys=joint_survey_keys,
    ap=True, z_bins=z_bins, Hz_fid=Hz_fid, DAz_fid=DAz_fid,
    n_gl=N_GL, num_mu=NUM_MU, num_phi=NUM_PHI,
    background_mode=BACKGROUND_MODE)

k = jnp.linspace(K_PK_MIN, K_PK_MAX, N_K)
dk = float(k[1] - k[0]); n_k = int(k.shape[0])
triangles, triangle_dk = build_bispectrum_triangles_from_k_grid(
    k, k_min=K_BK_MIN, k_max=K_BK_MAX, dk=dk)
n_tri = int(triangles.shape[0])
block_len = 3 * n_k + n_tri
if n_k != META["n_k"] or n_tri != META["n_tri"]:
    sys.exit(f"Config self-check failed: n_k={n_k}, n_tri={n_tri} vs META "
             f"{META['n_k']}/{META['n_tri']}.")
print(f"P grid n_k={n_k}, dk={dk:.5f}; B triangles={n_tri}; "
      f"per-bin block=3*{n_k}+{n_tri}={block_len}", flush=True)

# Sampled/marginalized split -- notebook cell "Varied block, comparison
# Fisher, and whitening scales". FIXED_COSMO=(5,6,7,8) is cosmology-independent
# (z/A_b/eta_b/logT_AGN); the varied-cosmo set is derived from n_cosmo_params, so
# nulcdm's mnu (packed index 9) enters cosmo_varied_global automatically.
fixed_cosmo = list(FIXED_COSMO)
split = split_marginal_indices(
    n_cosmo_params=n_cosmo_params, survey_keys=joint_survey_keys,
    n_bins=n_zbins, fixed_cosmo=fixed_cosmo,
    fixed_survey_keys={('shared', 'k_nl', None), ('shared', 'ndens', None)},
    lin_survey_keys=LIN_SURVEY_KEYS_BUILD)
n_nl = split.n_nl
varied_idx = sorted(list(split.nl_idx) + list(split.lin_idx))
cosmo_varied_global = [i for i in range(n_cosmo_params) if i not in fixed_cosmo]
nl_pos = {full_idx: pos for pos, full_idx in enumerate(split.nl_idx)}
cosmo_nl_pos = [nl_pos[i] for i in cosmo_varied_global]
fid_nl = packed_params[jnp.array(split.nl_idx)]
bin_lin_idx = [split.lin_idx[sl] for sl in bin_lin_slices(split, n_zbins)]
print(f"n_NL = {n_nl} ({len(cosmo_varied_global)} cosmo + {3 * n_zbins} bias), "
      f"n_lin marginalized = {split.n_lin}", flush=True)

# --- Build-time shape guards (cosmology-aware) --------------------------------
# nl-survey params per bin: b1/b2/bG2 (+ c1 when it is SAMPLED). Everything below
# is derived from the config so it holds for both cosmologies and both c1 modes.
_n_nl_survey_per_bin = 3 + (1 if C1_SAMPLED else 0)
_n_nl_expected = len(cosmo_varied_global) + _n_nl_survey_per_bin * n_zbins
assert n_nl == _n_nl_expected, (
    f"n_nl {n_nl} != expected {_n_nl_expected} "
    f"({len(cosmo_varied_global)} cosmo + {_n_nl_survey_per_bin}x{n_zbins} nl-survey)")
_c1_off = joint_survey_keys.index(_C1_KEY)
_c1_global = [n_cosmo_params + b * n_survey_params + _c1_off for b in range(n_zbins)]
if C1_SAMPLED:
    assert set(_c1_global) <= set(split.nl_idx), "sampled c1 must live in theta_NL"
else:
    assert set(_c1_global) <= set(split.lin_idx), "marginalized c1 must live in theta_lin"
if COSMOLOGY == "nulcdm":
    assert n_nl == 27, f"nuLCDM n_nl must be 27, got {n_nl}"
    assert split.n_lin == 77, f"nuLCDM n_lin must be 77, got {split.n_lin}"
    _mnu_global = list(cosmo.param_keys).index('mnu')
    assert _mnu_global == 9, f"mnu packed index {_mnu_global} != 9"
    assert nl_pos[_mnu_global] == 5, f"mnu theta_NL position {nl_pos[_mnu_global]} != 5"
    assert cosmo_nl_pos == [0, 1, 2, 3, 4, 5], cosmo_nl_pos

t_stage1 = time.perf_counter() - t_stage1

# --- Construction-time wiring proof (--dry-run) -------------------------------
# Build one bin's forward theory fn and evaluate it ONCE at the fiducial (a single
# eager pass, seconds -- NOT the Jacobian/Hessian build), assert finiteness, print
# the would-be META, then exit before the heavy Fisher/whitening + template stages.
if DRY_RUN:
    set_stage("dry-run: one-bin theory eval + wiring proof", 90.0)
    bin0_fn = jax.jit(make_joint_pk_bk_bin_fn(bin_index=0, **joint_theory_kwargs))
    pb0 = np.asarray(bin0_fn(packed_params, k=k, triangles=triangles))
    finite = bool(np.all(np.isfinite(pb0)))
    print(f"[dry-run] cosmology={COSMOLOGY} c1_treatment={C1_TREATMENT}", flush=True)
    print(f"[dry-run] n_cosmo_params={n_cosmo_params} "
          f"cosmo.param_keys={tuple(cosmo.param_keys)}", flush=True)
    print(f"[dry-run] n_nl={n_nl} n_lin={split.n_lin} "
          f"cosmo_varied_global={cosmo_varied_global} "
          f"cosmo_nl_pos={cosmo_nl_pos}", flush=True)
    print(f"[dry-run] c1 global idx={_c1_global} -> "
          f"{'theta_NL' if C1_SAMPLED else 'theta_lin'}", flush=True)
    if COSMOLOGY == "nulcdm":
        _mg = list(cosmo.param_keys).index('mnu')
        print(f"[dry-run] mnu packed idx={_mg} -> theta_NL pos {nl_pos[_mg]} "
              f"(fid_nl[{nl_pos[_mg]}]={float(fid_nl[nl_pos[_mg]]):.4f})", flush=True)
    print(f"[dry-run] bin-0 theory eval: shape={pb0.shape} finite={finite} "
          f"range=[{pb0.min():.4e}, {pb0.max():.4e}]", flush=True)
    assert finite, "bin-0 theory eval produced non-finite entries"
    print(f"[dry-run] would-be outputs: {TEMPLATES_PATH.name}, "
          f"{WHITENING_PATH.name}, {SUMMARY_PATH.name}", flush=True)
    print(f"[dry-run] TEMPLATE_META  = {json.dumps(TEMPLATE_META)}", flush=True)
    print(f"[dry-run] WHITENING_META = {json.dumps(WHITENING_META)}", flush=True)
    print("[dry-run] wiring proof OK; exiting WITHOUT building.", flush=True)
    sys.exit(0)

# ---------------------------------------------------------------------------
# Stage 2: data vector + covariance + Fisher + whitening (skipped by
# --templates-only). Literals from notebook cells "Joint P+B theory,
# covariance, and the data Fisher", "DESI DR2 BAO", "Priors", "Varied block"
# and "Combined likelihood".
# ---------------------------------------------------------------------------

t_stage2 = 0.0
if not TEMPLATES_ONLY:
    # 105 GB, not the planned 90: the fresh-process jacfwd compile transient
    # measured 91.5 GB (watchdog abort on the first attempt); the warm-kernel
    # notebook reference was 77.6 GB. 105 GB leaves ~23 GB headroom on 128 GB.
    set_stage("data vector + Fisher + whitening", 105.0)
    t_stage2 = time.perf_counter()

    joint_fn = make_joint_pk_bk_fn(**joint_theory_kwargs)
    joint_cov_fn = make_gaussian_joint_covariance_fn(
        pklin_emulator=pklin_emulator, ps1loop_model=ps1loop_model,
        cosmo_keys=cosmo.param_keys, cosmo_sizes=cosmo.param_sizes,
        survey_keys=joint_survey_keys,
        ap=True, z_bins=z_bins, Hz_fid=Hz_fid, DAz_fid=DAz_fid,
        bb_power_model=BB_POWER_MODEL, n_gl=N_GL,
        background_mode=BACKGROUND_MODE)
    jitted_joint_fn = jax.jit(joint_fn)

    t0 = time.perf_counter()
    jitted_joint_fn(packed_params, k=k, triangles=triangles).block_until_ready()
    print(f"joint_fn warmup compile: {time.perf_counter() - t0:.1f}s "
          f"(rss {_rss_gb():.1f} GB)", flush=True)

    # Noiseless mock data vector -- ALWAYS via the jitted fn (eager differs
    # ~1e-3 at the fiducial through the _ap_is_identity shortcut).
    pb_fid = jitted_joint_fn(packed_params, k=k, triangles=triangles)

    t0 = time.perf_counter()
    jac = jax.jacfwd(jitted_joint_fn, argnums=0)(
        packed_params, k=k, triangles=triangles)
    jax.block_until_ready(jac)
    print(f"Fisher jacfwd ({packed_params.shape[0]} tangents x {n_zbins} bins): "
          f"{time.perf_counter() - t0:.1f}s (rss {_rss_gb():.1f} GB)",
          flush=True)

    gauss_cov = joint_cov_fn(packed_params, V_survey=V_bins, k=k, dk=dk,
                             triangles=triangles, triangle_dk=triangle_dk)
    F_pfs_full = fisher_matrix(gauss_cov, jac)

    # -- DESI DR2 BAO (notebook cell 9) --
    bao_dr2 = load_desi_dr2("all", data_dir=BAO_DATA_DIR)
    fiducial_cosmo = packed_params[:n_cosmo_params]
    F_bao = bao_fisher_matrix(bao_dr2, fiducial_cosmo,
                              cosmo_keys=cosmo.param_keys,
                              cosmo_sizes=cosmo.param_sizes,)
    F_pfs_bao_full = add_bao_to_fullshape_fisher(F_pfs_full, F_bao,
                                                 n_cosmo=n_cosmo_params)
    bao_theory_fn = make_bao_theory_fn(
        bao_dr2, cosmo_keys=cosmo.param_keys, cosmo_sizes=cosmo.param_sizes,
        mnu_fixed=MNU_FIXED)
    bao_fid = bao_theory_fn(fiducial_cosmo)
    bao_cov_inv = inv(jnp.asarray(bao_dr2.cov))
    print(f"BAO data points: {len(bao_dr2.data_points)}", flush=True)

    # -- Priors: EFT/stochastic spec + BBN + ns10 (notebook cell 11) --
    prior_spec = load_prior_spec('eft_eq12_2405_02252')
    prior_sigmas = build_prior_sigmas_from_spec(
        cosmo_keys=cosmo.param_keys, cosmo_sizes=cosmo.param_sizes,
        survey_keys=joint_survey_keys,
        n_bins=n_zbins, observable='joint', spec=prior_spec,
        cosmo_priors=COSMO_PRIORS)
    # bGamma3 ~ N(fid, 1^2) (arXiv:2511.20757 Table I)
    bgamma3_off = joint_survey_keys.index(('shared', 'bias', 'bGamma3'))
    for b in range(n_zbins):
        prior_sigmas[n_cosmo_params + b * n_survey_params + bgamma3_off] = 1.0
    F_prior = gaussian_prior_fisher(packed_params.shape[0], prior_sigmas)
    F_pfs_bao_prior_full = F_pfs_bao_full + F_prior

    # -- Whitening: Schur the prior-included Fisher to theta_NL (cell 13) --
    F_pfs_bao_prior_cosmo = marginalized_fisher_block(
        marginalize_fisher(F_pfs_bao_prior_full, varied_idx),
        [varied_idx.index(i) for i in cosmo_varied_global])
    F_varied_prior = marginalize_fisher(F_pfs_bao_prior_full, varied_idx)
    nl_in_varied = [varied_idx.index(i) for i in split.nl_idx]
    F_nl_prior = marginalized_fisher_block(F_varied_prior, nl_in_varied)
    cov_nl_prior = inv(F_nl_prior)
    sig_fisher = np.sqrt(np.diag(np.asarray(inv(F_pfs_bao_prior_cosmo))))
    print("sig_fisher (cosmo):", np.array2string(sig_fisher, precision=6),
          flush=True)

    # Tripwire, rescoped 2026-08-23. The tier2 reference (cache/tier2_result
    # .json, recorded 2026-07-28) predates the bispectrum IR-resummation flip
    # (do_irres False -> True): the flip moves the B entries of the theory
    # vector (max ~4.8e-3 rel) and hence the Fisher sigmas by ~2e-4, so the
    # tier2 comparison is now a DATED PHYSICS-CHANGE RECORD, not a config-
    # identity check. tier2_result.json itself is a historical artifact of
    # the pre-flip run and is deliberately NOT rewritten. The live drift
    # guard is SIG_FISHER_LCDM_REF below: None on the bootstrap rebuild
    # (this run RECORDS the post-flip sigmas into the summary JSON; pin them
    # here immediately after), then a hard 1e-4 gate for every later build.
    # LCDM-only: the nuLCDM cosmo block is 6-wide, no reference exists.
    _t2 = CACHE / "tier2_result.json"
    if COSMOLOGY == "lcdm" and _t2.exists():
        ref = np.asarray(json.loads(_t2.read_text())["sig_fisher"])
        rel = float(np.max(np.abs(sig_fisher - ref) / ref))
        print(f"sig_fisher vs tier2 (PRE-FLIP) reference: max rel diff "
              f"{rel:.2e} -- expected ~2e-4 from the 2026-08-23 do_irres "
              "flip; this line is the dated record, not a gate.", flush=True)
    if COSMOLOGY == "lcdm":
        if SIG_FISHER_LCDM_REF is None:
            print("BOOTSTRAP: no post-flip sig_fisher pin yet; this build "
                  "records it. Pin the summary JSON's sig_fisher as "
                  "SIG_FISHER_LCDM_REF before committing.", flush=True)
            print("sig_fisher (full precision):",
                  [float(s) for s in sig_fisher], flush=True)
        else:
            _pin = np.asarray(SIG_FISHER_LCDM_REF)
            _rel_pin = float(np.max(np.abs(sig_fisher - _pin) / _pin))
            print(f"sig_fisher vs post-flip pin: max rel diff {_rel_pin:.2e}",
                  flush=True)
            if _rel_pin > 1e-4:
                sys.exit("ABORT: rebuilt Fisher sigmas disagree with the "
                         "post-flip SIG_FISHER_LCDM_REF beyond 1e-4 -- "
                         "config drift; gates would be invalid.")

    # -- Per-bin data / covariance blocks + marginalized-block priors
    #    (notebook cell "Combined likelihood") --
    bin_blocks = [slice(b * block_len, (b + 1) * block_len)
                  for b in range(n_zbins)]
    bin_data = [pb_fid[sl] for sl in bin_blocks]
    bin_cov_invs = [inv(gauss_cov[sl, sl]) for sl in bin_blocks]
    mu_p = packed_params[jnp.array(split.lin_idx)]
    sigma_p = jnp.array([prior_sigmas[i] for i in split.lin_idx])
    nl_prior_entries = [
        (nl_pos[i], float(packed_params[i]), prior_sigmas[i])
        for i in split.nl_idx if i in prior_sigmas
    ]

    np.savez(
        WHITENING_PATH,
        meta=np.asarray(json.dumps(WHITENING_META)),
        packed_params=np.asarray(packed_params),
        pb_fid=np.asarray(pb_fid),
        bin_cov_invs=np.stack([np.asarray(c) for c in bin_cov_invs]),
        bao_fid=np.asarray(bao_fid),
        bao_cov_inv=np.asarray(bao_cov_inv),
        mu_p=np.asarray(mu_p),
        sigma_p=np.asarray(sigma_p),
        nl_prior_pos=np.asarray([e[0] for e in nl_prior_entries],
                                dtype=np.int64),
        nl_prior_mean=np.asarray([e[1] for e in nl_prior_entries]),
        nl_prior_sigma=np.asarray([e[2] for e in nl_prior_entries]),
        nl_idx=np.asarray(split.nl_idx, dtype=np.int64),
        lin_idx=np.asarray(split.lin_idx, dtype=np.int64),
        cosmo_nl_pos=np.asarray(cosmo_nl_pos, dtype=np.int64),
        fid_nl=np.asarray(fid_nl),
        F_nl_prior=np.asarray(F_nl_prior),
        cov_nl_prior=np.asarray(cov_nl_prior),
        F_pfs_bao_prior_cosmo=np.asarray(F_pfs_bao_prior_cosmo),
        sig_fisher=sig_fisher,
    )
    print(f"-> {WHITENING_PATH} "
          f"({WHITENING_PATH.stat().st_size / 1048576:.1f} MB)", flush=True)

    # Free the Fisher-stage memory before the template build.
    del jac, gauss_cov, F_pfs_full, F_bao, F_pfs_bao_full, F_prior
    del F_pfs_bao_prior_full, F_varied_prior, pb_fid, bin_data, bin_cov_invs
    gc.collect()
    t_stage2 = time.perf_counter() - t_stage2

# ---------------------------------------------------------------------------
# Stage 3: Taylor-template build (chunked forward-over-forward).
# ---------------------------------------------------------------------------

# 105 GB, not the planned 70: the H build's inner full-d jacfwd peaks at
# ~81-84 GB regardless of chunk_H (both measured; see the CHUNK_H comment).
set_stage("taylor-template build", 105.0)
t_stage3 = time.perf_counter()

full_params_fn = make_full_params_fn(packed_params, split.nl_idx)
bin_theory_fns = [
    partial(make_joint_pk_bk_bin_fn(bin_index=b, **joint_theory_kwargs),
            k=k, triangles=triangles)
    for b in range(n_zbins)]


def _instrumented(fn, b):
    """Progress prints per Python-level theory trace (chunk granularity)."""
    calls = [0]

    def wrapped(p):
        calls[0] += 1
        print(f"  [tt] bin {b} theory trace #{calls[0]} "
              f"(t={time.perf_counter() - _T0:.0f}s, rss {_rss_gb():.1f} GB)",
              flush=True)
        return fn(p)

    return wrapped


tt = build_taylor_templates(
    bin_theory_fns=[_instrumented(fn, b) for b, fn in enumerate(bin_theory_fns)],
    bin_lin_idx=bin_lin_idx,
    full_params_fn=full_params_fn,
    theta0=fid_nl,
    order2_m0=True, chunk_J=CHUNK_J, chunk_H=CHUNK_H)

t_stage3 = time.perf_counter() - t_stage3
sym_errs = list(tt.build_diagnostics["H_sym_err"])
print(f"template build: {t_stage3:.1f}s; H sym_err per bin = "
      + np.array2string(np.asarray(sym_errs), precision=3), flush=True)

save_taylor_templates(tt, TEMPLATES_PATH, meta=TEMPLATE_META)
print(f"-> {TEMPLATES_PATH} "
      f"({TEMPLATES_PATH.stat().st_size / 1048576:.1f} MB)", flush=True)

summary = {
    "templates_only": TEMPLATES_ONLY,
    "cosmology": COSMOLOGY,
    "c1_treatment": C1_TREATMENT,
    "lin_keys_per_bin": len(LIN_SURVEY_KEYS_BUILD),
    "wall_s": {
        "setup": round(t_stage1, 1),
        "fisher_whitening": round(t_stage2, 1),
        "template_build": round(t_stage3, 1),
        "total": round(time.perf_counter() - _T0, 1),
    },
    "peak_rss_gb": {stage: round(v, 2) for stage, v in _stage_peaks.items()},
    "overall_peak_rss_gb": round(_watch["peak_gb"], 2),
    "H_sym_err": sym_errs,
    "chunk_J": CHUNK_J, "chunk_H": CHUNK_H,
    "n_nl": int(n_nl), "n_lin": int(split.n_lin),
    "block_len": int(block_len), "n_tri": int(n_tri),
    "meta": TEMPLATE_META,
}
SUMMARY_PATH.write_text(json.dumps(summary, indent=1))
print(f"-> {SUMMARY_PATH}", flush=True)
print(f"\nBUILD COMPLETE in {time.perf_counter() - _T0:.0f}s; overall peak RSS "
      f"{_watch['peak_gb']:.1f} GB", flush=True)
