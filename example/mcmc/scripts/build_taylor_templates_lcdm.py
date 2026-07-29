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
process if resident memory exceeds the per-stage limit: 105 GB for the Fisher
stage, 70 GB for the template stage (the machine has 128 GB). The Fisher limit
was measured, not guessed: a first run with the planned 90 GB limit aborted at
91.5 GB during the fresh-process jacfwd compile transient (the 77.6 GB
reference number was taken inside an already-warm notebook kernel). If the
template stage trips only because of residual heap left by the Fisher stage,
re-run in a fresh process with ``--templates-only`` (stage 2 is skipped; the
whitening npz from the earlier run is kept).
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

from jaxptpolypol import (
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
# Configuration -- copied VERBATIM from mcmc_joint_PFS_BAO_BBN_ns_LCDM.ipynb,
# cell "Configuration (mirrors the Fisher notebook)".
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
BB_POWER_MODEL = 'kaiser'
BACKGROUND_MODE = 'direct'

PFS_EMULATOR = '/Users/nguyenmn/cosmopower-jax-for-pfs/cosmology/jense2024/jense_2023_camb_lcdm/networks/jense_2023_camb_lcdm_Pk_lin.npz'
BAO_DATA_DIR = "../../ext_data/bao_data/desi_bao_dr2"   # chdir-sensitive: run from example/mcmc

COSMO_PRIORS = {'ombh2': 0.00055, 'ns': 0.042}  # BBN + ns10 (arXiv:2411.12022)

# Taylor-build knobs + the config stamp for the stale-template guard.
# META values MUST be native str/int/float/bool.
# chunk_H=1, not the planned 2: at chunk_H=2 the FIRST H (jacfwd-of-jacfwd)
# chunk spiked 10 -> 83.8 GB and tripped the 70 GB watchdog (measured
# 2026-07-29; the 18-25 GB prediction did not hold on production grids).
# Halving the outer tangent width is the knob the builder documents for
# exactly this; J chunks (4-wide, first order) stayed under 34 GB.
CHUNK_J, CHUNK_H = 4, 1
META = {
    "n_bins": 7, "n_k": 37, "n_tri": 264, "n_gl": 16,
    "num_mu": 65, "num_phi": 65,
    "k_min": 0.02, "k_max": 0.20, "k_bk_max": 0.08, "k_nl_rsd": 0.45,
    "order2_m0": True,
}

CACHE = pathlib.Path("cache")
WHITENING_PATH = CACHE / "taylor_whitening_lcdm.npz"
TEMPLATES_PATH = CACHE / "taylor_templates_lcdm.npz"
SUMMARY_PATH = CACHE / "taylor_build_summary.json"

if not pathlib.Path(BAO_DATA_DIR).is_dir():
    sys.exit(f"BAO data dir not found at {BAO_DATA_DIR!r} -- this script must "
             "be run from example/mcmc (the notebook's relative path).")
if not pathlib.Path(PFS_EMULATOR).is_file():
    sys.exit(f"PFS emulator not found at {PFS_EMULATOR!r}.")
CACHE.mkdir(exist_ok=True)
if TEMPLATES_ONLY and not WHITENING_PATH.exists():
    print(f"WARNING: --templates-only but {WHITENING_PATH} does not exist; "
          "the validation script will need it -- run the full build later.",
          flush=True)

# ---------------------------------------------------------------------------
# Stage 1: emulator, models, fiducial parameters -- copied VERBATIM from the
# notebook cell "Emulator, models, fiducial parameters".
# ---------------------------------------------------------------------------

set_stage("setup: models + fiducials", 90.0)
t_stage1 = time.perf_counter()

pklin_emulator = CosmoEmulator(probe='custom_log', emulator_path=PFS_EMULATOR)
ps1loop_model = PS1LoopModel(do_irres=True)
bispectrum_model = BispectrumTreeModel(do_AP=True, k_nl_rsd=K_NL_RSD)

cosmo_dict = {
    'ombh2': FIDUCIAL['ombh2'], 'omch2': FIDUCIAL['omch2'],
    'logA':  FIDUCIAL['logA'],  'ns':    FIDUCIAL['ns'], 'h': FIDUCIAL['h'],
    'z': 0.7, 'A_b': 3.13, 'eta_b': 0.603, 'logT_AGN': 7.8,
}
cosmo = CosmoParams(cosmo_dict)


# -- Bias / counterterm fiducials (arXiv:1907.06666), notebook cell 5 --
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
# Fisher, and whitening scales".
fixed_cosmo = [5, 6, 7, 8]
split = split_marginal_indices(
    n_cosmo_params=n_cosmo_params, survey_keys=joint_survey_keys,
    n_bins=n_zbins, fixed_cosmo=fixed_cosmo,
    fixed_survey_keys={('shared', 'k_nl', None), ('shared', 'ndens', None)})
n_nl = split.n_nl
varied_idx = sorted(list(split.nl_idx) + list(split.lin_idx))
cosmo_varied_global = [i for i in range(n_cosmo_params) if i not in fixed_cosmo]
nl_pos = {full_idx: pos for pos, full_idx in enumerate(split.nl_idx)}
cosmo_nl_pos = [nl_pos[i] for i in cosmo_varied_global]
fid_nl = packed_params[jnp.array(split.nl_idx)]
bin_lin_idx = [split.lin_idx[sl] for sl in bin_lin_slices(split, n_zbins)]
print(f"n_NL = {n_nl} ({len(cosmo_varied_global)} cosmo + {3 * n_zbins} bias), "
      f"n_lin marginalized = {split.n_lin}", flush=True)

t_stage1 = time.perf_counter() - t_stage1

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

    # Tripwire: the reference chain/tilt artifacts (cache/tier2_*) were built
    # with this exact configuration. If the rebuilt Fisher sigmas disagree,
    # the config has drifted and every gate comparison would be invalid.
    _t2 = CACHE / "tier2_result.json"
    if _t2.exists():
        ref = np.asarray(json.loads(_t2.read_text())["sig_fisher"])
        rel = float(np.max(np.abs(sig_fisher - ref) / ref))
        print(f"sig_fisher vs tier2 reference: max rel diff {rel:.2e}",
              flush=True)
        if rel > 1e-4:
            sys.exit("ABORT: rebuilt Fisher sigmas disagree with the tier2 "
                     "reference beyond 1e-4 -- config drift; gates would be "
                     "invalid.")

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
        meta=np.asarray(json.dumps(META)),
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

set_stage("taylor-template build", 70.0)
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

save_taylor_templates(tt, TEMPLATES_PATH, meta=META)
print(f"-> {TEMPLATES_PATH} "
      f"({TEMPLATES_PATH.stat().st_size / 1048576:.1f} MB)", flush=True)

summary = {
    "templates_only": TEMPLATES_ONLY,
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
    "meta": META,
}
SUMMARY_PATH.write_text(json.dumps(summary, indent=1))
print(f"-> {SUMMARY_PATH}", flush=True)
print(f"\nBUILD COMPLETE in {time.perf_counter() - _T0:.0f}s; overall peak RSS "
      f"{_watch['peak_gb']:.1f} GB", flush=True)
