"""Overnight delayed-acceptance (DA-MH) EXACT-TARGET chain for the 7-bin LCDM
joint P+B+BAO marginal posterior.

Purpose
-------
Retire the gate-2 REVIEW of ``scripts/taylor_surrogate_validation.py``. That
gate compared the Taylor surrogate chain against the reduced Tier-2 exact chain
whose ESS is only 30-83, so its own MC error (~0.11-0.18 sigma_F on means,
~8-13 % on widths) is TIGHTER than the gate tolerance. This driver produces a
high-ESS EXACT-target reference by running the two-stage delayed-acceptance
chain of Christen & Fox (2005): the cheap Taylor surrogate screens every
proposal (stage 1) and the expensive per-bin marginal posterior is evaluated
only on stage-1-accepted proposals (stage 2), so the chain targets the EXACT
posterior while touching the ~5 s/eval theory only ~24 % of the time.

Assembly (surrogate s(x) AND exact p(x)) is copied verbatim from
``scripts/taylor_surrogate_validation.py`` -- SAME Cholesky-whitened space
(``make_cholesky_transform(fid_nl, cov_nl_prior)``), SAME loaded data / covs /
priors / BAO. Both posteriors are the builders' ``@jax.jit`` closures and are
called directly (the jitted value differs from eager by ~1e-3 at the fiducial,
so eager is never used).

DA-MH step
----------
Inlined EXACTLY as the reviewed kernel ``jaxptpolypol.sampler.run_damh_python``
(``src/jaxptpolypol/sampler.py`` lines 1240-1261), reproduced per step so the
loop can checkpoint/resume:

    prop = cur + sigma * rng.normal(size=d)
    s_prop = surrogate(prop)                       # every proposal
    if log(rng.random()) < s_prop - s_cur:         # STAGE 1: min(1, s(y)/s(x))
        p_prop = exact(prop)                       # ONE exact eval
        stage2 = (p_prop + s_cur) - (p_cur + s_prop)   # min(1, p(y)s(x)/[p(x)s(y)])
        if log(rng.random()) < stage2:             # STAGE 2
            cur, s_cur, p_cur = prop, s_prop, p_prop
    kept[step] = cur

``p_cur`` and ``s_cur`` are cached and reused; the two Metropolis uniforms are
independent. Unlike the kernel (which seeds NumPy from a JAX key and jitters
the start), this driver uses ``numpy.random.default_rng(20260730)`` directly and
starts at ``x = 0`` for a fully deterministic, resumable segmented run.

Segmentation / checkpointing (the repo's proven overnight pattern; cf.
``cache/tier2_progress.json``): 500 steps per segment; after each segment the
cumulative draws are ``np.save``-d to ``cache/damh_chain_w.npy`` and the full
resumable state to ``cache/damh_progress.json``. Re-running RESUMES from those.

TOTAL = 22_000 steps (2_000 burn + 20_000 kept downstream). Expected ~24 %
stage-1 rate -> ~5.3k exact evals x ~5 s ~ 7.5 h.

Run from ``example/mcmc`` (detached, overnight)::

    cd example/mcmc
    caffeinate -i python3 scripts/damh_exact_chain_lcdm.py > /tmp/damh_chain.log 2>&1 &
    disown
"""

from functools import partial
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
jax.config.update("jax_compilation_cache_dir",
                  str(pathlib.Path.home() / ".jax_xla_cache"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 10)
import jax.numpy as jnp

from ps_1loop_jax import background as bg

from jaxptpolypol import (
    bin_lin_slices,
    load_taylor_templates,
    make_constant_prior_fns,
    make_marginal_log_posterior_perbin,
    make_marginal_log_posterior_taylor,
    split_marginal_indices,
)
from jaxptpolypol.bao import load_desi_dr2, make_bao_theory_fn
from jaxptpolypol.model import BispectrumTreeModel, CosmoEmulator, PS1LoopModel
from jaxptpolypol.params import (
    CosmoParams,
    FullShapeSurveyParams,
    pack_joint_params,
)
from jaxptpolypol.sampler import (
    make_cholesky_transform,
    make_full_params_fn,
    make_gaussian_log_prior,
)
from jaxptpolypol.theory import (
    build_bispectrum_triangles_from_k_grid,
    compute_fiducial_distances,
    make_joint_pk_bk_bin_fn,
)

# ---------------------------------------------------------------------------
# RSS watchdog (overnight-detached safety; the exact-posterior compile is the
# heaviest step, measured ~30 GB -- limit set well above at 90 GB).
# ---------------------------------------------------------------------------

_T0 = time.perf_counter()
_watch = {"stage": "init", "limit_gb": 90.0, "peak_gb": 0.0}


def _rss_gb():
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                         capture_output=True, text=True).stdout.strip()
    return int(out) / 1048576 if out else 0.0


def _watchdog():
    while True:
        gb = _rss_gb()
        _watch["peak_gb"] = max(_watch["peak_gb"], gb)
        if gb > _watch["limit_gb"]:
            print(f"!!! RSS WATCHDOG ABORT: {gb:.1f} GB > "
                  f"{_watch['limit_gb']:.0f} GB during '{_watch['stage']}'. "
                  "Aborting to protect the machine.", flush=True)
            os._exit(17)
        time.sleep(2.0)


threading.Thread(target=_watchdog, daemon=True).start()

# ---------------------------------------------------------------------------
# Configuration -- copied VERBATIM from taylor_surrogate_validation.py, which in
# turn mirrors mcmc_joint_PFS_BAO_BBN_ns_LCDM.ipynb.
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
BAO_DATA_DIR = "../../ext_data/bao_data/desi_bao_dr2"   # run from example/mcmc

SHARED_KEYS = ('ombh2', 'omch2', 'logA', 'ns', 'h')

META = {
    "n_bins": 7, "n_k": 37, "n_tri": 264, "n_gl": 16,
    "num_mu": 65, "num_phi": 65,
    "k_min": 0.02, "k_max": 0.20, "k_bk_max": 0.08, "k_nl_rsd": 0.45,
    "order2_m0": True,
}

# DA-MH driver parameters.
TOTAL = 22_000                 # 2_000 burn + 20_000 kept downstream
SEG = 500                      # steps per checkpoint segment
RNG_SEED = 20260730            # numpy default_rng seed (deterministic)

CACHE = pathlib.Path("cache")
WHITENING_PATH = CACHE / "taylor_whitening_lcdm.npz"
TEMPLATES_PATH = CACHE / "taylor_templates_lcdm.npz"
CHAIN_PATH = CACHE / "damh_chain_w.npy"
PROGRESS_PATH = CACHE / "damh_progress.json"

for p in (WHITENING_PATH, TEMPLATES_PATH):
    if not p.exists():
        sys.exit(f"Required artifact missing: {p} -- run from example/mcmc "
                 "after scripts/build_taylor_templates_lcdm.py.")
if not pathlib.Path(BAO_DATA_DIR).is_dir():
    sys.exit(f"BAO data dir not found at {BAO_DATA_DIR!r} -- run from "
             "example/mcmc.")

# ---------------------------------------------------------------------------
# Load artifacts (strict meta guards) and rebuild BOTH posteriors from the SAME
# loaded inputs -- assembly copied from taylor_surrogate_validation.py.
# ---------------------------------------------------------------------------

_watch["stage"] = "load + assemble posteriors"
print(f"===== {_watch['stage']} =====", flush=True)

tt = load_taylor_templates(TEMPLATES_PATH, expect_meta=META)
wz = np.load(WHITENING_PATH)
stored_meta = json.loads(str(wz["meta"].item()))
if stored_meta != META:
    sys.exit(f"Whitening npz meta mismatch:\nstored   {stored_meta}\n"
             f"expected {META}")

packed_params = jnp.asarray(wz["packed_params"])
pb_fid = jnp.asarray(wz["pb_fid"])
bin_cov_invs = [jnp.asarray(c) for c in wz["bin_cov_invs"]]
bao_fid = jnp.asarray(wz["bao_fid"])
bao_cov_inv = jnp.asarray(wz["bao_cov_inv"])
mu_p = jnp.asarray(wz["mu_p"])
sigma_p = jnp.asarray(wz["sigma_p"])
fid_nl = jnp.asarray(wz["fid_nl"])
cov_nl_prior = jnp.asarray(wz["cov_nl_prior"])
cosmo_nl_pos = [int(i) for i in wz["cosmo_nl_pos"]]
nl_prior_entries = [
    (int(p), float(m), float(s))
    for p, m, s in zip(wz["nl_prior_pos"], wz["nl_prior_mean"],
                       wz["nl_prior_sigma"])]

pklin_emulator = CosmoEmulator(probe='custom_log', emulator_path=PFS_EMULATOR)
ps1loop_model = PS1LoopModel(do_irres=True)
bispectrum_model = BispectrumTreeModel(do_AP=True, k_nl_rsd=K_NL_RSD)

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
n_cosmo_params = sum(cosmo.param_sizes)
Hz_fid, DAz_fid = compute_fiducial_distances(cosmo, z_bins)

# Config-drift tripwire.
packed_rebuilt = pack_joint_params(cosmo, surveys)
if not np.array_equal(np.asarray(packed_rebuilt), np.asarray(packed_params)):
    sys.exit("ABORT: rebuilt packed fiducial vector differs from the stored "
             "one -- config drift between build and this script.")

split = split_marginal_indices(
    n_cosmo_params=n_cosmo_params, survey_keys=joint_survey_keys,
    n_bins=n_zbins, fixed_cosmo=[5, 6, 7, 8],
    fixed_survey_keys={('shared', 'k_nl', None), ('shared', 'ndens', None)})
if (not np.array_equal(np.asarray(split.nl_idx), wz["nl_idx"])
        or not np.array_equal(np.asarray(split.lin_idx), wz["lin_idx"])):
    sys.exit("ABORT: rebuilt marginal split differs from the stored one.")
n_nl = split.n_nl
bin_lin_idx = [split.lin_idx[sl] for sl in bin_lin_slices(split, n_zbins)]

joint_theory_kwargs = dict(
    pklin_emulator=pklin_emulator, ps1loop_model=ps1loop_model,
    bispectrum_model=bispectrum_model,
    cosmo_keys=cosmo.param_keys, cosmo_sizes=cosmo.param_sizes,
    survey_keys=joint_survey_keys,
    ap=True, z_bins=z_bins, Hz_fid=Hz_fid, DAz_fid=DAz_fid,
    n_gl=N_GL, num_mu=NUM_MU, num_phi=NUM_PHI,
    background_mode=BACKGROUND_MODE)

k = jnp.linspace(K_PK_MIN, K_PK_MAX, N_K)
dk = float(k[1] - k[0])
triangles, _triangle_dk = build_bispectrum_triangles_from_k_grid(
    k, k_min=K_BK_MIN, k_max=K_BK_MAX, dk=dk)
block_len = 3 * int(k.shape[0]) + int(triangles.shape[0])
bin_blocks = [slice(b * block_len, (b + 1) * block_len)
              for b in range(n_zbins)]
bin_data = [pb_fid[sl] for sl in bin_blocks]

bao_dr2 = load_desi_dr2("all", data_dir=BAO_DATA_DIR)
bao_theory_fn = make_bao_theory_fn(
    bao_dr2, cosmo_keys=cosmo.param_keys, cosmo_sizes=cosmo.param_sizes,
    mnu_fixed=MNU_FIXED)
if not np.allclose(np.asarray(bao_theory_fn(packed_params[:n_cosmo_params])),
                   np.asarray(bao_fid), rtol=1e-10, atol=0.0):
    sys.exit("ABORT: recomputed BAO fiducial vector differs from the stored "
             "one beyond 1e-10.")

prior_mean_fn, prior_sigma_fn = make_constant_prior_fns(mu_p, sigma_p)
log_prior_nl = make_gaussian_log_prior(n_nl, nl_prior_entries)
to_whitened, to_physical = make_cholesky_transform(
    center=fid_nl, cov=cov_nl_prior)
full_params_fn = make_full_params_fn(packed_params, split.nl_idx)

shared_post_kwargs = dict(
    bin_data=bin_data, bin_cov_invs=bin_cov_invs,
    extra_theory_fn=lambda p: bao_theory_fn(p[:n_cosmo_params]),
    extra_data=bao_fid, extra_cov_inv=bao_cov_inv,
    prior_mean_fn=prior_mean_fn, prior_sigma_fn=prior_sigma_fn,
    log_prior_nl_fn=log_prior_nl, to_physical=to_physical,
    full_params_fn=full_params_fn, include_logdet=True)

# SURROGATE s(x) -- the cheap Taylor screen (jitted closure).
sur_log_post = make_marginal_log_posterior_taylor(tt, **shared_post_kwargs)

# EXACT p(x) -- the expensive per-bin marginal posterior (jitted closure).
bin_theory_fns = [
    partial(make_joint_pk_bk_bin_fn(bin_index=b, **joint_theory_kwargs),
            k=k, triangles=triangles)
    for b in range(n_zbins)]
exact_log_post = make_marginal_log_posterior_perbin(
    bin_theory_fns=bin_theory_fns, bin_lin_idx=bin_lin_idx,
    **shared_post_kwargs)

# ---------------------------------------------------------------------------
# Warm up both jitted posteriors at x = 0 (the exact one compiles here, ~1 min,
# ~30 GB). This is also the DA chain's cached initial p_cur / s_cur.
# ---------------------------------------------------------------------------

_watch["stage"] = "warmup surrogate"
x0 = jnp.zeros(n_nl)
t0 = time.perf_counter()
s0 = float(jax.block_until_ready(sur_log_post(x0)))
print(f"surrogate log_post(x0) = {s0:.6f} "
      f"(first call {time.perf_counter() - t0:.1f}s)", flush=True)

_watch["stage"] = "warmup exact (compile ~1 min, ~30 GB)"
t0 = time.perf_counter()
p0 = float(jax.block_until_ready(exact_log_post(x0)))
print(f"exact    log_post(x0) = {p0:.6f} (compile+eval "
      f"{time.perf_counter() - t0:.1f}s, rss {_rss_gb():.1f} GB); "
      f"exact - surrogate at center = {p0 - s0:.3e}", flush=True)

d = int(n_nl)
sigma_scalar = 2.38 / np.sqrt(d)              # optimal-scaling, whitened space
sigma_np = sigma_scalar * np.ones(d, dtype=np.float64)
print(f"d = {d}, proposal sigma = 2.38/sqrt({d}) = {sigma_scalar:.6f}",
      flush=True)

# ---------------------------------------------------------------------------
# Resume from checkpoint if present, else initialize the DA-MH state.
# ---------------------------------------------------------------------------

kept = np.empty((TOTAL, d), dtype=np.float64)

if PROGRESS_PATH.exists() and CHAIN_PATH.exists():
    st = json.loads(PROGRESS_PATH.read_text())
    n_done = int(st["n_done"])
    existing = np.load(CHAIN_PATH)
    if existing.shape != (n_done, d):
        sys.exit(f"ABORT: {CHAIN_PATH} shape {existing.shape} != "
                 f"({n_done}, {d}) from progress file.")
    kept[:n_done] = existing
    cur = np.asarray(st["cur"], dtype=np.float64)
    s_cur = float(st["s_cur"])
    p_cur = float(st["p_cur"])
    n_stage1 = int(st["n_stage1"])
    n_stage2 = int(st["n_stage2"])
    n_exact = int(st["n_exact"])
    elapsed_s = float(st.get("elapsed_s", 0.0))
    rng = np.random.default_rng()
    rng.bit_generator.state = st["rng_state"]
    print(f"RESUME: n_done={n_done}/{TOTAL}, n_exact={n_exact}, "
          f"stage1={n_stage1}, stage2={n_stage2}, elapsed={elapsed_s:.0f}s",
          flush=True)
else:
    n_done = 0
    cur = np.asarray(x0, dtype=np.float64)        # start x = 0 (no jitter)
    s_cur = s0
    p_cur = p0
    n_stage1 = 0
    n_stage2 = 0
    n_exact = 1                                    # the initial exact eval
    elapsed_s = 0.0
    rng = np.random.default_rng(RNG_SEED)
    print(f"FRESH START: seed={RNG_SEED}, x=0, s_cur={s_cur:.6f}, "
          f"p_cur={p_cur:.6f}", flush=True)


def save_checkpoint():
    np.save(CHAIN_PATH, kept[:n_done])
    sec_per_step = elapsed_s / n_done if n_done > 0 else 0.0
    est_remaining_h = (TOTAL - n_done) * sec_per_step / 3600.0
    state = {
        "n_done": n_done, "total": TOTAL,
        "n_stage1": n_stage1, "n_stage2": n_stage2, "n_exact": n_exact,
        "stage1_rate": n_stage1 / n_done if n_done > 0 else 0.0,
        "stage2_rate": n_stage2 / n_stage1 if n_stage1 > 0 else 0.0,
        "cur": cur.tolist(), "s_cur": s_cur, "p_cur": p_cur,
        "rng_state": rng.bit_generator.state,
        "elapsed_s": elapsed_s,
        "sec_per_step": sec_per_step,
        "est_remaining_h": est_remaining_h,
        "peak_gb": round(_watch["peak_gb"], 2),
    }
    PROGRESS_PATH.write_text(json.dumps(state))
    return sec_per_step, est_remaining_h


# ---------------------------------------------------------------------------
# Segmented DA-MH loop. The per-step body is inlined EXACTLY as
# run_damh_python (sampler.py:1240-1261).
# ---------------------------------------------------------------------------

_watch["stage"] = "DA-MH loop"
print(f"===== DA-MH loop: {n_done} -> {TOTAL} (seg {SEG}) =====", flush=True)

while n_done < TOTAL:
    seg_t0 = time.perf_counter()
    seg_end = min(n_done + SEG, TOTAL)

    for step in range(n_done, seg_end):
        prop = cur + sigma_np * rng.normal(size=d)
        s_prop = float(sur_log_post(jnp.asarray(prop)))

        # STAGE 1: surrogate screen, min(1, s(y)/s(x)).
        if np.log(rng.random()) < (s_prop - s_cur):
            n_stage1 += 1
            # STAGE 2: single exact eval, min(1, [p(y)s(x)]/[p(x)s(y)]).
            p_prop = float(exact_log_post(jnp.asarray(prop)))
            n_exact += 1
            stage2_log_ratio = (p_prop + s_cur) - (p_cur + s_prop)
            if np.log(rng.random()) < stage2_log_ratio:
                cur, s_cur, p_cur = prop, s_prop, p_prop
                n_stage2 += 1

        kept[step] = cur

    n_done = seg_end
    elapsed_s += time.perf_counter() - seg_t0
    sec_per_step, est_remaining_h = save_checkpoint()
    print(f"  [seg] {n_done}/{TOTAL}  stage1={n_stage1} "
          f"({100 * n_stage1 / n_done:.1f}%)  stage2={n_stage2}  "
          f"exact={n_exact}  {sec_per_step:.2f} s/step  "
          f"eta {est_remaining_h:.1f} h  rss {_rss_gb():.1f} GB", flush=True)

# ---------------------------------------------------------------------------
# Done.
# ---------------------------------------------------------------------------

stage1_rate = n_stage1 / TOTAL
stage2_rate = n_stage2 / n_stage1 if n_stage1 > 0 else 0.0
move_rate = n_stage2 / TOTAL
print(f"\n===== DA-MH COMPLETE =====", flush=True)
print(f"kept {TOTAL} draws -> {CHAIN_PATH}", flush=True)
print(f"stage-1 rate {stage1_rate:.4f}  stage-2 rate {stage2_rate:.4f}  "
      f"move rate {move_rate:.4f}  exact evals {n_exact}", flush=True)
print(f"total wall {elapsed_s:.0f}s ({elapsed_s / 3600:.2f} h), "
      f"{elapsed_s / TOTAL:.2f} s/step, peak RSS {_watch['peak_gb']:.1f} GB",
      flush=True)
