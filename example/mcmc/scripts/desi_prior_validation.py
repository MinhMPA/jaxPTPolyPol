"""Stream-B acceptance gate (branch stream-b-sigmap): surrogate chain vs
Hessian-Fisher under the DESI DR1-reanalysis (2511.20757) priors.

Gates (cosmology block, 5 params ombh2/omch2/logA/ns/h)
------------------------------------------------------
  G1  widths: chain posterior widths / Fisher widths in [0.9, 1.1]
  G2  corrs : max |corr_chain - corr_Fisher| < 0.1
  G3  means : |chain mean - mu_tilt| < 2.5 * MC-SE + 0.05 sigma_F per param,
              mu_tilt = fid + F^{-1} grad logpost(fid)  (AD-tilted center)

The surrogate marginal log-posterior embeds the spec priors, so "Fisher with
the same spec" is F = -hess logpost(fid) (jax.hessian on the whitened
surrogate; Tier-1 established curvature == Fisher-Schur at fiducial). The tilt
center now carries BOTH the logdet tilt AND the correlated non-fiducial
prior-mean pulls (c2->30 etc.) because the Hessian/gradient see the full
cov-mode ctr prior.

Amendment 1, sigmap branch
--------------------------
Templates stay UNROTATED (as cached by build_taylor_templates_lcdm.py): the
surrogate speaks OUR mu-space tilde ctr basis; the exact paper prior arrives
through the cov-mode per-bin prior blocks. ``make_desi_prior_fns`` auto-selects
cov-mode from the spec's ctr trio (the ``ctr_rotation`` token) and returns a
stacked ``(n_bins, 11, 11)`` ``prior_sigma_fn`` that
``make_marginal_log_posterior_taylor`` consumes natively (Task 5sigma). We do
NOT rotate the templates here.

Outputs (written into master's cache via the worktree symlink)
--------------------------------------------------------------
  cache/desi_prior_validation_sigmap.json  -- gate verdict + all numbers
  cache/branch_equiv_sigmap.json           -- 64 whitened-point log-posterior
      values for the machine-precision cross-branch equivalence test:
      {"points_seed": 20260731, "n": 64, "scale": 0.5, "log_post": [...]}
      points = 0.5 * jax.random.normal(PRNGKey(20260731), (64, n_nl)); each
      evaluated through the SAME jitted log-posterior the chain uses
      (include_logdet=True), float64. One PRNGKey call, no splitting -- the
      sister rotation branch generates the identical points, and
      max|Delta log_post| across branches is the primary cross-validation.

Reuses the template/whitening loading, theory-statics rebuild, config-drift
tripwires, BAO extra-term assembly and RWMH chain drive of
``scripts/taylor_surrogate_validation.py`` verbatim; only the prior
construction (DESI cov-mode + BBN/ns) and the gate logic are new.

Run from example/mcmc after build_taylor_templates_lcdm.py::

    cd example/mcmc
    PYTHONPATH=<worktree>/src python scripts/desi_prior_validation.py
    # smoke run (2000/200): prepend DESI_GATE_SMOKE=1
"""

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
    load_taylor_templates,
    make_marginal_log_posterior_taylor,
    split_marginal_indices,
)
from jaxptpolypol.bao import load_desi_dr2, make_bao_theory_fn
from jaxptpolypol.desi_priors import (
    load_desi_prior_spec,
    make_desi_prior_fns,
    make_lcdm_rescaling_fns,
)
from jaxptpolypol.model import CosmoEmulator
from jaxptpolypol.params import (
    CosmoParams,
    FullShapeSurveyParams,
    pack_joint_params,
)
from jaxptpolypol.sampler import (
    make_cholesky_transform,
    make_full_params_fn,
    make_gaussian_log_prior,
    run_rwmh_python,
)
from jaxptpolypol.theory import build_bispectrum_triangles_from_k_grid

# ---------------------------------------------------------------------------
# RSS watchdog (same protocol as taylor_surrogate_validation.py; surrogate-only
# here so the peak is far below the 70 GB abort limit -- kept as machine
# protection during the multi-minute full chain).
# ---------------------------------------------------------------------------

_T0 = time.perf_counter()
_watch = {"stage": "init", "limit_gb": 70.0, "peak_gb": 0.0}


def _rss_gb():
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                         capture_output=True, text=True).stdout.strip()
    return int(out) / 1048576 if out else 0.0


def _watchdog():
    n = 0
    while True:
        gb = _rss_gb()
        _watch["peak_gb"] = max(_watch["peak_gb"], gb)
        if gb > _watch["limit_gb"]:
            print(f"!!! RSS WATCHDOG ABORT: {gb:.1f} GB > "
                  f"{_watch['limit_gb']:.0f} GB during '{_watch['stage']}'. "
                  "Aborting to protect the machine.", flush=True)
            os._exit(17)
        n += 1
        if n % 30 == 0:
            print(f"  [watchdog] stage='{_watch['stage']}' rss={gb:.1f} GB "
                  f"t={time.perf_counter() - _T0:.0f}s", flush=True)
        time.sleep(2.0)


threading.Thread(target=_watchdog, daemon=True).start()

# ---------------------------------------------------------------------------
# Configuration -- copied VERBATIM from taylor_surrogate_validation.py (which in
# turn mirrors mcmc_joint_PFS_BAO_BBN_ns_LCDM.ipynb). Must stay in lockstep with
# build_taylor_templates_lcdm.py; the META guard + packed/split/bao tripwires
# enforce it.
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
BAO_DATA_DIR = "../../ext_data/bao_data/desi_bao_dr2"   # chdir-sensitive
# In a worktree ext_data is only partially checked out (like cache); the DR2
# BAO data lives in master's ext_data. Fall back to it if absent locally.
if not pathlib.Path(BAO_DATA_DIR).is_dir():
    _master_bao = pathlib.Path(
        "/Users/nguyenmn/jaxPTPolyPol/ext_data/bao_data/desi_bao_dr2")
    if _master_bao.is_dir():
        BAO_DATA_DIR = str(_master_bao)

SHARED_KEYS = ('ombh2', 'omch2', 'logA', 'ns', 'h')   # sampled cosmo (nl block)
N_COSMO_NL = len(SHARED_KEYS)                          # == 5, gate cosmo block

META = {
    "n_bins": 7, "n_k": 37, "n_tri": 264, "n_gl": 16,
    "num_mu": 65, "num_phi": 65,
    "k_min": 0.02, "k_max": 0.20, "k_bk_max": 0.08, "k_nl_rsd": 0.45,
    "order2_m0": True,
}

# Gate parameters (task 7sigma dispatch).
RNG_SEED_CHAIN = 20260731
EQUIV_SEED = 20260731
EQUIV_N = 64
EQUIV_SCALE = 0.5

SMOKE = os.environ.get("DESI_GATE_SMOKE") == "1"
NUM_SAMPLES = 2_000 if SMOKE else 200_000
BURN = 200 if SMOKE else 20_000

# ---------------------------------------------------------------------------
# Cache path. In a worktree, example/mcmc/cache is a symlink (or, as set up
# here, a directory carrying a nested ``cache`` symlink) into master's real
# cache; the required templates/whitening artifacts live there and outputs must
# land there too (Task E compares the two branches' branch_equiv_*.json in
# master's cache).
# ---------------------------------------------------------------------------

HERE = pathlib.Path(__file__).resolve().parents[1]      # example/mcmc
CACHE = HERE / "cache"
if (not (CACHE / "taylor_whitening_lcdm.npz").exists()
        and (CACHE / "cache" / "taylor_whitening_lcdm.npz").exists()):
    CACHE = CACHE / "cache"                              # nested symlink -> master

WHITENING_PATH = CACHE / "taylor_whitening_lcdm.npz"
TEMPLATES_PATH = CACHE / "taylor_templates_lcdm.npz"
RESULT_PATH = CACHE / "desi_prior_validation_sigmap.json"
EQUIV_PATH = CACHE / "branch_equiv_sigmap.json"

for p in (WHITENING_PATH, TEMPLATES_PATH):
    if not p.exists():
        sys.exit(f"Required artifact missing: {p} -- run from example/mcmc "
                 "after scripts/build_taylor_templates_lcdm.py.")
if not pathlib.Path(BAO_DATA_DIR).is_dir():
    sys.exit(f"BAO data dir not found at {BAO_DATA_DIR!r} -- run from "
             "example/mcmc.")

results = {"config": {"branch": "stream-b-sigmap", "meta": META,
                      "rng_seed_chain": RNG_SEED_CHAIN,
                      "num_samples": NUM_SAMPLES, "burn": BURN,
                      "smoke": SMOKE,
                      "equiv": {"points_seed": EQUIV_SEED, "n": EQUIV_N,
                                "scale": EQUIV_SCALE}}}


def save_results():
    RESULT_PATH.write_text(json.dumps(results, indent=1))


save_results()

# ---------------------------------------------------------------------------
# Load artifacts (strict meta guards) -- VERBATIM from
# taylor_surrogate_validation.py, minus the tilt/exact-chain pieces this gate
# does not use.
# ---------------------------------------------------------------------------

_watch["stage"] = "load + assemble surrogate"
print(f"===== {_watch['stage']} (smoke={SMOKE}) =====", flush=True)

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
fid_nl = jnp.asarray(wz["fid_nl"])
cov_nl_prior = jnp.asarray(wz["cov_nl_prior"])
sig_fisher = np.asarray(wz["sig_fisher"])
cosmo_nl_pos = [int(i) for i in wz["cosmo_nl_pos"]]
# BBN (ombh2) + ns cosmology priors on the SAMPLED (nl) parameters. These are
# the production priors that the DESI spec (b2/bG2 only) does NOT carry, so they
# are added to the DESI log-prior below on disjoint nl positions.
nl_prior_entries = [
    (int(p), float(m), float(s))
    for p, m, s in zip(wz["nl_prior_pos"], wz["nl_prior_mean"],
                       wz["nl_prior_sigma"])]

# Rebuild the theory statics (cannot be serialized). The surrogate needs only
# the emulator (for the DESI sigma8/A_AP rescaling), the cosmo/survey pytrees
# (for the split + config-drift tripwires), the BAO theory fn, and the k-grid
# triangles (for slicing pb_fid into per-bin data). ps1loop/bispectrum models
# and the exact per-bin posterior are NOT needed here.
pklin_emulator = CosmoEmulator(probe='custom_log', emulator_path=PFS_EMULATOR)

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

# Config-drift tripwire: the rebuilt fiducial packed vector must equal the one
# the templates/whitening were built with, bit for bit.
packed_rebuilt = pack_joint_params(cosmo, surveys)
if not np.array_equal(np.asarray(packed_rebuilt), np.asarray(packed_params)):
    sys.exit("ABORT: rebuilt packed fiducial vector differs from the stored "
             "one -- config drift between build and validation scripts.")

split = split_marginal_indices(
    n_cosmo_params=n_cosmo_params, survey_keys=joint_survey_keys,
    n_bins=n_zbins, fixed_cosmo=[5, 6, 7, 8],
    fixed_survey_keys={('shared', 'k_nl', None), ('shared', 'ndens', None)})
if (not np.array_equal(np.asarray(split.nl_idx), wz["nl_idx"])
        or not np.array_equal(np.asarray(split.lin_idx), wz["lin_idx"])):
    sys.exit("ABORT: rebuilt marginal split differs from the stored one.")
n_nl = split.n_nl

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

# ---------------------------------------------------------------------------
# DESI priors (NEW): cov-mode survey/EFT prior + BBN/ns cosmology prior.
# ---------------------------------------------------------------------------

spec = load_desi_prior_spec()
# The linear-Pk emulator's inputs include the baryon-feedback nuisances
# A_b/eta_b/logT_AGN, which are FIXED in the production LCDM layout (not in the
# sampled nl cosmo block). Inject them at their fiducial values as constants so
# sigma8 is evaluated at the right point with zero spurious theta-derivative.
FIXED_BARYON = {'A_b': cosmo_dict['A_b'], 'eta_b': cosmo_dict['eta_b'],
                'logT_AGN': cosmo_dict['logT_AGN']}
sigma8_bins_fn, a_ap_bins_fn, sigma8_ref_bins = make_lcdm_rescaling_fns(
    pklin_emulator=pklin_emulator, cosmo_keys=SHARED_KEYS,
    cosmo_sizes=(1,) * N_COSMO_NL, z_bins=z_bins,
    fid_cosmo_native=fid_nl[:N_COSMO_NL], mnu_fixed=MNU_FIXED,
    fixed_cosmo_extras=FIXED_BARYON)
desi_prior_mean_fn, desi_prior_sigma_fn, desi_log_prior_nl = make_desi_prior_fns(
    spec, split=split, knl_bins=knl_bins,
    sigma8_bins_fn=sigma8_bins_fn, a_ap_bins_fn=a_ap_bins_fn,
    sigma8_ref_bins=sigma8_ref_bins)

# The spec carries the ctr_rotation trio -> cov-mode: prior_sigma_fn returns
# stacked (n_bins, 11, 11) prior covariance blocks (paper c0/c2/c4 rotated into
# our tilde basis by L(f)).  Sanity-check the mode + shape once at fiducial.
_sig0 = desi_prior_sigma_fn(fid_nl)
if getattr(_sig0, "ndim", None) != 3 or _sig0.shape[0] != n_zbins:
    sys.exit(f"ABORT: expected cov-mode prior_sigma_fn (n_bins,11,11); got "
             f"shape {getattr(_sig0, 'shape', None)} -- spec ctr_rotation token "
             "missing?")
results["config"]["prior_sigma_mode"] = "cov"
results["config"]["prior_sigma_block_shape"] = list(_sig0.shape)
results["config"]["nl_prior_entries"] = nl_prior_entries
results["config"]["sigma8_ref_bins"] = np.asarray(sigma8_ref_bins).tolist()

bbn_ns_log_prior = make_gaussian_log_prior(n_nl, nl_prior_entries)


def log_prior_nl(theta_nl):
    """DESI (b2/bG2) + production BBN(ombh2)/ns Gaussian priors on nl params.

    Disjoint positions: DESI touches b2/bG2 (split.nl_b1_pos + offset per bin);
    BBN/ns touch nl positions 0 (ombh2) and 3 (ns).
    """
    return desi_log_prior_nl(theta_nl) + bbn_ns_log_prior(theta_nl)


to_whitened, to_physical = make_cholesky_transform(
    center=fid_nl, cov=cov_nl_prior)
full_params_fn = make_full_params_fn(packed_params, split.nl_idx)

log_post = make_marginal_log_posterior_taylor(
    tt, bin_data=bin_data, bin_cov_invs=bin_cov_invs,
    prior_mean_fn=desi_prior_mean_fn, prior_sigma_fn=desi_prior_sigma_fn,
    log_prior_nl_fn=log_prior_nl, to_physical=to_physical,
    full_params_fn=full_params_fn,
    extra_theory_fn=lambda p: bao_theory_fn(p[:n_cosmo_params]),
    extra_data=bao_fid, extra_cov_inv=bao_cov_inv, include_logdet=True)
log_post_j = jax.jit(log_post)

w0 = jnp.zeros(n_nl)
t0 = time.perf_counter()
lp0 = float(jax.block_until_ready(log_post_j(w0)))
print(f"surrogate log_post(x0) = {lp0:.6f} "
      f"(first call {time.perf_counter() - t0:.1f}s, rss {_rss_gb():.1f} GB)",
      flush=True)
results["surrogate"] = {"lp0": lp0}
save_results()

# ---------------------------------------------------------------------------
# Hessian-Fisher + AD tilt from the surrogate itself (whitened coords).
# ---------------------------------------------------------------------------

_watch["stage"] = "hessian + tilt"
print("===== Hessian-Fisher + AD tilt (whitened) =====", flush=True)
t0 = time.perf_counter()
H = jax.hessian(log_post)(w0)
g = jax.grad(log_post)(w0)
jax.block_until_ready((H, g))
F_w = -np.asarray(H)
cov_F_w = np.linalg.inv(F_w)
mu_tilt_w = np.linalg.solve(F_w, np.asarray(g))          # fid=0 in whitened
print(f"hessian+grad {time.perf_counter() - t0:.1f}s; "
      f"|mu_tilt_w| = {np.linalg.norm(mu_tilt_w):.4f}, "
      f"|g| = {np.linalg.norm(np.asarray(g)):.4f}", flush=True)
if not np.all(np.isfinite(mu_tilt_w)):
    sys.exit("ABORT: mu_tilt_w is not finite -- Hessian/gradient degenerate.")

# ---------------------------------------------------------------------------
# Equivalence dump (cross-branch machine-precision test). Deterministic in the
# chain length, so computed here regardless of smoke/full.
# ---------------------------------------------------------------------------

_watch["stage"] = "equivalence dump"
equiv_pts = EQUIV_SCALE * jax.random.normal(
    jax.random.PRNGKey(EQUIV_SEED), (EQUIV_N, n_nl))
equiv_lp = [float(log_post_j(equiv_pts[i])) for i in range(EQUIV_N)]
EQUIV_PATH.write_text(json.dumps(
    {"points_seed": EQUIV_SEED, "n": EQUIV_N, "scale": EQUIV_SCALE,
     "log_post": equiv_lp}, indent=1))
print(f"equivalence dump -> {EQUIV_PATH} "
      f"(lp[0]={equiv_lp[0]:.9f}, lp[-1]={equiv_lp[-1]:.9f})", flush=True)

# ---------------------------------------------------------------------------
# RWMH chain -- signature/step-scale transplanted from
# taylor_surrogate_validation.py (seed -> 20260731, log-post -> log_post_j).
# ---------------------------------------------------------------------------

_watch["stage"] = "chain"
print(f"===== CHAIN ({NUM_SAMPLES} RWMH steps, burn {BURN}) =====", flush=True)
_chain_t0 = time.perf_counter()


def _on_draw(_chain, _nchains, done, total):
    if done % (10_000 if not SMOKE else 500) == 0 or done == total:
        el = time.perf_counter() - _chain_t0
        print(f"  [chain] {done}/{total} draws, {el:7.0f}s, "
              f"{el / max(done, 1) * 1000:.2f} ms/step", flush=True)


samples_w, diagnostics = run_rwmh_python(
    jax.random.key(RNG_SEED_CHAIN), log_post_j, initial_position=w0,
    num_samples=NUM_SAMPLES, num_chains=1, thin=1,
    sample_progress_fn=_on_draw)
wall = time.perf_counter() - _chain_t0
s_per_step = wall / NUM_SAMPLES
acc = float(np.asarray(diagnostics["acceptance_rate"])[0])
print(f"chain wall {wall:.0f}s -> {s_per_step * 1000:.2f} ms/step; "
      f"acceptance {acc:.3f}", flush=True)

chain_w = np.asarray(samples_w[0])                        # (NUM_SAMPLES, n_nl)
draws = chain_w[BURN:]

# ---------------------------------------------------------------------------
# Gates on the cosmology block (whitened positions 0..4 == cosmo_nl_pos;
# physical projection via the loaded Cholesky whitening). Exactly the brief.
# ---------------------------------------------------------------------------

_watch["stage"] = "gates"
n_cosmo = N_COSMO_NL
L = np.linalg.cholesky(np.asarray(cov_nl_prior))
phys = np.asarray(fid_nl) + draws @ L.T
cov_F_phys = L @ cov_F_w @ L.T
mu_tilt_phys = np.asarray(fid_nl) + L @ mu_tilt_w

sig_chain = phys[:, :n_cosmo].std(0, ddof=1)
sig_F = np.sqrt(np.diag(cov_F_phys))[:n_cosmo]
width_ratio = sig_chain / sig_F
corr_chain = np.corrcoef(phys[:, :n_cosmo], rowvar=False)
corr_F = cov_F_phys[:n_cosmo, :n_cosmo] / np.outer(sig_F, sig_F)
iu = np.triu_indices(n_cosmo, 1)
corr_diff_max = float(np.abs(corr_chain - corr_F)[iu].max())


def ess_ips(x):
    x = x - x.mean(); n = x.size
    ac = np.correlate(x, x, "full")[n - 1:] / (np.arange(n, 0, -1) * x.var())
    tau = 1.0
    for lag in range(1, min(n // 2, 5000)):
        if ac[lag] <= 0:
            break
        tau += 2 * ac[lag]
    return n / tau


ess = np.array([ess_ips(draws[:, j]) for j in range(n_cosmo)])
mean_pull = (phys[:, :n_cosmo].mean(0) - mu_tilt_phys[:n_cosmo]) / sig_F
mc_se = sig_chain / np.sqrt(ess) / sig_F

g1 = bool(np.all((width_ratio > 0.9) & (width_ratio < 1.1)))
g2 = bool(corr_diff_max < 0.1)
g3 = bool(np.all(np.abs(mean_pull) < 2.5 * mc_se + 0.05))
verdict = "PASS" if (g1 and g2 and g3) else "REVIEW"

print(f"\n{'param':>7s} {'width s/F':>10s} {'mean_pull':>10s} "
      f"{'mc_se':>8s} {'ess':>9s}")
for i, key in enumerate(SHARED_KEYS):
    print(f"{key:>7s} {width_ratio[i]:10.4f} {mean_pull[i]:10.4f} "
          f"{mc_se[i]:8.4f} {ess[i]:9.1f}")
print(f"max corr diff = {corr_diff_max:.4f} (require < 0.1)")
print(f"G1 widths {g1}  G2 corrs {g2}  G3 means {g3}  ->  {verdict}", flush=True)

results["gates"] = {"G1_widths": g1, "G2_corrs": g2, "G3_means": g3}
results["verdict"] = verdict
results["numbers"] = {
    "names": list(SHARED_KEYS),
    "width_ratio": width_ratio.tolist(),
    "corr_diff_max": corr_diff_max,
    "mean_pull_vs_tilted": mean_pull.tolist(),
    "mc_se": mc_se.tolist(),
    "ess": ess.tolist(),
    "sig_chain_phys": sig_chain.tolist(),
    "sig_F_phys": sig_F.tolist(),
    "sig_fisher_stored": sig_fisher.tolist(),
    "acceptance": acc,
    "s_per_step": s_per_step,
    "wall_s": wall,
    "n_kept": int(draws.shape[0]),
    "corr_chain": corr_chain.tolist(),
    "corr_F": corr_F.tolist(),
    "mu_tilt_w": np.asarray(mu_tilt_w).tolist(),
    "peak_rss_gb": round(_watch["peak_gb"], 2),
    "total_wall_s": round(time.perf_counter() - _T0, 1),
}
save_results()

print(f"\n===== VALIDATION {verdict} (smoke={SMOKE}) =====")
print(f"-> {RESULT_PATH}")
print(f"-> {EQUIV_PATH}")
print(f"peak RSS {_watch['peak_gb']:.1f} GB, total "
      f"{time.perf_counter() - _T0:.0f}s", flush=True)
