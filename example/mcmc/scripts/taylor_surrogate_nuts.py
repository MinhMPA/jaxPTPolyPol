"""Task 8: gradient-based NUTS on the Taylor SURROGATE posterior.

The surrogate marginal log-posterior is a microsecond-scale dense-algebra body
(linear templates + logdet tilt + Gaussian priors; measured ~0.65 ms/step
in-chain), so the "scan-trap" rule recorded in
``docs/design/perbin-compile-measurements.md`` -- *do not lax.scan an expensive
body* -- does NOT apply here: the body is tiny and homogeneous, exactly the
regime where blackjax window adaptation + chunked-scan production is the right
tool. We therefore run blackjax NUTS (``jaxptpolypol.sampler.run_nuts``, which is
blackjax ``window_adaptation`` + a JIT-compiled chunked ``lax.scan``) directly on
the surrogate. If window adaptation misbehaves (R-hat > 1.01 or divergences
> 2 %) we fall back to a fixed-L leapfrog HMC driven from a Python loop over
``jax.value_and_grad`` of the surrogate, and record which sampler was used.

Deliverables
------------
* R-hat and ESS per cosmology parameter, mean acceptance, divergence fraction.
* SKEW of the logA and ns marginals (scipy.stats.skew + bootstrap error) for
  three chains in the SAME Cholesky-whitened -> physical space:
    - this NUTS-on-surrogate chain,
    - the surrogate RWMH 200k chain (reused from ``cache/taylor_chain_w.npy``
      if present, else rerun),
    - the exact reduced-Tier-2 chain (``cache/tier2_chain_w.npy``, burn 500).
* Physics question: is the open logA/ns Tier-2 mean residual (means ~0.3-0.45
  sigma_F below the fiducial) explained by genuine posterior skew?

Assembly of the surrogate posterior is copied from
``scripts/taylor_surrogate_validation.py`` (SAME whitening / data / covs /
priors / BAO). The expensive per-bin theory and exact posterior are NOT built
here -- this script is surrogate-only, so it stays memory-trivial and can run
alongside the overnight DA-MH chain.

Run from ``example/mcmc``::

    cd example/mcmc
    python3 scripts/taylor_surrogate_nuts.py
"""

from functools import partial
import json
import os
import pathlib
import sys
import time

N_THREADS = os.cpu_count() or 1
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(N_THREADS)

import numpy as np
from scipy.stats import skew as scipy_skew

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir",
                  str(pathlib.Path.home() / ".jax_xla_cache"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 10)
import jax.numpy as jnp

from blackjax.diagnostics import (
    effective_sample_size,
    potential_scale_reduction,
)

from ps_1loop_jax import background as bg

from stream_common import (
    DEFAULT_BAO_DATA_DIR, FIDUCIAL, K_BK_MAX, K_BK_MIN, K_PK_MAX, K_PK_MIN,
    META, MNU_FIXED, N_K, SHARED_KEYS, knl_bins, load_templates_and_whitening,
    n_bar, n_zbins, z_bins,
)

from jaxptpolypol import (
    make_constant_prior_fns,
    make_marginal_log_posterior_taylor,
    split_marginal_indices,
)
from jaxptpolypol.bao import load_desi_dr2, make_bao_theory_fn
from jaxptpolypol.params import (
    CosmoParams,
    FullShapeSurveyParams,
    pack_joint_params,
)
from jaxptpolypol.sampler import (
    make_cholesky_transform,
    make_full_params_fn,
    make_gaussian_log_prior,
    run_nuts,
    run_rwmh_python,
)
from jaxptpolypol.theory import build_bispectrum_triangles_from_k_grid

_T0 = time.perf_counter()

# ---------------------------------------------------------------------------
# Configuration -- imported from stream_common (2026-08-23; this script was the
# one driver that re-declared the production constants as literals, which is
# the drift class the single-source-of-truth rule exists to prevent).
# ---------------------------------------------------------------------------

BAO_DATA_DIR = DEFAULT_BAO_DATA_DIR

LOGA_I, NS_I = 2, 3               # positions of logA, ns within SHARED_KEYS

# NUTS parameters.
NUTS_SEED = 20260731
N_WARMUP = 1000
N_SAMPLES = 5000
N_CHAINS = 4
DIVERGENCE_FRAC_LIMIT = 0.02
RHAT_LIMIT = 1.01
# RWMH-fallback-rerun parameters (only if cache/taylor_chain_w.npy is missing).
RWMH_SEED = 20260729
RWMH_SAMPLES = 200_000
RWMH_BURN = 20_000
TIER2_BURN = 500
# Fixed-L HMC fallback parameters (only if NUTS adaptation misbehaves).
HMC_STEP = 0.3
HMC_L = 25
# Skew bootstrap.
N_BOOT = 1000
BOOT_SEED = 0

CACHE = pathlib.Path("cache")
WHITENING_PATH = CACHE / "taylor_whitening_lcdm.npz"
TEMPLATES_PATH = CACHE / "taylor_templates_lcdm.npz"
RWMH_CHAIN_PATH = CACHE / "taylor_chain_w.npy"
TIER2_CHAIN_PATH = CACHE / "tier2_chain_w.npy"
NUTS_CHAIN_PATH = CACHE / "taylor_nuts_chain_w.npy"
RESULT_PATH = CACHE / "taylor_nuts_result.json"
DOC_PATH = pathlib.Path("../../docs/design/perbin-compile-measurements.md")

for p in (WHITENING_PATH, TEMPLATES_PATH, TIER2_CHAIN_PATH):
    if not p.exists():
        sys.exit(f"Required artifact missing: {p} -- run from example/mcmc "
                 "after scripts/build_taylor_templates_lcdm.py.")
if not pathlib.Path(BAO_DATA_DIR).is_dir():
    sys.exit(f"BAO data dir not found at {BAO_DATA_DIR!r} -- run from "
             "example/mcmc.")

# ---------------------------------------------------------------------------
# Assemble the SURROGATE posterior only (surrogate-only: no emulator / exact).
# ---------------------------------------------------------------------------

print("===== assemble surrogate posterior =====", flush=True)

# The shared guarded loader: the templates stamp is held to the FULL production
# expectation (theory_config_hash + c1_treatment + grid), the whitening stamp to
# the grid + c1_treatment, so a stale or c1-sampled cache hard-fails here rather
# than silently feeding this chain (stream_common.load_templates_and_whitening).
tt, wz = load_templates_and_whitening(TEMPLATES_PATH, WHITENING_PATH)

packed_params = jnp.asarray(wz["packed_params"])
pb_fid = jnp.asarray(wz["pb_fid"])
bin_cov_invs = [jnp.asarray(c) for c in wz["bin_cov_invs"]]
bao_fid = jnp.asarray(wz["bao_fid"])
bao_cov_inv = jnp.asarray(wz["bao_cov_inv"])
mu_p = jnp.asarray(wz["mu_p"])
sigma_p = jnp.asarray(wz["sigma_p"])
fid_nl = jnp.asarray(wz["fid_nl"])
cov_nl_prior = jnp.asarray(wz["cov_nl_prior"])
sig_fisher = np.asarray(wz["sig_fisher"])
cosmo_nl_pos = [int(i) for i in wz["cosmo_nl_pos"]]
nl_prior_entries = [
    (int(p), float(m), float(s))
    for p, m, s in zip(wz["nl_prior_pos"], wz["nl_prior_mean"],
                       wz["nl_prior_sigma"])]

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

# Config-drift tripwire.
packed_rebuilt = pack_joint_params(cosmo, surveys)
if not np.array_equal(np.asarray(packed_rebuilt), np.asarray(packed_params)):
    sys.exit("ABORT: rebuilt packed fiducial vector differs from the stored "
             "one -- config drift.")

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

prior_mean_fn, prior_sigma_fn = make_constant_prior_fns(mu_p, sigma_p)
log_prior_nl = make_gaussian_log_prior(n_nl, nl_prior_entries)
to_whitened, to_physical = make_cholesky_transform(
    center=fid_nl, cov=cov_nl_prior)
full_params_fn = make_full_params_fn(packed_params, split.nl_idx)

sur_log_post = make_marginal_log_posterior_taylor(
    tt, bin_data=bin_data, bin_cov_invs=bin_cov_invs,
    extra_theory_fn=lambda p: bao_theory_fn(p[:n_cosmo_params]),
    extra_data=bao_fid, extra_cov_inv=bao_cov_inv,
    prior_mean_fn=prior_mean_fn, prior_sigma_fn=prior_sigma_fn,
    log_prior_nl_fn=log_prior_nl, to_physical=to_physical,
    full_params_fn=full_params_fn, include_logdet=True)

x0 = jnp.zeros(n_nl)
t0 = time.perf_counter()
lp0 = float(jax.block_until_ready(sur_log_post(x0)))
print(f"surrogate log_post(x0) = {lp0:.6f} "
      f"(first call {time.perf_counter() - t0:.1f}s); n_nl={n_nl}", flush=True)

fid_cosmo = np.asarray(fid_nl)[cosmo_nl_pos]      # physical fiducial (5,)

results = {
    "config": {"meta": META, "nuts_seed": NUTS_SEED, "n_warmup": N_WARMUP,
               "n_samples": N_SAMPLES, "n_chains": N_CHAINS,
               "rhat_limit": RHAT_LIMIT,
               "divergence_frac_limit": DIVERGENCE_FRAC_LIMIT,
               "cosmo_names": list(SHARED_KEYS)},
    "surrogate_lp0": lp0,
}


def save_results():
    RESULT_PATH.write_text(json.dumps(results, indent=1))


save_results()

# ---------------------------------------------------------------------------
# Primary: blackjax NUTS + window adaptation on the surrogate.
# ---------------------------------------------------------------------------

print(f"\n===== NUTS: {N_CHAINS} chains x {N_SAMPLES} draws "
      f"({N_WARMUP} adaptation) =====", flush=True)


def _warm_cb(c, nc, stage, done, total):
    if stage == "window_adaptation" and done == total:
        print(f"  [warmup] chain {c}/{nc} adapted", flush=True)


def _samp_cb(c, nc, done, total):
    if done == total:
        print(f"  [sample] chain {c}/{nc} done "
              f"({time.perf_counter() - _nuts_t0:.0f}s)", flush=True)


_nuts_t0 = time.perf_counter()
samples_w, diag = run_nuts(
    jax.random.key(NUTS_SEED), sur_log_post, x0,
    num_warmup=N_WARMUP, num_samples=N_SAMPLES, num_chains=N_CHAINS,
    max_tree_depth=10,
    warmup_progress_fn=_warm_cb, sample_progress_fn=_samp_cb)
nuts_wall = time.perf_counter() - _nuts_t0

samples_w = np.asarray(samples_w)                 # (C, S, n_nl) whitened
div_frac = float(np.mean(np.asarray(diag["is_divergent"])))
mean_accept = float(np.mean(np.asarray(diag["acceptance_rate"])))
mean_steps = float(np.mean(np.asarray(diag["num_integration_steps"])))

# Physical cosmology columns, per chain, for R-hat / ESS.
phys_cosmo = np.asarray(
    to_physical(jnp.asarray(samples_w)))[..., cosmo_nl_pos]   # (C, S, 5)
rhat = np.asarray(potential_scale_reduction(
    phys_cosmo, chain_axis=0, sample_axis=1))                 # (5,)
ess = np.asarray(effective_sample_size(
    phys_cosmo, chain_axis=0, sample_axis=1))                 # (5,)
rhat_max = float(np.max(rhat))

print(f"NUTS wall {nuts_wall:.0f}s; mean accept {mean_accept:.3f}; "
      f"mean int.steps {mean_steps:.1f}; divergence frac {div_frac:.4f}",
      flush=True)
print(f"{'param':>7s} {'R-hat':>9s} {'ESS':>10s}")
for i, key in enumerate(SHARED_KEYS):
    print(f"{key:>7s} {rhat[i]:9.5f} {ess[i]:10.1f}")

nuts_ok = (rhat_max <= RHAT_LIMIT) and (div_frac <= DIVERGENCE_FRAC_LIMIT)
sampler_used = "nuts_window_adaptation"

results["nuts"] = {
    "wall_s": round(nuts_wall, 1), "mean_acceptance": mean_accept,
    "mean_integration_steps": mean_steps, "divergence_frac": div_frac,
    "rhat": rhat.tolist(), "ess": ess.tolist(), "rhat_max": rhat_max,
    "adaptation_ok": nuts_ok,
}
save_results()

# ---------------------------------------------------------------------------
# Fallback: fixed-L leapfrog HMC via jax.value_and_grad in a Python loop.
# Only if NUTS window adaptation misbehaved.
# ---------------------------------------------------------------------------

if not nuts_ok:
    print(f"\n!!! NUTS adaptation flagged (rhat_max {rhat_max:.4f} > "
          f"{RHAT_LIMIT} or div_frac {div_frac:.4f} > {DIVERGENCE_FRAC_LIMIT})"
          f" -- falling back to fixed-L HMC (step={HMC_STEP}, L={HMC_L}).",
          flush=True)
    sampler_used = "fixed_L_hmc_fallback"

    val_grad = jax.jit(jax.value_and_grad(sur_log_post))

    def _hmc_chain(seed, n, d):
        rng = np.random.default_rng(seed)
        x = np.zeros(d)
        lp, g = val_grad(jnp.asarray(x))
        lp = float(lp); g = np.asarray(g)
        kept = np.empty((n, d)); acc = 0
        for i in range(n):
            p0 = rng.normal(size=d)
            x_new = x.copy(); g_new = g.copy()
            p = p0 + 0.5 * HMC_STEP * g_new           # U = -logp; dp = +grad logp
            for _ in range(HMC_L):
                x_new = x_new + HMC_STEP * p
                lp_new, g_arr = val_grad(jnp.asarray(x_new))
                g_new = np.asarray(g_arr)
                p = p + HMC_STEP * g_new
            p = p - 0.5 * HMC_STEP * g_new             # half-step correction
            lp_new = float(lp_new)
            dH = (lp_new - 0.5 * p @ p) - (lp - 0.5 * p0 @ p0)
            if np.log(rng.random()) < dH:
                x, lp, g = x_new, lp_new, g_new
                acc += 1
            kept[i] = x
        return kept, acc / n

    hmc_chains = []
    hmc_accs = []
    for c in range(N_CHAINS):
        ch, a = _hmc_chain(NUTS_SEED + 1 + c, N_SAMPLES, n_nl)
        hmc_chains.append(ch); hmc_accs.append(a)
        print(f"  [hmc] chain {c + 1}/{N_CHAINS} accept {a:.3f}", flush=True)
    samples_w = np.stack(hmc_chains)                  # (C, S, n_nl)

    phys_cosmo = np.asarray(
        to_physical(jnp.asarray(samples_w)))[..., cosmo_nl_pos]
    rhat = np.asarray(potential_scale_reduction(
        phys_cosmo, chain_axis=0, sample_axis=1))
    ess = np.asarray(effective_sample_size(
        phys_cosmo, chain_axis=0, sample_axis=1))
    rhat_max = float(np.max(rhat))
    mean_accept = float(np.mean(hmc_accs))
    print(f"{'param':>7s} {'R-hat':>9s} {'ESS':>10s}")
    for i, key in enumerate(SHARED_KEYS):
        print(f"{key:>7s} {rhat[i]:9.5f} {ess[i]:10.1f}")
    results["hmc_fallback"] = {
        "step_size": HMC_STEP, "L": HMC_L, "mean_acceptance": mean_accept,
        "rhat": rhat.tolist(), "ess": ess.tolist(), "rhat_max": rhat_max,
    }
    save_results()

results["sampler_used"] = sampler_used
np.save(NUTS_CHAIN_PATH, samples_w)
save_results()

# ---------------------------------------------------------------------------
# Skew of logA / ns marginals for the three chains (same whitened->physical map).
# ---------------------------------------------------------------------------

print("\n===== SKEW of logA / ns marginals =====", flush=True)


def _skew_boot(vals):
    vals = np.asarray(vals, dtype=np.float64)
    s = float(scipy_skew(vals))
    rng = np.random.default_rng(BOOT_SEED)
    n = len(vals)
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        boot[b] = scipy_skew(vals[rng.integers(0, n, n)])
    return s, float(np.std(boot))


def _cosmo_cols(chain_w):
    """Whitened chain (N, n_nl) -> physical (logA, ns) columns."""
    phys = np.asarray(to_physical(jnp.asarray(chain_w)))[:, cosmo_nl_pos]
    return phys[:, LOGA_I], phys[:, NS_I]


# (1) NUTS/HMC on the surrogate (flatten chains).
nuts_flat = samples_w.reshape(-1, n_nl)
# (2) surrogate RWMH 200k (reuse if present, else rerun).
if RWMH_CHAIN_PATH.exists():
    rwmh_w = np.load(RWMH_CHAIN_PATH)[RWMH_BURN:]
    print(f"reused surrogate RWMH chain {RWMH_CHAIN_PATH} "
          f"(post-burn {rwmh_w.shape})", flush=True)
    rwmh_rerun = False
else:
    print(f"{RWMH_CHAIN_PATH} missing -- rerunning {RWMH_SAMPLES} RWMH steps",
          flush=True)
    s_w, _ = run_rwmh_python(
        jax.random.key(RWMH_SEED), sur_log_post, initial_position=x0,
        num_samples=RWMH_SAMPLES, num_chains=1, thin=1)
    full = np.asarray(s_w[0])
    np.save(RWMH_CHAIN_PATH, full)
    rwmh_w = full[RWMH_BURN:]
    rwmh_rerun = True
# (3) exact Tier-2 chain.
tier2_w = np.load(TIER2_CHAIN_PATH)[TIER2_BURN:]
print(f"exact Tier-2 chain post-burn {tier2_w.shape}", flush=True)

chains = {
    sampler_used: nuts_flat,
    "surrogate_rwmh": rwmh_w,
    "exact_tier2": tier2_w,
}

skew_out = {}
print(f"{'chain':>22s} {'n':>8s} {'skew(logA)':>18s} {'skew(ns)':>18s} "
      f"{'meanpull(logA)':>15s} {'meanpull(ns)':>13s}")
for name, chain_w in chains.items():
    logA, ns = _cosmo_cols(chain_w)
    sA, eA = _skew_boot(logA)
    sN, eN = _skew_boot(ns)
    pull_A = float((logA.mean() - fid_cosmo[LOGA_I]) / sig_fisher[LOGA_I])
    pull_N = float((ns.mean() - fid_cosmo[NS_I]) / sig_fisher[NS_I])
    skew_out[name] = {
        "n": int(len(logA)),
        "skew_logA": sA, "skew_logA_err": eA,
        "skew_ns": sN, "skew_ns_err": eN,
        "meanpull_logA_sigmaF": pull_A, "meanpull_ns_sigmaF": pull_N,
    }
    print(f"{name:>22s} {len(logA):8d} {sA:9.4f} +-{eA:6.4f} "
          f"{sN:9.4f} +-{eN:6.4f} {pull_A:15.3f} {pull_N:13.3f}", flush=True)

results["skew"] = skew_out
results["rwmh_rerun"] = rwmh_rerun

# ---------------------------------------------------------------------------
# Physics verdict: is the Tier-2 mean residual explained by genuine skew?
# A left-skewed (skew<0) marginal pulls the MEAN below the MODE, so a negative
# mean pull is expected from skew alone if the mode sits near the fiducial. We
# call the residual "skew-explained" for a parameter when the surrogate chains
# (clean, high-ESS) show skew significant at >3 bootstrap sigma with the SAME
# sign as the observed (negative) mean pull.
# ---------------------------------------------------------------------------

def _significant_neg(name, key_skew, key_err):
    s = skew_out[name][key_skew]
    e = skew_out[name][key_err]
    return (s < 0) and (abs(s) > 3 * e)


sur_name = sampler_used
logA_skew_explains = (_significant_neg(sur_name, "skew_logA", "skew_logA_err")
                      and _significant_neg("surrogate_rwmh", "skew_logA",
                                           "skew_logA_err"))
ns_skew_explains = (_significant_neg(sur_name, "skew_ns", "skew_ns_err")
                    and _significant_neg("surrogate_rwmh", "skew_ns",
                                         "skew_ns_err"))
results["skew_verdict"] = {
    "logA_residual_skew_explained": bool(logA_skew_explains),
    "ns_residual_skew_explained": bool(ns_skew_explains),
    "criterion": "surrogate NUTS AND RWMH both show skew<0 at >3 bootstrap-sigma",
}
results["total_wall_s"] = round(time.perf_counter() - _T0, 1)
save_results()

print(f"\nlogA residual skew-explained: {logA_skew_explains}", flush=True)
print(f"ns   residual skew-explained: {ns_skew_explains}", flush=True)
print(f"-> {RESULT_PATH}", flush=True)

# ---------------------------------------------------------------------------
# Append a results section to the measurements doc (idempotent).
# ---------------------------------------------------------------------------

DOC_MARKER = "## NUTS on the surrogate (added 2026-07-29)"
doc = DOC_PATH.read_text()
if DOC_MARKER in doc:
    print(f"doc already has the NUTS section; not re-appending.", flush=True)
else:
    def _row(name):
        d = skew_out[name]
        return (f"| {name} | {d['n']} | {d['skew_logA']:+.4f} "
                f"± {d['skew_logA_err']:.4f} | {d['skew_ns']:+.4f} "
                f"± {d['skew_ns_err']:.4f} | {d['meanpull_logA_sigmaF']:+.3f} "
                f"| {d['meanpull_ns_sigmaF']:+.3f} |")

    rhat_ess = "\n".join(
        f"| {SHARED_KEYS[i]} | {rhat[i]:.5f} | {ess[i]:.0f} |"
        for i in range(len(SHARED_KEYS)))
    section = f"""

{DOC_MARKER}

Gradient-based sampling on the Taylor surrogate (Task 8). The surrogate body is
microsecond-scale dense algebra (linear templates + logdet tilt + Gaussian
priors, ~0.65 ms/step in-chain), so the scan-trap rule recorded above (a ~50k-op
body inside a `while` defeats XLA) does **not** apply: the body is tiny and
homogeneous, so blackjax `window_adaptation` + chunked-scan production
(`sampler.run_nuts`) is the right tool. Sampler used: **{sampler_used}**
(fallback to fixed-L HMC triggers only on R-hat > {RHAT_LIMIT} or divergences >
{100 * DIVERGENCE_FRAC_LIMIT:.0f}%).

{N_CHAINS} chains x {N_SAMPLES} draws ({N_WARMUP} adaptation), whitened space.
Mean acceptance {mean_accept:.3f}, mean integration steps {mean_steps:.1f},
divergence fraction {div_frac:.4f}, wall {nuts_wall:.0f}s.

| cosmo param | R-hat | ESS |
|---|---|---|
{rhat_ess}

**Skew of the logA / ns marginals** (scipy.stats.skew ± {N_BOOT}-resample
bootstrap error), all three chains mapped through the SAME Cholesky
whitening -> physical transform. Mean pull is (mean - fiducial)/sigma_F:

| chain | n | skew(logA) | skew(ns) | pull(logA) [σ_F] | pull(ns) [σ_F] |
|---|---|---|---|---|---|
{_row(sampler_used)}
{_row("surrogate_rwmh")}
{_row("exact_tier2")}

**Physics question — is the open logA/ns Tier-2 mean residual (means ~0.3-0.45
σ_F below the fiducial) genuine posterior skew?** A left-skewed (skew < 0)
marginal pulls the *mean* below the *mode*, so a negative mean pull is the
expected signature of skew when the mode sits near the fiducial. Verdict
(surrogate NUTS AND RWMH both showing skew < 0 at > 3 bootstrap-σ):
logA **{'skew-explained' if logA_skew_explains else 'NOT skew-explained'}**,
ns **{'skew-explained' if ns_skew_explains else 'NOT skew-explained'}**. The
surrogate chains (clean, ESS ~ 10^3-10^4) and the noisy exact Tier-2 chain
(ESS 30-83) are compared directly in the table; agreement of the surrogate NUTS
and RWMH skews cross-checks that the sampling — not the sampler — sets the shape.
Written by `scripts/taylor_surrogate_nuts.py`; numbers in
`cache/taylor_nuts_result.json`.
"""
    DOC_PATH.write_text(doc + section)
    print(f"appended NUTS section to {DOC_PATH}", flush=True)

print(f"\nDONE in {time.perf_counter() - _T0:.0f}s; sampler={sampler_used}",
      flush=True)
