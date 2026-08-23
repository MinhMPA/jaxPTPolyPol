"""Three-gate validation of the production Taylor-surrogate marginal posterior.

Loads ``cache/taylor_templates_lcdm.npz`` (with the strict ``expect_meta``
stale-config guard) and ``cache/taylor_whitening_lcdm.npz`` written by
``scripts/build_taylor_templates_lcdm.py``, rebuilds the SURROGATE
(``make_marginal_log_posterior_taylor``) and the EXACT reference
(``make_marginal_log_posterior_perbin``) posteriors from the SAME loaded
inputs, and runs three gates. Every number is written incrementally to
``cache/taylor_validation.json`` (after each gate, not only at the end) and a
PASS/REVIEW verdict is printed per gate.

Gates
-----
1. TILT: central FD gradient (h=0.02, whitened space) of the SURROGATE at
   x=0 vs the recorded EXACT-posterior logdet tilt ``cache/logdet_tilt.json``
   key ``g_w``. PASS: cosine > 0.99 AND |norm ratio - 1| < 0.1.
2. CHAIN-VS-CHAIN: 200_000-step RWMH on the surrogate (``jax.random.key
   (20260729)``, 1 chain, thin=1; measured s/step is THE headline number;
   burn the first 20_000) vs the exact reduced-Tier-2 chain
   ``cache/tier2_chain_w.npy`` (drop its first 500 burn). Both converted to
   physical space with the SAME loaded Cholesky whitening; cosmology columns
   via the notebook's ``cosmo_nl_pos`` construction. PASS: width ratios
   surrogate/exact within [0.9, 1.1] AND max correlation diff < 0.1. Mean
   differences are reported in sigma_F units (``sig_fisher`` from
   ``inv(F_pfs_bao_prior_cosmo)``); the exact chain's own MC error is
   ~0.15-0.2 sigma_F.
3. IS: ``importance_reweight`` of the surrogate chain against the exact
   posterior (subsample=400 exact evals ~ 400 x 5.06 s ~ 34 min, seed=0).
   PASS: ess_frac >= 0.2 (well-covered: > 0.5). Reports ess_frac,
   max_weight, and reweighted-vs-raw cosmology means in sigma_F.

STOP-FOR-USER conditions (any gate REVIEW): tilt cosine < 0.99, width ratios
outside +-10%, or ess_frac < 0.2.

Run from ``example/mcmc`` after the build script::

    cd example/mcmc
    caffeinate -i python3 scripts/taylor_surrogate_validation.py
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

from stream_common import (
    BACKGROUND_MODE, BK_DO_IRRES, DEFAULT_BAO_DATA_DIR, K_NL_RSD, META, N_GL, NUM_MU,
    NUM_PHI, PFS_EMULATOR, SHARED_KEYS, build_bao, build_fiducial_surveys,
    build_kgrid_and_blocks, build_split, load_templates_and_whitening,
    n_zbins, z_bins,
)

from jaxptpolypol import (
    bin_lin_slices,
    importance_reweight,
    make_constant_prior_fns,
    make_marginal_log_posterior_perbin,
    make_marginal_log_posterior_taylor,
    reweighted_moments,
)
from jaxptpolypol.model import BispectrumTreeModel, CosmoEmulator, PS1LoopModel
from jaxptpolypol.params import pack_joint_params
from jaxptpolypol.sampler import (
    make_cholesky_transform,
    make_full_params_fn,
    make_gaussian_log_prior,
    run_rwmh_python,
)
from jaxptpolypol.theory import (
    compute_fiducial_distances,
    make_joint_pk_bk_bin_fn,
)

# ---------------------------------------------------------------------------
# RSS watchdog (same protocol as the build script; abort limit 70 GB -- the
# exact-posterior compile is the heaviest step here, measured ~28.5 GB).
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
# Configuration. Shared production constants (fiducial cosmology, redshift bins,
# k-grid, quadrature, PFS_EMULATOR, SHARED_KEYS, META) are imported from
# stream_common (single source of truth, in lockstep with
# scripts/build_taylor_templates_lcdm.py; the META guard + the packed-vector
# tripwire below enforce it).
# ---------------------------------------------------------------------------

BAO_DATA_DIR = DEFAULT_BAO_DATA_DIR   # chdir-sensitive: run from example/mcmc

# Gate parameters (task 7 dispatch).
FD_H = 0.02                       # matches cache/logdet_tilt.json fd_h
RNG_SEED_CHAIN = 20260729
NUM_SAMPLES_SUR = 200_000
BURN_SUR = 20_000
EXACT_BURN = 500                  # tier2_chain_w.npy burn-in (its run mode)
IS_SUBSAMPLE = 400
IS_SEED = 0

CACHE = pathlib.Path("cache")
WHITENING_PATH = CACHE / "taylor_whitening_lcdm.npz"
TEMPLATES_PATH = CACHE / "taylor_templates_lcdm.npz"
TILT_PATH = CACHE / "logdet_tilt.json"
EXACT_CHAIN_PATH = CACHE / "tier2_chain_w.npy"
CHAIN_OUT_PATH = CACHE / "taylor_chain_w.npy"
RESULT_PATH = CACHE / "taylor_validation.json"

for p in (WHITENING_PATH, TEMPLATES_PATH, TILT_PATH, EXACT_CHAIN_PATH):
    if not p.exists():
        sys.exit(f"Required artifact missing: {p} -- run from example/mcmc "
                 "after scripts/build_taylor_templates_lcdm.py.")
if not pathlib.Path(BAO_DATA_DIR).is_dir():
    sys.exit(f"BAO data dir not found at {BAO_DATA_DIR!r} -- run from "
             "example/mcmc.")

results = {"config": {"meta": META, "fd_h": FD_H,
                      "rng_seed_chain": RNG_SEED_CHAIN,
                      "num_samples_surrogate": NUM_SAMPLES_SUR,
                      "burn_surrogate": BURN_SUR,
                      "exact_chain_burn": EXACT_BURN,
                      "is_subsample": IS_SUBSAMPLE, "is_seed": IS_SEED}}


def save_results():
    RESULT_PATH.write_text(json.dumps(results, indent=1))


save_results()

# ---------------------------------------------------------------------------
# Load artifacts (strict meta guards) and rebuild both posteriors from the
# SAME loaded inputs.
# ---------------------------------------------------------------------------

_watch["stage"] = "load + assemble posteriors"
print(f"===== {_watch['stage']} =====", flush=True)

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
F_pfs_bao_prior_cosmo = jnp.asarray(wz["F_pfs_bao_prior_cosmo"])
sig_fisher = np.asarray(wz["sig_fisher"])
cosmo_nl_pos = [int(i) for i in wz["cosmo_nl_pos"]]
nl_prior_entries = [
    (int(p), float(m), float(s))
    for p, m, s in zip(wz["nl_prior_pos"], wz["nl_prior_mean"],
                       wz["nl_prior_sigma"])]

# Rebuild the theory statics (cannot be serialized) -- notebook cell
# "Emulator, models, fiducial parameters", verbatim.
pklin_emulator = CosmoEmulator(probe='custom_log', emulator_path=PFS_EMULATOR)
ps1loop_model = PS1LoopModel(do_irres=True)
bispectrum_model = BispectrumTreeModel(
    do_irres=BK_DO_IRRES, do_AP=True, k_nl_rsd=K_NL_RSD)

cosmo_dict, cosmo, surveys, joint_survey_keys = build_fiducial_surveys()
n_cosmo_params = sum(cosmo.param_sizes)
Hz_fid, DAz_fid = compute_fiducial_distances(cosmo, z_bins)

# Config-drift tripwire: the rebuilt fiducial packed vector must equal the one
# the templates/whitening were built with, bit for bit.
packed_rebuilt = pack_joint_params(cosmo, surveys)
if not np.array_equal(np.asarray(packed_rebuilt), np.asarray(packed_params)):
    sys.exit("ABORT: rebuilt packed fiducial vector differs from the stored "
             "one -- config drift between build and validation scripts.")

split = build_split(n_cosmo_params, joint_survey_keys)
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

k, dk, triangles, block_len, bin_blocks = build_kgrid_and_blocks()
bin_data = [pb_fid[sl] for sl in bin_blocks]

bao_dr2, bao_theory_fn = build_bao(
    BAO_DATA_DIR, cosmo, packed_params[:n_cosmo_params], bao_fid)

# Shared posterior pieces -- notebook cell "Combined likelihood", using the
# LOADED priors/whitening.
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

sur_log_post = make_marginal_log_posterior_taylor(tt, **shared_post_kwargs)

x0 = jnp.zeros(n_nl)
t0 = time.perf_counter()
lp0_sur = float(jax.block_until_ready(sur_log_post(x0)))
print(f"surrogate log_post(x0) = {lp0_sur:.6f} "
      f"(first call {time.perf_counter() - t0:.1f}s)", flush=True)
t0 = time.perf_counter()
_ = float(sur_log_post(x0 * 0.0 + 0.01))
sur_eval_s = time.perf_counter() - t0
print(f"surrogate cached eval: {sur_eval_s * 1000:.2f} ms", flush=True)
results["surrogate"] = {"lp0": lp0_sur,
                        "cached_eval_ms": round(sur_eval_s * 1000, 3)}
save_results()

# ---------------------------------------------------------------------------
# GATE 1: logdet-tilt fidelity (52 surrogate evals, seconds).
# ---------------------------------------------------------------------------

_watch["stage"] = "gate 1: tilt"
print(f"\n===== GATE 1: TILT (FD h={FD_H}, whitened) =====", flush=True)

tilt_ref = json.loads(TILT_PATH.read_text())
assert tilt_ref["fd_h"] == FD_H, \
    f"logdet_tilt.json used fd_h={tilt_ref['fd_h']}, script has {FD_H}"
g_ref = np.asarray(tilt_ref["g_w"])

g_sur = np.zeros(n_nl)
t0 = time.perf_counter()
for i in range(n_nl):
    e = np.zeros(n_nl); e[i] = FD_H
    lp_p = float(sur_log_post(jnp.asarray(e)))
    lp_m = float(sur_log_post(jnp.asarray(-e)))
    g_sur[i] = (lp_p - lp_m) / (2.0 * FD_H)
print(f"FD gradient of the surrogate: {time.perf_counter() - t0:.1f}s",
      flush=True)

cos_tilt = float(g_sur @ g_ref
                 / (np.linalg.norm(g_sur) * np.linalg.norm(g_ref)))
norm_ratio = float(np.linalg.norm(g_sur) / np.linalg.norm(g_ref))
gate1_pass = (cos_tilt > 0.99) and (abs(norm_ratio - 1.0) < 0.1)

print(f"cosine(g_sur, g_exact) = {cos_tilt:.6f}   (require > 0.99)")
print(f"|g_sur|/|g_exact|      = {norm_ratio:.6f}   (require within 1 +- 0.1)")
print(f"GATE 1: {'PASS' if gate1_pass else 'REVIEW'}", flush=True)

results["gate1_tilt"] = {
    "cosine": cos_tilt, "norm_ratio": norm_ratio,
    "g_norm_surrogate": float(np.linalg.norm(g_sur)),
    "g_norm_exact": float(np.linalg.norm(g_ref)),
    "g_surrogate": g_sur.tolist(),
    "status": "PASS" if gate1_pass else "REVIEW",
}
save_results()

# ---------------------------------------------------------------------------
# GATE 2: surrogate chain vs exact chain.
# ---------------------------------------------------------------------------

_watch["stage"] = "gate 2: chain"
print(f"\n===== GATE 2: CHAIN ({NUM_SAMPLES_SUR} RWMH steps on the surrogate) "
      "=====", flush=True)

_chain_t0 = time.perf_counter()


def _on_draw(_chain, _nchains, done, total):
    if done % 10_000 == 0 or done == total:
        el = time.perf_counter() - _chain_t0
        print(f"  [chain] {done}/{total} draws, {el:7.0f}s, "
              f"{el / max(done, 1) * 1000:.2f} ms/step", flush=True)


samples_w, diagnostics = run_rwmh_python(
    jax.random.key(RNG_SEED_CHAIN), sur_log_post, initial_position=x0,
    num_samples=NUM_SAMPLES_SUR, num_chains=1, thin=1,
    sample_progress_fn=_on_draw)
chain_wall = time.perf_counter() - _chain_t0
s_per_step = chain_wall / NUM_SAMPLES_SUR
acc = float(np.asarray(diagnostics["acceptance_rate"])[0])
print(f"chain wall {chain_wall:.0f}s -> {s_per_step * 1000:.2f} ms/step "
      f"(HEADLINE); acceptance {acc:.3f}", flush=True)

chain_w_full = np.asarray(samples_w[0])            # (NUM_SAMPLES_SUR, n_nl)
np.save(CHAIN_OUT_PATH, chain_w_full)
print(f"-> {CHAIN_OUT_PATH}", flush=True)

sur_w = chain_w_full[BURN_SUR:]
ex_w = np.load(EXACT_CHAIN_PATH)[EXACT_BURN:]
print(f"surrogate post-burn {sur_w.shape}, exact post-burn {ex_w.shape}",
      flush=True)

# Physical space via the SAME loaded whitening; cosmology columns as in the
# notebook (cosmo_nl_pos).
sur_phys = np.asarray(to_physical(jnp.asarray(sur_w)))
ex_phys = np.asarray(to_physical(jnp.asarray(ex_w)))
cs = sur_phys[:, cosmo_nl_pos]                     # (n, 5)
ce = ex_phys[:, cosmo_nl_pos]

mean_diff_sig = np.abs(cs.mean(axis=0) - ce.mean(axis=0)) / sig_fisher
width_ratio = cs.std(axis=0) / ce.std(axis=0)
corr_s = np.corrcoef(cs, rowvar=False)
corr_e = np.corrcoef(ce, rowvar=False)
corr_diff_max = float(np.max(np.abs(corr_s - corr_e)))
# Surrogate-chain mean pulls off the fiducial (the logdet tilt, sigma_F units)
fid_cosmo = np.asarray(fid_nl)[cosmo_nl_pos]
sur_pulls = (cs.mean(axis=0) - fid_cosmo) / sig_fisher
ex_pulls = (ce.mean(axis=0) - fid_cosmo) / sig_fisher

widths_ok = bool(np.all((width_ratio >= 0.9) & (width_ratio <= 1.1)))
corr_ok = corr_diff_max < 0.1
gate2_pass = widths_ok and corr_ok

print(f"{'param':>7s} {'|dmean|/sigF':>12s} {'width s/e':>10s} "
      f"{'sur pull':>9s} {'ex pull':>9s}")
for i, key in enumerate(SHARED_KEYS):
    print(f"{key:>7s} {mean_diff_sig[i]:12.4f} {width_ratio[i]:10.4f} "
          f"{sur_pulls[i]:9.3f} {ex_pulls[i]:9.3f}")
print(f"max corr diff = {corr_diff_max:.4f}  (require < 0.1)")
print(f"width ratios in [0.9, 1.1]: {widths_ok}; note: exact-chain MC error "
      "on means is ~0.15-0.2 sigma_F (ESS 30-83)")
print(f"GATE 2: {'PASS' if gate2_pass else 'REVIEW'}", flush=True)

results["gate2_chain"] = {
    "s_per_step": s_per_step,
    "chain_wall_s": round(chain_wall, 1),
    "acceptance": acc,
    "n_kept_surrogate": int(sur_w.shape[0]),
    "n_kept_exact": int(ex_w.shape[0]),
    "names": list(SHARED_KEYS),
    "mean_diff_sigma_F": mean_diff_sig.tolist(),
    "width_ratio": width_ratio.tolist(),
    "corr_diff_max": corr_diff_max,
    "surrogate_mean_pull_sigma_F": sur_pulls.tolist(),
    "exact_mean_pull_sigma_F": ex_pulls.tolist(),
    "sig_fisher": sig_fisher.tolist(),
    "widths_ok": widths_ok, "corr_ok": corr_ok,
    "status": "PASS" if gate2_pass else "REVIEW",
}
save_results()

# ---------------------------------------------------------------------------
# GATE 3: importance-sampling diagnostics against the exact posterior.
# ---------------------------------------------------------------------------

_watch["stage"] = "gate 3: exact compile"
print(f"\n===== GATE 3: IS (subsample={IS_SUBSAMPLE} exact evals) =====",
      flush=True)

bin_theory_fns = [
    partial(make_joint_pk_bk_bin_fn(bin_index=b, **joint_theory_kwargs),
            k=k, triangles=triangles)
    for b in range(n_zbins)]
exact_log_post = make_marginal_log_posterior_perbin(
    bin_theory_fns=bin_theory_fns, bin_lin_idx=bin_lin_idx,
    **shared_post_kwargs)

t0 = time.perf_counter()
lp0_exact = float(jax.block_until_ready(exact_log_post(x0)))
exact_compile_s = time.perf_counter() - t0
center_diff = lp0_exact - lp0_sur
print(f"exact log_post(x0) = {lp0_exact:.6f} (compile+eval "
      f"{exact_compile_s:.1f}s, rss {_rss_gb():.1f} GB); "
      f"exact - surrogate at center = {center_diff:.3e}", flush=True)

_watch["stage"] = "gate 3: IS loop"
_is_t0 = time.perf_counter()
_is_count = [0]


def _exact_counting(x):
    v = exact_log_post(x)
    _is_count[0] += 1
    if _is_count[0] % 10 == 0:
        el = time.perf_counter() - _is_t0
        print(f"  [IS] exact evals {_is_count[0]}/{IS_SUBSAMPLE}, "
              f"{el:6.0f}s ({el / _is_count[0]:.2f} s/eval)", flush=True)
    return v


res = importance_reweight(sur_w, _exact_counting, sur_log_post,
                          subsample=IS_SUBSAMPLE, seed=IS_SEED)
is_wall = time.perf_counter() - _is_t0

rw_mean, rw_std = reweighted_moments(cs, res["weights"], idx=res["idx"])
raw_mean = cs.mean(axis=0)
rw_shift_sig = (rw_mean - raw_mean) / sig_fisher

ess_frac = float(res["ess_frac"])
max_weight = float(res["max_weight"])
gate3_pass = ess_frac >= 0.2
well_covered = ess_frac > 0.5

print(f"IS wall {is_wall:.0f}s; ess = {res['ess']:.1f} / {IS_SUBSAMPLE} "
      f"(ess_frac {ess_frac:.3f}), max_weight {max_weight:.4f}")
print(f"{'param':>7s} {'raw mean':>12s} {'reweighted':>12s} "
      f"{'shift/sigF':>10s}")
for i, key in enumerate(SHARED_KEYS):
    print(f"{key:>7s} {raw_mean[i]:12.6g} {rw_mean[i]:12.6g} "
          f"{rw_shift_sig[i]:10.4f}")
print(f"well-covered (ess_frac > 0.5): {well_covered}")
print(f"GATE 3: {'PASS' if gate3_pass else 'REVIEW'}", flush=True)

results["gate3_is"] = {
    "ess": float(res["ess"]),
    "ess_frac": ess_frac,
    "max_weight": max_weight,
    "well_covered": well_covered,
    "is_wall_s": round(is_wall, 1),
    "exact_compile_s": round(exact_compile_s, 1),
    "lp0_exact": lp0_exact,
    "center_diff_exact_minus_surrogate": center_diff,
    "names": list(SHARED_KEYS),
    "raw_mean": raw_mean.tolist(),
    "reweighted_mean": rw_mean.tolist(),
    "reweighted_std": rw_std.tolist(),
    "reweighted_shift_sigma_F": rw_shift_sig.tolist(),
    "status": "PASS" if gate3_pass else "REVIEW",
}
save_results()

# ---------------------------------------------------------------------------
# Summary.
# ---------------------------------------------------------------------------

overall = ("PASS" if gate1_pass and gate2_pass and gate3_pass else "REVIEW")
results["overall"] = {
    "status": overall,
    "gate1": results["gate1_tilt"]["status"],
    "gate2": results["gate2_chain"]["status"],
    "gate3": results["gate3_is"]["status"],
    "peak_rss_gb": round(_watch["peak_gb"], 2),
    "total_wall_s": round(time.perf_counter() - _T0, 1),
}
save_results()

print(f"\n===== VALIDATION {overall} =====")
print(f"gate1 tilt   : {results['gate1_tilt']['status']} "
      f"(cos {cos_tilt:.4f}, ratio {norm_ratio:.4f})")
print(f"gate2 chain  : {results['gate2_chain']['status']} "
      f"(widths {np.round(width_ratio, 3)}, corr diff {corr_diff_max:.3f}, "
      f"{s_per_step * 1000:.2f} ms/step)")
print(f"gate3 IS     : {results['gate3_is']['status']} "
      f"(ess_frac {ess_frac:.3f}, max_w {max_weight:.4f})")
print(f"-> {RESULT_PATH}", flush=True)
print(f"VALIDATION COMPLETE in {time.perf_counter() - _T0:.0f}s; peak RSS "
      f"{_watch['peak_gb']:.1f} GB", flush=True)
