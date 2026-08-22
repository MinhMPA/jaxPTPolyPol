"""Tier-3 validation: SAMPLED-c1 vs MARGINALIZED-c1 posteriors on the mock.

The last outstanding c1 validation (theory.md: Why c1 sits in the linear
block): build two surrogate
posteriors that differ ONLY in how the c1 bispectrum FoG counterterm is handled,
under the SAME DESI DR1-reanalysis (2511.20757) priors, and confirm their
COSMOLOGY posteriors are indistinguishable.

  MARGINALIZED side  c1 in theta_lin, integrated analytically -- the production
                     path. The marginal templates model the theory as LINEAR in
                     c1 (m0/M interrogated only at theta_lin=0), so the D'Amico
                     c1^2 (2502.14758 eq 3.14; relative size ~6e-4) is dropped.
                     Uses the EXISTING cache (taylor_{templates,whitening}_lcdm.npz).
  SAMPLED side       c1 in theta_NL, explored by the chain. The theory is EXACTLY
                     quadratic in c1, and the order-2 m0 Taylor surrogate captures
                     that c1^2 EXACTLY -- so the sampled surrogate carries the
                     physics the marginalized path linearizes away. Uses the
                     _c1s cache built by
                     ``build_taylor_templates_lcdm.py --c1-sampled`` (PART 2).

Both sides carry an EQUIVALENT c1 prior: N(0, (1.0125 / (A_AP*A_amp))^2). On the
marginalized side it is the c1 marginalized-row diagonal entry (make_desi_prior_fns
default 11 lin keys); on the sampled side it is a per-bin sampled prior added to
log_prior_nl_fn via ``sampled_marginal_priors`` with the same rescale.

Gates (cosmology block, 5 params ombh2/omch2/logA/ns/h)
------------------------------------------------------
  G1 means : per-param |mean_marg - mean_samp| < 2.5 * combined-MC-SE + 0.02 sig_F
  G2 widths: sig_samp / sig_marg in [0.95, 1.05]
  G3 corrs : max |corr_marg - corr_samp| (upper triangle) < 0.05
Plus a report of the SAMPLED chain's c1 marginals (expect prior-dominated:
mean ~ 0, sigma ~ 1.0 dimensionless).

Writes cache/tier3_c1_validation.json with the verdict + all numbers.

Run from example/mcmc AFTER both builds::

    cd example/mcmc
    caffeinate -i python3 scripts/build_taylor_templates_lcdm.py            # marginalized (or reuse cache)
    caffeinate -i python3 scripts/build_taylor_templates_lcdm.py --c1-sampled  # sampled (PART 2)
    python3 scripts/tier3_c1_validation.py
    # smoke run (2000/200): prepend TIER3_SMOKE=1

FAILS FAST with a clear message if the _c1s cache is absent (PART 2 not run).
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

from stream_common import (
    DEFAULT_BAO_DATA_DIR, META, MNU_FIXED, PFS_EMULATOR, SHARED_KEYS,
    build_bao, build_fiducial_surveys, build_kgrid_and_blocks, build_split,
    knl_bins, load_templates_and_whitening, n_zbins, z_bins,
)

from jaxptpolypol import LIN_SURVEY_KEYS, make_marginal_log_posterior_taylor
from jaxptpolypol.desi_priors import (
    load_desi_prior_spec,
    make_desi_prior_fns,
    make_lcdm_rescaling_fns,
)
from jaxptpolypol.marginal_likelihood import split_marginal_indices
from jaxptpolypol.model import CosmoEmulator
from jaxptpolypol.params import pack_joint_params
from jaxptpolypol.sampler import (
    make_cholesky_transform,
    make_full_params_fn,
    make_gaussian_log_prior,
    run_rwmh_python,
)

_C1_KEY = ("bk", "ctr", "c1")
LIN_KEYS_NO_C1 = tuple(k for k in LIN_SURVEY_KEYS if k != _C1_KEY)
N_COSMO_NL = len(SHARED_KEYS)                              # == 5, gate cosmo block

# ---------------------------------------------------------------------------
# RSS watchdog (surrogate-only; well below the abort limit).
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
# Config / paths.
# ---------------------------------------------------------------------------

BAO_DATA_DIR = DEFAULT_BAO_DATA_DIR
if not pathlib.Path(BAO_DATA_DIR).is_dir():
    _master_bao = pathlib.Path(
        "/Users/nguyenmn/jaxPTPolyPol/ext_data/bao_data/desi_bao_dr2")
    if _master_bao.is_dir():
        BAO_DATA_DIR = str(_master_bao)

SMOKE = os.environ.get("TIER3_SMOKE") == "1"
NUM_SAMPLES = 2_000 if SMOKE else 200_000
BURN = 200 if SMOKE else 20_000
SEED_MARG = 20260731
SEED_SAMP = 20260801                       # distinct seed per side

HERE = pathlib.Path(__file__).resolve().parents[1]        # example/mcmc
CACHE = HERE / "cache"
if (not (CACHE / "taylor_whitening_lcdm.npz").exists()
        and (CACHE / "cache" / "taylor_whitening_lcdm.npz").exists()):
    CACHE = CACHE / "cache"                                # nested symlink -> master

MARG_TEMPLATES = CACHE / "taylor_templates_lcdm.npz"
MARG_WHITENING = CACHE / "taylor_whitening_lcdm.npz"
SAMP_TEMPLATES = CACHE / "taylor_templates_lcdm_c1s.npz"
SAMP_WHITENING = CACHE / "taylor_whitening_lcdm_c1s.npz"
# A SMOKE run writes to its own filename: the production result is a TRACKED
# gate artifact, and a 2000-step smoke chain must never be able to overwrite it.
RESULT_PATH = CACHE / (f"tier3_c1_validation{'_smoke' if SMOKE else ''}.json")

for p in (MARG_TEMPLATES, MARG_WHITENING):
    if not p.exists():
        sys.exit(f"Required MARGINALIZED artifact missing: {p} -- run from "
                 "example/mcmc after scripts/build_taylor_templates_lcdm.py.")
# FAIL FAST if PART 2 (the sampled build) has not run yet.
_missing_samp = [p for p in (SAMP_TEMPLATES, SAMP_WHITENING) if not p.exists()]
if _missing_samp:
    sys.exit(
        "Sampled-c1 cache missing: "
        + ", ".join(str(p) for p in _missing_samp)
        + "\nPART 2 has not been run. Build it first (from example/mcmc):\n"
        "    caffeinate -i python3 scripts/build_taylor_templates_lcdm.py --c1-sampled\n"
        "then re-run this validation.")
if not pathlib.Path(BAO_DATA_DIR).is_dir():
    sys.exit(f"BAO data dir not found at {BAO_DATA_DIR!r} -- run from example/mcmc.")

results = {"config": {"meta": META, "smoke": SMOKE,
                      "num_samples": NUM_SAMPLES, "burn": BURN,
                      "seed_marg": SEED_MARG, "seed_samp": SEED_SAMP}}


def save_results():
    RESULT_PATH.write_text(json.dumps(results, indent=1))


save_results()

# ---------------------------------------------------------------------------
# Shared theory statics (rebuilt once; identical for both sides).
# ---------------------------------------------------------------------------

_watch["stage"] = "load + assemble"
print(f"===== Tier-3 c1 validation (smoke={SMOKE}) =====", flush=True)

pklin_emulator = CosmoEmulator(probe='custom_log', emulator_path=PFS_EMULATOR)
cosmo_dict, cosmo, surveys, joint_survey_keys = build_fiducial_surveys()
n_cosmo_params = sum(cosmo.param_sizes)
packed_rebuilt = pack_joint_params(cosmo, surveys)
k, dk, triangles, block_len, bin_blocks = build_kgrid_and_blocks()
n_survey = len(joint_survey_keys)
c1_survey_off = joint_survey_keys.index(_C1_KEY)

FIXED_BARYON = {'A_b': cosmo_dict['A_b'], 'eta_b': cosmo_dict['eta_b'],
                'logT_AGN': cosmo_dict['logT_AGN']}
spec = load_desi_prior_spec()


def _build_side(name, templates_path, whitening_path, split,
                sampled_marginal_priors, lin_keys):
    """Assemble one side's surrogate log-posterior + chain inputs.

    ``name`` IS the c1 treatment ('marginalized' | 'sampled'), so both sides get
    their treatment AND the theory-config hash verified against the stamps on
    the npz they load -- a stale cache or a swapped marg/sampled pair is a hard
    failure, not a warning (stream_common.load_templates_and_whitening).

    Returns a dict with the jitted log-posterior, whitening (center/cov),
    fid_nl, cosmo_nl_pos and the sampled-c1 nl positions (empty for marg)."""
    tt, wz = load_templates_and_whitening(
        templates_path, whitening_path, treatment=name)
    packed_params = jnp.asarray(wz["packed_params"])
    pb_fid = jnp.asarray(wz["pb_fid"])
    bin_cov_invs = [jnp.asarray(c) for c in wz["bin_cov_invs"]]
    bao_fid = jnp.asarray(wz["bao_fid"])
    bao_cov_inv = jnp.asarray(wz["bao_cov_inv"])
    fid_nl = jnp.asarray(wz["fid_nl"])
    cov_nl_prior = jnp.asarray(wz["cov_nl_prior"])
    sig_fisher = np.asarray(wz["sig_fisher"])
    cosmo_nl_pos = [int(i) for i in wz["cosmo_nl_pos"]]
    # BBN(ombh2)/ns cosmology priors only -- filter to cosmo nl positions so a
    # legacy c1 prior_sigma that lands in the sampled side's nl block is NOT
    # double-counted with the DESI sampled c1 prior below.
    nl_prior_entries = [
        (int(p), float(m), float(s))
        for p, m, s in zip(wz["nl_prior_pos"], wz["nl_prior_mean"],
                           wz["nl_prior_sigma"]) if int(p) < N_COSMO_NL]

    # Config-drift tripwires (identical to desi_prior_validation.py).
    if not np.array_equal(np.asarray(packed_rebuilt), np.asarray(packed_params)):
        sys.exit(f"ABORT[{name}]: rebuilt packed fiducial vector differs from "
                 "the stored one -- config drift.")
    if (not np.array_equal(np.asarray(split.nl_idx), wz["nl_idx"])
            or not np.array_equal(np.asarray(split.lin_idx), wz["lin_idx"])):
        sys.exit(f"ABORT[{name}]: rebuilt marginal split differs from stored.")

    bin_data = [pb_fid[sl] for sl in bin_blocks]
    bao_dr2, bao_theory_fn = build_bao(
        BAO_DATA_DIR, cosmo, packed_params[:n_cosmo_params], bao_fid)

    sigma8_bins_fn, a_ap_bins_fn, sigma8_ref_bins = make_lcdm_rescaling_fns(
        pklin_emulator=pklin_emulator, cosmo_keys=SHARED_KEYS,
        cosmo_sizes=(1,) * N_COSMO_NL, z_bins=z_bins,
        fid_cosmo_native=fid_nl[:N_COSMO_NL], mnu_fixed=MNU_FIXED,
        fixed_cosmo_extras=FIXED_BARYON)

    # Sampled-c1 nl positions (production order: b1,b2,bG2,c1 -> nl_b1_pos + 3).
    nl_pos = {full: pos for pos, full in enumerate(split.nl_idx)}
    c1_positions = ([nl_pos[n_cosmo_params + b * n_survey + c1_survey_off]
                     for b in range(n_zbins)] if sampled_marginal_priors else [])
    smp = ([(_C1_KEY, c1_positions)] if sampled_marginal_priors else ())

    desi_mean_fn, desi_sigma_fn, desi_log_prior_nl = make_desi_prior_fns(
        spec, split=split, knl_bins=knl_bins, sigma8_bins_fn=sigma8_bins_fn,
        a_ap_bins_fn=a_ap_bins_fn, sigma8_ref_bins=sigma8_ref_bins,
        lin_keys=lin_keys, sampled_marginal_priors=smp)

    # cov-mode sanity (ctr trio -> (n_bins, len(lin_keys), len(lin_keys))).
    _sig0 = desi_sigma_fn(fid_nl)
    exp_lin = len(lin_keys)
    if getattr(_sig0, "ndim", None) != 3 or _sig0.shape != (
            n_zbins, exp_lin, exp_lin):
        sys.exit(f"ABORT[{name}]: expected cov-mode prior_sigma_fn "
                 f"(n_bins,{exp_lin},{exp_lin}); got {getattr(_sig0,'shape',None)}.")

    bbn_ns_log_prior = make_gaussian_log_prior(split.n_nl, nl_prior_entries)

    def log_prior_nl(theta_nl):
        return desi_log_prior_nl(theta_nl) + bbn_ns_log_prior(theta_nl)

    to_whitened, to_physical = make_cholesky_transform(
        center=fid_nl, cov=cov_nl_prior)
    full_params_fn = make_full_params_fn(packed_params, split.nl_idx)

    log_post = make_marginal_log_posterior_taylor(
        tt, bin_data=bin_data, bin_cov_invs=bin_cov_invs,
        prior_mean_fn=desi_mean_fn, prior_sigma_fn=desi_sigma_fn,
        log_prior_nl_fn=log_prior_nl, to_physical=to_physical,
        full_params_fn=full_params_fn,
        extra_theory_fn=lambda p: bao_theory_fn(p[:n_cosmo_params]),
        extra_data=bao_fid, extra_cov_inv=bao_cov_inv, include_logdet=True)
    log_post_j = jax.jit(log_post)

    w0 = jnp.zeros(split.n_nl)
    t0 = time.perf_counter()
    lp0 = float(jax.block_until_ready(log_post_j(w0)))
    print(f"[{name}] log_post(0) = {lp0:.6f} (first call "
          f"{time.perf_counter() - t0:.1f}s, rss {_rss_gb():.1f} GB)", flush=True)

    return {
        "name": name, "log_post_j": log_post_j, "w0": w0,
        "fid_nl": np.asarray(fid_nl),
        "cov_nl_prior": np.asarray(cov_nl_prior),
        "sig_fisher": sig_fisher, "cosmo_nl_pos": cosmo_nl_pos,
        "c1_positions": c1_positions, "lp0": lp0,
    }


# Marginalized side: existing cache, default 11 lin keys (c1 marginalized).
marg_split = build_split(n_cosmo_params, joint_survey_keys)   # standard split
marg = _build_side(
    "marginalized", MARG_TEMPLATES, MARG_WHITENING, marg_split,
    sampled_marginal_priors=None, lin_keys=LIN_SURVEY_KEYS)

# Sampled side: _c1s cache, reduced 10 lin keys (c1 sampled + its DESI prior).
samp_split = split_marginal_indices(
    n_cosmo_params=n_cosmo_params, survey_keys=joint_survey_keys, n_bins=n_zbins,
    fixed_cosmo=[5, 6, 7, 8],
    fixed_survey_keys={('shared', 'k_nl', None), ('shared', 'ndens', None)},
    lin_survey_keys=LIN_KEYS_NO_C1)
samp = _build_side(
    "sampled", SAMP_TEMPLATES, SAMP_WHITENING, samp_split,
    sampled_marginal_priors=True, lin_keys=LIN_KEYS_NO_C1)

# The two sides must share the cosmology fiducial + Fisher widths.
if not np.allclose(marg["fid_nl"][:N_COSMO_NL], samp["fid_nl"][:N_COSMO_NL]):
    sys.exit("ABORT: cosmology fiducial differs between the two sides.")
if not np.allclose(marg["sig_fisher"], samp["sig_fisher"], rtol=1e-6):
    print("WARNING: sig_fisher differs between sides "
          f"(marg={marg['sig_fisher']}, samp={samp['sig_fisher']}).", flush=True)
sig_F = marg["sig_fisher"][:N_COSMO_NL]
results["config"]["n_nl"] = {"marginalized": int(marg_split.n_nl),
                             "sampled": int(samp_split.n_nl)}
results["surrogate_lp0"] = {"marginalized": marg["lp0"], "sampled": samp["lp0"]}
save_results()

# ---------------------------------------------------------------------------
# Chains (distinct seeds; each side in its own whitened space).
# ---------------------------------------------------------------------------


def _run_chain(side, seed):
    _watch["stage"] = f"chain[{side['name']}]"
    print(f"===== CHAIN [{side['name']}] ({NUM_SAMPLES} steps, burn {BURN}) =====",
          flush=True)
    t0 = time.perf_counter()

    def _on_draw(_c, _nc, done, total):
        if done % (10_000 if not SMOKE else 500) == 0 or done == total:
            el = time.perf_counter() - t0
            print(f"  [{side['name']}] {done}/{total} draws, {el:7.0f}s, "
                  f"{el / max(done, 1) * 1000:.2f} ms/step", flush=True)

    samples_w, diag = run_rwmh_python(
        jax.random.key(seed), side["log_post_j"], initial_position=side["w0"],
        num_samples=NUM_SAMPLES, num_chains=1, thin=1, sample_progress_fn=_on_draw)
    wall = time.perf_counter() - t0
    acc = float(np.asarray(diag["acceptance_rate"])[0])
    print(f"[{side['name']}] chain wall {wall:.0f}s -> "
          f"{wall / NUM_SAMPLES * 1000:.2f} ms/step; acceptance {acc:.3f}",
          flush=True)
    chain_w = np.asarray(samples_w[0])[BURN:]              # (n_kept, n_nl)
    # Physical draws (full nl vector) via the loaded Cholesky whitening.
    L = np.linalg.cholesky(side["cov_nl_prior"])
    phys = side["fid_nl"] + chain_w @ L.T
    return phys, acc, wall


def _ess_ips(x):
    x = x - x.mean(); n = x.size
    ac = np.correlate(x, x, "full")[n - 1:] / (np.arange(n, 0, -1) * x.var())
    tau = 1.0
    for lag in range(1, min(n // 2, 5000)):
        if ac[lag] <= 0:
            break
        tau += 2 * ac[lag]
    return n / tau


phys_marg, acc_marg, wall_marg = _run_chain(marg, SEED_MARG)
phys_samp, acc_samp, wall_samp = _run_chain(samp, SEED_SAMP)

# ---------------------------------------------------------------------------
# Gates on the cosmology block (physical positions 0..4).
# ---------------------------------------------------------------------------

_watch["stage"] = "gates"
cm = phys_marg[:, :N_COSMO_NL]
cs = phys_samp[:, :N_COSMO_NL]
mean_m, mean_s = cm.mean(0), cs.mean(0)
sig_m = cm.std(0, ddof=1)
sig_s = cs.std(0, ddof=1)
ess_m = np.array([_ess_ips(cm[:, j]) for j in range(N_COSMO_NL)])
ess_s = np.array([_ess_ips(cs[:, j]) for j in range(N_COSMO_NL)])
se_m = sig_m / np.sqrt(ess_m)
se_s = sig_s / np.sqrt(ess_s)
combined_se = np.sqrt(se_m ** 2 + se_s ** 2)

dmean = mean_m - mean_s
mean_tol = 2.5 * combined_se + 0.02 * sig_F
width_ratio = sig_s / sig_m
corr_m = np.corrcoef(cm, rowvar=False)
corr_s = np.corrcoef(cs, rowvar=False)
iu = np.triu_indices(N_COSMO_NL, 1)
corr_diff_max = float(np.abs(corr_m - corr_s)[iu].max())

g1 = bool(np.all(np.abs(dmean) < mean_tol))
g2 = bool(np.all((width_ratio > 0.95) & (width_ratio < 1.05)))
g3 = bool(corr_diff_max < 0.05)
verdict = "PASS" if (g1 and g2 and g3) else "REVIEW"

# Sampled chain c1 marginals (prior-dominated: mean ~ 0, sigma ~ 1.0).
c1_cols = samp["c1_positions"]
c1_draws = phys_samp[:, c1_cols]                          # (n_kept, n_bins)
c1_mean = c1_draws.mean(0).tolist()
c1_sigma = c1_draws.std(0, ddof=1).tolist()

print(f"\n{'param':>7s} {'dmean/sigF':>11s} {'tol/sigF':>10s} "
      f"{'width s/m':>10s}")
for i, key in enumerate(SHARED_KEYS):
    print(f"{key:>7s} {dmean[i] / sig_F[i]:11.4f} {mean_tol[i] / sig_F[i]:10.4f} "
          f"{width_ratio[i]:10.4f}")
print(f"max corr diff = {corr_diff_max:.4f} (require < 0.05)")
print(f"c1 marginals (sampled): mean={np.array2string(np.asarray(c1_mean), precision=3)}")
print(f"                        sigma={np.array2string(np.asarray(c1_sigma), precision=3)}")
print(f"G1 means {g1}  G2 widths {g2}  G3 corrs {g3}  ->  {verdict}", flush=True)

results["gates"] = {"G1_means": g1, "G2_widths": g2, "G3_corrs": g3}
results["verdict"] = verdict
results["numbers"] = {
    "names": list(SHARED_KEYS),
    "mean_marg": mean_m.tolist(), "mean_samp": mean_s.tolist(),
    "dmean": dmean.tolist(), "mean_tol": mean_tol.tolist(),
    "sig_marg": sig_m.tolist(), "sig_samp": sig_s.tolist(),
    "width_ratio": width_ratio.tolist(),
    "combined_se": combined_se.tolist(),
    "sig_F": sig_F.tolist(),
    "ess_marg": ess_m.tolist(), "ess_samp": ess_s.tolist(),
    "corr_marg": corr_m.tolist(), "corr_samp": corr_s.tolist(),
    "corr_diff_max": corr_diff_max,
    "acceptance": {"marginalized": acc_marg, "sampled": acc_samp},
    "wall_s": {"marginalized": wall_marg, "sampled": wall_samp},
    "n_kept": {"marginalized": int(cm.shape[0]), "sampled": int(cs.shape[0])},
    "c1_marginals_sampled": {"mean": c1_mean, "sigma": c1_sigma},
    "peak_rss_gb": round(_watch["peak_gb"], 2),
    "total_wall_s": round(time.perf_counter() - _T0, 1),
}
save_results()

print(f"\n===== TIER-3 c1 VALIDATION {verdict} (smoke={SMOKE}) =====")
print(f"-> {RESULT_PATH}")
print(f"peak RSS {_watch['peak_gb']:.1f} GB, total "
      f"{time.perf_counter() - _T0:.0f}s", flush=True)
