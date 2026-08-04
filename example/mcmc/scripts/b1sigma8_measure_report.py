"""Option D: measure the b1-measure shift by reweighting the production chain.

Loads cache/desi_chain_w.npy (the raw-measure DESI-prior surrogate chain
persisted by scripts/desi_prior_validation.py), computes the importance weights
w ~ exp(sum_b log sigma8) via ``b1sigma8_log_weights`` (plus the [0,3] bounds on
b1*sigma8), and reports the 5 cosmology means/widths under BOTH measures, the
measured shift in sigma_F units, the Kish ESS/N, and the comparison against the
first-order prediction  F^-1 g = (-0.006, -0.028, +0.172, +0.097, -0.016) sigma_F
for (ombh2, omch2, logA, ns, h).  Writes cache/b1sigma8_measure.json.

This is the production chain-level twin of Task 4's cross-check
(b1sigma8_crosscheck.py), which proved flag-ON == reweighted-raw pointwise to
5.95e-14.  Here the SAME reweighting turns the predicted first-order tilt into a
MEASURED chain-level shift on the full 180000-draw production chain.

Assembly note: only ``sigma8_bins_fn`` (from ``make_lcdm_rescaling_fns``) and the
whitening pieces (fid_nl / cov_nl_prior / split.nl_b1_pos) are needed -- the
minimal subset transplanted from desi_prior_validation.py / b1sigma8_crosscheck.py.
The DESI Hessian widths SIG_F_DESI come from the refreshed gate JSON's
``numbers.sig_F_phys`` (first 5 = cosmology), NOT the legacy whitening sig_fisher.

Run from example/mcmc after scripts/desi_prior_validation.py::

    cd example/mcmc
    python3 scripts/b1sigma8_measure_report.py
"""

import json
import os
import pathlib
import sys

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
    MNU_FIXED, PFS_EMULATOR, SHARED_KEYS,
    build_fiducial_surveys, build_split, n_zbins, z_bins,
)

from jaxptpolypol.desi_priors import b1sigma8_log_weights, make_lcdm_rescaling_fns
from jaxptpolypol.marginal_taylor import reweighted_moments
from jaxptpolypol.model import CosmoEmulator
from jaxptpolypol.params import pack_joint_params

# ---------------------------------------------------------------------------
# Configuration + cache-path resolution (mirrors b1sigma8_crosscheck.py).
# ---------------------------------------------------------------------------
N_COSMO_NL = len(SHARED_KEYS)                                # == 5, cosmo block
WEIGHT_CHUNK = 20_000        # bound the emulator vmap memory over 180k draws

# First-order prediction (Task 3): F^-1 g in sigma_F units, cosmo order.
PRED = np.array([-0.006, -0.028, +0.172, +0.097, -0.016])

HERE = pathlib.Path(__file__).resolve().parents[1]           # example/mcmc
CACHE = HERE / "cache"
if (not (CACHE / "taylor_whitening_lcdm.npz").exists()
        and (CACHE / "cache" / "taylor_whitening_lcdm.npz").exists()):
    CACHE = CACHE / "cache"                                  # worktree nested symlink

WHITENING_PATH = CACHE / "taylor_whitening_lcdm.npz"
CHAIN_PATH = CACHE / "desi_chain_w.npy"
GATE_JSON = CACHE / "desi_prior_validation_sigmap.json"
CROSSCHECK_JSON = CACHE / "b1sigma8_crosscheck.json"
RESULT_PATH = CACHE / "b1sigma8_measure.json"

for p in (WHITENING_PATH, CHAIN_PATH, GATE_JSON):
    if not p.exists():
        sys.exit(f"Required artifact missing: {p} -- run from example/mcmc "
                 "after scripts/desi_prior_validation.py (full, non-smoke).")

# ---------------------------------------------------------------------------
# Whitening pieces + config-drift tripwires (fid_nl / cov_nl_prior / split).
# ---------------------------------------------------------------------------
wz = np.load(WHITENING_PATH)
fid_nl = jnp.asarray(wz["fid_nl"])
cov_nl_prior = np.asarray(wz["cov_nl_prior"])
cosmo_nl_pos = [int(i) for i in wz["cosmo_nl_pos"]]
packed_params = np.asarray(wz["packed_params"])

cosmo_dict, cosmo, surveys, joint_survey_keys = build_fiducial_surveys()
n_cosmo_params = sum(cosmo.param_sizes)

packed_rebuilt = np.asarray(pack_joint_params(cosmo, surveys))
if not np.array_equal(packed_rebuilt, packed_params):
    sys.exit("ABORT: rebuilt packed fiducial vector differs from the stored one "
             "-- config drift vs the whitening npz.")

split = build_split(n_cosmo_params, joint_survey_keys)
if (not np.array_equal(np.asarray(split.nl_idx), wz["nl_idx"])
        or not np.array_equal(np.asarray(split.lin_idx), wz["lin_idx"])):
    sys.exit("ABORT: rebuilt marginal split differs from the stored one.")

# ---------------------------------------------------------------------------
# sigma8_bins_fn (the only theory static the reweighting needs).
# ---------------------------------------------------------------------------
pklin_emulator = CosmoEmulator(probe='custom_log', emulator_path=PFS_EMULATOR)
FIXED_BARYON = {'A_b': cosmo_dict['A_b'], 'eta_b': cosmo_dict['eta_b'],
                'logT_AGN': cosmo_dict['logT_AGN']}
sigma8_bins_fn, _a_ap_bins_fn, _sigma8_ref_bins = make_lcdm_rescaling_fns(
    pklin_emulator=pklin_emulator, cosmo_keys=SHARED_KEYS,
    cosmo_sizes=(1,) * N_COSMO_NL, z_bins=z_bins,
    fid_cosmo_native=fid_nl[:N_COSMO_NL], mnu_fixed=MNU_FIXED,
    fixed_cosmo_extras=FIXED_BARYON)

# ---------------------------------------------------------------------------
# DESI Hessian widths for the cosmology block (from the refreshed gate JSON).
# ---------------------------------------------------------------------------
gate = json.loads(GATE_JSON.read_text())
if gate["numbers"]["names"][:N_COSMO_NL] != list(SHARED_KEYS):
    sys.exit("ABORT: gate JSON cosmology names do not match SHARED_KEYS.")
sig_F = np.asarray(gate["numbers"]["sig_F_phys"][:N_COSMO_NL])

# Task-4 pointwise-identity numbers (for provenance in the report + doc).
try:
    xc = json.loads(CROSSCHECK_JSON.read_text())["pointwise"]
    task4 = {"max_pointwise_resid": xc["max_pointwise_resid"],
             "identity_resid": xc["identity_resid"], "jac_fid": xc["jac_fid"]}
except Exception:
    task4 = None

# ---------------------------------------------------------------------------
# Project the raw whitened chain to physical space; compute the b1sigma8 weights.
# ---------------------------------------------------------------------------
draws_w = np.load(CHAIN_PATH)                                # (n, 26), raw measure
n_draws = draws_w.shape[0]
L = np.linalg.cholesky(cov_nl_prior)
phys = np.asarray(fid_nl) + draws_w @ L.T                    # (n, 26) physical

# Chunk the emulator vmap so the 180k-draw sigma8 evaluation stays bounded.
lw_parts = []
for s in range(0, n_draws, WEIGHT_CHUNK):
    sub = jnp.asarray(phys[s:s + WEIGHT_CHUNK])
    lw_parts.append(np.asarray(b1sigma8_log_weights(
        sub, sigma8_bins_fn, b1_pos=split.nl_b1_pos, lower=0.0, upper=3.0)))
lw = np.concatenate(lw_parts)

n_bound_hits = int(np.sum(~np.isfinite(lw)))
w = np.exp(lw - lw.max())
w /= w.sum()
ess = 1.0 / np.sum(w ** 2)
ess_frac = ess / n_draws
max_weight = float(w.max())
print(f"reweighting: ESS = {ess:.0f} / {n_draws} = {ess_frac:.4f}; "
      f"bound hits {n_bound_hits}; max weight {max_weight:.2e}", flush=True)

# ---------------------------------------------------------------------------
# Both-measure cosmology moments + the measured shift vs the prediction.
# ---------------------------------------------------------------------------
cosmo_phys = phys[:, :N_COSMO_NL]
mean_raw = cosmo_phys.mean(0)
std_raw = cosmo_phys.std(0, ddof=1)
mean_rw, std_rw = reweighted_moments(cosmo_phys, w)
mean_rw = np.asarray(mean_rw)
std_rw = np.asarray(std_rw)

shift = (mean_rw - mean_raw) / sig_F
width_ratio = std_rw / std_raw


def ess_ips(x):
    """Initial-positive-sequence autocorrelation ESS (transplanted from the
    gate script)."""
    x = x - x.mean(); n = x.size
    ac = np.correlate(x, x, "full")[n - 1:] / (np.arange(n, 0, -1) * x.var())
    tau = 1.0
    for lag in range(1, min(n // 2, 5000)):
        if ac[lag] <= 0:
            break
        tau += 2 * ac[lag]
    return n / tau


# MC standard error of the shift, combined-SE form (the reviewed crosscheck
# pattern): raw-mean SE (autocorr ESS) and reweighted-mean SE (autocorr ESS x
# Kish fraction), in sigma_F units, added in quadrature. Conservative -- the two
# estimators share draws and are positively correlated -- so the true shift SE is
# smaller; the 3x-combined-SE tripwire below is therefore not over-eager.
ess_raw = np.array([ess_ips(draws_w[:, cosmo_nl_pos[i]]) for i in range(N_COSMO_NL)])
ess_rw = ess_raw * ess_frac
se_raw = (std_raw / sig_F) / np.sqrt(ess_raw)
se_rw = (std_rw / sig_F) / np.sqrt(ess_rw)
combined_se = np.sqrt(se_raw ** 2 + se_rw ** 2)

resid = shift - PRED
resid_over_3se = np.abs(resid) / (3.0 * combined_se)
match = bool(np.all(resid_over_3se < 1.0))
bounds_ok = (n_bound_hits == 0)
widths_ok = bool(np.all((width_ratio > 0.9) & (width_ratio < 1.1)))
verdict = "MATCH" if (match and bounds_ok) else "CONCERN"

print(f"\n{'param':>7s} {'shift':>9s} {'pred':>8s} {'|s-p|':>8s} "
      f"{'3*SE':>8s} {'|s-p|/3SE':>10s} {'w_rw/w_raw':>11s} {'ess_raw':>9s}")
for i, key in enumerate(SHARED_KEYS):
    print(f"{key:>7s} {shift[i]:9.4f} {PRED[i]:8.4f} {abs(resid[i]):8.4f} "
          f"{3 * combined_se[i]:8.4f} {resid_over_3se[i]:10.4f} "
          f"{width_ratio[i]:11.4f} {ess_raw[i]:9.1f}")
print(f"ESS/N {ess_frac:.4f}  bound hits {n_bound_hits}  widths_ok {widths_ok}  "
      f"->  {verdict}", flush=True)

# ---------------------------------------------------------------------------
# Persist the measured shift + both-measure table + diagnostics.
# ---------------------------------------------------------------------------
results = {
    "config": {
        "n_draws": int(n_draws), "weight_chunk": WEIGHT_CHUNK,
        "sig_F_source": "desi_prior_validation_sigmap.json:numbers.sig_F_phys[:5]",
        "pred": PRED.tolist(), "chain": "desi_chain_w.npy (raw measure)",
        "measure": "b1sigma8 (arXiv:2511.20757 Table I), bounds b1*sigma8 in [0,3]"},
    "task4_identity": task4,
    "reweighting": {
        "ess": float(ess), "n": int(n_draws), "ess_over_n": float(ess_frac),
        "n_bound_hits": n_bound_hits, "max_weight": max_weight},
    "cosmology": {
        "names": list(SHARED_KEYS),
        "mean_raw": mean_raw.tolist(), "std_raw": std_raw.tolist(),
        "mean_rw": mean_rw.tolist(), "std_rw": std_rw.tolist(),
        "sig_F": sig_F.tolist(),
        "shift_sigmaF": shift.tolist(), "pred_sigmaF": PRED.tolist(),
        "shift_minus_pred": resid.tolist(),
        "combined_se_sigmaF": combined_se.tolist(),
        "resid_over_3se": resid_over_3se.tolist(),
        "width_ratio": width_ratio.tolist(),
        "ess_raw": ess_raw.tolist(), "ess_rw": ess_rw.tolist()},
    "gates": {"shift_matches_pred": match, "bounds_ok": bounds_ok,
              "widths_ok": widths_ok},
    "verdict": verdict,
}
RESULT_PATH.write_text(json.dumps(results, indent=1))
print(f"-> {RESULT_PATH}", flush=True)

if verdict != "MATCH":
    print("\n*** CONCERN: measured shift disagrees with the first-order "
          "prediction beyond 3x MC SE (or bound hits > 0). Per the brief HARD "
          "RULE this is a FINDING (posterior non-Gaussianity beyond first "
          "order) -- do NOT tune; record and report.", flush=True)
