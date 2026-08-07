# Joint PFS P+B + BAO + Gaussian-CMB + BBN MCMC Forecasts (LCDM + nuLCDM) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two production MCMC forecast notebooks (LCDM, nuLCDM) sampling the joint posterior PFS P+B (Taylor-surrogate marginal) + DESI DR2 BAO + fiducial-centered Gaussian CMB block + fiducial-centered BBN prior, with tau as a new CMB-constrained sampled dimension, matching inline comparison Fishers, and the profile-likelihood check carried over.

**Architecture:** The PFS marginal posterior (surrogate, 26/27-dim θ_NL) is reused UNTOUCHED; tau is appended LAST to the sampled vector and never reaches the PFS templates (`theta[:n_pfs]` slicing). The CMB enters as `−½ Δs F_cmb Δs` on the shared basis `s = (ombh2, omch2, logA, ns, h, tau[, mnu])`, where `F_cmb_shared` is precomputed by a build script (Hessian of the 5-term candl stack at fiducial, CMB nuisances Schur-marginalized, H0→h projected) and cached as a hash-guarded npz. BBN is a 1-D Gaussian on ombh2 with Mossa width but FIDUCIAL center. New tested helpers live in a new module `src/jaxptpolypol/joint_forecast.py` (cmb.py is user-WIP and off-limits).

**Tech Stack:** Existing jaxptpolypol machinery (marginal_taylor, stream_common, sampler), candl + clipy + cosmopower-jax Cl emulators (build script only — never a notebook runtime dependency), BlackJAX NUTS (LCDM) / RWMH (nuLCDM).

## Supersession notes (added at close-out 2026-08-07)

This plan is committed AS WRITTEN, for provenance. Three places were overtaken by
what actually landed — read the plan against these:

1. **Task 2 Step 3's observed-Hessian `F_cmb` is SUPERSEDED.** The plan specifies the
   Hessian of the 5-term candl stack (see also the Architecture line above). A
   two-branch experiment during Task 2 found the nuLCDM observed Hessian indefinite,
   and the user decided in favour of the **hybrid Gauss–Newton expected Fisher** —
   GN (`JᵀC⁻¹J`) for the Gaussian-bandpower terms, observed Hessian for the
   non-Gaussian low-ℓ terms (CONTEXT.md, "F_cmb method (DECIDED 2026-08-07)"). The
   rejected PSD-clip branch (`expt/cmb-psd-clip`) stays unmerged as the documented
   fallback. The fix wave that landed this is
   `docs/superpowers/plans/2026-08-07-cmb-block-branchB-fixes.md`.
2. **Task 3 Step 2/3's whitened-concatenation code is SUPERSEDED.** Those snippets
   mix whitened and physical coordinates (the joint vector is built by
   concatenating onto a whitened PFS start while the CMB/BBN blocks are physical).
   The implementer's resolution — a **physical joint vector composed with the
   PFS-internal whitening** — was adjudicated in Task 3's review as correct and the
   unique reading that satisfies all the plan's constraints (no double whitening, no
   missing Jacobian; `lp0 = −167.752302` reproduced across four paths). Task 4 Step
   2/3 inherit the same correction.
3. **E14 landed in the WEAKER form the plan's own Step-1 test specified — a
   plan-internal inconsistency.** The E-table row says
   `make_forecast_joint_log_post` "validates `n_pfs` against a probe call at build
   time", but Task 1 Step 1's test only pins `n_pfs=0 → ValueError`, and the
   implementation matches the test (`n_pfs <= 0` check, no probe call). The probe-call
   guarantee does NOT exist; a wrong positive `n_pfs` still surfaces only as a later
   shape error.

## Global Constraints

- Decisions of record (CONTEXT.md, 2026-08-06): CMB = fiducial-centered Gaussian (rejected: real Planck data, mock-injected candl); probe set = PFS P+B + DESI BAO + CMB(Gaussian) + BBN; BBN = ombh2 **0.02242 (fiducial) ± 0.00036** (Mossa width — the Mossa MEAN 0.02233 is FORBIDDEN: it injects a 0.25σ_BBN spurious pull); **no ns10 prior**; F_cmb provisioned as a cached artifact.
- Shared basis order is FIXED: `('ombh2','omch2','logA','ns','h','tau')` LCDM / `('ombh2','omch2','logA','ns','h','tau','mnu')` nuLCDM — identical to `fisher_joint_PFS_BAO_CMB_*` `SHARED_KEYS`. tau fiducial 0.0561. All artifact matrices are stored in this order.
- Sampled vector layout: `theta = concat(theta_NL, [tau])` — tau LAST (index 26 LCDM / 27 nuLCDM). θ_NL layout is unchanged from production (cosmo block first, then 7×(b1,b2,bG2)). Index maps sampled→shared: LCDM `[0,1,2,3,4,26]`; nuLCDM `[0,1,2,3,4,27,5]` (mnu sits at θ_NL position 5, tau after all θ_NL).
- **No tau prior** (post-simall-fix doctrine): tau is constrained solely by the CMB block's curvature. No tau bound (σ(tau)≈0.007 puts 0 at ~8σ; the notebook prints `min(tau samples)` as a sanity line).
- The PFS surrogate, templates, whitening artifacts, spec YAMLs, phase gates, and fiducial-centered marginalized-prior policy (`PRIOR_VARIANT="fiducial_centered"` default, `marginal_means="fiducial"`) carry over UNCHANGED. Taylor templates are NOT rebuilt (tau never enters them).
- Samplers by doctrine: LCDM → NUTS (wall-free; tau adds one smooth dimension); nuLCDM → RWMH 200k×4/20k burn, seed 20260806 (mnu≥0 + b1σ8 U[0,3] walls are −∞ indicators, NUTS-hostile).
- Fiducial: ombh2 0.02242, omch2 0.11933, logA 3.047, ns 0.9665, h 0.6766, tau 0.0561, mnu 0.06 (varied in nuLCDM only). LCDM data vectors remain NOISELESS mocks; every likelihood term peaks at fiducial; `chi2(fid)=0` tripwires are mandatory.
- OFF-LIMITS (must appear in NO diff): `src/jaxptpolypol/cmb.py`, `tests/test_cmb_priors.py` (user WIP — import from cmb.py is allowed, editing is not), the four committed production/reference notebooks (`example/mcmc/mcmc_joint_PFS_BAO_BBN_ns_{LCDM,nuLCDM}.ipynb`, `example/fisher/fisher_joint_PFS_BAO_CMB_{LCDM,nuLCDM}.ipynb`), `example/mcmc/mcmc_cmb_bao_bbn_LCDM.ipynb`, existing cache artifacts.
- Test suite baseline at plan start: `pytest tests/ -q` → **193 passed, 15 deselected**. Additions only; never fewer.
- Notebook edits ONLY via Python scripts using nbformat/json (NotebookEdit has a wrong-cell hazard in this repo); semantic cell-diff vs the pre-edit git state before every commit. WIP-commit before every production run. Long runs: nohup + fresh tagged log under `example/mcmc/cache/`, bounded `while kill -0` waits. Production runs serial, never concurrent.
- Verification-before-completion: every claim in reports backed by verbatim fresh command output.
- x64 everywhere (`jax.config.update("jax_enable_x64", True)`).

## Expected behaviors and edge cases (verification targets — each is asserted or printed somewhere below)

| # | Behavior / edge case | Where enforced |
|---|---|---|
| E1 | simall (low-ℓ EE) tau gradient is LIVE in the installed clipy — else the CMB block has no tau information and the whole design fails | Task 2 build-script hard gate (nonzero grad + σ(tau) range check); it FAILS LOUDLY, no fallback |
| E2 | F_cmb_shared is symmetric positive-definite after nuisance marginalization | Task 2 gate: symmetrize, then `min(eigvals) > 0` else abort |
| E3 | H0↔h factor-100 landmine | Task 2 gate: Jacobian element `J[h_row, H0_col] == 0.01` exactly; unit test on the projection |
| E4 | Wrong-cosmology / stale artifact loaded into a notebook | `load_cmb_fisher_block` META guard (cosmology + shared_keys + theory-config hash, enforce-if-present) + unit tests |
| E5 | tau is constrained ONLY by CMB | Notebook assert: comparison Fisher without the CMB block has an exactly zero tau row/column |
| E6 | σ(tau) sane | Task 2 gate + notebook print: `sqrt(inv(F_cmb_shared)[tau,tau]) ∈ [0.004, 0.02]` |
| E7 | logA–tau degeneracy present (A_s e^(−2τ)) | Notebook sanity print: chain corr(logA, tau) > 0.3 |
| E8 | Information only adds: joint marginal σ ≤ PFS-only production σ for every shared param except tau (which PFS lacks) | Notebook table prints joint σ next to the committed PFS-only σ; soft assert ratio ≤ 1.02 (MC tolerance) |
| E9 | BBN near-redundancy quantified, not assumed | Notebook print: σ(ombh2) from inline Fisher with vs without F_bbn (expect ~few % difference) |
| E10 | Mossa-mean contamination | Task 1 unit test pins the BBN center argument to the fiducial; notebook asserts `BBN_MEAN == cosmo_dict['ombh2']` |
| E11 | Noiseless-forecast exactness incl. new terms | `chi2_prof(fid) < 1e-10` tripwire (CMB and BBN are fiducial-centered ⇒ contribute exactly 0 at fid); profile-likelihood check re-run over ALL shared params incl. tau |
| E12 | tau profile is exactly parabolic (Gaussian term) with min at fid | Profile table row; offset < 0.1 σ_F like the rest |
| E13 | mnu wall still active in nuLCDM; truncation ratio remeasured under tighter CMB-informed σ | nuLCDM notebook mnu-marginal cell (wall-hit %, width ratio vs joint Fisher) |
| E14 | The surrogate never sees tau (dim mismatch would be an immediate shape error — make it a clear one) | `make_forecast_joint_log_post` validates `n_pfs` against a probe call at build time (Task 1 test) |
| E15 | Whitening covariance PD after adding CMB+BBN information | Notebook: Cholesky of the joint whitening cov succeeds (raises otherwise); assert no NaN in `log_post(x0)` |
| E16 | LCDM path bit-identical | No off-limits file in any diff; suite count never drops below 193 |
| E17 | RWMH proposal must include a sane tau scale (else tau mixes catastrophically slowly) | nuLCDM whitening includes tau via the joint Fisher; ESS(tau) reported in the R-hat/ESS table |
| E18 | Artifact regeneration documented | META carries build command; notebook markdown states the regen one-liner |

## File Structure

| File | Change |
|---|---|
| `src/jaxptpolypol/joint_forecast.py` | NEW — `make_gaussian_fisher_loglike`, `make_forecast_joint_log_post`, `embed_fisher` |
| `tests/test_joint_forecast.py` | NEW — unit tests for the three helpers (values, gradients, validation, E10/E14) |
| `example/mcmc/scripts/build_cmb_fisher_block.py` | NEW — `--cosmology {lcdm,nulcdm}` builder porting `fisher_joint_PFS_BAO_CMB_*` cells; writes `cache/cmb_fisher_{lcdm,nulcdm}.npz`; hard gates E1–E3 |
| `example/mcmc/scripts/stream_common.py` | ADDITIVE — `SHARED_KEYS_CMB_{LCDM,NULCDM}`, `TAU_FID`, `BBN_SIGMA_MOSSA`, `cmb_fisher_path(cosmology)`, `load_cmb_fisher_block(cosmology, ...)` with META guard |
| `tests/test_stream_common_meta.py` | ADDITIVE — loader guard tests (E4) with synthetic tmp artifacts |
| `example/mcmc/mcmc_joint_PFS_BAO_CMB_BBN_LCDM.ipynb` | NEW — replicated from `mcmc_joint_PFS_BAO_BBN_ns_LCDM.ipynb`, joint-posterior deltas below, NUTS production |
| `example/mcmc/mcmc_joint_PFS_BAO_CMB_BBN_nuLCDM.ipynb` | NEW — replicated from `mcmc_joint_PFS_BAO_BBN_ns_nuLCDM.ipynb`, RWMH production |
| `docs/design/perbin-compile-measurements.md` | ADDITIVE — "Joint PFS+BAO+CMB+BBN MCMC" section |
| `CONTEXT.md` | ADDITIVE — implementation-status sentence appended to the 2026-08-06 decision paragraphs |

Sequencing: Task 1 → Task 2 → Task 3 (LCDM notebook) → Task 4 (nuLCDM notebook) → Task 5 (docs). Strictly serial (single committer; production runs must not overlap).

---

### Task 1: `joint_forecast` helpers (TDD)

**Files:**
- Create: `src/jaxptpolypol/joint_forecast.py`
- Test: `tests/test_joint_forecast.py`

**Interfaces:**
- Consumes: nothing new (pure JAX).
- Produces (used verbatim by Tasks 3–4):
  - `make_gaussian_fisher_loglike(fisher, center, index_map) -> Callable[[Array], Array]` — returns `loglike(theta) = −½ (theta[index_map]−center)ᵀ F (theta[index_map]−center)`.
  - `make_forecast_joint_log_post(pfs_log_post, *, n_pfs, extra_loglike_fns=()) -> Callable[[Array], Array]` — `log_post(theta) = pfs_log_post(theta[:n_pfs]) + Σ fn(theta)`; each `fn` receives the FULL theta.
  - `embed_fisher(F_sub, index_map, n) -> Array` — n×n zeros with `F_sub` added at `ix_(index_map, index_map)` (whitening/comparison-Fisher assembly).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_joint_forecast.py
import jax, jax.numpy as jnp, numpy as np, pytest
jax.config.update("jax_enable_x64", True)

from jaxptpolypol.joint_forecast import (
    make_gaussian_fisher_loglike, make_forecast_joint_log_post, embed_fisher,
)

F2 = jnp.array([[4.0, 1.0], [1.0, 9.0]])
CENTER = jnp.array([0.5, -1.0])
IDX = [1, 3]          # picks theta[1], theta[3]
THETA = jnp.array([9.0, 0.7, 9.0, -0.6])   # d = [0.2, 0.4]

def test_gaussian_fisher_loglike_value():
    ll = make_gaussian_fisher_loglike(F2, CENTER, IDX)
    d = np.array([0.2, 0.4])
    expected = -0.5 * d @ np.asarray(F2) @ d
    assert np.isclose(float(ll(THETA)), expected, rtol=0, atol=1e-14)

def test_gaussian_fisher_loglike_zero_at_center():
    ll = make_gaussian_fisher_loglike(F2, CENTER, IDX)
    theta = jnp.zeros(4).at[jnp.array(IDX)].set(CENTER)
    assert float(ll(theta)) == 0.0

def test_gaussian_fisher_loglike_gradient_hits_only_mapped_indices():
    ll = make_gaussian_fisher_loglike(F2, CENTER, IDX)
    g = np.asarray(jax.grad(ll)(THETA))
    assert g[0] == 0.0 and g[2] == 0.0
    assert g[1] != 0.0 and g[3] != 0.0

def test_gaussian_fisher_loglike_shape_validation():
    with pytest.raises(ValueError):
        make_gaussian_fisher_loglike(F2, jnp.zeros(3), IDX)      # center wrong
    with pytest.raises(ValueError):
        make_gaussian_fisher_loglike(F2, CENTER, [0, 1, 2])      # idx wrong

def test_bbn_center_is_fiducial_not_mossa():
    # E10: the 1-D BBN block used by the notebooks must center on the FIDUCIAL.
    ll = make_gaussian_fisher_loglike(
        jnp.array([[1.0 / 0.00036**2]]), jnp.array([0.02242]), [0])
    assert float(ll(jnp.array([0.02242]))) == 0.0
    pull = -2.0 * float(ll(jnp.array([0.02233])))       # Mossa mean, in chi2
    assert np.isclose(np.sqrt(pull), 0.25, atol=0.01)   # the documented 0.25 sigma

def test_joint_log_post_composition_and_slicing():
    pfs = lambda th: -jnp.sum(th**2)                     # sees ONLY theta[:3]
    extra = make_gaussian_fisher_loglike(F2, CENTER, IDX)
    lp = make_forecast_joint_log_post(pfs, n_pfs=3, extra_loglike_fns=(extra,))
    expected = -float(jnp.sum(THETA[:3]**2)) + float(extra(THETA))
    assert np.isclose(float(lp(THETA)), expected, atol=1e-14)
    g = np.asarray(jax.grad(lp)(THETA))                  # tau-analog theta[3]:
    assert g[3] != 0.0                                   # reached ONLY via extra

def test_joint_log_post_n_pfs_validation():
    with pytest.raises(ValueError):
        make_forecast_joint_log_post(lambda th: th.sum(), n_pfs=0)

def test_embed_fisher():
    F = np.asarray(embed_fisher(F2, IDX, 5))
    expected = np.zeros((5, 5))
    expected[np.ix_(IDX, IDX)] = np.asarray(F2)
    assert np.array_equal(F, expected)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_joint_forecast.py -q`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'jaxptpolypol.joint_forecast'`

- [ ] **Step 3: Write the implementation**

```python
# src/jaxptpolypol/joint_forecast.py
"""Joint forecast log-posterior helpers.

Compose the PFS Taylor-surrogate marginal posterior with fiducial-centered
Gaussian external blocks (CMB Fisher block, BBN prior) on an extended sampled
vector theta = concat(theta_NL, [tau]).  The PFS posterior sees only
theta[:n_pfs]; external blocks address the full vector through index maps.
"""
from __future__ import annotations

from typing import Callable, Sequence

import jax.numpy as jnp

__all__ = [
    "make_gaussian_fisher_loglike",
    "make_forecast_joint_log_post",
    "embed_fisher",
]


def make_gaussian_fisher_loglike(fisher, center, index_map) -> Callable:
    """loglike(theta) = -1/2 (theta[index_map] - center)^T F (theta[index_map] - center).

    Fiducial-centered by construction: contributes exactly 0 at theta[index_map]==center,
    preserving the noiseless-forecast chi2(fid)=0 tripwire.
    """
    F = jnp.asarray(fisher, dtype=jnp.float64)
    c = jnp.asarray(center, dtype=jnp.float64)
    idx = jnp.asarray(index_map, dtype=int)
    k = idx.shape[0]
    if F.shape != (k, k):
        raise ValueError(f"fisher shape {F.shape} does not match index_map length {k}")
    if c.shape != (k,):
        raise ValueError(f"center shape {c.shape} does not match index_map length {k}")

    def loglike(theta):
        d = jnp.asarray(theta)[idx] - c
        return -0.5 * d @ F @ d

    return loglike


def make_forecast_joint_log_post(pfs_log_post: Callable, *, n_pfs: int,
                                 extra_loglike_fns: Sequence[Callable] = ()) -> Callable:
    """log_post(theta) = pfs_log_post(theta[:n_pfs]) + sum(fn(theta) for fn in extras)."""
    if n_pfs <= 0:
        raise ValueError(f"n_pfs must be positive, got {n_pfs}")
    fns = tuple(extra_loglike_fns)

    def log_post(theta):
        theta = jnp.asarray(theta)
        total = pfs_log_post(theta[:n_pfs])
        for fn in fns:
            total = total + fn(theta)
        return total

    return log_post


def embed_fisher(F_sub, index_map, n: int):
    """n x n zeros with F_sub added at ix_(index_map, index_map)."""
    idx = jnp.asarray(index_map, dtype=int)
    return jnp.zeros((n, n)).at[jnp.ix_(idx, idx)].add(jnp.asarray(F_sub, dtype=jnp.float64))
```

- [ ] **Step 4: Run tests to verify they pass, then the full suite**

Run: `pytest tests/test_joint_forecast.py -q` → Expected: `8 passed`
Run: `pytest tests/ -q` → Expected: `201 passed, 15 deselected` (193 + 8; `test_cmb_priors.py` untouched)

- [ ] **Step 5: Commit**

```bash
git add src/jaxptpolypol/joint_forecast.py tests/test_joint_forecast.py
git commit -m "feat(joint): Gaussian-Fisher block + joint log-post composition helpers (tau-last layout)"
```

---

### Task 2: CMB Fisher block build script + loader

**Files:**
- Create: `example/mcmc/scripts/build_cmb_fisher_block.py`
- Modify: `example/mcmc/scripts/stream_common.py` (additive block at the end, after the NULCDM block)
- Test: `tests/test_stream_common_meta.py` (additive)

**Interfaces:**
- Consumes: `jaxptpolypol.cmb` public API (import-only — the file is off-limits to EDIT): `load_candl_likelihood`, `make_candl_loglike_fn`, `make_candl_pars_to_theory_specs_fn`, `get_candl_default_parameters`, `make_joint_loglike_fn`, `CandlParameterLayout`; `jaxptpolypol.inference`: `marginalized_fisher_block`, `project_fisher_to_derived`. All constants (likelihood term paths, `CMB_EMULATOR_FILENAMES`, `COSMO_KEYS_CMB`, nuisance-union construction) are **copied verbatim from `example/fisher/fisher_joint_PFS_BAO_CMB_{LCDM,nuLCDM}.ipynb` cells 3, 13, 14, 16** — do not retype paths from memory; extract them programmatically (`json.load` the notebook, locate the cells) or copy-paste exactly.
- Produces:
  - Artifact `example/mcmc/cache/cmb_fisher_{lcdm,nulcdm}.npz` with keys: `F_cmb_shared` (k×k, shared-basis order), `fid_shared` (k,), `shared_keys` (str array), `F_cmb_native`, `fid_native`, `native_keys`, `sigma_tau` (scalar), `meta_json` (single JSON string: `{"cosmology", "shared_keys", "terms", "emulator_files", "theory_config_hash", "build_command", "gates": {...}}`).
  - `stream_common.load_cmb_fisher_block(cosmology, cache_dir=None) -> dict` — loads, META-guards, returns `{"F_shared", "fid_shared", "shared_keys", "sigma_tau", "meta"}`.
  - `stream_common.SHARED_KEYS_CMB_LCDM = ('ombh2','omch2','logA','ns','h','tau')`, `SHARED_KEYS_CMB_NULCDM = (..., 'mnu')`, `TAU_FID = 0.0561`, `BBN_SIGMA_MOSSA = 0.00036`, `cmb_fisher_path(cosmology)`.

- [ ] **Step 1: Write the failing loader-guard tests (E4)**

Append to `tests/test_stream_common_meta.py` (keep existing content untouched):

```python
# --- CMB Fisher block loader guards (joint PFS+CMB forecasts) ---
import json as _json
import numpy as _np
import pytest as _pytest

def _write_cmb_artifact(path, *, cosmology="lcdm", shared_keys=None, hash_val=None):
    import example.mcmc.scripts.stream_common as sc  # adjust to the repo's import pattern
    keys = shared_keys or list(sc.SHARED_KEYS_CMB_LCDM)
    k = len(keys)
    meta = {"cosmology": cosmology, "shared_keys": keys,
            "theory_config_hash": hash_val or sc.THEORY_CONFIG_HASH}
    _np.savez(path, F_cmb_shared=_np.eye(k), fid_shared=_np.zeros(k),
              shared_keys=_np.array(keys), F_cmb_native=_np.eye(k),
              fid_native=_np.zeros(k), native_keys=_np.array(keys),
              sigma_tau=_np.float64(0.007), meta_json=_json.dumps(meta))

def test_cmb_loader_roundtrip(tmp_path):
    import example.mcmc.scripts.stream_common as sc
    _write_cmb_artifact(tmp_path / "cmb_fisher_lcdm.npz")
    out = sc.load_cmb_fisher_block("lcdm", cache_dir=tmp_path)
    assert out["shared_keys"] == tuple(sc.SHARED_KEYS_CMB_LCDM)
    assert out["F_shared"].shape == (6, 6)

def test_cmb_loader_rejects_wrong_cosmology(tmp_path):
    import example.mcmc.scripts.stream_common as sc
    _write_cmb_artifact(tmp_path / "cmb_fisher_nulcdm.npz", cosmology="lcdm")
    with _pytest.raises(ValueError, match="cosmology"):
        sc.load_cmb_fisher_block("nulcdm", cache_dir=tmp_path)

def test_cmb_loader_rejects_wrong_hash(tmp_path):
    import example.mcmc.scripts.stream_common as sc
    _write_cmb_artifact(tmp_path / "cmb_fisher_lcdm.npz", hash_val="deadbeef")
    with _pytest.raises(ValueError, match="hash"):
        sc.load_cmb_fisher_block("lcdm", cache_dir=tmp_path)

def test_cmb_loader_rejects_wrong_shared_keys(tmp_path):
    import example.mcmc.scripts.stream_common as sc
    bad = ["ombh2", "omch2", "logA", "ns", "tau", "h"]   # swapped order
    _write_cmb_artifact(tmp_path / "cmb_fisher_lcdm.npz", shared_keys=bad)
    with _pytest.raises(ValueError, match="shared_keys"):
        sc.load_cmb_fisher_block("lcdm", cache_dir=tmp_path)
```

NOTE for the implementer: match the import idiom the EXISTING tests in `tests/test_stream_common_meta.py` use to reach `stream_common` (sys.path manipulation or package import) — copy it, do not invent a new one. Expected hash: LCDM artifacts guard against `THEORY_CONFIG_HASH`, nuLCDM against `NULCDM_THEORY_CONFIG_HASH` (both already in stream_common).

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_stream_common_meta.py -q`
Expected: new tests FAIL with `AttributeError: ... 'SHARED_KEYS_CMB_LCDM'` (existing tests still pass).

- [ ] **Step 3: Add the stream_common additive block**

```python
# --- CMB Fisher block (joint PFS+BAO+CMB+BBN forecasts, 2026-08-06 decisions) ---
SHARED_KEYS_CMB_LCDM = ('ombh2', 'omch2', 'logA', 'ns', 'h', 'tau')
SHARED_KEYS_CMB_NULCDM = ('ombh2', 'omch2', 'logA', 'ns', 'h', 'tau', 'mnu')
TAU_FID = 0.0561
BBN_SIGMA_MOSSA = 0.00036   # Mossa et al. 2020 WIDTH; center is ALWAYS the fiducial ombh2

def cmb_fisher_path(cosmology, cache_dir=None):
    base = Path(cache_dir) if cache_dir is not None else CACHE_DIR   # reuse the existing cache-dir constant
    return base / f"cmb_fisher_{cosmology}.npz"

def load_cmb_fisher_block(cosmology, cache_dir=None):
    """Load the precomputed fiducial-centered CMB Fisher block with META guards."""
    if cosmology not in ("lcdm", "nulcdm"):
        raise ValueError(f"unknown cosmology {cosmology!r}")
    path = cmb_fisher_path(cosmology, cache_dir)
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"]))
        expected_keys = SHARED_KEYS_CMB_LCDM if cosmology == "lcdm" else SHARED_KEYS_CMB_NULCDM
        if meta.get("cosmology") != cosmology:
            raise ValueError(f"artifact cosmology {meta.get('cosmology')!r} != requested {cosmology!r}")
        if tuple(meta.get("shared_keys", ())) != expected_keys:
            raise ValueError(f"artifact shared_keys {meta.get('shared_keys')} != expected {expected_keys}")
        expected_hash = THEORY_CONFIG_HASH if cosmology == "lcdm" else NULCDM_THEORY_CONFIG_HASH
        got = meta.get("theory_config_hash")
        if got is not None and got != expected_hash:
            raise ValueError(f"artifact theory_config_hash {got} != expected {expected_hash}")
        return {"F_shared": jnp.asarray(z["F_cmb_shared"]),
                "fid_shared": jnp.asarray(z["fid_shared"]),
                "shared_keys": expected_keys,
                "sigma_tau": float(z["sigma_tau"]),
                "meta": meta}
```

(Adjust `CACHE_DIR`/`Path`/`json`/`np`/`jnp` to the names stream_common already imports — inspect the file first; it already has a cache-dir convention used by `template_meta_for`/`load_templates_and_whitening`.)

- [ ] **Step 4: Run loader tests to verify they pass**

Run: `pytest tests/test_stream_common_meta.py -q` → Expected: all pass (existing + 4 new).

- [ ] **Step 5: Write the build script**

`example/mcmc/scripts/build_cmb_fisher_block.py` — structure (constants copied verbatim from the fisher_joint notebooks; the script must refuse to run if the source cells can't be located):

```python
#!/usr/bin/env python3
"""Build the fiducial-centered Gaussian CMB Fisher block artifact.

Ports cells 3/13/14/16 of example/fisher/fisher_joint_PFS_BAO_CMB_{LCDM,nuLCDM}.ipynb:
  1. load the 5 candl/clipy likelihood terms (Planck highl TTTEEE, lowl TT,
     lowl EE simall, Planck lensing, ACT DR6 lensing) with internal priors ON;
  2. joint loglike over (COSMO_KEYS_CMB + sampled CMB nuisances) at the fiducial;
  3. F_full = -0.5*(H + H^T) with H = jax.hessian(joint_loglike)(theta_fid);
  4. Schur-marginalize nuisances -> cosmo-native block;
  5. project H0->h into the shared basis (ombh2, omch2, logA, ns, h, tau[, mnu]);
  6. HARD GATES (abort loudly, no fallback):
       G1 (E1): |grad of the lowl_EE term wrt tau| > 0 at fiducial, AND
                sigma_tau = sqrt(inv(F_shared)[tau,tau]) in [0.004, 0.02];
       G2 (E2): min(eigvals(F_shared)) > 0;
       G3 (E3): d(shared)/d(native) Jacobian has J[h_row, H0_col] == 0.01.
  7. save npz + META (incl. gate results and the exact build command).

Usage: python3 build_cmb_fisher_block.py --cosmology {lcdm,nulcdm} [--dry-run]
--dry-run: run steps 1-2 shapes-only (no Hessian), print the layout, exit 0.
"""
```

Implementation requirements (each is a review checkpoint, not prose):
- `--cosmology lcdm` uses the LCDM notebook's constants + `jense_2023_camb_lcdm_Cl_*` emulators; `nulcdm` the nuLCDM notebook's (7-key native basis incl. mnu, `jense_2023_camb_mnu_Cl_*`).
- Fiducial values imported from `stream_common` (`FIDUCIAL`-equivalents already there) + `TAU_FID` — NOT retyped.
- G1 implementation: build the `planck_lowl_ee` term's standalone loglike (`make_candl_loglike_fn` on that term alone), take `jax.grad` at the fiducial, assert the tau component is nonzero (`abs > 1e-3`; a dead clipy spline gives exactly 0.0 — that is the E1 failure mode, message must say "installed clipy simall has no tau gradient — the 2026-07-14 cubic-spline fix is missing from this environment").
- The Hessian is O((7+~20)²) candl evaluations under jit — minutes-scale; print wall time.
- META `theory_config_hash`: `THEORY_CONFIG_HASH` (lcdm) / `NULCDM_THEORY_CONFIG_HASH` (nulcdm) from stream_common.
- Output paths via `cmb_fisher_path(...)` — never a hand-built string (output-path lesson).

- [ ] **Step 6: Dry-run gate**

Run: `python3 example/mcmc/scripts/build_cmb_fisher_block.py --cosmology lcdm --dry-run`
Expected: prints term list, native/shared key layout, nuisance union, exits 0. Same for `--cosmology nulcdm`.

- [ ] **Step 7: Real builds (serial)**

Run: `python3 ... --cosmology lcdm` then `--cosmology nulcdm`.
Expected: both artifacts written; gates G1–G3 printed as PASS with numbers (quote `sigma_tau` for both — expect ≈0.007); wall time printed.

- [ ] **Step 8: Full suite + commit**

Run: `pytest tests/ -q` → Expected: `205 passed, 15 deselected` (201 + 4).

```bash
git add example/mcmc/scripts/build_cmb_fisher_block.py example/mcmc/scripts/stream_common.py tests/test_stream_common_meta.py
git commit -m "feat(cmb): fiducial-centered CMB Fisher block builder + hash-guarded loader (lcdm+nulcdm artifacts)"
```

(Cache npz files stay untracked, consistent with the Taylor-template convention; regeneration is the one-liner in META.)

---

### Task 3: LCDM joint notebook + NUTS production run

**Files:**
- Create: `example/mcmc/mcmc_joint_PFS_BAO_CMB_BBN_LCDM.ipynb` (replicate from `example/mcmc/mcmc_joint_PFS_BAO_BBN_ns_LCDM.ipynb` at HEAD — a `cp` then scripted edits; the source notebook is NOT modified)

**Interfaces:**
- Consumes: `joint_forecast.{make_gaussian_fisher_loglike, make_forecast_joint_log_post, embed_fisher}` (Task 1), `stream_common.{load_cmb_fisher_block, SHARED_KEYS_CMB_LCDM, TAU_FID, BBN_SIGMA_MOSSA}` (Task 2), plus everything the source notebook already uses.
- Produces: the committed executed notebook; fresh tripwire values (lp0, chi2_prof(fid)) recorded in the config-cell comments for Task 5's docs.

Cell-level delta specification (edit via an nbformat script; cell numbers refer to the SOURCE notebook's layout — locate each cell by content match, never by hardcoded index):

- [ ] **Step 1: Copy + title/config edits**

`cp example/mcmc/mcmc_joint_PFS_BAO_BBN_ns_LCDM.ipynb example/mcmc/mcmc_joint_PFS_BAO_CMB_BBN_LCDM.ipynb`, then scripted edits:
- Title markdown: new name + one paragraph: probe set, CMB-as-Gaussian decision pointer to CONTEXT.md, tau doctrine (no tau prior), artifact regen one-liner (E18).
- Config cell: `COSMO_PRIORS = {}` **replacing** `{'ombh2': 0.00055, 'ns': 0.042}` — the BBN(0.00055)+ns10 sampled-cosmo priors are REMOVED (cosmology information now enters as CMB+BBN likelihood terms; keeping them would double-count ombh2/ns information). Add:

```python
TAU_FID = 0.0561
CMB_BLOCK = load_cmb_fisher_block("lcdm")            # E4-guarded
SIGMA_BBN = BBN_SIGMA_MOSSA                          # 0.00036 — WIDTH only
BBN_MEAN = cosmo_dict['ombh2']                       # FIDUCIAL center (E10)
assert BBN_MEAN == 0.02242
SHARED_IDX_MAP = [0, 1, 2, 3, 4, N_NL]               # theta -> shared basis; N_NL=26, tau appended last
```

- [ ] **Step 2: Joint posterior assembly cell (after the existing surrogate cell)**

```python
n_pfs = N_NL                                          # 26: the surrogate's full input
theta0 = jnp.concatenate([x0, jnp.array([TAU_FID])])  # x0 = existing PFS fiducial start

cmb_loglike = make_gaussian_fisher_loglike(
    CMB_BLOCK["F_shared"], CMB_BLOCK["fid_shared"], SHARED_IDX_MAP)
bbn_loglike = make_gaussian_fisher_loglike(
    jnp.array([[1.0 / SIGMA_BBN**2]]), jnp.array([BBN_MEAN]), [0])

log_post_joint = make_forecast_joint_log_post(
    log_post_surr, n_pfs=n_pfs, extra_loglike_fns=(cmb_loglike, bbn_loglike))

lp0 = float(log_post_joint(theta0))
# CMB and BBN are fiducial-centered => both contribute EXACTLY 0 at theta0:
assert np.isclose(lp0, float(log_post_surr(x0)), atol=1e-10)   # E11 seed
print(f"log_post_joint(theta0) = {lp0:.6f}")                    # record as the fresh tripwire
```

- [ ] **Step 3: Whitening extension cell (replaces the source `make_cholesky_transform` call)**

```python
# Joint sampled-block Fisher for whitening: PFS block + CMB + BBN information (E15, E17)
F_white = embed_fisher(jnp.linalg.inv(cov_nl_prior), list(range(N_NL)), N_NL + 1)
F_white = F_white + embed_fisher(CMB_BLOCK["F_shared"], SHARED_IDX_MAP, N_NL + 1)
F_white = F_white + embed_fisher(jnp.array([[1.0 / SIGMA_BBN**2]]), [0], N_NL + 1)
cov_joint = jnp.linalg.inv(F_white)
to_whitened, to_physical = make_cholesky_transform(center=theta0, cov=cov_joint)  # raises if not PD (E15)
```

- [ ] **Step 4: Sampler cells** — identical NUTS config to the source production branch (4 chains × 5000, same seed variable), operating on `log_post_joint`/dim 27. SMOKE branch kept.

- [ ] **Step 5: Comparison-Fisher + results cells**

```python
# Inline matching Fisher (this probe set has no committed fisher_joint counterpart):
N_SH = 6
F_pfs_part = embed_fisher(F_pfs_bao_prior_cosmo, [0, 1, 2, 3, 4], N_SH)  # NOTE: rebuild
#   F_pfs_bao_prior_cosmo WITHOUT the removed BBN/ns10 cosmo priors — it must be the
#   PFS P+B + BAO + DESI-EFT-prior marginal-cosmo Fisher ONLY (rename to F_pfs_bao_cosmo).
F_cmp_nocmb = F_pfs_part + embed_fisher(jnp.array([[1.0/SIGMA_BBN**2]]), [0], N_SH)
assert np.allclose(np.asarray(F_cmp_nocmb)[5, :], 0.0)        # E5: tau row zero w/o CMB
F_cmp = F_cmp_nocmb + CMB_BLOCK["F_shared"]
sigma_tau = float(np.sqrt(np.linalg.inv(np.asarray(F_cmp))[5, 5]))
assert 0.004 < sigma_tau < 0.02                                # E6
# E9: BBN redundancy quantification
s_with = np.sqrt(np.linalg.inv(np.asarray(F_cmp))[0, 0])
s_without = np.sqrt(np.linalg.inv(np.asarray(F_cmp - embed_fisher(jnp.array([[1.0/SIGMA_BBN**2]]), [0], N_SH)))[0, 0])
print(f"BBN effect on sigma(ombh2): {100*(s_without/s_with - 1):.1f}% (expect few %)")
```

Results table over `('ombh2','omch2','logA','ns','h','tau')`: fid | MCMC mean | Fisher σ (from `inv(F_cmp)`) | MCMC σ | ratio, plus residual-pulls line. Sanity prints: `corr(logA, tau)` from the chain with `assert > 0.3` soft-check (E7); `min(tau)` (tau>0 sanity); E8 line comparing joint σ against the committed PFS-only production σ values `(0.00047985, 0.0032185, 0.060783, 0.027632, 0.0035686)` with soft assert `joint ≤ 1.02 × PFS-only` for the 5 shared params.

- [ ] **Step 6: Corner + profile cells**

- Corner: 6 params incl. tau; fiducial crosshairs incl. `TAU_FID`; Fisher ellipses from `F_cmp` centered at fiducial.
- Profile-likelihood check (port of the committed cell pair): objective gains `+ (-2*cmb_loglike(theta)) + (-2*bbn_loglike(theta))` — both fiducial-centered so `chi2_prof(fid) < 1e-10` still holds (E11); the scan loops over all 6 shared params (σ_F from `F_cmp`); the ±6σ_F free-dim validity box carries over and now includes tau; expect the tau profile exactly parabolic with offset <0.1 σ_F (E12).

- [ ] **Step 7: SMOKE gate → WIP commit → production run → verification**

- Semantic cell-diff vs the copied source (exactly the intended cells changed) — repo lesson, mandatory.
- SMOKE=True execution: 0 errors; assertions E5/E6/E10/E11 pass; record smoke lp0. WIP-commit (explicit path).
- SMOKE=False production (nohup + tagged log, kill-0 waits): NUTS acceptance ~0.8–0.95, 0 divergences, R-hat ≤ 1.01 all 6 params, ESS table incl. tau (E17), all asserts green, corner rendered.
- `pytest tests/ -q` → `205 passed, 15 deselected`.

- [ ] **Step 8: Commit**

```bash
git add example/mcmc/mcmc_joint_PFS_BAO_CMB_BBN_LCDM.ipynb
git commit -m "feat(joint): LCDM PFS+BAO+CMB+BBN MCMC forecast — NUTS production, tau sampled via CMB block"
```

---

### Task 4: nuLCDM joint notebook + RWMH production run

**Files:**
- Create: `example/mcmc/mcmc_joint_PFS_BAO_CMB_BBN_nuLCDM.ipynb` (replicate from `example/mcmc/mcmc_joint_PFS_BAO_BBN_ns_nuLCDM.ipynb` at HEAD; source NOT modified)

**Interfaces:** Consumes Task 1 helpers + `load_cmb_fisher_block("nulcdm")` + `SHARED_KEYS_CMB_NULCDM`. Everything else parallels Task 3 with these differences — repeat of the full delta spec with nuLCDM values (the implementer sees only this task):

- [ ] **Step 1: Copy + config edits** — as Task 3 Step 1 but: `CMB_BLOCK = load_cmb_fisher_block("nulcdm")` (7×7, `('ombh2','omch2','logA','ns','h','tau','mnu')`); `N_NL = 27`; **`SHARED_IDX_MAP = [0, 1, 2, 3, 4, 27, 5]`** (tau appended at 27; mnu is θ_NL position 5); `COSMO_PRIORS = {}` replacing the source's BBN/ns10 entries; `BBN_MEAN = cosmo_dict['ombh2']; assert BBN_MEAN == 0.02242`; b1σ8 spec + `phase="nulcdm"` + mnu≥0 indicator + `PRIOR_VARIANT="fiducial_centered"` ALL unchanged from the source.

- [ ] **Step 2: Joint posterior cell** — identical to Task 3 Step 2 with `n_pfs = 27`, `theta0 = concat(x0, [TAU_FID])` (dim 28). The mnu≥0 wall stays inside `log_post_surr`'s prior term — untouched.

```python
cmb_loglike = make_gaussian_fisher_loglike(
    CMB_BLOCK["F_shared"], CMB_BLOCK["fid_shared"], SHARED_IDX_MAP)
bbn_loglike = make_gaussian_fisher_loglike(
    jnp.array([[1.0 / SIGMA_BBN**2]]), jnp.array([BBN_MEAN]), [0])
log_post_joint = make_forecast_joint_log_post(
    log_post_surr, n_pfs=27, extra_loglike_fns=(cmb_loglike, bbn_loglike))
assert np.isclose(float(log_post_joint(theta0)), float(log_post_surr(x0)), atol=1e-10)
```

- [ ] **Step 3: Whitening** — as Task 3 Step 3 with `N_NL=27`, `SHARED_IDX_MAP=[0,1,2,3,4,27,5]` (E15/E17: tau proposal scale now comes from the CMB block; mnu scale tightens from CMB lensing information).

- [ ] **Step 4: Sampler** — RWMH 200k×4, 20k burn, seed 20260806, dim 28 (walls forbid NUTS — source doctrine unchanged).

- [ ] **Step 5: Comparison Fisher + results** — as Task 3 Step 5 with `N_SH = 7`, PFS embed map `[0,1,2,3,4,6]` (skipping tau at 5 — mnu goes to shared position 6; this mirrors the fisher_joint `pfs_idx` convention), E5 assert on row 5 (tau), E8 comparison against the committed nuLCDM production σ values `(0.00048188, 0.0032276, 0.078534, 0.033284, 0.0036311, 0.095725)` for `(ombh2, omch2, logA, ns, h, mnu)`. Expected physics (report, don't force): σ(mnu) tightens substantially vs 0.0957 (CMB lensing + primary), so remeasure the wall-truncation ratio and wall-hit % against the JOINT Fisher σ (E13) and present the mnu marginal with the same truncated-shape framing as the source notebook.

- [ ] **Step 6: Corner + profile** — 7 params incl. tau; crosshairs incl. TAU_FID and mnu 0.06; profile check over all 7 shared params — the mnu grid stays clipped to `mnu ≥ 0` and the optimizer keeps the mnu≥0 + ±6σ_F validity boxes; tau parabolic (E12).

- [ ] **Step 7: SMOKE → WIP commit → production (serial after Task 3's run; nohup + tagged log) → verification** — RWMH acceptance ~0.2–0.35, R-hat ≤ 1.01 all 7, ESS incl. tau, `min(tau)` print, all asserts green; `pytest tests/ -q` → `205 passed, 15 deselected`.

- [ ] **Step 8: Commit**

```bash
git add example/mcmc/mcmc_joint_PFS_BAO_CMB_BBN_nuLCDM.ipynb
git commit -m "feat(joint): nuLCDM PFS+BAO+CMB+BBN MCMC forecast — RWMH production, tau + CMB-tightened mnu"
```

---

### Task 5: Docs close-out

**Files:**
- Modify: `docs/design/perbin-compile-measurements.md` (append section), `CONTEXT.md` (append status sentences to the two 2026-08-06 decision paragraphs)

- [ ] **Step 1:** Measurement-doc section "Joint PFS+BAO+CMB+BBN MCMC forecasts (2026-08-06)": both notebooks' lp0 tripwires, sampler health (acceptance/R-hat/ESS incl. tau), both Fisher-vs-MCMC tables, σ(tau), corr(logA,tau), the E8 joint-vs-PFS-only width comparison, E9 BBN-redundancy numbers, nuLCDM mnu truncation remeasurement (E13), profile-check tables (E11/E12), artifact provenance (META gates G1–G3 values, build wall times).
- [ ] **Step 2:** CONTEXT.md: one sentence per decision paragraph marking implementation landed, with notebook + artifact + script paths.
- [ ] **Step 3:** `pytest tests/ -q` (expect `205 passed, 15 deselected`), then:

```bash
git add docs/design/perbin-compile-measurements.md CONTEXT.md
git commit -m "docs(joint): PFS+BAO+CMB+BBN MCMC forecast results + decision close-out"
```

---

## Self-review (performed at write time)

1. **Spec coverage:** CMB-as-Gaussian (T2 builds, T3/4 consume) ✓; BBN fiducial-centered Mossa-width (T1 E10 test, T3/4 config asserts) ✓; no ns10 + old cosmo-prior removal with double-count rationale (T3/4 Step 1) ✓; inline comparison Fisher (T3/4 Step 5) ✓; artifact + guards (T2, E4) ✓; tau doctrine (E1/E5/E6, no prior, no bound) ✓; samplers per doctrine ✓; profile-check carry-over incl. tau (E11/E12) ✓; fiducial crosshairs incl. tau ✓; cmb.py untouched (new module) ✓; docs ✓.
2. **Placeholder scan:** the only "copy from source" instructions name exact cells of committed notebooks (constants that must not be retyped from memory — a correctness measure, not a placeholder); all code steps carry code.
3. **Type consistency:** `make_gaussian_fisher_loglike(fisher, center, index_map)`, `make_forecast_joint_log_post(pfs_log_post, *, n_pfs, extra_loglike_fns)`, `embed_fisher(F_sub, index_map, n)`, `load_cmb_fisher_block(cosmology, cache_dir=None) -> {"F_shared", "fid_shared", "shared_keys", "sigma_tau", "meta"}` used identically in T1/T2/T3/T4 ✓; SHARED_IDX_MAP values consistent with the tau-last layout stated in Global Constraints ✓; suite counts 193→201 (T1) →205 (T2) stable through T3–T5 ✓.
