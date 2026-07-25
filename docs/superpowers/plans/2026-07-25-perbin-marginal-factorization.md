# Phase 1 — Exact per-bin factorization of the marginal likelihood

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Evaluate the *same joint* marginal log-posterior as a sum of per-bin terms, with **bit-exact** results, to reduce the compile/runtime cost of the 7-bin P+B posterior.

> **Measured outcome (2026-07-25, supersedes the original hypothesis).** The premise below — "cutting op count ≈/7" — is **falsified**: on the 2-bin reference config the op counts of monolith / per-bin / scan are 19 624 / 20 096 / 20 090, i.e. equal. The monolith already performs `n_bins` of theory work, and only a genuine single-compiled-body would cut ops (blocked by static `bin_index`; see Task 3). What per-bin factorization *does* buy, measured: **peak compile RSS 6.3 → 4.7 GB (−25%)** and **2× faster per-evaluation (2.04 → 0.99 s)**, exactly. `docs/design/perbin-compile-measurements.md` has the full table.
>
> **Follow-up lead (not in this plan's scope):** the lowered StableHLO text is ~142 MB because the **emulator weights are inlined as constants**, and `theory.py:401` calls `pklin_emulator.predict` *inside* the per-bin loop — so they are inlined once per bin. That plausibly dominates compile time/RSS and would explain why ~40× grid coarsening barely moved the footprint. Candidate fix: one **batched** emulator call over all bin redshifts (only `z` differs per bin). Investigate before any further graph-size work.

**Architecture:** The joint likelihood already factorizes exactly: the data covariance is `block_diag` across z-bins (`covariance.py:766`), P/B are block-diagonal within a bin (`pb_cov=None`, `covariance.py:127-128`), BAO is an independent cosmology-only block, and each bin owns its own 11 θ_lin (no lin parameter crosses bins). Therefore

  `ln L_marg(θ_NL) = Σ_b ln L_marg^(b)(cosmo, bias_b) + ln L_BAO(cosmo)`

with each per-bin term a self-contained 11-parameter Gaussian marginalization (11×11 `A_b`, not one dense 77×77). **This is a compilation strategy, not a statistical change**: one joint posterior, one sampler, shared cosmology coupling all bins at every step.

**Tech Stack:** JAX (float64), existing `jaxptpolypol.marginal_likelihood` + `theory.py`, pytest.

## Global Constraints

- **Purely additive.** Do NOT modify the behavior of `make_marginal_templates`, `make_marginal_log_posterior`, `make_joint_pk_bk_fn`, or `gaussian_marginal_loglike`. New code paths live beside them. All existing tests must stay green **without edits**: `tests/test_marginal_likelihood.py` (8), `tests/test_marginal_pipeline.py` (3 — reconstruction 1e-10, c1-ratio ==4.0, Fisher-Schur 1e-8), `tests/test_sampler_rwmh.py` (3), full suite currently **71 passed**.
- **Exactness is the product.** Every new path is validated against the monolith with `np.testing.assert_allclose(..., rtol=1e-10)` — **rtol, not atol**: the log-likelihood is O(1e3–1e4), so a legitimate ~1e-13 re-association reaches ~1e-9 absolute.
- `jax.config.update("jax_enable_x64", True)` at the top of every new test file, before `jax.numpy` import.
- Parameter names/keys stay static (compile-time); only values are traced.
- `git add` ONLY the files named in each task — the tree has unrelated dirty files (fisher notebooks, `cmb.py`). Never `git add -A`.
- Conventional commits with scope, e.g. `feat(theory):`, `feat(marginal):`, `test(marginal):`.
- Run tests from repo root: `cd /Users/nguyenmn/jaxPTPolyPol && python -m pytest <file> -v`.
- **No heavy runs.** Tasks 1–3 use the small 2-bin config (`N_K=8`, `n_gl=8`, `num_mu=num_phi=8`, `n_gl>=7` is enforced by a real guard). Do not execute the 7-bin notebook except where Task 4 explicitly says to.
- Structural facts (verified, cite in code comments where useful): per-bin block length `3*n_k + n_tri`; production notebook = 7 bins × 375 = 2625 P+B rows + 13 BAO = 2638; `n_NL = 26`; `n_lin = 77 = 11 × 7`; `LIN_SURVEY_KEYS` order is fixed and `split_marginal_indices` emits `lin_idx` **bin-major, 11 contiguous per bin**.

---

### Task 1: Per-bin theory evaluation (`bins=` selector + single-bin block factory)

**Files:**
- Modify: `src/jaxptpolypol/theory.py` (`_make_theory_context_evaluator` ~305-439; add a new factory next to `make_joint_pk_bk_fn` ~923-1017)
- Test: `tests/test_theory_perbin.py` (new)

**Interfaces:**
- Produces: `make_joint_pk_bk_bin_fn(*, bin_index, **same_kwargs_as_make_joint_pk_bk_fn) -> bin_fn(params, *, k, triangles) -> (3*n_k + n_tri,)` — bin `bin_index`'s block **only**, byte-comparable to the corresponding slice of `make_joint_pk_bk_fn`'s output. Carries attributes `bin_fn.bin_index`, `bin_fn.n_bins`, `bin_fn.ells`, `bin_fn.ap`, `bin_fn.layout` (mirroring `make_joint_pk_bk_fn`'s attribute style).
  - **Amended 2026-07-25 (controller):** an earlier draft specified a `bin_fn.block_len` attribute and a matching assertion. That is not implementable as a build-time value — the block length is `3*len(k) + len(triangles)` and both are *call-time* arguments — and faking it via a trace-time attribute mutation is a side effect, so it was dropped. The `assert blk.shape == (3 * k.shape[0] + triangles.shape[0],)` line already checks the identical property.
- Consumes: nothing new.

**Design (follow this, do not duplicate the loop body):** give the closure returned by `_make_theory_context_evaluator` an optional bin selector rather than copying its body:

```python
def evaluate_contexts(params, bins=None):
    ...
    bin_indices = tuple(range(n_bins)) if bins is None else tuple(int(b) for b in bins)
    # AP precompute: DELIBERATELY over ALL bins, even for a subset (see below)
    if ap and multi_bin:
        Hz_true_all = bg.Hz(omb, omc, h, z_bins_arr, mnu)      # unchanged from pre-selector code
        ...                                                     # same for the chi/DAz and tabulated paths
    contexts = []
    for i in bin_indices:                                       # `i` is the GLOBAL bin index throughout
        ...                                                     # existing body: Hz_true_all[i], DAz_true_all[i], DAz_fid[i], Hz_fid[i], z_bins[i]
```

> **Corrected 2026-07-25 after implementation (this replaces an earlier draft that batched the background over the *selected* bins only and used a separate `slot` index space).** Batching the background over a subset is **measurably wrong**: vmapping over 1 vs 2 redshifts shifts `D_A` by 1 ulp, which flips the exact-equality branch `float(alpha_perp) == 1.0 and float(alpha_para) == 1.0` in `ps_1loop_jax/ps_1loop.py:121` (`_ap_is_identity`) into an entirely different no-AP code path. Measured effect: the per-bin vs monolith agreement degrades from `rtol=1e-12` to **~8.5e-4 in P0/P2/P4** — six orders of magnitude, and precisely at the fiducial cosmology where the tests evaluate.
>
> Computing the background over all bins and indexing everything by the **global** `i` makes per-bin blocks match the monolith by construction and removes the `slot` index space (and its mispairing bug class) entirely. The wasted work is negligible: closed-form `bg.Hz` plus a `vmap` over `chi`'s 512-point `cumulative_simpson` (~3.6k integrand evaluations for 7 bins) against one bin's tree bispectrum at `num_mu=num_phi=65` (4225 angular nodes × n_triangles) plus a 1-loop `P(k,μ)`. In `background_mode="tabulated"` the cost is exactly zero, since `chi` runs on the bin-independent `z_bg_grid` either way.
>
> **Related upstream landmine, worth knowing before any eager-vs-jitted comparison:** because `_ap_is_identity` casts with `float()`, it raises under tracing and is swallowed, so the shortcut is live in **eager mode only** — eager and jitted evaluations of the same theory differ by ~1e-3 at the fiducial. Compare jit-to-jit only; a ~1e-3 mismatch means eager-vs-jit, not bad algebra.

Then the new factory mirrors `make_joint_pk_bk_fn` but calls `evaluate_contexts(params, bins=(bin_index,))` and returns the single concatenated `[P0,P2,P4,B0]` block (no outer loop, no final multi-bin concatenate).

- [ ] **Step 1: Write the failing test** — the equivalence that guards the whole plan.

```python
# tests/test_theory_perbin.py
"""Per-bin theory evaluation must reproduce the monolithic joint theory exactly."""
import os
import pathlib
from functools import partial

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

EMULATOR_PATH = os.environ.get(
    "PFS_EMULATOR_PATH",
    "/Users/nguyenmn/cosmopower-jax-for-pfs/cosmology/jense2024/jense_2023_camb_lcdm/networks/jense_2023_camb_lcdm_Pk_lin.npz",
)
needs_emulator = pytest.mark.skipif(
    not pathlib.Path(str(EMULATOR_PATH)).exists(),
    reason="PFS emulator not available (set PFS_EMULATOR_PATH)",
)


@pytest.fixture(scope="module")
def cfg():
    """Small 2-bin config; mirrors tests/test_marginal_pipeline.py's fixture."""
    from jaxptpolypol.model import CosmoEmulator, PS1LoopModel, BispectrumTreeModel
    from jaxptpolypol.params import CosmoParams, FullShapeSurveyParams, pack_joint_params
    from jaxptpolypol.theory import (
        build_bispectrum_triangles_from_k_grid, compute_fiducial_distances)
    from ps_1loop_jax import background as bg

    MNU_FIXED, K_NL_RSD = 0.06, 0.45
    z_bins, knl_bins, n_bar = (0.7, 0.9), (0.52, 0.65), (3.06e-4, 9.61e-4)
    cosmo_dict = {'ombh2': 0.02237, 'omch2': 0.1200, 'logA': 3.044,
                  'ns': 0.9649, 'h': 0.6736,
                  'z': 0.7, 'A_b': 3.13, 'eta_b': 0.603, 'logT_AGN': 7.8}
    cosmo = CosmoParams(cosmo_dict)

    def b1z(z): return 0.9 + 0.4 * z
    def b2z(z): return -0.704 - 0.208 * z + 0.183 * z**2 - 0.00771 * z**3
    def bG2z(z): return -(2. / 7.) * (b1z(z) - 1.)
    def bGamma3z(z): return (23. / 42.) * (b1z(z) - 1.)
    def Dplusz(z):
        return float(bg.growth_factor(cosmo_dict['ombh2'], cosmo_dict['omch2'],
                                      cosmo_dict['h'], z, mnu=MNU_FIXED))

    surveys = []
    for z, knl, nd in zip(z_bins, knl_bins, n_bar):
        surveys.append(FullShapeSurveyParams(
            shared={'bias': {'b1': b1z(z), 'b2': b2z(z), 'bG2': bG2z(z),
                             'bGamma3': bGamma3z(z)},
                    'stoch': {'P_shot': 1.0}, 'k_nl': knl, 'ndens': nd},
            pk={'ctr': {'c0': 25. * Dplusz(z)**2, 'c2': 25. * Dplusz(z)**2,
                        'c4': Dplusz(z)**2, 'cfog': knl**(-4)},
                'stoch': {'a0': 0., 'a2': 0.}},
            bk={'ctr': {'c1': 0.0}, 'stoch': {'B_shot': 1.0, 'A_shot': 1.0}},
        ))
    survey_keys = surveys[0].joint_param_keys
    packed = pack_joint_params(cosmo, surveys)
    Hz_fid, DAz_fid = compute_fiducial_distances(cosmo, z_bins)
    emulator = CosmoEmulator(probe='custom_log', emulator_path=EMULATOR_PATH)
    kwargs = dict(
        pklin_emulator=emulator, ps1loop_model=PS1LoopModel(do_irres=True),
        bispectrum_model=BispectrumTreeModel(do_AP=True, k_nl_rsd=K_NL_RSD),
        cosmo_keys=cosmo.param_keys, cosmo_sizes=cosmo.param_sizes,
        survey_keys=survey_keys, ap=True, z_bins=z_bins,
        Hz_fid=Hz_fid, DAz_fid=DAz_fid, n_gl=8, num_mu=8, num_phi=8,
        background_mode="direct")
    k = jnp.linspace(0.02, 0.18, 8)
    triangles, _ = build_bispectrum_triangles_from_k_grid(
        k, k_min=0.02, k_max=0.10, dk=float(k[1] - k[0]))
    return kwargs, packed, k, triangles, len(z_bins)


@needs_emulator
def test_perbin_blocks_concatenate_to_joint(cfg):
    from jaxptpolypol.theory import make_joint_pk_bk_fn, make_joint_pk_bk_bin_fn
    kwargs, packed, k, triangles, n_bins = cfg

    joint = make_joint_pk_bk_fn(**kwargs)
    full = joint(packed, k=k, triangles=triangles)

    blocks = []
    for b in range(n_bins):
        bin_fn = make_joint_pk_bk_bin_fn(bin_index=b, **kwargs)
        blk = bin_fn(packed, k=k, triangles=triangles)
        assert blk.shape == (3 * k.shape[0] + triangles.shape[0],)
        assert blk.shape[0] == bin_fn.block_len
        blocks.append(blk)
    recon = jnp.concatenate(blocks)
    assert recon.shape == full.shape
    np.testing.assert_allclose(np.asarray(recon), np.asarray(full), rtol=1e-12)


@needs_emulator
def test_perbin_block_is_bin_specific(cfg):
    """Guards the slot-vs-global index pairing: bin 1 must NOT equal bin 0."""
    from jaxptpolypol.theory import make_joint_pk_bk_bin_fn
    kwargs, packed, k, triangles, _ = cfg
    b0 = make_joint_pk_bk_bin_fn(bin_index=0, **kwargs)(packed, k=k, triangles=triangles)
    b1 = make_joint_pk_bk_bin_fn(bin_index=1, **kwargs)(packed, k=k, triangles=triangles)
    assert not np.allclose(np.asarray(b0), np.asarray(b1), rtol=1e-6)
```

- [ ] **Step 2: Run to verify it fails** — `python -m pytest tests/test_theory_perbin.py -v` → FAIL with `ImportError: cannot import name 'make_joint_pk_bk_bin_fn'`.
- [ ] **Step 3: Implement** the `bins=` selector + `make_joint_pk_bk_bin_fn` per the Design block above. Keep `evaluate_contexts(params)` (no `bins`) behaviorally identical.
- [ ] **Step 4: Run the new tests AND the existing suite**

```bash
python -m pytest tests/test_theory_perbin.py -v
python -m pytest tests/ -q --ignore=tests/test_marginal_pipeline.py   # expect 71 passed
python -m pytest tests/test_marginal_pipeline.py -v                    # expect 3 passed (~15 min: the Hessian/Fisher-Schur test dominates -- budget for it, do not treat a quiet agent as stalled)
```

- [ ] **Step 5: Commit** — `git add src/jaxptpolypol/theory.py tests/test_theory_perbin.py` then `git commit -m "feat(theory): per-bin theory block evaluation via bins= selector"`

---

### Task 2: Per-bin marginal log-posterior (Python-unrolled)

**Files:**
- Modify: `src/jaxptpolypol/marginal_likelihood.py`, `src/jaxptpolypol/__init__.py`
- Test: `tests/test_marginal_perbin.py` (new)

**Interfaces:**
- Consumes: Task 1's `make_joint_pk_bk_bin_fn`; existing `gaussian_marginal_loglike`, `make_marginal_templates`, `split_marginal_indices`.
- Produces:
  - `bin_lin_slices(split, n_bins) -> tuple[slice, ...]` — position of each bin's 11 lin entries **within the (n_lin,) prior vectors**, i.e. `slice(b*11, (b+1)*11)`; assert `split.lin_keys` is bin-major with equal counts and raise `ValueError` otherwise.
  - `make_marginal_log_posterior_perbin(*, bin_theory_fns, bin_data, bin_cov_invs, bin_lin_idx, extra_theory_fn=None, extra_data=None, extra_cov_inv=None, prior_mean_fn, prior_sigma_fn, log_prior_nl_fn, to_physical, full_params_fn, include_logdet=True) -> log_posterior(x)` (jitted).
    - `bin_theory_fns[b](full_params) -> (n_data_b,)` (statics pre-bound with `partial`).
    - `bin_lin_idx[b]` = that bin's 11 **full-vector** indices (from `split.lin_idx[b*11:(b+1)*11]`).
    - `prior_mean_fn/prior_sigma_fn(theta_nl) -> (n_lin,)` — unchanged Stream-A/B contract; sliced per bin internally with `bin_lin_slices`.
    - `extra_*` is the BAO term: no lin params ⇒ plain `-0.5 r^T Cinv r` added once.
    - Sum over bins of `gaussian_marginal_loglike(data_b, m0_b, M_b, cov_inv_b, mu_b, sigma_b, include_logdet=...)`, plus the extra term, plus `log_prior_nl_fn(theta_nl)`.

**Why this is exact:** `ln det(A Σ_p)` of a block-diagonal `A` is `Σ_b ln det(A_b Σ_p,b)`, and the quadratic forms are block-separable — so the per-bin sum equals the monolith term-by-term, not merely in the limit.

- [ ] **Step 1: Write the failing tests** — equivalence against the monolith at the fiducial *and* at a displaced θ_NL (a fiducial-only check would pass even if the θ_NL-dependence were wired wrong), plus the BAO-term and slice-helper checks.

```python
# tests/test_marginal_perbin.py
# (reuse the `cfg` fixture pattern from tests/test_theory_perbin.py — copy it verbatim,
#  adding `survey_keys` and `n_cosmo` to the returned tuple.)

@needs_emulator
def test_perbin_logpost_equals_monolith(cfg):
    """Same joint posterior, computed two ways: sum-of-bins == dense 77x77."""
    from jaxptpolypol.marginal_likelihood import (
        make_constant_prior_fns, make_marginal_log_posterior,
        make_marginal_log_posterior_perbin, split_marginal_indices)
    from jaxptpolypol.sampler import make_full_params_fn
    from jaxptpolypol.theory import make_joint_pk_bk_fn, make_joint_pk_bk_bin_fn
    kwargs, packed, k, triangles, n_bins, survey_keys, n_cosmo = cfg

    joint = make_joint_pk_bk_fn(**kwargs)
    theory_fn = partial(joint, k=k, triangles=triangles)
    split = split_marginal_indices(
        n_cosmo_params=n_cosmo, survey_keys=survey_keys, n_bins=n_bins,
        fixed_cosmo=(5, 6, 7, 8),
        fixed_survey_keys={('shared', 'k_nl', None), ('shared', 'ndens', None)})

    data = theory_fn(packed)                       # noiseless mock
    n_data = data.shape[0]
    block = n_data // n_bins
    rng = np.random.default_rng(0)
    # Block-diagonal, non-identity covariance (identity would hide block-pairing bugs)
    blocks = []
    for _ in range(n_bins):
        d = rng.uniform(0.5, 2.0, size=block)
        blocks.append(np.diag(d))
    from scipy.linalg import block_diag as _bd
    cov = _bd(*blocks)
    cov_inv = jnp.asarray(np.linalg.inv(cov))
    mu_p = packed[jnp.array(split.lin_idx)]
    sigma_p = jnp.asarray(rng.uniform(0.5, 2.0, size=split.n_lin))
    prior_mean_fn, prior_sigma_fn = make_constant_prior_fns(mu_p, sigma_p)
    fpf = make_full_params_fn(packed, split.nl_idx)

    mono = make_marginal_log_posterior(
        theory_fn=theory_fn, data=data, cov_inv=cov_inv, lin_idx=split.lin_idx,
        prior_mean_fn=prior_mean_fn, prior_sigma_fn=prior_sigma_fn,
        log_prior_nl_fn=lambda t: 0.0, to_physical=lambda x: x,
        full_params_fn=fpf, include_logdet=True)

    bin_fns = [partial(make_joint_pk_bk_bin_fn(bin_index=b, **kwargs),
                       k=k, triangles=triangles) for b in range(n_bins)]
    per = make_marginal_log_posterior_perbin(
        bin_theory_fns=bin_fns,
        bin_data=[data[b * block:(b + 1) * block] for b in range(n_bins)],
        bin_cov_invs=[jnp.asarray(np.linalg.inv(blocks[b])) for b in range(n_bins)],
        bin_lin_idx=[split.lin_idx[b * 11:(b + 1) * 11] for b in range(n_bins)],
        prior_mean_fn=prior_mean_fn, prior_sigma_fn=prior_sigma_fn,
        log_prior_nl_fn=lambda t: 0.0, to_physical=lambda x: x,
        full_params_fn=fpf, include_logdet=True)

    x0 = packed[jnp.array(split.nl_idx)]
    np.testing.assert_allclose(float(per(x0)), float(mono(x0)), rtol=1e-10)
    # displaced point — catches wrong theta_NL wiring that a fiducial-only test misses
    x1 = x0 * (1.0 + 0.01 * jnp.arange(x0.shape[0]) / max(x0.shape[0], 1))
    np.testing.assert_allclose(float(per(x1)), float(mono(x1)), rtol=1e-10)


@needs_emulator
def test_perbin_logpost_with_extra_bao_term_equals_monolith(cfg):
    """BAO as a separate cosmology-only chi^2 == BAO concatenated into the monolith."""
    # Build a stand-in cosmology-only 'BAO' map: bao_fn(full_params) = A @ full_params[:n_cosmo]
    # with fixed random A (13, n_cosmo) and its own 13x13 covariance; append to the monolith's
    # data/cov via block_diag and pass as extra_* to the per-bin builder. Assert rtol=1e-10 at x0
    # and at the same displaced x1 as above.
    ...


def test_bin_lin_slices_and_validation():
    """Pure-python: slices are 11-wide bin-major; non-bin-major input raises."""
    ...
```

Write the two `...` bodies out fully following the pattern of the first test (they are ordinary code, not placeholders — the `...` marks where to write, not what to skip).

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_marginal_perbin.py -v` → `ImportError`.
- [ ] **Step 3: Implement** `bin_lin_slices` + `make_marginal_log_posterior_perbin`; export both from `marginal_likelihood.__all__` and `src/jaxptpolypol/__init__.py` (match the existing names-import block).
- [ ] **Step 4: Run** the new file, then `python -m pytest tests/ -q --ignore=tests/test_marginal_pipeline.py` (71 + new), then `tests/test_marginal_pipeline.py -v` (3).
- [ ] **Step 5: Commit** — `git add src/jaxptpolypol/marginal_likelihood.py src/jaxptpolypol/__init__.py tests/test_marginal_perbin.py` then `git commit -m "feat(marginal): exact per-bin factorized marginal log-posterior (Python-unrolled)"`

---

### Task 3: `lax.scan` form + compile/memory measurement

**Files:**
- Modify: `src/jaxptpolypol/marginal_likelihood.py`, `src/jaxptpolypol/__init__.py`
- Test: `tests/test_marginal_perbin.py` (append)
- Create: `docs/design/perbin-compile-measurements.md`

**Interfaces:**
- Produces: `make_marginal_log_posterior_scan(*, bin_theory_fn, bin_params_fn, bin_data, bin_cov_invs, bin_lin_idx, ...same as perbin...) -> log_posterior(x)`.
  - `bin_theory_fn(full_params, bin_index)` — ONE function reused for every bin (this is the whole point: one compiled body). Build it from Task 1's factory with a traced/`lax.switch`-free path if possible; if the bin index must be static, accept a `bin_theory_fns` tuple and scan over **stacked per-bin data/cov/lin arrays** while the theory call stays inside the scan body via `jax.lax.switch` — document whichever is used and why.
  - Stack `bin_data` → `(n_bins, block)`, `bin_cov_invs` → `(n_bins, block, block)`, `bin_lin_idx` → `(n_bins, 11)`; scan carries the running scalar.
- If a genuine blocker makes a single reusable body impossible (e.g. per-bin static `z` baked into the closure), STOP and report BLOCKED with the specific obstruction — do not silently fall back to the unrolled form.

- [ ] **Step 1: Write the failing test** — `test_scan_logpost_equals_perbin_and_monolith`: same fixture, assert `scan(x0) == perbin(x0) == mono(x0)` and the same at the displaced `x1`, all `rtol=1e-10`.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** the scan form; export it.
- [ ] **Step 4: Run** the new test + full suite + pipeline suite (all green).
- [ ] **Step 5: Measure and record.** Write a small script (scratchpad, not committed) that, on the **2-bin** config, times first-compile and peak RSS for: monolith, perbin, scan. Record the three numbers plus `n_data`, `n_lin` in `docs/design/perbin-compile-measurements.md` with the exact command used. This is the evidence for the plan's central claim; if scan is NOT better than perbin, say so plainly in the doc.
- [ ] **Step 6: Commit** — `git add src/jaxptpolypol/marginal_likelihood.py src/jaxptpolypol/__init__.py tests/test_marginal_perbin.py docs/design/perbin-compile-measurements.md` then `git commit -m "feat(marginal): lax.scan per-bin marginal posterior + compile measurements"`

---

### Task 4: Wire the LCDM notebook to the factorized posterior + smoke gate

**Files:**
- Modify: `example/mcmc/mcmc_joint_PFS_BAO_BBN_ns_LCDM.ipynb`

**Interfaces:** consumes Tasks 1–3. The notebook currently builds `combined_theory` (7-bin `joint_fn` ++ BAO) and calls `make_marginal_log_posterior`. Replace **only** the posterior construction in the `# Combined (noiseless) forecast data vector:` cell:
- per-bin theory fns via `make_joint_pk_bk_bin_fn(bin_index=b, ...)` reusing the notebook's existing `joint_fn` kwargs;
- per-bin data slices of `pb_fid` (block = `3*n_k + n_tri`) and per-bin `cov_inv` blocks — slice `gauss_cov` block-diagonally (`block_len` stride) and invert **per block** (cheaper and better-conditioned than inverting 2625×2625);
- BAO as `extra_theory_fn=lambda p: bao_theory_fn(p[:n_cosmo_params])`, `extra_data=bao_fid`, `extra_cov_inv=inv(bao_dr2.cov)`;
- **`make_marginal_log_posterior_perbin`** — amended 2026-07-25 on Task 3's measured evidence: `..._scan` compiles ~1.7× *slower* than `..._perbin` (28.7 s vs 16.9 s) for identical memory and op count, because `make_joint_pk_bk_bin_fn` bakes `bin_index` statically, forcing a `lax.switch` body that traces all `n_bins` branches. See `docs/design/perbin-compile-measurements.md`. Use `_perbin`; `_scan` stays in the library as a measured negative result.
Keep the MH cells, `run_rwmh`, diagnostics, and all Fisher-comparison cells untouched.

- [ ] **Step 1: Add an assertion cell** (immediately after the posterior is built) that the factorized `log_post(x0)` is finite and, if the monolith path is still cheap enough to build, agrees with it — otherwise just assert finite and print the value.
- [ ] **Step 2: Coarse smoke.** Copy the notebook to `example/mcmc/_smoke_tmp.ipynb` (same directory — relative BAO paths must resolve), append `N_K = 8; N_GL = 8; NUM_MU = NUM_PHI = 10` to the config cell, set `SMOKE_TEST = True`, and run:
  `cd example/mcmc && jupyter nbconvert --to notebook --execute --inplace _smoke_tmp.ipynb --ExecutePreprocessor.timeout=None`
  Record wall time, `log_post(x0)`, MH acceptance. Delete `_smoke_tmp.ipynb` afterwards. Expect minutes, not hours.
- [ ] **Step 3: Full 7-bin forward gate.** With production grids and `SMOKE_TEST = True` (tiny chain), run the real notebook the same way **in the background**, capturing wall time and peak RSS (sample `ps -o rss=` on the kernel every 15 s to a log). GATE: first `log_post(x0)` compile should be **minutes, ~10–15 GB** — the plan's central claim. If it exceeds 45 min or 60 GB, STOP and report BLOCKED with the measured numbers rather than letting it run.
- [ ] **Step 4: Restore** the production `NUM_*`/`SMOKE_TEST` values and commit — `git add example/mcmc/mcmc_joint_PFS_BAO_BBN_ns_LCDM.ipynb` then `git commit -m "feat(mcmc): LCDM notebook uses the factorized per-bin marginal posterior"`

---

## Verification before completion (Phase-1 exit criteria)

Phase 1 is complete only when **all** of these are evidenced by pasted command output — not asserted:

1. `python -m pytest tests/ -v` → all green, including the 3 pipeline guards, with **no edits to existing tests**.
2. Per-bin and scan posteriors match the monolith at `rtol=1e-10` at the fiducial **and** at a displaced θ_NL (Tasks 2–3).
3. Per-bin theory blocks concatenate to the monolithic joint theory at `rtol=1e-12`, and bin 0 ≠ bin 1 (Task 1).
4. `docs/design/perbin-compile-measurements.md` records measured compile time + peak RSS for monolith vs perbin vs scan.
5. The 7-bin notebook's forward `log_post(x0)` compiles within the Task-4 gate and returns a finite value, with the wall-time/RSS numbers recorded.
