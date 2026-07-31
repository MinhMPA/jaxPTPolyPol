# Stream-B: DESI DR1-Reanalysis Priors (arXiv:2511.20757) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adopt the EFT/stochastic nuisance priors of arXiv:2511.20757 Table I via a dual-verified two-layer convention map, a mapped+validated packaged spec, and θ_NL-dependent prior functions, ending with the production LCDM MCMC notebook running on the new spec.

**Architecture:** Two independent agents derive the layer-1 coefficient-convention map from the primary papers and the ps_1loop_jax operators; a reconciliation pass produces the authoritative map document with a machine-readable spec appendix. A new `desi_priors.py` module loads the spec with load-time reconciliation checks and builds `(prior_mean_fn, prior_sigma_fn, log_prior_nl_fn)` with runtime A_AP·A_amp rescaling. The acceptance gate runs a surrogate chain under the new priors against the surrogate's own Hessian-Fisher (templates are prior-independent, so no rebuild).

**Tech Stack:** JAX (float64), PyYAML, existing jaxptpolypol marginal-likelihood + Taylor-surrogate machinery, ps_1loop_jax background module.

## Global Constraints

- All seven grill decisions in `CONTEXT.md` § "Stream-B decisions (grill session 2026-07-30)" are binding; the plan implements them exactly.
- Layer-1 factors are parameterized by the **production config**, never library defaults: `k_nl_rsd = 0.45` (bispectrum c1 normalization), per-bin `knl_bins = (0.52, 0.65, 0.82, 1.02, 1.29, 1.82, 2.88)` (stochastic a0/a2 normalization), from `example/mcmc/scripts/build_taylor_templates_lcdm.py:173,179`.
- a0/a2 widths map per-bin: σ_ours = σ_paper × (knl_b/0.45)² (grill decision 4).
- Spec format is mapped+validated: every entry stores our-convention value, verbatim Table-I paper value, and the layer-1 factor (+ affine offset); the loader raises unless `mean == paper_mean*factor + offset` and `sigma == paper_sigma*|factor|` (grill decision 2).
- θ_lin ordering per bin is `LIN_SURVEY_KEYS` (`src/jaxptpolypol/marginal_likelihood.py:80-92`): bGamma3, P_shot, c0, c2, c4, cfog, a0, a2, c1, B_shot, A_shot. `prior_mean_fn`/`prior_sigma_fn` return `(n_lin,)` = (n_bins × 11,) arrays in bin-major order (bin 0's 11 entries first).
- A Gaussian prior on the paper's rescaled variable y = x·R implies on our raw coefficient x: mean m/R, width s/R. Runtime rescaling **divides** by R(θ_NL); R = 1 at fiducial by construction (σ8_ref = fiducial σ8).
- `jax.config.update("jax_enable_x64", True)` in every script/test.
- Compare jitted-to-jitted only (the `_ap_is_identity` eager/jit trap).
- Notebook scope: `example/mcmc/mcmc_joint_PFS_BAO_BBN_ns_LCDM.ipynb` only; keep `run_rwmh_python`; nuLCDM stays on hold; Fisher example notebooks stay on the legacy spec.
- Subagents run on Opus 4.8 (user preference). Tasks 1 and 2 MUST be executed by different agents, and the Task-2 agent MUST NOT read Task 1's artifact (independence is the point).
- Legacy spec `eft_eq12_2405_02252.yaml` is retained untouched.

## File Structure

| File | Responsibility |
|------|----------------|
| `docs/design/desi-convention-map-A.md` (new, Task 1) | Derivation A working artifact |
| `docs/design/desi-convention-map-B.md` (new, Task 2) | Derivation B working artifact (independent) |
| `docs/design/desi-convention-map.md` (new, Task 3) | Authoritative reconciled map + machine-readable spec-rows appendix |
| `src/jaxptpolypol/data/priors/desi_dr1_reanalysis_2511_20757.yaml` (new, Task 4) | The packaged spec (verbatim copy of the Task-3 appendix) |
| `src/jaxptpolypol/desi_priors.py` (new, Tasks 4–6) | Spec dataclasses + validating loader; `make_desi_prior_fns`; `make_lcdm_rescaling_fns`; `build_prior_sigmas_from_desi_spec` |
| `tests/test_desi_priors.py` (new, Tasks 4–6) | Loader validation, width/mean unit tests (toy closures), Fisher-consumption tests |
| `example/mcmc/scripts/desi_prior_validation.py` (new, Task 7) | Acceptance gate: surrogate chain vs Hessian-Fisher under new priors |
| `example/mcmc/mcmc_joint_PFS_BAO_BBN_ns_LCDM.ipynb` (modify, Task 8) | Switch prior construction to the spec |
| `src/jaxptpolypol/__init__.py` (modify, Tasks 4–5) | Export new public API |
| `CONTEXT.md`, `docs/design/perbin-compile-measurements.md` (modify, Tasks 3, 7) | Deviations + gate results records |

---

### Task 1: Convention-map derivation A

**Files:**
- Create: `docs/design/desi-convention-map-A.md`

**Interfaces:**
- Produces: a per-parameter derivation document in the fixed table format below, consumed by Task 3.

This is a research/derivation task, not a coding task. The deliverable is a document.

- [ ] **Step 1: Fetch and read the primary sources**

Fetch (WebFetch) and read:
- `https://arxiv.org/pdf/2511.20757` — §II.3, Table I **including its footnote** (the A_AP/A_amp rescaling definitions), and any appendix defining the P and B model operators.
- `https://arxiv.org/pdf/2004.10607` (CLASS-PT) — the counterterm and stochastic operator definitions (their eqs. for `−2 c̃_ℓ k² μ^{2ℓ} P`, the k⁴μ⁴ FoG term, and the stochastic `1/n̄ [1 + P_shot + a0(k/0.45)² + a2 μ²(k/0.45)²]` form).
- If 2511.20757 defers its bispectrum shot-noise definitions to a companion (e.g. its refs for the B likelihood), fetch that companion's B_shot/A_shot operator equation too.

Transcribe **Table I verbatim** (every nuisance row: variable as printed, prior mean, prior width, units, marginalized-vs-sampled type) into a "Table I verbatim" section of the artifact. This transcription is the audit anchor for everything downstream.

- [ ] **Step 2: Read the code-side operator definitions**

Read these exact locations (the operators our coefficients multiply):
- `/Users/nguyenmn/ps_1loop_jax-for-pfs/src/ps_1loop_jax/ps_1loop.py:581-600` — k² counterterms: `P_ctr = −2k²[c0 + c2·fμ² + c4·f²μ⁴]·P(k,μ)`, coefficients in (Mpc/h)².
- `ps_1loop.py:603-622` — k⁴ FoG: `P_ctr4 = −k⁴·cfog·f⁴μ⁴(b1+fμ²)²·P(k,μ)`.
- `ps_1loop.py:657-672` — stochastic: `(1/n̄)[P_shot + a0(k/k_nl)² + a2(k/k_nl)²μ²]` with **per-bin** k_nl.
- `/Users/nguyenmn/ps_1loop_jax-for-pfs/src/ps_1loop_jax/bs_tree.py:160-180` — c1 in `Z1_fog = Z1 − c1·μ²(k/k_nl_rsd)²`, and the surrounding B tree expression.
- `bs_tree.py:85-140` — B_shot/A_shot shot-noise operators (trace the full expressions they multiply).
- bGamma3: locate its kernel contribution (grep `bGamma3` in ps_1loop_jax) — linear in the model.
- Production config: `/Users/nguyenmn/jaxPTPolyPol/example/mcmc/scripts/build_taylor_templates_lcdm.py:173` (`knl_bins`), `:179` (`K_NL_RSD = 0.45`), `:203` (k-ranges), `:258-264` (fiducial parameter values incl. `cfog = knl**(-4)`).

- [ ] **Step 3: Derive the layer-1 map per parameter**

For each of the 11 θ_lin parameters + sampled b2, bG2: equate our operator's coefficient with the paper's operator's coefficient at identical (k, μ, z) and solve for `ours = paper × factor + offset`. Rules:
- Factors must be expressed in terms of config symbols where config-dependent: a0/a2 factor is `(knl_b/0.45)²` (per-bin formula); c1 factor uses the production `k_nl_rsd = 0.45`, not the bs_tree.py default 0.3.
- P_shot: ours is a mean-1 Poisson amplitude (`(1/n̄)·P_shot`), theirs a mean-0 deviation — derive the exact affine map (expected: factor 1, offset 1; verify their operator normalization `1/n̄` matches, including whether their n̄ convention differs).
- Check for extra f/b1 factors: the paper's c1 operator may carry factors ours folds into the kernel (or vice versa). Same check for cfog vs their c̃ (theirs may define the k⁴ term with `(b1+fμ²)²` explicitly or absorbed).
- Identify each row's layer-2 rescale flag from the Table-I footnote: which of A_AP, A_AP·A_amp, A_AP·A_amp² multiplies each prior variable, verbatim.
- bGamma3: record the paper's mean expression (expected `23/42·(b1−1)`, in whichever b1 variable they use — note explicitly whether it is raw b1 or b1σ8) and its width.
- b2/bG2 (sampled block): paper priors are on σ8-scaled combos (`b2σ8²`, `b𝒢2σ8²` expected 𝒩(0,5²)) — record the raw-variable width rule `5/σ8²(z)`.

- [ ] **Step 4: Write the artifact in the fixed format**

`docs/design/desi-convention-map-A.md` must contain:
1. "Table I verbatim" section (from Step 1).
2. The map table with EXACTLY these columns, one row per parameter (11 marginalized + b2 + bG2):

```
| param | our operator (file:line) | paper operator (eq. ref) | our units | paper units | factor | offset | layer-2 rescale | paper mean | paper sigma | our mean | our sigma | notes |
```

Numeric factors as exact expressions (e.g. `0.45^2 = 0.2025`), config-formula factors as `(knl_b/0.45)^2`. `our mean`/`our sigma` are the layer-1-mapped values BEFORE layer-2 (layer-2 is runtime).
3. A "confidence + open issues" section: any row where the paper is ambiguous, any factor you could not pin, any Table-I row that contradicts `CONTEXT.md` (e.g. its recorded c2 mean 30, c̃ 𝒩(400,400²), c1 𝒩(0,5²)).

- [ ] **Step 5: Commit**

```bash
git add docs/design/desi-convention-map-A.md
git commit -m "docs(stream-b): convention-map derivation A (independent)"
```

---

### Task 2: Convention-map derivation B (independent)

**Files:**
- Create: `docs/design/desi-convention-map-B.md`

**Interfaces:**
- Produces: same fixed-format artifact as Task 1, derived independently; consumed by Task 3.

Repeat Task 1's Steps 1–5 exactly, writing to `docs/design/desi-convention-map-B.md` with commit message `docs(stream-b): convention-map derivation B (independent)`. **Hard constraint: do not open, read, or search for `desi-convention-map-A.md` or any Task-1 material.** Work only from the primary papers, the ps_1loop_jax code, the production config, and CONTEXT.md. The value of this task is an independent second derivation; any contamination voids it.

---

### Task 3: Reconciliation → authoritative map + spec-rows appendix

**Files:**
- Create: `docs/design/desi-convention-map.md`
- Modify: `CONTEXT.md` (deviations section, only if new deviations emerge)

**Interfaces:**
- Consumes: `desi-convention-map-A.md`, `desi-convention-map-B.md`.
- Produces: the authoritative map; its final section is a fenced YAML block titled `## Spec rows (machine-readable)` in EXACTLY the schema of Task 4, which Task 4 copies verbatim into the packaged spec file.

- [ ] **Step 1: Diff the two derivations row by row**

For every row, compare: factor, offset, layer-2 flag, paper mean/sigma, units. Record three lists: AGREE (identical), RECONCILABLE (same physics, different presentation — normalize and confirm), DISAGREE (different factor/offset/flag).

- [ ] **Step 2: Resolve disagreements**

For each DISAGREE row: re-derive from the primary source yourself, quoting the paper equation and the code lines in the final artifact. If after re-derivation the row is still ambiguous (the paper genuinely under-specifies the operator), STOP and report BLOCKED — per grill decision 1 the fallback is a CLASS-PT numerical cross-check, which is a user decision (new heavy dependency).

- [ ] **Step 3: Write the authoritative map**

`docs/design/desi-convention-map.md`: the fixed-format table (same columns as Task 1 Step 4) with every row carrying a provenance line "A: <value> | B: <value> | resolution: <agree/how resolved>", the verbatim Table I, and the reconciliation log. Then append the machine-readable spec-rows YAML block in Task 4's schema (fill every field; the `factor_formula`/`mean_formula`/`rescale` tokens must come from Task 4's allowed vocabularies).

- [ ] **Step 4: Cross-check against CONTEXT.md recorded facts**

Verify the map is consistent with CONTEXT.md's recorded values: c1 paper prior 𝒩(0,5²) [Mpc/h]² on `c1·A_AP·A_amp`; c2 paper mean 30; c̃ 𝒩(400,400²) [Mpc/h]⁴; b2/bG2 𝒩(0,5²) σ8-scaled; bΓ3 mean 23/42(b1−1); P_shot mean-0↔mean-1 shift; Table I has B_shot/A_shot rows (14 = 11+3 arithmetic). Any contradiction: update CONTEXT.md's record with the primary-source value and note the correction in the commit message — the primary PDF governs.

- [ ] **Step 5: Commit**

```bash
git add docs/design/desi-convention-map.md CONTEXT.md
git commit -m "docs(stream-b): reconciled two-layer convention map + spec-rows appendix"
```

---

### Task 4: Packaged spec + validating loader

**Files:**
- Create: `src/jaxptpolypol/data/priors/desi_dr1_reanalysis_2511_20757.yaml`
- Create: `src/jaxptpolypol/desi_priors.py`
- Create: `tests/test_desi_priors.py`
- Modify: `src/jaxptpolypol/__init__.py` (add exports)

**Interfaces:**
- Consumes: the spec-rows appendix of `docs/design/desi-convention-map.md`.
- Produces: `load_desi_prior_spec(name="desi_dr1_reanalysis_2511_20757") -> DesiPriorSpec`; `DesiPriorSpec(metadata: dict, marginalized: dict[tuple, MarginalRow], sampled: dict[str, SampledRow])`; `MarginalRow(paper_mean, paper_sigma, paper_units, paper_variable, factor, offset, mean, sigma, rescale, factor_formula, mean_formula)`; `SampledRow(kind, paper_mean, paper_sigma, paper_variable, rescale)`; `SpecValidationError(ValueError)`. Tasks 5–8 rely on these names.

**Schema** (governs the toy fixture below, the real spec, and the Task-3 appendix). Allowed vocabularies: `rescale ∈ {none, A_AP, A_AP*A_amp, A_AP*A_amp^2}`; `factor_formula ∈ {null, knl_over_0p45_sq}`; `mean_formula ∈ {null, coevolution_bGamma3}`; sampled `rescale ∈ {none, sigma8_sq}`, `kind ∈ {flat, gaussian}`.

- [ ] **Step 1: Write the failing loader tests (toy fixture)**

In `tests/test_desi_priors.py`:

```python
"""Tests for the desi_dr1_reanalysis_2511_20757 prior spec machinery."""
import jax
jax.config.update("jax_enable_x64", True)

import textwrap
import numpy as np
import pytest

from jaxptpolypol.desi_priors import (
    DesiPriorSpec, SpecValidationError, load_desi_prior_spec,
)
from jaxptpolypol.marginal_likelihood import LIN_SURVEY_KEYS

TOY_YAML = textwrap.dedent("""\
metadata:
  source: "toy"
  paper_knl: 0.45
  production_k_nl_rsd: 0.45
marginalized:
  shared.bias.bGamma3:
    {paper_mean: null, paper_sigma: 1.0, paper_units: "unit",
     paper_variable: "bGamma3*A_AP*A_amp^2", factor: 1.0, offset: 0.0,
     mean: null, sigma: 1.0, rescale: "A_AP*A_amp^2",
     factor_formula: null, mean_formula: "coevolution_bGamma3"}
  shared.stoch.P_shot:
    {paper_mean: 0.0, paper_sigma: 2.0, paper_units: "unit",
     paper_variable: "P_shot*A_AP", factor: 1.0, offset: 1.0,
     mean: 1.0, sigma: 2.0, rescale: "A_AP",
     factor_formula: null, mean_formula: null}
  pk.ctr.c0:
    {paper_mean: 0.0, paper_sigma: 30.0, paper_units: "(Mpc/h)^2",
     paper_variable: "c0*A_AP*A_amp", factor: 1.0, offset: 0.0,
     mean: 0.0, sigma: 30.0, rescale: "A_AP*A_amp",
     factor_formula: null, mean_formula: null}
  pk.ctr.c2:
    {paper_mean: 30.0, paper_sigma: 30.0, paper_units: "(Mpc/h)^2",
     paper_variable: "c2*A_AP*A_amp", factor: 0.5, offset: 0.0,
     mean: 15.0, sigma: 15.0, rescale: "A_AP*A_amp",
     factor_formula: null, mean_formula: null}
  pk.ctr.c4:
    {paper_mean: 0.0, paper_sigma: 30.0, paper_units: "(Mpc/h)^2",
     paper_variable: "c4*A_AP*A_amp", factor: 1.0, offset: 0.0,
     mean: 0.0, sigma: 30.0, rescale: "A_AP*A_amp",
     factor_formula: null, mean_formula: null}
  pk.ctr.cfog:
    {paper_mean: 400.0, paper_sigma: 400.0, paper_units: "(Mpc/h)^4",
     paper_variable: "ctilde*A_AP*A_amp", factor: 1.0, offset: 0.0,
     mean: 400.0, sigma: 400.0, rescale: "A_AP*A_amp",
     factor_formula: null, mean_formula: null}
  pk.stoch.a0:
    {paper_mean: 0.0, paper_sigma: 1.0, paper_units: "unit",
     paper_variable: "a0*A_AP", factor: 1.0, offset: 0.0,
     mean: 0.0, sigma: 1.0, rescale: "A_AP",
     factor_formula: "knl_over_0p45_sq", mean_formula: null}
  pk.stoch.a2:
    {paper_mean: 0.0, paper_sigma: 1.0, paper_units: "unit",
     paper_variable: "a2*A_AP", factor: 1.0, offset: 0.0,
     mean: 0.0, sigma: 1.0, rescale: "A_AP",
     factor_formula: "knl_over_0p45_sq", mean_formula: null}
  bk.ctr.c1:
    {paper_mean: 0.0, paper_sigma: 5.0, paper_units: "(Mpc/h)^2",
     paper_variable: "c1*A_AP*A_amp", factor: 0.2025, offset: 0.0,
     mean: 0.0, sigma: 1.0125, rescale: "A_AP*A_amp",
     factor_formula: null, mean_formula: null}
  bk.stoch.B_shot:
    {paper_mean: 0.0, paper_sigma: 1.0, paper_units: "unit",
     paper_variable: "B_shot*A_AP", factor: 1.0, offset: 1.0,
     mean: 1.0, sigma: 1.0, rescale: "A_AP",
     factor_formula: null, mean_formula: null}
  bk.stoch.A_shot:
    {paper_mean: 0.0, paper_sigma: 1.0, paper_units: "unit",
     paper_variable: "A_shot*A_AP", factor: 1.0, offset: 0.0,
     mean: 0.0, sigma: 1.0, rescale: "A_AP",
     factor_formula: null, mean_formula: null}
sampled:
  b1: {kind: flat}
  b2:
    {kind: gaussian, paper_mean: 0.0, paper_sigma: 5.0,
     paper_variable: "b2*sigma8(z)^2", rescale: "sigma8_sq"}
  bG2:
    {kind: gaussian, paper_mean: 0.0, paper_sigma: 5.0,
     paper_variable: "bG2*sigma8(z)^2", rescale: "sigma8_sq"}
""")


@pytest.fixture
def toy_spec_path(tmp_path):
    p = tmp_path / "toy_spec.yaml"
    p.write_text(TOY_YAML)
    return p


def test_toy_spec_loads_and_reconciles(toy_spec_path):
    spec = load_desi_prior_spec(toy_spec_path)
    assert isinstance(spec, DesiPriorSpec)
    assert set(spec.marginalized) == set(LIN_SURVEY_KEYS)
    row = spec.marginalized[("pk", "ctr", "c2")]
    assert row.mean == pytest.approx(30.0 * 0.5)
    assert row.sigma == pytest.approx(30.0 * 0.5)
    assert spec.marginalized[("shared", "stoch", "P_shot")].offset == 1.0
    assert spec.sampled["b1"].kind == "flat"


def _mutated_spec_path(tmp_path, mutate):
    """Parse TOY_YAML, apply ``mutate(raw_dict)``, dump to a temp file."""
    import yaml as _yaml
    raw = _yaml.safe_load(TOY_YAML)
    mutate(raw)
    p = tmp_path / "mutated.yaml"
    p.write_text(_yaml.safe_dump(raw))
    return p


def test_reconciliation_failure_raises(tmp_path):
    def mutate(raw):
        raw["marginalized"]["pk.ctr.c2"]["sigma"] = 14.0
    with pytest.raises(SpecValidationError, match="sigma"):
        load_desi_prior_spec(_mutated_spec_path(tmp_path, mutate))


def test_mean_reconciliation_includes_offset(tmp_path):
    def mutate(raw):
        raw["marginalized"]["shared.stoch.P_shot"]["mean"] = 0.0
    with pytest.raises(SpecValidationError, match="mean"):
        load_desi_prior_spec(_mutated_spec_path(tmp_path, mutate))


def test_missing_lin_key_raises(tmp_path):
    def mutate(raw):
        del raw["marginalized"]["bk.stoch.A_shot"]
    with pytest.raises(SpecValidationError, match="A_shot"):
        load_desi_prior_spec(_mutated_spec_path(tmp_path, mutate))


def test_unknown_rescale_token_raises(tmp_path):
    def mutate(raw):
        raw["marginalized"]["shared.bias.bGamma3"]["rescale"] = "A_AP^3"
    with pytest.raises(SpecValidationError, match="rescale"):
        load_desi_prior_spec(_mutated_spec_path(tmp_path, mutate))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_desi_priors.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jaxptpolypol.desi_priors'`.

- [ ] **Step 3: Implement the loader**

Create `src/jaxptpolypol/desi_priors.py`:

```python
"""DESI DR1-reanalysis (arXiv:2511.20757 Table I) prior spec machinery.

Layer-1 (constant coefficient-convention map) values are stored mapped AND
validated: each entry carries the verbatim paper value, the map factor (+
affine offset), and the our-convention value; loading raises unless they
reconcile. Layer-2 (theta_NL-dependent A_AP * A_amp rescaling, Table I
footnote) is applied at runtime by make_desi_prior_fns. Convention-map
provenance: docs/design/desi-convention-map.md. See CONTEXT.md
"Stream-B decisions (grill session 2026-07-30)".
"""
from __future__ import annotations

import importlib.resources
from dataclasses import dataclass

import yaml

__all__ = [
    "DesiPriorSpec", "MarginalRow", "SampledRow", "SpecValidationError",
    "load_desi_prior_spec",
]

_RESCALE_TOKENS = ("none", "A_AP", "A_AP*A_amp", "A_AP*A_amp^2")
_FACTOR_FORMULAS = (None, "knl_over_0p45_sq")
_MEAN_FORMULAS = (None, "coevolution_bGamma3")
_SAMPLED_RESCALE = ("none", "sigma8_sq")
_RECONCILE_RTOL = 1e-12


class SpecValidationError(ValueError):
    """Raised when a spec entry fails load-time reconciliation."""


@dataclass(frozen=True)
class MarginalRow:
    paper_mean: float | None
    paper_sigma: float
    paper_units: str
    paper_variable: str
    factor: float
    offset: float
    mean: float | None
    sigma: float
    rescale: str
    factor_formula: str | None
    mean_formula: str | None


@dataclass(frozen=True)
class SampledRow:
    kind: str
    paper_mean: float | None = None
    paper_sigma: float | None = None
    paper_variable: str | None = None
    rescale: str = "none"


@dataclass(frozen=True)
class DesiPriorSpec:
    metadata: dict
    marginalized: dict
    sampled: dict


def _close(a, b):
    return abs(a - b) <= _RECONCILE_RTOL * max(1.0, abs(a), abs(b))


def _validate_row(key, row):
    if row.rescale not in _RESCALE_TOKENS:
        raise SpecValidationError(
            f"{key}: unknown rescale token {row.rescale!r}")
    if row.factor_formula not in _FACTOR_FORMULAS:
        raise SpecValidationError(
            f"{key}: unknown factor_formula {row.factor_formula!r}")
    if row.mean_formula not in _MEAN_FORMULAS:
        raise SpecValidationError(
            f"{key}: unknown mean_formula {row.mean_formula!r}")
    if row.paper_sigma is None or row.paper_sigma <= 0.0:
        raise SpecValidationError(f"{key}: paper_sigma must be positive")
    if not _close(row.sigma, row.paper_sigma * abs(row.factor)):
        raise SpecValidationError(
            f"{key}: sigma {row.sigma} != paper_sigma*|factor| "
            f"{row.paper_sigma * abs(row.factor)}")
    if row.mean_formula is None:
        if row.paper_mean is None or row.mean is None:
            raise SpecValidationError(
                f"{key}: numeric mean required when mean_formula is null")
        if not _close(row.mean, row.paper_mean * row.factor + row.offset):
            raise SpecValidationError(
                f"{key}: mean {row.mean} != paper_mean*factor+offset "
                f"{row.paper_mean * row.factor + row.offset}")
    else:
        if row.mean is not None or row.paper_mean is not None:
            raise SpecValidationError(
                f"{key}: mean/paper_mean must be null with mean_formula")


def load_desi_prior_spec(name_or_path="desi_dr1_reanalysis_2511_20757"):
    """Load and validate a DESI prior spec (packaged name or explicit path)."""
    from .marginal_likelihood import LIN_SURVEY_KEYS

    path = str(name_or_path)
    if not path.endswith((".yaml", ".yml")):
        ref = importlib.resources.files("jaxptpolypol.data") / "priors" / f"{path}.yaml"
        raw = yaml.safe_load(ref.read_text())
    else:
        with open(path) as fh:
            raw = yaml.safe_load(fh)

    marginalized = {}
    for dotted, entry in raw["marginalized"].items():
        key = tuple(dotted.split("."))
        if len(key) != 3:
            raise SpecValidationError(f"bad marginalized key {dotted!r}")
        row = MarginalRow(**entry)
        _validate_row(dotted, row)
        marginalized[key] = row

    expected = set(LIN_SURVEY_KEYS)
    got = set(marginalized)
    if got != expected:
        missing = sorted(".".join(k) for k in expected - got)
        extra = sorted(".".join(k) for k in got - expected)
        raise SpecValidationError(
            f"marginalized keys mismatch: missing={missing} extra={extra}")

    sampled = {}
    for name, entry in raw["sampled"].items():
        row = SampledRow(**entry)
        if row.kind not in ("flat", "gaussian"):
            raise SpecValidationError(f"sampled {name}: bad kind {row.kind!r}")
        if row.rescale not in _SAMPLED_RESCALE:
            raise SpecValidationError(
                f"sampled {name}: unknown rescale {row.rescale!r}")
        if row.kind == "gaussian" and (row.paper_sigma is None
                                       or row.paper_sigma <= 0.0):
            raise SpecValidationError(
                f"sampled {name}: gaussian needs positive paper_sigma")
        sampled[name] = row
    for required in ("b1", "b2", "bG2"):
        if required not in sampled:
            raise SpecValidationError(f"sampled block missing {required!r}")

    return DesiPriorSpec(metadata=raw.get("metadata", {}),
                         marginalized=marginalized, sampled=sampled)
```

Add to `src/jaxptpolypol/__init__.py` exports (match the file's existing export style): `DesiPriorSpec`, `SpecValidationError`, `load_desi_prior_spec`.

- [ ] **Step 4: Run toy tests to verify they pass**

Run: `pytest tests/test_desi_priors.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Author the real spec + real-spec tests**

Copy the `## Spec rows (machine-readable)` YAML block from `docs/design/desi-convention-map.md` **verbatim** into `src/jaxptpolypol/data/priors/desi_dr1_reanalysis_2511_20757.yaml`. Add a `metadata` block: `source: "arXiv:2511.20757"`, `table: "I"`, `convention_map: "docs/design/desi-convention-map.md"`, `paper_knl: 0.45`, `production_k_nl_rsd: 0.45`, and a `deviations` list naming any documented deviations from the map's notes.

Append to `tests/test_desi_priors.py`:

```python
def test_real_spec_loads():
    spec = load_desi_prior_spec()
    assert set(spec.marginalized) == set(LIN_SURVEY_KEYS)
    assert spec.metadata["source"] == "arXiv:2511.20757"


def test_real_spec_verbatim_anchor_rows():
    """Rows recorded in CONTEXT.md from the primary PDF; if the reconciled
    map contradicts one of these, the MAP governs -- update CONTEXT.md and
    this test together in the same commit, quoting the paper."""
    spec = load_desi_prior_spec()
    c1 = spec.marginalized[("bk", "ctr", "c1")]
    assert c1.paper_mean == 0.0 and c1.paper_sigma == 5.0
    cfog = spec.marginalized[("pk", "ctr", "cfog")]
    assert cfog.paper_mean == 400.0 and cfog.paper_sigma == 400.0
    c2 = spec.marginalized[("pk", "ctr", "c2")]
    assert c2.paper_mean == 30.0
    for nm in ("b2", "bG2"):
        assert spec.sampled[nm].paper_sigma == 5.0
        assert spec.sampled[nm].rescale == "sigma8_sq"
    bg3 = spec.marginalized[("shared", "bias", "bGamma3")]
    assert bg3.mean_formula == "coevolution_bGamma3"
    for k in (("pk", "stoch", "a0"), ("pk", "stoch", "a2")):
        assert spec.marginalized[k].factor_formula == "knl_over_0p45_sq"
```

- [ ] **Step 6: Run all spec tests**

Run: `pytest tests/test_desi_priors.py -v`
Expected: 7 PASS. (If an anchor-row assertion fails, the reconciled map contradicts CONTEXT.md's record — re-read the map's provenance for that row; if the map quotes the paper, update CONTEXT.md and the test values in the same commit and say so in the commit message.)

- [ ] **Step 7: Commit**

```bash
git add src/jaxptpolypol/desi_priors.py src/jaxptpolypol/data/priors/desi_dr1_reanalysis_2511_20757.yaml tests/test_desi_priors.py src/jaxptpolypol/__init__.py
git commit -m "feat(priors): desi_dr1_reanalysis_2511_20757 spec + mapped-and-validated loader"
```

---

### Task 5: make_desi_prior_fns + make_lcdm_rescaling_fns

**Files:**
- Modify: `src/jaxptpolypol/desi_priors.py` (append)
- Modify: `tests/test_desi_priors.py` (append)
- Modify: `src/jaxptpolypol/__init__.py` (add exports)

**Interfaces:**
- Consumes: `DesiPriorSpec` (Task 4); `MarginalSplit` (has `.n_nl`, `.nl_b1_pos` tuple of per-bin b1 positions in θ_NL — b2 = b1_pos+1, bG2 = b1_pos+2); `LIN_SURVEY_KEYS` ordering; `sigma8_from_linear_pk` (`derived.py:58`); `CosmoParams.from_array`; `bg.Hz(omb, omc, h, z, mnu)` and `bg.angular_diameter_distance(omb, omc, h, z, mnu)` from `ps_1loop_jax.background`.
- Produces:
  - `make_desi_prior_fns(spec, *, split, knl_bins, sigma8_bins_fn, a_ap_bins_fn, sigma8_ref_bins) -> (prior_mean_fn, prior_sigma_fn, log_prior_nl_fn)` — the first two return `(n_lin,)` bin-major arrays; all three take the physical θ_NL vector.
  - `make_lcdm_rescaling_fns(*, pklin_emulator, cosmo_keys, cosmo_sizes, z_bins, fid_cosmo_native, mnu_fixed=0.06) -> (sigma8_bins_fn, a_ap_bins_fn, sigma8_ref_bins)` — closures mapping θ_NL → `(n_bins,)` arrays; `sigma8_ref_bins` is the fiducial σ8(z_b) (so A_amp(fid) = 1 exactly).

**Semantics (from Global Constraints):** paper prior 𝒩(m, s²) on `x·R` ⇒ prior on raw x is 𝒩(m/R, (s/R)²). Per bin b and θ_lin slot j:

```
R_b(θ) = 1                          if rescale == "none"
       = A_AP_b(θ)                  if rescale == "A_AP"
       = A_AP_b(θ)·A_amp_b(θ)       if rescale == "A_AP*A_amp"
       = A_AP_b(θ)·A_amp_b(θ)²      if rescale == "A_AP*A_amp^2"
A_amp_b(θ) = sigma8_bins_fn(θ)[b]² / sigma8_ref_bins[b]²
f_bj = (knl_bins[b]/0.45)²          if factor_formula == "knl_over_0p45_sq" else 1
mu_bj(θ)    = mean_bj(θ) / R_b(θ)         # mean_bj = row.mean, or 23/42·(b1_b−1)·row.factor + row.offset for coevolution_bGamma3
sigma_bj(θ) = row.sigma · f_bj / R_b(θ)
```

Sampled block: b1 flat (no term); b2/bG2 with `rescale: sigma8_sq`: log N(θ_i; 0, (paper_sigma/σ8_b(θ)²)²) **including the normalization term** (the width is θ-dependent).

- [ ] **Step 1: Write the failing factory tests (toy closures — exact analytic oracles)**

Append to `tests/test_desi_priors.py`:

```python
import jax.numpy as jnp

from jaxptpolypol.desi_priors import make_desi_prior_fns
from jaxptpolypol.marginal_likelihood import split_marginal_indices


N_COSMO = 2
N_BINS = 2
KNL_BINS = (0.52, 0.65)
SURVEY_KEYS_TOY = tuple(LIN_SURVEY_KEYS) + (
    ('shared', 'bias', 'b1'), ('shared', 'bias', 'b2'),
    ('shared', 'bias', 'bG2'))


@pytest.fixture
def toy_setup(toy_spec_path):
    spec = load_desi_prior_spec(toy_spec_path)
    split = split_marginal_indices(
        n_cosmo_params=N_COSMO, survey_keys=SURVEY_KEYS_TOY, n_bins=N_BINS)
    # theta_NL layout: [2 cosmo | (b1,b2,bG2) x 2 bins] -> n_nl = 8
    sigma8_ref = jnp.array([0.6, 0.5])

    def sigma8_bins_fn(theta_nl):
        # depends on theta so gradients/off-fiducial tests are meaningful
        return sigma8_ref * (1.0 + 0.1 * theta_nl[0])

    def a_ap_bins_fn(theta_nl):
        return jnp.ones(N_BINS) * (1.0 + 0.2 * theta_nl[1])

    fns = make_desi_prior_fns(
        spec, split=split, knl_bins=KNL_BINS,
        sigma8_bins_fn=sigma8_bins_fn, a_ap_bins_fn=a_ap_bins_fn,
        sigma8_ref_bins=sigma8_ref)
    return spec, split, fns


def test_prior_fns_shapes_and_fiducial_values(toy_setup):
    spec, split, (mean_fn, sigma_fn, _) = toy_setup
    theta0 = jnp.zeros(split.n_nl)          # fiducial: R = 1 everywhere
    mu, sig = mean_fn(theta0), sigma_fn(theta0)
    assert mu.shape == sig.shape == (N_BINS * len(LIN_SURVEY_KEYS),)
    j_c2 = LIN_SURVEY_KEYS.index(('pk', 'ctr', 'c2'))
    j_a0 = LIN_SURVEY_KEYS.index(('pk', 'stoch', 'a0'))
    j_ps = LIN_SURVEY_KEYS.index(('shared', 'stoch', 'P_shot'))
    n = len(LIN_SURVEY_KEYS)
    for b, knl in enumerate(KNL_BINS):
        assert sig[b * n + j_c2] == pytest.approx(15.0)
        assert sig[b * n + j_a0] == pytest.approx(1.0 * (knl / 0.45) ** 2)
        assert mu[b * n + j_ps] == pytest.approx(1.0)


def test_bGamma3_coevolution_mean(toy_setup):
    spec, split, (mean_fn, _, _) = toy_setup
    theta = jnp.zeros(split.n_nl)
    b1_vals = (1.7, 2.1)
    for pos, v in zip(split.nl_b1_pos, b1_vals):
        theta = theta.at[pos].set(v)
    mu = mean_fn(theta)
    j = LIN_SURVEY_KEYS.index(('shared', 'bias', 'bGamma3'))
    n = len(LIN_SURVEY_KEYS)
    row = spec.marginalized[('shared', 'bias', 'bGamma3')]
    for b, b1 in enumerate(b1_vals):
        expected = (23.0 / 42.0) * (b1 - 1.0) * row.factor + row.offset
        assert mu[b * n + j] == pytest.approx(expected)   # R=1 at theta cosmo=0


def test_layer2_rescaling_divides(toy_setup):
    spec, split, (mean_fn, sigma_fn, _) = toy_setup
    theta = jnp.zeros(split.n_nl).at[1].set(0.5)   # a_ap = 1.1, sigma8 unchanged
    a_ap = 1.0 + 0.2 * 0.5
    sig0 = sigma_fn(jnp.zeros(split.n_nl))
    sig = sigma_fn(theta)
    j_c0 = LIN_SURVEY_KEYS.index(('pk', 'ctr', 'c0'))    # rescale A_AP*A_amp
    j_ps = LIN_SURVEY_KEYS.index(('shared', 'stoch', 'P_shot'))  # rescale A_AP
    n = len(LIN_SURVEY_KEYS)
    assert sig[j_c0] == pytest.approx(sig0[j_c0] / a_ap)     # A_amp = 1 here
    assert sig[j_ps] == pytest.approx(sig0[j_ps] / a_ap)
    mu = mean_fn(theta)
    assert mu[j_ps] == pytest.approx(1.0 / a_ap)


def test_layer2_amp_powers(toy_setup):
    spec, split, (mean_fn, sigma_fn, _) = toy_setup
    theta = jnp.zeros(split.n_nl).at[0].set(0.5)   # sigma8 *= 1.05 -> A_amp = 1.05**2
    a_amp = 1.05 ** 2
    sig0, sig = sigma_fn(jnp.zeros(split.n_nl)), sigma_fn(theta)
    n = len(LIN_SURVEY_KEYS)
    j_c0 = LIN_SURVEY_KEYS.index(('pk', 'ctr', 'c0'))       # A_AP*A_amp
    j_bg3 = LIN_SURVEY_KEYS.index(('shared', 'bias', 'bGamma3'))  # A_AP*A_amp^2
    j_a0 = LIN_SURVEY_KEYS.index(('pk', 'stoch', 'a0'))     # A_AP only
    assert sig[j_c0] == pytest.approx(sig0[j_c0] / a_amp)
    assert sig[j_bg3] == pytest.approx(sig0[j_bg3] / a_amp ** 2)
    assert sig[j_a0] == pytest.approx(sig0[j_a0])            # A_amp-independent


def test_log_prior_nl_gaussian_and_flat(toy_setup):
    spec, split, (_, _, log_prior_nl) = toy_setup
    theta0 = jnp.zeros(split.n_nl)
    theta_b1 = theta0.at[split.nl_b1_pos[0]].set(3.0)
    assert log_prior_nl(theta_b1) == pytest.approx(float(log_prior_nl(theta0)))
    b2_pos = split.nl_b1_pos[0] + 1
    theta_b2 = theta0.at[b2_pos].set(1.0)
    sigma8_ref = 0.6
    w = 5.0 / sigma8_ref ** 2
    expected_delta = -0.5 * (1.0 / w) ** 2
    got = float(log_prior_nl(theta_b2) - log_prior_nl(theta0))
    assert got == pytest.approx(expected_delta)


def test_prior_fns_are_jit_and_grad_safe(toy_setup):
    spec, split, (mean_fn, sigma_fn, log_prior_nl) = toy_setup
    theta = jnp.full(split.n_nl, 0.1)
    assert jnp.allclose(jax.jit(mean_fn)(theta), mean_fn(theta))
    g = jax.grad(lambda t: jnp.sum(sigma_fn(t)) + log_prior_nl(t))(theta)
    assert jnp.all(jnp.isfinite(g))
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_desi_priors.py -k "prior_fns or bGamma3 or layer2 or log_prior" -v`
Expected: FAIL with `ImportError: cannot import name 'make_desi_prior_fns'`.

- [ ] **Step 3: Implement the factory + rescaling builder**

Append to `src/jaxptpolypol/desi_priors.py`:

```python
import jax.numpy as jnp

from .marginal_likelihood import LIN_SURVEY_KEYS

__all__ += ["make_desi_prior_fns", "make_lcdm_rescaling_fns"]

_LOG2PI = 1.8378770664093453


def _rescale_power(token):
    return {"none": (0, 0), "A_AP": (1, 0),
            "A_AP*A_amp": (1, 1), "A_AP*A_amp^2": (1, 2)}[token]


def make_desi_prior_fns(spec, *, split, knl_bins, sigma8_bins_fn,
                        a_ap_bins_fn, sigma8_ref_bins):
    """Build (prior_mean_fn, prior_sigma_fn, log_prior_nl_fn) from a spec.

    All three receive the physical theta_NL vector. The first two return
    (n_bins*11,) bin-major arrays laid out per LIN_SURVEY_KEYS. Layer-2
    rescaling divides by R_b(theta) per the Table-I footnote convention
    (prior on x*R => prior on x with mean m/R, width s/R).
    """
    n_bins = len(split.nl_b1_pos)
    n_lin_keys = len(LIN_SURVEY_KEYS)
    knl_arr = jnp.asarray(knl_bins, dtype=jnp.float64)
    if knl_arr.shape != (n_bins,):
        raise ValueError(f"knl_bins must have length {n_bins}")
    sigma8_ref = jnp.asarray(sigma8_ref_bins, dtype=jnp.float64)
    paper_knl = float(spec.metadata.get("paper_knl", 0.45))
    rows = [spec.marginalized[k] for k in LIN_SURVEY_KEYS]

    base_mean = jnp.array([0.0 if r.mean is None else r.mean for r in rows])
    base_sigma = jnp.array([r.sigma for r in rows])
    fac_knl = jnp.array([r.factor_formula == "knl_over_0p45_sq" for r in rows])
    coevo = jnp.array([r.mean_formula == "coevolution_bGamma3" for r in rows])
    coevo_factor = jnp.array([r.factor for r in rows])
    coevo_offset = jnp.array([r.offset for r in rows])
    ap_pow = jnp.array([_rescale_power(r.rescale)[0] for r in rows])
    amp_pow = jnp.array([_rescale_power(r.rescale)[1] for r in rows])
    b1_pos = jnp.asarray(split.nl_b1_pos)

    def _per_bin(theta_nl):
        theta_nl = jnp.asarray(theta_nl, dtype=jnp.float64)
        a_ap = a_ap_bins_fn(theta_nl)                       # (n_bins,)
        a_amp = sigma8_bins_fn(theta_nl) ** 2 / sigma8_ref ** 2
        R = (a_ap[:, None] ** ap_pow[None, :]
             * a_amp[:, None] ** amp_pow[None, :])          # (n_bins, 11)
        f_bin = jnp.where(fac_knl[None, :],
                          (knl_arr[:, None] / paper_knl) ** 2, 1.0)
        b1 = theta_nl[b1_pos]                               # (n_bins,)
        coevo_mean = ((23.0 / 42.0) * (b1[:, None] - 1.0) * coevo_factor[None, :]
                      + coevo_offset[None, :])
        mean = jnp.where(coevo[None, :], coevo_mean, base_mean[None, :])
        return (mean / R).reshape(-1), (base_sigma[None, :] * f_bin / R).reshape(-1)

    def prior_mean_fn(theta_nl):
        return _per_bin(theta_nl)[0]

    def prior_sigma_fn(theta_nl):
        return _per_bin(theta_nl)[1]

    gaussian_sampled = [(nm, spec.sampled[nm]) for nm in ("b2", "bG2")
                        if spec.sampled[nm].kind == "gaussian"]
    offsets = {"b2": 1, "bG2": 2}

    def log_prior_nl_fn(theta_nl):
        theta_nl = jnp.asarray(theta_nl, dtype=jnp.float64)
        s8 = sigma8_bins_fn(theta_nl)                       # (n_bins,)
        total = 0.0
        for nm, row in gaussian_sampled:
            pos = b1_pos + offsets[nm]
            width = (row.paper_sigma / s8 ** 2 if row.rescale == "sigma8_sq"
                     else jnp.full_like(s8, row.paper_sigma))
            x = theta_nl[pos] - (row.paper_mean or 0.0)
            total = total + jnp.sum(
                -0.5 * (x / width) ** 2 - jnp.log(width) - 0.5 * _LOG2PI)
        return total

    return prior_mean_fn, prior_sigma_fn, log_prior_nl_fn


def make_lcdm_rescaling_fns(*, pklin_emulator, cosmo_keys, cosmo_sizes,
                            z_bins, fid_cosmo_native, mnu_fixed=0.06):
    """Build (sigma8_bins_fn, a_ap_bins_fn, sigma8_ref_bins) closures.

    theta_NL[: n_cosmo] is interpreted as the native cosmology vector
    (all cosmology parameters sampled, the production LCDM layout).
    sigma8_ref_bins is sigma8(z_b) at fid_cosmo_native, so A_amp(fid) = 1
    exactly. Mirrors the emulator wiring of make_lcdm_derived_params_fn
    (derived.py:90-134); background from ps_1loop_jax (physical units).
    """
    from ps_1loop_jax import background as bg

    from .derived import _emulator_input_dict, sigma8_from_linear_pk
    from .params import CosmoParams

    n_cosmo = int(sum(cosmo_sizes))
    has_mnu = "mnu" in cosmo_keys
    z_arr = tuple(float(z) for z in z_bins)
    emulator_modes = jnp.asarray(pklin_emulator.modes, dtype=jnp.float64)
    emulator_parameters = getattr(pklin_emulator, "parameters", None)
    if emulator_parameters is not None:
        emulator_parameters = tuple(emulator_parameters)

    def _cosmo_obj(theta_nl):
        vec = jnp.asarray(theta_nl, dtype=jnp.float64)[:n_cosmo]
        return CosmoParams.from_array(vec, cosmo_keys, cosmo_sizes)

    def _bg_args(cosmo_obj):
        h = cosmo_obj.h[0]
        omb = cosmo_obj.omega_b[0]
        omc = cosmo_obj.omega_cdm[0]
        mnu = (cosmo_obj.mnu[0] if has_mnu
               else jnp.asarray(mnu_fixed, dtype=jnp.float64))
        return omb, omc, h, mnu

    def sigma8_bins_fn(theta_nl):
        cosmo_obj = _cosmo_obj(theta_nl)
        h = cosmo_obj.h[0]
        vals = []
        for z in z_arr:
            emulator_input = _emulator_input_dict(
                cosmo_obj, emulator_parameters=emulator_parameters,
                sigma8_redshift=z)
            pklin = jnp.ravel(jnp.asarray(
                pklin_emulator.predict(emulator_input), dtype=jnp.float64))
            vals.append(sigma8_from_linear_pk(emulator_modes / h, pklin))
        return jnp.stack(vals)

    fid_obj = CosmoParams.from_array(
        jnp.asarray(fid_cosmo_native, dtype=jnp.float64), cosmo_keys, cosmo_sizes)
    omb_f, omc_f, h_f, mnu_f = _bg_args(fid_obj)
    Hz_fid = jnp.array([float(bg.Hz(omb_f, omc_f, h_f, z, mnu_f)) for z in z_arr])
    DA_fid = jnp.array([float(bg.angular_diameter_distance(
        omb_f, omc_f, h_f, z, mnu_f)) for z in z_arr])
    H0_fid = 100.0 * float(h_f)

    def a_ap_bins_fn(theta_nl):
        omb, omc, h, mnu = _bg_args(_cosmo_obj(theta_nl))
        Hz = jnp.stack([bg.Hz(omb, omc, h, z, mnu) for z in z_arr])
        DA = jnp.stack([bg.angular_diameter_distance(omb, omc, h, z, mnu)
                        for z in z_arr])
        return (H0_fid / (100.0 * h)) ** 3 * (Hz / Hz_fid) * (DA_fid / DA) ** 2

    sigma8_ref_bins = sigma8_bins_fn(
        jnp.asarray(fid_cosmo_native, dtype=jnp.float64))
    return sigma8_bins_fn, a_ap_bins_fn, sigma8_ref_bins
```

Note: `__all__ +=` requires `__all__` to be a list — change the Task-4 `__all__` to a list if it was written as one already (it was). Export `make_desi_prior_fns`, `make_lcdm_rescaling_fns` from `src/jaxptpolypol/__init__.py`.

- [ ] **Step 4: Run the new tests**

Run: `pytest tests/test_desi_priors.py -v`
Expected: all PASS (13 total).

- [ ] **Step 5: Commit**

```bash
git add src/jaxptpolypol/desi_priors.py tests/test_desi_priors.py src/jaxptpolypol/__init__.py
git commit -m "feat(priors): make_desi_prior_fns with runtime A_AP*A_amp rescaling + LCDM rescaling closures"
```

---

### Task 6: Fisher consumption of the spec

**Files:**
- Modify: `src/jaxptpolypol/desi_priors.py` (append)
- Modify: `tests/test_desi_priors.py` (append)
- Modify: `src/jaxptpolypol/__init__.py`

**Interfaces:**
- Consumes: `DesiPriorSpec`; existing Fisher-side convention `build_prior_sigmas` (`inference.py`) accepts per-bin survey-prior dicts (list of dicts).
- Produces: `build_prior_sigmas_from_desi_spec(spec, *, knl_bins, sigma8_ref_bins) -> (survey_sigma_dicts, sampled_sigma_bins)` where `survey_sigma_dicts` is a list (one dict per bin) mapping `(section, group, key) -> sigma` at fiducial (R = 1, per-bin knl formula applied), and `sampled_sigma_bins` is a list of dicts `{"b2": 5/σ8_ref_b², "bG2": 5/σ8_ref_b²}` (fiducial widths for the sampled block; `b1` absent = no prior).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_desi_priors.py`:

```python
from jaxptpolypol.desi_priors import build_prior_sigmas_from_desi_spec


def test_fisher_sigmas_from_spec(toy_spec_path):
    spec = load_desi_prior_spec(toy_spec_path)
    sigma8_ref = np.array([0.6, 0.5])
    survey_dicts, sampled_bins = build_prior_sigmas_from_desi_spec(
        spec, knl_bins=KNL_BINS, sigma8_ref_bins=sigma8_ref)
    assert len(survey_dicts) == len(KNL_BINS)
    for b, knl in enumerate(KNL_BINS):
        assert survey_dicts[b][("pk", "ctr", "c2")] == pytest.approx(15.0)
        assert survey_dicts[b][("pk", "stoch", "a0")] == pytest.approx(
            (knl / 0.45) ** 2)
        assert survey_dicts[b][("bk", "ctr", "c1")] == pytest.approx(1.0125)
        assert sampled_bins[b]["b2"] == pytest.approx(5.0 / sigma8_ref[b] ** 2)
        assert "b1" not in sampled_bins[b]
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_desi_priors.py::test_fisher_sigmas_from_spec -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement**

Append to `src/jaxptpolypol/desi_priors.py`:

```python
__all__ += ["build_prior_sigmas_from_desi_spec"]


def build_prior_sigmas_from_desi_spec(spec, *, knl_bins, sigma8_ref_bins):
    """Fiducial (R = 1) per-bin prior widths for the Fisher side.

    Returns (survey_sigma_dicts, sampled_sigma_bins): one dict per bin for
    the marginalized survey block, and one dict per bin for the sampled
    bias block (b2/bG2 raw widths at sigma8_ref; flat b1 omitted).
    """
    paper_knl = float(spec.metadata.get("paper_knl", 0.45))
    survey_sigma_dicts = []
    sampled_sigma_bins = []
    for knl, s8_ref in zip(knl_bins, sigma8_ref_bins):
        d = {}
        for key, row in spec.marginalized.items():
            f_bin = ((knl / paper_knl) ** 2
                     if row.factor_formula == "knl_over_0p45_sq" else 1.0)
            d[key] = row.sigma * f_bin
        survey_sigma_dicts.append(d)
        sb = {}
        for nm in ("b2", "bG2"):
            row = spec.sampled[nm]
            if row.kind == "gaussian":
                sb[nm] = (row.paper_sigma / float(s8_ref) ** 2
                          if row.rescale == "sigma8_sq" else row.paper_sigma)
        sampled_sigma_bins.append(sb)
    return survey_sigma_dicts, sampled_sigma_bins
```

Export from `__init__.py`.

- [ ] **Step 4: Run all tests**

Run: `pytest tests/test_desi_priors.py -v`
Expected: 14 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/jaxptpolypol/desi_priors.py tests/test_desi_priors.py src/jaxptpolypol/__init__.py
git commit -m "feat(priors): Fisher-side fiducial widths from the DESI spec"
```

---

### Task 7: Acceptance gate — surrogate chain vs Hessian-Fisher under the new priors

**Files:**
- Create: `example/mcmc/scripts/desi_prior_validation.py`
- Modify: `docs/design/perbin-compile-measurements.md` (append results section)

**Interfaces:**
- Consumes: `load_desi_prior_spec`, `make_desi_prior_fns`, `make_lcdm_rescaling_fns` (Tasks 4–5); the existing Taylor-surrogate assembly in `example/mcmc/scripts/taylor_surrogate_validation.py` (its template/whitening loading around lines 220–260 and its posterior assembly around lines 320–340 are the pattern to reuse — same cache artifacts `cache/taylor_templates_lcdm.npz`, `cache/taylor_whitening_lcdm.npz`); `run_rwmh_python`, `make_cholesky_transform` from `jaxptpolypol.sampler`; the production config constants from `build_taylor_templates_lcdm.py` (`z_bins`, `knl_bins`, cosmo fiducial).
- Produces: `example/mcmc/cache/desi_prior_validation.json` with the gate verdict.

**Gate design (grill decision 6):** the surrogate marginal log-posterior embeds the spec priors, so "Fisher with the same spec" is `F = −∇²logpost(fid)` (`jax.hessian` on the 26-dim surrogate — cheap; Tier-1 established curvature == Fisher–Schur at fiducial). The mean check uses the AD-tilted center `μ_tilt = fid + F⁻¹∇logpost(fid)` (the gradient at fiducial now contains BOTH the logdet tilt and the non-fiducial prior-mean pulls — c2→30, c̃→400, bΓ3 coevolution, P_shot centering).

- [ ] **Step 1: Write the gate script**

`example/mcmc/scripts/desi_prior_validation.py` — structure (reuse the loading/assembly blocks from `taylor_surrogate_validation.py` verbatim where marked; only the prior construction and the gate logic are new):

```python
"""Stream-B acceptance gate: surrogate chain vs Hessian-Fisher, DESI priors.

Gates:
  G1  widths: chain posterior widths / Fisher widths in [0.9, 1.1] (cosmo block)
  G2  correlations: max |corr_chain - corr_Fisher| < 0.1 (cosmo block)
  G3  means: |chain mean - mu_tilt| < 2.5 * MC-SE + 0.05 sigma_F per cosmo param,
      mu_tilt = fid + F^{-1} grad logpost(fid)
Writes cache/desi_prior_validation.json. Uses the prior-independent Taylor
templates (no rebuild) -- see CONTEXT.md Stream-B decision 6.
"""
import json
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from jaxptpolypol.desi_priors import (
    load_desi_prior_spec, make_desi_prior_fns, make_lcdm_rescaling_fns)
from jaxptpolypol.marginal_taylor import (
    load_taylor_templates, make_marginal_log_posterior_taylor)
from jaxptpolypol.sampler import make_cholesky_transform, run_rwmh_python

# --- config: copy z_bins, knl_bins, cosmo fiducial, emulator setup, and the
# --- BAO/BBN/ns extra-term construction VERBATIM from
# --- taylor_surrogate_validation.py (they are the production config).

# --- load templates + whitening npz: copy the loading block from
# --- taylor_surrogate_validation.py (cache/taylor_templates_lcdm.npz,
# --- cache/taylor_whitening_lcdm.npz -> tt, fid_nl, cov_nl_prior, split, ...)

spec = load_desi_prior_spec()
sigma8_bins_fn, a_ap_bins_fn, sigma8_ref_bins = make_lcdm_rescaling_fns(
    pklin_emulator=pklin_emulator, cosmo_keys=cosmo_keys,
    cosmo_sizes=cosmo_sizes, z_bins=z_bins,
    fid_cosmo_native=fid_nl[:n_cosmo])
prior_mean_fn, prior_sigma_fn, log_prior_nl = make_desi_prior_fns(
    spec, split=split, knl_bins=knl_bins,
    sigma8_bins_fn=sigma8_bins_fn, a_ap_bins_fn=a_ap_bins_fn,
    sigma8_ref_bins=sigma8_ref_bins)

to_physical = make_cholesky_transform(center=fid_nl, cov=cov_nl_prior)
log_post = make_marginal_log_posterior_taylor(
    tt, bin_data=bin_data, bin_cov_invs=bin_cov_invs,
    prior_mean_fn=prior_mean_fn, prior_sigma_fn=prior_sigma_fn,
    log_prior_nl_fn=log_prior_nl, to_physical=to_physical,
    full_params_fn=full_params_fn, extra_theory_fn=extra_theory_fn,
    extra_data=extra_data, extra_cov_inv=extra_cov_inv)
log_post_j = jax.jit(log_post)

# --- Fisher + tilt from the surrogate itself (whitened coords)
w0 = jnp.zeros(fid_nl.shape[0])
H = jax.hessian(log_post)(w0)
g = jax.grad(log_post)(w0)
F_w = -np.asarray(H)
cov_F_w = np.linalg.inv(F_w)
mu_tilt_w = np.linalg.solve(F_w, np.asarray(g))          # fid=0 in whitened coords

# --- chain: 200_000 draws, seed 20260731, burn 20_000. Use EXACTLY the
# --- run_rwmh_python call signature and step-scale choice of
# --- taylor_surrogate_validation.py's chain block (copy it; change only
# --- the seed to 20260731 and the log-posterior to log_post_j).
t0 = time.time()
chain_w, acc = run_rwmh_python(  # signature per the transplanted block
    jax.random.PRNGKey(20260731), log_post_j, w0, num_samples=200_000)
wall = time.time() - t0
draws = np.asarray(chain_w)[20_000:]

# --- gates on the cosmology block (whitened positions 0..4; physical
# --- projection via chol as in the tier-2 analysis)
n_cosmo = 5
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

out = dict(width_ratio=width_ratio.tolist(), corr_diff_max=corr_diff_max,
           mean_pull_vs_tilted=mean_pull.tolist(), mc_se=mc_se.tolist(),
           ess=ess.tolist(), acceptance=float(acc), wall_s=wall,
           mu_tilt_w=np.asarray(mu_tilt_w).tolist(),
           gates=dict(G1_widths=g1, G2_corrs=g2, G3_means=g3),
           verdict="PASS" if (g1 and g2 and g3) else "REVIEW")
with open("example/mcmc/cache/desi_prior_validation.json", "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out["gates"], indent=1), "\nverdict:", out["verdict"])
```

The `# --- copy ... VERBATIM from taylor_surrogate_validation.py` markers are instructions to transplant those existing repo blocks (config, template/whitening loading, extra-term assembly, chain-drive parameters), keeping this script consistent with the validated pipeline; everything else above is complete as written. Resolve paths relative to the script's parent (`Path(__file__).resolve().parents[1]`) exactly as `taylor_surrogate_validation.py` does.

- [ ] **Step 2: Smoke-run the script at reduced size**

Temporarily set `num_samples=2_000`, burn 200 (do not commit these values) and run:
`cd example/mcmc && python scripts/desi_prior_validation.py`
Expected: completes in ~1–2 min after compile; JSON written; gates computed (REVIEW acceptable at this ESS — the check is that the machinery runs and mu_tilt is finite).

- [ ] **Step 3: Full run**

Restore `num_samples=200_000`, burn 20_000. Run again (~3–5 min).
Expected: verdict PASS. If REVIEW: check which gate — G1/G2 failures at ESS ≳ thousands indicate a real prior-wiring bug (inspect per-parameter width ratios against `build_prior_sigmas_from_desi_spec` values); G3 failure indicates the tilt prediction is off — verify `g` at fiducial is dominated by the expected prior-mean pulls before escalating.

- [ ] **Step 4: Record results**

Append a "Stream-B gate (DESI priors)" section to `docs/design/perbin-compile-measurements.md` with the gate table (width ratios, corr diff, mean pulls vs tilted center, ESS, acceptance, wall time) and one paragraph noting the templates were reused unmodified (prior-independence).

- [ ] **Step 5: Commit**

```bash
git add example/mcmc/scripts/desi_prior_validation.py example/mcmc/cache/desi_prior_validation.json docs/design/perbin-compile-measurements.md
git commit -m "feat(stream-b): acceptance gate -- surrogate chain vs Hessian-Fisher under DESI priors"
```

---

### Task 8: Notebook switchover

**Files:**
- Modify: `example/mcmc/mcmc_joint_PFS_BAO_BBN_ns_LCDM.ipynb`

**Interfaces:**
- Consumes: everything above; the notebook's existing variable names `prior_mean_fn`, `prior_sigma_fn`, `log_prior_nl` (downstream cells keep working if these names are preserved).

- [ ] **Step 1: Locate the prior-construction cell(s)**

Search the notebook for `make_constant_prior_fns` and the manual `mu_p`/`sigma_p`/`nl_prior_entries` construction (the same pattern as `taylor_surrogate_validation.py:228-330`). Also locate the imports cell.

- [ ] **Step 2: Replace prior construction with the spec-driven bundle**

Replace the located construction (keeping the SAME output variable names) with:

```python
from jaxptpolypol.desi_priors import (
    load_desi_prior_spec, make_desi_prior_fns, make_lcdm_rescaling_fns)

spec = load_desi_prior_spec()          # desi_dr1_reanalysis_2511_20757
sigma8_bins_fn, a_ap_bins_fn, sigma8_ref_bins = make_lcdm_rescaling_fns(
    pklin_emulator=pklin_emulator, cosmo_keys=cosmo_keys,
    cosmo_sizes=cosmo_sizes, z_bins=z_bins,
    fid_cosmo_native=fid_nl[:n_cosmo_params])
prior_mean_fn, prior_sigma_fn, log_prior_nl = make_desi_prior_fns(
    spec, split=split, knl_bins=knl_bins,
    sigma8_bins_fn=sigma8_bins_fn, a_ap_bins_fn=a_ap_bins_fn,
    sigma8_ref_bins=sigma8_ref_bins)
```

(Adapt only the local variable names on the right-hand sides — emulator, cosmo key tuple, z/knl tuples, fid vector — to what the notebook already defines; do not rename its outputs.) Keep the existing Fisher/whitening proposal (`make_cholesky_transform` on the existing `cov_nl_prior`) — the MH proposal need not match the target priors; add a one-line markdown note saying so. Add a markdown cell citing the spec name, `docs/design/desi-convention-map.md`, and CONTEXT.md decisions 1–7.

- [ ] **Step 3: Smoke-run the notebook**

Run the notebook in SMOKE mode (its existing SMOKE branch: 60 steps / 10 burn) end-to-end:
`cd example/mcmc && jupyter nbconvert --to notebook --execute --inplace mcmc_joint_PFS_BAO_BBN_ns_LCDM.ipynb` (with the SMOKE flag set as the notebook's first cell defines).
Expected: completes; acceptance > 0; posterior cells render. Note: with the new prior means (c2→30 etc.) the SMOKE posterior summaries will shift relative to the previous run — that is expected, not a bug.

- [ ] **Step 4: Commit**

```bash
git add example/mcmc/mcmc_joint_PFS_BAO_BBN_ns_LCDM.ipynb
git commit -m "feat(notebook): switch LCDM MCMC notebook to desi_dr1_reanalysis_2511_20757 priors"
```

---

## Self-Review Notes

- Spec coverage: grill decision 1 → Tasks 1–3; decision 2 → Task 4; decision 3 → Tasks 5–6; decision 4 → factor_formula machinery (Tasks 4–6); decision 5 → Task 3 Step 4; decision 6 → Task 7; decision 7 → Task 8 (+ exclusions honored).
- The toy fixture's numbers (c1 factor 0.2025, P_shot offset 1, c2 factor 0.5) are FIXTURE values exercising the machinery, deliberately including a non-trivial factor; the real spec's numbers come only from the Task-3 appendix.
- Anchor-row tests encode CONTEXT.md's paper-verified values; the escape hatch (map governs, update both in one commit) is stated in the test docstring.
- Type consistency: `make_desi_prior_fns(spec, *, split, knl_bins, sigma8_bins_fn, a_ap_bins_fn, sigma8_ref_bins)` used identically in Tasks 5, 7, 8; loader name `load_desi_prior_spec` throughout; `MarginalSplit.nl_b1_pos` is the only split field consumed.

---

## Amendment 1 (2026-07-31): two-branch resolution of the ctr-basis escalation

Task 3 confirmed (CLASS-PT Eqs 2.21–2.23, quoted in `docs/design/desi-convention-map.md`
§3.1) that Table I's c0/c2/c4 priors live on the per-multipole basis while our code's
coefficients are the μ-space tilde basis, related per bin by the upper-triangular
`L(f) = [[1, −f/3, 3f²/35], [0, 1, −6f/7], [0, 0, 1]]` mapping (c0,c2,c4)_paper →
(c0,c2,c4)_ours. USER DECISION: implement BOTH exact representations on two branches;
their gates plus a machine-precision equivalence test decide the merge. The marginal
likelihood (including ln det(AΣ_p)) is exactly invariant under a linear θ_lin
reparameterization with consistently transformed priors, so the two branches must agree
to float64 precision — this equivalence is the primary cross-validation.

**Branches** (both from master after 6a6d5d7):
- `stream-b-sigmap` — option 1: full per-bin prior covariance Σ_p.
- `stream-b-rotation` — option 3: θ_lin ctr slots redefined to the paper basis via L(f).

**Worktree mechanics (both branches):**
- Worktrees: `/Users/nguyenmn/jaxPTPolyPol-sigmap`, `/Users/nguyenmn/jaxPTPolyPol-rotation`.
- Tests MUST run with `PYTHONPATH=<worktree>/src` prepended — the editable install points
  at the master checkout and would silently shadow the branch code.
- `example/mcmc/cache` is untracked and absent in worktrees: each worktree gets a symlink
  to `/Users/nguyenmn/jaxPTPolyPol/example/mcmc/cache` (created at branch setup). Branch
  outputs use per-branch filenames: `desi_prior_validation_<branch>.json`,
  `branch_equiv_<branch>.json`.

**Shared constants (both branches, verbatim):**
- `F_FID = (0.8155, 0.8579, 0.8893, 0.9126, 0.9301, 0.9489, 0.9649)` for
  `z_bins = (0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.2)` — from the map §3.1 table. Each branch
  must also verify `ps_1loop_jax.background.growth_rate_approx(omb, omc, h, z, 0.06)`
  at the production fiducial (ombh2=0.02242, omch2=0.11933, h=0.6766) reproduces these
  within 2e-3 and report if not (the hardcoded tuple governs either way — cross-branch
  identity of L is what the equivalence test needs).
- ctr slot positions within a bin's 11 θ_lin entries: indices (2, 3, 4) per
  LIN_SURVEY_KEYS (bGamma3, P_shot, c0, c2, c4, cfog, a0, a2, c1, B_shot, A_shot).
- Layer-2: R_b(θ) (rescale A_AP*A_amp) divides PAPER-basis quantities in both branches:
  μ_paper_b = (0, 30, 0)/R_b; σ_paper_b = 30/R_b for all three rows.

**Branch `stream-b-sigmap` contract (Tasks 4σ–7σ = Tasks 4–7 with these deltas):**
- Spec: c0/c2/c4 rows keep paper values verbatim (factor 1, offset 0) and gain a new
  optional field `ctr_rotation: "multipole_to_tilde"` (vocabulary {null,
  multipole_to_tilde}); loader validates the trio carries it all-or-none. PROVISIONAL
  markers removed on this branch (the representation is exact).
- `gaussian_marginal_loglike`: `sigma_p` accepts ndim==1 (diag widths, current behavior,
  backward compatible) or ndim==2 (full Σ_p): then `A = MᵀC⁻¹M + Σ_p⁻¹` via Cholesky
  solve and `ln det Σ_p` via `2·sum(log(diag(chol(Σ_p))))`. All existing tests must stay
  green unchanged.
- `make_marginal_log_posterior_perbin` and `_taylor`: `prior_sigma_fn(θ)` may return
  `(n_lin,)` (current) or `(n_bins, 11, 11)` stacked per-bin blocks; per-bin consumption
  takes `[b]`. `prior_mean_fn` unchanged `(n_lin,)`.
- `make_desi_prior_fns`: when the spec carries ctr_rotation, returns cov-mode
  `prior_sigma_fn` building per bin: start from the diagonal entries as today, then
  overwrite the (2:5, 2:5) block with `L(F_FID[b]) · diag(σ_paper_b²) · L(F_FID[b])ᵀ`
  and set `prior_mean_fn` ctr entries to `L(F_FID[b]) · μ_paper_b`. New toy tests:
  cov-mode block matches the analytic L·Σ·Lᵀ oracle at and off fiducial; diag-mode
  path bit-identical to before.
- Fisher (Task 6σ): `build_prior_sigmas_from_desi_spec` emits the marginal widths
  `sqrt(diag(LΣLᵀ))` for the ctr trio (documented: legacy-Fisher consumers are
  diagonal-only; the gate's Hessian-Fisher carries the full block).
- Gate (Task 7σ): as Task 7, writing `desi_prior_validation_sigmap.json`, PLUS dump
  `branch_equiv_sigmap.json`: `{"points_seed": 20260731, "n": 64, "scale": 0.5,
  "log_post": [...]}` — 64 whitened points `0.5 * jax.random.normal(PRNGKey(20260731),
  (64, n_nl))`, log-posterior at each (include_logdet=True).

**Branch `stream-b-rotation` contract (Tasks 4r–7r = Tasks 4–7 with these deltas):**
- Spec: c0/c2/c4 rows are the paper's diagonal VERBATIM (factor 1, offset 0, rescale
  A_AP*A_amp) with `metadata.ctr_basis: "multipole"`; θ_lin keys pk.ctr.{c0,c2,c4} now
  MEAN the per-multipole coefficients on this branch. PROVISIONAL markers removed.
- New helper in `desi_priors.py`: `ctr_rotation_matrices(f_bins)` returning the stacked
  `(n_bins, 3, 3)` L(f) matrices, and `rotate_taylor_templates(tt, L_bins)` returning a
  new TaylorTemplates with, for each bin b, columns (2,3,4) of `bin_M0[b]` and of
  `bin_dM[b]` right-multiplied by `L(F_FID[b])` (m0/J/H untouched — they live at
  θ_lin=0, which L fixes). Unit test: rotating then evaluating the surrogate marginal
  with the diagonal paper prior == evaluating the UNrotated surrogate with the
  L-pushforward correlated prior, to 1e-10 (this is the invariance identity in-branch).
- Exact perbin path: wrap each bin theory fn so the inserted ctr slots are
  `L(F_FID[b]) @ θ_paper[2:5]` (identity elsewhere) — one wrapper in the notebook/script
  assembly, NOT a theory.py change.
- `make_desi_prior_fns`: diagonal machinery unchanged from the pre-amendment plan
  (the paper prior applies verbatim); no cov-mode.
- Fisher (Task 6r): paper diagonal verbatim.
- Gate (Task 7r): as Task 7 on rotated templates, writing
  `desi_prior_validation_rotation.json` + `branch_equiv_rotation.json` (same seed/spec
  as σp — the same 64 whitened points).

**Task E (controller-level, after both 7σ and 7r):** compare the two
`branch_equiv_*.json`: require `max|Δ log_post| < 1e-5` (expect ~1e-9); compare the two
gate JSONs (same verdicts, width ratios within MC noise). Record in the measurement doc.
Winner selection is a user decision informed by both gates + code-review simplicity;
Task 8 (notebook switchover) runs only on the winning branch after merge.
