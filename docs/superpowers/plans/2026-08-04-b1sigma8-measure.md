# b1σ8 Prior-Measure Package (F1 + D + E + F2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the b1 prior measure an explicit, switchable, cross-validated, and gated choice — spec field + Jacobian term (F1), importance-reweighting machinery and a measured chain-level shift (D), a flag-ON ≡ reweighted cross-check (E), and a loader-level phase gate for real-data/nuLCDM (F2) — with the default (`raw`) behavior bit-identical to today.

**Architecture:** The paper (arXiv:2511.20757 Table I) samples y_b = b1·σ8(z_b) ~ 𝒰[0,3]; we sample raw b1 flat. The two measures differ by the prior weight Π_b σ8(z_b; θ) plus cosmology-dependent bounds. The spec's b1 row gains a `measure` field; `make_desi_prior_fns` adds `Σ_b log σ8` (+ bounds) to `log_prior_nl_fn` when `measure: b1sigma8`; a weights helper converts raw-measure chains post-hoc; a cross-check script proves flag-ON ≡ reweighted pointwise and at chain level; the loader refuses `measure: raw` in real-data/nuLCDM phases.

**Tech Stack:** JAX (float64), PyYAML, existing desi_priors/sampler/marginal machinery. No new dependencies.

## Global Constraints

- Default-path behavior MUST be bit-identical: suite baseline **155 passed, 15 deselected** stays green with no pre-existing test edited; the wiring tripwire `log_post(x0) = -172.996046` (smoke gate `DESI_GATE_SMOKE=1 SMOKE=1 python3 scripts/desi_prior_validation.py`) is unchanged.
- Do NOT touch `src/jaxptpolypol/cmb.py`, `tests/test_cmb_priors.py` (uncommitted user WIP), or any `.ipynb`.
- The b2/bG2 rows already carry their σ8² Jacobian via the Gaussian `-log(width)` normalization — do NOT add anything for them; the new term is for b1 only, power **1**.
- Known reference numbers (verified this session, cite in docs/tests): gradient `d(Σ_b log σ8)/dθ = (-65.385, +28.404, +3.500, +2.650, +6.155)` for (ombh2, omch2, logA, ns, h) with `d/dlogA = n_bins/2 = 3.5` exact; predicted full shift `F⁻¹g = (-0.006, -0.028, +0.172, +0.097, -0.016) σ_F` (DESI-Hessian units); predicted reweighting efficiency `ESS/N ≈ exp(-gᵀCg) ≈ 0.973`; fiducial σ8(z_b) = (0.5660, 0.5158, 0.4727, 0.4356, 0.4034, 0.3628, 0.3194); fiducial b1σ8 = 0.57–0.67 (bounds [0,3] non-binding).
- Flag-ON chains use RWMH/DA (the [0,3] walls are hostile to NUTS — the recorded CONTEXT.md rationale; do not run NUTS with `measure: b1sigma8`).
- `jax.config.update("jax_enable_x64", True)` in every test/script.
- Commit with explicit paths only (never `git add -A`); commit as soon as each task is green.

## File Structure

| File | Change |
|---|---|
| `src/jaxptpolypol/desi_priors.py` | `SampledRow.measure/paper_lower/paper_upper`; loader vocab + phase gate; Jacobian+bounds in `log_prior_nl_fn`; `b1sigma8_log_weights` helper |
| `src/jaxptpolypol/data/priors/desi_dr1_reanalysis_2511_20757.yaml` | b1 row fields + `metadata.deviations` entry (option H) |
| `tests/test_desi_priors.py` | all new unit tests (append-only) |
| `example/mcmc/scripts/desi_prior_validation.py` | save the post-burn chain (non-smoke) |
| `example/mcmc/scripts/b1sigma8_crosscheck.py` (new) | option E: pointwise identity + short-chain cross-check |
| `example/mcmc/scripts/b1sigma8_measure_report.py` (new) | option D: reweight the production chain, report both measures |
| `CONTEXT.md`, `docs/design/perbin-compile-measurements.md` | corrected deviation record + measured-shift section |

---

### Task 1: Spec field + loader vocabulary + phase gate (F1 spec half + H + F2)

**Files:**
- Modify: `src/jaxptpolypol/desi_priors.py` (SampledRow ~line 60, vocab constants ~line 34, `load_desi_prior_spec` ~line 112, sampled-row validation ~line 162)
- Modify: `src/jaxptpolypol/data/priors/desi_dr1_reanalysis_2511_20757.yaml` (b1 row line 85 + metadata)
- Test: `tests/test_desi_priors.py` (append)

**Interfaces:**
- Consumes: existing `SampledRow(kind, paper_mean, paper_sigma, paper_variable, rescale)`, `load_desi_prior_spec(name_or_path)`, `SpecValidationError`, the tests' `TOY_YAML` + `_mutated_spec_path` helper.
- Produces (later tasks rely on these exact names): `SampledRow` gains `measure: str = "raw"`, `paper_lower: float | None = None`, `paper_upper: float | None = None`; module constants `_B1_MEASURES = ("raw", "b1sigma8")`, `_PHASES = ("forecast", "real_data", "nulcdm")`; `load_desi_prior_spec(name_or_path=..., phase="forecast")`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_desi_priors.py`:

```python
# ---------------------------------------------------------------------------
# b1 sigma8 measure (F1) + phase gate (F2): spec/loader layer
# ---------------------------------------------------------------------------

def test_b1_measure_defaults_raw(toy_spec_path):
    spec = load_desi_prior_spec(toy_spec_path)
    assert spec.sampled["b1"].measure == "raw"
    assert spec.sampled["b1"].paper_lower is None


def test_b1_measure_b1sigma8_loads_with_bounds(tmp_path):
    def mutate(raw):
        raw["sampled"]["b1"] = {"kind": "flat", "measure": "b1sigma8",
                                "paper_lower": 0.0, "paper_upper": 3.0,
                                "paper_variable": "b1*sigma8(z)"}
    spec = load_desi_prior_spec(_mutated_spec_path(tmp_path, mutate))
    assert spec.sampled["b1"].measure == "b1sigma8"
    assert spec.sampled["b1"].paper_upper == 3.0


def test_b1_measure_bad_token_raises(tmp_path):
    def mutate(raw):
        raw["sampled"]["b1"] = {"kind": "flat", "measure": "b1_sigma_8"}
    with pytest.raises(SpecValidationError, match="measure"):
        load_desi_prior_spec(_mutated_spec_path(tmp_path, mutate))


def test_b1sigma8_requires_both_bounds(tmp_path):
    def mutate(raw):
        raw["sampled"]["b1"] = {"kind": "flat", "measure": "b1sigma8",
                                "paper_lower": 0.0}
    with pytest.raises(SpecValidationError, match="paper_upper"):
        load_desi_prior_spec(_mutated_spec_path(tmp_path, mutate))


def test_measure_on_non_b1_row_raises(tmp_path):
    def mutate(raw):
        raw["sampled"]["b2"]["measure"] = "b1sigma8"
    with pytest.raises(SpecValidationError, match="only the b1 row"):
        load_desi_prior_spec(_mutated_spec_path(tmp_path, mutate))


def test_phase_gate_blocks_raw_measure(toy_spec_path):
    """F2: real-data / nuLCDM phases refuse the raw measure (CONTEXT.md)."""
    for phase in ("real_data", "nulcdm"):
        with pytest.raises(SpecValidationError, match="measure"):
            load_desi_prior_spec(toy_spec_path, phase=phase)
    load_desi_prior_spec(toy_spec_path, phase="forecast")   # default path OK


def test_phase_gate_passes_with_b1sigma8(tmp_path):
    def mutate(raw):
        raw["sampled"]["b1"] = {"kind": "flat", "measure": "b1sigma8",
                                "paper_lower": 0.0, "paper_upper": 3.0}
    p = _mutated_spec_path(tmp_path, mutate)
    load_desi_prior_spec(p, phase="real_data")               # no raise


def test_unknown_phase_raises(toy_spec_path):
    with pytest.raises(SpecValidationError, match="phase"):
        load_desi_prior_spec(toy_spec_path, phase="production")


def test_real_spec_b1_row_and_deviation_note():
    spec = load_desi_prior_spec()
    b1 = spec.sampled["b1"]
    assert b1.measure == "raw" and b1.paper_lower == 0.0 and b1.paper_upper == 3.0
    devs = " ".join(str(d) for d in spec.metadata.get("deviations", []))
    assert "b1" in devs and "measure" in devs and "sigma8" in devs.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_desi_priors.py -k "b1_measure or phase_gate or unknown_phase or b1_row_and_deviation" -v`
Expected: FAIL (`TypeError: unexpected keyword 'phase'`, missing-attribute/field errors).

- [ ] **Step 3: Implement the loader layer**

In `src/jaxptpolypol/desi_priors.py`:

Near line 34, alongside `_SAMPLED_RESCALE`:

```python
_B1_MEASURES = ("raw", "b1sigma8")
_PHASES = ("forecast", "real_data", "nulcdm")
```

Extend `SampledRow` (line ~60):

```python
@dataclass(frozen=True)
class SampledRow:
    kind: str
    paper_mean: float | None = None
    paper_sigma: float | None = None
    paper_variable: str | None = None
    rescale: str = "none"
    #: b1 only -- which coordinate the flat prior is flat IN. "raw": flat in
    #: raw b1 (project default; differs from the paper's measure by the
    #: cosmology-dependent weight prod_b sigma8(z_b) -- see CONTEXT.md
    #: deviation 3). "b1sigma8": flat in y = b1*sigma8(z) on
    #: [paper_lower, paper_upper], the Table-I measure (adds the Jacobian
    #: sum_b log sigma8 and the bounds to log_prior_nl_fn).
    measure: str = "raw"
    paper_lower: float | None = None
    paper_upper: float | None = None
```

Change the signature at line ~112 and add validation inside `load_desi_prior_spec` (in the sampled-row loop, next to the existing `rescale` check at ~162, and after the loop for the phase gate):

```python
def load_desi_prior_spec(name_or_path="desi_dr1_reanalysis_2511_20757",
                         phase="forecast"):
```

```python
        if row.measure not in _B1_MEASURES:
            raise SpecValidationError(
                f"sampled {name}: unknown measure {row.measure!r} "
                f"(allowed: {_B1_MEASURES})")
        if name != "b1" and row.measure != "raw":
            raise SpecValidationError(
                f"sampled {name}: 'measure' applies to only the b1 row")
        if row.measure == "b1sigma8":
            if row.paper_lower is None or row.paper_upper is None:
                raise SpecValidationError(
                    "sampled b1: measure=b1sigma8 requires numeric "
                    "paper_lower and paper_upper")
            if not row.paper_lower < row.paper_upper:
                raise SpecValidationError(
                    "sampled b1: paper_lower must be < paper_upper")
```

```python
    if phase not in _PHASES:
        raise SpecValidationError(f"unknown phase {phase!r} (allowed: {_PHASES})")
    if phase != "forecast" and sampled["b1"].measure == "raw":
        raise SpecValidationError(
            f"phase={phase!r} requires the Table-I b1 measure: set the spec's "
            "b1 row to measure: b1sigma8 (raw-b1 flat differs from "
            "arXiv:2511.20757 by the prod_b sigma8(z_b) prior weight, which "
            "lands on Sum m_nu in nuLCDM -- see CONTEXT.md deviation 3)")
```

Update the packaged YAML b1 row (line 85) and metadata:

```yaml
  b1: {kind: flat, measure: raw, paper_lower: 0.0, paper_upper: 3.0,
       paper_variable: "b1*sigma8(z)"}
```

and append to `metadata.deviations`:

```yaml
  - >
    b1 measure: the paper samples b1*sigma8(z) ~ U[0,3]; this spec's default
    measure: raw samples raw b1 flat/unbounded, which differs by the prior
    weight prod_b sigma8(z_b) (a cosmology tilt: +0.17 sigma_F on logA,
    +0.10 on ns for the 7-bin PFS forecast; widths unaffected). Set
    measure: b1sigma8 for the Table-I measure; loader refuses measure: raw
    for phase real_data/nulcdm. Evidence: cache/b1sigma8_measure.json.
```

- [ ] **Step 4: Run the new tests + full suite**

Run: `pytest tests/test_desi_priors.py -v` then `pytest tests/`
Expected: new tests PASS; full suite = 155 + 9 new, 15 deselected, nothing pre-existing broken.

- [ ] **Step 5: Commit**

```bash
git add src/jaxptpolypol/desi_priors.py src/jaxptpolypol/data/priors/desi_dr1_reanalysis_2511_20757.yaml tests/test_desi_priors.py
git commit -m "feat(priors): b1 measure spec field (raw|b1sigma8) + real-data/nuLCDM phase gate"
```

---

### Task 2: Jacobian + bounds in `log_prior_nl_fn` (F1 runtime half)

**Files:**
- Modify: `src/jaxptpolypol/desi_priors.py` (`make_desi_prior_fns` — the `log_prior_nl_fn` closure, currently ~lines 483–506; `b1_pos` is already bound at ~line 375, `s8 = sigma8_bins_fn(theta_nl)` already computed at ~line 485)
- Test: `tests/test_desi_priors.py` (append)

**Interfaces:**
- Consumes: `spec.sampled["b1"].measure/.paper_lower/.paper_upper` (Task 1); the existing closure variables `b1_pos`, `sigma8_bins_fn`.
- Produces: flag-ON `log_prior_nl_fn` satisfying, at every θ_NL with b1σ8 inside the bounds, the exact identity `log_prior_ON(θ) − log_prior_OFF(θ) = Σ_b log σ8(z_b; θ)`; `-inf` outside the bounds. Tasks 4–5 rely on this identity.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_desi_priors.py` (reuses the existing `toy_setup`-style closures: 2 bins, `sigma8_bins_fn = ref*(1+0.1*θ[0])` with `ref = (0.6, 0.5)`):

```python
# ---------------------------------------------------------------------------
# b1 sigma8 measure: runtime Jacobian + bounds
# ---------------------------------------------------------------------------

def _fns_for_measure(tmp_path, measure):
    def mutate(raw):
        if measure == "b1sigma8":
            raw["sampled"]["b1"] = {"kind": "flat", "measure": "b1sigma8",
                                    "paper_lower": 0.0, "paper_upper": 3.0}
    spec = load_desi_prior_spec(_mutated_spec_path(tmp_path, mutate))
    split = split_marginal_indices(
        n_cosmo_params=N_COSMO, survey_keys=SURVEY_KEYS_TOY, n_bins=N_BINS)
    sigma8_ref = jnp.array([0.6, 0.5])
    s8_fn = lambda t: sigma8_ref * (1.0 + 0.1 * t[0])
    a_ap_fn = lambda t: jnp.ones(N_BINS) * (1.0 + 0.2 * t[1])
    fns = make_desi_prior_fns(
        spec, split=split, knl_bins=KNL_BINS, sigma8_bins_fn=s8_fn,
        a_ap_bins_fn=a_ap_fn, sigma8_ref_bins=sigma8_ref)
    return spec, split, s8_fn, fns


def test_b1sigma8_jacobian_pointwise_identity(tmp_path):
    """log_prior_ON - log_prior_OFF == sum_b log sigma8(z_b; theta), exactly."""
    _, split, s8_fn, (_, _, lp_off) = _fns_for_measure(tmp_path, "raw")
    _, _, _, (_, _, lp_on) = _fns_for_measure(tmp_path, "b1sigma8")
    rng = np.random.default_rng(20260804)
    for _ in range(8):
        theta = jnp.asarray(rng.normal(0.0, 0.3, size=split.n_nl))
        # keep b1*sigma8 inside [0, 3]: set b1 slots to ~1.5
        for p in split.nl_b1_pos:
            theta = theta.at[p].set(1.5)
        expected = float(jnp.sum(jnp.log(s8_fn(theta))))
        got = float(lp_on(theta)) - float(lp_off(theta))
        assert got == pytest.approx(expected, abs=1e-12)


def test_b1sigma8_bounds_give_minus_inf(tmp_path):
    _, split, _, (_, _, lp_on) = _fns_for_measure(tmp_path, "b1sigma8")
    theta = jnp.zeros(split.n_nl)
    ok = theta.at[split.nl_b1_pos[0]].set(1.0)          # y = 0.6 in [0,3]
    bad_hi = theta.at[split.nl_b1_pos[0]].set(6.0)      # y = 3.6 > 3
    bad_lo = theta.at[split.nl_b1_pos[0]].set(-0.5)     # y < 0
    assert np.isfinite(float(lp_on(ok)))
    assert float(lp_on(bad_hi)) == -np.inf
    assert float(lp_on(bad_lo)) == -np.inf


def test_b1sigma8_gradient_slope(tmp_path):
    """d(Jacobian)/d theta0 = sum_b d log s8/d theta0 = sum_b 0.1/(1+0.1 t0)."""
    _, split, _, (_, _, lp_off) = _fns_for_measure(tmp_path, "raw")
    _, _, _, (_, _, lp_on) = _fns_for_measure(tmp_path, "b1sigma8")
    theta = jnp.zeros(split.n_nl)
    for p in split.nl_b1_pos:
        theta = theta.at[p].set(1.5)
    diff = lambda t: lp_on(t) - lp_off(t)
    g = jax.grad(diff)(theta)
    assert float(g[0]) == pytest.approx(N_BINS * 0.1, rel=1e-10)
    assert float(g[split.nl_b1_pos[0]]) == pytest.approx(0.0, abs=1e-12)


def test_raw_measure_bitwise_unchanged(tmp_path):
    """Default path must be BIT-identical to the pre-change behavior."""
    _, split, _, (mu_a, sig_a, lp_a) = _fns_for_measure(tmp_path, "raw")
    theta = jnp.full(split.n_nl, 0.2)
    # raw measure adds no term and no bounds:
    assert np.isfinite(float(lp_a(theta.at[split.nl_b1_pos[0]].set(50.0))))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_desi_priors.py -k "b1sigma8 or raw_measure_bitwise" -v`
Expected: the identity/bounds/gradient tests FAIL (ON and OFF priors identical); `raw_measure_bitwise` PASSES (it pins the invariant).

- [ ] **Step 3: Implement**

Inside `make_desi_prior_fns`, before the closure definitions, read the row once (static):

```python
    b1_row = spec.sampled["b1"]
    b1_measure = b1_row.measure
    if b1_measure == "b1sigma8":
        b1_lower = float(b1_row.paper_lower)
        b1_upper = float(b1_row.paper_upper)
```

Inside `log_prior_nl_fn`, immediately after `s8 = sigma8_bins_fn(theta_nl)` (line ~485):

```python
        if b1_measure == "b1sigma8":
            # Table-I measure: flat in y_b = b1_b * sigma8(z_b) on
            # [b1_lower, b1_upper]. Relative to flat-in-raw-b1 this adds the
            # change-of-variables Jacobian sum_b log sigma8 (a LIVE cosmology
            # tilt: d/dlogA = n_bins/2) plus the bounds indicator. b2/bG2 need
            # nothing here -- their Gaussian -log(width) already carries the
            # sigma8^2 Jacobian.
            total = total + jnp.sum(jnp.log(s8))
            y = theta_nl[b1_pos] * s8
            inside = jnp.all((y >= b1_lower) & (y <= b1_upper))
            total = total + jnp.where(inside, 0.0, -jnp.inf)
```

(`total` starts at `0.0` before the b2/bG2 loop; place this after the loop so `s8` is in scope and the raw path is untouched by construction — the `if` is a Python-level static branch on the spec, not a traced branch.)

- [ ] **Step 4: Run the new tests + full suite**

Run: `pytest tests/test_desi_priors.py -v` then `pytest tests/`
Expected: all PASS; full suite 155+13, 15 deselected.

- [ ] **Step 5: Commit**

```bash
git add src/jaxptpolypol/desi_priors.py tests/test_desi_priors.py
git commit -m "feat(priors): b1sigma8 measure -- sigma8 Jacobian + bounds in log_prior_nl_fn (default raw untouched)"
```

---

### Task 3: Reweighting helper + chain persistence (D machinery)

**Files:**
- Modify: `src/jaxptpolypol/desi_priors.py` (append `b1sigma8_log_weights`; export)
- Modify: `src/jaxptpolypol/__init__.py` (export)
- Modify: `example/mcmc/scripts/desi_prior_validation.py` (save the post-burn chain, non-smoke only)
- Test: `tests/test_desi_priors.py` (append)

**Interfaces:**
- Consumes: `sigma8_bins_fn` closures; existing `reweighted_moments(samples, weights, idx=None)` in `src/jaxptpolypol/marginal_taylor.py:762`.
- Produces: `b1sigma8_log_weights(theta_nl_samples, sigma8_bins_fn, *, b1_pos=None, lower=None, upper=None) -> jnp.ndarray (n_samples,)` — unnormalized log-weights converting a raw-measure chain to the b1sigma8 measure (`Σ_b log σ8` per sample; `-inf` outside bounds when `b1_pos`+bounds given). Task 4/5 rely on this exact signature. Also: `example/mcmc/cache/desi_chain_w.npy` (post-burn whitened draws, non-smoke runs).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_desi_priors.py`:

```python
# ---------------------------------------------------------------------------
# b1 sigma8 measure: post-hoc reweighting helper (option D)
# ---------------------------------------------------------------------------
from jaxptpolypol.desi_priors import b1sigma8_log_weights
from jaxptpolypol.marginal_taylor import reweighted_moments


def test_log_weights_match_pointwise_jacobian(tmp_path):
    _, split, s8_fn, _ = _fns_for_measure(tmp_path, "raw")
    rng = np.random.default_rng(4)
    samples = jnp.asarray(rng.normal(0.0, 0.3, size=(16, split.n_nl)))
    lw = b1sigma8_log_weights(samples, s8_fn)
    assert lw.shape == (16,)
    for i in range(16):
        assert float(lw[i]) == pytest.approx(
            float(jnp.sum(jnp.log(s8_fn(samples[i])))), abs=1e-12)


def test_log_weights_bounds(tmp_path):
    _, split, s8_fn, _ = _fns_for_measure(tmp_path, "raw")
    samples = jnp.zeros((2, split.n_nl))
    samples = samples.at[1, split.nl_b1_pos[0]].set(10.0)   # y = 6 > 3
    lw = b1sigma8_log_weights(samples, s8_fn,
                              b1_pos=split.nl_b1_pos, lower=0.0, upper=3.0)
    assert np.isfinite(float(lw[0])) and float(lw[1]) == -np.inf


def test_reweighting_gaussian_tilt_analytic_oracle():
    """Reweighting N(0,1) draws by exp(a*x) must give N(a,1): the exact
    finite-sample check is that reweighted moments match the ANALYTIC
    importance estimate, and at n=200k they must be within MC error of (a, 1)."""
    rng = np.random.default_rng(20260804)
    n, a = 200_000, 0.35
    x = rng.normal(0.0, 1.0, size=(n, 1))
    lw = a * x[:, 0]
    w = np.exp(lw - lw.max()); w /= w.sum()
    mean, std = reweighted_moments(x, w)
    se = 1.0 / np.sqrt(n * float((w.sum() ** 2) / (w ** 2).sum()) / n)  # ~1/sqrt(ESS)
    ess = 1.0 / np.sum(w ** 2)
    assert ess / n > 0.85                       # exp(-a^2) = 0.885 predicted
    assert float(mean[0]) == pytest.approx(a, abs=4.0 / np.sqrt(ess))
    assert float(std[0]) == pytest.approx(1.0, abs=4.0 / np.sqrt(ess))
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_desi_priors.py -k "log_weights or gaussian_tilt" -v`
Expected: FAIL with `ImportError: cannot import name 'b1sigma8_log_weights'`.

- [ ] **Step 3: Implement the helper**

Append to `src/jaxptpolypol/desi_priors.py`:

```python
__all__ += ["b1sigma8_log_weights"]


def b1sigma8_log_weights(theta_nl_samples, sigma8_bins_fn, *,
                         b1_pos=None, lower=None, upper=None):
    """Unnormalized log-weights converting a RAW-measure chain to the Table-I
    b1*sigma8 measure (arXiv:2511.20757): lw_i = sum_b log sigma8(z_b; theta_i),
    optionally -inf where any b1_b*sigma8_b leaves [lower, upper].

    Exact on the interior: the analytic theta_lin marginalization conditions
    on b1, so the measure change is a pure prior reweighting of theta_NL
    (see docs: options report / CONTEXT.md deviation 3). Weights are one
    scalar per sample, a function of the cosmology block only (plus b1 for
    the bounds). Normalize downstream (softmax) before use with
    ``marginal_taylor.reweighted_moments``.
    """
    theta_nl_samples = jnp.asarray(theta_nl_samples, dtype=jnp.float64)

    def one(theta):
        s8 = sigma8_bins_fn(theta)
        lw = jnp.sum(jnp.log(s8))
        if b1_pos is not None:
            y = theta[jnp.asarray(b1_pos)] * s8
            inside = jnp.all((y >= lower) & (y <= upper))
            lw = jnp.where(inside, lw, -jnp.inf)
        return lw

    return jax.vmap(one)(theta_nl_samples)
```

(Add `import jax` at the top of the file if not already present.) Export `b1sigma8_log_weights` from `src/jaxptpolypol/__init__.py`.

- [ ] **Step 4: Persist the production chain**

In `example/mcmc/scripts/desi_prior_validation.py`, directly after `draws = chain_w[BURN:]` (line ~421):

```python
if not SMOKE:
    CHAIN_OUT = CACHE / "desi_chain_w.npy"
    np.save(CHAIN_OUT, draws)          # post-burn whitened draws, raw-b1 measure
    print(f"chain -> {CHAIN_OUT} {draws.shape}", flush=True)
```

(Match the script's actual `SMOKE`/`CACHE` variable names — read the file first; the smoke-suffix refactor of 5c3b5ba defined them.)

- [ ] **Step 5: Run tests + tripwire**

Run: `pytest tests/ -q` (expect 155+16, 15 deselected) and
`cd example/mcmc && DESI_GATE_SMOKE=1 SMOKE=1 python3 scripts/desi_prior_validation.py 2>&1 | grep "log_post(x0)"`
Expected: `log_post(x0) = -172.996046`; NO `desi_chain_w.npy` written in smoke mode; `git status` clean for tracked cache artifacts.

- [ ] **Step 6: Commit**

```bash
git add src/jaxptpolypol/desi_priors.py src/jaxptpolypol/__init__.py tests/test_desi_priors.py example/mcmc/scripts/desi_prior_validation.py
git commit -m "feat(priors): b1sigma8_log_weights reweighting helper + persist production gate chain"
```

---

### Task 4: Cross-check script — flag-ON ≡ reweighted (option E)

**Files:**
- Create: `example/mcmc/scripts/b1sigma8_crosscheck.py`
- Test: run of the script itself (writes `cache/b1sigma8_crosscheck.json`)

**Interfaces:**
- Consumes: `load_desi_prior_spec`, `make_desi_prior_fns`, `b1sigma8_log_weights`, `dataclasses.replace`; the surrogate assembly pattern of `desi_prior_validation.py` (`stream_common.load_templates_and_whitening`, `make_marginal_log_posterior_taylor`, `run_rwmh_python` — transplant the loading/assembly blocks per that script; the flag-ON posterior is the SAME assembly with a modified spec).
- Produces: `example/mcmc/cache/b1sigma8_crosscheck.json` with the pointwise-identity residual, the flag-ON tripwire value, and short-chain moment agreement. Task 6 cites it.

- [ ] **Step 1: Write the script**

`example/mcmc/scripts/b1sigma8_crosscheck.py` — structure (transplant the template/whitening/prior/BAO assembly verbatim from `desi_prior_validation.py`; only what is shown here is new logic):

```python
"""Option E: prove flag-ON (b1sigma8 measure) == reweighted raw, two ways.

(1) POINTWISE (exact): log_post_ON(theta) - log_post_OFF(theta) must equal
    sum_b log sigma8(z_b; theta) at every point (interior) -- machine precision.
(2) CHAIN-LEVEL (MC): a short flag-ON RWMH chain's cosmology moments must match
    the b1sigma8_log_weights-reweighted moments of a short raw chain within MC
    error. RWMH only -- the [0,3] walls are hostile to NUTS.
Also records the flag-ON fiducial value lp0_on = lp0_raw + sum_b log sigma8(fid).
Writes cache/b1sigma8_crosscheck.json.
"""
from dataclasses import replace
# ... transplanted assembly: spec, split, rescaling fns, templates, whitening,
#     to_physical, full_params_fn, extra/BAO terms, log_post_raw  ...

b1_on = replace(spec.sampled["b1"], measure="b1sigma8",
                paper_lower=0.0, paper_upper=3.0)
spec_on = replace(spec, sampled={**spec.sampled, "b1": b1_on})
mean_on, sig_on, lp_nl_on = make_desi_prior_fns(
    spec_on, split=split, knl_bins=knl_bins, sigma8_bins_fn=sigma8_bins_fn,
    a_ap_bins_fn=a_ap_bins_fn, sigma8_ref_bins=sigma8_ref_bins)
log_post_on = make_marginal_log_posterior_taylor(
    tt, bin_data=bin_data, bin_cov_invs=bin_cov_invs,
    prior_mean_fn=mean_on, prior_sigma_fn=sig_on, log_prior_nl_fn=lp_nl_on,
    to_physical=to_physical, full_params_fn=full_params_fn,
    extra_theory_fn=extra_theory_fn, extra_data=extra_data,
    extra_cov_inv=extra_cov_inv)

# (1) pointwise identity at 64 whitened points (scale 0.5, fixed seed)
pts = 0.5 * jax.random.normal(jax.random.key(20260804), (64, n_nl))
def jac_at(w):
    th = to_physical(w)
    return jnp.sum(jnp.log(sigma8_bins_fn(th)))
resid = jnp.abs(jax.vmap(log_post_on)(pts) - jax.vmap(log_post_raw)(pts)
                - jax.vmap(jac_at)(pts))
max_resid = float(jnp.max(resid))
assert max_resid < 1e-8, max_resid

lp0_raw = float(log_post_raw(jnp.zeros(n_nl)))
lp0_on = float(log_post_on(jnp.zeros(n_nl)))
jac_fid = float(jac_at(jnp.zeros(n_nl)))
assert abs(lp0_raw - (-172.996046)) < 1e-5
assert abs(lp0_on - lp0_raw - jac_fid) < 1e-9

# (2) short chains: 20_000 draws each (burn 2_000), same step scale as the
#     gate script, seeds 20260804/20260805; reweight the raw one via
#     b1sigma8_log_weights on to_physical(draws); compare the 5 cosmology
#     means/stds: |d mean| < 4*combined SE, width ratio in [0.9, 1.1].
# ... run_rwmh_python calls transplanted from desi_prior_validation.py ...
```

The JSON must record: `max_pointwise_resid`, `lp0_raw`, `lp0_on`, `jac_fid`, chain moment tables, ESS/N of the reweighting, and a `verdict` PASS/FAIL from the two assertions plus the chain gates.

- [ ] **Step 2: Run it**

Run: `cd example/mcmc && python3 scripts/b1sigma8_crosscheck.py`
Expected: pointwise residual ≲ 1e-10; `jac_fid ≈ -5.87` (Σ log of the seven fiducial σ8 values); lp0_on ≈ -178.87; chain moments agree; verdict PASS; total wall ≲ 3 min (2 × 20k surrogate draws at ~2 ms/step + compile).

- [ ] **Step 3: Commit**

```bash
git add example/mcmc/scripts/b1sigma8_crosscheck.py example/mcmc/cache/b1sigma8_crosscheck.json
git commit -m "feat(priors): option-E cross-check -- flag-ON == reweighted raw, pointwise (1e-10) and chain-level"
```

---

### Task 5: Production measurement + report (option D deliverable)

**Files:**
- Create: `example/mcmc/scripts/b1sigma8_measure_report.py`
- Modify: `docs/design/perbin-compile-measurements.md` (append section)

**Interfaces:**
- Consumes: `cache/desi_chain_w.npy` (Task 3 — produced by ONE full gate re-run), `b1sigma8_log_weights`, `reweighted_moments`, the whitening npz (`to_physical` reconstruction as in the tier-2 analysis: `phys = fid_nl + draws @ L.T`).
- Produces: `example/mcmc/cache/b1sigma8_measure.json` — the measured chain-level shift, both-measure moment table, ESS/N.

- [ ] **Step 1: Produce the chain**

Run: `cd example/mcmc && caffeinate -i python3 scripts/desi_prior_validation.py`
Expected: full 200k gate run (~10 min), writes `cache/desi_chain_w.npy` (180000 × 26) and refreshes the tracked gate JSONs (same seed → statistically identical numbers; commit the refreshed artifacts).

- [ ] **Step 2: Write the report script**

`example/mcmc/scripts/b1sigma8_measure_report.py`:

```python
"""Option D: measure the b1-measure shift by reweighting the production chain.

Loads cache/desi_chain_w.npy (raw-measure DESI-prior surrogate chain),
computes w ~ exp(sum_b log sigma8) via b1sigma8_log_weights, and reports the
5 cosmology means/widths under BOTH measures, the shift in sigma_F units,
Kish ESS/N, and the comparison against the first-order prediction
F^-1 g = (-0.006, -0.028, +0.172, +0.097, -0.016) sigma_F.
Writes cache/b1sigma8_measure.json.
"""
# assembly: spec/split/rescaling fns transplanted from desi_prior_validation.py
# (sigma8_bins_fn is all that is needed), plus the whitening npz for
# fid_nl / cov_nl_prior / the DESI Hessian sig_F used by the gate.
import numpy as np, json, jax.numpy as jnp

draws_w = np.load(CACHE / "desi_chain_w.npy")
L = np.linalg.cholesky(np.asarray(cov_nl_prior))
phys = np.asarray(fid_nl) + draws_w @ L.T

lw = np.asarray(b1sigma8_log_weights(jnp.asarray(phys), sigma8_bins_fn,
                                     b1_pos=split.nl_b1_pos, lower=0.0, upper=3.0))
w = np.exp(lw - lw.max()); w /= w.sum()
ess_frac = 1.0 / (len(w) * np.sum(w ** 2))

mean_raw, std_raw = phys[:, :5].mean(0), phys[:, :5].std(0, ddof=1)
mean_rw, std_rw = reweighted_moments(phys[:, :5], w)
sig_F = np.asarray(SIG_F_DESI)           # from the gate artifact, 5 cosmology
shift = (np.asarray(mean_rw) - mean_raw) / sig_F
pred = np.array([-0.006, -0.028, +0.172, +0.097, -0.016])
# report table + JSON with: shift, pred, |shift - pred| vs MC SE, width
# ratios std_rw/std_raw (expect ~1), ess_frac (expect ~0.97), bounds hits (0).
```

- [ ] **Step 3: Run it and check against prediction**

Run: `cd example/mcmc && python3 scripts/b1sigma8_measure_report.py`
Expected: ESS/N ≈ 0.97; zero bound violations; width ratios 0.99–1.01; logA shift ≈ +0.17 σ_F and ns ≈ +0.10 σ_F within MC error of the prediction (MC SE ≈ 0.03–0.05 σ_F at this chain's ESS). If the measured shift disagrees with the prediction by ≫ MC error, STOP and report — do not tune; the discrepancy would itself be the finding (posterior non-Gaussianity beyond first order).

- [ ] **Step 4: Doc section**

Append to `docs/design/perbin-compile-measurements.md` a section `## b1 sigma8 measure (2026-08-04)`: the both-measure table, the measured-vs-predicted shift, ESS/N, the pointwise identity number from Task 4, and one paragraph stating the framing (prior-measure sensitivity as a robustness result; widths measure-independent; centers shift ≤0.17 σ_F; raw remains the default with the phase gate guarding real-data/nuLCDM).

- [ ] **Step 5: Commit**

```bash
git add example/mcmc/scripts/b1sigma8_measure_report.py example/mcmc/cache/b1sigma8_measure.json example/mcmc/cache/desi_prior_validation_sigmap.json example/mcmc/cache/branch_equiv_sigmap.json docs/design/perbin-compile-measurements.md
git commit -m "feat(priors): measured b1-measure shift via chain reweighting (option D) + doc section"
```

---

### Task 6: CONTEXT.md correction + ledger close

**Files:**
- Modify: `CONTEXT.md` (deviation 3 ~line 40, the b1 note ~line 42, the nuLCDM-hold note)
- Modify: `.superpowers/sdd/progress.md` (append)

**Interfaces:**
- Consumes: the measured numbers from Tasks 4–5 (`cache/b1sigma8_crosscheck.json`, `cache/b1sigma8_measure.json`).

- [ ] **Step 1: Rewrite the deviation record**

In `CONTEXT.md` deviation 3: DELETE the sentence claiming results "can always be *reported* in the paper's variables via the existing derived-parameter projection" and replace with (adapting the measured numbers from Task 5's JSON):

```
Raw-basis sampling differs from the paper's measure for b1: flat-in-raw-b1 vs
flat-in-b1σ8 differ by the prior weight Π_b σ8(z_b;θ) — a cosmology tilt
(measured by chain reweighting, 2026-08-04: logA +0.17 σ_F, ns +0.10, others
≤0.03; widths unaffected; evidence cache/b1sigma8_measure.json). The earlier
"report in paper variables via projection" rationale was INCORRECT — projection
changes coordinates of samples, not the measure they were drawn under. The
b2/bG2 rows are measure-correct (their Gaussian -log(width) normalization
carries the σ8² Jacobian). The spec's b1 row now has measure: raw|b1sigma8
(default raw; flag-ON ≡ reweighted proven to 1e-10 pointwise,
cache/b1sigma8_crosscheck.json), and load_desi_prior_spec(phase=...) REFUSES
measure: raw for real_data/nulcdm — in nuLCDM the dropped weight lands on
Σm_ν. Any real-data or nuLCDM assembly must pass the phase argument.
```

Extend the ~line 42 b1 note with: `(bounds remain non-binding at fiducial: b1σ8 = 0.57–0.67 vs [0,3])`. Add one line to the nuLCDM-hold note (grep "nuLCDM remains on hold"): `nuLCDM assemblies must call load_desi_prior_spec(..., phase="nulcdm") — the loader enforces the b1sigma8 measure there.`

- [ ] **Step 2: Verify + full suite**

Run: `pytest tests/ -q` (155+16 expected) and re-grep CONTEXT.md that no stale "projection" escape-hatch text survives: `rg -n "derived-parameter projection" CONTEXT.md` should show only historical/corrected phrasing.

- [ ] **Step 3: Commit + ledger**

```bash
git add CONTEXT.md
git commit -m "docs(context): correct deviation 3 -- b1 measure recorded, projection rationale retracted, phase gate documented"
```

Append to `.superpowers/sdd/progress.md` a plan-complete stamp with the measured numbers and commit list.

---

## Self-Review Notes

- Spec coverage: F1 → Tasks 1–2; H → Task 1 (YAML deviations) + Task 6 (CONTEXT.md); D → Tasks 3 + 5; E → Task 4; F2 → Task 1 (loader phase gate) + Task 6 (policy note). Declined options need no tasks.
- The `test_reweighting_gaussian_tilt_analytic_oracle` intentionally uses a pure-numpy synthetic (no project machinery) so it pins `reweighted_moments` + the weighting algebra against a closed-form answer.
- Type consistency: `b1sigma8_log_weights(theta_nl_samples, sigma8_bins_fn, *, b1_pos, lower, upper)` used identically in Tasks 3, 4, 5; `load_desi_prior_spec(name_or_path, phase="forecast")` in Tasks 1, 4, 5; `SampledRow.measure/paper_lower/paper_upper` throughout.
- The flag-ON tripwire is DERIVED (lp0_on − lp0_raw == Σ log σ8(fid), asserted to 1e-9) rather than hardcoded, so it cannot go stale.
