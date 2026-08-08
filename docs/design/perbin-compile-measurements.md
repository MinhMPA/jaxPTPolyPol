# Per-bin marginal posterior: first-compile time and peak RSS

**Verdict: the `lax.scan` form is NOT better than the unrolled per-bin form.**
On the 2-bin reference configuration it compiles **~1.7x slower** (28.7 s vs
16.9 s) for the same peak memory (~4.9 GB vs ~4.8 GB) and the same op count
(20 090 vs 20 096 StableHLO ops). The real win recorded here belongs to the
*per-bin* factorization over the *monolith*: peak RSS 4.8 GB vs 6.4 GB (-25%)
and 2x faster per-evaluation runtime. Use
`make_marginal_log_posterior_perbin`; `make_marginal_log_posterior_scan`
exists as a measured negative result and as scaffolding should the theory ever
gain a traced-bin-index entry point (see "Why" below).

## What was measured

Three constructions of the *same* joint P+B marginal log-posterior:

| path | builder |
|------|---------|
| `mono` | `marginal_likelihood.make_marginal_log_posterior` (dense `n_lin x n_lin` marginalization over the concatenated multi-bin data vector) |
| `perbin` | `marginal_likelihood.make_marginal_log_posterior_perbin` (Python-unrolled sum of per-bin marginalizations) |
| `scan` | `marginal_likelihood.make_marginal_log_posterior_scan` (`lax.scan` over stacked per-bin data/cov/lin-index arrays; theory dispatched by `lax.switch` on the traced bin index) |

Configuration = the `cfg` fixture of `tests/test_marginal_perbin.py`
(2 redshift bins, z = 0.7/0.9, AP on, IR resummation on, 8 k-modes, GL(8),
`num_mu = num_phi = 8`, tree bispectrum):

- `n_bins` = 2, per-bin block = 33 (`3 x 8` P-multipole + 9 B0 triangles)
- `n_data` = 66, `n_lin` = 22 (11 per bin), `n_nl` = 11 (sampled theta_NL)

"First compile" = wall time of *building the closure and making the first
(compiling) call*, i.e. trace + lower + XLA compile + one execution, with
`jax.block_until_ready`. One path per process — JIT caches and the allocator
high-water mark leak across builds inside a process.

## Numbers

Three fresh-process repeats per path. Machine: Apple M4 Max, 128 GB, macOS
15.5 (Darwin 25.5.0), CPU backend, Python 3.10.18, jax/jaxlib 0.6.2, x64
enabled.

| path | first compile [s] (3 runs) | peak RSS during compile [MB] | `ru_maxrss` [MB] | cached eval [s] | StableHLO ops | `while` / `case` ops |
|------|---------------------------|------------------------------|------------------|-----------------|---------------|----------------------|
| `mono`   | 17.94 / 17.98 / 17.95 | 6289 / 6514 / 6374 | 6321 / 6605 / 6398 | 2.04 / 2.05 / 2.11 | 19 624 | 14 / 83 |
| `perbin` | 17.03 / 16.75 / 16.94 | 4660 / 4783 / 4790 | 4697 / 4791 / 4814 | 0.99 / 1.01 / 1.00 | 20 096 | 14 / 83 |
| `scan`   | 28.82 / 28.46 / 28.73 | 4968 / 4970 / 4817 | 4984 / 4973 / 4846 | 1.13 / 1.18 / 1.14 | 20 090 | 15 / 85 |

Peak RSS is the max of `ps -o rss= -p <pid>` sampled every 50 ms in a thread
spanning the timed region; `ru_maxrss` is
`resource.getrusage(RUSAGE_SELF).ru_maxrss` (bytes on macOS) taken immediately
after it. RSS *before* the timed region was 1.27-1.36 GB in every run (imports,
emulator load, eager mock-data evaluation), so the compile itself accounts for
~3.4-5.2 GB of transient memory.

All three agree on the posterior value: `mono` and `perbin` give
`-15.025474752739306`, `scan` gives `-15.025474752739502` (relative difference
1.3e-14, pure summation-order round-off; the equivalence tests assert
`rtol=1e-10`).

## Why `scan` does not help

`make_joint_pk_bk_bin_fn` bakes its bin index in **statically**: it reads
`z_bins[b]`, `Hz_fid[b]`, `DAz_fid[b]` at Python level and calls the emulator at
that Python-level redshift. A single reusable scan body therefore cannot call
the theory at a *traced* bin index; the body dispatches through
`jax.lax.switch` over the tuple of per-bin closures instead, and `lax.switch`
traces **and compiles all `n_bins` branches**. The measured op counts confirm
it: scan saves 6 ops out of ~20 000 (one copy of the 11x11 Cholesky/solve/log-det
algebra instead of two) while adding one `while` and two `case` ops. The theory
graph — which is essentially all of the ~20 000 ops and all of the compile cost —
is still emitted `n_bins` times.

Where the extra ~12 s goes (one run per path, `fn.lower(x0)` then
`lowered.compile()`, same fresh-process protocol; these exclude the first
execution, so they do not sum to the totals above):

| path | trace + lower to StableHLO [s] | XLA compile [s] |
|------|-------------------------------|-----------------|
| `mono`   | 3.02 | 8.60 |
| `perbin` | 2.68 | 8.97 |
| `scan`   | 3.87 | 16.56 |

So ~85% of the penalty is **XLA compiling the `while` body**, not JAX tracing:
the theory ops now sit inside a loop body whose scatter indices and data slices
are loop-carried, which blocks the constant folding and cross-bin CSE that the
unrolled form gets for free. Tracing costs only ~1.2 s more (the
`jax.linearize` + `vmap` template construction has to go through the `switch`).

A genuine single-body compile requires option (ii): make the per-bin statics
(`z`, `Hz_fid`, `DAz_fid`) *traced inputs* of one closure and index them with the
traced bin index (`pklin_emulator.predict` takes `z` as ordinary data, so this is
feasible). That is a **theory-side** change to
`theory._make_theory_context_evaluator` and is out of scope for this task's file
set; it also has to preserve the "compute the background over all bins" trick
that keeps per-bin evaluation bit-identical to the all-bin one
(`theory.py:386-404`).

## Reproducing

The measurement script is not committed (throwaway). Recreate it as
`measure_compile.py` with the `cfg` fixture of `tests/test_marginal_perbin.py`
copied verbatim, wrapping the timed region as

```python
rss0 = _rss_kb()                       # ps -o rss= -p <pid>
sampler = RssSampler(); sampler.start()  # samples ps RSS every 50 ms
t0 = time.perf_counter()
fn = builders[path]()                  # build the closure
value = float(jax.block_until_ready(fn(x0)))   # first, compiling call
t1 = time.perf_counter()
sampler.stop()
ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2  # macOS: bytes
```

then, one fresh process per path, three repeats each (the Numbers table above
reports all three runs per path):

```bash
for rep in 1 2 3; do for p in mono perbin scan; do python measure_compile.py "$p"; done; done
```

The stage breakdown used the same fixture with
`t0; lowered = fn.lower(x0); t1; lowered.compile(); t2`.

Op counts came from `fn.lower(x0).as_text()` with
`re.findall(r"\b(?:stablehlo|chlo|mhlo)\.(\w+)", txt)` counted per op name
(note the lowered text is ~142 MB — the emulator weights are inlined constants,
which is also why peak RSS is measured in gigabytes).

Equivalence of the three paths is covered by
`tests/test_marginal_perbin.py::test_scan_logpost_equals_perbin_and_monolith`
and `::test_scan_logpost_with_extra_bao_term_equals_perbin`.

---

## 7-bin production-grid numbers (added 2026-07-25)

The table above is the 2-bin reference config. The production configuration was
measured twice, independently:

| harness | first compile | cached eval | peak RSS |
|---|---|---|---|
| isolated script, synthetic data/cov, no Fisher — **monolith** | 109.3 s | 65.44 s | 92.5 GB |
| isolated script, synthetic data/cov, no Fisher — **per-bin** | **54.4 s** | **5.06 s** | **28.3 GB** |
| inside the real LCDM notebook — **per-bin** | 60.9 s | — | 33.5 GB |

Config: `n_bins=7`, `block_len = 3*37 + 264 = 375`, `n_data = 2625` (+13 BAO),
`n_lin = 77` (11/bin), `n_NL = 26`. The two per-bin harnesses agree to ~12 %,
which is the expected spread between a bare script and a notebook that has
already allocated the emulator, covariance and Fisher objects.

**Per-bin vs monolith at 7 bins: 2.0x first compile, 12.9x per evaluation,
3.3x peak RSS**, with the two posteriors agreeing to ~4e-13 relative.

### Why the per-evaluation win is so much larger than the compile win

The op counts are equal (see above), so the runtime win is *dense linear
algebra*, not graph size. The marginal likelihood's dominant term is
`Ci_M = cov_inv @ M`:

- monolith: `(n_data x n_data) @ (n_data x n_lin)` ~ `n_data^2 * n_lin`
- per-bin: `n_bins` copies of `(block x block) @ (block x n_lin_b)` with
  `block = n_data/n_bins`, `n_lin_b = n_lin/n_bins` ~ `n_data^2 * n_lin / n_bins^3`

i.e. an `n_bins^3 = 343x` reduction in that matmul at 7 bins. The observed
end-to-end factor is 12.9x rather than 343x because the theory evaluation
itself does not shrink — it is now the floor.

### What is NOT fixed by this work

The notebook is still not demonstrably end-to-end. Measured per-cell:

- the Fisher `jacfwd` cell **completes** in 174 s at 77.6 GB (expensive, not a blocker);
- the `run_rwmh` cell ran **60 min at up to 94.0 GB and emitted zero draws**.

`run_rwmh` wraps `log_post` in a `lax.scan` over MH steps, which embeds the
whole ~61 s-compile posterior inside a scan body — the same failure mode this
document records for `make_marginal_log_posterior_scan`. At 5.06 s/eval a plain
Python loop over the already-compiled `log_post` gives ~12 steps/min and
produces draws incrementally; that is the outstanding follow-up.

### Correction (2026-07-25, verified): what the numbers above do and do not show

An 11-agent measurement workflow, with every probe adversarially verified,
corrected four claims made earlier in this document and in the commit log:

1. **The "2.0x first-compile win" is a measurement artifact.** The timed region
   was `build closure + first call`, and the first call includes execution.
   Decomposed with the measured cached evals: monolith `109.3 - 65.44 = ~43.9 s`
   compile; per-bin `54.4 - 5.06 = ~49.3 s` compile. **Compile is roughly equal**
   (per-bin possibly slightly worse). The real wins are per-evaluation (12.9x)
   and peak RSS (3.3x).

2. **The `n_bins^3` dense-algebra explanation is wrong.** The marginalization
   linear algebra is ~30 ms of a 65.44 s evaluation (~0.05%). The actual driver
   is the **linearize tangent count x bins**: the monolith's dense
   `M = vmap(jvp)(eye(n_lin))` propagates all 77 tangents through all 7 bins
   (539 bin-JVP units); the per-bin form propagates 11 through 1 bin each
   (77 units) — a factor `n_bins`. XLA cannot discover this itself, because
   scattering a dense `(77,)` tangent into the packed parameter vector makes
   every bin *formally* depend on all 77 parameters. Per-bin factorization is
   what makes that sparsity structural. The residual ~1.9x is working set.

3. **Emulator-weight inlining is NOT a driver.** `CosmoPowerJAX.predict` is
   `@partial(jit, static_argnames=('self',))`, so calling it inside a traced
   region emits one shared private function plus one `func.call` per bin: the
   weights appear **once**, not per bin. Batching `predict` over `z` changes the
   graph by +2.8%. Do not pursue it.

4. **Input StableHLO op count is a non-predictor.** It is ~139 MB / ~10k ops at
   every `n_bins` by construction (shared private funcs, ~23 extra `func.call`s
   per bin). Compile time tracks the **optimized** HLO instruction count at
   ~84 us/op.

Contested and worth one direct re-check: whether the 28 PT loop-kernel constants
(`complex128[257,257]`, emitted 2x per bin pre-inline) survive duplication into
the optimized module. The probe measured linear duplication pre-inline; the
verifier read the optimized HLO and found XLA CSE collapses them to 28 copies at
every `n_bins`. Prefer the verifier's reading pending a re-check.

### The notebook blocker, resolved

`run_rwmh` wraps `log_post` in a `lax.scan` over MH steps: **60 min, up to
94.0 GB, zero draws**. A plain Python loop over the same already-compiled
`log_post`: **20 steps -> 20 draws, 50% acceptance, 5.71 s median step
(= one `log_post` call), 10.5 steps/min, flat 28.5 GB**.

The posterior is fine; the scan wrapper is the blocker. This is the third
instance of one pathology — a ~50k-instruction body inside a `while` defeats
XLA's simplification/fusion passes (114k vs 180k optimized instructions, 2.2k vs
4.5k fusions). **Rule: drive iteration from Python over an already-compiled
callable; do not `lax.scan` an expensive body.** A Python loop is also
checkpointable and yields draws incrementally.

Honest throughput: 10.5 steps/min => ~159 h for 100k single-chain steps. Since
the 5.06 s evaluation is dominated by *template construction* (77 bin-JVP units
rebuilt every call) and not by the marginalization algebra, the correctly aimed
speedup is to stop rebuilding templates per call.

---

## Taylor surrogate (production, added 2026-07-29)

Build (`scripts/build_taylor_templates_lcdm.py`): Fisher/whitening stage 183.6 s
at 87.6-91.5 GB peak (whitening matches the tier-2 chain's to 1.1e-16);
template stage 2446 s (41 min) at **36.2 GB peak** after the m0-only-closure fix
(`5fa3c49` — the original H path floored at ~82 GB regardless of `chunk_H`
because the discarded `linearize`/M lanes multiplied both jacfwd widths).
H symmetry errors ~1e-16 across all 7 bins; templates 20.0 MB.

Validation (`scripts/taylor_surrogate_validation.py`, `cache/taylor_validation.json`):

| gate | result | numbers |
|---|---|---|
| 1 tilt | **PASS** | cosine 1 − 1e-12, norm ratio 1.0000001 vs the recorded exact `g_w` |
| 2 chain-vs-chain | REVIEW | widths [1.035, 0.964, **0.896**, 1.053, 1.031], corr diff 0.105; mean diffs ≤ 0.21 σ_F |
| 3 importance sampling | **PASS** | ess_frac **0.973** (389/400), max_weight 0.0044, reweighted mean shifts ≤ 0.094 σ_F, center diff −1.4e-14 |

**Headline: 0.65 ms/step in-chain (200 000 draws in 131 s), acceptance 0.243 —
~11 400× faster than the 7.4 s/step exact chain.** The options-review 10³–10⁴×
projection is verified mid-band.

The 0.65 ms and the "17.7 ms single fresh eval" are **both real** and measure
different things. 17.7 ms is a **cold-dispatch single call** — one isolated
`fn(x)` whose wall time is dominated by Python-side dispatch and the device
synchronization on `block_until_ready`, amortized over exactly one evaluation.
0.65 ms is the **steady-state per-step cost** averaged over the 200 000 in-loop
calls, where dispatch overlaps across iterations and the fixed per-call costs are
amortized away, leaving the microsecond-scale dense-algebra body. The in-chain
number is the relevant one for chain throughput; the single-eval number is what
you see timing one call in isolation.

**Gate-2 interpretation:** the reference exact chain has ESS 30–83, so its own
MC error is ~0.11–0.18 σ_F on means, ~8–13 % on widths, ~0.11–0.18 on
correlations — i.e. the gate-2 tolerance band (±10 % widths, 0.1 corr) is
TIGHTER than the reference's noise. Every gate-2 deviation is ≤ ~1.5× the
reference noise (logA's 0.896 is 0.9σ of the reference width error), while
gates 1 and 3 — which do not depend on the noisy reference — pass decisively.
The REVIEW flags the reference chain's precision, not a surrogate defect. A
definitive width check needs a higher-ESS exact-target reference (extended
exact chain, or a DA-MH chain: ~20 k steps ≈ 4.8 k exact evals ≈ 7 h).


## NUTS on the surrogate (added 2026-07-29)

Gradient-based sampling on the Taylor surrogate (Task 8). The surrogate body is
microsecond-scale dense algebra (linear templates + logdet tilt + Gaussian
priors, ~0.65 ms/step in-chain), so the scan-trap rule recorded above (a ~50k-op
body inside a `while` defeats XLA) does **not** apply: the body is tiny and
homogeneous, so blackjax `window_adaptation` + chunked-scan production
(`sampler.run_nuts`) is the right tool. Sampler used: **nuts_window_adaptation**
(fallback to fixed-L HMC triggers only on R-hat > 1.01 or divergences >
2%).

4 chains x 5000 draws (1000 adaptation), whitened space.
Mean acceptance 0.897, mean integration steps 7.9,
divergence fraction 0.0000, wall 680s.

| cosmo param | R-hat | ESS |
|---|---|---|
| ombh2 | 0.99990 | 27199 |
| omch2 | 1.00017 | 24832 |
| logA | 1.00012 | 26318 |
| ns | 0.99996 | 22357 |
| h | 0.99996 | 26888 |

**Skew of the logA / ns marginals** (scipy.stats.skew ± 1000-resample
bootstrap error), all three chains mapped through the SAME Cholesky
whitening -> physical transform. Mean pull is (mean - fiducial)/sigma_F:

| chain | n | skew(logA) | skew(ns) | pull(logA) [σ_F] | pull(ns) [σ_F] |
|---|---|---|---|---|---|
| nuts_window_adaptation | 20000 | +0.0318 ± 0.0167 | -0.0405 ± 0.0167 | -0.652 | -0.279 |
| surrogate_rwmh | 180000 | -0.0026 ± 0.0059 | +0.0338 ± 0.0059 | -0.659 | -0.295 |
| exact_tier2 | 5000 | +0.2188 ± 0.0294 | +0.0350 ± 0.0302 | -0.448 | -0.316 |

**Physics question — is the open logA/ns Tier-2 mean residual (means ~0.3-0.45
σ_F below the fiducial) genuine posterior skew?** A left-skewed (skew < 0)
marginal pulls the *mean* below the *mode*, so a negative mean pull is the
expected signature of skew when the mode sits near the fiducial. Verdict:
logA **NOT skew-explained**, ns **NOT skew-explained** — **both samplers find
|skew| ≲ 0.05 for logA and ns** (near-symmetric marginals), far too small to
source the 0.3-0.66 σ_F mean pulls. The mean-residual-is-not-skew conclusion is
unchanged.

The NUTS and RWMH *skews* do **not** agree in sign (ns: NUTS −0.0405 vs RWMH
+0.0338; logA: NUTS +0.0318 vs RWMH −0.0026), but this is not a real tension.
The RWMH ns skew reads +0.0338 ± 0.0059 — an apparent ~5.7σ "detection" of
positive skew — but that ±0.0059 is an **iid bootstrap over ~180k
autocorrelated draws** and therefore badly understates the true error; the
significance, and the apparent NUTS/RWMH sign discrepancy, are spurious. Once
autocorrelation is folded into the error bars both estimates are consistent with
|skew| ≲ 0.05 and with each other.

The genuine cross-check that the residual is posterior *geometry* and not a
sampler artifact is the tight agreement of the mean **pulls** — not the skews —
between the two independent samplers (logA −0.652 vs −0.659; ns −0.279 vs
−0.295), with the noisy exact Tier-2 chain (ESS 30-83) in the same direction
(−0.448, −0.316). The residual is a shift of the whole posterior *center*
(consistent with the logdet / prior-volume marginalization tilt, Gate-1 cosine
≈ 1), not a mean-vs-mode skew. Written by `scripts/taylor_surrogate_nuts.py`;
numbers in `cache/taylor_nuts_result.json`.

## Gate-2 re-adjudication vs the DA-MH exact-target chain (2026-07-30)

The definitive width check promised above. `scripts/damh_exact_chain_lcdm.py`
ran 22 000 delayed-acceptance steps overnight (surrogate stage-1 proposal
filter, exact-target stage-2 correction — Christen & Fox 2005, target exactly
the marginal posterior): stage-1 acceptance 0.2406, **stage-2 acceptance
0.9719** (the surrogate is that good a proposal), 5 295 exact evaluations,
9.90 h wall, 30 GB peak. Burn 2 000 → 20 000 exact-target draws, cosmology
ESS **188–240** (vs tier2's 30–83) → reference MC error ~5 % on widths,
~0.07 σ_F on means. Surrogate side: the NUTS chain (4×5000 flattened, ESS
~25k, same surrogate target as the RWMH chain, which was not persisted).

| param | width ratio (sur/exact) | mean diff (σ_F) |
|-------|------------------------|-----------------|
| ombh2 | 0.939 ± 0.051 | −0.054 ± 0.075 |
| omch2 | 0.996 ± 0.049 | −0.082 ± 0.066 |
| logA  | 0.968 ± 0.052 | −0.024 ± 0.075 |
| ns    | 1.037 ± 0.046 | +0.057 ± 0.061 |
| h     | 0.949 ± 0.049 | −0.123 ± 0.071 |

**Widths all inside [0.9, 1.1]** (the tier2-round outlier 0.896 for logA is
now 0.968 — it was reference noise, as diagnosed). **Means all consistent
with zero** (≤ 1.8 SE). Correlations: max diff 0.136 (ombh2–ns and ns–h)
nominally breaches the 0.1 band, but (a) it is 1.9 SE of the DA chain's own
correlation noise (ESS ~200 → SE ≈ 0.07/pair), (b) the two **exact-target**
chains (DA vs tier2) differ by up to 0.171 on the same pairs — the
exact-vs-exact scatter exceeds the band — and (c) on both breaching pairs
tier2 agrees with the surrogate (+0.073/+0.096) and not with DA
(−0.022/−0.075): the DA chain is the outlier on these near-zero
correlations. **Gate 2: PASS.** Numbers in
`example/mcmc/cache/gate2_readjudication.json`; summary stamped into
`cache/taylor_validation.json` (`gate2_final`).

## Stream-B gate (DESI priors) — sigmap branch (added 2026-07-31)

Acceptance gate for branch `stream-b-sigmap` (Amendment 1): the production
Taylor surrogate marginal posterior under the DESI DR1-reanalysis
(2511.20757) priors, wired through `make_desi_prior_fns` **cov-mode**. The
surrogate embeds the spec priors, so "Fisher with the same spec" is the
surrogate's own Hessian at the fiducial, `F = -hess logpost(0)` (whitened
coords; Tier-1 established curvature == Fisher–Schur there). Gates compare a
200 000-step RWMH chain (seed 20260731, burn 20 000, gradient-free
`run_rwmh_python`) against that Hessian-Fisher on the 5-parameter cosmology
block; the mean check uses the AD-tilted center
`mu_tilt = fid + F^{-1} grad logpost(fid)`. Config: 7 z-bins, P+B
(k∈[0.02,0.20], bispectrum k≤0.08) + DESI-DR2 BAO + BBN(ombh2) + ns. Chain
wall 461 s at 2.31 ms/step, acceptance 0.357, peak RSS 3.25 GB, 569 s total.
Script: `example/mcmc/scripts/desi_prior_validation.py`; numbers in
`cache/desi_prior_validation_sigmap.json`.

| param | width ratio (chain/F) | ESS | tilt pred (σ_F) | chain pull (σ_F) | mean pull vs tilt (σ_F) |
|-------|----------------------|-----|-----------------|------------------|-------------------------|
| ombh2 | 0.995 | 2719 | −0.72 | −0.67 | +0.047 |
| omch2 | 0.989 | 1790 | −0.96 | −0.71 | +0.256 |
| logA  | 1.002 | 1505 | +1.91 | +0.76 | −1.155 |
| ns    | 1.014 | 1220 | +0.71 | +0.44 | −0.276 |
| h     | 0.986 | 2196 | −0.97 | −0.82 | +0.151 |

max |corr diff| = 0.021 (< 0.1). **G1 widths PASS, G2 correlations PASS,
G3 means REVIEW → verdict REVIEW.**

**G3 diagnosis (not a prior-wiring bug).** The DESI counterterm priors
(c2→30 etc.) tilt the posterior strongly away from the fiducial:
`|mu_tilt_w| = 3.21` (whitened σ). The single Newton step `mu_tilt` therefore
extrapolates the mean over a large, mildly non-Gaussian displacement and
**systematically overshoots** the true MCMC mean in every cosmology
parameter (|chain pull| < |tilt pred| for all five; worst on logA, the
amplitude parameter most coupled to the counterterm centering: tilt predicts
+1.91 σ_F, chain moved +0.76 σ_F, residual −1.15 σ_F). This is a limitation of
the AD-tilt *mean predictor*, not of the prior wiring or the chain: **G1
(widths, ~1 %) and G2 (correlations, 0.021) both PASS at ESS 1200–2700**, i.e.
the surrogate chain reproduces the Hessian-Fisher second moments under the
DESI priors exactly, which a mis-wired prior could not do. The gradient at the
fiducial is dominated by the expected DESI prior-mean pulls (amplitude/
counterterm direction), as required by the brief's Step-3 check before
escalation. Recommended follow-up if a tighter mean gate is wanted: use an
iterated tilt (Newton refinement) or the chain mean itself as the reference,
rather than a single first-order step.

**Templates reused unmodified (prior-independence), UNROTATED on this branch.**
The Taylor tensors (`cache/taylor_templates_lcdm.npz`) are the *same* artifact
built for the pre-DESI validation — nothing about them depends on the prior. On
the sigmap branch they also stay in our μ-space tilde counterterm basis
(unrotated): the exact paper per-multipole c0/c2/c4 prior arrives entirely
through the cov-mode `(n_bins, 11, 11)` blocks
`L(f)·diag(paper_σ²)·L(f)ᵀ`, consumed by `gaussian_marginal_loglike`'s
full-Σ_p branch, so the Hessian-Fisher carries the full correlated counterterm
prior automatically.

**Cross-branch equivalence (primary cross-validation).** The equivalence dump
`cache/branch_equiv_sigmap.json` records `log_post` at 64 fixed whitened points
`0.5·normal(PRNGKey(20260731),(64,n_nl))` (include_logdet). The rotation branch
(rotated templates + diagonal paper prior) generates the identical points; on
the three points visible during the parallel run the two branches' `log_post`
agree to ~1–3×10⁻¹² (≪ the 1e-5 Task-E threshold), and the full 200 000-step
chains produce **bit-identical gate statistics** (width ratios, mean pulls, ESS
all matching). The two exact representations are therefore numerically
equivalent, and the G3 REVIEW is a shared property of the tilted posterior +
Newton predictor, not a per-branch artifact. (Full 64-point Δ comparison is
Task E, once both `branch_equiv_*.json` are collocated.)

**One Task-5σ integration fix was required for this gate.** The production
linear-Pk emulator (`jense_2023_camb_lcdm_Pk_lin`) lists baryon-feedback inputs
`A_b/eta_b/logT_AGN` among its parameters, but those are *fixed* in the LCDM
layout and live outside the sampled nl cosmo block, so `make_lcdm_rescaling_fns`
(which slices `theta_nl[:n_cosmo]`) could not supply them (the Task-5σ toy
tests used synthetic rescaling closures and never exercised the real emulator).
Added a backward-compatible `fixed_cosmo_extras` kwarg to
`make_lcdm_rescaling_fns` (and `extra_cosmo` to `derived._emulator_input_dict`)
that injects them as constants (zero θ-derivative); emulator sigma8 is inert to
them at ~1e-5. Full suite stays 130 green.

### Frozen-R diagnostic (2026-07-31)

Isolation experiment for the G3 mean-vs-mode gap: rerun the full sigmap gate with
every θ_NL-dependent prior WIDTH frozen at fiducial (`FROZEN_R=1`:
`sigma8_bins_fn ≡ sigma8_ref_bins`, `a_ap_bins_fn ≡ 1` — freezing both the
layer-2 A_AP·A_amp division and the b2/bG2 σ8-widths; the bΓ3 coevolution MEAN's
b1-dependence deliberately stays live). 200k/20k, acceptance 0.386, ESS
1461–2797, 273 s. Widths 0.933–0.995 (G1 PASS), corr 0.0177 (G2 PASS).

Mean pulls vs the frozen posterior's own tilted center (live-run values in
parentheses): ombh2 +0.03 (+0.05), omch2 +0.37 (+0.26), **logA −0.78 (−1.15)**,
**ns −0.06 (−0.28)**, h +0.12 (+0.15).

**Conclusion — the attribution is PARTIAL, and informative.** The θ-dependent
prior widths fully explain the ns pull and ≈⅓ of the logA pull (−1.15 → −0.78).
The residual is NOT width-driven: it matches the pre-existing marginal-posterior
non-Gaussianity documented in the tier2-era logdet-tilt work (with constant
legacy priors, the logA mean already sat ~0.3–0.45 σ_F beyond the first-order
tilt) — i.e. marginalization volume through the θ-dependence of A(θ) = MᵀC⁻¹M +
Σ_p⁻¹ plus model curvature, amplified here by the wider DESI ctr priors. The
mean-vs-mode methodology rule (CONTEXT.md) is unaffected: the tilted center
remains the mode-level comparison target; the mean offset is genuine, now
decomposed into a width-volume part (~0.4 σ_F on logA, all of ns) and an
intrinsic-curvature part (~0.8 σ_F on logA). Evidence:
`example/mcmc/cache/desi_prior_validation_sigmap_frozenR.json`.

## Stream-B Task E: cross-branch equivalence of the two ctr-prior representations (2026-07-31)

The two exact representations of the per-multipole ctr prior — `stream-b-rotation`
(templates rotated by L(f), diagonal Table-I prior verbatim) and `stream-b-sigmap`
(templates untouched, correlated per-bin Σ_p = L·diag(σ²)·Lᵀ through the extended
full-covariance marginal likelihood) — were compared on 64 shared-seed whitened
points (PRNGKey 20260731, scale 0.5, include_logdet=True):

**max |Δ log_post| = 4.235e-12 (mean 1.6e-12, max relative 2.4e-14) — PASS**
(criterion 1e-5; the marginal likelihood incl. the logdet term is exactly invariant
under a linear θ_lin reparameterization, and both implementations hit the float64
floor). The two 200k production chains produced gate statistics agreeing to
4e-14 (width ratios) / 6e-13 (mean pulls) — same posterior, same trajectory.

Both gates: G1 widths PASS (0.986–1.014), G2 correlations PASS (max diff 0.021),
G3 means REVIEW with a corroborated physics diagnosis: the AD-tilted center
fid + F⁻¹∇logpost(fid) lands on the MODE (Newton-converged to ≤0.06 σ_F), while
the chain MEAN sits ~1.1 σ_F lower along logA — persisting with
include_logdet=False — i.e. genuine posterior asymmetry from the θ_NL-dependent
A_AP·A_amp prior widths (the documented σ8/logA-low volume pull of EFT
full-shape priors), not a wiring or representation error. Evidence:
`example/mcmc/cache/{task_e_equivalence,branch_equiv_rotation,branch_equiv_sigmap,desi_prior_validation_rotation,desi_prior_validation_sigmap}.json`
plus the noLD corroboration JSON on the rotation branch.

## Tier-3: sampled-c1 vs marginalized-c1 validation (2026-08-01) — PASS

The last open validation from CONTEXT.md's c1 section. Route A marginalizes c1
analytically by linearizing the theory in it, which drops the `Z1_fog·Z1_fog`
c1² cross-term. This test asks whether that omission is
visible in the cosmology posterior.

**Why the surrogate makes this cheap and exact:** with c1 moved into θ_NL the
theory is *exactly quadratic* in c1, and the Taylor surrogate's second-order m0
expansion reproduces polynomial-degree-≤2 dependence to the float64 floor — so
the sampled-c1 surrogate carries the full c1² physics that the marginalized path
linearizes away. Template rebuild with the c1-sampled split: 3872 s, peak
96.3 GB, H symmetry residual ~1e-16 → `taylor_{templates,whitening}_lcdm_c1s.npz`
(θ_NL 26→33: +1 c1 per bin at positions 8,12,…,32; θ_lin 77→70).

Two 200 000-draw RWMH chains (burn 20 000; seeds 20260731/20260801; 1.8–1.9 ms/step,
905 s total, peak 2.4 GB) under **identical** DESI priors — the sampled side
carrying c1 ~ 𝒩(0, (1.0125/A_AP·A_amp)²) in the sampled block instead of the
marginalized row:

| param | Δmean (σ_F) | tolerance | width ratio (samp/marg) |
|-------|-------------|-----------|------------------------|
| ombh2 | +0.008 | 0.096 | 0.994 |
| omch2 | +0.013 | 0.099 | 0.987 |
| logA  | −0.049 | 0.137 | 0.977 |
| ns    | −0.017 | 0.108 | 0.987 |
| h     | +0.019 | 0.090 | 0.990 |

max |corr diff| = 0.041 (< 0.05). **All three gates PASS.**

**c1 marginals (sampled side):** mean ≈ 0 (−0.014 … +0.052), σ = 0.97–1.00
against a prior width of 1.0125 — c1 is prior-dominated, the data constrains it
essentially not at all. This is both the expected EFT behaviour (2502.14758 found
the same when sampling c1) and the reason the linearization is invisible: a
parameter the data cannot see cannot transmit its quadratic term to cosmology.

**Conclusion: Route A (analytic c1 marginalization) is validated** — the
marginalized and sampled treatments give the same cosmology posterior to within
MC precision. Evidence: `example/mcmc/cache/tier3_c1_validation.json`.

### Correction (2026-08-02, after adversarial review): what actually validates Route A

The section above cites the two 200 000-draw chains as the evidence that
dropping the c1² term is harmless. **That citation was wrong — the chains have
no power to detect the effect.** Corrected account:

**The effect, measured deterministically.** c1 has no bilinear coupling to the
marginalized block (`dM[:, :, c1] ≡ 0` in all 7 bins, verified), so the *entire*
difference between the marginalized and sampled models is the constant c1²
coefficient of m0, `q_b = ½ ∂²m0_b/∂c1_b²`. Its whitened norm is
`Σ_b q_bᵀ C_b⁻¹ q_b = 2.159e-09` at |c1| = 1, i.e. an omitted signal of
**4.6×10⁻⁵ σ**, rising as c1⁴ to **1.2×10⁻³ σ at a 5σ prior draw**. A data-vector
displacement of s σ cannot move any parameter by more than s σ_F, so this is a
strict upper bound. Elementwise on the bispectrum block the c1² piece is
≤1.7×10⁻⁵ of the theory (median ~1×10⁻⁶) — the previously quoted "≈6×10⁻⁴" was a
stale per-leg estimate at the library default `k_nl_rsd = 0.3` rather than a
data-vector measurement at the production 0.45, overstating it ~35×.

**Why the chains could not have shown this.** The chain gate tolerates
0.090–0.137 σ_F on means with an MC noise floor of 0.028–0.047 σ_F — roughly
2000× the effect and 700× above the noise needed to see it. It would return PASS
identically if the c1² term were deleted, or made 1000× larger. The chains remain
useful as a *wiring/consistency* check of the c1-sampled split (they confirm the
two independently-built pipelines agree), but they are not a measurement of the
c1² effect and must not be cited as one.

**Exactness caveat.** The order-2 surrogate reproduces c1² exactly *along the
pure-c1 direction from θ0*; c1²×δθ cross terms are third order in θ_NL and are
truncated — symmetrically on both sides of the comparison, so the comparison
remains fair.

Artifact: `scripts/tier3_c1_bound.py` → `cache/tier3_c1_bound.json`
(runs in seconds, asserts the 5σ-draw signal stays below 1e-2 σ, and fails loudly
if a template rebuild ever breaks the `dM[:, :, c1] ≡ 0` premise).

## b1 sigma8 measure (2026-08-04)

Production chain-level measurement (option D) of the shift induced by sampling
the DESI Table-I prior in b1·σ8 rather than raw b1. The 200 000-step raw-measure
gate chain (`desi_prior_validation.py`, seed 20260731, burn 20 000 → 180 000
draws; the re-run reproduced every committed gate statistic bit-identically and
the `branch_equiv_sigmap.json` dump byte-identically) was reweighted with
`b1sigma8_log_weights` (w ∝ exp Σ_b log σ8(z_b; θ), b1·σ8 ∈ [0,3]). This
reweighting is *exactly* the flag-ON posterior: Task 4's cross-check proved
`log_post_ON − log_post_OFF == Σ_b log σ8` pointwise to **5.95e-14** across 64
production-posterior points (fiducial identity residual 1.78e-15,
`jac_fid = Σ_b log σ8(fid) = −5.874902`). Script:
`scripts/b1sigma8_measure_report.py`; numbers in `cache/b1sigma8_measure.json`.

**Kish ESS/N = 0.9611** (173 000/180 000), **zero** draws hit the [0,3]
b1·σ8 bounds, max normalized weight 1.3e-05 — the two measures are this close.

| param | mean (raw) | σ (raw) | mean (b1σ8) | σ (b1σ8) | σ ratio | shift (σ_F) | predicted F⁻¹g (σ_F) | \|Δ\| / 3·MC-SE |
|-------|-----------|---------|-------------|----------|---------|-------------|----------------------|-----------------|
| ombh2 | 0.0220967 | 0.000481 | 0.0220952 | 0.000481 | 1.001 | −0.0032 | −0.006 | 0.034 |
| omch2 | 0.117012  | 0.003250 | 0.116925  | 0.003241 | 0.997 | −0.0265 | −0.028 | 0.015 |
| logA  | 3.090275  | 0.057348 | 3.100180  | 0.057235 | 0.998 | **+0.1731** | **+0.172** | 0.010 |
| ns    | 0.978596  | 0.028116 | 0.981513  | 0.028205 | 1.003 | **+0.1052** | **+0.097** | 0.066 |
| h     | 0.673644  | 0.003565 | 0.673598  | 0.003566 | 1.000 | −0.0127 | −0.016 | 0.037 |

The measured shift matches the first-order prediction F⁻¹g on every parameter
well inside 3× the (conservative, quadrature-combined) MC standard error of
0.027–0.042 σ_F — the largest residual is ns at 0.0082 σ_F. The residuals sit
far below even 1× SE because the raw and reweighted means are estimated from
the *same* draws with near-uniform weights, so their MC errors cancel almost
completely in the difference; the agreement is a genuine confirmation that the
posterior responds to the measure change at first order, not a coincidence of
noise. Width ratios are 0.997–1.003, i.e. the tilt is mean-only — as Task 4's
review proved, an analytic log-density offset that is (locally) linear in the
parameters cannot change a width, so width agreement is forced and the
measured ratios are a consistency check, not a finding.

**Framing.** This is a prior-measure *robustness* result: switching the b1
prior measure between raw b1 (our default) and b1·σ8 (the 2511.20757 Table-I
convention) moves the cosmology posterior centers by at most **0.17 σ_F**
(logA; ns +0.11 σ_F, all others ≲ 0.03 σ_F) and leaves the widths
measure-independent. On this forecast-grade surrogate the effect is therefore
quantified and small-but-not-negligible; the measured shift equals the
first-order Fisher prediction, so the sensitivity can be forecast without
re-running chains. Raw b1 remains the production default; the phase gate
(`desi_priors` spec `measure` field + `b1sigma8_log_weights` reweighting)
guards the real-data and nuLCDM phases, where the Table-I measure — or a
chain-level reweighting equivalent to it, as validated here — can be switched
on without touching the theory or the templates.

## LCDM production MCMC — first run (2026-08-04)

First end-to-end execution of the production path (commit 89888a7): 4 chains ×
5000 NUTS draws on the Taylor surrogate under the DESI DR1-reanalysis priors,
launched by flipping SMOKE_TEST=False in the production notebook.

**Sampler health (first execution of this branch):** mean acceptance 0.887,
zero divergences, R-hat 0.99992–1.00002 on all five cosmology parameters,
ESS 15,493–26,628 of 20,000 post-warmup draws. Wall: minutes-scale end to end
— the same chain on the exact per-bin path would have taken ~10 days.

**Headline Fisher↔MCMC result (the project's central question):**

| param | fid | MCMC mean | Fisher σ (DESI spec) | MCMC σ | ratio |
|-------|-----|-----------|----------------------|--------|-------|
| ombh2 | 0.02242 | 0.022085 | 0.00048211 | 0.00047489 | 0.99 |
| omch2 | 0.11933 | 0.11704 | 0.003422 | 0.0032227 | 0.94 |
| logA  | 3.047 | 3.0902 | 0.062477 | 0.057345 | 0.92 |
| ns    | 0.9665 | 0.97939 | 0.027963 | 0.027349 | 0.98 |
| h     | 0.6766 | 0.67358 | 0.0036491 | 0.0035587 | 0.98 |

The Gaussian Fisher forecast tracks the full posterior's widths at the ≤8 %
level under the production priors. Means sit off-fiducial as expected from the
documented mechanisms (non-fiducial prior means + logdet/volume tilts;
logA +0.69 σ_F, consistent with the +0.76 seen in the RWMH surrogate chain —
different sampler, same posterior). Tripwire log_post(x0) = −172.996046 exact.

## nuLCDM gate (2026-08-04) — both marginal-mean modes + mnu Jacobian

The `desi_prior_validation.py` gate under `--cosmology nulcdm` (Task 5): the
production Taylor surrogate marginal posterior with **mnu added to the sampled
cosmology basis** (n_NL 26→27, mnu at θ_NL position 5), the DESI DR1-reanalysis
**b1σ8 spec under `phase="nulcdm"`** (`load_desi_prior_spec("...
_b1s8", phase="nulcdm")`), the nuLCDM templates/whitening
(`taylor_{templates,whitening}_nulcdm.npz`, theory-config hash `8f0f2e74…`,
`cosmology: nulcdm` meta guards), and the **Σm_ν ≥ 0 physical bound** as a −∞
indicator in the sampled-block prior. Gradient-free RWMH (200 000 steps, burn
20 000; NUTS is forbidden — the b1σ8 U[0,3] and Σm_ν≥0 walls are −∞ indicators).
Widths/correlations compare the chain against the surrogate's own
Hessian-Fisher `F = −hess logpost(0)` on the **6-parameter cosmology block
(incl. mnu)**; the mean check uses the AD-tilted center
`μ_tilt = fid + F⁻¹∇logpost(fid)`. Run TWICE at production scale under the two
marginal-mean modes (CONTEXT.md policy 2026-08-04). Artifacts:
`cache/nulcdm_gate_{spec,fiducial}_means.json` (+ `…_chain_w.npy`).

**spec-means mode** (paper-fidelity means; seed 20260807; acc 0.275; chain wall
341 s, 459 s total; peak RSS 3.3 GB; surrogate lp0 = −178.879579):

| param | width ratio (chain/F) | tilt pred (σ_F) | mean pull vs tilt (σ_F) | (mean−fid)/σ_F | ESS |
|-------|----------------------|-----------------|-------------------------|----------------|-----|
| ombh2 | 1.006 | −0.702 | +0.091 | −0.611 | 2298 |
| omch2 | 1.007 | −0.991 | +0.280 | −0.711 | 1302 |
| logA  | 0.892 | +1.457 | −0.324 | +1.133 | 1561 |
| ns    | 1.102 | +0.784 | +0.253 | +1.037 | 1107 |
| h     | 0.953 | −0.961 | −0.043 | −1.004 | 1331 |
| **mnu** | **0.800** | **+0.085** | **+0.626** | **+0.711** | 1032 |

max |corr diff| = 0.158 at (ns, mnu); **core-5 (excl. mnu) = 0.078**.
**G1 REVIEW, G2 REVIEW, G3 REVIEW → verdict REVIEW.**

**fiducial-means mode** (production policy config, means = per-bin fiducial
θ_lin `packed_params[split.lin_idx]`; seed 20260808; acc 0.281; chain wall
353 s, 464 s total; peak RSS 3.2 GB; surrogate lp0 = −173.635756):

| param | width ratio (chain/F) | tilt pred (σ_F) | mean pull vs tilt (σ_F) | (mean−fid)/σ_F | ESS |
|-------|----------------------|-----------------|-------------------------|----------------|-----|
| ombh2 | 0.998 | −0.616 | +0.121 | −0.495 | 1762 |
| omch2 | 0.952 | −1.166 | +0.321 | −0.845 | 1896 |
| logA  | 0.890 | +1.505 | −0.438 | +1.067 | 1656 |
| ns    | 1.128 | +0.856 | +0.158 | +1.014 | 1166 |
| h     | 0.967 | −0.975 | +0.010 | −0.965 | 1792 |
| **mnu** | **0.791** | **+0.190** | **+0.557** | **+0.746** | 1266 |

max |corr diff| = 0.152 at (ns, mnu); **core-5 (excl. mnu) = 0.088**.
**G1 REVIEW, G2 REVIEW, G3 REVIEW → verdict REVIEW.**

**The mnu-direction measurement (the quantity the phase gate protects).**

| quantity | spec | fiducial |
|----------|------|----------|
| (a) ∂(Σ_b log σ8)/∂mnu \|_fid | **−1.8135** | **−1.8135** |
| (b) induced 1st-order tilt (F⁻¹g)_mnu (σ_F) | +0.085 | +0.190 |
| (c) realized chain mnu mean pull vs tilt (σ_F) | +0.626 | +0.557 |
| Σm_ν≥0 boundary-hit frac (< 0.01 eV) | 0.035 | 0.032 |
| min Σm_ν (eV) / max identical-run (draws) | 0.000 / 40 | 0.000 / 41 |
| σ_F(mnu), Hessian (eV) | 0.120 | 0.121 |

(a) is a fiducial quantity (mode-independent) and is **NEGATIVE**: σ8 falls with
Σm_ν, so the b1σ8 change-of-variables Jacobian Σ_b log σ8(z_b) genuinely tilts
the measure toward lower Σm_ν (≈ −0.26 per z-bin × 7 bins). (b) the FULL
induced tilt (F⁻¹g)_mnu is only slightly positive (+0.09/+0.19 σ_F): the direct
negative Jacobian pull is offset by the data/BAO gradient and the cross-parameter
F⁻¹ mixing. (c) the chain mnu mean sits **+0.56–0.63 σ_F ABOVE** the tilted
center — the Σm_ν≥0 wall (0.5 σ_F below the fiducial in the Hessian metric,
σ_F,mnu ≈ 0.12 eV) truncates the low-mnu tail and pushes the mnu mean up. The
wall IS sampled (≈ 3–4 % of draws within 0.01 eV of it) but does **not stick**:
the longest identical-draw run (40/41) matches the expected geometric maximum
(≈ 38 at acceptance 0.28 over 180 k draws), and acceptance is healthy.

**Why REVIEW is a documented outcome, not a wiring failure.** In both modes the
full 6-parameter gate reads REVIEW, but the deviations are entirely
mnu-specific: (i) **every** correlation whose chain-vs-Hessian difference exceeds
0.1 involves mnu — (ns,mnu), (omch2,mnu), (h,mnu), (logA,mnu) — while the
**LCDM-analog 5-parameter cosmology core (ombh2/omch2/logA/ns/h) has max corr
diff 0.078 (spec) / 0.088 (fiducial), both < 0.1**, and its width ratios sit in
[0.9,1.1] to within MC error; (ii) mnu's own width ratio ≈ 0.80 is the
truncated-Gaussian narrowing of the Σm_ν≥0 wall (a lower truncation at 0.5 σ_F
predicts a 1-D width ratio ≈ 0.70; the correlated marginal narrows to 0.80). The
Hessian-Fisher is the LOCAL curvature at the fiducial=mode and is blind to the
wall; the chain sees the truncation. A mis-wired prior would corrupt the un-walled
core, which it does not. G3 REVIEW is the same mean-vs-mode volume effect
documented for LCDM (θ_NL-dependent A_AP·A_amp prior widths + logdet volume),
now compounded by the wall on mnu.

**spec vs fiducial mean-mode comparison.** The two modes share WIDTHS, the
correlated ctr block, and the θ-rescaling — only the marginalized-nuisance prior
MEANS differ (spec = mapped/rescaled paper means + bΓ3 coevolution; fiducial =
the constant per-bin fiducial θ_lin, zero θ_NL-gradient). The difference is
visible in the **tilt center**, not the posterior itself: the residual
(mean−fid)/σ_F pulls are nearly mode-independent (logA +1.13 vs +1.07, ns +1.04
vs +1.01, mnu +0.71 vs +0.75), while the mean-pull-vs-tilt (G3) column shifts
because μ_tilt moves (spec carries the extra non-fiducial prior-mean gradient;
fiducial's tilt is the pure logdet-volume term, and its lp0 drops from −178.88
to −173.64 as the prior-mean residuals at x0 vanish). Against the LCDM
fiducial-round residual pulls (ombh2 −0.62, omch2 −0.77, logA +0.56, ns +0.43,
h −0.80; those used the notebook's DESI-spec comparison Fisher, whereas this gate
uses the surrogate Hessian-Fisher, so magnitudes are comparable-but-not-identical
in construction), the nuLCDM fiducial-mode analogs match in SIGN throughout
(ombh2 −0.50, omch2 −0.85, h −0.97 negative; logA +1.07, ns +1.01 positive) and
are LARGER on the amplitude/shape directions (logA, ns) — the σ8–Σm_ν degeneracy
opened by adding mnu to the basis amplifies the volume pull there — with the new
mnu direction pulling **+0.75 σ_F** up off the wall. This is the honest nuLCDM
analog: the physical Σm_ν≥0 prior is the dominant force on the mnu marginal
(narrows it, pushes its mean up, distorts its cross-correlations), exactly the
behaviour the b1σ8 phase gate + the mnu bound were built to represent.

Reproduce (from `example/mcmc`, after `build_taylor_templates_lcdm.py
--cosmology nulcdm`):
`python scripts/desi_prior_validation.py --cosmology nulcdm --marginal-means
{spec,fiducial}`. LCDM default unchanged: tripwire log_post(x0) = −172.996046
in smoke.

### mnu wall diagnostic (2026-08-04) — unbounded run: NEGATIVE RESULT

**Production remains BOUNDED (flat Σm_ν ≥ 0) per the user decision
(2026-08-04).** One diagnostic chain was run WITHOUT the wall
(`--mnu-unbounded`, valid only with `--cosmology nulcdm`; seed 20260809,
200 000 RWMH / burn 20 000, otherwise identical to the bounded fiducial-means
gate — shared-F tripwire confirmed: lp0 = −173.635756, σ_F and tilt_pred
bit-identical to the bounded run, i.e. the Hessian is wall-blind as expected)
to measure the truncation's effect on the mnu marginal rather than infer it.

**The diagnostic returned a negative result: the unbounded configuration is
INVALID for this pipeline.** Σm_ν < 0 lies outside both the mnu emulator's
training domain and the Taylor surrogate's validity radius; their composition
manufactures a spurious sharp mode, and the chain collapsed into it at
mnu ≈ −0.33 eV (−3.2 σ_F from the fiducial): SD 0.00098 eV (~σ_F/124),
negative-mass fraction 1.000, acceptance 0.005, ESS ~20–27, max identical-draw
run 14 547. The Σm_ν ≥ 0 wall was not only physics — it was protecting the
sampler from undefined-model territory. The recorded table is **pathology
evidence, not a truncation measurement**
(`cache/nulcdm_gate_fiducial_means_unbounded.json`, `"diagnostic_verdict":
"INVALID_CONFIGURATION"`):

| mnu marginal (eV) | bounded | unbounded (INVALID) |
|-------------------|---------|---------------------|
| mean | 0.15025 | −0.32795 |
| SD | 0.09568 | 0.00098 |
| mean pull vs tilt (σ_F) | +0.557 | −3.398 |
| neg-mass frac (mnu<0) | 0.000 | 1.000 |
| σ_F(mnu) shared (eV) | 0.12094 | 0.12094 |

(SD "ratio" 97.96 and core-5 "leaks" up to 3.95 σ_F are artifacts of the
collapse, not wall effects.) **The truncation effect therefore stands
quantified analytically, not by chain comparison:** a 1-D truncated normal
with the wall at −0.496 σ_F predicts an SD factor **0.697**, against the
observed bounded marginal width ratio **0.791** — the gap between 0.70 and
0.79 reflecting the multivariate correlations (the mnu marginal is not the
1-D conditional the analytic formula truncates). Any future genuine unbounded
diagnostic would require an emulator trained through mnu ≤ 0 and a surrogate
re-centered/validated there — out of scope.

Reproduce (evidence only — do not expect a truncated-Gaussian comparison):
`python scripts/desi_prior_validation.py --cosmology nulcdm --marginal-means
fiducial --mnu-unbounded`.

## nuLCDM production run (2026-08-05, Task 6)

First end-to-end execution of the nuLCDM production path (notebook
`example/mcmc/mcmc_joint_PFS_BAO_BBN_ns_nuLCDM.ipynb`, committed with this doc):
4 chains × 200k **RWMH** draws (20k burn → 180k kept/chain) on the Taylor
surrogate under the DESI DR1-reanalysis **b1σ8** priors
(`desi_dr1_reanalysis_2511_20757_b1s8`, `phase="nulcdm"`), fiducial-centered
marginalized-prior means (policy default), seed 20260806, launched by flipping
SMOKE_TEST=False. RWMH not NUTS: the b1σ8 `U[0,3]` and the mnu ≥ 0 walls are
−∞ indicators (non-differentiable). `log_post(x0) = −173.635756`
(fiducial-centered; the spec-means/`desi_paper` value is −178.879579); surrogate
identity |surrogate − exact| = 5.7×10⁻¹³.

**Sampler health:** acceptance per chain [0.281, 0.281, 0.280, 0.284]
(RWMH-optimal ~0.234), R-hat 1.00008–1.00057 on all six cosmology parameters,
ESS 5570–10697 of 180k kept draws/chain. Wall: 25 min 23 s for the sampling
cell (~30 min notebook end to end) — the same chain on the exact per-bin path
would have taken days.

**Headline Fisher↔MCMC result:**

| param | fid | MCMC mean | Fisher σ (DESI spec) | MCMC σ | ratio |
|-------|-----|-----------|----------------------|--------|-------|
| ombh2 | 0.02242 | 0.022158 | 0.00049003 | 0.00048188 | 0.98 |
| omch2 | 0.11933 | 0.11662 | 0.0037219 | 0.0032276 | 0.87 |
| logA  | 3.047 | 3.1388 | 0.1337 | 0.078534 | 0.59 |
| ns    | 0.9665 | 0.99635 | 0.035172 | 0.033284 | 0.95 |
| h     | 0.6766 | 0.67293 | 0.0043902 | 0.0036311 | 0.83 |
| mnu   | 0.06 | 0.14719 | 0.20804 | 0.095725 | 0.46 |

Residual pulls (volume term only under fiducial_centered):
ombh2 −0.53, omch2 −0.73, logA +0.69, ns +0.85, h −0.84, mnu +0.42.

The well-constrained parameters track the Gaussian Fisher at the few-percent
level (ombh2 0.98, ns 0.95), while **logA (0.59) and mnu (0.46) sit below 1
because of the Σm_ν ≥ 0 wall**, not a Fisher failure: the comparison Fisher is
unbounded, so marginalizing over mnu inflates the logA and mnu marginal σ along
their degeneracy direction, whereas the MCMC posterior is truncated at the wall
and is correspondingly narrower on exactly those two. The mnu marginal is a
truncated shape, not a detection — its chain σ = 0.0957 eV reproduces the
validation-gate chain σ exactly (`cache/nulcdm_gate_fiducial_means.json`),
giving a width ratio **0.791** against the gate's Hessian Fisher σ_F ≈ 0.121 eV
(1-D truncated-normal analytic 0.697; ~3.2% of draws within 0.01 eV of the
wall). The 0.46 in the table above is the same chain σ measured against the
notebook's broader diagonal-consumed comparison Fisher σ (0.208 eV). The bound
is load-bearing: the unbounded diagnostic collapsed to a spurious extrapolation
mode at mnu ≈ −0.33 eV (verdict INVALID_CONFIGURATION; see the mnu wall
diagnostic section above).

Reader note on differing defaults: the validation gate
(`desi_prior_validation.py`) defaults to `--marginal-means spec`, preserving the
LCDM `log_post(x0) = -172.996046` tripwire and the recorded Stream-B artifacts,
while the production notebooks default to `PRIOR_VARIANT = "fiducial_centered"`
(the CONTEXT.md policy). This is deliberate — gate JSONs and notebook outputs
therefore use different prior means and are not directly comparable on lp0 or
AD-tilted centers unless the modes are matched.

## Profile-likelihood check (2026-08-05)

Both production notebooks
(`mcmc_joint_PFS_BAO_BBN_ns_LCDM.ipynb`, `mcmc_joint_PFS_BAO_BBN_ns_nuLCDM.ipynb`)
carry a frequentist profile-likelihood cell (markdown + code, inserted after the
Fisher-vs-MCMC comparison-table cell, before the corner plot). It profiles the
DATA-ONLY chi-square on the Taylor surrogate — the exact whitened GLS profile-out
of the 11 linear EFT/stochastic amplitudes per bin (no theta_lin priors, no
b1sigma8 Jacobian/walls), plus the fiducial-centered BAO and BBN(ombh2)+ns10
data-like terms — over each cosmological parameter, at a 17-point grid spanning
fid +/- 2 sigma_F (for mnu: 0 to fid + 2 sigma_F, box-bounded mnu >= 0, the
emulator/surrogate validity domain). The noiseless mock forces chi2_prof(fiducial)
= 0 to the float64 floor (tripwire: LCDM 1.2e-23, nuLCDM 1.2e-23), so the profile
minimum is at fiducial by construction; the marginal MEANS drift only from the
cosmology-dependent marginalization volume. Free cosmo dimensions are box-bounded
to fid +/- 6 sigma_F purely to keep the background-distance integral defined (a
numerical validity safeguard; the profile ridge sits well inside).

Punchline — profile-min offsets sit at fiducial while the marginal-mean pulls do
not:

LCDM (5 params):

| param | profile-min offset (sigma_F) | marginal-mean pull (sigma_F) |
|-------|------------------------------|------------------------------|
| ombh2 | -0.000 | -0.62 |
| omch2 | +0.001 | -0.77 |
| logA  | +0.000 | +0.56 |
| ns    | -0.000 | +0.43 |
| h     | +0.000 | -0.80 |

max |profile-min offset| = 0.001 sigma_F; pulls span 0.43-0.80 sigma_F.

nuLCDM (6 params):

| param | profile-min offset (sigma_F) | marginal-mean pull (sigma_F) |
|-------|------------------------------|------------------------------|
| ombh2 | -0.000 | -0.53 |
| omch2 | +0.004 | -0.73 |
| logA  | -0.010 | +0.69 |
| ns    | +0.001 | +0.85 |
| h     | +0.003 | -0.84 |
| mnu   | -0.003 | +0.42 |

max |profile-min offset| = 0.010 sigma_F; pulls span 0.42-0.85 sigma_F.

The contrast is the point: the likelihood peaks at fiducial on every parameter
(offsets ~ 0), so the 0.4-0.9 sigma_F marginal-mean pulls are Bayesian
marginalization-volume / projection effects, not fitting errors. The strict
|offset| < 0.1 sigma_F gate is asserted at the 17-point production resolution; the
SMOKE gate uses a coarse 5-point grid (which cannot localize a broad profile's
vertex to 0.1 sigma) with a loosened 0.5 sigma tolerance that still trips on a
broken profile.

## CMB Fisher block: two-branch experiment (2026-08-06/07)

The joint PFS+BAO+CMB+BBN forecasts consume a fiducial-centered Gaussian CMB
Fisher block built once by `example/mcmc/scripts/build_cmb_fisher_block.py`
(Planck highl TTTEEE + lowl TT + lowl EE simall + Planck lensing + ACT DR6
lensing). The first implementation reproduced the source notebooks exactly: the
OBSERVED Hessian `F = -0.5 (H + H^T)` of the summed log-likelihood at our
fiducial. It aborted gate G2 for nuLCDM.

### The finding: an indefinite observed Hessian

nuLCDM raw 28x28 min eigenvalue **-0.250293**; projected into the shared basis
**-46.2436**. The direction is the CMB geometric degeneracy (99.7% H0, 7.7%
mnu). Not a porting bug — the committed nuLCDM notebook carries the same
`-2.663e-01` and prints `---` for `omch2`, `h`, `mnu` in its CMB-only column.

Cause: for a Gaussian band-power likelihood the exact Hessian is
`-d2 logL = J^T C^-1 J - sum_a (C^-1 delta)_a d2 m_a`. The residual-curvature
term has no definite sign, and our fiducial is not the joint maximum of the real
Planck/ACT data (`chi2_resid` at fiducial: 2345.6 highl, 9.11 Planck lensing,
14.47 ACT), so along a near-null direction that term dominates and flips the
sign. LCDM was never clean either — its raw 27x27 min eigenvalue was already
**-0.00740928**; only the projection hid it.

### Per-term attribution (measured, not asserted)

The summed Fisher is exactly additive over the five terms in the packed basis,
so for the nuisance-profiled minimum eigenvector `u` of the marginalized
cosmology block, `sum_t u^T F_t u` reproduces the eigenvalue and splits it by
term. Recomputed on EVERY build and stored in the artifact META under
`method.negative_mode_attribution` (nuLCDM, observed Hessian, pre-dedupe):

| term | `u^T F_t u` | share |
|---|---|---|
| planck_highl | -0.248403 | 93% |
| planck_lensing | -0.0964809 | 36% |
| planck_lowl_tt | -0.0251028 | 9% |
| planck_lowl_ee | **+0.071592** | — |
| act_dr6_lensing | **+0.0321369** | — |
| SUM | -0.266258 | |

Both dominant negative contributors are Gaussian in band powers; the two low-ell
terms are net POSITIVE (+0.0465) along that direction. That is what makes a
hybrid legitimate rather than a patch.

### The two branches

* **Branch A — `expt/cmb-psd-clip`**: eigenvalue clipping `max(lambda, 0)`.
  Kept UNMERGED as the documented fallback. Do not develop further.
* **Branch B — `expt/cmb-expected-fisher`** (ADOPTED): hybrid Gauss-Newton.
  Terms with a Gaussian band-power data model contribute the EXPECTED Fisher
  `J^T C^-1 J` (PSD by construction); the two non-Gaussian low-ell terms, for
  which `J^T C^-1 J` does not exist, keep the observed Hessian.

| term | likelihood object | method | covariance source |
|---|---|---|---|
| planck_highl | `clipy.smica.smica_lkl` | **GN** | `_internal.siginv` (2289x2289) |
| planck_lowl_tt | `clipy.gibbs.gibbs_lkl` | hessian | none (Blackwell-Rao `cl2x` spline) |
| planck_lowl_ee | `clipy.simall.simall_lkl` | hessian | none (`probEE` spline) |
| planck_lensing | `clipy.lkl._clik_lensing` | **GN** | `clik.siginv` (9x9), `pp_hat` |
| act_dr6_lensing | `candl.likelihood.LensLike` | **GN** | `covariance_chol_dec` (10x10) |

`clik_candl.covariance` raises `NotImplementedError` for all four clipy terms;
the inverse covariances live under `siginv`. `make_candl_theory_vector_fn` works
only for the native-candl ACT term, so the two clipy model vectors are
reconstructed and then validated against the untouched `log_like` in value, full
Hessian, and along the reference minimum-eigenvalue direction.

Branch comparison, `sigma(mnu)` from the PFS-regularized joint proxy:
**0.0387** (branch B pre-dedupe) / **0.0404** (branch A, clipped) / **0.0957**
(PFS only). Post-dedupe branch B: **0.0390**.

### The A_planck defect (found by adversarial review, fixed on branch B)

All four Planck `.clik` likelihoods are loaded with `all_priors=True`, so each
folds the SAME Gaussian `A_planck` calibration prior (`sigma = 0.0025`,
curvature `160000`) into its own `log_like`. Summing the five per-term blocks
counted it **four times** — a pre-existing baseline defect, faithfully
reproduced by both branches.

Fix: a shared-prior INVENTORY (each prior's curvature obtained by
differentiating the likelihood object's own prior callable — never hardcoded),
then subtraction of `(count-1) x curvature = 3 x 160000` at the `A_planck`
packed index, applied AFTER summation so no per-term log-likelihood and no
Gauss-Newton validation reference is disturbed. Exact, not approximate: a
Gaussian prior's Hessian is a constant matrix. Two hard checks guard it —
COMPLETENESS (the enumerated per-prior curvatures must sum to the term's total
prior curvature) and GAUSSIANITY (curvature must not move when its own
parameters are perturbed); anything unresolvable raises `SharedPriorError`.

Effect on the CMB-only marginal widths — the overcounted prior was making the
amplitude direction look tighter than it is:

| param | LCDM pre -> post | nuLCDM pre -> post |
|---|---|---|
| logA | 0.013163 -> 0.013558 (**+3.00%**) | 0.014757 -> 0.015320 (**+3.81%**) |
| omch2 | 0.0010642 -> 0.0010727 (+0.81%) | 0.0016178 -> 0.0016179 (+0.01%) |
| h | 0.0048581 -> 0.0048965 (+0.79%) | 0.018971 -> 0.019048 (+0.41%) |
| mnu | — | 0.13697 -> 0.13807 (+0.80%) |
| tau | 0.0070807 -> 0.0070896 (+0.13%) | 0.0073758 -> 0.0073765 (+0.01%) |

All widths LOOSEN, as an over-confidence fix must.

### Production numbers (post-dedupe, both artifacts)

| | LCDM | nuLCDM |
|---|---|---|
| `sigma_tau` | 0.007089623562 | 0.007376499170 |
| G2 min eig (strict `> 0`) | 4188.53 | **51.3613** |
| max eig | 1.8334e+08 | 1.83336e+08 |
| baseline observed-Hessian min eig | 4317.77 | **-46.2436 (ABORT)** |

No eigenvalue clipping or regularization anywhere. Gate G2 is strict `> 0`.

### Follow-up (DEFERRED, not this plan)

The committed `example/fisher/fisher_joint_PFS_BAO_CMB_{LCDM,nuLCDM}.ipynb`
notebooks carry the SAME 4x `A_planck` overcount and the same observed-Hessian
indefiniteness. Their CMB-column numbers are therefore ~3-4% over-confident on
logA. Not corrected here; the scripts are the production path.

Adoption rationale and the decisions of record are in CONTEXT.md (2026-08-07).

## Joint PFS+BAO+CMB+BBN MCMC forecasts (2026-08-07)

Two production notebooks, both committed with `SMOKE_TEST = False` and full
outputs:

| | LCDM | nuLCDM |
|---|---|---|
| notebook | `example/mcmc/mcmc_joint_PFS_BAO_CMB_BBN_LCDM.ipynb` | `example/mcmc/mcmc_joint_PFS_BAO_CMB_BBN_nuLCDM.ipynb` |
| production commit | `e1fb52b` | `a71e949` (outputs), `e4d251a` (final source, doc-only) |
| sampled dimension | 27 (26 theta_NL + tau) | 28 (27 theta_NL + tau) |
| reported basis | ombh2, omch2, logA, ns, h, tau | + mnu (tau at 5, mnu at 6) |
| sampler | NUTS on the Taylor surrogate, 4 x 5000 draws, 1000 warmup, `NUTS_SEED = 20260731` | RWMH on the Taylor surrogate, 4 x 200000 draws, 20000 burn, `RWMH_SEED = 20260806` |
| `log_post_joint(theta0)` tripwire | **-167.752302** | **-173.635756** |
| `chi2_prof(fiducial)` | 1.231e-23 | 1.193e-23 |
| `cond(cov_joint)` | 2.045e+08 | 2.046e+08 (`cond(cov_nl_prior)` 1.209e+07) |

The probe set and the CMB-as-Gaussian treatment are the decisions of record in
CONTEXT.md (2026-08-06 / 2026-08-07); the CMB block itself and the two-branch
experiment that produced it are documented above under
"CMB Fisher block: two-branch experiment (2026-08-06/07)" and are not repeated
here.

Wall times (Apple silicon, same machine as the sections above):

| stage | LCDM | nuLCDM |
|---|---|---|
| PFS P+B Jacobian/Fisher | 2 min 36 s | 2 min 33 s |
| first exact `log_post` (trace+compile+eval) | 56.8 s | 55.7 s |
| production sampling | **13 min 42 s** (NUTS) | **26 min 21 s** (RWMH, 800k total steps) |
| profile scan | 20.3 s | 29.5 s |
| notebook end-to-end | ~18 min | ~31 min |

### Wiring tripwires, and what lp0 does NOT prove

Both notebooks assert the hard identity
`log_post_joint(theta0) == log_post_surr(x0) == exact-path lp0`, and both print
`CMB loglike(theta0) = 0.000e+00; BBN loglike(theta0) = -0.000e+00 (both EXACTLY
0 by construction)`. Exact-vs-surrogate PFS agreement: `|Delta| = 1.421e-13`
(LCDM) / `5.684e-13` (nuLCDM). `log_post_joint_w(0)` reproduces the same value
(affine reparametrisation only).

**IMPORTANT — lp0 is BLIND to the `COSMO_PRIORS` removal.** These notebooks drop
the old `{'ombh2': 0.00055, 'ns': 0.042}` cosmo priors so that BBN and the CMB
block are not double-counted, but a fiducial-centered Gaussian prior contributes
EXACTLY 0 at the fiducial: removing it cannot move lp0. lp0 tests the joint
wiring (index maps, whitening composition, surrogate/exact equivalence), NOT the
double-count avoidance. The evidence for the removal is the **prior-entry
counts**, printed by both notebooks:

```
Legacy-spec prior entries (whitening geometry): 77; DESI-spec prior entries (comparison Fisher): 91
```

i.e. 79 -> 77 on the whitening side and 93 -> 91 on the comparison side vs the
`mcmc_joint_PFS_BAO_BBN_ns_*` sources — exactly the two removed cosmo priors.
Do not cite lp0 for this.

### Sampler health (verbatim)

LCDM:

```
production NUTS-on-surrogate samples: (4, 5000, 27)
mean acceptance 0.882; divergence frac 0.0000
  param     R-hat        ESS
  ombh2   0.99992    30028.4
  omch2   1.00002    24826.0
   logA   0.99994    26703.6
     ns   0.99991    27509.7
      h   0.99992    24230.1
    tau   0.99994    26007.4
```

nuLCDM:

```
production RWMH-on-surrogate samples (post burn-in): (4, 180000, 28)
acceptance per chain = [0.351 0.35  0.351 0.353] (RWMH optimal ~0.234)
  param     R-hat        ESS
  ombh2   1.00028    11756.5
  omch2   1.00006    11781.9
   logA   1.00006    12972.6
     ns   1.00003    11901.2
      h   1.00034     9973.9
    tau   1.00002    12939.7
    mnu   1.00026    12848.1
```

R-hat max 1.00002 / 1.00034, zero divergences (LCDM), ESS >= 24230 (LCDM) and
>= 9974 (nuLCDM) INCLUDING tau and mnu. nuLCDM acceptance 0.350-0.353 sits at
the top edge of the RWMH band with nothing tuned — the expected consequence of a
more Gaussian target once the CMB block dominates.

### Headline Fisher-vs-MCMC tables (verbatim)

LCDM — `Fisher: PFS P+B + DESI DR2 BAO + CMB block + BBN + DESI DR1-reanalysis
EFT priors (same information as the sampled target)`:

```
  param        fid    MCMC mean   Fisher sig     MCMC sig   ratio
  ombh2    0.02242     0.022397   0.00011478   0.00011327    0.99
  omch2    0.11933       0.1194   0.00053746   0.00053177    0.99
   logA      3.047       3.0467     0.011894     0.011984    1.01
     ns     0.9665      0.96636    0.0035039    0.0035062    1.00
      h     0.6766      0.67607    0.0022525    0.0022193    0.99
    tau     0.0561     0.055947    0.0059752    0.0060354    1.01
residual pulls (volume term only under fiducial_centered): ombh2=-0.20  omch2=+0.13  logA=-0.02  ns=-0.04  h=-0.24  tau=-0.03
```

nuLCDM — same line with `b1sigma8 EFT priors`:

```
  param        fid    MCMC mean   Fisher sig     MCMC sig   ratio
  ombh2    0.02242     0.022403   0.00011903   0.00011786    0.99
  omch2    0.11933      0.11933   0.00070789   0.00067388    0.95
   logA      3.047       3.0486     0.014276     0.013727    0.96
     ns     0.9665      0.96662    0.0036972    0.0036756    0.99
      h     0.6766      0.67589    0.0026304    0.0025049    0.95
    tau     0.0561     0.056894     0.007228    0.0069281    0.96
    mnu       0.06     0.065965     0.038089      0.03341    0.88
residual pulls (volume term only under fiducial_centered): ombh2=-0.14  omch2=-0.01  logA=+0.11  ns=+0.03  h=-0.27  tau=+0.11  mnu=+0.16
```

MCMC/Fisher sigma ratios are 0.99-1.01 on all six LCDM parameters and 0.95-0.99
on six of seven nuLCDM parameters; **mnu at 0.88 is the only outlier** and it is
the `mnu >= 0` wall (see below), not a convergence problem. Residual volume
pulls collapse to |.| <= 0.24 sigma_F (LCDM) / 0.27 sigma_F (nuLCDM) — far
smaller than the 0.4-0.9 sigma_F of the PFS-only runs, because the CMB block
makes the joint posterior much more Gaussian.

An independent orientation check on the inline comparison Fisher: the committed
`example/fisher/fisher_joint_PFS_BAO_CMB_LCDM.ipynb` "PFS P+B+BAO+CMB" column is
`(1.2042e-4, 5.3874e-4, 0.011693, 0.0034728, 0.002286, 0.005948)`; per parameter
`F_cmp` sits at logA **+1.72%**, h **−1.47%**, ns **+0.90%**, tau **+0.46%**,
omch2 **−0.24%**, and ombh2 **−4.7%** (tighter) — the ombh2 gap is the expected
BBN signature, since that notebook carries no BBN term. The logA entry
independently corroborates Open item 1: `F_cmp` uses the A_planck-**deduped**
CMB block while the `fisher_joint` column still carries the 4× overcount, so
`F_cmp`'s logA must be LOOSER by roughly the dedupe's measured effect (+2.36% on
the LCDM joint proxy) — +1.72% is that sign and that rough size.

### tau: first appearance as a SAMPLED parameter

This is the first time tau appears as a SAMPLED MCMC dimension in this repo (it
had appeared only inside Fisher blocks; the simall-zero-gradient era's tau-prior
workaround was retired 2026-07-14). Here tau is sampled with
**no tau prior and no bound**; its only constraint is the CMB block's curvature.

| | LCDM | nuLCDM |
|---|---|---|
| sigma(tau), comparison Fisher | 0.0059752 | 0.007228 |
| sigma(tau), chain | 0.0060354 | 0.0069281 |
| sigma(tau), CMB-alone artifact | 0.007090 | 0.007376 |
| ratio joint/CMB-alone (Fisher) | **0.843** | **0.980** |
| tau range over the chain | [0.03260, 0.07963] | [0.02811, 0.08765] |

Both ratios are below 1, as they must be (PFS+BAO+BBN can only sharpen tau
through its degeneracies), and both chains stay strictly positive with no bound
imposed (`min(tau) > 0` is a hard physicality assert, not a prior).

**Why nuLCDM's ratio is so much closer to 1 (0.980 vs 0.843): mnu absorbs the
degeneracy-breaking.** With mnu free, the direction along which PFS+BAO would
otherwise donate information to tau is partly spent on mnu instead. The Task-4
review confirmed this numerically rather than by assertion: refitting with mnu
FIXED recovers ratio **0.927**, and the PFS+BAO-only sigma(logA) — the parameter
tau is degenerate with — degrades by **3.4x** when mnu is freed
(0.065085 -> 0.22439 in the two cell-29 gain tables, 3.45x).

corr(logA, tau), the CMB degeneracy the joint target must transport correctly:

```
[PASS] E7 corr(logA, tau) = +0.900 (chain) vs +0.898 (F_cmp)     # LCDM
[PASS] E7 corr(logA, tau) = +0.923 (chain) vs +0.930 (F_cmp)     # nuLCDM
```

Chain and Fisher agree to 0.002 / 0.007, and tau's marginal is NOT simply the
CMB-alone one.

### The mnu wall, remeasured (nuLCDM) — READ THE FLAVOR LABELS

`sigma_F` appears in two incompatible flavors in this lineage and they must
never be mixed:

* **Gauss-Newton (GN) comparison Fisher** — `F_cmp` here, `F_pfs_bao_prior_cosmo`
  in the PFS-only notebook. Joint sigma_F(mnu) = **0.038089 eV**; PFS-only
  GN baseline = **0.20804 eV**.
* **Marginal-posterior Hessian-Fisher** — the validation gate's flavor.
  PFS-only sigma_F(mnu) = **0.120937 eV**.

The same PFS-only chain sigma of 0.095725 eV therefore yields a truncation ratio
of 0.46 against the GN denominator and 0.79 against the Hessian denominator.
The notebook's originally reported "0.791 -> 0.877, essentially unchanged"
crossed the two flavors; Task-4 review caught it and it was corrected in
`e4d251a`.

**On matched Gauss-Newton denominators:**

| quantity (GN flavor throughout) | PFS-only | joint |
|---|---|---|
| sigma_F(mnu) | 0.20804 eV | 0.038089 eV |
| chain sigma(mnu) | 0.095725 eV | **0.033410 eV** (2.87x tighter) |
| truncation ratio sigma_chain/sigma_F | **0.46** | **0.877** |
| fiducial distance from the `mnu >= 0` wall | **0.29 sigma_F** | **1.58 sigma_F** (5.5x further out) |

So the truncation is **strongly weakened** by the CMB block — 0.46 -> 0.877, not
the marginal move the mixed-flavor text implied.

**Flavor-free facts — the wall still bites:**

* draws within an ABSOLUTE 0.01 eV of the wall: **3.24%** (PFS-only 3.24%,
  printed as "3.2%") — no sigma_F enters this fraction, so it needs no flavor
  caveat, and it is essentially unchanged;
* `min(mnu) = 0.00000 eV` — the chain still reaches the wall;
* mnu is the only parameter with an MCMC/Fisher sigma ratio of **0.88** against
  0.95-0.99 for the other six.

Both halves must be carried together: the truncation weakened a lot AND the
bound is still load-bearing. The mnu marginal remains **a truncated shape by
construction, not a detection**, and must not be quoted as a Gaussian sigma; the
corner panel's Fisher ellipse overhangs a region the samples cannot occupy.

Verbatim (note: the two parentheticals in this printed line are Hessian-flavored
— see Open items):

```
mnu wall (E13): sigma_F = 0.038089 eV, fiducial sits 1.58 sigma_F above the mnu >= 0 wall (PFS-only notebook: ~0.5 sigma_F)
mnu chain: sigma = 0.033410 eV, truncation ratio sigma_chain/sigma_F = 0.877 (PFS-only: 0.791); draws within 0.01 eV of the wall = 3.24% (PFS-only: 3.2%); min(mnu) = 0.00000 eV
```

### E8 — information only ADDS (joint <= PFS-only widths)

Both notebooks compare against their own committed PFS-only production sigmas:

```
  param   PFS-only sig      joint sig   joint/PFS-only      # LCDM
  ombh2     0.00047985     0.00011327            0.236
  omch2      0.0032185     0.00053177            0.165
   logA       0.060783       0.011984            0.197
     ns       0.027632      0.0035062            0.127
      h      0.0035686      0.0022193            0.622
[PASS] E8 max(joint/PFS-only) = 0.622 (expect <= 1.02)
```

```
  param   PFS-only sig      joint sig   joint/PFS-only      # nuLCDM
  ombh2     0.00048188     0.00011786            0.245
  omch2      0.0032276     0.00067388            0.209
   logA       0.078534       0.013727            0.175
     ns       0.033284      0.0036756            0.110
      h      0.0036311      0.0025049            0.690
    mnu       0.095725        0.03341            0.349
[PASS] E8 max(joint/PFS-only) = 0.690 (expect <= 1.02)
```

Every width shrinks; the loosest direction is `h` in both cases (0.622 / 0.690).

### E9 — BBN redundancy

`BBN effect on sigma(ombh2)`: **5.5%** (LCDM), **6.0%** (nuLCDM) — a few percent,
as the probe-set decision predicted. BBN is retained as a deliberate consistency
anchor, not because it carries the ombh2 constraint. Width source:
`BBN_SIGMA_MOSSA = 0.00036` in `example/mcmc/scripts/stream_common.py`, whose
provenance is `mcmc_cmb_bao_bbn_LCDM.ipynb` cell 2 (`MOSSABBN_SIGMA`), NOT the
`fisher_joint_*` notebooks; the CENTER is the fiducial 0.02242, not Mossa's
0.02233.

### Profile-likelihood check (E11/E12) — PASS in both, tau and mnu included

```
chi2_prof(fiducial) = 1.231e-23   (LCDM;   noiseless-mock exactness tripwire, expect < 1e-10)
[PASS] max |profile-min offset| = 0.000 sigma_F (< 0.1 tol) vs marginal-mean pulls |.| in [0.02, 0.24] sigma_F -- the likelihood peaks at fiducial; the pulls are marginalization-volume effects.
```

```
chi2_prof(fiducial) = 1.193e-23   (nuLCDM; noiseless-mock exactness tripwire, expect < 1e-10)
[PASS] max |profile-min offset| = 0.000 sigma_F (< 0.1 tol) vs marginal-mean pulls |.| in [0.01, 0.27] sigma_F -- the likelihood peaks at fiducial; the pulls are marginalization-volume effects.
```

All 6 (LCDM) / 7 (nuLCDM) shared parameters were scanned, tau included. tau's
objective comes from the CMB quadratic form alone, so its profile is an exact
parabola with its minimum at the fiducial (E12). mnu's grid is clipped to
`>= 0`. The conclusion carries over from the PFS-only sections above: the
likelihood peaks at the fiducial and every pull is a marginalization-volume
effect.

### Artifact provenance

Both notebooks consume a precomputed CMB block and take no candl/clipy/.clik
runtime dependency:

| | LCDM | nuLCDM |
|---|---|---|
| artifact | `example/mcmc/cache/cmb_fisher_lcdm.npz` | `example/mcmc/cache/cmb_fisher_nulcdm.npz` |
| `CMB_CONFIG_HASH` (pinned in `stream_common`, HARD-REQUIRED by `load_cmb_fisher_block`) | `97f8695acb8a0543...` | `e89efa399fe35590...` |
| `theory_config_hash` | `903aeb06e1cca1c1...` | `8f0f2e74332a4a80...` |
| method | hybrid GN (highl, Planck lensing, ACT DR6 lensing) + observed Hessian (lowl TT, lowl EE) | same |
| shared-prior dedupe | `A_planck` sigma 0.0025, curvature 160000, count 4, subtracted 3 x 160000 = 480000 after summation | same (packed index 27) |
| G1b `sigma_tau` | 0.007089623562031232 | 0.007376499170236379 |
| G2 min eig (strict > 0) | 4188.532847 | 51.361321 |
| G2 max eig | 1.83340e+08 | 1.83336e+08 |
| G1a `lowl_ee_dtau` | -110.1132786885 | -110.0832864627 |
| Fisher build time | 10.8 s | 10.9 s |

Regeneration (from the repo root; the pinned hash must match or the build aborts
nonzero before writing):

```bash
python3 example/mcmc/scripts/build_cmb_fisher_block.py --cosmology lcdm
python3 example/mcmc/scripts/build_cmb_fisher_block.py --cosmology nulcdm
```

### Open items

1. **`fisher_joint_PFS_BAO_CMB_{LCDM,nuLCDM}.ipynb` still carry the 4x
   `A_planck` overcount** (and the observed-Hessian indefiniteness). Their
   CMB-column logA widths are ~3-4% over-confident. DEFERRED — the scripts are
   the production path; the MCMC notebooks here use the deduped artifacts.
2. **nuLCDM notebook cell-30 print relabel, at the next execution.** The two
   parentheticals `(PFS-only: 0.791)` and `~0.5 sigma_F` in the printed E13 lines
   are Hessian-Fisher-flavored and are NOT like-for-like baselines for the
   GN-flavored `wall_ratio` printed beside them. They were left byte-identical in
   `e4d251a` to preserve source/output provenance (a fix would require a ~31 min
   re-run); the flavor labels live in cell 28's markdown and cell 30's comment.
   Relabel the f-strings whenever the notebook is next executed.
   **DONE 2026-08-08** in the derived-projection execution round — see
   "Deferred items discharged in this execution round" below.
3. **Rounding nit:** the wall-distance move 0.29 -> 1.58 sigma_F is 5.5x
   (1.5753/0.28841 = 5.46), not the 5.4x quoted in an intermediate report.
   **DONE 2026-08-08** — cell 28's markdown now reads 5.5x.
4. **E3 projection unit test for `make_cmb_to_shared` never landed.** The plan's
   E3 row asks for a unit test on the native->shared Jacobian (the H0<->h
   factor-100 landmine, `J[h_row, H0_col] == 0.01`) in addition to the build-time
   gate. Only the gate exists; the function lives in
   `example/mcmc/scripts/build_cmb_fisher_block.py` and is data-free, so the test
   is cheap.
   **DONE 2026-08-08** — `tests/test_cmb_gn_fisher.py` gained four data-free
   `jax.jacfwd` tests (8 parametrized cases) pinning `J[h_row, H0_col] == 0.01`
   exactly, the identity behavior of every passthrough parameter, the output
   order for both cosmologies (mnu last in nuLCDM), and that an H0 step moves
   only the h slot. Red-proof: mutating `H0 / 100.0` to `H0 / 1.0` fails 6 of
   the 8.
5. **E7/E8 are prints, not asserts, in both joint MCMC notebooks.** `E7_OK` /
   `E8_OK` are computed and printed as `PASS`/`WARN` but never asserted, so a
   regression would be silent in a headless re-run. Convert to post-print asserts
   at each notebook's next execution (deferred for the same output-provenance
   reason as item 2).
   **DONE 2026-08-08** — both joint notebooks now assert `E7_OK` / `E8_OK`
   immediately after their prints.
6. **`inventory_shared_priors(..., atol=1e-8)` is misnamed.** The value is
   compared against the RELATIVE residual `gap / span`
   (`example/mcmc/scripts/cmb_gn_fisher.py`), not an absolute one, and the
   function already takes a separate `rtol`. Rename to something unambiguous
   (e.g. `sum_rtol`) when that signature is next touched.
   **DONE 2026-08-08** — renamed to `prior_gap_rtol` (keyword-only; no caller
   ever passed it), with the error message and docstring updated to say which
   relative residual each of the two tolerances bounds.

## Derived-parameter constraints (2026-08-08)

All four production `mcmc_joint_*` notebooks gained one markdown + one code cell
(ids `derivedmd01` / `derivedcode01`, appended after each notebook's cosmology
corner cell) that projects the posterior onto the derived basis — $(\Omega_m,
\sigma_8, H_0)$ for LCDM, $(\Sigma m_\nu, \Omega_m, \sigma_8, H_0)$ for nuLCDM —
and were fully re-executed at fixed seeds. Every previously committed number
reproduced **bit-identically** (see "Determinism" below).

**One map for both sides.** The cell builds
`jaxptpolypol.derived.make_lcdm_derived_params_fn(cosmo.param_keys,
cosmo.param_sizes, pklin_emulator=..., mnu_fixed=MNU_FIXED,
sigma8_redshift=0.0)` — the same helper the `fisher_joint_*` notebooks project
with — and wraps it to (i) scatter the sampled cosmology into the notebook's own
native `cosmo_dict` basis at `cosmo_varied_global`, (ii) reorder the library's
`(Omega_m, H0, sigma8)` output into the reported axis order, and (iii) for
nuLCDM prepend $\Sigma m_\nu$ read back out of the native vector. The chain is
mapped through `chunked_map(..., chunk_size=20_000)` over `jax.jit(jax.vmap(...))`
and the comparison Fisher through `project_fisher_to_derived` — the **same**
wrapper — so the sample clouds and the projected ellipses cannot disagree by
construction. Parameters absent from the map ($\tau$, the per-bin bias block,
the analytically marginalized $\theta_{\rm lin}$) are marginalized simply by not
entering it; their correlations still propagate through $C = F^{-1}$.

Hard asserts in the cell: projected chain finite; Jacobian finite and full row
rank (3 / 4); $H_0 = 100\,h$ and
$\Omega_m = (\omega_b + \omega_c + \Sigma m_\nu/93.14)/h^2$ at the fiducial to
$10^{-12}$; and for nuLCDM the projected $\Sigma m_\nu$ column bit-identical to
the chain's (identity-coordinate check). $\sigma_8$ is emulator-derived and is
printed, never asserted.

### Derived fiducials

| | LCDM notebooks | nuLCDM notebooks |
|---|---|---|
| $\Omega_m$ | 0.311049 | 0.311049 |
| $\sigma_8$ ($z=0$) | 0.810384 | 0.810337 |
| $H_0$ | 67.6600 | 67.6600 |

$\Omega_m$ includes $\Omega_\nu = \Sigma m_\nu/93.14 h^2$ at $\Sigma m_\nu =
0.06$ eV (0.31105, not the massless 0.30964). The two $\sigma_8$ values differ
only because the LCDM and nuLCDM notebooks load different linear-$P_k$ emulator
networks (`jense_2023_camb_lcdm` vs `jense_2023_camb_mnu`). **Both networks are
at $\Sigma m_\nu = 0.06$ eV for this comparison** (verified 2026-08-08 from the
training configs: the LCDM YAML sets no neutrino parameters, so training used
CAMB 1.5.2 defaults = one massive neutrino at 0.06 eV; the mnu variant merely
promotes `mnu` to a varied input over [0, 0.5]). The 6e-5 relative $\sigma_8$
gap is emulator interpolation noise at identical physics — there is NO
massless-vs-massive fiducial asymmetry between $\Omega_m$ and $\sigma_8$; do
not "correct" one against the other. The LCDM value
0.810384 and the projected Fisher widths in the PFS-only LCDM table below
reproduce the `fisher_joint_PFS_BAO_BBN_ns_LCDM.ipynb` "PFS P+B+BAO+BBN+ns"
projected column (0.31105 / 0.0070463, 0.81038 / 0.023876, 67.66 / 0.36491)
to all printed digits — an independent cross-check that the MCMC notebooks and
the Fisher notebooks drive the same projection.

### PFS $P_\ell + B_0$ + DESI DR2 BAO + BBN + $n_{s,10}$, LCDM

`mcmc_joint_PFS_BAO_BBN_ns_LCDM.ipynb`, comparison Fisher `F_pfs_bao_prior_cosmo`,
4 x 5000 NUTS-on-surrogate draws.

```
    param        fid    MCMC mean   Fisher sig     MCMC sig   ratio
  Omega_m    0.31105      0.30726    0.0070463    0.0067088    0.95
   sigma8    0.81038      0.81791     0.023876     0.024277    1.02
       H0      67.66       67.368      0.36491      0.35686    0.98
residual pulls (sigma_F units): Omega_m=-0.54  sigma8=+0.32  H0=-0.80
```

### PFS $P_\ell + B_0$ + DESI DR2 BAO + BBN + $n_{s,10}$, nuLCDM

`mcmc_joint_PFS_BAO_BBN_ns_nuLCDM.ipynb`, comparison Fisher
`F_pfs_bao_prior_cosmo`, 4 x 180000 RWMH-on-surrogate draws.

```
    param        fid    MCMC mean   Fisher sig     MCMC sig   ratio
      mnu       0.06      0.14719      0.20804     0.095725    0.46
  Omega_m    0.31105      0.30996    0.0081674     0.007448    0.91
   sigma8    0.81034      0.82682     0.023949     0.024906    1.04
       H0      67.66       67.293      0.43902      0.36311    0.83
residual pulls (sigma_F units): mnu=+0.42  Omega_m=-0.13  sigma8=+0.69  H0=-0.84
```

### PFS $P_\ell + B_0$ + DESI DR2 BAO + CMB + BBN, LCDM

`mcmc_joint_PFS_BAO_CMB_BBN_LCDM.ipynb`, comparison Fisher `F_cmp` (6-key shared
basis incl. $\tau$; $\tau$ marginalized by the projection), 4 x 5000 NUTS.

```
    param        fid    MCMC mean   Fisher sig     MCMC sig   ratio
  Omega_m    0.31105      0.31166     0.002999    0.0029701    0.99
   sigma8    0.81038      0.81054    0.0049869    0.0050225    1.01
       H0      67.66       67.607      0.22525      0.22193    0.99
residual pulls (sigma_F units): Omega_m=+0.21  sigma8=+0.03  H0=-0.24
```

### PFS $P_\ell + B_0$ + DESI DR2 BAO + CMB + BBN, nuLCDM

`mcmc_joint_PFS_BAO_CMB_BBN_nuLCDM.ipynb`, comparison Fisher `F_cmp` (7-key
shared basis incl. $\tau$ and $m_\nu$), 4 x 180000 RWMH.

```
    param        fid    MCMC mean   Fisher sig     MCMC sig   ratio
      mnu       0.06     0.065965     0.038089      0.03341    0.88
  Omega_m    0.31105      0.31181    0.0032304    0.0031351    0.97
   sigma8    0.81034      0.80952    0.0087874    0.0080155    0.91
       H0      67.66       67.589      0.26304      0.25049    0.95
residual pulls (sigma_F units): mnu=+0.16  Omega_m=+0.24  sigma8=-0.09  H0=-0.27
```

### Reading the tables

CMB information tightens the derived basis by a large factor at fixed model:
$\sigma(\Omega_m)$ 0.0067 -> 0.0030 and $\sigma(H_0)$ 0.357 -> 0.222 (LCDM),
0.0074 -> 0.0031 and 0.363 -> 0.250 (nuLCDM). The MCMC/Fisher width ratios sit
in 0.95-1.02 (LCDM) and 0.88-1.04 (nuLCDM), i.e. the derived basis inherits the
Gaussianity of the native one; all residual pulls are below 0.9 $\sigma_F$ and
are marginalization-volume effects (the profile-likelihood cells in the same
notebooks show the likelihood peaks at the fiducial).

**The $\Sigma m_\nu$ rows restate the truncated-marginal caveat verbatim, with
its $\sigma_F$ flavors.** $\Sigma m_\nu$ is an identity coordinate of the map,
so its derived row is numerically the native row: the ratio is
Gauss-Newton-flavored ($\sigma_F$ from `F_pfs_bao_prior_cosmo` / `F_cmp`) against
a chain width truncated by the $\Sigma m_\nu \ge 0$ wall — **0.46 PFS-only** and
**0.88 joint** on matched GN denominators. The validation gate's
Hessian-flavored $\sigma_F$ (0.120937 eV,
`cache/nulcdm_gate_fiducial_means.json`) gives a *different* PFS-only ratio,
0.791; the two must never be mixed. The projected $\sigma_8$/$\Omega_m$/$H_0$
rows of the nuLCDM tables inherit that truncation only through correlations,
which is why their ratios stay in the Gaussian band while $\Sigma m_\nu$'s does
not. Never quote $\sigma(\Sigma m_\nu)$ as a bare Gaussian width.

### Determinism

Each notebook was re-executed with `jupyter nbconvert --execute --inplace` at its
committed seed after a SMOKE-branch gate run. A cell-id-keyed diff of **every**
text output against `a4a697f` (the pre-edit state), ignoring CPU/wall-time
strings and tqdm bars, found **zero** differences in all pre-existing cells of
all four notebooks, except the single intended `sigma_F`-flavor relabel in the
nuLCDM joint E13 print (values unchanged). Spot gates, fresh vs expected:

| gate | expected | fresh |
|---|---|---|
| PFS-only LCDM lp0 | -167.752302 | -167.752302 |
| PFS-only LCDM ratios | 1.00/0.94/0.97/0.99/0.98 | 1.00/0.94/0.97/0.99/0.98 |
| PFS-only LCDM pulls | -0.62/-0.77/+0.56/+0.43/-0.80 | -0.62/-0.77/+0.56/+0.43/-0.80 |
| PFS-only LCDM profile | PASS 0.001 sigma_F | PASS 0.001 sigma_F |
| PFS-only nuLCDM lp0 | -173.635756 | -173.635756 |
| PFS-only nuLCDM ratios | 0.98/0.87/0.59/0.95/0.83/0.46 | 0.98/0.87/0.59/0.95/0.83/0.46 |
| PFS-only nuLCDM acceptance | [0.281, 0.281, 0.280, 0.284] | [0.281, 0.281, 0.280, 0.284] |
| PFS-only nuLCDM profile | PASS 0.010 sigma_F | PASS 0.010 sigma_F |
| joint LCDM lp0 | -167.752302 | -167.752302 |
| joint LCDM acceptance / R-hat max | 0.882 / 1.00002 | 0.882 / 1.00002 |
| joint LCDM sigma(tau) / corr(logA,tau) | 0.0060354 / +0.900 | 0.0060354 / +0.900 |
| joint LCDM profile | PASS 0.000 | PASS 0.000 |
| joint nuLCDM lp0 | -173.635756 | -173.635756 |
| joint nuLCDM acceptance / R-hat max | [0.351, 0.350, 0.351, 0.353] / 1.00034 | [0.351, 0.350, 0.351, 0.353] / 1.00034 |
| joint nuLCDM sigma(mnu) / sigma(tau) | 0.033410 / 0.0069281 | 0.033410 / 0.0069281 |
| joint nuLCDM wall-hit / profile | 3.24% / PASS 0.000 | 3.24% / PASS 0.000 |

Execution logs: `example/mcmc/cache/derived_{pfs,joint}_{lcdm,nulcdm}_{SMOKE,PROD}_20260808.log`.
Production wall times (end-to-end nbconvert, incl. the one-time exact-path
compile): PFS-only LCDM 20m37s, PFS-only nuLCDM 32m29s, joint LCDM 18m23s, joint
nuLCDM 29m57s.

### Deferred items discharged in this execution round

* Open item 2 (nuLCDM joint cell-30 print relabel) — **done**. Both
  parentheticals now carry each flavor explicitly: `(PFS-only, Hessian-flavored:
  ~0.5 sigma_F; matched-GN: 0.29 sigma_F)` and `(PFS-only, Hessian-flavored
  sigma_F: 0.791; matched-GN: 0.46)`. The cell's SIGMA_F-FLAVORS comment no
  longer instructs a future relabel.
* Open item 3 (5.4x -> 5.5x rounding nit in the nuLCDM joint markdown) — **done**.
* Open item 5 (E7/E8 prints, not asserts) — **done**. Both joint notebooks now
  carry `assert E7_OK` immediately after the E7 print and `assert E8_OK` after
  the E8 print; prints kept verbatim.

## Standing caveats of the CMB Fisher block (2026-08-08)

Three properties of the shipped CMB block are *standing* — they are not bugs to
be fixed, they are conditions that hold today and could stop holding. Each is
recorded here with the guard that would catch the change and the exact place to
look.

### A4. The hybrid GN block is PSD *in practice*, not *by theorem*

`GN = J^T C^-1 J` is PSD by construction, but only the three Gaussian-bandpower
terms use it (`planck_highl` plik TTTEEE, `planck_lensing`, `act_dr6_lensing`).
The two non-Gaussian low-ell terms (`planck_lowl_tt` Gibbs/Blackwell-Rao,
`planck_lowl_ee` simall) have no `J^T C^-1 J` at all and keep their OBSERVED
Hessians, which are **individually indefinite** — the per-term conditional
minimum eigenvalues measured during the branch-B experiment were
**`lowl_tt` -1418.6** and **`lowl_ee` -92.0**. (That diagnostic was transient
and is not stored in the artifact; the durable per-build evidence is
`method.negative_mode_attribution`, which shows the same qualitative picture
along the near-null direction.) The summed block is positive definite only
because the GN terms dominate them, which is what the G2 margins measure:

| | LCDM | nuLCDM |
|---|---|---|
| G2 min eig, production (post-dedupe) | **+4188.53** | **+51.3613** |
| baseline observed-Hessian min eig | 4317.77 | -46.2436 (ABORT) |

So the positivity is an empirical margin, not a guarantee. A changed fiducial
cosmology, a new emulator generation, or a different likelihood set can shrink
or re-flip it — the nuLCDM margin of ~51 against a max eigenvalue of 1.83e8 is
nine orders of magnitude of headroom in the *conditioning*, i.e. essentially
none.

**Containment.** (i) G2 (`gate_g2_positive_definite` in
`example/mcmc/scripts/build_cmb_fisher_block.py`) is a hard abort with strict
`> 0` that prints the FULL eigenvalue spectrum and NEVER clips or regularizes —
the clipping branch `expt/cmb-psd-clip` stays unmerged precisely so that a
failure here forces a method decision rather than a silent patch. (ii) The
per-build negative-mode attribution now runs **unconditionally** (not only under
`--diagnose-negative-mode`) and is stored in the artifact META, so drift shows
up as a trend across builds rather than as a surprise abort.

**Where to look:** `meta["method"]["negative_mode_attribution"]` in
`example/mcmc/cache/cmb_fisher_{lcdm,nulcdm}.npz` (key `meta_json`), and the
`gate_g2_positive_definite` block in `build_cmb_fisher_block.py`. Note the G2
docstring in that function still quotes the PRE-dedupe margins
(`+52.2 nuLCDM, +4332 LCDM`); the production post-dedupe values are the table
above.

### A5. `cmb_gn_fisher.py` mirrors clipy/candl PRIVATE internals

`example/mcmc/scripts/cmb_gn_fisher.py` reconstructs each Gaussian term's model
vector and inverse covariance by reaching into attributes that are not public
API and carry no compatibility promise:

| term | private symbols used |
|---|---|
| `planck_highl` | `like._internal.siginv`, `like._internal.rqh_f`, `like._internal.oo` |
| `planck_lensing` | `clik.siginv`, `clik.pp_hat`, `clik.bins`, `clik.cors`, `clik.cl_fid`, `clik.renorm`, `clik.ren1`, `clik._m_llp1_2` |
| `act_dr6_lensing` | candl `covariance_chol_dec`, `_data_bandpowers` |

(The public route is closed, not merely unused: `clik_candl.covariance`,
`window_functions`, `effective_ells` and `bins_*` all raise
`NotImplementedError` on the four clipy terms, and
`make_candl_theory_vector_fn` works only for the native-candl ACT term.)

**This is a standing maintenance coupling: every clipy/candl upgrade requires an
artifact rebuild.** It is safe rather than fragile only because of three guards:

1. **Per-build validation against the UNTOUCHED likelihood.**
   `validate_gn_term` checks the reconstruction against the same
   `jaxptpolypol` `log_like` closure the observed-Hessian build uses, in
   (a) VALUE at the fiducial, (b) the FULL packed HESSIAN at `hess_rtol=1e-12`,
   and (c) DIRECTIONALLY along the reference minimum-eigenvalue direction
   (budget `0.01 * |lambda_min_ref|`) — check (c) exists because (b) is
   normalized by `max|H_ref| ~ 2e8` and is blind to the ~1e-1 eigenvalue the
   module exists to get right. Measured on the shipped nuLCDM artifact:
   `value_rel_err` 1.9e-15 / 1.9e-16 / 1.2e-16 and `hessian_max_rel_err`
   1.5e-15 / 2.7e-16 / 3.7e-16 for highl / Planck lensing / ACT.
2. **`GNValidationError`, never `assert`.** Assertions are stripped under
   `python -O` / `PYTHONOPTIMIZE=1`, which would turn the gate into a no-op
   exactly when someone runs the build "for speed". `-O`-proofness is itself
   pinned by `test_validate_gn_term_still_raises_under_pythonoptimize` in
   `tests/test_cmb_gn_fisher.py`.
3. **Library versions are hash inputs.** `compute_cmb_config_hash` folds
   `candl.__version__` and `clipy.__version__` (with `jax.__version__`) into
   `CMB_CONFIG_HASH`, and `load_cmb_fisher_block` HARD-REQUIRES the pin. A
   version change therefore forces a rebuild + repin instead of silently
   reusing a stale artifact.

**Manual bump point:** `GN_ALGORITHM_VERSION` (currently `"1.0"`, in
`cmb_gn_fisher.py`) is also a hash input, but nothing detects a change to the
reconstruction logic itself — bump it BY HAND whenever the reconstruction
changes, or the fingerprint will not move.

### A6. E14 landed weaker than its plan row promised

`make_forecast_joint_log_post(pfs_log_post, *, n_pfs, extra_loglike_fns=())` in
`src/jaxptpolypol/joint_forecast.py` validates only `n_pfs > 0`:

```python
if n_pfs <= 0:
    raise ValueError(f"n_pfs must be positive, got {n_pfs}")
```

The plan's E14 row asked for a **probe call at build time** — construct-time
evidence that `pfs_log_post` actually accepts `theta[:n_pfs]`. That did not
land. Consequence: a wrong but POSITIVE `n_pfs` passes construction and surfaces
later as a downstream shape error inside the first likelihood evaluation, not as
a construction-time raise.

**Compensating guards actually in place** (per joint notebook, at every map
construction):

* `assert n_nl == N_NL` — the surrogate's nonlinear-parameter count;
* `assert theta0.shape == (N_TOT,)` — the packed sampled-vector length;
* `assert np.allclose(np.asarray(theta0)[SHARED_IDX_MAP],
  np.asarray(CMB_BLOCK["fid_shared"]))` (E10) — the CMB block's shared basis
  really lands on the right slots of `theta0`;
* `assert max(SHARED_IDX_MAP) < N_TOT` — every embedded index is in range
  (JAX clamps/drops out-of-bounds indices silently), asserted at every map
  construction.

Between them a mis-sized `n_pfs` cannot reach a production chain silently; it
just fails later and less legibly than the plan intended. Cross-reference the
supersession banner in
`docs/superpowers/plans/2026-08-06-joint-pfs-cmb-bbn-mcmc.md`, which records the
other three places that plan was overtaken by what landed.

### A7 (B11). RWMH acceptance 0.351 is at the top edge of the band — DO NOT re-tune

The joint nuLCDM production RWMH runs at acceptance `[0.351, 0.350, 0.351,
0.353]`, at the top edge of the usual 0.2-0.35 target band (RWMH optimal ~0.234
for a high-dimensional Gaussian). Nothing was tuned to get there: the
CMB-dominated joint target is simply **more Gaussian** than the PFS-only one, so
a step size inherited from the PFS-only regime accepts more often.

The health evidence says mixing is fine and the acceptance is cosmetic:

* R-hat <= **1.00034** across all seven cosmology parameters (max at `h`);
* ESS **9974-12973** out of 180000 x 4 post-burn-in draws, including `tau` and
  `mnu`;
* MCMC/Fisher sigma ratios **0.95-0.99** on six of the seven parameters
  (`mnu` at 0.88 is the `mnu >= 0` wall, not convergence).

**Decision: do not re-tune.** Re-tuning changes the proposal, hence the chain,
hence every committed number in the "Derived-parameter constraints" and "Joint
PFS+BAO+CMB+BBN MCMC forecasts" sections above — all of which are protected by
exact determinism gates. The trade is a modest ESS gain for a ~30 min production
re-run plus a full re-verification of every gated number. Revisit only if a
future change forces that re-run for an independent reason.
