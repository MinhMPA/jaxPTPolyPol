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
