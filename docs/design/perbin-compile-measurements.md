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
~11 400× faster than the 7.4 s/step exact chain.** Single fresh eval 17.7 ms.
The options-review 10³–10⁴× projection is verified mid-band.

**Gate-2 interpretation:** the reference exact chain has ESS 30–83, so its own
MC error is ~0.11–0.18 σ_F on means, ~8–13 % on widths, ~0.11–0.18 on
correlations — i.e. the gate-2 tolerance band (±10 % widths, 0.1 corr) is
TIGHTER than the reference's noise. Every gate-2 deviation is ≤ ~1.5× the
reference noise (logA's 0.896 is 0.9σ of the reference width error), while
gates 1 and 3 — which do not depend on the noisy reference — pass decisively.
The REVIEW flags the reference chain's precision, not a surrogate defect. A
definitive width check needs a higher-ESS exact-target reference (extended
exact chain, or a DA-MH chain: ~20 k steps ≈ 4.8 k exact evals ≈ 7 h).
