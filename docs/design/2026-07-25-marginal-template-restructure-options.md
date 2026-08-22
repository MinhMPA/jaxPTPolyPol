# Marginal-likelihood template restructure — design options (verified)

**Date:** 2026-07-25
**Scope:** restructure the marginal-likelihood template extraction (`m0`, `M` = 77 tangent
columns over a 7-bin P+B theory, recomputed every posterior call) so that (a) the forward
compile stops blowing up host RAM, (b) per-eval runtime allows long MH, (c) the NUTS gradient
compiles. Targets: laptop validation (128 GB M-series CPU, x64) and V100 production (16/32 GB
device, fp64).
**Input:** 45-agent exploration, 36 candidate options, each with an *adversarial* verdict.
Where a verdict contradicts an option's own claims, the verdict wins. Every number below is
taken from the verified JSON (`corrected_estimates` preferred over option self-claims).
**Guards:** exact-reconstruction 1e-10, c1 quadratic-ratio ==4.0, Fisher-Schur 1e-8
(`tests/test_marginal_pipeline.py`, `tests/test_marginal_likelihood.py`).

---

## 1. Executive summary

The measured 61–85 GB peak and 132 s-clean / 45 min-under-pressure compile are **compile /
graph-size cost** (HLO op count from the 7-bin Python unroll × 77-wide tangent batch), *not*
runtime buffers — proven by the null result of ~40× grid coarsening. Two other, distinct axes
ride on the same structure: ~1 min/eval runtime (kills 100k-step MH) and the reverse-over-77-
tangent mixed second-order graph (kills NUTS). The single decisive fix is the **exact per-bin
`lax.scan` factorization** (block-diagonal covariance + per-bin lin params make the 77×77 system
exactly seven independent 11×11 systems): it collapses op count ≈/7 (→ ~10–15 GB, ~20–40 s
compile), keeps the answer bit-exact, fits a 16 GB V100, and is the prerequisite for every other
win. Ship it first as a de-risked **Python-unrolled** bridge, then the `lax.scan` form. Layer on:
a **Fisher-preconditioned gradient-free sampler** (ESS/pCN or surrogate-MALA, ~8–10× fewer evals,
exact) for the forward-MH / 2511.20757 path, and — if NUTS is wanted — the **envelope-theorem
`custom_vjp`** for an exact gradient with no second-order graph. The exact per-eval *speed* endgame
(analytic bias-polynomial) and the fast-but-approximate Taylor surrogate are high-leverage upgrades,
not prerequisites.

---

## 2. The bottleneck, as established by verification

Three orthogonal axes, repeatedly conflated by the weaker option write-ups:

**(A) Compile / graph size — the dominant, binding constraint.**
Measured 61–85 GB host RSS peak on first compile; 132 s clean, 45+ min under memory pressure,
kernel death under pressure. This is **op-count / XLA-graph-size** driven, from the 7-bin Python
unroll feeding a 77-wide tangent batch (`make_marginal_templates`, `jax.linearize` +
`vmap(jvp)(eye(77))`). *Decisive evidence:* coarsening the angular/k grids ~40× barely moved the
64 GB — runtime array buffers scale with grid, so if they dominated, 40× coarsening would have cut
~40×; it didn't → the footprint is grid-independent → compile/graph, not runtime.

**(B) Per-eval runtime — ~1 min/eval.** 77 JVPs through the full theory, FLOPs-bound in the shared
emulator + FFTLog + AP + Gauss-Legendre bispectrum work. 100k serial MH steps are therefore
infeasible; a 44 h chain crashed. Grid coarsening and plain tangent chunking do not help — the work
is fixed, not resolution-bound.

**(C) NUTS gradient — mixed second-order graph → kernel death.** `grad(log_post)` for HMC/NUTS is
reverse-over-(linearize+vmap-77): a reverse pass whose residual store multiplies the 77-wide × 7-bin
forward footprint. The 2-bin `jax.hessian` that *did* compile took 15–24 min; the full 7-bin path
dies.

**Why grid coarsening failed:** it attacked array-element size (axis of runtime buffers), but the
binding peak is op-count/compile (axis A) — grid-independent. Same class of mistake recurs in
several options.

**Why plain `lax.map(batch_size=k)` failed:** it chunks the *tangent-batch width* (77→k) — again an
array-size / runtime axis. It does **not** reduce HLO op count: the 7-bin unroll lives *inside* the
linearized theory before batching, and `vmap` never multiplied op count by the batch dim anyway. So
the compile peak (axis A) is untouched; verifiers measured the "64 GB → 9 GB" claim as a category
error (misapplying the CPU-compile number to a device buffer). Worse for NUTS: reverse-over-scan
**stacks** the per-iteration forward residuals across all ceil(77/k) chunks → back to ~77-wide, so
there is no reverse-mode memory win without an added `jax.checkpoint`.

**What actually moves axis A:** rolling the 7-bin Python unroll into one `lax.scan` body (op count
≈/7, compiled once, n_bins-independent) — exploiting that the marginal likelihood factorizes
*exactly* per bin. What moves axis B: replacing the per-step 77-JVP recompute (analytic
bias-polynomial, or one-time Taylor precompute) or cutting the eval *count* (Fisher-preconditioned
sampler). What moves axis C: removing the nested second-order graph (envelope `custom_vjp`, analytic
`M`, or surrogate), or shrinking it via the per-bin scan.

---

## 3. Ranked options (all 36)

Ranked by (i) attacks the dominant verified bottleneck, (ii) exactness preserved, (iii) effort/risk,
(iv) composability. Convergent duplicates collapsed to one rank (others cross-referenced).
Axes: **C**=compile/graph, **R**=runtime/eval, **G**=NUTS-grad, **D**=device-mem (V100 fit).

| # | Option (family) | Verdict | Buys (axis) | Exact | Effort | One-line caveat (verifier) |
|---|---|---|---|---|---|---|
| 1 | **Per-bin `lax.scan` factorization** — F1-B (F1-exact) *[= F2-A, = F4.1 via other families]* | SOUND* | **C** decisive (op /7, 61–85 GB→~10–15 GB, 132 s→~20–40 s), **D** fits 16 GB (~1–2 GB/bin), **R** ~2–7× fewer JVP-FLOPs, enables **G** | Exact ~1e-13 | High, 1–2 d | Reverse-through-scan-through-`linearize` (NUTS) is the one unverified composition — build NUTS `M` via `jacfwd`, smoke-test 1 bin, have `checkpoint` ready |
| 2 | **Per-bin factorization, Python-unrolled bridge** — F1-C | SOUND* | **C** ~7× tangent graph (132 s→~30–50 s), **R** ~4–6× (~10–15 s/step), forward **D** ~1–3 GB fits 16 GB | Exact (== #1) | **Low, ~0.5 d** | De-risked (no scan/reverse-through-linearize); but NUTS grad memory stays 7× (~20–40 GB) → does NOT fit 16 GB V100, dominated by #1 on the grad axis; use as the safe first step |
| 3 | **Analytic bias-polynomial assembly of (m0,M)** (critic) | SOUND* | **C** (removes 77-tangent batch entirely, peak→primal ~10 GB), **R** ~3–15×, kills **G** 2nd-order graph, **D** fits 16–32 GB | Exact to fp64 round-off | **HIGH** (understated) | Needs a new hand-derived tree-bispectrum monomial + c1-insertion basis in `ps_1loop_jax` (cross-repo, error-prone); runtime "40–78×" corrected to 3–15× (shared emulator/FFTLog is the floor) |
| 4 | **`custom_vjp` envelope-theorem gradient (no-logdet)** — F3-A | SOUND* | **G** enabler (no 2nd-order graph; grad ≈1.04× forward, +0 memory over forward) | Exact *only* for 76 linear cols + no-logdet + constant priors | Low-mod, ~30–40 ln | **Hessian-through-it is WRONG** (envelope cross-term hidden by `stop_gradient`; 5.1 vs 10.2) — gradient only; c1 col O(c1²); WRONG for Stream-B NL-priors; grad peak = forward peak → needs #1/#2 first |
| 5 | **Elliptical-slice / pCN, Fisher-Schur reference** — F6a | SOUND* | **R** ~8× fewer evals vs RWMH (EPIS ~2–5 measured at d=26), gradient-free (sidesteps **G**) | Exact for any reference C | Low, ~40 ln | Wraps the same `log_post`: memory/compile unchanged → needs #1/#2 to fit V100; use the FULL correlated whitened C; degrades (not breaks) if strongly non-Gaussian far from fiducial |
| 6 | **Surrogate-gradient MALA + exact Metropolis accept** — F6b | SOUND* | **R** ~7–12× fewer evals (τ~d^{1/3}≈3), forward-only (sidesteps **G**) | Exact (accept uses true `log_post`) | Med, ~50 ln | Do NOT use `blackjax.mala` (it autodiffs → reintroduces the OOM graph); needs ε tuning; needs #1/#2 to fit V100 |
| 7 | **First-order Taylor → polynomial marginal-loglike** — F5-a | SOUND* | **R** 10⁴–10⁵×/step (cost → one-time precompute), free **G**, **C**+**D** for sampling loop (<100 MB) | **Approximate** (Tier-2; O(dθ³) trunc; logdet-ON peak-width error) | Moderate | Circular vs Fisher (partly reproduces it by construction); build **per-bin** (full-graph one-time build ~120–170 GB → OOM); needs cov_inv AND priors fixed |
| 8 | **Importance-reweight a surrogate-NUTS chain with exact `log_post`** — F5-d | SOUND* | Restores **exactness** on #7; exact evals embarrassingly parallel | Asymptotically exact (ESS-gated) | **Lowest** in family | ESS collapse if surrogate under-covers tails; laptop parallelism ~1–2 procs (memory-bound), not 16; each exact eval still 61–85 GB → needs #1/#2 |
| 9 | **Persistent CPU compile cache + config** — F2-D | SOUND* | **C** amortizes compile across processes/restarts (dev loop, cluster preemption) | Exact (replays executable) | Trivial, 1 line | "132 s→1 s" corrected to **~10–40 s** (trace + StableHLO lowering re-run every process); **zero** benefit if the first compile crashes under pressure (nothing written); no effect on peak RSS or runtime |
| 10 | **Delayed-acceptance MH (Taylor stage-1, exact stage-2)** — F5-c | SOUND* | **R** ~2.5–4× fewer exact evals; exact target for any surrogate; no gradient (sidesteps **G**) | Exact stationary law | Moderate | Memory unchanged (61–85 GB) → inert until #1/#2; do NOT embed the 60 s exact eval in `lax.scan`+`cond` (loses checkpointing); serial exact chain caps throughput |
| 11 | **Second-order-in-m0 surrogate (add H=d²m0/dθ²)** — F5-b | SOUND* | De-circularizes the Fisher comparison; leading non-Gaussianity | Approximate, O(dθ³)-consistent *in mean sector only* | Low increment on #7 | "Consistent to O(dθ³)" overstated — dropping d2M leaves same-order gaps in b^TA⁻¹b + logdet; H build is a 676-wide batch (chunk it) |
| 12 | **`shard_map` bins across 1–4 V100** — F4.4 | SOUND* | **R/latency** ~3.5× on 4 GPUs (few-long-chain NUTS) | Exact to fp64 eps (psum reorder) | Moderate | On top of #1, zero laptop benefit; traps: `pvary` the scan carry, BAO **outside** the psum (double-count), null-bin weight-0 (singular Cholesky); MH-throughput prefers #18, not this |
| 13 | **`remat`/checkpoint on theory primal (NUTS reverse only)** — F2-C | SOUND* | **D** caps V100 backward peak to ~1-bin | Exact (Δ≤2e-14) | Low-mod | Not standalone (recompute transient ≈ full forward without #1); worsens compile RSS; no-op-to-harmful on forward MH; under vmap, `dots_with_no_batch_dims_saveable` recomputes the emulator/GL matmuls |
| 14 | **Within-bin P/B sub-staging (exact zero-cross-block)** — F1-D | SOUND* | **R** extra ~1.5–1.8× on #1/#2 (bispectrum tangent 11→5), **D** ~2× headroom on the dominant block | Exact to 1e-12 | ~1 d on top of #1/#2 | Rider only (worthless standalone); the 2 shared params (bGamma3,P_shot) are double-touched — the scatter-add + staged-A==dense-A@1e-12 test is mandatory; breaks if C_PB is enabled |
| 15 | **Micro-tactics audit (jacfwd/concat/donate/unroll)** — F2-E | SOUND* | Decision record; adopt per-bin Cinv storage (55.7→7.9 MB, /7) with #1 | Exact (order/layout only) | ~0 | jacfwd ≠ linearize+vmap (needs an extra full primal for m0); donate_argnums saves ~nothing (x is 208 B); de-concatenate null alone |
| 16 | **Metropolis-within-Gibbs (cosmo block + cheap bias blocks)** — critic | SOUND* | **R** amortizes cosmology grid work over ~n_sub cheap bias substeps; cheap exact bias gradients | Exact (standard MwG) | Med / med-high | Bispectrum is NOT P-separable → substep is ~10–300 ms not "ms"; "keeps current templates()" FALSE (needs table-injection API in both repos); block A still needs #1's bin-scan for memory; b1–σ8 cross-block degeneracy |
| 17 | **Lagged/frozen-M delayed-acceptance (cache M across steps)** — critic | SOUND* | **R** ~2–3× fewer expensive M rebuilds | Exact per fixed-M window | Medium | "Exact for any w" corrected — adaptive refresh = biased (freeze M_ref for production); memory unchanged → needs #1; collapses if M drifts fast (large cosmo moves) |
| 18 | **Multi-process CPU MH chains + cache** — F4.6 | SOUND* | **R/throughput** many chains for R-hat | Exact (independent chains) | Low | Multiplies footprint by N → needs #1 first (8×60 GB≫128 GB); thread-partition flag is a **silent no-op**, realistic aggregate ~1–3× not 8×; must pre-warm cache serially (else N× cold-compile OOM) |
| 19 | **Chains-axis `vmap`/`shard_map` of the forward log-post** — critic | SOUND* | Enables ensemble/SMC/independence kernels needing simultaneous chains; **D** N× on multi-GPU | Exact (~1e-12/lane) | Low-med | Memory ×C is the whole risk → needs footprint reducer; CPU throughput ~1× not C× (core-saturating kernel); incremental value over #18 = simultaneous-chain kernels only |
| 20 | **`lax.map(batch_size=k)` tangent chunk** — F1-A (F1-exact) *[= F2-B, = F4.2]* | SOUND* | **D** V100 forward device-buffer only (~k/77); innermost knob under #1 | Bit-identical (measured 0.0) | Trivial, 1 line | **Does NOT reduce compile op count** (dominant axis) — "64 GB→9 GB" is a category error; no NUTS-memory win (reverse re-stacks residuals); F4.2's literal `.T` double-transposes — drop it |
| 21 | **`custom_vjp` with analytic logdet cotangents** — F3-B | SOUND* | **G** for the include_logdet=True estimand | Exact (to machine eps) | Moderate | Memory claim FALSE by ~3 orders — the logdet backward is the SAME reverse-over-77-JVP tape that OOM'd (removing Cholesky-AD is <0.2%); dominated by #4 when logdet is optional (2511.20757 §II.3 drops it); needs #1 |
| 22 | **Differentiation-order recipe (inner fwd / outer choice)** — F3-D | SOUND* | Memory-vs-flops policy for the logdet term + Schur Hessian | Exact (any ordering) | Low (guidance) | Not standalone; inner ∂/∂θ_lin MUST be forward (reverse = 34× worse); fwd-over-fwd is ~2× forward memory (not "flat") and ~50× runtime |
| 23 | **Fully forward-mode gradient (oracle / last resort)** — F3-C | SOUND* | **G** flat-ish memory for logdet-ON when reverse OOMs; independent oracle to certify #4/#21 | Exact (fwd==rev ~5e-13) | Trivial | ~2× forward memory (not 1×) and **~50×** runtime/gradient → MALA/short-HMC only; standalone V100 fit NO (forward alone > device) |
| 24 | **closure_convert / custom_transpose / jet** — F3-E | SOUND* | closure_convert = glue for vmap/shard_map of #4 | Value-exact | Few lines | Honest negative — jet/custom_transpose give no leverage on the 77-wide trace; `jax.experimental.custom_transpose` import is wrong; closure_convert is optional, not "necessary" |
| 25 | **Many independent chains, pooled** — F6c | SOUND* | **R/throughput** wall-clock ÷ M | Exact (pooling unbiased) | Low | Pure multiplier — 0 standalone benefit, gated on #1 (and its headline numbers silently assume a 20× per-eval speedup #1 does not give); unparallelizable burn-in floor; distinct-RNG-per-process pitfall |
| 26 | **Adaptive tempered SMC (forward-only mutation)** — F6e | SOUND* | Unbiased **ln Z** + multimodal robustness + particle parallelism | Consistent, O(1/N_p) bias | Medium | Highest eval count of F6 (72k); dominated by #5/#6 on the unimodal near-Gaussian target; laptop "3.75 h" corrected to ~15 h even granting 3 s/eval; production V100 evidence path only |
| 27 | **Mixed precision fp32 theory / fp64 algebra** — F4.3 | SOUND* | **R/D** ~2× on V100 (fp64 is half-rate) | **Approximate** (~1e-6) | **HIGH** (~100+ x64-leak sites) | ~5–15% relief on the compile peak only (dtype-invariant); breaks reconstruction 1e-10 + Fisher-Schur 1e-8 (needs fp64-configurable test path, not threshold edits); c1-ratio may hard-fail; fp32 grads risk NUTS divergences |
| 28 | **RWMH baseline (shipped `run_rwmh`)** — F6f | SOUND* | Regression baseline + Fisher cross-check | Exact | Zero (shipped) | Dominated 8–10× by #5/#6 (EPIS ~40 vs ~4–5); diagonal-only whitening leaves ~1.5–2× on the table (full Fisher-Cholesky recovers τ≈26) |
| 29 | **Host-RAM offload of remat residuals** — F4.5 | SOUND* | (V100 only) HBM→host for an un-scanned monolith gradient | Exact | Moderate | Laptop no-op by construction (CPU can't offload; `pinned_host` ValueError); once #1 lands, per-bin residuals fit HBM → offload unnecessary; dominated |
| 30 | **NN/GP emulator of log_post over the Fisher ball** — F5-e | **FLAWED** | (claims cheap inference) | Approximate, no error control | HIGH | Re-incurs the full bottleneck ~1e3–1e4× to build training data; V100 data-gen infeasible without the restructure it claims to replace; realistic budget ~1e4 pts (~1 week); leaves guards green while un-guarding the sampled posterior |
| 31 | **Affine-invariant ensemble (emcee stretch)** — F6d | **FLAWED** | (tuning-free preconditioning) | Exact (stretch DB) | Low-med | EPIS ~40 (really ~100+ at d=26, z^{25} stall) → 8–10× WORSE than the Fisher-whitened RWMH already in the repo; needs #1 to even run; redundant (whitening already does the job) |
| 32 | **Reverse-Woodbury folding of "easy" columns** — critic | **FLAWED** | (shrink the tangent batch) | Exact (Gaussian assoc.) | Low-med | Under AP (the target), **0 of 77** columns are globally foldable — A_shot carries the traced AP Jacobian 1/(α⊥⁴α∥²); removes zero tangents, doesn't touch the bottleneck; P_shot mis-classified as both E and H |

\* All non-FLAWED verdicts are `SOUND_WITH_CAVEATS`; "Exact" reflects the *verified* exactness, and
the caveat column is the load-bearing correction. **Convergent-duplicate reconciliation:** **F2-A**
and **F4.1** are the same per-bin scan as **#1** — all three verifiers re-derived the block-diagonal
factorization from source and independently corrected the compile factor from the optimistic /49 to
a reliable **/7**; F1-B's ~9–10 s/step runtime is a projection that F2-A's verifier corroborates as
2–7×, while F4.1's verifier is more conservatively "throughput ~unchanged" — I trust the 2–7× because
each monolith JVP mechanically propagates through all 7 bins whereas the factorized JVP touches one.
**F1-C** (#2) is the *Python-unrolled* sibling of that scan — same exact factorization and forward
win, lower risk, but keeps the 7-bin unroll so its NUTS-grad memory stays 7× (the axis that killed
the kernel), hence ranked just below #1 and dominated by it on the grad/V100 axis. **F2-B** and
**F4.2** are the same `lax.map` chunk as **#20**; all three verifiers agree it is bit-identical and
compile-neutral, F4.2 additionally caught the `.T` double-transpose bug, F1-A most sharply refuted the
memory arithmetic.

---

## 4. Recommended composition (2-phase)

The dominant bottleneck (compile/graph) has exactly one clean structural fix (#1/#2). Everything else
is a rider that presupposes it. Below, existing guards that already cover a step are named; every new
path needs a **`new-path == monolith` equivalence test at 1e-10 (use `rtol`, not raw `atol` — an
O(1e4) loglike × ~1e-13 re-association can reach ~1e-9 absolute)** plus per-bin analogs of
reconstruction and c1-ratio==4.

### Phase 1 — laptop validation (128 GB CPU, x64)

1. **#2 Python-unrolled per-bin factorization first** (de-risked, ~0.5 d, no scan/reverse-through-
   `linearize` risk). Factor a single-bin theory builder out of `theory.py:384–428` / `joint_fn`, loop
   b in range(7), accumulate `gaussian_marginal_loglike` per bin, split BAO out as a cosmology-only
   additive χ². Delivers the full forward win: 132 s → ~30–50 s compile, ~4–6× per-eval, forward
   ~1–3 GB. This is the safe landing.
2. **#1 `lax.scan` form next** for the n_bins-independent compile (op /7 → ~10–15 GB, ~20–40 s) and the
   bounded reverse-through-scan memory that NUTS needs. Add `make_marginal_log_posterior_scan`
   **alongside** the monolith (do not replace it), so all three shipped pipeline guards + 8 unit tests
   stay literally untouched; add the scan==monolith equivalence test + per-bin reconstruction/c1-ratio.
   Optionally layer **#14** (within-bin P/B sub-staging, extra ~1.5–1.8×) once the base path is
   green — with its mandatory staged-A==dense-A@1e-12 test.
3. **#9 Persistent CPU compile cache** (`jax_compilation_cache_dir` set once) — turns every dev
   iteration into a warm ~10–40 s launch; highest value-per-effort for the iteration loop. **#15**
   per-bin Cinv storage (55.7 → 7.9 MB) adopted with the scan.
4. **Forward-MH validation:** run **#5 (ESS/pCN)** or **#6 (surrogate-MALA)** on the factorized
   `log_post` — exact, gradient-free (sidesteps the NUTS graph), ~8–10× fewer evals than RWMH. **#28
   RWMH** stays as the regression / Fisher cross-check. Build the reference C from the existing
   Fisher-Schur marginal precision, **in whitened space** (`C_white = D⁻¹ C_phys D⁻¹`).
5. **NUTS validation (optional on laptop):** **#4 envelope `custom_vjp`** gives an exact gradient with
   no second-order graph on the factorized forward. Validate `grad_custom == jax.grad(log_post)`
   pointwise to <1e-8; **do not** `jax.hessian` through it (keep the Fisher-Schur Hessian test on the
   naive path). Valid only for include_logdet=False / constant-prior (the target
   `mcmc_joint_PFS_BAO_BBN_ns_LCDM` notebook uses Stream-A constant priors — OK; if logdet is required,
   use **#21** and expect the 77-wide reverse tape back).

### Phase 2 — V100 production (16/32 GB device, fp64)

1. **#1 is mandatory** to fit device memory (per-bin ~1–2 GB < 16 GB; the monolith 61–85 GB was
   host-compile, but the runtime tangent batch will not fit a 16 GB device either). Compile still runs
   in cluster host RAM.
2. **Forward-MH production (2511.20757 path):** **#5/#6** sampler + **#18 multi-process, 1 chain/GPU**
   (with **#9** shared warm cache, pre-warmed serially) or **#19 chains-axis shard_map**. On V100 the
   per-eval drops to sub-second → 100k-equivalent MH in hours. Prefer #18 for many-chain throughput,
   #5/#6 for eval-count efficiency; they compose.
3. **NUTS enablement (judged worth it):** **#4 envelope `custom_vjp`** on the scan forward is the
   recommended enabler — it removes the mixed second-order graph entirely, adds ≈0 memory over the
   forward, and matches the 2511.20757 §II.3 profile/best-fit (no-logdet) convention the pipeline can
   already select. Smoke-test a **1-bin reverse-through-scan** gradient first (the one composition the
   env audit did *not* verify); if reverse-through-`linearize`-in-scan misbehaves, build `M` inside the
   scan via `jax.jacfwd(bin_fn_of_lin)(zeros(11))` so the outer `grad` is a clean reverse-over-scan, and
   keep **#13 `remat`** on the scan body ready to cap the V100 backward peak. For a few-long-chain NUTS,
   **#12 `shard_map` bins** cuts per-gradient latency ~3.5× across 4 GPUs.
4. **If per-eval speed is still the wall (very long exact chains):** two exact endgames —
   **#3 analytic bias-polynomial** (exact, ~3–15× per-eval, also kills the NUTS graph; **HIGH** effort:
   a new tree-bispectrum monomial + c1-insertion basis cross-repo), or the fast-but-approximate **#7
   Taylor surrogate** made exact by **#8 importance reweighting** (diagnose ESS / max-weight before
   trusting). Skip **#27 mixed precision** unless V100 fp64 throughput is the proven blocker — its
   effort is high and it breaks two of three exactness guards.

**New tests required:** (i) scan==monolith equivalence (1e-10, rtol); (ii) per-bin reconstruction +
c1-ratio==4; (iii) for #4, `grad_custom==jax.grad` <1e-8 (and *keep* the Hessian/Schur tests on the
naive path); (iv) for #14, staged-A==dense-A@1e-12 + the two exact-zero-block assertions; (v) for
#7/#8, surrogate-vs-exact at whitened |x|=1,2,3 + ESS/max-weight diagnostics; (vi) for
#5/#6/#16/#18, sampler stationarity vs a long exact-MH / Fisher reference. The three shipped guards
(1e-10, 4.0, 1e-8) stay green throughout because every recommended path is **additive**.

---

## 5. Rejected / dominated options

**FLAWED (do not build):**
- **#30 F5-e NN/GP emulator** — dominated by the training-free Taylor surrogate; must re-run the full
  61–85 GB / ~60 s bottleneck ~1e3–1e4× to generate training data (V100-infeasible without the very
  restructure it tries to avoid); leaves the guards green while no longer guarding the sampled posterior.
- **#31 F6d affine-invariant ensemble** — EPIS ~40 (realistically ~100+ at d=26 from the z^{d−1} stretch
  stall) is 8–10× *worse* than the Fisher-whitened RWMH already shipped; its only edge (tuning-free
  preconditioning) is redundant with `make_whitening_fns`; needs #1 to even run.
- **#32 critic reverse-Woodbury folding** — under AP (the production target) 0 of 77 columns are
  globally cosmology-independent (A_shot carries the traced AP Jacobian), so it removes zero tangents
  and does nothing for the bottleneck; only pays off in the noAP notebooks or *inside* #16.

**Dominated / conditional (keep as documentation or narrow riders, not primary levers):**
- **#22 F3-D, #23 F3-C, #24 F3-E** — supporting AD theory (ordering rule, forward-mode oracle,
  plumbing); no standalone bottleneck leverage.
- **#25 F6c pooled chains** — pure wall-clock multiplier, zero standalone benefit, subsumed by #18/#19.
- **#26 F6e SMC** — genuine value is ln Z + multimodality, not speed; overkill for a unimodal target.
- **#29 F4.5 host offload** — laptop no-op; V100-unnecessary once #1 lands.
- **#27 F4.3 mixed precision** — high-effort, approximate, off the dominant (compile) axis; defer.
- **#21 F3-B logdet `custom_vjp`** — dominated by #4 whenever the Occam factor is optional; its headline
  memory justification is false (retains the OOM tape).

---

## 6. Notable verifier corrections (lessons)

1. **The `lax.map` compile-memory misattribution (#20 / F1-A, F2-B, F4.2).** The headline
   "64 GB → 9 GB via k/77" is a **category error**: the 61–85 GB is the *compile/graph* peak (proven by
   the 40× grid-coarsening null result), while `lax.map` only chunks the *runtime tangent-batch width*.
   It leaves HLO op count — and thus the compile peak — untouched. Same failure class as grid coarsening:
   attacking an array-size axis when the binding constraint is op count. Real (narrower) value: shrinking
   the V100 forward *device* buffer, and as an inner knob under #1.
2. **Scan/`lax.map`-reverse residual stacking refutes the NUTS-memory claim (#20, #13, #12).**
   Reverse-mode through `lax.scan`/`lax.map` **stacks** the per-iteration forward residuals along a new
   axis, so the backward pass holds ~77-wide residuals *simultaneously* — the same order as
   reverse-over-vmap. F1-A's "grad path also benefits" is refuted; any reverse-mode memory win needs an
   explicit `jax.checkpoint` on the scan body (#13), not the scan alone.
3. **The `custom_vjp` Hessian trap and estimand narrowness (#4 F3-A, #21 F3-B).** `stop_gradient` on the
   frozen linear-block optimum hides the envelope cross-term, so differentiating the wrapper a *second*
   time is wrong (measured Hessian 5.1 vs true 10.2; residual 1158 in a large-curvature toy).
   `custom_vjp` defines the *first* derivative only — exactly what NUTS/MALA consume, but the shipped
   Fisher-Schur Hessian guards must stay on the naive path. Compounding: F3-A's gradient is exact only
   for the 76 linear columns **with no-logdet and constant priors** — WRONG for Stream-B NL-dependent
   priors, O(c1²) off on the c1 column, unavailable for include_logdet=True.
4. **The analytic-cotangent memory claim is false by ~3 orders (#21 F3-B).** "May fit where the naive
   kernel OOM'd because the Cholesky/cho_solve tape is gone" is wrong: the logdet backward *is* the same
   reverse-over-77-forward-JVP second-order tape that killed the kernel; the linalg-AD it removes
   (<100 MB) is <0.2 % of the 61–85 GB peak. Lesson: the binding term is the *theory* second-order graph,
   not the linear-algebra AD around it.
5. **The compile-cache "132 s → 1 s" is really ~10–40 s, and 0 when it matters most (#9 F2-D).** JAX's
   persistent cache stores only the XLA `backend.compile()` output; Python tracing + StableHLO lowering
   (which the fact sheet attributes the 132 s to) re-run every process to produce the cache key. And under
   the memory-pressure regime where the first compile *crashes*, no entry is ever written — the
   amortization yields exactly zero benefit precisely when it is most wanted. It solves launch latency,
   never peak RSS or the ~1 min/eval runtime.
6. **Delayed-acceptance / surrogate caveats (#10 F5-c, #8 F5-d, #17).** DA/reweighting reduce the *count*
   of exact evals but leave the 61–85 GB per-eval footprint **unchanged** — inert until #1/#2 lands.
   "Exact for any window w" (#17) is only true per *fixed* M_ref; an indefinite accept-triggered refresh
   is non-diminishing adaptation → residual bias (freeze M_ref for production). Laptop "run many chains"
   is void — one forward saturates RAM (parallelism ~1–2 procs, not 16); and never embed the 60 s exact
   stage-2 inside `lax.scan`+`cond` (it compiles the whole forward into the scan body and loses per-step
   checkpointing).

**Meta-lesson:** the recurring error across the weaker options is attacking an array-size / runtime axis
(grid resolution, tangent-batch width, dtype bytes, sampler eval-count, chain parallelism) while
believing it addresses the compile/graph peak. Only the exact per-bin structural factorization (#1/#2)
reduces the op count that the 40×-grid-coarsening null result fingered as the true driver; the exact
per-eval-*speed* fix (#3) removes the tangent batch outright; everything else is a rider.
