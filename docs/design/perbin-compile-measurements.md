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
