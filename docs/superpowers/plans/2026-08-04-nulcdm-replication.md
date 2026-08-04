# nuLCDM P+B Forecast Replication (Fisher done; MCMC) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the nuLCDM PFS P+B MCMC forecast to full parity with the validated LCDM production pipeline — nuLCDM Taylor templates + whitening, a paper-measure (b1σ8) spec variant satisfying the phase gate, a replicated production notebook, and its own validation gate.

**Architecture:** Everything reuses the validated LCDM machinery with mnu added to the sampled cosmology basis (n_NL 26→27): a nuLCDM config block beside the LCDM one in `stream_common.py`, a `--cosmology nulcdm` build mode, a `*_b1s8.yaml` spec variant (measure: b1sigma8) so `load_desi_prior_spec(phase="nulcdm")` passes its own gate legitimately, and a notebook replicated from the CURRENT LCDM notebook (the untracked nuLCDM notebook is a pre-marginalization fossil — replace, do not migrate). Sampling is RWMH-on-surrogate (flag-ON walls forbid NUTS).

**Tech Stack:** Existing jaxptpolypol + ps_1loop_jax machinery; the mnu emulator (`jense_2023_camb_mnu`) already used by the nuLCDM Fisher notebooks.

## Global Constraints

- The LCDM production path must remain untouched and bit-identical: LCDM tripwire `log_post(x0) = -172.996046`; suite baseline at plan start + additions only.
- nuLCDM MCMC assemblies MUST call `load_desi_prior_spec(..., phase="nulcdm")` and therefore MUST use a spec whose b1 row is `measure: b1sigma8` — never bypass the gate by loading with `phase="forecast"` and mutating afterwards (CONTEXT.md nuLCDM note).
- Flag-ON sampling is RWMH/DA only (𝒰[0,3] walls hostile to NUTS).
- **mnu prior (default, confirm at the gate task): flat with the physical bound Σm_ν ≥ 0** implemented as a −inf indicator in the sampled-block prior (RWMH-safe); fiducial 0.06 eV. Cosmology priors otherwise as the LCDM notebook (BBN ombh2 + ns10).
- Heavy compute (template build ~1 h / ~96 GB peak; Fisher/whitening stage) must not run concurrently with other heavy jobs — coordinate with the controller.
- The prerequisite hash-coverage task (theory-config hash includes cosmo basis + emulator + fiducials) must be merged BEFORE the nuLCDM build task runs, so LCDM and nuLCDM caches are guard-distinguishable.
- x64 everywhere; explicit-path commits; verification-before-completion (verbatim outputs in reports).

## File Structure

| File | Change |
|---|---|
| `example/mcmc/scripts/stream_common.py` | additive `NULCDM` config block (emulator path, 6-key basis, fiducial with mnu, fixed-cosmo indices, per-cosmology `template_meta_for`) |
| `example/mcmc/scripts/build_taylor_templates_lcdm.py` | `--cosmology {lcdm,nulcdm}` mode → `taylor_templates_nulcdm[,_c1s].npz`, `taylor_whitening_nulcdm.npz` |
| `src/jaxptpolypol/data/priors/desi_dr1_reanalysis_2511_20757_b1s8.yaml` | spec variant: identical except b1 `measure: b1sigma8` (+ deviations note) |
| `tests/test_desi_priors.py`, `tests/test_stream_common_meta.py` | additive tests (variant loads under phase="nulcdm"; hash disambiguation; mnu-bound prior) |
| `example/mcmc/mcmc_joint_PFS_BAO_BBN_ns_nuLCDM.ipynb` | REPLACED wholesale from the current LCDM notebook (fossil discarded; note in commit message) |
| `example/mcmc/scripts/desi_prior_validation.py` | `--cosmology nulcdm` gate variant (or a thin wrapper script — implementer's call, documented) |
| `CONTEXT.md`, `docs/design/perbin-compile-measurements.md` | nuLCDM sections + hold-note retirement |

## Task List (briefs to be extracted per-task; complete code lives in each task's step blocks, to be finalized against the post-prereq HEAD before dispatch)

1. **nuLCDM config + spec variant + tests** — config block (basis `('ombh2','omch2','logA','ns','h','mnu')`, emulator `jense_2023_camb_mnu` path from the nuLCDM Fisher notebooks, fiducial mnu 0.06, fixed-cosmo recomputed for the 10-key packed basis), the `_b1s8.yaml` variant, loader test `load_desi_prior_spec("..._b1s8", phase="nulcdm")` passes while the base spec raises, hash-disambiguation test (LCDM vs nuLCDM configs → different hashes).
2. **Build mode** — `--cosmology nulcdm`: n_NL 27, mnu-aware `make_lcdm_rescaling_fns` wiring (has_mnu path), whitening stage from the nuLCDM Fisher (mirroring the LCDM build's Fisher/whitening stage), META stamped with the nuLCDM hash + `cosmology: nulcdm`. Toy-scale validation only in this task (real build is Task 3's step).
3. **The heavy build** (controller-scheduled): `--cosmology nulcdm` production build; verify H symmetry ~1e-16, shapes (27), META; record wall/RSS.
4. **Notebook replication** — copy the current LCDM notebook; adapt: emulator/basis/fiducial/template paths/spec variant + `phase="nulcdm"`; sampler branch = RWMH production (200k) instead of NUTS (walls); mnu ≥ 0 indicator in the sampled-block prior; comparison Fisher from the spec variant via `build_prior_sigmas_from_desi_spec` with nuLCDM σ8_ref; mean-vs-mode + b1σ8-measure notes adapted (under flag-ON the chain IS the paper measure — the dual-measure cell reports the *raw* measure via inverse weights, the mirror of the LCDM direction). Smoke-run gate (SMOKE branch) before commit.
5. **nuLCDM validation gate** — the `desi_prior_validation` pattern under the nuLCDM config: surrogate-vs-Hessian widths/corrs, AD-tilted-center means, equivalence dump; PLUS the mnu-specific check: the b1σ8 Jacobian's mnu-direction tilt (∂Σlogσ8/∂mnu < 0) measured and recorded — this is the quantity the phase gate exists to protect.
6. **Production run + docs + CONTEXT close-out** — production RWMH chain in the notebook; CONTEXT.md nuLCDM hold-note retired with pointers; measurement-doc section; ledger stamp.

Sequencing: Task 1–2 can proceed immediately after the hash prereq lands; Task 3 waits for a free machine (no concurrent heavy jobs); Tasks 4–6 serial after 3.
