# Branch B (hybrid-GN CMB Fisher block) Review-Fix Wave Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. NOTE: this plan executes as the FIX LOOP of Task 2 of `2026-08-06-joint-pfs-cmb-bbn-mcmc.md` — Round 1 resumes the original branch-B implementer with this plan as the findings package; work happens ON the branch, in its worktree.

> **Close-out (2026-08-07):** executed as fix rounds 1–3 of the parent plan's Task 2 — commits `063d5eb` / `909de00` / `7f699b1` (round 1, merged to master as `e776f4f`), `2405d6a` (round 2), `dea4e0a` (round 3).

**Goal:** Land every accepted finding from the three branch-B reviews (Codex adversarial, superpowers deep review, branch reports) on `expt/cmb-expected-fisher`, regenerate both artifacts, document the two-branch experiment, and merge to master.

**Architecture:** All changes on branch `expt/cmb-expected-fisher` in worktree `/Users/nguyenmn/jaxPTPolyPol-wt-expected-fisher` (base faffcf0). Three tasks: (1) physics fix + gate hardening in `cmb_gn_fisher.py`/`build_cmb_fisher_block.py` with unit tests; (2) content-derived provenance fingerprint hard-required by the loader; (3) regeneration, docs, merge, cleanup. Reports append to the SDD workspace of the parent plan.

**Tech Stack:** Existing branch-B code; clipy/candl (import + attribute introspection only); pytest.

## Global Constraints

- Decisions of record (CONTEXT.md 2026-08-07): hybrid GN adopted; branch A (`expt/cmb-psd-clip`) stays unmerged as documented fallback — do NOT fix or touch it; A_planck fix = shared-prior inventory + duplicate-curvature subtraction AFTER summation, prior widths read programmatically (never hardcoded), per-term log_likes and validation references untouched, hard abort if a shared prior's curvature can't be located; fingerprint = `CMB_CONFIG_HASH`, hard-required, no enforce-if-present grace.
- Gates fail loudly: no `assert` for enforcement (removable by `python -O`); `sys.exit`/`raise` only; no warn-and-continue; no fallback paths.
- OFF-LIMITS: `src/jaxptpolypol/**` (import-only), `tests/test_cmb_priors.py`, all committed notebooks, branch A's worktree.
- The committed `fisher_joint_PFS_BAO_CMB_*` notebooks' A_planck defect is a DEFERRED follow-up — not this plan.
- Suite in the worktree runs with `PYTHONPATH=<worktree>/src` (documented editable-install quirk): baseline `213 passed, 15 deselected`. Additions only.
- Artifacts stay untracked; regenerate via the build script only; quote gate printouts verbatim in reports.
- Review-finding provenance: findings below cite the deep review (DR) and Codex (CX) with their file:line anchors; verify anchors against the working tree before editing (line numbers may drift).

## File Structure

| File (in the worktree) | Change |
|---|---|
| `example/mcmc/scripts/cmb_gn_fisher.py` | shared-prior inventory + dedupe; assert→raise; tolerance tightening + directional check; `HESSIAN_TERMS` guard; `GN_ALGORITHM_VERSION`; minors |
| `example/mcmc/scripts/build_cmb_fisher_block.py` | G2 strict restore; `--diagnose-negative-mode` (also routine per-build print); fingerprint computation; summary-JSON generator absorbed; minors (dead `joint_loglike`) |
| `example/mcmc/scripts/stream_common.py` | ADDITIVE: pinned `CMB_CONFIG_HASH_{LCDM,NULCDM}`; loader hard-require |
| `tests/test_cmb_gn_fisher.py` | NEW: 5 data-free unit tests (incl. negative test + PYTHONOPTIMIZE subprocess test) |
| `tests/test_stream_common_meta.py` | loader-guard tests updated for hard-required fingerprint |
| `docs/design/perbin-compile-measurements.md` | two-branch experiment section (numbers below) |

---

### Task 1: Physics fix + gate hardening + unit tests

**Files:** Modify `cmb_gn_fisher.py`, `build_cmb_fisher_block.py`; Create `tests/test_cmb_gn_fisher.py`.

**Findings addressed (verify each anchor first):**
- CX[high] A_planck 4× (cmb_gn_fisher.py:326-339 region)
- DR-Imp1 G2 `>=`→`>` (build_cmb_fisher_block.py:474; also guard the cond print at :480)
- DR-Imp2/CX[med] asserts vanish under -O (cmb_gn_fisher.py:385-390; convert to `raise GNValidationError(...)`, catch in build at :391)
- DR-Imp3 tolerance shape (cmb_gn_fisher.py:343,370-371): `hess_rtol=1e-12` AND directional check `|vᵀ(H_got−H_ref)v| < 0.01·|λ_min_ref|` along the min-eig direction of the reference
- DR-Imp4 unknown-term silent Hessian path (cmb_gn_fisher.py:306-307): `_BUILDERS.get` miss must raise unless `term_name in HESSIAN_TERMS` (make the dead constants the live guard)
- DR-Imp5 hand-retyped diagnostic (build script :596-600): add `--diagnose-negative-mode` recomputing the per-term attribution along the joint min-eig direction; ALSO run it unconditionally per build (cheap once eigendecomposed) and write STRUCTURED numbers to META (`meta["method"]["negative_mode_attribution"] = {term: float}`), deleting the prose numbers
- DR minors: remove dead `joint_loglike`+`make_joint_loglike_fn` import; write the per-term `"source"` strings into `meta["method"]["sources"]`; drop unused 3rd return of `_clipy_cls_and_tot`; replace silent `S = 0.5*(S+S.T)` with a symmetry check that raises above 1e-8 relative

**Steps:**
- [ ] 1. Shared-prior inventory: implement `inventory_shared_priors(terms) -> dict[str, {"sigma": float, "count": int, "terms": [...]}]` reading each likelihood object's prior metadata (clipy: the priors attached under `all_priors=True`; candl: `required_prior_parameters` + prior specs). Identify parameters appearing in >1 term. Hard-fail (`raise SharedPriorError`) if a shared parameter's prior is non-Gaussian or its width can't be read.
- [ ] 2. Dedupe: after summing the 5 per-term (28×28) blocks, subtract `(count−1)/sigma**2` at the parameter's packed index for every shared prior. Print the inventory + subtraction verbatim in the build log; record it in META (`meta["prior_policy"]`).
- [ ] 3. Inventory regression test (data-free, in `tests/test_cmb_gn_fisher.py`): synthetic 3-term stub where two terms share a Gaussian prior on the same parameter — assert the deduped sum equals the analytic single-count matrix exactly.
- [ ] 4. Gate hardening per the findings list above; define `class GNValidationError(RuntimeError)` in `cmb_gn_fisher.py`.
- [ ] 5. Remaining unit tests (all data-free, no candl/clipy imports at test time — stub objects only): (a) whitener round-trip `|Wx|² == xᵀSx` for random SPD S (both `_whitener_from_inv_cov` and `_whitener_from_cov_chol`); (b) `gn_fisher` == closed-form `JᵀC⁻¹J + prior` for a linear-Gaussian stub, and PSD; (c) NEGATIVE test: `validate_gn_term` raises `GNValidationError` when `model_fn` is perturbed ×(1+1e-6); (d) `make_gn_pieces` raises on an unknown term name; (e) subprocess test: run (c)'s reproduction under `PYTHONOPTIMIZE=1` and assert it STILL raises (proves no assert-based enforcement remains).
- [ ] 6. `pytest tests/test_cmb_gn_fisher.py -q` green; full suite `PYTHONPATH=<worktree>/src pytest tests/ -q` → 218+ passed (213 + ≥5), 15 deselected.
- [ ] 7. Commit (explicit paths): `fix(cmb-B): shared-prior dedupe + hard gates (A_planck 4x, G2 strict, raise-not-assert, term guard, directional tol) + unit tests`

### Task 2: Provenance fingerprint

**Files:** Modify `build_cmb_fisher_block.py`, `cmb_gn_fisher.py` (`GN_ALGORITHM_VERSION = "1.0"`), `stream_common.py` (additive), `tests/test_stream_common_meta.py`.

- [ ] 1. Builder computes `cmb_config_hash`: sha256 over the canonical JSON of {per-file sha256 of each Cl-emulator .npz; per-.clik-dir sha256 of sorted (relpath, file-sha256) listing; ACT dataset identifier; clipy/candl/jax `__version__`; per-term method map + `GN_ALGORITHM_VERSION`; shared-prior inventory; fiducial vector + shared/native basis keys}. Components stored individually in META; hash printed at build end.
- [ ] 2. `stream_common`: `CMB_CONFIG_HASH_LCDM` / `CMB_CONFIG_HASH_NULCDM` pinned constants (placeholder `None` until Task 3 pins real values — loader treats a `None` pin as "refuse to load", so nothing consumes an unpinned artifact); `load_cmb_fisher_block` HARD-requires `meta["cmb_config_hash"]` present AND equal to the pin. Update the 4 existing loader tests' synthetic artifacts to carry a fingerprint; add mismatch + absent-fingerprint failure tests.
- [ ] 3. Full suite green; commit: `feat(cmb-B): content-derived CMB_CONFIG_HASH pinned in stream_common, hard-required by loader`

### Task 3: Regenerate, document, merge, clean up

- [ ] 1. Rebuild both artifacts in the worktree (lcdm, nulcdm; serial). Quote verbatim: inventory/subtraction lines, all gates (G1/G2-strict/G3/validation errs), negative-mode attribution, new sigma_tau + min-eig, the new fingerprints. Expected: logA-related widths loosen ~3.0%/3.8% (the dedupe); min-eig nuLCDM stays positive (the subtraction removes 3×160000 from A_planck's nuisance diagonal — nuisance-block only, marginalization redistributes it; if G2-strict FAILS after dedupe, STOP and report, do not clip).
- [ ] 2. Pin the real hash values in `stream_common`; regenerate `cmb_block_branchB_summary.json` via a new `--summary` flag on the build script (kills the generator-less JSON minor); refresh the regularized joint-proxy numbers.
- [ ] 3. Docs: measurement-doc section "CMB Fisher block: two-branch experiment (2026-08-06/07)" — the indefinite-Hessian finding, per-term attribution, branch comparison table (σ(mnu) triple 0.0387/0.0404/0.0957 pre-dedupe; post-dedupe values quoted alongside), A_planck defect + dedupe numbers, adoption rationale pointer to CONTEXT.md, branch-A fallback note, fisher_joint follow-up note.
- [ ] 4. Full suite in worktree; commit docs; **merge `expt/cmb-expected-fisher` into master** (no-ff), verify `git merge-base --is-ancestor faffcf0 master`; run the suite ON MASTER from the main checkout (expect 218+/15, no PYTHONPATH quirk on master since worktree-only issue); remove BOTH worktrees (`git worktree remove`), keep both branch refs.
- [ ] 5. Ledger the parent plan: Task 2 fix rounds recorded; then the parent plan's Task 2 task-review runs on the full combined diff (210b544..master-head) before Task 3 of the parent plan dispatches.

## Self-review

Spec coverage: every accepted finding from the synthesis maps to a step (CX high×2, CX med, DR Imp 1–7, DR minors, summary-generator minor) ✓. Placeholders: none — steps carry contracts, test definitions, and exact anchors; implementation details are delegated to the resumed implementer who owns the module context ✓. Type consistency: `GNValidationError`, `inventory_shared_priors`, `GN_ALGORITHM_VERSION`, `CMB_CONFIG_HASH_*`, `--diagnose-negative-mode`, `--summary` used consistently across tasks ✓.
