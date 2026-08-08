# fisher_joint CMB Port + Number Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retire the last committed defective numbers in the repo (the `fisher_joint_PFS_BAO_CMB_*` notebooks' A_planck 4× overcount, indefinite nuLCDM Hessian, and inverted σ(tau)) by making those notebooks consume the cached hybrid-GN CMB artifact, and make the two report-attested-only numbers re-derivable from committed artifacts.

> **Supersession note (2026-08-08, added at close-out; plan body otherwise kept as-written for provenance).**
> The Goal sentence above overclaims. This plan retired those three defects in the two
> `fisher_joint_PFS_BAO_CMB_*` notebooks **only**. It did not retire "the last committed defective
> numbers in the repo": `example/fisher/fisher_cmb_candl_LCDM.ipynb` and
> `example/fisher/fisher_cmb_candl_nuLCDM.ipynb` were out of scope, remain un-ported, and still carry
> the same uncorrected 4× `A_planck` overcount and indefinite committed Fishers — plus outputs that are
> stale against their own source. Both now carry a `SUPERSEDED / stale outputs` banner as cell 0;
> re-executing them is explicitly not recommended (it would reintroduce the candl/clipy/`.clik` runtime
> dependency this plan removed). Details: `docs/design/perbin-compile-measurements.md`, section
> "fisher_joint CMB port (2026-08-08)" → "Scope: two sibling candl notebooks remain UN-PORTED and
> defective". Use `stream_common.load_cmb_fisher_block(cosmology)` for any CMB Fisher block.

**Architecture:** The two Fisher notebooks currently rebuild the CMB Fisher inline (load 5 candl/clipy terms → `jax.hessian` of the summed log-like → Schur-marginalize nuisances → project H0→h). That entire block is replaced by `stream_common.load_cmb_fisher_block(cosmology)["F_shared"]`, which is already nuisance-marginalized, shared-basis, A_planck-deduped, hybrid-GN (PD), and fingerprint-guarded — one source of truth with the MCMC notebooks. Separately, the build script gains pre-dedupe width bookkeeping in META so the dedupe's effect is re-derivable, and the `--summary` output gains the mnu-fixed σ(tau) refit.

**Tech Stack:** Existing `stream_common` loader + cached artifacts; the Fisher notebooks' existing `embed_fisher`/`marginalized_fisher_block` machinery; no new dependencies (the port REMOVES candl/clipy/.clik as runtime dependencies of the two notebooks).

## Global Constraints

- The artifact is authoritative: `cmb_fisher_{lcdm,nulcdm}.npz`, shared basis `('ombh2','omch2','logA','ns','h','tau'[,'mnu'])`, pinned `CMB_CONFIG_HASH_{LCDM,NULCDM}` = `97f8695acb8a0543…` / `e89efa399fe35590…`, hard-required by the loader. Notebooks MUST NOT bypass the loader or rebuild a CMB Fisher inline.
- Numbers that MUST change on re-execution (this is the point of the task) and must be reported explicitly: CMB-column and CMB-containing-combination logA widths loosen ~3–4% (A_planck dedupe); the nuLCDM CMB-only σ(h)/σ(omch2)/σ(mnu) stop being `---`/NaN (indefinite Hessian retired); nuLCDM σ(tau) moves from the inverted-through-negative-curvature 0.0069061 to the artifact's 0.007376.
- Numbers that MUST NOT change: every non-CMB column (PFS-only, BAO-only, PFS+BAO, and their BBN/ns10 variants) — these never touched the CMB block. Treat any drift there as a STOP condition.
- Off-limits: `src/jaxptpolypol/cmb.py`, `tests/test_cmb_priors.py` (user WIP), the four `mcmc_joint_*` notebooks, `mcmc_cmb_bao_bbn_LCDM.ipynb`, spec YAMLs, CONTEXT.md decision paragraphs (append-only if needed).
- Notebook edits ONLY via nbformat scripts; semantic cell-diff before every commit; SMOKE/short-run gate where the notebook has one, else a dry validation cell run; WIP-commit before long executions; serial executions; verification-before-completion with verbatim outputs.
- Suite must remain green (`pytest tests/ -q`; current 236 passed, 15 deselected — the count moves only if the user lands their WIP).
- x64 everywhere.

## File Structure

| File | Change |
|---|---|
| `example/mcmc/scripts/build_cmb_fisher_block.py` | META gains `prior_policy.pre_dedupe_marginal_sigmas` + `post_dedupe_marginal_sigmas` + `dedupe_width_shift_pct`; `--summary` gains the mnu-fixed σ(tau) refit |
| `example/fisher/fisher_joint_PFS_BAO_CMB_LCDM.ipynb` | inline CMB Hessian block → `load_cmb_fisher_block("lcdm")`; re-executed |
| `example/fisher/fisher_joint_PFS_BAO_CMB_nuLCDM.ipynb` | same with `"nulcdm"`; re-executed |
| `docs/design/perbin-compile-measurements.md` | new section + Open items 1/2/3-provenance closed |

---

### Task 1: Build-script provenance (items 2 and 3)

**Files:** Modify `example/mcmc/scripts/build_cmb_fisher_block.py`; artifacts regenerated.

**Interfaces produced:** META key `prior_policy.dedupe_width_effect = {"pre_dedupe_marginal_sigmas": [...], "post_dedupe_marginal_sigmas": [...], "shift_pct": [...], "basis": [...]}`; `--summary` JSON key `sigma_tau_mnu_fixed_refit` (nuLCDM only).

- [ ] **Step 1: Add pre/post-dedupe width bookkeeping.** In the build path, the dedupe (`apply_shared_prior_dedupe`) acts on the summed full matrix before `marginalized_fisher_block`. Marginalize BOTH the pre-dedupe and post-dedupe full matrices to the cosmo block, project both to shared, take `sqrt(diag(inv(·)))` of each, and store all three vectors (pre, post, percent shift) in META under `prior_policy.dedupe_width_effect`, in `shared_keys` order. Cost is one extra marginalize+invert (milliseconds).

- [ ] **Step 2: Add the mnu-fixed σ(tau) refit to `--summary` (nuLCDM only).**

```python
# Item-3 provenance: sigma(tau) with mnu held fixed, to quantify how much of
# nuLCDM's weaker tau sharpening is mnu absorbing the degeneracy-breaking.
if cosmology == "nulcdm":
    keep = [i for i, k in enumerate(shared_keys) if k != "mnu"]
    F_fixed = np.asarray(F_shared)[np.ix_(keep, keep)]
    i_tau = [k for k in shared_keys if k != "mnu"].index("tau")
    summary["sigma_tau_mnu_fixed_refit"] = float(np.sqrt(np.linalg.inv(F_fixed)[i_tau, i_tau]))
```

- [ ] **Step 3: Rebuild both artifacts.** Run the two build commands. VERIFY: `F_cmb_shared`, `fid_shared`, `sigma_tau` and the pinned `cmb_config_hash` are **byte-identical** to the pre-rebuild artifacts (the new META fields are outputs, not hash inputs — a hash change means step 1 accidentally touched a hash input and is a STOP condition). Quote the md5s before/after and the gate printouts.

- [ ] **Step 4: Verify the recorded numbers reproduce the report claims.** The stored `shift_pct` for logA must be ≈ +3.0% (lcdm) / +3.8% (nulcdm) — the previously report-attested-only numbers, now artifact-backed. Quote them.

- [ ] **Step 5: Commit.**

```bash
git add example/mcmc/scripts/build_cmb_fisher_block.py
git commit -m "feat(cmb): record pre/post-dedupe marginal widths + mnu-fixed sigma(tau) refit in artifact META (number provenance)"
```

---

### Task 2: Port `fisher_joint_PFS_BAO_CMB_LCDM.ipynb` to the cached artifact

**Files:** Modify `example/fisher/fisher_joint_PFS_BAO_CMB_LCDM.ipynb`.

**Interfaces consumed:** `load_cmb_fisher_block("lcdm") -> {"F_shared" (6×6, order ombh2,omch2,logA,ns,h,tau), "fid_shared", "shared_keys", "sigma_tau", "meta"}` (from `example/mcmc/scripts/stream_common.py`; the notebook must add that scripts dir to `sys.path` the way the MCMC notebooks do, or import via the same idiom already used elsewhere in `example/`).

- [ ] **Step 1: Read the notebook and identify the exact cell range to replace.** The block spans: the candl/clipy term loading (`load_candl_likelihood` ×5), the nuisance union, `make_candl_loglike_fn`/`make_joint_loglike_fn`, `jax.hessian` → `F_cmb_full`, `marginalized_fisher_block(..., cmb_cosmo_idx)`, and `project_fisher_to_derived(..., cmb_to_shared)`. Record the cell ids. Everything downstream consumes `F_cmb_shared` — that variable name is the seam.

- [ ] **Step 2: Replace with the loader.** New cell body (keeping the variable name `F_cmb_shared` so downstream cells are untouched):

```python
# CMB Fisher block: loaded from the cached hybrid-GN artifact (single source of
# truth with the joint MCMC notebooks). Replaces the former inline observed-Hessian
# construction, which (a) counted the shared A_planck prior once per Planck term
# (4x -> ~3-4% over-confident logA) and (b) was indefinite for nuLCDM.
# Provenance/regeneration: see the artifact META and
# docs/design/perbin-compile-measurements.md "CMB Fisher block: two-branch experiment".
CMB_BLOCK = load_cmb_fisher_block("lcdm")
assert tuple(CMB_BLOCK["shared_keys"]) == SHARED_KEYS, (CMB_BLOCK["shared_keys"], SHARED_KEYS)
F_cmb_shared = jnp.asarray(CMB_BLOCK["F_shared"])
fid_cmb_shared = np.asarray(CMB_BLOCK["fid_shared"])
assert np.allclose(fid_cmb_shared, np.array([FIDUCIAL[k] for k in SHARED_KEYS])), \
    "artifact fiducial disagrees with the notebook fiducial"
assert np.min(np.linalg.eigvalsh(np.asarray(F_cmb_shared))) > 0.0, "CMB block not PD"
print(f"CMB block: {CMB_BLOCK['meta']['method']['per_term']}")
print(f"  A_planck dedupe: {CMB_BLOCK['meta']['prior_policy']['shared_prior_inventory']}")
print(f"  sigma(tau) CMB-alone = {CMB_BLOCK['sigma_tau']:.6f}")
```

Delete the now-dead candl/clipy imports and constants that this change orphans (and ONLY those — CLAUDE.md §3).

- [ ] **Step 3: Semantic cell-diff** vs the pre-edit git state; confirm only the intended cells changed and no downstream cell was touched.

- [ ] **Step 4: Re-execute the notebook** (nohup, tagged log under `example/mcmc/cache/`, bounded `kill -0` waits).

- [ ] **Step 5: Verify the expected/forbidden changes.** Print a before/after table for EVERY column. Expected: CMB and CMB-containing columns move (logA ~3–4% looser); `sigma(tau)` CMB-alone becomes 0.007090. Forbidden: any change in PFS-only / BAO-only / PFS+BAO / BBN+ns10 columns — STOP if any moves. Quote both lists verbatim.

- [ ] **Step 6: Commit.**

```bash
git add example/fisher/fisher_joint_PFS_BAO_CMB_LCDM.ipynb
git commit -m "fix(fisher): LCDM joint CMB Fisher from the deduped hybrid-GN artifact (retires the 4x A_planck overcount)"
```

---

### Task 3: Port `fisher_joint_PFS_BAO_CMB_nuLCDM.ipynb` to the cached artifact

**Files:** Modify `example/fisher/fisher_joint_PFS_BAO_CMB_nuLCDM.ipynb`.

Same as Task 2 with these differences (the implementer sees only this task):

- [ ] **Step 1: Identify the inline CMB block cells** (same construction, 7-key native basis incl. mnu, `jense_2023_camb_mnu_Cl_*` emulators). Seam variable is again `F_cmb_shared`.

- [ ] **Step 2: Replace with the loader:**

```python
CMB_BLOCK = load_cmb_fisher_block("nulcdm")
assert tuple(CMB_BLOCK["shared_keys"]) == SHARED_KEYS, (CMB_BLOCK["shared_keys"], SHARED_KEYS)
F_cmb_shared = jnp.asarray(CMB_BLOCK["F_shared"])
fid_cmb_shared = np.asarray(CMB_BLOCK["fid_shared"])
assert np.allclose(fid_cmb_shared, np.array([FIDUCIAL[k] for k in SHARED_KEYS]))
assert np.min(np.linalg.eigvalsh(np.asarray(F_cmb_shared))) > 0.0, "CMB block not PD"
print(f"CMB block: {CMB_BLOCK['meta']['method']['per_term']}")
print(f"  sigma(tau) CMB-alone = {CMB_BLOCK['sigma_tau']:.6f}")
```

Delete the orphaned candl/clipy imports/constants.

- [ ] **Step 3: Semantic cell-diff.**

- [ ] **Step 4: Re-execute** (serial — after Task 2's run completes).

- [ ] **Step 5: Verify.** Expected changes, all of which must be reported verbatim: the CMB-only column's `---` / `invalid value encountered in sqrt` entries for h/omch2/mnu are GONE (finite values); σ(tau) CMB-alone 0.0069061 → 0.007376; logA widths ~3.8% looser in CMB-containing columns. Forbidden: any non-CMB column moving — STOP. Additionally assert in-notebook that the previously-negative eigenvalue is gone: `min(eigvalsh(F_cmb_shared)) > 0`.

- [ ] **Step 6: Commit.**

```bash
git add example/fisher/fisher_joint_PFS_BAO_CMB_nuLCDM.ipynb
git commit -m "fix(fisher): nuLCDM joint CMB Fisher from the deduped hybrid-GN artifact (retires 4x A_planck + indefinite Hessian + inverted sigma(tau))"
```

---

### Task 4: Docs close-out

**Files:** Modify `docs/design/perbin-compile-measurements.md`.

- [ ] **Step 1: New section** "fisher_joint CMB port (2026-08-08)": what changed and why (three defects retired), the before/after column tables for both notebooks, the confirmation that non-CMB columns are untouched, the note that these notebooks no longer import candl/clipy at runtime, and the artifact-provenance pointer.
- [ ] **Step 2: Close Open item 1** (mark DONE with the commit refs); annotate the dedupe-shift entry and the 0.927 entry as artifact-backed now, citing the new META keys and the `--summary` key from Task 1.
- [ ] **Step 3:** `pytest tests/ -q` (docs+script only; expect unchanged) and commit.

```bash
git add docs/design/perbin-compile-measurements.md
git commit -m "docs(fisher): record the CMB-artifact port results and close the A_planck open item"
```

## Self-review

Spec coverage: item 1 → Tasks 2+3 (both defective notebooks, all three defects); item 2 → Task 1 Steps 1/4 (dedupe shift artifact-backed); item 3 → Task 1 Step 2 (0.927 re-derivable); docs → Task 4 ✓. Placeholders: none — every code step carries its code, every verification names its expected and forbidden changes. Type consistency: `load_cmb_fisher_block(cosmology) -> {"F_shared","fid_shared","shared_keys","sigma_tau","meta"}` and the `F_cmb_shared` seam variable used identically in Tasks 2 and 3; META key `prior_policy.dedupe_width_effect` produced in Task 1 and cited in Task 4 ✓.
