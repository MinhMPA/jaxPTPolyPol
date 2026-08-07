# jaxPTPolyPol — Domain Context

Glossary of domain terms with a single agreed meaning. Implementation details live in code and ADRs, not here.

## Terms

### Linear nuisance block (θ_lin)
The EFT/stochastic survey parameters in which the joint P+B theory vector is **exactly linear** (verified empirically to the float64 floor, 2026-07-14): per redshift bin `{c0, c2, c4, cfog, a0, a2, P_shot, bGamma3, B_shot, A_shot}` — 10 per bin. Their Gaussian-prior marginalization is analytic. Note: `P_shot` is a single shared parameter entering **both** P (as `P_shot/n̄`) and B (in the shot-noise term) — one column, never duplicated.

### Sampled block (θ_NL)
Parameters the sampler explores: cosmology + per-bin `{b1, b2, bG2}` — matching 2511.20757 Table I's sampled block ("via linearity, all except three can be marginalized over analytically"). **c1 is marginalized** (see c1 section).

### c1 FoG counterterm (FINAL 2026-07-17)
`ps_1loop_jax` folds the c1 fingers-of-god counterterm into the first-order kernel: `Z1_fog(k) = b1 + fμ² − c1·μ²(k/k_nl_rsd)²`, `k_nl_rsd=0.3`, then forms `Z1_fog(k_i)·Z1_fog(k_j)·Z2` — which contains a c1² cross-term. **Size (CORRECTED 2026-08-02, measured on the production data vector):** ≤1.7×10⁻⁵ of the bispectrum block elementwise at |c1|=1 (median ~1×10⁻⁶), equivalently an omitted signal of **4.6×10⁻⁵ σ in whitened data space** (1.2×10⁻³ σ even at a 5σ prior draw, since the χ² scales as c1⁴). The previously recorded "≈6×10⁻⁴" was a stale per-leg estimate at the library default `k_nl_rsd=0.3`, not a data-vector measurement at the production 0.45 — it overstated the effect ~35×. Evidence: `scripts/tier3_c1_bound.py` → `cache/tier3_c1_bound.json`.
**Construction lineage (verified from primary PDF):** this is the published D'Amico-school form — arXiv:2502.14758 eq. 3.14 is *identical* (`Z1 → Z1 − c1μ²(k/k_NL)²`, same `k_NL=0.3 h/Mpc`); that paper **samples** c1 (𝒰[−1,1]) and finds it prior-dominated. The Ivanov school — our reference 2511.20757 (Chudaykin, Ivanov, Philcox), **verified from primary PDF Table I + §II C** — has c1 strictly linear in the model and **analytically marginalizes** it with prior `c1·A_AP·A_amp ~ 𝒩(0,5²) [Mpc/h]²`.
**Decision (Route A, locked 2026-07-17): marginalize c1 via free template linearization.** c1 ∈ θ_lin → **11 params/bin marginalized**. The route-1 templates (`m₀ = t(θ_lin=0)`, `M = ∂t/∂θ_lin`) model the theory as linear in c1 by construction, dropping the c1² — this *is* the 2511.20757 model for c1. No `ps_1loop_jax` change; theory repo stays faithful to its 2502.14758 lineage; inference layer reproduces the Chudaykin treatment. Physics license: c1² is O(k⁴μ⁴), same order as omitted terms — spurious precision either way. Statistics: Fisher already linearizes c1 (Schur column), so Route A makes the Fisher↔MCMC partition identical on both sides.
**Tests:** (1) exact-reconstruction stays float64-sharp via the quadratic-ratio tripwire — c1 is the only θ_lin member with self-curvature and has no θ_lin cross-terms, so residual(2·c1) = 4·residual(c1) exactly; non-c1 columns exact at machine precision. (2) Tier-3: is Route A's dropped c1² visible? **VALIDATED 2026-08-02 — but by a deterministic bound, NOT by the chains** (evidence corrected after adversarial review; the original write-up cited the chains, which have no power here).
   - **Primary evidence — deterministic bound (`scripts/tier3_c1_bound.py` → `cache/tier3_c1_bound.json`).** c1 has *no* bilinear coupling to θ_lin (verified: `dM[:, :, c1] ≡ 0` in all 7 bins), so the entire model difference between the marginalized and sampled treatments is the constant c1² coefficient `q_b = ½ ∂²m0_b/∂c1_b²`. Its whitened norm gives an omitted signal of **4.6×10⁻⁵ σ at |c1|=1 and 1.2×10⁻³ σ at a 5σ prior draw** — a strict upper bound on any parameter shift, ~1000× below anything the chains could resolve. This is what actually establishes Route A is safe.
   - **Secondary — the chain comparison (`cache/tier3_c1_validation.json`)** is a *wiring/consistency* check only: two 200k-draw surrogate chains under identical DESI priors (sampled side rebuilt with c1 in θ_NL: n_NL 26→33, n_lin 77→70) give cosmology mean shifts 0.008–0.049 σ_F, width ratios 0.977–0.994, max corr diff 0.041. Its gate tolerates 0.09–0.14 σ_F with an MC noise floor of 0.028–0.047 σ_F, i.e. **~2000× looser than the effect — it would pass identically with the c1² term deleted or 1000× larger.** Do not cite it as a measurement of the c1² effect.
   - Sampled c1 marginals confirm the physical reason: prior-dominated, mean ≈ 0 (−0.014…0.052), σ = 0.97–1.00 vs the 1.0125 prior width — the data barely constrains c1, so its quadratic term cannot reach cosmology.
   - Exactness caveat: the order-2 surrogate reproduces the c1² dependence exactly **along the pure-c1 direction from θ0**; c1²×δθ cross terms are third order in θ_NL and truncated — on *both* sides of the comparison, so the comparison stays fair.
**Units (Stream-B layer-1 row):** their c1 is in [Mpc/h]² (counterterm ~ c1·k²μ²·…); ours dimensionless via (k/k_nl_rsd)². CORRECTED 2026-07-30: the **production config overrides k_nl_rsd to 0.45** (`build_taylor_templates_lcdm.py` K_NL_RSD=0.45 → `BispectrumTreeModel`), not the ps_1loop_jax default 0.3 (`bs_tree.py:31`) — naive scaling is width 5 [Mpc/h]² → ≈ 5·0.45² ≈ 1.01 dimensionless. Layer-1 derivations MUST read normalization scales from the production config, never from library defaults. **RESOLVED 2026-07-31 (Stream-B Task 3, primary source):** the paper c1 operator is confirmed from ref [160] = arXiv:2110.10161 Eq (3.11), `Z1 → Z1 − c1·μ²(k/kNL^r)²` — **additive, no extra f/b1 factor** (bispectrum uses `Z1^FoG` per leg), character-identical to `bs_tree.py:169` and D'Amico 2502.14758 eq 3.14. Layer-1 factor = **0.45² = 0.2025**, offset 0, `σ_ours = 1.0125` (dimensionless). See `docs/design/desi-convention-map.md` §3.2.
**Corroborating precedent (2511.20757 §II):** their MCMC is Metropolis-Hastings (Montepython) — marginalization, not gradients, is what makes P+B sampling tractable; and they drop the log-det ("Jeffreys prior") term for best-fit extraction, matching our log-det API flag.

### Analytic (Gaussian) marginalization / marginal likelihood
The closed-form integral over θ_lin with Gaussian priors: theory t = m₀(θ_NL) + M(θ_NL)·θ_lin;
−2 ln L_marg = r̃ᵀC⁻¹r̃ − bᵀA⁻¹b + ln det(A Σ_p), with r̃ = d − m₀ − Mμ_p, A = MᵀC⁻¹M + Σ_p⁻¹, b = MᵀC⁻¹r̃.
**Status (2026-07-14): this is the production inference path** for P+B posteriors in this repo — full-space NUTS over all EFT nuisances is computationally infeasible (XLA compile wall), and marginalization is the literature-standard method (BOSS 2112.04515, DESI 2507.13433/2511.20757). Equivalent to the Schur-complement marginalization already used on the Fisher side, which is what makes Fisher↔MCMC comparisons well-defined.

**Mean-vs-mode under the DESI spec (measured 2026-07-31, Stream-B gates):** with the `desi_dr1_reanalysis_2511_20757` priors, the AD-tilted center μ = fid + F⁻¹∇logpost(fid) predicts the posterior **MODE** (Newton-converged agreement ≤0.06 σ_F), but the chain **MEAN** sits ~1.1 σ_F lower along logA (ns −0.28, omch2 +0.26, h +0.15) — persisting with `include_logdet=False`. Mechanism (decomposed by the frozen-R diagnostic, 2026-07-31): the θ_NL-dependent prior widths (layer-2 A_AP·A_amp division; b2/bG2 ∝ 1/σ8²) account for the ns pull entirely and ≈⅓ of logA's (−1.15 → −0.78 with widths frozen); the residual is intrinsic marginal-posterior curvature (A(θ)-volume + model nonlinearity, the tier2-era effect amplified by the wider DESI ctr priors). **Methodology rule: under this spec, mean-level Fisher↔MCMC checks must target the mode (tilted center), not the raw mean; second moments (widths 0.986–1.014, corr diff ≤0.021 vs Hessian-Fisher) are the quantitative gates.** Cross-validated by two numerically identical representations (max|Δlogpost| = 4.2e-12; see the ctr-basis entry below). Evidence: `example/mcmc/cache/{desi_prior_validation_rotation,desi_prior_validation_sigmap,task_e_equivalence}.json` (+ frozen-R diagnostic, same dir).

**Logdet volume tilt (measured 2026-07-28):** with `include_logdet=True` the `ln det(A Σ_p)` term shifts the marginal-posterior MEAN off the fiducial by O(0.1–0.5) σ_F even on a noiseless mock — measured on the 7-bin production chain (all five cosmology pulls share the tilt's sign; cosine(∇logdet, measured shift)=0.76; omch2/h reproduced quantitatively). The Gauss–Newton Fisher cannot contain this term, so **mean-level Fisher↔MCMC comparisons must either use the tilted center μ = fid + F⁻¹∇(−½ ln det A Σ_p) or comparison chains with `include_logdet=False`** — the latter being exactly Chudaykin et al.'s Jeffreys best-fit convention (2511.20757 §II.3). Width and correlation comparisons are unaffected (validated: ratios 0.93–1.08, corr diffs ≤0.12 at 5000 steps). Evidence: `example/mcmc/cache/{tier2_result,logdet_tilt}.json`.

### Notation reference (established 2026-07-15)
The reference chain for symbols and terminology is **arXiv:2511.20757 §II.3 + Table 1** ("Reanalyzing DESI DR1: 2"), whose marginalized likelihood defers to its ref. [122] and Paper 1 (2507.13433); the explicit equations trace to the CLASS-PT paper (arXiv:2004.10607). Code, docstrings, and notebooks use this chain as the citable source. Their phrasing "parameters that appear **quadratically in the likelihood** are marginalized analytically" ≡ our "linear in the model" (a Gaussian likelihood is quadratic in a model-linear parameter).

**POLICY (2026-08-04, user decision): marginalized-nuisance priors are FIDUCIAL-CENTERED in forecast runs.** The θ_lin prior MEANS default to the per-bin fiducial values (widths remain the DESI spec's — per-bin knl factors, correlated ctr block, θ_NL-dependent A_AP·A_amp rescaling all unchanged), so the noiseless-mock posterior peaks at truth up to the residual volume terms (logdet + width-volume; the prior-mean gradient mechanism is removed by construction). The paper's Table-I means (c2→30, c̃→400, shot centering, coevolution bΓ3) remain available as the `marginal_means="spec"` variant for paper-fidelity studies and are what the recorded Stream-B gate artifacts used. Scope: MARGINALIZED nuisances only — the sampled-block priors (b1 measure, b2/bG2) are governed by their own sections.

**Documented deviations from 2511.20757:**
1. **c1 is analytically marginalized, matching 2511.20757 Table I exactly** (verified from primary PDF 2026-07-17; see c1 section). The only residual difference: our *underlying* theory carries the D'Amico c1² (2502.14758 eq. 3.14 lineage), which the linearized marginal model drops — an omitted signal of ≤1.2×10⁻³ σ in data space even at a 5σ draw of a prior-dominated parameter (see the c1 section for the measurement). bΓ3 is marginalized in both (linear in our model — verified).
2. **EFT priors: adopt 2511.20757 Table 1 via an explicit convention map** (decided 2026-07-15, superseding the earlier "keep `eft_eq12_2405_02252`" position). The legacy spec is mistranslated: its `scale` metadata is never applied, so paper widths (~30 for c0, ~400 for c̃, in their normalization) became width-1 in our coefficient units — our own fiducial c0≈12 sits 12σ outside its own 𝒩(0,1) prior. A new packaged spec (`desi_dr1_reanalysis_2511_20757`) carries per-parameter mapped means/widths in our conventions, incl. the P_shot variable shift (theirs = mean-0 deviation-from-Poisson; ours = mean-1 amplitude). Both Fisher and MCMC use the same spec — shared priors are what make the Fisher↔MCMC comparison meaningful. Legacy spec retained for reproducibility of pre-2026-07-15 results.
3. **Sampled bias variables are raw `(b1, b2, bG2)`, not the paper's σ8-scaled combinations `(b1σ8, b2σ8², b𝒢2σ8²)`** (decided 2026-07-15; c1 marginalized per c1 section). Rationale: basis identity with the packed-parameter Fisher block being validated against; Fisher whitening already provides the conditioning benefit. Raw-basis sampling differs from the paper's *measure* for b1: flat-in-raw-b1 vs flat-in-b1σ8 differ by the prior weight Π_b σ8(z_b;θ) — a cosmology tilt (measured by chain reweighting, 2026-08-04: logA +0.1731 σ_F [predicted +0.172], ns +0.1052 [+0.097], others ≤0.03; widths 0.997–1.003, measure-independent as proven; ESS/N 0.9611, zero bound hits; evidence `example/mcmc/cache/b1sigma8_measure.json`). The earlier "report in paper variables via projection" rationale was **INCORRECT** — projection changes the coordinates of samples, not the measure they were drawn under. The b2/bG2 rows are measure-correct (their Gaussian −log(width) normalization already carries the σ8² Jacobian). The spec's b1 row now carries `measure: raw|b1sigma8` (default `raw`; flag-ON ≡ raw reweighted by Π_b σ8, proven identical to 5.95e-14 pointwise / 1.78e-15 at the fiducial, jac_fid = −5.874902, flag-ON lp0 = −178.870948, `example/mcmc/cache/b1sigma8_crosscheck.json`), and `load_desi_prior_spec(phase=...)` **REFUSES** `measure: raw` for `real_data`/`nulcdm` — in nuLCDM the dropped weight lands on Σm_ν. Any real-data or nuLCDM assembly must pass the phase argument. Revisit σ8-scaled sampling only if real-data chains show geometry problems.

**Sampled-block priors (2511.20757 Table 1, mapped to raw variables):** b1 flat (paper: b1σ8 ~ 𝒰[0,3]; unbounded here — the wall is irrelevant on a noiseless mock and harmful to NUTS; bounds remain non-binding at fiducial: b1σ8 = 0.57–0.67 vs [0,3]); b2, bG2 ~ wide Gaussians (paper: 𝒩(0,5²) on the σ8²-scaled combos → width 5/σ8²(z) on raw); c1 ~ mapped 𝒩(0,5²·map). Verified against Table 1 on 2026-07-15; the table's "Type" column as web-extracted mislabels b2/bG2 as marginalized — §II.3 prose ("directly sampled") + the 14−3 arithmetic + their genuine nonlinearity settle it: three sampled bias combos per bin.

**Prior rescaling factors (2511.20757 Table 1 footnote), both cosmology-dependent:**
- `A_AP ≡ (H₀ᶠⁱᵈ/H₀)³ · H(z)/Hᶠⁱᵈ(z) · (D_Aᶠⁱᵈ(z)/D_A(z))²` (=1 at fiducial; computable from `compute_fiducial_distances`/background).
- `A_amp ≡ σ₈²(z)/σ₈,ref²(z)`, ref = Planck 2018 (≈1 at fiducial; σ8 already computed for the derived-parameter projection).
Priors are on `c0·A_AP·A_amp`, `bΓ3·A_AP·A_amp²`, etc. — so the effective prior on our raw coefficient has a **θ_NL-dependent width** (A_AP·A_amp) and, for bΓ3, a **θ_NL-dependent mean** (23/42·(b1−1)).

**API requirement (firm):** the marginal-likelihood API must accept BOTH the prior mean μ_p(θ_NL) AND the prior covariance Σ_p(θ_NL) as functions of θ_NL — constant-only APIs cannot represent the reference priors. Both enter exactly: r̃ = d − m₀ − M·μ_p(θ_NL); A = MᵀC⁻¹M + Σ_p⁻¹(θ_NL). At the fiducial A_AP=A_amp=1, so the Fisher comparison (evaluated at fiducial) sees the un-rescaled widths — the two sides still agree there. Other Table-1 specifics: c2 mean = 30 (not 0); c̃ ~ 𝒩(400,400²) [Mpc/h]⁴.

**Two-layer convention map (the high-risk item):** (1) a per-parameter factor converting our raw coefficient definition (e.g. counterterm `−2k²c0·P`) to the paper's, in matching units; (2) the *θ_NL-dependent* A_AP·A_amp rescaling above. Layer 1 must be derived by matching the coefficient definitions in ps_1loop_jax vs the paper/CLASS-PT — error here silently biases every posterior. **REFINEMENT 2026-07-31 (Stream-B Task 3, dual-derivation reconciliation, primary source):** the "per-parameter factor" framing holds for every layer-1 row EXCEPT the counterterms `c0/c2/c4`. The paper's Table I `c0/c2/c4` priors sit on CLASS-PT's **per-multipole** counterterm basis (arXiv:2004.10607 Eqs 2.21–2.23), while our code coefficients are the **μ-space "tilde"** basis (Eq 2.15). The true map is a **triangular, f-dependent basis rotation** `c̃ = L(f)·c`, NOT a scalar: `our_c0 = paper_c0 − (f/3)paper_c2 + (3f²/35)paper_c4`, `our_c2 = paper_c2 − (6f/7)paper_c4`, `our_c4 = paper_c4`. The paper's diagonal prior becomes a per-bin correlated, f-shifted prior on our coefficients (`c0` mean shift ≈ −8 to −10 [Mpc/h]²; `c2` width inflation ≈ 22–30%; corr(c2,c4) ≈ −0.57 to −0.64). NOT representable as a scalar-factor spec row → **ADOPTED (2026-07-31):** the packaged spec's `c0/c2/c4` rows now ship the **exact correlated per-bin prior** via the `ctr_rotation: multipole_to_tilde` token (cov-mode; sigmap representation), assembled at runtime as `L(f)·diag(σ_paper²)·Lᵀ`. Validated against the rotated-basis (template-rotation) representation at max|Δlogpost| = 4.2e-12; the O(f)-diagonal provisional encoding and the three design alternatives documented in `docs/design/desi-convention-map.md` §3.1 were resolved by the two-branch experiment (sigmap merged).

**Stream-B decisions (grill session 2026-07-30):**
1. **Layer-1 verification standard: dual independent derivation.** Two agents derive the full map independently from 2004.10607 + the ps_1loop_jax code; controller diffs the write-ups; per-parameter unit tests pin each mapped width; CLASS-PT numerical cross-check only if the derivations disagree.
2. **Spec format: mapped + validated.** Each `desi_dr1_reanalysis_2511_20757` entry stores our-convention value, verbatim Table-1 paper value, AND the layer-1 factor; the loader returns our value but raises unless value == paper × factor (+ offset for affine maps like the P_shot mean-1↔mean-0 shift). θ_NL-dependent layer-2 factors are per-entry flags consumed at runtime.
3. **Scope: full bundle.** `make_desi_prior_fns` returns (μ_p_fn, Σ_p_fn, log_prior_nl_fn) — marginalized block and sampled block (b2/bG2 raw widths 5/σ8²(z), b1 flat) from one spec; Fisher consumes the same spec at fiducial.
4. **a0/a2: per-bin exact map** σ_ours = σ_paper·(knl_b/0.45)² — exactly the paper's physical prior at every bin. The factor is a config-parameterized formula (knl_bins is an input to `make_desi_prior_fns`), not a hardcoded constant; layer-1 factors must never bake in library defaults (see the k_nl_rsd correction in the c1 section).
5. **B-side rows (B_shot, A_shot) come from the same Table I** — its 14-per-bin arithmetic (11 marginalized + 3 sampled) includes them; verify verbatim during derivation.
6. **Acceptance gate:** (i) load-time spec validation; (ii) per-parameter width/mean unit tests incl. θ_NL-dependent entries at and off fiducial; (iii) end-to-end surrogate chain under the new priors (Taylor templates are prior-independent — no rebuild) vs Fisher with the same spec: widths/corrs in the established bands, means vs the AD-tilted center fid + F⁻¹∇logpost(fid) (extends the logdet-tilt rule — paper prior means are non-fiducial: c2→30, c̃→400, bΓ3→23/42(b1−1)). Overnight DA exact-chain confirmation optional.
7. **Scope:** Stream-B ends with the LCDM MCMC notebook switched to the new spec (keeping the exact-RWMH sampler; surrogate/DA promotion is a separate follow-up). Fisher example notebooks stay on the legacy spec pending a separate migration; the gate itself exercises the Fisher-from-spec path. nuLCDM assemblies must call `load_desi_prior_spec(..., phase="nulcdm")` — the loader enforces the b1σ8 measure there.

**Joint PFS+CMB MCMC forecasts — CMB treatment (DECIDED 2026-08-06).** For the joint
MCMC forecast notebooks (PFS P+B + Planck CMB [+ BBN], LCDM and nuLCDM), the CMB
enters as a **fiducial-centered Gaussian likelihood term**: −½ Δθᵀ F_cmb Δθ in the
sampled cosmology(+tau) basis, where F_cmb is the Hessian-Fisher of the full candl
stack (Planck high-ℓ TTTEEE + low-ℓ TT + low-ℓ EE simall + Planck/ACT lensing,
CMB nuisances Schur-marginalized) at the fiducial — the same object the
`fisher_joint_PFS_BAO_CMB_*` notebooks build. Rationale: the PFS data vector is a
NOISELESS fiducial mock (Poisson noise in the covariance only), so a consistent
forecast requires every likelihood term to peak at the fiducial; the Gaussian CMB
term preserves the chi2(fid)=0 tripwire, the profile-likelihood check, and exact
MCMC↔Fisher comparability. Rejected: real-Planck candl data (posterior would center
on Planck's best fit ≠ fiducial — reserve for a separate "PFS + real Planck"
analysis); mock-injection into .clik files (plumbing risk, payoff limited to
CMB-side non-Gaussianity, which is not what these forecasts measure). Consequences:
tau becomes a genuinely sampled dimension constrained by the CMB curvature (no tau
prior — post-simall-fix doctrine), and ALL non-Gaussianity of the joint posterior is
PFS-side (volume effects + the Σm_ν wall); the tau marginal is exactly Gaussian by
construction. **IMPLEMENTED 2026-08-07** in `example/mcmc/mcmc_joint_PFS_BAO_CMB_BBN_LCDM.ipynb`
(commit `e1fb52b`, NUTS 4×5000, `log_post_joint(θ0) = −167.752302`) and
`..._nuLCDM.ipynb` (`a71e949`/`e4d251a`, RWMH 4×200k seed 20260806,
`−173.635756`): tau is sampled with no prior and no bound, σ(tau) = 0.0060 (LCDM,
0.843× CMB-alone 0.007090) / 0.0069 (nuLCDM, 0.980× CMB-alone 0.007376 — m_ν
absorbs the degeneracy-breaking), corr(logA,tau) = +0.900/+0.923 chain vs
+0.898/+0.930 Fisher, and both profile checks PASS at 0.000 σ_F with
chi2_prof(fid) ≈ 1.2e-23. Full numbers:
`docs/design/perbin-compile-measurements.md` §"Joint PFS+BAO+CMB+BBN MCMC
forecasts (2026-08-07)".

**Joint PFS+CMB MCMC forecasts — probe set (DECIDED 2026-08-06).** The combination is
**PFS P+B + DESI DR2 BAO + CMB(Gaussian, above) + BBN(fiducial-centered)**, for both
LCDM and nuLCDM. BBN prior: **ombh2 = 0.02242 (fiducial) ± 0.00036** — Mossa et al.
(2020) width, but the center is the fiducial, NOT Mossa's 0.02233: a real-data mean
breaks the peaks-at-fiducial forecast doctrine (0.25σ_BBN spurious pull). BBN is
nearly redundant with primary TT/TE/EE on ombh2 (~2% width effect) and is retained as
a deliberate consistency anchor. **No ns10 prior** in CMB-containing combinations
(CMB constrains ns; matches the fisher_joint exclusion convention). This probe set
has no committed `fisher_joint_*` counterpart (those exclude BBN when CMB is
present), so each MCMC notebook builds its **own matching comparison Fisher inline**:
F_pfs_bao_shared + F_cmb_shared + F_bbn, in the shared basis
(ombh2, omch2, logA, ns, h, tau[, mnu]) with tau constrained only by the CMB block.
The DESI DR1-reanalysis EFT/stochastic priors stay on the PFS block as in production.
F_cmb is provisioned as a **precomputed cached artifact** (`example/mcmc/cache/
cmb_fisher_{lcdm,nulcdm}.npz`, built by `scripts/build_cmb_fisher_block.py` reusing
the fisher_joint cells; META carries likelihood terms, emulator paths, and the
theory-config hash with the enforce-if-present guard pattern) — the MCMC notebooks
load it and never take candl/clipy/.clik as runtime dependencies.
**IMPLEMENTED 2026-08-07** (same two notebooks): the inline comparison Fisher
`F_cmp` lands within ~1% of the committed `fisher_joint_PFS_BAO_CMB_LCDM.ipynb`
column, MCMC/Fisher σ ratios are 0.99–1.01 (LCDM, 6 params) and 0.95–0.99
(nuLCDM, 6 of 7 — m_ν 0.88 is the wall), σ(m_ν) = 0.033410 eV = 2.87× tighter
than PFS-only 0.095725 eV, and information only adds (E8 max joint/PFS-only
0.622 / 0.690). The old `COSMO_PRIORS = {'ombh2','ns'}` are removed to avoid
double-counting BBN/CMB — evidenced by the prior-entry counts 79→77 (whitening)
and 93→91 (comparison), NOT by lp0, which a fiducial-centered Gaussian prior
leaves exactly unchanged. BBN's measured effect on σ(ombh2) is 5.5% (LCDM) /
6.0% (nuLCDM), confirming the "nearly redundant, kept as an anchor" rationale.

**F_cmb method (DECIDED 2026-08-07, two-branch experiment): hybrid Gauss–Newton
expected Fisher**, not the observed Hessian. The nuLCDM observed Hessian is
indefinite (one eig −0.25 raw, ~1e-9 relative, along the H0–mnu geometric
degeneracy — real-Planck-data residual curvature at our fiducial, sourced 93% by
plik TTTEEE per the per-term diagnostic). Adopted branch `expt/cmb-expected-fisher`:
GN (JᵀC⁻¹J) for the Gaussian-bandpower terms (plik high-ℓ TTTEEE, Planck lensing,
ACT DR6 lensing), observed Hessian for the non-Gaussian low-ℓ terms (commander TT,
simall EE — net positive along the degeneracy direction); per-build validation of
the reconstructed models against the untouched log_like (value + full 28-dim
Hessian, ~1e-15). Rejected: PSD-clip (branch `expt/cmb-psd-clip`, kept unmerged as
the documented fallback) — a clipped eigenvalue is a repair, not a forecast
statement, and it retains real-data contamination in retained directions. The
joint-proxy forecast is insensitive to the choice (σ(mnu) 0.0387 GN vs 0.0404
clip vs 0.0957 PFS-only), so the decision is doctrinal. Known cross-cutting defect
found by review during the experiment (fix required before artifacts are consumed):
the shared A_planck internal prior is counted once per Planck term (4×) in ANY
sum-of-terms Fisher/Hessian with all_priors=True — this also afflicts the committed
`fisher_joint_PFS_BAO_CMB_*` notebooks (~3–4% overconfident logA widths; separate
follow-up). Fix mechanism (DECIDED 2026-08-07): automated shared-prior inventory
across terms + duplicate-curvature subtraction after summation, prior widths read
programmatically from the likelihood objects (per-term log_likes and their
validation references stay untouched); hard abort if a shared prior's curvature
cannot be located analytically. Provenance (DECIDED 2026-08-07): artifacts carry a
content-derived `CMB_CONFIG_HASH` (sha256 over Cl-emulator file hashes, .clik
content listings, clipy/candl/jax versions, per-term method map +
GN_ALGORITHM_VERSION, shared-prior inventory, fiducial+basis) pinned in
stream_common and HARD-REQUIRED by `load_cmb_fisher_block` — absent or mismatched
fingerprint fails; no enforce-if-present grace (no legacy CMB artifacts exist).
**IMPLEMENTED and CONSUMED 2026-08-07**: `example/mcmc/cache/cmb_fisher_{lcdm,
nulcdm}.npz` (hashes pinned `97f8695a…` / `e89efa39…`, G2 min eig 4188.53 /
51.3613 strict > 0, A_planck dedupe 3×160000 subtracted after summation) are
loaded by both joint MCMC notebooks. The deferred `fisher_joint_PFS_BAO_CMB_*`
A_planck overcount remains OPEN (~3–4% overconfident logA there). Measured
consequences of the hybrid-GN choice for the m_ν wall, with σ_F flavors labelled:
on matched Gauss–Newton denominators the truncation ratio moves 0.46 → 0.877 and
the fiducial sits 0.29 → 1.58 σ_F above the wall (a large weakening), while the
flavor-free evidence shows the bound is still load-bearing (wall-hit 3.24%,
unchanged from PFS-only 3.24%; min(m_ν) = 0; σ-ratio 0.88 vs 0.95–0.99 for every
other parameter). The 0.791 truncation ratio in the older notes is the
Hessian-Fisher flavor (σ_F = 0.120937 eV) and must not be compared with the GN
0.877.

**nuLCDM production status (2026-08-05, retires the "on hold" note in Stream-B decision 7 above).** The nuLCDM P+B MCMC is now a committed production run: `example/mcmc/mcmc_joint_PFS_BAO_BBN_ns_nuLCDM.ipynb` — 6-key mnu-last cosmology basis (n_NL 26→27), RWMH 200k×4 draws (20k burn) on the prior-independent Taylor surrogate, seed 20260806, R-hat ≤ 1.0006, acceptance ≈ 0.28, and `log_post(x0) = −173.635756` under the fiducial-centered default (the earlier spec-means value −178.879579 is the `desi_paper` variant). Priors load via `load_desi_prior_spec("desi_dr1_reanalysis_2511_20757_b1s8", phase="nulcdm")` — the `_b1s8` variant's b1σ8 measure is MANDATORY in nuLCDM (the phase gate REFUSES `measure: raw`, since the dropped Π_b σ8 weight tilts Σm_ν; deviation 3). The Σm_ν prior is flat with the physical bound Σm_ν ≥ 0 (a −∞ indicator, RWMH-safe / NUTS-hostile), fiducial 0.06 eV. **The m_ν marginal is a truncated shape by construction, not a detection:** observed chain width 0.791× the untruncated Fisher σ (1-D truncated-normal analytic 0.697), 3.2% wall-hit — `example/mcmc/cache/nulcdm_gate_fiducial_means.json` (spec-means variant: `nulcdm_gate_spec_means.json`). The bound is load-bearing: the unbounded diagnostic collapsed to a spurious extrapolation mode at m_ν ≈ −0.33 eV outside the emulator/surrogate domain — `example/mcmc/cache/nulcdm_gate_fiducial_means_unbounded.json`, verdict **INVALID_CONFIGURATION**. Full convergence/measure/mnu-Jacobian evidence: `scripts/desi_prior_validation.py --cosmology nulcdm`.

### Templates (m₀, M)
m₀(θ_NL) = theory at θ_lin = 0; M(θ_NL) = ∂t/∂θ_lin (exact, since t is linear in θ_lin). Template columns may depend on θ_NL (e.g. cfog's column ∝ (b1+fμ²)²) — that does not break the decomposition.
