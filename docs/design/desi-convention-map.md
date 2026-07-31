# DESI (2511.20757) ↔ ps_1loop_jax convention map — AUTHORITATIVE (reconciled)

**Status:** reconciliation of two independent derivations (A =
`desi-convention-map-A.md`, B = `desi-convention-map-B.md`) into the authoritative
layer-1 + layer-2 prior-convention map. Produced by Stream-B Task 3.

**Verdict summary (three controller battlegrounds):**
1. **c0/c2/c4 basis — A is CORRECT** (re-derived from CLASS-PT primary source). The
   paper's Table I priors sit on the CLASS-PT **per-multipole** counterterm basis (Eqs
   2.21–2.23), while our code coefficients are the **μ-space "tilde"** basis (Eq 2.15).
   The true map is a **triangular, f-dependent basis rotation**, not a scalar factor.
   B's "factor 1, no caveat" is a genuine miss. → **DONE_WITH_CONCERNS / escalation.**
2. **c1 operator form — RESOLVED.** Ref [160] (arXiv:2110.10161) Eq (3.11) prints the
   operator: `Z1 → Z1 − c1 μ²(k/k_NL^r)²`, **additive, no extra (b1+fμ²)**. Factor
   `0.45² = 0.2025` confirmed. Both A and B were right on the number; the deferred
   operator is now verified from the primary source.
3. **Layer-2 stochastic flags — A and B AGREE and match the printed Table I.** Every
   stochastic row's runtime rescale flag is resolved verbatim below.

---

## Two-layer structure (CONTEXT.md)

- **Layer 1** (this document): a per-parameter conversion `ours = paper × factor +
  offset` in matching units, derived by equating operators at identical `(k, μ, z)`.
  **Exception (battleground 1):** the counterterms `c0/c2/c4` are related by a
  triangular *matrix* `L(f)`, not a scalar — documented in full below.
- **Layer 2** (runtime, θ_NL-dependent): the `A_AP` / `A_amp` rescaling from the Table-I
  footnote. `A_AP = A_amp = 1` at the fiducial cosmology, so the Fisher comparison
  (evaluated at fiducial) sees only layer-1. The `our mean`/`our sigma` columns are the
  **layer-1-mapped values BEFORE layer-2**.

---

## Sources verified (primary, this session)

| Source | Route | What was read |
|---|---|---|
| **arXiv:2511.20757** (DESI-2; Chudaykin, Ivanov, Philcox) | PDF `pdftotext -layout` + prior transcriptions (A: rendered page 5) | Table I (14 rows/bin) + caption/footnote; §II AP-rescaling text |
| **arXiv:2507.13433** (Paper 1) | PDF `pdftotext -layout` | Eq (57) param set; Eq (58) P-stoch (kNL=0.45); Eq (59) B-stoch; §V.B–C; bibliography [160]/[165] |
| **arXiv:2004.10607** (CLASS-PT) | PDF `pdftotext -layout` (Eqs 2.15/2.19–2.23) | μ-space counterterm (2.15); multipole model (2.21a–c); per-multipole basis (2.22); basis map (2.23) |
| **arXiv:2110.10161** (ref [160]; Ivanov, Philcox, Nishimichi, Simonović, Takada, Zaldarriaga, PRD 105, 063512) | PDF `pdftotext -layout` | FoG counterterm Eq (3.10)/(3.11) `Z1^FoG`; bispectrum Eq (3.14); c1 prior Eq (5.15) |
| `ps_1loop_jax` code | direct read | `ps_1loop.py` L581–672/236/306; `bs_tree.py` L168–170/246–252 |
| production config | direct read | `build_taylor_templates_lcdm.py` L163–266 (fiducial cosmology, `knl_bins`, `K_NL_RSD=0.45`) |

DESI-2 prints **no operator equations** for the stochastic/bispectrum terms — it defers
to Paper 1 (Eqs 58/59) and, for the bispectrum FoG operator, to ref [160]. CLASS-PT is
the operator source for the power-spectrum counterterms `c0/c2/c4/c̃`.

---

## 1. Table I verbatim (arXiv:2511.20757) — audit anchor

Columns as printed: **Type | Parameter | Default | Prior | Units**. Default is blank for
every nuisance row. `𝒰[a,b]` uniform; `𝒩(μ,σ²)` Gaussian mean μ, variance σ².

| Type | Parameter (as printed) | Prior | Units | Marg/Sampled |
|---|---|---|---|---|
| nuisance (sampled) | `b₁σ₈(z)` | `𝒰[0, 3]` | — | sampled |
| nuisance (sampled) | `b₂σ₈²(z)` | `𝒩[0, 5²]` | — | sampled |
| nuisance (sampled) | `b_𝒢₂σ₈²(z)` | `𝒩[0, 5²]` | — | sampled |
| nuisance (analytically marginalized) | `b_Γ₃ A_AP A²_amp` | `𝒩((23/42)(b₁−1), 1²)` | — | marginalized |
| nuisance (analytically marginalized) | `c₀ A_AP A_amp` | `𝒩(0, 30²)` | `[Mpc/h]²` | marginalized |
| nuisance (analytically marginalized) | `c₂ A_AP A_amp` | `𝒩(30, 30²)` | `[Mpc/h]²` | marginalized |
| nuisance (analytically marginalized) | `c₄ A_AP A_amp` | `𝒩(0, 30²)` | `[Mpc/h]²` | marginalized |
| nuisance (analytically marginalized) | `c̃ A_AP A_amp` | `𝒩(400, 400²)` | `[Mpc/h]⁴` | marginalized |
| nuisance (analytically marginalized) | `c₁ A_AP A_amp` | `𝒩(0, 5²)` | `[Mpc/h]²` | marginalized |
| nuisance (analytically marginalized) | `P_shot` | `𝒩(0, 1²)` | — | marginalized |
| nuisance (analytically marginalized) | `a₀` | `𝒩(0, 1²)` | — | marginalized |
| nuisance (analytically marginalized) | `a₂` | `𝒩(0, 1²)` | — | marginalized |
| nuisance (analytically marginalized) | `B_shot A_AP A_amp` | `𝒩(0, 1²)` | — | marginalized |
| nuisance (analytically marginalized) | `A_shot` | `𝒩(0, 1²)` | — | marginalized |

`14` params/bin = `11` marginalized + `3` sampled (§II.3: "fourteen nuisance parameters
for each redshift chunk, three of which enter only in the bispectrum model … all except
three can be marginalized").

**Caption / footnote (verbatim):**
> TABLE I. **Model Parameters**: … 𝒰[a, b] refers to a uniform prior between a and b,
> whilst 𝒩(µ, σ²) denotes a Gaussian distribution with mean µ and variance σ². Bias
> parameters b₁σ₈, b₂σ₈², b_𝒢₂σ₈² are directly sampled in the MCMC chains, whilst the
> nuisance parameters that appear quadratically in the likelihood are marginalized over
> analytically… We use the Alcock-Paczynski parameter,
> **A_AP ≡ (H₀ᶠⁱᵈ/H₀)³ · [H(z)/Hᶠⁱᵈ(z)] · [D_Aᶠⁱᵈ(z)/D_A(z)]²** … **The A_AP factor is
> absorbed into the definition of the stochastic parameters.** Similarly,
> **A_amp ≡ σ₈²(z)/σ₈,ref²(z)**, where σ₈,ref(z) is the late-time fluctuation amplitude
> at the Planck 2018 best-fit cosmology.

**Paper 1 footnote 8 (verbatim, corroborating layer-2):**
> We do not rescale b₁, b₂ and b_𝒢₂ with the AP amplitude because these bias parameters
> enter the theoretical model in different combinations with A_AP. All other nuisance
> parameters appear in unique combinations with the AP amplitude, motivating our
> AP-rescaled priors.

---

## 2. Reconciliation log (Step 1: row-by-row diff)

`ours = paper × factor + offset` unless noted. Provenance line per row: `A: … | B: … |
resolution: …`.

### AGREE (identical in A and B, confirmed against primary source)

| param | factor | offset | layer-2 | provenance |
|---|---|---|---|---|
| **c4** | 1 | 0 | A_AP·A_amp | A: 1/0 \| B: 1/0 \| resolution: AGREE — `c4=c̃4` exact (CLASS-PT 2.23 `c₄≡c̃₄`). |
| **cfog (c̃)** | 1 | 0 | A_AP·A_amp | A: 1/0 \| B: 1/0 \| resolution: AGREE — CLASS-PT 2.16 char-for-char = `ps_1loop.py:612–617`, incl. explicit `(b1+fμ²)²`. |
| **a0** | (knl_b/0.45)² | 0 | none | A: (knl_b/0.45)²/0 \| B: (knl_b/0.45)²/0 \| resolution: AGREE — Paper 1 Eq 58 kNL=0.45 vs per-bin `knl_bins`. |
| **a2** | (knl_b/0.45)² | 0 | none | A: same as a0 \| B: same as a0 \| resolution: AGREE. |
| **P_shot** | 1 | +1 | none | A: 1/+1 \| B: 1/+1 \| resolution: AGREE — Eq 58 explicit `1+`, mean-1 amp ↔ mean-0 deviation; shared P/B param. |
| **B_shot** | 1 | +1 | A_AP·A_amp | A: 1/+1 \| B: 1/+1 \| resolution: AGREE — Eq 59 ≡ `bs_tree.py:246–250`; linearity + Poisson subtraction. |
| **A_shot** | 1 | +1 | none | A: 1/+1 \| B: 1/+1 \| resolution: AGREE — Eq 59 `A_shot/n̄²`. |
| **bGamma3** | 1 | 0 | A_AP·A²_amp | A: 1/0 \| B: 1/0 \| resolution: AGREE — mean `(23/42)(b1−1)` raw b1, width 1. |
| **b2** | 1/σ8²(z) | 0 | none (σ8-scale = layer-1) | A: 1/σ8²(z), 0 \| B: 1/σ8²(z), 0 \| resolution: AGREE — paper on `b2σ8²`, raw width `5/σ8²(z)`. |
| **bG2** | 1/σ8²(z) | 0 | none | A: 1/σ8²(z), 0 \| B: 1/σ8²(z), 0 \| resolution: AGREE — same as b2. |

### RECONCILABLE (same physics; one presented a caveat the other did not)

*(none besides the DISAGREE rows below — A and B present identical numbers for every
AGREE row; the only substantive divergence is battleground 1.)*

### DISAGREE (resolved from primary source — battlegrounds 1 & 2)

| param | A | B | resolution |
|---|---|---|---|
| **c0** | factor 1 **diagonal only**; NON-diagonal `our_c0 = paper_c0 − (f/3)paper_c2 + (3f²/35)paper_c4`; true mean ≈ −(f/3)·30 ≈ −8, width ≈ 31.1, correlated | factor 1, offset 0, **no basis caveat** | **A CORRECT.** CLASS-PT priors are on the per-multipole basis; our code is the tilde basis. See §3.1. |
| **c2** | factor 1 diagonal only; `our_c2 = paper_c2 − (6f/7)paper_c4`; true width ≈ 36.6, corr with c4 ≈ −0.57 | factor 1, offset 0, no caveat | **A CORRECT.** See §3.1. |
| **c1** | 0.45²=0.2025 (bare-k² assumed, **ref [160] not read**) | 0.45²=0.2025 (additive assumed, **ref [160] not read**) | **BOTH CORRECT; now VERIFIED.** Ref [160] Eq (3.11) prints the additive bare-Z1 form. See §3.2. |

---

## 3. Battleground adjudications (Step 2: primary-source re-derivation)

### 3.1 Battleground 1 — c0/c2/c4 basis (A CORRECT; headline escalation)

**The question.** Our code applies the k² counterterm as a single μ-space polynomial
(`ps_1loop.py:589–595`):
```
P_ctr,k2(k,μ) = −2 k² P_lin · [c0 + c2 f μ² + c4 f² μ⁴]
```
This is **character-for-character CLASS-PT Eq (2.15)** (the μ-space "tilde" basis):
> "P_ctr,∇²δ = −2 c̃₀ k² P_lin − 2 c̃₂ f μ² k² P_lin − 2 c̃₄ f² μ⁴ k² P_lin"   (CLASS-PT 2.15)

so **our `{c0,c2,c4}` ≡ CLASS-PT `{c̃0,c̃2,c̃4}` (tilde / μ-space).** A and B agree here.

**The deciding evidence.** CLASS-PT writes the *galaxy multipoles it actually computes*
(Eqs 2.21a–c) with a **different, per-multipole** set of coefficients — one per
multipole:
> "P₀(z,k) = … + c₀(z) P₀,∇²δ(z,k) + c̃(z) P₀,∇⁴z δ(z,k) + P_shot(z),   (2.21a)
>  P₂(z,k) = … + c₂(z) P₂,∇²δ(z,k) + c̃(z) P₂,∇⁴z δ(z,k),               (2.21b)
>  P₄(z,k) = … + c₄(z) P₄,∇²δ(z,k) + c̃(z) P₄,∇⁴z δ(z,k),               (2.21c)"

and states explicitly, immediately after:
> "Note that the **basis of counterterms has been changed to have a single free
> coefficient for each multipole moment**, and the new contributions are defined as
> P_ℓ,∇²δ(z,k) ≡ (2ℓ+1)/2 ∫dµ L_ℓ(µ) µ^ℓ f^{ℓ/2} k² P_lin(k).   (2.22)
> The mapping between the old and new coefficients is given by
>   c₀ ≡ c̃₀ + (f/3) c̃₂ + (f²/5) c̃₄,   c₂ ≡ c̃₂ + (6f/7) c̃₄,   c₄ ≡ c̃₄.   (2.23)"

So the coefficients CLASS-PT **exposes and imposes priors on** — the coefficients DESI-2
Table I labels `c₀, c₂, c₄` — are the **per-multipole** ones (Eq 2.21/2.23), *not* the
tilde ones our code uses. DESI-2's pipeline is CLASS-PT (ref [126]); Paper 1 §V confirms
"the parameters cℓ in (57) correspond to the **leading-order counterterms for multipoles
ℓ = 0, 2, 4**." The phrase "has been changed" is decisive: the μ-space tilde basis (2.15)
is the *old* basis; the analysis/priors use the *new* per-multipole basis.

**Verified algebra (Legendre-overlap check).** Eq (2.23)'s coefficients are exactly the
monopole/quadrupole projection weights of `μ²` and `μ⁴`: `μ² = ⅓L₀ + ⅔L₂`,
`μ⁴ = ⅕L₀ + (4/7)L₂ + (8/35)L₄`. The `c̃₄`-into-`c₂` weight `(4f²/7)/(2f/3) = 6f/7` and the
`c̃₂`,`c̃₄`-into-`c₀` weights `f/3`, `f²/5` reproduce (2.23) with a **unit diagonal** — a
genuine relabelling of the same 3-D space, so the diagonal factor is **+1** (both bases
share the `−2 k²P_lin` normalization; A's "+1 not −2/−½" is confirmed). The paper's
footnote to (2.23) ("This mapping … is not exact when IR resummation and the AP effect
are present, but … smaller than our baseline accuracy of 0.1%") confirms it is the map in
force.

**Inverse map (paper per-multipole → our tilde), `f = f(z)` at the fiducial cosmology:**
```
our_c4 = paper_c4
our_c2 = paper_c2 − (6f/7)·paper_c4
our_c0 = paper_c0 − (f/3)·paper_c2 + (3f²/35)·paper_c4        [3f²/35 = (f/3)(6f/7) − f²/5]
```
i.e. `c̃ = L(f)·c` with the upper-triangular `L(f) = [[1, −f/3, 3f²/35], [0, 1, −6f/7],
[0, 0, 1]]`.

**Consequence — the paper's DIAGONAL prior becomes a CORRELATED, f-SHIFTED prior on our
coefficients.** With paper prior `c0~𝒩(0,30²)`, `c2~𝒩(30,30²)`, `c4~𝒩(0,30²)`
(independent), the induced prior on `(our_c0, our_c2, our_c4) = L(f)·(c0,c2,c4)` is
Gaussian with mean `L(f)·(0,30,0) = (−(f/3)·30, 30, 0)` and covariance `30²·L(f)Lᵀ(f)`.

**Per-bin numbers at the production fiducial** (`ombh2=0.02242, omch2=0.11933, h=0.6766,
mnu=0.06`; `f(z)` from `ps_1loop_jax.background.growth_rate`, source = the growth ODE at
that fiducial):

| bin | z | f(z) | our_c0 mean | our_c0 width | our_c2 mean | our_c2 width | our_c4 width | corr(c2,c4) | corr(c0,c2) | corr(c0,c4) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.7 | 0.8155 | −8.15 | 31.14 | 30 | 36.60 | 30 | −0.573 | −0.246 | +0.055 |
| 1 | 0.9 | 0.8579 | −8.58 | 31.26 | 30 | 37.24 | 30 | −0.592 | −0.257 | +0.061 |
| 2 | 1.1 | 0.8893 | −8.89 | 31.36 | 30 | 37.72 | 30 | −0.606 | −0.265 | +0.065 |
| 3 | 1.3 | 0.9126 | −9.13 | 31.43 | 30 | 38.09 | 30 | −0.616 | −0.271 | +0.068 |
| 4 | 1.5 | 0.9301 | −9.30 | 31.49 | 30 | 38.37 | 30 | −0.623 | −0.275 | +0.071 |
| 5 | 1.8 | 0.9489 | −9.49 | 31.55 | 30 | 38.67 | 30 | −0.631 | −0.280 | +0.073 |
| 6 | 2.2 | 0.9649 | −9.65 | 31.60 | 30 | 38.93 | 30 | −0.637 | −0.284 | +0.076 |

(`our_c0 mean = −(f/3)·30 = −10f`; `our_c2 mean = 30` unchanged; `our_c4` untouched.)

**Why this cannot go into the scalar-factor schema.** The truth is a per-bin,
f-dependent, off-diagonal `3×3` prior covariance block coupling `c0/c2/c4` *within* each
bin, plus an f-dependent mean shift on `c0`. The Task-4 schema has (i) one row per θ_lin
key applied uniformly to all bins, (ii) no `f`-token in `factor_formula`/`mean_formula`,
(iii) diagonal per-parameter widths only. Per the resolution rule, **we do not force it.**

**Design alternatives (for the controller to rule on):**
- **(A) O(f) approximation (interim, no code change):** place the paper's diagonal prior
  directly on our tilde `c0/c2/c4` (factor 1, offset 0). Bias = the shift/inflation above
  (largest effect: `c0` mean off by ≈ −8 to −10 [Mpc/h]²; `c2` width under-stated ≈ 22–30%).
  Tolerable *only* because these are wide, analytically-marginalized nuisances. **This is
  what the PROVISIONAL spec rows below encode**, and what B implicitly assumed.
- **(B) Full triangular rotation (recommended for correctness):** add a per-bin
  counterterm-rotation to `make_desi_prior_fns` that builds `μ_p` and the **off-diagonal**
  `Σ_p` block from `f(z)` via `L(f)`. The marginal-likelihood API already accepts a full
  `Σ_p(θ_NL)` (CONTEXT.md "API requirement (firm)"), so no likelihood change is needed —
  only a new spec token (e.g. `mean_formula:/factor_formula: ctr_multipole_rotation`) and
  the per-bin block assembly. Matches A's recommendation.
- **(C) Rotate the templates:** apply `L(f)⁻¹` to the `c0/c2/c4` template columns so the
  model is expressed in the paper's per-multipole basis, then impose the diagonal prior
  directly. Cleanest conceptually; requires a template rebuild.

### 3.2 Battleground 2 — c1 bispectrum FoG operator (RESOLVED, factor 0.2025)

**Ref [160] identified** (Paper 1 bibliography, verbatim):
> "[160] M. M. Ivanov, O. H. E. Philcox, T. Nishimichi, M. Simonović, M. Takada, and
> M. Zaldarriaga, Phys. Rev. D 105, 063512" — = **arXiv:2110.10161**, cited by Paper 1 for
> "a k² fingers-of-God-term, parameterized by the nuisance parameter c1 [160]" (§V.C).

**Operator form, verbatim from ref [160]:**
> "The inclusion of the c1 counterterm amounts to correcting the kernel Z1 as
>  Z1 → Z1^FoG = b1 + f µ² − c1 µ² (k/k_NL^r)².   (3.11)
>  In what follows we set k_NL^r = 0.3 h Mpc⁻¹ …"

and the bispectrum uses `Z1^FoG` on each leg:
> "Bggg = 2 Z2(k1,k2) Z1^FoG(k1) Z1^FoG(k2) P_tree …"   (ref [160] §III, Eq 3.14 region)

This is **additive** (`Z1 − c1 μ² (k/k_NL)²`), with **no extra `(b1+fμ²)` or `f` factor** —
character-identical to `ps_1loop_jax` `bs_tree.py:169`
(`Z1_fog = (b1 + f μ²) − c1·μ²·(k/k_nl_rsd)²`) and to the D'Amico form (2502.14758 Eq 3.14,
per CONTEXT.md). The **multiplicative** hypothesis (`Z1(1 − c1 k²μ²)`, which would inject
an extra `b1 + fμ²`) is **excluded**. This closes the sole MED-confidence concern shared
by A and B.

**Factor derivation.** Ref [160] uses dimensionless `c1` with `(k/0.3)²`. The DESI papers
(Paper 1 Table III; DESI-2 Table I) re-cast `c1` as a **dimensionful `[Mpc/h]²`**
coefficient multiplying **bare `k²μ²`** (the only reading consistent with the printed
`[Mpc/h]²` units — `c1·k²μ²` is dimensionless iff `c1` carries `[Mpc/h]²`), with an
independent EFT-naturalness prior `𝒩(0,5²)`. Our **production** config overrides
`k_nl_rsd = 0.45` (`build_taylor_templates_lcdm.py:179`, not the `bs_tree.py:31` default
0.3). Equate at identical `(k,μ)`:
```
c1_ours · (k/0.45)² = c1_paper · k²   ⟹   c1_ours = c1_paper · 0.45² = 0.2025 · c1_paper
```
**factor `0.45² = 0.2025`, offset 0**; `our σ = 5 · 0.2025 = 1.0125` (dimensionless).
A: 0.2025/0 | B: 0.2025/0 | resolution: **AGREE, now verified against ref [160] Eq 3.11.**

### 3.3 Battleground 3 — layer-2 stochastic flags (AGREE; resolved verbatim)

A and B assign **identical** layer-2 flags; both match the printed Table I. Row-by-row
from the PDF-verbatim "Parameter" column (the factor printed next to each prior):

| stochastic row | printed factor in Table I | runtime rescale flag | why |
|---|---|---|---|
| `P_shot` | (none printed) | **none** | Poisson-normalized by `1/n̄`; the footnote absorbs `A_AP` into the stochastic *definition*, so no explicit factor is printed. |
| `a₀` | (none printed) | **none** | same — `1/n̄` normalization carries the AP volume dependence. |
| `a₂` | (none printed) | **none** | same. |
| `A_shot` | (none printed) | **none** | `A_shot/n̄²`; AP absorbed via `1/n̄²`. |
| `B_shot` | `A_AP A_amp` | **A_AP·A_amp** | `B_shot` multiplies `b1² P^tree` (the *linear power*), so it takes an explicit `A_AP·A_amp` like the counterterms — NOT absorbed into `1/n̄`. |

**Resolution of the apparent footnote↔table tension.** The footnote "The A_AP factor is
absorbed into the definition of the stochastic parameters" is an *explanation* of why
`P_shot/a₀/a₂/A_shot` carry **no** explicit `A_AP` (their `1/n̄`, `1/n̄²` normalization
already carries the AP volume Jacobian), **not** a runtime instruction to multiply them by
`A_AP`. `B_shot` is printed *with* `A_AP·A_amp` precisely because it multiplies the linear
power spectrum rather than a pure `1/n̄`. Paper 1 footnote 8 corroborates: only the
parameters that "appear in unique combinations with the AP amplitude" are AP-rescaled.
**Runtime flags: `P_shot/a₀/a₂/A_shot → none`; `B_shot → A_AP·A_amp`.**

**Off-fiducial caveat (runtime note, not a fiducial issue).** The paper's
`P_shot/a₀/a₂/A_shot` definitions carry an *implicit* `A_AP` through `1/n̄`; our code uses
the plain **fiducial** `n̄` (a static constant), so off-fiducial it does not reproduce that
absorbed `A_AP`. At the fiducial (`A_AP=1`, where the Fisher comparison lives) the two are
identical, so the spec flag `none` is exact there. Flagged for a future runtime refinement.

---

## 4. Layer-1 map table (authoritative, fixed format)

`factor`/`offset`: `ours = paper × factor + offset`. `our mean`/`our sigma` are
layer-1-mapped, **pre-layer-2**. Config symbols: `knl_b` = per-bin `knl_bins`; `f = f(z)`;
`σ8(z)` = linear σ8 at effective z, fiducial cosmology. Per-row provenance lines follow §2.

| param | our operator (file:line) | paper operator (eq. ref) | our units | paper units | factor | offset | layer-2 rescale | paper mean | paper sigma | our mean | our sigma | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **c0** | `ps_1loop.py:589–595` | CLASS-PT 2.21a/2.22/2.23; Table I `c₀` | `[Mpc/h]²` | `[Mpc/h]²` | **matrix `L(f)`** (diag 1) | see §3.1 | A_AP·A_amp | `0` | `30` | **−(f/3)·30** (≈−8) | **30·√(1+(f/3)²+(3f²/35)²)** (≈31.1) | **NON-diagonal**, correlated w/ c2,c4. Provisional spec row = diagonal (0,30). §3.1. |
| **c2** | `ps_1loop.py:589–595` | CLASS-PT 2.21b/2.22/2.23; Table I `c₂` | `[Mpc/h]²` | `[Mpc/h]²` | **matrix `L(f)`** (diag 1) | see §3.1 | A_AP·A_amp | `30` | `30` | **30** | **30·√(1+(6f/7)²)** (≈36.6) | **NON-diagonal**, corr(c2,c4)≈−0.57. Provisional spec row = diagonal (30,30). §3.1. |
| **c4** | `ps_1loop.py:589–595` | CLASS-PT 2.21c/2.22/2.23; Table I `c₄` | `[Mpc/h]²` | `[Mpc/h]²` | `1` | `0` | A_AP·A_amp | `0` | `30` | `0` | `30` | **Exact** (`c4=c̃4`), no mixing. |
| **cfog (`c̃`)** | `ps_1loop.py:612–617` | CLASS-PT 2.16; Table I `c̃` | `[Mpc/h]⁴` | `[Mpc/h]⁴` | `1` | `0` | A_AP·A_amp | `400` | `400` | `400` | `400` | Operators identical incl. explicit `(b1+fμ²)²`. Single coefficient (no basis mixing). |
| **a0** | `ps_1loop.py:667–669` | Paper 1 Eq 58; Table I `a₀` | dimensionless | dimensionless | `(knl_b/0.45)²` | `0` | none | `0` | `1` | `0` | `(knl_b/0.45)²` | Per-bin σ: 1.335, 2.086, 3.320, 5.138, 8.218, 16.36, 40.96. |
| **a2** | `ps_1loop.py:667–669` | Paper 1 Eq 58; Table I `a₂` | dimensionless | dimensionless | `(knl_b/0.45)²` | `0` | none | `0` | `1` | `0` | `(knl_b/0.45)²` | Same per-bin σ as a0. |
| **P_shot** | `ps_1loop.py:667` + `bs_tree.py:238` | Paper 1 Eqs 58 & 59; Table I `P_shot` | dimensionless | dimensionless | `1` | `+1` | none | `0` | `1` | `1` | `1` | Mean-1 amplitude vs mean-0 deviation. Shared P/B param; offset +1 consistent both sides. |
| **b_Γ3** | `ps_1loop.py:236,306` | CLASS-PT 2.21 `(2bG2+0.8bΓ3)F_G2`; Table I `b_Γ₃` | dimensionless | dimensionless | `1` | `0` | A_AP·A²_amp | `(23/42)(b1−1)` | `1` | `(23/42)(b1−1)` | `1` | Mean uses raw b1; same `0.8=4/5`. |
| **c1** | `bs_tree.py:168–169` | ref [160] Eq 3.11 (`Z1−c1μ²(k/kNL)²`); Table I `c₁` | dimensionless | `[Mpc/h]²` | `0.45² = 0.2025` | `0` | A_AP·A_amp | `0` | `5` | `0` | `1.0125` | Additive bare-Z1 form **verified** (§3.2). `k_nl_rsd=0.45` (production). No extra f/b1. |
| **B_shot** | `bs_tree.py:246–250` | Paper 1 Eq 59; Table I `B_shot` | dimensionless | dimensionless | `1` | `+1` | A_AP·A_amp | `0` | `1` | `1` | `1` | Structure identical to Eq 59. |
| **A_shot** | `bs_tree.py:252` | Paper 1 Eq 59; Table I `A_shot` | dimensionless | dimensionless | `1` | `+1` | none | `0` | `1` | `1` | `1` | `A_shot/n̄²` term. |
| **b2** (sampled) | `ps_1loop.py:230,232,234` | Table I `b₂σ₈²` | dimensionless (raw) | dimensionless (`b2σ8²`) | `1/σ8²(z)` | `0` | none | `0` | `5` | `0` | `5/σ8²(z)` | Paper samples `b2σ8²`; we sample raw b2. |
| **bG2** (sampled) | `ps_1loop.py:231,233` | Table I `b_𝒢₂σ₈²` | dimensionless (raw) | dimensionless (`bG2σ8²`) | `1/σ8²(z)` | `0` | none | `0` | `5` | `0` | `5/σ8²(z)` | Same as b2. |

`b1` (sampled) is not a map row: paper `b1σ8 ~ 𝒰[0,3]`; we sample raw b1 flat/unbounded
(CONTEXT deviation 3). Recover paper variable via `b1σ8 = b1·σ8(z)` for reporting only.

---

## 5. Cross-check against CONTEXT.md (Step 4)

All recorded CONTEXT.md values are **confirmed** against the primary PDFs — no numeric
contradiction:
- ✅ `c1` paper prior `𝒩(0,5²) [Mpc/h]²` on `c1·A_AP·A_amp` — Table I row present, exact.
- ✅ `c2` paper mean `30` — Table I `𝒩(30,30²)`.
- ✅ `c̃` `𝒩(400,400²) [Mpc/h]⁴` — exact.
- ✅ `b2/bG2` `𝒩(0,5²)` σ8-scaled → raw width `5/σ8²(z)`.
- ✅ `bΓ3` mean `(23/42)(b1−1)` raw b1, width 1.
- ✅ `P_shot` mean-0 ↔ mean-1 shift (offset +1); extended here to `B_shot`, `A_shot`.
- ✅ Table I has `B_shot`/`A_shot` rows; `14 = 11 marginalized + 3 sampled`.

**Two refinements written back to CONTEXT.md** (primary source governs; no numeric value
changed):
1. **c1 open item RESOLVED.** CONTEXT's c1 section said "Exact factor (possible extra f/b1
   factors) still to be derived from CLASS-PT's operator before use." Ref [160]
   (arXiv:2110.10161) Eq (3.11) now settles it: additive `Z1 − c1μ²(k/kNL)²`, **no extra
   f/b1**, factor `0.45² = 0.2025`. Marked resolved.
2. **NEW structural deviation (counterterm basis).** CONTEXT's "Two-layer convention map"
   framed layer-1 as "a per-parameter factor." That holds for every parameter **except
   `c0/c2/c4`**, whose true map is the triangular f-dependent basis rotation `L(f)` above.
   Recorded as a new Stream-B deviation with the escalation.

---

## Spec rows (machine-readable)

<!--
  Task-4 schema. rescale ∈ {none, A_AP, A_AP*A_amp, A_AP*A_amp^2};
  factor_formula ∈ {null, knl_over_0p45_sq}; mean_formula ∈ {null, coevolution_bGamma3};
  sampled rescale ∈ {none, sigma8_sq}; sampled kind ∈ {flat, gaussian}.
  Loader reconciliation: sigma == paper_sigma*|factor|; mean == paper_mean*factor+offset
  (unless mean_formula is set). Values are LAYER-1 (pre-layer-2); layer-2 applied at
  runtime by make_desi_prior_fns via `rescale`. Per-bin a0/a2 factor via factor_formula.

  *** PROVISIONAL rows: pk.ctr.c0, pk.ctr.c2, pk.ctr.c4 ***
  Battleground 1 (see §3.1): the paper's c0/c2/c4 priors live in CLASS-PT's PER-MULTIPOLE
  basis (Eqs 2.21-2.23); our code coefficients are the MU-SPACE tilde basis (Eq 2.15).
  The true prior on our (c0,c2,c4) is CORRELATED and f-SHIFTED (per-bin), via the
  triangular rotation c_tilde = L(f)*c_paper:
      our_c0 = paper_c0 - (f/3)paper_c2 + (3f^2/35)paper_c4
      our_c2 = paper_c2 - (6f/7)paper_c4
      our_c4 = paper_c4
  Marginal-at-fiducial-f (dropping correlations), per §3.1 table:
      our_c0: mean -(f/3)*30 (approx -8..-10), width ~31.1-31.6
      our_c2: mean 30,                          width ~36.6-38.9  (corr(c2,c4) ~ -0.57..-0.64)
      our_c4: mean 0,                           width 30 (exact)
  This per-bin f-dependent correlated 3x3 block CANNOT be represented in the scalar-factor
  schema (no f-token; one uniform row per key; diagonal widths only). Per the resolution
  rule we DO NOT force it. The rows below encode design-alternative (A): the O(f)
  approximation (paper diagonal placed directly on our tilde coefficients, factor 1,
  offset 0). This is chosen over baking a single fiducial-f marginal because the row is
  applied identically to all 7 bins and no single f is correct for all. CONTROLLER
  DECISION REQUIRED: adopt (A) as-is, or implement (B) full triangular rotation (add a
  ctr-rotation token to make_desi_prior_fns; the marginal-likelihood API already accepts
  a full Sigma_p(theta_NL)), or (C) rotate the templates. See §3.1.
-->

```yaml
metadata:
  source: "arXiv:2511.20757"
  table: "I"
  convention_map: "docs/design/desi-convention-map.md"
  paper_knl: 0.45
  production_k_nl_rsd: 0.45
  deviations:
    - "c0/c2/c4 are PROVISIONAL O(f) diagonal rows: paper priors are in the CLASS-PT
       per-multipole basis (Eq 2.21-2.23), our coefficients are the mu-space tilde basis
       (Eq 2.15); the true prior is a per-bin f-dependent correlated rotation L(f) not
       representable in this schema. See desi-convention-map.md section 3.1."
    - "P_shot/B_shot/A_shot use offset +1 (our mean-1 Poisson amplitude vs paper mean-0
       deviation)."
    - "a0/a2 factor is per-bin (knl_b/0.45)^2 via factor_formula knl_over_0p45_sq."
    - "c1 factor 0.45^2=0.2025 (production k_nl_rsd=0.45); additive bare-Z1 operator
       verified from ref [160] arXiv:2110.10161 Eq (3.11)."
    - "b2/bG2 sampled raw; paper priors on b*sigma8^2, raw width 5/sigma8(z)^2."
marginalized:
  shared.bias.bGamma3:
    {paper_mean: null, paper_sigma: 1.0, paper_units: "unit",
     paper_variable: "bGamma3*A_AP*A_amp^2", factor: 1.0, offset: 0.0,
     mean: null, sigma: 1.0, rescale: "A_AP*A_amp^2",
     factor_formula: null, mean_formula: "coevolution_bGamma3"}
  shared.stoch.P_shot:
    {paper_mean: 0.0, paper_sigma: 1.0, paper_units: "unit",
     paper_variable: "P_shot", factor: 1.0, offset: 1.0,
     mean: 1.0, sigma: 1.0, rescale: "none",
     factor_formula: null, mean_formula: null}
  # PROVISIONAL (battleground 1, section 3.1): true prior correlated + f-shifted.
  # This row = O(f) approximation: paper diagonal on our tilde c0 (factor 1, offset 0).
  pk.ctr.c0:
    {paper_mean: 0.0, paper_sigma: 30.0, paper_units: "(Mpc/h)^2",
     paper_variable: "c0*A_AP*A_amp", factor: 1.0, offset: 0.0,
     mean: 0.0, sigma: 30.0, rescale: "A_AP*A_amp",
     factor_formula: null, mean_formula: null}
  # PROVISIONAL (section 3.1): true our_c2 width ~36.6-38.9, corr(c2,c4) ~ -0.57..-0.64.
  pk.ctr.c2:
    {paper_mean: 30.0, paper_sigma: 30.0, paper_units: "(Mpc/h)^2",
     paper_variable: "c2*A_AP*A_amp", factor: 1.0, offset: 0.0,
     mean: 30.0, sigma: 30.0, rescale: "A_AP*A_amp",
     factor_formula: null, mean_formula: null}
  # Marginal EXACT (c4 = c_tilde4, no basis mixing of c4 itself); its
  # correlation with our c0/c2 (induced by their c4-mixing) is dropped in
  # this diagonal encoding, same as the other two PROVISIONAL rows.
  pk.ctr.c4:
    {paper_mean: 0.0, paper_sigma: 30.0, paper_units: "(Mpc/h)^2",
     paper_variable: "c4*A_AP*A_amp", factor: 1.0, offset: 0.0,
     mean: 0.0, sigma: 30.0, rescale: "A_AP*A_amp",
     factor_formula: null, mean_formula: null}
  pk.ctr.cfog:
    {paper_mean: 400.0, paper_sigma: 400.0, paper_units: "(Mpc/h)^4",
     paper_variable: "ctilde*A_AP*A_amp", factor: 1.0, offset: 0.0,
     mean: 400.0, sigma: 400.0, rescale: "A_AP*A_amp",
     factor_formula: null, mean_formula: null}
  pk.stoch.a0:
    {paper_mean: 0.0, paper_sigma: 1.0, paper_units: "unit",
     paper_variable: "a0", factor: 1.0, offset: 0.0,
     mean: 0.0, sigma: 1.0, rescale: "none",
     factor_formula: "knl_over_0p45_sq", mean_formula: null}
  pk.stoch.a2:
    {paper_mean: 0.0, paper_sigma: 1.0, paper_units: "unit",
     paper_variable: "a2", factor: 1.0, offset: 0.0,
     mean: 0.0, sigma: 1.0, rescale: "none",
     factor_formula: "knl_over_0p45_sq", mean_formula: null}
  bk.ctr.c1:
    {paper_mean: 0.0, paper_sigma: 5.0, paper_units: "(Mpc/h)^2",
     paper_variable: "c1*A_AP*A_amp", factor: 0.2025, offset: 0.0,
     mean: 0.0, sigma: 1.0125, rescale: "A_AP*A_amp",
     factor_formula: null, mean_formula: null}
  bk.stoch.B_shot:
    {paper_mean: 0.0, paper_sigma: 1.0, paper_units: "unit",
     paper_variable: "B_shot*A_AP*A_amp", factor: 1.0, offset: 1.0,
     mean: 1.0, sigma: 1.0, rescale: "A_AP*A_amp",
     factor_formula: null, mean_formula: null}
  bk.stoch.A_shot:
    {paper_mean: 0.0, paper_sigma: 1.0, paper_units: "unit",
     paper_variable: "A_shot", factor: 1.0, offset: 1.0,
     mean: 1.0, sigma: 1.0, rescale: "none",
     factor_formula: null, mean_formula: null}
sampled:
  b1: {kind: flat}
  b2:
    {kind: gaussian, paper_mean: 0.0, paper_sigma: 5.0,
     paper_variable: "b2*sigma8(z)^2", rescale: "sigma8_sq"}
  bG2:
    {kind: gaussian, paper_mean: 0.0, paper_sigma: 5.0,
     paper_variable: "bG2*sigma8(z)^2", rescale: "sigma8_sq"}
```
