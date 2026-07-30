# DESI convention map — derivation B (independent)

**Purpose.** Independent second derivation (Stream-B, Task 2) of the layer-1
prior-convention map between the `ps_1loop_jax` EFT/stochastic coefficient
conventions and those of **arXiv:2511.20757** ("Reanalyzing DESI DR1: 2",
Chudaykin, Ivanov & Philcox) / CLASS-PT. Derived *without* reference to
derivation A. To be diffed against A by the controller.

For every parameter the rule is: **equate our operator's coefficient with the
paper's operator's coefficient at identical (k, μ, z) and solve `ours = paper ×
factor + offset`.** Layer-2 (the θ_NL-dependent `A_AP`/`A_amp` runtime rescale)
is recorded per row but *not* folded into `our mean`/`our sigma`.

---

## Sources used (primary)

| Source | Route | What was taken |
|--------|-------|----------------|
| arXiv:2511.20757 (Paper 2, reference chain) | PDF → `pdftotext -layout` | **Table I verbatim** (audit anchor), its footnote (A_AP, A_amp defs), §II text. Defers operator eqs to Paper 1 + CLASS-PT `[126]`. |
| arXiv:2004.10607 (CLASS-PT) | ar5iv HTML | P-spectrum counterterm eqs **(2.15)**, **(2.16)**; stochastic **(2.18)**. |
| arXiv:2507.13433 (Paper 1, companion) | PDF → `pdftotext -layout` | Bispectrum model §V.B–C: parameter set **(57)**, stochastic P **(58)** with `kNL`, bispectrum stochastic **(59)** (B_shot/A_shot), Table III (priors, ΛCDM). |
| `ps_1loop_jax` code | Read | operator definitions (lines cited below). |
| `build_taylor_templates_lcdm.py` | Read | production config: `knl_bins` (:173), `K_NL_RSD=0.45` (:179), fiducials (:258–266). |

**Code operator locations read** (`/Users/nguyenmn/ps_1loop_jax-for-pfs/src/ps_1loop_jax/`):
- `ps_1loop.py:581–600` — k² counterterm `get_pkmu_ctr_k2`
- `ps_1loop.py:603–622` — k⁴ FoG `get_pkmu_ctr_k4`
- `ps_1loop.py:656–672` — P stochastic `get_pkmu_stoch`
- `ps_1loop.py:236, 306` — bGamma3 kernel entry `(4/5)·b1·bGamma3·F_G2`, `(2/5)·bGamma3·F_G2`
- `bs_tree.py:168–169` — c1 FoG `Z1_fog = Z1(bias,f,μ) − c1·μ²·(k/k_nl_rsd)²`
- `bs_tree.py:218–252` — B stochastic `_get_bkmuphi_stoch_from_geometry` (B_shot/A_shot)

---

## Table I verbatim (arXiv:2511.20757, from PDF)

> **TABLE I.** Model Parameters: Parameters and priors used in the galaxy clustering analyses. Here, U[a, b] refers to a uniform prior between a and b, whilst N(μ, σ²) denotes a Gaussian distribution with mean μ and variance σ². Bias parameters b₁σ₈, b₂σ₈², b_G2σ₈² are directly sampled in the MCMC chains, whilst the nuisance parameters that appear quadratically in the likelihood are marginalized over analytically, as discussed in the text.

| Type | Parameter | Default | Prior | Units |
|------|-----------|:-------:|-------|-------|
| nuisance (sampled) | b₁ σ₈(z) | — | U[0, 3] | — |
| (sampled) | b₂ σ₈²(z) | — | N[0, 5²] | — |
| (sampled) | b_G2 σ₈²(z) | — | N[0, 5²] | — |
| nuisance (analytically marginalized) | b_Γ3 A_AP A²_amp | — | N(23/42·(b₁−1), 1²) | — |
| (analytically marginalized) | c₀ A_AP A_amp | — | N(0, 30²) | [Mpc/h]² |
| (analytically marginalized) | c₂ A_AP A_amp | — | N(30, 30²) | [Mpc/h]² |
| (analytically marginalized) | c₄ A_AP A_amp | — | N(0, 30²) | [Mpc/h]² |
| (analytically marginalized) | c̃ A_AP A_amp | — | N(400, 400²) | [Mpc/h]⁴ |
| (analytically marginalized) | c₁ A_AP A_amp | — | N(0, 5²) | [Mpc/h]² |
| (analytically marginalized) | P_shot | — | N(0, 1²) | — |
| (analytically marginalized) | a₀ | — | N(0, 1²) | — |
| (analytically marginalized) | a₂ | — | N(0, 1²) | — |
| (analytically marginalized) | B_shot A_AP A_amp | — | N(0, 1²) | — |
| (analytically marginalized) | A_shot | — | N(0, 1²) | — |

**Footnote (verbatim):**
> We use the Alcock-Paczynski parameter, `A_AP ≡ (H₀^fid/H₀)³ · [H(z)/H^fid(z)] · [D_A^fid(z)/D_A(z)]²`, where the super-script "fid" refers to quantities evaluated in the fiducial cosmology assumed when converting redshifts to distances in the DESI data. **The A_AP factor is absorbed into the definition of the stochastic parameters.** Similarly, we define `A_amp ≡ σ₈²(z)/σ₈,ref²(z)`, where σ₈,ref²(z) is the late-time fluctuation amplitude at the Planck 2018 best-fit cosmology.
>
> (§II text, footnote 8:) We do not rescale b₁, b₂ and b_G2 with the AP amplitude because these bias parameters enter the theoretical model in different combinations with A_AP. All other nuisance parameters appear in unique combinations with the AP amplitude, motivating our AP-rescaled priors.

**Count:** 14 nuisance per redshift chunk; "three of which enter only in the
bispectrum model" (= c₁, B_shot, A_shot); "Via linearity, all except three can
be marginalized" → 3 sampled (b₁σ₈, b₂σ₈², b_G2σ₈²) × 6 chunks = 18 sampled
nuisance ⇒ **11 marginalized + 3 sampled per bin.**

**Paper 1 (2507.13433) Table III** agrees on all priors/means but is ΛCDM-only
and rescales with `A ≡ σ₈²(z)/σ₈,ref²(z)` (the amplitude factor **only**, no
A_AP). Its two-column `pdftotext` rendering scrambled the amplitude *powers*
(appeared as `A² c̃`, `A b_Γ3`); Paper 2's Table I is verbatim-clean and
physically self-consistent (see open issue #4), so **Table I above is the
authority for the layer-2 flags.**

---

## Operator equations (verbatim quotes)

**CLASS-PT (2004.10607), power-spectrum counterterms:**
- Eq. (2.15): `P_ctr,∇²δ(z,k,μ) = −2 c̃₀(z) k² P_lin − 2 c̃₂(z) f(z) μ² k² P_lin − 2 c̃₄(z) f²(z) μ⁴ k² P_lin`
- Eq. (2.16): `P_ctr,∇⁴δ(z,k,μ) = − c̃(z) f⁴(z) μ⁴ k⁴ (b₁(z) + f(z) μ²)² P_lin(z,k)`
- Eq. (2.18): `P_εε,RSD = P_shot + a₀ k² + a₂ μ² k²` (bare k²; a₀,a₂ neglected in CLASS-PT itself — normalization supplied by Paper 1 eq 58).

**Paper 1 (2507.13433):**
- Eq. (57) parameter set: `{b1,b2,bG2,bΓ3} × {c0,c2,c4,c̃,c1} × {Pshot,a0,a2,Bshot,Ashot}`.
- Eq. (58): `P_stoch(k,μ) = (1/n̄)[ 1 + P_shot + a0 (k/kNL)² + a2 μ² (k/kNL)² ]`, with **kNL = 0.45 h/Mpc**; "the Poisson limit is reproduced with P_shot → 0".
- Eq. (59): `B_stoch(k1,k2,k3) = (1/n̄) b1² P^tree(k1) [ B_shot + β μ1² (P_shot + B_shot) + β² μ1⁴ P_shot ] + cycl. + (1/n̄²) A_shot`, β = f/b1; "the Poisson limit is reproduced with P_shot, B_shot, A_shot → 0."

**`ps_1loop_jax` code (what our coefficients multiply):**
- `ps_1loop.py:589–595`: `P_ctr_k2 = −2 k² [ c0 + c2·f·μ² + c4·f²·μ⁴ ] · P(k,μ)`
- `ps_1loop.py:612–617`: `P_ctr_k4 = − k⁴ · cfog · f⁴ μ⁴ (b1 + f μ²)² · P(k,μ)`
- `ps_1loop.py:667–670`: `P_stoch = (1/n̄) [ P_shot + a0 (k/k_nl)² + a2 μ² (k/k_nl)² ]`, `k_nl = knl_bins[b]` (per bin)
- `bs_tree.py:169`: `Z1_fog = (b1 + f μ²) − c1 · μ² · (k/k_nl_rsd)²`, `k_nl_rsd = 0.45` (production)
- `bs_tree.py:246–252`: `B_stoch = (1/n̄) Σ_i b1² P(k_i) (1 + β μ_i²)(B_shot + P_shot·β μ_i²) + A_shot/n̄²`, β = f/b1
  — expanding the per-leg bracket: `(1+βμ²)(B_shot + P_shot βμ²) = B_shot + βμ²(P_shot + B_shot) + β²μ⁴ P_shot`, i.e. **term-by-term identical to Paper 1 eq (59).**

---

## Layer-1 derivation, per parameter (show work)

Everything below uses the production config (`knl_bins`, `k_nl_rsd = 0.45`), not
library defaults.

**c0, c2, c4** — match ours `−2k²[c0 + c2 f μ² + c4 f² μ⁴]P` to CLASS-PT (2.15)
`−2k²[c̃0 + c̃2 f μ² + c̃4 f² μ⁴]P`, term by term. Same −2, same k², same f/μ
powers, same P (both IR-resummed linear), same units [Mpc/h]².
⇒ **c0=c̃0, c2=c̃2, c4=c̃4: factor 1, offset 0.** No knl normalization on
counterterms (bare k² both sides).

**cfog (their c̃)** — match ours `−k⁴ cfog f⁴μ⁴(b1+fμ²)² P` to CLASS-PT (2.16)
`−c̃ f⁴μ⁴k⁴(b1+fμ²)² P`. Identical sign, k⁴, f⁴μ⁴, **explicit (b1+fμ²)²** (no
absorbed b1/f), units [Mpc/h]⁴. ⇒ **cfog = c̃: factor 1, offset 0.**

**P_shot** — ours `(1/n̄)[P_shot^o + …]` vs Paper 1 (58) `(1/n̄)[1 + P_shot^p + …]`.
Same 1/n̄. Matching the constant: `P_shot^o = 1 + P_shot^p`.
⇒ **factor 1, offset +1** (our mean-1 Poisson amplitude ↔ their mean-0 deviation).

**a0, a2** — ours `(1/n̄) a^o (k/knl_b)²·{1 or μ²}` vs Paper 1 (58) `(1/n̄) a^p
(k/0.45)²·{1 or μ²}`. At identical k: `a^o (k/knl_b)² = a^p (k/0.45)²` ⇒
`a^o = a^p (knl_b/0.45)²`. ⇒ **factor (knl_b/0.45)², offset 0** (per-bin; from
`knl_bins`, not the 0.45 the paper uses). This is the *only* discrepancy for the
scale-dependent stochastic terms: our per-bin `knl_bins` vs their fixed 0.45.

**c1 (bispectrum FoG)** — ours `Z1_fog = Z1 − c1^o μ² (k/0.45)²` (dimensionless
c1); paper c1 has units [Mpc/h]², so it enters `Z1 − c1^p μ² k²` (bare k²). At
identical (k,μ): `c1^o (k/0.45)² = c1^p k²` ⇒ **c1^o = c1^p · 0.45² = 0.2025·c1^p:
factor 0.45² = 0.2025, offset 0.** (Factor uses production `k_nl_rsd = 0.45`, NOT
the `bs_tree.py:31` default 0.3.) Extra-factor check: our c1 enters Z1 *additively
with no f or b1 prefactor* — identical to the verified D'Amico form (2502.14758
eq 3.14, per CONTEXT.md); assumed identical to the Ivanov/[160] form (see open
issue #2).

**B_shot, A_shot** — Paper 1 eq (59) is linear-homogeneous in (P_shot, B_shot,
A_shot) with F(0,0,0)=0 (Poisson subtracted from their data). Our operator is the
*identical function F* (verified term-by-term above), but our data *includes*
Poisson, and our fiducial (P_shot,B_shot,A_shot)=(1,1,1) reproduces the full
redshift-space Poisson bispectrum shot noise `F(1,1,1) = (1/n̄)Σ_i (b1+fμ_i²)²P(k_i)
+ 1/n̄²`. Physical identity `F(our) = Poisson + F(paper) = F(1,1,1)+F(paper)
= F(1+P^p, 1+B^p, 1+A^p)` (linearity) ⇒ **B_shot^o = 1 + B_shot^p and
A_shot^o = 1 + A_shot^p: factor 1, offset +1** — the same mean-1↔mean-0 shift as
P_shot, and the shared P_shot's offset +1 is consistent across eqs (58) and (59).

**bGamma3** — enters our P kernel linearly as `(4/5)b1·bGamma3·F_G2` (+ `(2/5)
bGamma3 F_G2` in real space); dimensionless, standard CLASS-PT McDonald–Roy
normalization. Paper's b_Γ3 dimensionless, same operator. ⇒ **factor 1, offset 0.**
Prior mean `23/42·(b1−1)` uses **raw b1** (McDonald–Roy coevolution relation), not
b1σ8. Our fiducial `bGamma3z = 23/42(b1z−1)` equals the prior mean by construction.

**b2, bG2 (sampled block)** — paper priors are on the σ8-scaled combos `b2σ8²(z)`,
`b_G2σ8²(z)` ~ N[0,5²]. Our sampled variables are **raw** b2, bG2. Map
`b2σ8²(z) = b2^o · σ8²(z)` ⇒ `b2^o = (b2σ8²)/σ8²(z)`: **factor 1/σ8²(z), offset 0**,
so `σ_ours = 5/σ8²(z)` (θ_NL-dependent — *not* a scalar constant). Same for bG2.
(b1 is sampled as raw b1, flat/unbounded per CONTEXT.md; paper's b1σ8 ~ U[0,3];
no numeric row required.)

---

## The map table

Columns exactly as specified. `factor`/`offset` solve `ours = paper × factor +
offset` at identical (k,μ,z). `our mean = paper mean × factor + offset`;
`our sigma = |paper sigma × factor|`. **Values are layer-1 only (pre layer-2).**
Config symbols: `knl_b` = per-bin `knl_bins`; `σ8(z)` = growth-scaled σ8 at bin z.

| param | our operator (file:line) | paper operator (eq. ref) | our units | paper units | factor | offset | layer-2 rescale | paper mean | paper sigma | our mean | our sigma | notes |
|-------|--------------------------|--------------------------|-----------|-------------|--------|--------|-----------------|-----------|------------|----------|-----------|-------|
| c0 | `ps_1loop.py:589,595` (`−2k²·c0·P`) | CLASS-PT (2.15) c̃0 | [Mpc/h]² | [Mpc/h]² | 1 | 0 | A_AP·A_amp | 0 | 30 | 0 | 30 | exact term match; bare k² both sides |
| c2 | `ps_1loop.py:590,595` (`−2k²·c2·fμ²·P`) | CLASS-PT (2.15) c̃2 | [Mpc/h]² | [Mpc/h]² | 1 | 0 | A_AP·A_amp | 30 | 30 | 30 | 30 | **mean 30, not 0**; same f¹μ² power |
| c4 | `ps_1loop.py:591,595` (`−2k²·c4·f²μ⁴·P`) | CLASS-PT (2.15) c̃4 | [Mpc/h]² | [Mpc/h]² | 1 | 0 | A_AP·A_amp | 0 | 30 | 0 | 30 | same f²μ⁴ power |
| cfog | `ps_1loop.py:612–617` (`−k⁴·cfog·f⁴μ⁴(b1+fμ²)²·P`) | CLASS-PT (2.16) c̃ | [Mpc/h]⁴ | [Mpc/h]⁴ | 1 | 0 | A_AP·A_amp | 400 | 400 | 400 | 400 | `(b1+fμ²)²` explicit both sides; no absorbed b1/f |
| c1 | `bs_tree.py:169` (`Z1 − c1·μ²(k/0.45)²`) | Paper1 §V.C / [160]; b'spec k² FoG | dimensionless | [Mpc/h]² | 0.45² = 0.2025 | 0 | A_AP·A_amp | 0 | 5 | 0 | 1.0125 | factor = k_nl_rsd² (production 0.45, **not 0.3**); additive bare-Z1 form assumed (open #2) |
| P_shot | `ps_1loop.py:667,670` (`(1/n̄)P_shot`) + `bs_tree.py:247` | Paper1 (58) P_shot | dimensionless | dimensionless | 1 | +1 | none (A_AP absorbed) | 0 | 1 | 1 | 1 | mean-1 amplitude ↔ mean-0 deviation; single shared param (P & B) |
| a0 | `ps_1loop.py:668,670` (`(1/n̄)a0(k/knl_b)²`) | Paper1 (58) a0 | dimensionless | dimensionless | (knl_b/0.45)² | 0 | none | 0 | 1 | 0 | (knl_b/0.45)² | per-bin knl vs fixed 0.45; see per-bin table |
| a2 | `ps_1loop.py:669,670` (`(1/n̄)a2 μ²(k/knl_b)²`) | Paper1 (58) a2 | dimensionless | dimensionless | (knl_b/0.45)² | 0 | none | 0 | 1 | 0 | (knl_b/0.45)² | same per-bin factor as a0 |
| B_shot | `bs_tree.py:247–250` (`(1/n̄)Σ b1²P(1+βμ²)(B_shot+…)`) | Paper1 (59) B_shot | dimensionless | dimensionless | 1 | +1 | A_AP·A_amp | 0 | 1 | 1 | 1 | identical operator; offset +1 by linearity (Poisson subtracted their side) |
| A_shot | `bs_tree.py:252` (`A_shot/n̄²`) | Paper1 (59) A_shot | dimensionless | dimensionless | 1 | +1 | none | 0 | 1 | 1 | 1 | identical operator; offset +1 |
| bGamma3 | `ps_1loop.py:236,306` (`(4/5)b1·bGamma3·F_G2`, `(2/5)bGamma3·F_G2`) | Table I b_Γ3 / Paper1 §V.C | dimensionless | dimensionless | 1 | 0 | A_AP·A²_amp | 23/42·(b1−1) | 1 | 23/42·(b1−1) | 1 | **θ_NL-dependent mean**; b1 = raw b1 |
| b2 | sampled bias (raw b2) | Table I b₂σ₈²(z) | dimensionless | dimensionless | 1/σ8²(z) | 0 | none | 0 | 5 | 0 | 5/σ8²(z) | **sampled**; θ_NL-dependent width (not scalar) |
| bG2 | sampled bias (raw bG2) | Table I b_G2σ₈²(z) | dimensionless | dimensionless | 1/σ8²(z) | 0 | none | 0 | 5 | 0 | 5/σ8²(z) | **sampled**; θ_NL-dependent width |

### Per-bin a0/a2 layer-1 factor `(knl_b/0.45)²` (production `knl_bins`)

| bin | z | knl_b | (knl_b/0.45)² = our σ(a0)=σ(a2) |
|-----|-----|-------|--------------------------------|
| 0 | 0.7 | 0.52 | 1.3353 |
| 1 | 0.9 | 0.65 | 2.0864 |
| 2 | 1.1 | 0.82 | 3.3205 |
| 3 | 1.3 | 1.02 | 5.1378 |
| 4 | 1.5 | 1.29 | 8.2178 |
| 5 | 1.8 | 1.82 | 16.3575 |
| 6 | 2.2 | 2.88 | 40.9600 |

---

## Confidence + open issues

**High confidence (primary-source, exact factor):**
- **c0, c2, c4** = c̃0, c̃2, c̃4, factor 1: CLASS-PT eq (2.15) and `ps_1loop.py`
  match term-by-term (−2, k², f/μ powers, [Mpc/h]²). No knl on counterterms.
- **cfog = c̃**, factor 1: CLASS-PT eq (2.16) writes `(b1+fμ²)²` and `f⁴μ⁴k⁴`
  explicitly, identical to `ps_1loop.py:612–617`; nothing absorbed. Units [Mpc/h]⁴.
- **P_shot / a0 / a2**: Paper 1 eq (58) verbatim gives the `1 + P_shot + a0(k/kNL)²
  + a2 μ²(k/kNL)²` form with **kNL = 0.45** confirmed in-text; P_shot offset +1,
  a0/a2 factor `(knl_b/0.45)²` fall out algebraically.
- **B_shot / A_shot**: Paper 1 eq (59) verbatim is the identical function to our
  `bs_tree.py:246–252` (verified by expanding our per-leg bracket); linearity +
  Poisson-subtraction ⇒ factor 1, offset +1 for both, consistent with P_shot.
- **bGamma3**: factor 1, mean `23/42(b1−1)` (raw b1), width 1 — all verbatim.
- **Table I priors**: c2 mean **30**, c̃ N(400,400²), c1 N(0,5²), c0/c2/c4 σ=30,
  bΓ3 mean 23/42(b1−1) — all transcribed from the PDF.

**Medium confidence / open issues:**

1. **a0/a2 kNL = 0.45 vs CLASS-PT's bare k².** CLASS-PT eq (2.18) as fetched
   writes *unnormalized* k² and states a0/a2 are neglected in CLASS-PT itself.
   The `(k/0.45)²` normalization is supplied by **Paper 1 eq (58)** (kNL = 0.45
   confirmed verbatim), which is the operative DESI convention. So the factor
   `(knl_b/0.45)²` rests on Paper 1 eq (58), not on CLASS-PT (2.18) directly.
   Solid, but noting the two-source chain.

2. **c1 exact operator form (possible extra f/b1) — NOT independently re-derived.**
   2511.20757 and Paper 1 both *defer* the c1 bispectrum-FoG operator to ref
   [160] (Ivanov), whose explicit equation I could not fetch. I assumed the
   **additive, bare-Z1** form `Z1 − c1 μ²(k/knl)²` (no f, no b1 prefactor),
   matching (a) our `bs_tree.py:169` and (b) the D'Amico form verified in
   CONTEXT.md (2502.14758 eq 3.14). A *multiplicative* form `Z1(1 − c1 k²μ²)`
   would introduce an extra `(b1+fμ²)` factor and change the map. Factor
   **0.45² = 0.2025** holds only under the additive assumption. CONTEXT.md itself
   flags this exact residual ("possible extra f/b1 factors still to be derived").
   **Recommend the CLASS-PT numerical cross-check here if A and B agree elsewhere.**

3. **B_shot/A_shot offset via the Poisson-subtraction argument.** The +1 offsets
   are *derived*, not read from a table: they follow from (i) our operator ≡
   Paper 1 eq (59) function, (ii) eq (59) being linear-homogeneous with Poisson
   limit at 0 ("subtracted from the data at the level of estimators"), (iii) our
   fiducial (1,1,1) = full Poisson. If either paper subtracts Poisson differently
   for B vs P, or our n̄ convention differs from theirs, the A_shot 1/n̄² offset
   in particular could carry an n̄-power factor — I assumed identical n̄
   convention (background number density, same units, both `(1/n̄)`/`(1/n̄²)`).

4. **Paper 1 vs Paper 2 amplitude-power discrepancy (layer-2 only, resolved).**
   `pdftotext` of Paper 1 Table III (two-column) appeared to show `A² c̃` and
   `A b_Γ3` (i.e. c̃→σ8⁴, bΓ3→σ8²), the *opposite* of Paper 2 Table I (`c̃ A_amp`,
   `b_Γ3 A²_amp`). Physical scaling settles it in Paper 2's favor: c̃ multiplies
   P_lin ∝ σ8² (one power A_amp); bΓ3 enters a one-loop term ∝ σ8⁴ (two powers).
   The Paper 1 rendering scrambled the superscripts. **Layer-2 flags in the table
   above follow Paper 2's verbatim Table I** (my reference chain).

**CONTEXT.md contradictions:** none. Every CONTEXT.md-recorded paper value was
*confirmed* against the PDF-verbatim Table I: c2 mean 30 ✓, c̃ N(400,400²) ✓,
c1 N(0,5²) [Mpc/h]² ✓, c0 σ=30 ✓, bΓ3 mean 23/42(b1−1) width 1 ✓, P_shot
mean-1↔mean-0 shift ✓, a0/a2 factor `(knl_b/0.45)²` ✓, c1 factor `0.45²` (not the
0.3 default) ✓. One transcription note: the **ar5iv HTML** of 2511.20757
mislabeled b2σ8²/bG2σ8² as "analytically marginalized"; the **PDF-verbatim**
Table I (and its caption) lists them under **(sampled)** — matching CONTEXT.md's
resolution that b2/bG2 are directly sampled. PDF is authoritative.

**Layer-2 (runtime, not in `our mean/sigma`):** priors are imposed on
`X·A_AP^{a}·A_amp^{p}`; from Table I: bΓ3 → A_AP·A²_amp; c0,c2,c4,c̃,c1 →
A_AP·A_amp; B_shot → A_AP·A_amp; P_shot,a0,a2,A_shot → none (A_AP absorbed into
the stochastic definition; not A_amp-rescaled); b1,b2,bG2 → no A_AP/A_amp (their
σ8-scaling is the layer-1 factor, not layer-2). A_AP = A_amp = 1 at fiducial, so
the layer-1 `our mean/sigma` above are exactly what a Fisher forecast evaluated
at the fiducial sees.
