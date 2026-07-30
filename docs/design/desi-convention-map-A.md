# DESI (2511.20757) ↔ ps_1loop_jax convention map — Derivation A (independent)

**Purpose.** First of two *independent* derivations (Stream-B decision 1, CONTEXT.md
"grill session 2026-07-30") of the layer-1 prior-convention map between the
`ps_1loop_jax` EFT/stochastic coefficient conventions and those of the DESI DR1
reanalysis **arXiv:2511.20757** (Table I) and its companion **Paper 1,
arXiv:2507.13433**, whose operator equations trace to **CLASS-PT, arXiv:2004.10607**.
Derivation B is produced independently; the controller diffs the two.

**Scope.** This is a *derivation/documentation* deliverable only — no code, no tests.
The map is consumed by Task 3 (the `desi_dr1_reanalysis_2511_20757` spec).

**Two-layer structure (CONTEXT.md).**
- *Layer 1* (this document): a per-parameter conversion `ours = paper × factor + offset`
  in matching units, derived by equating operators at identical `(k, μ, z)`.
- *Layer 2* (runtime, θ_NL-dependent): the `A_AP` / `A_amp` rescaling from the Table-I
  footnote. `A_AP = A_amp = 1` at the fiducial cosmology, so the Fisher comparison
  (evaluated at fiducial) sees only layer-1. The `our mean`/`our sigma` columns below
  are the **layer-1-mapped values BEFORE layer-2**.

---

## Sources verified (primary, this session)

| Source | Route | What was read | Verification |
|---|---|---|---|
| **arXiv:2511.20757** ("Reanalyzing DESI DR1: 2", Chudaykin/Ivanov/Philcox) | PDF rendered to image, page 5 | Table I (all 14 rows/bin) + caption/footnote; §II.3 text (pp. 5–6, PDF `pdftotext`) | Table I read **visually** from the rendered page (audit anchor). Caption read from text layer. |
| **arXiv:2507.13433** ("Reanalyzing DESI DR1: 1 / Paper 1") | PDF rendered to image, pages 22–23 | Eq (57) parameter list; Eq (58) P-stochastic; Eq (59) B-stochastic; Table III | Eqs 58/59 and Table III read **visually** from rendered pages. |
| **arXiv:2004.10607** (CLASS-PT, Chudaykin/Ivanov/Simonović/Zaldarriaga) | PDF rendered to image, pages 11–13; text layer | Eqs (2.15)–(2.23) counterterms/stochastic; Eq (6.5)/(C.3)/(C.4) BOSS priors | Eqs 2.15–2.23 read **visually** from rendered page 12 (printed p. 12). |
| `ps_1loop_jax` code | direct read | `ps_1loop.py` L581–672, L236, L306; `bs_tree.py` L168–170, L246–252 | direct |
| production config | direct read | `build_taylor_templates_lcdm.py` L171–179, L244–266 | direct |

DESI-2 (2511.20757) **prints no operator equations** for the stochastic or bispectrum
terms — it defers them to Paper 1 (2507.13433), which is therefore the operator source
for `P_shot/a0/a2` (Eq 58) and `c1/B_shot/A_shot` (Eq 59). CLASS-PT (2004.10607) is the
operator source for the power-spectrum counterterms `c0/c2/c4/c̃` (Eqs 2.15–2.23).

---

## 1. Table I verbatim (arXiv:2511.20757, page 5) — audit anchor

Columns as printed: **Type | Parameter | Default | Prior | Units**. The *Default*
column is blank for every nuisance row. `𝒰[a,b]` = uniform on `[a,b]`; `𝒩(μ,σ²)` =
Gaussian with mean `μ`, variance `σ²`.

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

**Rescaling combination per row (verbatim, the "Parameter" column above):**
`b_Γ₃·A_AP·A²_amp`; `{c₀,c₂,c₄}·A_AP·A_amp`; `c̃·A_AP·A_amp`; `c₁·A_AP·A_amp`;
`P_shot` (none); `a₀` (none); `a₂` (none); `B_shot·A_AP·A_amp`; `A_shot` (none).
`14` params/bin = `11` marginalized + `3` sampled (matches §II.3: "fourteen nuisance
parameters for each redshift chunk, three of which enter only in the bispectrum
model … all except three can be marginalized").

**Caption / footnote (verbatim, text layer):**
> TABLE I. **Model Parameters**: Parameters and priors used in the galaxy clustering
> analyses. Here, 𝒰[a, b] refers to a uniform prior between a and b, whilst 𝒩(µ, σ²)
> denotes a Gaussian distribution with mean µ and variance σ². Bias parameters
> b₁σ₈, b₂σ₈², b_𝒢₂σ₈² are directly sampled in the MCMC chains, whilst the nuisance
> parameters that appear quadratically in the likelihood are marginalized over
> analytically, as discussed in the text. We use the Alcock-Paczynski parameter,
> **A_AP ≡ (H₀ᶠⁱᵈ/H₀)³ · [H(z)/Hᶠⁱᵈ(z)] · [D_Aᶠⁱᵈ(z)/D_A(z)]²**, where the
> super-script "fid" refers to quantities evaluated in the fiducial cosmology assumed
> when converting redshifts to distances in the DESI data [1]. **The A_AP factor is
> absorbed into the definition of the stochastic parameters.** Similarly, we define
> **A_amp ≡ σ₈²(z)/σ₈,ref²(z)**, where σ₈,ref(z) is the late-time fluctuation amplitude
> at the Planck 2018 best-fit cosmology [139].

The clause "the A_AP factor is absorbed into the definition of the stochastic
parameters" explains why `P_shot`, `a₀`, `a₂`, `A_shot` carry **no explicit** A_AP in
the table: their `1/n̄` (and `1/n̄²`) normalization already carries the AP volume
dependence. (`B_shot` *is* shown with an explicit `A_AP·A_amp` because it multiplies the
linear power spectrum `b₁²P^tree`.)

### 1b. Corroboration — Paper 1 (2507.13433) Table III

Paper 1's Table III lists the identical parameter set and priors but with **one power
of `A ≡ A_amp` only, and NO `A_AP`** (Paper 1 predates the AP-rescaling improvement
that 2511.20757 introduces; see §II.3 of 2511.20757: "Second, motivated by [152], we
rescale the nuisance parameters that multiply the linear power spectrum by the AP
amplitude"). Verbatim Paper-1 rescalings: `A·b_Γ₃`, `A·c₀`, `A·c₂`, `A·c₄`, **`A²·c̃`**,
`A·c₁`, `A·B_shot`, and none on `P_shot/a₀/a₂/A_shot`. Prior *values* are identical to
Table I (`c₂→𝒩(30,30²)`, `c̃→𝒩(400,400²)`, `c₁→𝒩(0,5²)`, etc.).

⚠️ **Inter-paper discrepancy (flagged in §Confidence):** the `A_amp` **power on `c̃` and
`b_Γ₃` is swapped** between the two papers — Paper 1: `c̃·A²`, `b_Γ₃·A¹`; DESI-2:
`c̃·A_amp¹`, `b_Γ₃·A²_amp` (the DESI-2 body text confirms "c̃ … and b_Γ₃ are rescaled by
σ₈²(z) and σ₈⁴(z), respectively", i.e. `A_amp¹` and `A_amp²`). This is a **layer-2**
issue only; it does not affect any layer-1 number (all `A`-factors = 1 at fiducial).

---

## 2. Code-side operators (what our coefficients multiply)

All from `ps_1loop_jax` (companion repo). `P(k,μ)` below is the IR-resummed **linear**
power `_get_pk_irres_rsd` (= `pk_nw + e^{−damp}·pk_w`), i.e. `P_lin` with BAO damping.

- **k² counterterm** — `ps_1loop.py:589–595`:
  `P_ctr,k2 = −2·k²·[c0 + c2·f·μ² + c4·f²·μ⁴]·P(k,μ)`  — coefficients in `(Mpc/h)²`.
- **k⁴ FoG** — `ps_1loop.py:612–617`:
  `P_ctr,k4 = −k⁴·cfog·f⁴·μ⁴·(b1+f·μ²)²·P(k,μ)`  — `cfog` in `(Mpc/h)⁴`.
- **P stochastic** — `ps_1loop.py:667–670` (`get_pkmu_stoch`, used by `get_pkmu_pair`→multipoles):
  `P_stoch = (1/n̄)·[P_shot + a0·(k/k_nl)² + a2·(k/k_nl)²·μ²]` — `k_nl` = **per-bin** `knl_bins`.
- **c1 bispectrum FoG** — `bs_tree.py:168–169`:
  `Z1_fog = Z1(b1,f,μ) − c1·μ²·(k/k_nl_rsd)²`, with `Z1 = b1+f·μ²`. `k_nl_rsd = 0.45`
  in production (`build_taylor_templates_lcdm.py:179,233`), **not** the `bs_tree.py:31`
  default `0.3`. `c1` here is **dimensionless**.
- **B stochastic** — `bs_tree.py:246–252`:
  `B_stoch = (1/n̄)·Σ_i b1²P_lin(k_i)·(1+βμ_i²)·(B_shot + P_shot·βμ_i²) + A_shot/n̄²`,
  `β=f/b1`.
- **b_Γ3** — `ps_1loop.py:236` (real) / `:306` (cross): enters linearly as
  `+(4/5)·b1·b_Γ3·F_G2` and `(b_G2 + (2/5)·b_Γ3)·F_G2` (RSD multipoles: same `0.8=4/5`).
- **b2, b_G2** — `ps_1loop.py:230–234`: linear/quadratic in the loop bias terms;
  sampled as **raw** `b2, bG2` (CONTEXT decision, deviation 3).

Production config (`build_taylor_templates_lcdm.py`): `knl_bins = (0.52, 0.65, 0.82,
1.02, 1.29, 1.82, 2.88)` at `z = (0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.2)`; `K_NL_RSD =
0.45`; fiducial `cfog = knl**(-4)`, `P_shot=B_shot=A_shot=1.0`, `a0=a2=c1=0`,
`bGamma3 = (23/42)(b1(z)−1)`, `b1(z)=0.9+0.4z`.

## 3. Paper-side operators (matched primary equations)

- **CLASS-PT Eq (2.15)** μ-space counterterm (the "tilde" basis):
  `P_ctr,∇²δ = −2c̃0·k²P_lin − 2c̃2·f·μ²·k²P_lin − 2c̃4·f²·μ⁴·k²P_lin`, `c̃ℓ` in `[Mpc/h]²`.
- **CLASS-PT Eq (2.16)** k⁴ FoG: `P_ctr,∇⁴δ = −c̃·f⁴·μ⁴·k⁴·(b1+f·μ²)²·P_lin`, `c̃` in `[Mpc/h]⁴`.
- **CLASS-PT Eqs (2.21)–(2.23)** per-multipole basis (the **Table-I** `c0/c2/c4`):
  multipole `ℓ` receives `+c_ℓ·P_ℓ,∇²δ` with a single coefficient per multipole, and the
  **old↔new map (2.23)** is `c0 ≡ c̃0 + (f/3)c̃2 + (f²/5)c̃4`, `c2 ≡ c̃2 + (6f/7)c̃4`,
  `c4 ≡ c̃4`. (§4 below.)
- **Paper 1 Eq (58)** P stochastic:
  `P_stoch = (1/n̄)·[1 + P_shot + a0·(k/k_NL)² + a2·μ²·(k/k_NL)²]`, **k_NL = 0.45 h/Mpc**,
  Poisson limit `P_shot=a0=a2→0`. Dimensionless `P_shot,a0,a2`.
- **Paper 1 Eq (59)** B stochastic:
  `B_stoch = (1/n̄)·b1²P^tree(k1)·[B_shot + βμ1²(P_shot+B_shot) + β²μ1⁴P_shot] + cycl. + A_shot/n̄²`,
  `β=f/b1`, Poisson limit `P_shot=B_shot=A_shot→0`.
  Algebraically `[B_shot+βμ²(P_shot+B_shot)+β²μ⁴P_shot] = (1+βμ²)(B_shot+P_shot·βμ²)`,
  so Eq (59) **≡ `bs_tree.py:246–252` term-for-term.**
- **Paper 1 Eq (57) / CLASS-PT Eq (2.21)** `b_Γ3`: `(2b_G2 + 0.8·b_Γ3)·F_G2` — same `0.8`.

---

## 4. The c0/c2/c4 basis derivation (highest-risk finding)

Our code applies the counterterm as a **single μ-space polynomial** then Legendre-projects:
`P_ctr,k2(k,μ) = −2k²P_lin·[c0 + c2·f·μ² + c4·f²·μ⁴]`. This is **identical in form to
CLASS-PT Eq (2.15)**, i.e. **our `{c0,c2,c4}` are the tilde coefficients `{c̃0,c̃2,c̃4}`**
(a constant/μ²/μ⁴ term that each leak across multipoles: μ² → ℓ=0,2; μ⁴ → ℓ=0,2,4).

DESI Table I / Paper 1 / CLASS-PT put priors on the **per-multipole** `{c0,c2,c4}`
(Eq 2.23; "the basis of counterterms has been changed to have a single free coefficient
for each multipole moment"). Hence **our variables ≠ the paper's variables**; they are
related by Eq (2.23). Inverting Eq (2.23) (`ours = c̃` in terms of `paper = c`):

```
our_c4 = paper_c4
our_c2 = paper_c2 − (6f/7)·paper_c4
our_c0 = paper_c0 − (f/3)·paper_c2 + (3f²/35)·paper_c4
```
(`3f²/35 = (f/3)(6f/7) − f²/5`.) `f = f(z)` at the effective redshift, fiducial cosmology.

**Consequence for the prior.** The paper's *diagonal* prior on `(c0,c2,c4)` maps to a
**correlated, f-shifted** Gaussian on our `(c0,c2,c4)`:
- `our_c4`: `𝒩(0, 30²)` — **exact, no mixing** (`c4=c̃4`).
- `our_c2`: mean `30`, var `30²[1+(6f/7)²]`, and `cov(our_c2,our_c4) = −(6f/7)·30²`.
  (At `f≈0.8`: width `≈36.4` not `30`; corr with c4 `≈−0.55`.)
- `our_c0`: **mean `−(f/3)·30 ≈ −8`** (not 0), var `30²[1+(f/3)²+(3f²/35)²]≈(31.1)²`,
  correlated with c2,c4.

A scalar `factor+offset` therefore captures **only the diagonal** (`factor=1, offset=0`,
units `[Mpc/h]²`). The off-diagonal `f`-mixing and the `≈−8 [Mpc/h]²` mean shift on `c0`
are **not** representable in the per-parameter spec. See §Confidence, item **A**.

*Normalization aside:* CLASS-PT's rendered Eq (2.22) `P_ℓ,∇²δ ≡ (2ℓ+1)/2 ∫dμ Lℓ μ^ℓ
f^{ℓ/2} k²P_lin` shows no `−2` while Eq (2.15) has `−2`; the paper's stated Eq (2.23)
relates the two coefficient sets with **no `−2` and no sign flip**, so the diagonal
factor is `+1` (both bases share the `−2` convention). Independent derivation B should
confirm this `+1` (not `−2`/`−½`).

---

## 5. Layer-1 map table

`factor`/`offset`: `ours = paper × factor + offset`. `our mean`/`our sigma` are
layer-1-mapped, **pre**-layer-2. Config symbols: `knl_b` = per-bin `knl_bins`;
`f = f(z)`; `σ8(z)` = linear `σ8` at effective `z`, fiducial cosmology.

| param | our operator (file:line) | paper operator (eq. ref) | our units | paper units | factor | offset | layer-2 rescale | paper mean | paper sigma | our mean | our sigma | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **c0** | `ps_1loop.py:589–595` | CLASS-PT 2.15/2.21a/2.23; Table I `c₀` | `[Mpc/h]²` | `[Mpc/h]²` | `1` (diag only) | `0` | `A_AP·A_amp` | `0` | `30` | `0` (diag) | `30` (diag) | **NON-diagonal**: `our_c0 = paper_c0 − (f/3)paper_c2 + (3f²/35)paper_c4`; true prior mean `≈ −(f/3)·30 ≈ −8`, width `≈31.1`, correlated. §4. |
| **c2** | `ps_1loop.py:589–595` | CLASS-PT 2.15/2.21b/2.23; Table I `c₂` | `[Mpc/h]²` | `[Mpc/h]²` | `1` (diag only) | `0` | `A_AP·A_amp` | `30` | `30` | `30` (diag) | `30` (diag) | **NON-diagonal**: `our_c2 = paper_c2 − (6f/7)paper_c4`; true width `≈36.4`, corr with c4. §4. |
| **c4** | `ps_1loop.py:589–595` | CLASS-PT 2.15/2.21c/2.23; Table I `c₄` | `[Mpc/h]²` | `[Mpc/h]²` | `1` | `0` | `A_AP·A_amp` | `0` | `30` | `0` | `30` | **Exact** (`c4=c̃4`), no mixing. |
| **cfog** (`c̃`) | `ps_1loop.py:612–617` | CLASS-PT 2.16; Table I `c̃` | `[Mpc/h]⁴` | `[Mpc/h]⁴` | `1` | `0` | `A_AP·A_amp` | `400` | `400` | `400` | `400` | Operators identical incl. explicit `(b1+fμ²)²`. Single coefficient (no basis mixing). |
| **a0** | `ps_1loop.py:667–669` | Paper 1 Eq 58; Table I `a₀` | dimensionless | dimensionless | `(knl_b/0.45)²` | `0` | none (A_AP absorbed) | `0` | `1` | `0` | `(knl_b/0.45)²` | Per-bin `knl_b` vs paper fixed `k_NL=0.45`. Per-bin σ: `1.335, 2.086, 3.320, 5.138, 8.218, 16.36, 40.96`. |
| **a2** | `ps_1loop.py:667–669` | Paper 1 Eq 58; Table I `a₂` | dimensionless | dimensionless | `(knl_b/0.45)²` | `0` | none (A_AP absorbed) | `0` | `1` | `0` | `(knl_b/0.45)²` | Same per-bin σ as a0. |
| **P_shot** | `ps_1loop.py:667` + `bs_tree.py:238` | Paper 1 Eqs 58 & 59; Table I `P_shot` | dimensionless | dimensionless | `1` | `+1` | none (A_AP absorbed) | `0` | `1` | `1` | `1` | Mean-1 amplitude (ours) vs mean-0 deviation (Poisson subtracted, paper). **Shared** P- and B-side param; offset `+1` verified consistent in *both* (§6). |
| **b_Γ3** | `ps_1loop.py:236,306` | CLASS-PT 2.21 `(2bG2+0.8bΓ3)F_G2`; Table I `b_Γ₃` | dimensionless | dimensionless | `1` | `0` | `A_AP·A²_amp` | `(23/42)(b1−1)` | `1` | `(23/42)(b1−1)` | `1` | Mean uses **raw b1** (= `(b1σ8)/σ8(z)` at runtime), not `b1σ8`. Same `0.8=4/5` coefficient. Our fiducial matches. |
| **c1** | `bs_tree.py:168–169` | Paper 1 Eq 57 / ref [160] (`k²μ²` FoG); Table I `c₁` | dimensionless | `[Mpc/h]²` | `0.45² = 0.2025` | `0` | `A_AP·A_amp` | `0` | `5` | `0` | `1.0125` | `k_nl_rsd = 0.45` (production), not `0.3`. Paper c1 assumed **bare `k²μ²`** (units imply it); if ref [160] normalizes by a `k_NL`, revisit. No extra f/b1 (both fold c1 into `Z1`). |
| **B_shot** | `bs_tree.py:246–250` | Paper 1 Eq 59; Table I `B_shot` | dimensionless | dimensionless | `1` | `+1` | `A_AP·A_amp` | `0` | `1` | `1` | `1` | Mean-1 amplitude vs mean-0 deviation. Structure identical to Eq 59. |
| **A_shot** | `bs_tree.py:252` | Paper 1 Eq 59; Table I `A_shot` | dimensionless | dimensionless | `1` | `+1` | none | `0` | `1` | `1` | `1` | `A_shot/n̄²` term; mean-1 amplitude vs mean-0 deviation. |
| **b2** (sampled) | `ps_1loop.py:230,232,234` | Table I `b₂σ₈²` (sampled) | dimensionless (raw b2) | dimensionless (`b2σ8²`) | `1/σ8²(z)` | `0` | none (bias not AP-rescaled, fn. 9) | `0` | `5` | `0` | `5/σ8²(z)` | Paper samples `b2σ8²`; we sample raw `b2`. Width rule `5/σ8²(z)`. |
| **bG2** (sampled) | `ps_1loop.py:231,233` | Table I `b_𝒢₂σ₈²` (sampled) | dimensionless (raw bG2) | dimensionless (`bG2σ8²`) | `1/σ8²(z)` | `0` | none (fn. 9) | `0` | `5` | `0` | `5/σ8²(z)` | Same as b2. |

`b1` (sampled) is not a map row: paper `b1σ8 ~ 𝒰[0,3]`; we sample raw `b1` **flat/unbounded**
(CONTEXT deviation 3). Recover paper variable via `b1σ8 = b1·σ8(z)` for reporting only.

---

## 6. Per-parameter derivation notes

**c0/c2/c4** — see §4. Diagonal factor `1`, units `[Mpc/h]²`; off-diagonal `f`-mixing
per inverse Eq (2.23) is the headline caveat.

**cfog (`c̃`)** — `ps_1loop.py:612–617` `−k⁴·cfog·f⁴μ⁴(b1+fμ²)²P_lin` vs CLASS-PT Eq
(2.16) `−c̃·f⁴μ⁴k⁴(b1+fμ²)²P_lin`. Character-for-character identical ⇒ `cfog = c̃`,
factor `1`, `[Mpc/h]⁴`. `c̃` is a single coefficient (not per-multipole) ⇒ no basis mixing.

**a0, a2** — set `our_a0·(k/knl_b)² = paper_a0·(k/0.45)²` (Eq 58) ⇒
`our_a0 = paper_a0·(knl_b/0.45)²`. Factor is a **config formula** `(knl_b/0.45)²`
(per-bin, since our stochastic uses per-bin `knl_bins` while the paper fixes `k_NL=0.45`).
Paper 1 confirms: "a_n are scaled by `k_NL^{−2} n̄^{−1}`". Layer-2 = none (A_AP absorbed
in `1/n̄`).

**P_shot / B_shot / A_shot (offset +1)** — Paper Eqs 58/59 subtract the Poisson term at
the estimator level, so their shot params are **mean-0 deviations** (Poisson limit → 0).
Our code carries the Poisson `1` inside the coefficients (production fiducials
`P_shot=B_shot=A_shot=1.0`). Verification that `offset=+1` is consistent across P **and**
B for the *shared* `P_shot`:
- P-side: `(1/n̄)[P_shot_ours + …] = (1/n̄)[1 + P_shot_paper + …]` ⇒ `P_shot_ours = 1 + P_shot_paper`. ✓
- B-side: with `P_shot_ours=1+P_p`, `B_shot_ours=1+B_p`,
  `(1+βμ²)(B_shot_ours + P_shot_ours·βμ²) = (1+βμ²)[(1+βμ²) + B_p + P_p βμ²]`
  `= [paper Poisson (1+βμ²)²] + [Eq 59 deviation (1+βμ²)(B_p + P_p βμ²)]`. ✓
  i.e. our fiducial `=1` **exactly reproduces the full Kaiser-boosted Poisson**
  `Σ b1²P(k_i)(1+βμ_i²)²/n̄ + 1/n̄²`, and the deviation maps 1:1 to Eq (59). So
  `factor=1, offset=+1` for all three, with no hidden μ/β factor.
- `1/n̄` convention matches: paper "n̄ = background galaxy density"; code `params['ndens']`
  = per-bin `n_bar` (`~3×10⁻⁴ (h/Mpc)³`). At fiducial identical; off-fiducial the paper's
  P_shot/a0/a2 *definition* absorbs A_AP (hence layer-2 = none, matching the Table-I
  footnote), whereas ours uses the plain fiducial `n̄` — consistent at fiducial (where the
  Fisher comparison lives).

**b_Γ3** — dimensionless bias, identical operator coefficient (`0.8=4/5`) and same
co-evolution prior mean `(23/42)(b1−1)` with width `1`. Factor `1`, offset `0`. The mean
is expressed in **raw b1**; the paper obtains raw `b1 = (b1σ8)/σ8(z)` from its sampled
variable. Our fiducial `bGamma3z = (23/42)(b1z−1)` (`build_taylor…:247`) matches. Layer-2
`A_AP·A²_amp` (DESI-2 convention; Paper 1 uses `A¹` — see §1b/Confidence).

**c1** — both fold c1 into the linear Kaiser factor as a FoG counterterm:
ours `Z1 − c1_ours·μ²(k/0.45)²`; paper's c1 is a **`k²`** FoG term in `[Mpc/h]²` (Paper 1
line ~1517; the D'Amico/Ivanov `Z1 → Z1 − c1·k²μ²` lineage, CONTEXT c1 section). Equate:
`c1_ours·μ²(k/0.45)² = c1_paper·k²μ²` ⇒ `c1_ours = c1_paper·0.45²`. Factor `0.45²=0.2025`,
offset `0`; `our σ = 5·0.2025 = 1.0125`. No extra `f`/`b1` factor. **Assumption:** the
paper's c1 operator is bare `k²μ²` (units `[Mpc/h]²` on c1 imply this). If ref [160]
instead writes `(k/k_NL)²μ²` with some fixed `k_NL`, the factor becomes `(0.45/k_NL)²`
(= 1 if that `k_NL` is also 0.45). Flagged in Confidence, item **C**.

**b2, bG2** — paper samples `b2σ8²`, `bG2σ8²` (`𝒩(0,5²)`); we sample raw. `b2σ8² =
b2·σ8²(z)` ⇒ prior on raw `b2` is `𝒩(0, (5/σ8²(z))²)`. Factor `1/σ8²(z)`
(cosmology/z-dependent), offset `0`. No AP rescale (footnote 9).

---

## 7. Confidence + open issues

**Overall:** all four primary sources reached and read at the equation/table level
(DESI-2 Table I and Paper 1 Eqs 58/59 + Table III read from rendered page images; CLASS-PT
Eqs 2.15–2.23 from rendered page 12). Every prior value in Table I is reproduced. Agreement
with CONTEXT.md on the checked values (below). Confidence is **high** except items A and C.

**Cross-check vs CONTEXT.md (no silent deference):**
- ✅ `c2` mean `30` — matches Table I `𝒩(30,30²)`.
- ✅ `c̃` `𝒩(400,400²) [Mpc/h]⁴` — matches.
- ✅ `c1·A_AP·A_amp ~ 𝒩(0,5²) [Mpc/h]²` — matches Table I exactly; c1 IS in Table I
  (row present, `[Mpc/h]²`), confirming CONTEXT's "c1 strictly linear, analytically
  marginalized in 2511.20757".
- ✅ `a0/a2` factor `(knl_b/0.45)²`, per-bin; `k_nl_rsd=0.45` production override — matches
  CONTEXT decision 4 and the c1 section correction.
- ✅ `P_shot` mean-1↔mean-0 (offset +1) — matches; **extended** here to `B_shot`, `A_shot`.
- ✅ `b2σ8²/bG2σ8² → 5/σ8²(z)` raw width — matches CONTEXT deviation 3 / decision 3.
- **No contradiction found** with any recorded CONTEXT value.

**Open issues / items derivation B must independently confirm:**

**A. (HIGH RISK — headline) c0/c2/c4 are in a different basis than the paper's priors.**
Our `{c0,c2,c4}` are the μ-space `{c̃0,c̃2,c̃4}` (Eq 2.15); Table I priors are on the
per-multipole `{c0,c2,c4}` (Eqs 2.21–2.23). A scalar `factor+offset` captures only the
diagonal (`factor=1, offset=0`); the true map is the triangular `f`-dependent inverse of
Eq (2.23) (§4), which turns the paper's diagonal prior into a **correlated, f-shifted**
prior on our coefficients — notably a ≈`−(f/3)·30 ≈ −8 [Mpc/h]²` mean shift on `c0` and a
≈20% width inflation on `c2`. **This contradicts the framework's per-parameter-factor
assumption** (CONTEXT.md "Two-layer convention map": "a per-parameter factor converting
our raw coefficient definition (e.g. counterterm −2k²c0·P) to the paper's"). Task 3 must
decide: (i) accept the basis mismatch and place the paper's diagonal priors directly on
our `c̃` (a documented O(f) approximation, tolerable because these are wide-prior,
analytically-marginalized nuisances), or (ii) implement the full triangular transform
`c = L(f)·c̃` so the paper's prior is imposed in the correct basis. My recommendation:
**do (ii) for correctness**, or explicitly document (i). This is exactly the kind of
silent bias the dual derivation exists to catch — if derivation B reports `factor=1` with
no basis caveat, that is the discrepancy to escalate.

**B. (MEDIUM) Layer-2 A_amp powers on `c̃` and `b_Γ3` differ between the two DESI papers.**
DESI-2 (anchor): `c̃·A_AP·A_amp¹`, `b_Γ3·A_AP·A²_amp` (matches its own body text σ₈²/σ₈⁴).
Paper 1: `c̃·A²`, `b_Γ3·A¹` (swapped, and no A_AP). I follow **DESI-2** (the designated
reference). Layer-2 only ⇒ **no layer-1 number affected** (A=1 at fiducial). Derivation B
should confirm the DESI-2 powers and note the swap.

**C. (LOW–MEDIUM) c1 paper-operator normalization not read from a printed equation.**
Paper 1 states c1 is a "k² fingers-of-God term" (`[Mpc/h]²`) and defers the explicit
operator to ref [160]. I inferred bare `k²μ²` (⇒ factor `0.45²`) from the `[Mpc/h]²`
units and the CONTEXT-verified D'Amico `Z1 → Z1 − c1·μ²(k/k_NL)²` lineage. **Not**
confirmed from ref [160]'s equation this session. If [160] uses `(k/k_NL)²μ²` with a
fixed `k_NL`, the factor is `(0.45/k_NL)²` (`=1` if `k_NL=0.45`). Numerically the
CONTEXT value (`≈1.01`) agrees with factor `0.45²`. Derivation B should try to reach ref
[160] (likely Ivanov et al. tree-bispectrum) for the printed c1 operator.

**D. (LOW) Stochastic `1/n̄` AP absorption.** The Table-I footnote absorbs A_AP into the
stochastic-parameter *definition*; our code uses the plain fiducial `n̄`. Identical at
fiducial (where Fisher lives); off-fiducial the paper's P_shot/a0/a2 carry an implicit
A_AP our layer-2=none does not apply. Consistent with the Table (no explicit A_AP on those
rows) but worth a runtime note for Task 3.

**E. (LOW) `f`, `σ8(z)` in factors are cosmology/z-dependent.** The `a0/a2` `(knl_b/0.45)²`
is pure-config (cosmology-independent). But `b2/bG2` `1/σ8²(z)` and the c0/c2/c4 `f`-mixing
depend on the (fiducial) cosmology and effective redshift — they are constants only once
the fiducial cosmology and effective z per bin are fixed. The spec/loader must evaluate
them at fiducial (as the Fisher side does).
