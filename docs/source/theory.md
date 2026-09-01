# Theory

This page explains what jaxPTPolyPol computes and why it is built the way it is. It is
background reading, not a recipe — for runnable code see {doc}`usage`, and for exact
signatures see the {doc}`api/index`.

The short version: the package forecasts and infers cosmological parameters from the
galaxy power-spectrum and bispectrum multipoles of a PFS-like spectroscopic survey,
optionally combined with BAO, CMB, and BBN information. The effective-field-theory model
that makes this possible carries a large number of nuisance parameters — 14 per redshift
bin, 98 across the seven production bins — and almost all of the design in this repository
exists to make that nuisance load tractable without approximating it away.

## The model

### Observables

The data vector is built from two statistics of the galaxy density field, evaluated
independently in each redshift bin:

- the redshift-space **power-spectrum multipoles** $P_\ell(k)$ for $\ell = 0, 2, 4$, at
  1-loop order in perturbation theory;
- the **bispectrum monopole** $B_0(k_1, k_2, k_3)$ at tree level, on the triangle
  configurations that close within the chosen $k$ range.

Both come from `ps_1loop_jax`; jaxPTPolyPol wraps them in {doc}`api/model` and assembles
them into a single per-bin block by {doc}`api/theory`'s `make_joint_pk_bk_fn`. Stacking
bins gives the full data vector, and because bins are treated as independent volumes the
Gaussian covariance is block diagonal across them — a fact the marginalization exploits
heavily.

### Counterterms

Perturbation theory at 1-loop order is not a closed prediction: the loop integrals reach
into wavenumbers where the perturbative expansion has broken down, and the EFT repairs
this with counterterms whose coefficients are free parameters. Two enter the power
spectrum. The $k^2$ counterterm is a single polynomial in $\mu$,

$$
P^{k^2}_{\rm ctr}(k,\mu) = -2\,k^2 P(k)\,
\bigl[\, c_0 + c_2 f \mu^2 + c_4 f^2 \mu^4 \,\bigr],
$$

and the $k^4$ "fingers-of-God" counterterm carries the squared linear Kaiser factor,

$$
P^{k^4}_{\rm ctr}(k,\mu) = -\,k^4 \, c_{\rm fog}\, f^4 \mu^4 \,
\bigl(b_1 + f\mu^2\bigr)^2 P(k).
$$

Here $f$ is the linear growth rate and $P(k)$ is the (IR-resummed, when enabled) linear
power spectrum. The bispectrum carries its own fingers-of-God counterterm, folded into the
first-order kernel on each triangle leg:

$$
Z_1^{\rm FoG}(k,\mu) = b_1 + f\mu^2 - c_1\,\mu^2 \bigl(k / k^r_{\rm nl}\bigr)^2 .
$$

`k_nl_rsd` — the $k^r_{\rm nl}$ above — defaults to $0.3\,h/\mathrm{Mpc}$ in
`ps_1loop_jax`, but the production configuration in this repository sets it to
$0.45\,h/\mathrm{Mpc}$. That distinction matters: it changes the units in which $c_1$ is
measured, and therefore the width of its prior. Prior conventions must always be derived
from the production configuration, never from a library default.

### Stochasticity

Galaxies are a discrete tracer, so the model adds a shot-noise-like term with free
amplitude and scale dependence:

$$
P_{\rm stoch}(k,\mu) = \frac{1}{\bar n}
\Bigl[\, P_{\rm shot} + a_0 \bigl(k/k_{\rm nl}\bigr)^2
       + a_2 \mu^2 \bigl(k/k_{\rm nl}\bigr)^2 \,\Bigr].
$$

$P_{\rm shot}$ is an *amplitude* here, with fiducial value 1 (pure Poisson); the DESI
reference paper parameterises the same freedom as a mean-zero deviation from Poisson, so
translating between the two conventions requires an offset, not just a rescaling. The
bispectrum adds two more stochastic amplitudes, $B_{\rm shot}$ multiplying the linear
power and $A_{\rm shot}/\bar n^2$ a pure constant, and it re-uses the *same* $P_{\rm shot}$
— one parameter, two statistics, never duplicated as two columns.

### IR resummation and the Alcock–Paczyński effect

Large-scale bulk flows smear the BAO wiggles. `ps_1loop_jax` handles this by splitting
$P_{\rm lin}$ into smooth and wiggle parts and damping the wiggle by
$\exp(-k^2\Sigma^2_{\rm tot})$, with an anisotropic
$\Sigma^2_{\rm tot} = [1 + \mu^2 f(2+f)]\Sigma^2 + f^2\mu^2(\mu^2-1)\,\delta\Sigma^2$.

Converting observed angles and redshifts into distances requires assuming a cosmology. If
the assumed (fiducial) cosmology differs from the true one, the inferred clustering is
distorted — the Alcock–Paczyński effect. The distortion is parameterised by two ratios,

$$
\alpha_\perp = \frac{D_A^{\rm true}(z)}{D_A^{\rm fid}(z)}, \qquad
\alpha_\parallel = \frac{H^{\rm fid}(z)}{H^{\rm true}(z)} ,
$$

and the work is split across the two repositories: `ps_1loop_jax` accepts
$(\alpha_\perp, \alpha_\parallel)$ as inputs, while jaxPTPolyPol computes them from
cosmology. The fiducial distances are evaluated *once*, outside any JIT boundary, by
`compute_fiducial_distances`, and captured as static constants in the theory closure.

### Factory closures

Every theory function in {doc}`api/theory` is produced by a factory:
`make_pk_ell_fn(...)` returns a closure `pk_fn(params, *, k)`. All static configuration —
emulator, 1-loop model, multipole list, AP switch, redshift bins, fiducial distances,
whether $\Sigma m_\nu$ is varied — is captured in the closure scope, so the returned
function's only traced input is the parameter vector. This is why nothing in the package
needs `static_argnames`: `jax.jit(pk_fn)` and `jax.jacfwd(pk_fn)` just work.

## The parameter vector

Parameters live in two JAX pytrees, `CosmoParams` and `FullShapeSurveyParams`
({doc}`api/params`), and are flattened into a single packed array for differentiation:

```text
[ cosmology | survey bin 0 | survey bin 1 | ... | survey bin N-1 ]
```

Cosmological parameters are **shared** across bins; EFT and stochastic parameters are
**per bin**. Parameter *names and ordering* are static — they are compilation constants
baked into the pytree's aux data — while parameter *values* are traced and
differentiable. Nothing in the pipeline may make the key set dynamic.

## Two kinds of nuisance parameter

The central structural fact of this codebase is that the joint $P+B$ theory vector is
**exactly linear** in most of its nuisance parameters. Write the theory as

$$
t(\theta) = m_0(\theta_{\rm NL}) + M(\theta_{\rm NL})\,\theta_{\rm lin},
$$

where $m_0$ is the theory evaluated at $\theta_{\rm lin} = 0$ and $M = \partial t /
\partial \theta_{\rm lin}$ is a matrix of *templates* — one column per linear parameter.
For all but one of the linear-block parameters below, this decomposition is not an
approximation — it has been verified empirically to the float64 floor. The exception is
$c_1$, where the linearity is imposed rather than found; the next section explains why
that is safe. Note that the template columns may themselves depend on
$\theta_{\rm NL}$ (the $c_{\rm fog}$ column carries a factor $(b_1 + f\mu^2)^2$, for
instance); that is perfectly compatible with linearity in $\theta_{\rm lin}$.

The split is:

$\theta_{\rm lin}$ — the linear (marginalized) block
: eleven parameters per redshift bin:
  $\{c_0, c_2, c_4, c_{\rm fog}, a_0, a_2, P_{\rm shot}, b_{\Gamma_3}, c_1, B_{\rm shot},
  A_{\rm shot}\}$. On the seven production bins that is 77 parameters. They are never
  sampled; they are integrated out in closed form.

$\theta_{\rm NL}$ — the sampled block
: the varied cosmological parameters plus $\{b_1, b_2, b_{\mathcal{G}_2}\}$ per bin. For
  the production $\Lambda$CDM configuration — five cosmological parameters and seven bins
  — that is 26 dimensions; $\nu\Lambda$CDM adds $\Sigma m_\nu$ for 27.

Eleven plus three is fourteen nuisances per bin, matching Table I of the DESI DR1
reanalysis (arXiv:2511.20757), whose phrasing — parameters that "appear quadratically in
the likelihood" are marginalized analytically — is the same statement seen from the
likelihood side: a Gaussian likelihood is quadratic in any parameter the model is linear
in.

### Why $c_1$ sits in the linear block

`ps_1loop_jax` folds $c_1$ into $Z_1^{\rm FoG}$ *before* forming the product
$Z_1^{\rm FoG}(k_i)\,Z_1^{\rm FoG}(k_j)\,Z_2$, so the underlying theory contains a $c_1^2$
cross term and is not strictly linear in $c_1$. Placing $c_1$ in $\theta_{\rm lin}$
therefore means modelling it by its first-order template and dropping that quadratic
piece.

This is a deliberate choice, and it is the treatment the reference paper uses. Its cost
has been bounded rather than assumed: the omitted signal is $4.6\times10^{-5}\sigma$ in
whitened data space at $|c_1| = 1$, and $1.2\times10^{-3}\sigma$ even at a $5\sigma$ draw
of the prior. Two things make the bound clean. First, $c_1$ has no bilinear coupling to
any other linear parameter, so the entire discrepancy between the marginalized and
sampled treatments is one constant coefficient. Second, $c_1$ is prior-dominated in
practice — the data barely constrains it — so its quadratic term cannot propagate to
cosmology. The physics licence is that $c_1^2$ is $\mathcal{O}(k^4\mu^4)$, the same order
as terms the model omits anyway.

There is also a statistical reason to prefer it: the Fisher formalism *already* linearises
$c_1$ when it takes a Schur complement. Marginalizing $c_1$ on the MCMC side makes the
sampled/marginalized partition identical on both sides, which is what makes Fisher and
MCMC results directly comparable.

## The marginal likelihood

Because the theory is linear in $\theta_{\rm lin}$ and the priors on $\theta_{\rm lin}$
are Gaussian, the integral over those 77 parameters has a closed form. This is the
production inference path: full-space NUTS over every EFT nuisance is not merely slow but
computationally infeasible (the XLA compile wall), and analytic marginalization is the
literature-standard alternative.

$$
-2 \ln L_{\rm marg}
 = \tilde r^{\mathsf T} C^{-1} \tilde r
 - b^{\mathsf T} A^{-1} b
 + \ln \det\bigl(A \Sigma_p\bigr),
$$

with

$$
\tilde r = d - m_0 - M\mu_p, \qquad
A = M^{\mathsf T} C^{-1} M + \Sigma_p^{-1}, \qquad
b = M^{\mathsf T} C^{-1} \tilde r .
$$

The symbols:

$d$
: the data vector (a noiseless fiducial mock in every forecast in this repository).

$C$
: the data covariance; $C^{-1}$ is precomputed per bin.

$m_0, M$
: the templates defined above, both functions of $\theta_{\rm NL}$.

$\mu_p, \Sigma_p$
: the Gaussian prior mean and covariance on $\theta_{\rm lin}$. Both are **functions of**
  $\theta_{\rm NL}$ — see [Priors](#priors) — and the API is required to accept them as
  such. A constant-only interface cannot express the reference priors.

$A$
: the posterior precision of $\theta_{\rm lin}$ at fixed $\theta_{\rm NL}$: data
  information plus prior information.

$b$
: the data's pull on $\theta_{\rm lin}$, so $A^{-1}b$ is the conditional best fit and
  $b^{\mathsf T}A^{-1}b$ is the $\chi^2$ that fitting the linear block buys back.

A parameter-independent constant — $\ln \det C$ in the $-2\ln L$ convention
used here — is dropped throughout.

### The log-determinant term

$\ln \det (A\Sigma_p)$ is the volume factor of the marginalization: it measures how much
$\theta_{\rm lin}$ space the data has closed off, and it varies with $\theta_{\rm NL}$
through $M$ and $\Sigma_p$. It is a genuine part of the marginal posterior, and it has
consequences. Measured on the seven-bin production chain, this term alone shifts the
posterior *mean* off the fiducial by $\mathcal{O}(0.1\text{–}0.5)\,\sigma_F$ even on a
noiseless mock.

A Gauss–Newton Fisher matrix cannot contain that term. So mean-level Fisher-versus-MCMC
comparisons are only meaningful if one of two things is done: compare against the tilted
centre $\mu = \theta^{\rm fid} + F^{-1}\nabla(-\tfrac12 \ln\det A\Sigma_p)$, or run the
comparison chain with the term switched off. `include_logdet=False` does the latter, and
is exactly the "Jeffreys prior" best-fit convention of the reference paper. Widths and
correlations are unaffected either way.

### Mean versus mode

Under the full DESI prior specification the effect goes further. The AD-tilted centre
$\mu = \theta^{\rm fid} + F^{-1}\nabla \ln p(\theta^{\rm fid})$ predicts the posterior
**mode** to within $0.06\,\sigma_F$, but the chain **mean** sits about $1.1\,\sigma_F$
away along $\log A$ — and this persists with the log-determinant term removed. The
mechanism is decomposable: the $\theta_{\rm NL}$-dependent prior *widths* account for the
$n_s$ pull entirely and roughly a third of $\log A$'s; the remainder is intrinsic
curvature of the marginal posterior.

The resulting methodology rule is worth stating plainly, because it governs how every
validation gate in this repository is written: **under this prior specification,
mean-level Fisher–MCMC checks must target the mode, not the raw mean. Second moments —
widths and correlations — are the quantitative gates.** In practice widths agree at
0.986–1.014 and correlation differences stay below 0.021.

## The Taylor surrogate

Evaluating the marginal likelihood exactly requires rebuilding $m_0$ and $M$ from the full
theory graph on every call — a `jax.linearize` pass through the whole `ps_1loop_jax`
computation. On the production seven-bin $P+B$ configuration that costs about five seconds
per posterior evaluation, which puts a converged chain out of reach.

The surrogate replaces the reconstruction with a low-order Taylor expansion of the
templates about the fiducial $\theta_{\rm NL}$, built once and cached:

$$
m_0(\theta_0 + u) \simeq m_{00} + J u + \tfrac12 u^{\mathsf T} H u, \qquad
M(\theta_0 + u) \simeq M_0 + \mathrm{d}M \cdot u .
$$

Each subsequent evaluation is then a handful of dense tensor contractions — milliseconds
instead of seconds — which is what turns a multi-day chain into a run of minutes.

Three things about this are worth understanding.

**$M$ must be expanded too.** It would be tempting to freeze $M$ at $M(\theta_0)$ and
expand only $m_0$. That would silently delete part of the log-determinant tilt: the only
route by which the *templates* reach $\ln\det(A\Sigma_p)$ is through how $M$ varies with
$\theta_{\rm NL}$, so a frozen $M$ would make that contribution vanish. (The prior
covariance $\Sigma_p(\theta_{\rm NL})$ carries its own $\theta_{\rm NL}$ dependence, but it
is supplied at runtime rather than expanded — which is what keeps the templates
prior-independent.) So $M$
carries its first-order variation $\mathrm{d}M$, while $m_0$ — which enters the residual
quadratically — is carried to second order.

**It is exact at the expansion point.** At $\theta_{\rm NL} = \theta_0$ the surrogate
reproduces the exact marginal posterior to the float64 floor. Every notebook that uses it
asserts this identity as a wiring tripwire before sampling.

**It is prior-independent.** The templates encode the theory, not the priors, so changing
the prior specification does not require a rebuild — the cached `.npz` is reused. This
matters because the build is expensive: roughly 41 minutes of wall time at about 36 GB
peak resident memory for the production configuration.

The whole expansion is computed forward-over-forward (`jax.jacfwd` on top of the inner
`jax.linearize`), in column chunks. A reverse-mode tape over that graph is what exhausts
compile memory, so it is never taken.

## Priors

The EFT and stochastic parameters are only weakly constrained by the data, so their
priors do real work. This repository adopts the specification of the DESI DR1 reanalysis
(arXiv:2511.20757, Table I), packaged as `desi_dr1_reanalysis_2511_20757` and loaded by
`load_desi_prior_spec` ({doc}`api/desi_priors`). Fisher and MCMC consume the same spec —
shared priors are what make the comparison between them meaningful.

Adopting someone else's priors is not a matter of copying numbers. The paper's
coefficients are defined by *their* operators in *their* units, and translating them into
ours takes two distinct layers.

### Layer 1 — convention mapping

A per-parameter conversion `ours = paper × factor + offset`, derived by equating operators
at identical $(k, \mu, z)$. Most rows are simple: $c_4$ and $c_{\rm fog}$ map with factor
1; $a_0$ and $a_2$ pick up a per-bin factor $(k_{\rm nl,b}/0.45)^2$ because the paper's
stochastic normalisation uses a single fixed scale; $b_2$ and $b_{\mathcal{G}_2}$ pick up
$1/\sigma_8^2(z)$ because the paper samples the $\sigma_8$-scaled combinations;
$P_{\rm shot}$, $B_{\rm shot}$, and $A_{\rm shot}$ take an offset of $+1$ from the
mean-1-amplitude versus mean-0-deviation convention; and $c_1$ takes the factor
$0.45^2 = 0.2025$ from the production `k_nl_rsd`, turning a width of 5 $[\mathrm{Mpc}/h]^2$
into a dimensionless 1.0125.

The counterterms $c_0, c_2, c_4$ are the exception, and they are the reason this
translation was treated as a high-risk item. The paper's priors sit on CLASS-PT's
*per-multipole* counterterm basis — one coefficient per multipole moment — while the code
uses the $\mu$-space "tilde" basis. The two are related by a triangular, $f$-dependent
rotation:

$$
L(f) = \begin{pmatrix}
1 & -f/3 & 3f^2/35 \\
0 & 1 & -6f/7 \\
0 & 0 & 1
\end{pmatrix},
\qquad \tilde c = L(f)\, c .
$$

A *diagonal* prior in the paper's basis is therefore a *correlated* prior in ours. At the
production fiducial this shifts the $c_0$ mean by roughly $-8$ to $-10\,[\mathrm{Mpc}/h]^2$
across the seven bins, inflates the $c_2$ width by 22–30%, and induces
$\mathrm{corr}(c_2, c_4) \approx -0.57$ to $-0.64$. The spec ships the exact correlated
per-bin block, assembled at runtime as $L(f)\,\mathrm{diag}(\sigma_{\rm paper}^2)\,L(f)^{\mathsf T}$,
rather than approximating it as a diagonal. Getting this wrong would bias every posterior
silently, which is why it was derived twice, independently, from the primary sources.

### Layer 2 — runtime rescaling

The paper's priors are imposed not on the bare coefficients but on rescaled combinations,
with two cosmology-dependent factors:

$$
A_{\rm AP} \equiv \Bigl(\frac{H_0^{\rm fid}}{H_0}\Bigr)^{3}
\frac{H(z)}{H^{\rm fid}(z)}
\Bigl(\frac{D_A^{\rm fid}(z)}{D_A(z)}\Bigr)^{2},
\qquad
A_{\rm amp} \equiv \frac{\sigma_8^2(z)}{\sigma_{8,\rm ref}^2(z)} .
$$

Here $\sigma_{8,\rm ref}(z)$ is the late-time fluctuation amplitude at the Planck 2018
best-fit cosmology.

Priors are on $c_0 A_{\rm AP} A_{\rm amp}$, on $b_{\Gamma_3} A_{\rm AP} A_{\rm amp}^2$, and
so on. The effective prior on our raw coefficient therefore has a
$\theta_{\rm NL}$-dependent *width* — and, for $b_{\Gamma_3}$ whose mean follows the
coevolution relation $\tfrac{23}{42}(b_1 - 1)$, a $\theta_{\rm NL}$-dependent *mean*. Both
factors equal 1 at the fiducial cosmology, which is why a Fisher matrix evaluated at the
fiducial sees only layer 1 and still agrees with the MCMC there. Not every parameter is
rescaled: $P_{\rm shot}$, $a_0$, $a_2$, and $A_{\rm shot}$ carry no explicit factor,
because their $1/\bar n$ and $1/\bar n^2$ normalisation already absorbs the AP volume
dependence — exactly at the fiducial, and only approximately away from it. The absorption
runs through $\bar n$, and this code uses the **fiducial** $\bar n$ as a static constant,
so the implicit $A_{\rm AP}$ is reproduced only where $A_{\rm AP} = 1$. A Fisher matrix
evaluated at the fiducial is therefore exact; the MCMC, which samples off-fiducial, carries
a small residual mismatch. See the off-fiducial caveat in
`docs/design/desi-convention-map.md`, which flags it for a future runtime refinement.

### The $b_1\sigma_8$ measure

The paper samples $b_1\sigma_8$ with a flat prior; this repository samples raw $b_1$,
flat. These are *not* the same statement. Flat-in-$b_1\sigma_8$ and flat-in-$b_1$ differ
by the prior weight $\prod_b \sigma_8(z_b; \theta)$, which is cosmology-dependent and so
tilts the posterior. Measured by chain reweighting, the tilt is $+0.173\,\sigma_F$ in
$\log A$ and $+0.105\,\sigma_F$ in $n_s$; widths are unaffected.

Two consequences follow. Reporting results in the paper's variables by projecting samples
does **not** fix this — projection changes the coordinates of samples, not the measure
they were drawn under. And in $\nu\Lambda$CDM the dropped weight lands on $\Sigma m_\nu$,
so `load_desi_prior_spec` **refuses** the raw measure for the `nulcdm` and `real_data`
phases; those assemblies must pass the phase argument and use the `_b1s8` spec variant.

### Fiducial-centered prior means

In forecast runs, the marginalized-nuisance prior *means* default to the per-bin fiducial
values rather than the paper's Table-I means. The widths, the correlated counterterm
block, and the layer-2 rescaling are all unchanged; only the means move. The reason is
consistency: a noiseless mock should peak at the truth, and a prior mean displaced from
the fiducial introduces a gradient that pulls it away. With fiducial-centered means that
mechanism is removed by construction, and what remains at the fiducial is only the residual
volume terms. The paper's means remain available as the `spec` variant for paper-fidelity
studies. This policy scopes to marginalized nuisances only — the sampled-block priors are
governed by the sections above.

## External probes

Three external data sets can be composed onto the full-shape posterior. All of them are
**fiducial-centered** in forecast mode, for the same reason the prior means are: the PFS
data vector is a noiseless mock, so a consistent joint forecast requires every likelihood
term to peak at the fiducial. This preserves the $\chi^2(\rm fid) = 0$ tripwire and keeps
the MCMC and Fisher comparisons exact.

### BAO

{doc}`api/bao` implements the standard BAO distance observables — $D_M/r_s$, $D_H/r_s$,
and $D_V/r_s$ — as differentiable JAX functions, and loads DESI measurements in the
`CobayaSampler/bao_data` format. `make_bao_theory_fn` returns a closure with the same
shape as the full-shape theory factories, so BAO enters the marginal posterior as one more
$-\tfrac12 r^{\mathsf T} C^{-1} r$ term. It carries no $\theta_{\rm lin}$ dependence, so it
sits outside the per-bin marginalization. On the Fisher side,
`add_bao_to_fullshape_fisher` adds the BAO block into the cosmology corner of the
full-shape Fisher matrix.

### CMB

The CMB enters the joint MCMC forecasts as a **Gaussian likelihood term** in the sampled
cosmology basis,

$$
\ln L_{\rm CMB} = -\tfrac12\, \Delta\theta^{\mathsf T} F_{\rm CMB}\, \Delta\theta ,
$$

where $F_{\rm CMB}$ is a Fisher matrix built once from the full `candl` stack — Planck
high-$\ell$ TTTEEE, low-$\ell$ TT, low-$\ell$ EE, and Planck/ACT lensing, with the CMB
nuisance parameters Schur-marginalized — evaluated at the fiducial cosmology and cached as
an `.npz` artifact. Notebooks load the artifact; they never take `candl` as a runtime
dependency.

Using real Planck data directly was considered and rejected for these forecasts: the
posterior would centre on Planck's best fit rather than on the fiducial, which breaks the
peaks-at-fiducial doctrine that the rest of the forecast depends on. That combination is
worth doing, but as a separate "PFS + real Planck" analysis.

$F_{\rm CMB}$ is a **hybrid Gauss–Newton** object, not an observed Hessian. The reason is
concrete: the $\nu\Lambda$CDM observed Hessian at our fiducial is indefinite — one negative
eigenvalue along the $H_0$–$m_\nu$ geometric degeneracy, sourced overwhelmingly by the
plik TTTEEE term. That negative curvature is residual real-data structure at a point that
is not Planck's best fit, and it is not a forecast statement. So the Gaussian-bandpower
terms use $J^{\mathsf T} C^{-1} J$, while the two genuinely non-Gaussian low-$\ell$ terms
(commander TT, simall EE) keep their observed Hessians, which are net positive along the
degeneracy direction. Clipping the negative eigenvalue was the rejected alternative: a
clipped eigenvalue is a repair, not a forecast, and it leaves real-data contamination in
the directions it retains.

One consequence is worth flagging. An internal prior (`A_planck`, the overall Planck
absolute calibration, $1.0000 \pm 0.0025$) that the `candl` implementation shares across
four Planck likelihood terms gets counted once per term in any naive
sum-of-terms Fisher matrix. The cached block subtracts that duplicate curvature after
summation, so every notebook that loads the artifact inherits the correction — the joint
MCMC notebooks and the `fisher_joint_PFS_BAO_CMB_*` Fisher notebooks alike. Two older
notebooks, `example/fisher/fisher_cmb_candl_LCDM.ipynb` and
`example/fisher/fisher_cmb_candl_nuLCDM.ipynb`, build the CMB Fisher matrix inline from
the likelihoods instead, and therefore still carry the overcount. They are kept as
**superseded reference material** — a readable record of how the block was originally
constructed — and carry banners saying not to cite their numbers and not to re-execute
them.

Because $\tau$ is constrained by the CMB curvature and by nothing else in the combination,
it becomes a genuinely sampled dimension with no separate prior, and its marginal is
Gaussian by construction. All non-Gaussianity in the joint posterior is therefore
full-shape-side: volume effects, and the $\Sigma m_\nu \ge 0$ wall.

### BBN

A Gaussian prior on $\omega_b$ with the Mossa et al. (2020) width, $0.00036$ — but centred
on the fiducial $0.02242$, not on the measured value. A real-data centre would introduce a
$0.25\sigma$ spurious pull and break the forecast doctrine. BBN is nearly redundant with
primary TT/TE/EE information on $\omega_b$ (it tightens $\sigma(\omega_b)$ by about 5–6%
once the CMB block is in) and is retained as a deliberate consistency anchor rather than
for constraining power.

### The $\Sigma m_\nu$ wall

In $\nu\Lambda$CDM the prior on $\Sigma m_\nu$ is flat with the physical bound
$\Sigma m_\nu \ge 0$, implemented as a $-\infty$ indicator. This makes the marginal a
*truncated* shape by construction, not a detection, and any width quoted for it must be
read that way. The bound is load-bearing rather than cosmetic: an unbounded diagnostic run
did not merely widen the posterior, it collapsed into a spurious mode at
$m_\nu \approx -0.33$ eV, outside the emulator's and the surrogate's domain of validity.

## Fisher and MCMC as two views of the same partition

The Fisher side and the MCMC side use the same model, the same prior spec, and the same
sampled/marginalized partition — deliberately.

On the Fisher side, {doc}`api/inference` builds $F = J^{\mathsf T} C^{-1} J + F_{\rm prior}$
and then takes a Schur complement to marginalize the nuisance block. On the MCMC side, the
analytic Gaussian marginalization integrates the same block out in closed form. These are
the same operation: the Schur complement *is* Gaussian marginalization. That equivalence is
what makes "does the MCMC reproduce the Fisher forecast?" a well-posed question rather than
an apples-to-oranges comparison — and, in the other direction, what makes a disagreement
diagnostic. When widths agree but means do not, the culprit is one of the volume terms
described above, not a bug.

Both sides finish by projecting to derived parameters
$(\Omega_m, \sigma_8, H_0\,[, \Sigma m_\nu])$ through the same map
({doc}`api/derived`), so the reported constraints are directly comparable.

Credible regions in the corner plots enclose 68% and 95% of the **joint** 2-D
posterior mass ($\Delta\chi^2 = 2.279$ and $5.991$ for two parameters, i.e.
$1.510\,\sigma$ and $2.448\,\sigma$). The 1-D intervals quoted on the diagonal enclose
the same masses in one dimension ($0.994\,\sigma$ and $1.960\,\sigma$ — approximately
$1\sigma$ and $2\sigma$). The two differ because probability mass in the tails of a
2-D Gaussian grows with the area of the annulus, so a fixed $\sigma$ multiple captures
less mass in two dimensions than in one. Overlaying a chain's 2-D HDI contour
(`chain_analysis.plot_credible_contours`) on a Fisher ellipse drawn with the wrong
convention overstates the joint region's width by up to $\sim 1.5\times$ at the 68%
level and $\sim 1.2\times$ at 95% — see {doc}`usage` for the `level_kind="mass2d"`
calling convention that keeps the two sides consistent.
