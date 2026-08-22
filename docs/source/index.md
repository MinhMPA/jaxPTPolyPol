# jaxPTPolyPol

Fisher forecasts and Bayesian inference for the **galaxy power spectrum and bispectrum
multipoles** in perturbation theory, written in JAX — end-to-end differentiable, from the
linear-$P(k)$ emulator through the EFT model to the posterior.

The package targets PFS-like spectroscopic surveys and combines them with DESI BAO,
Planck CMB, and BBN information in a single joint likelihood.

:::{important}
This is a **forecasting** framework, not a measurement pipeline. Every likelihood term is
centered on a noiseless fiducial mock rather than on observed data: the data vector is the
theory evaluated at the fiducial parameters, and the external BAO and CMB blocks are
Gaussian blocks expanded about that same point. Consequently the best fit sits at the
fiducial by construction, and $\chi^2(\theta_{\rm fid})$ is zero to the float64 floor —
which is why that identity is used as a wiring tripwire (see {doc}`testing`). The numbers
this package produces are *forecast uncertainties*, never measured central values.
:::

## What it does

- **Theory** — 1-loop galaxy $P_\ell(k)$ and tree-level $B_0(k_1,k_2,k_3)$ multipoles with
  EFT counterterms, stochasticity, IR resummation, and Alcock–Paczyński distortion.
- **Covariance** — Gaussian $P$, $B$, and joint $P{+}B$ covariances on the survey grid.
- **Fisher forecasts** — packed multi-bin parameter vectors, Schur-complement
  marginalization over nuisances, and projection to derived parameters
  $(\Omega_m, \sigma_8, H_0[, \Sigma m_\nu])$.
- **MCMC** — full posteriors over the same model via BlackJAX NUTS or random-walk
  Metropolis, made tractable by two devices: **analytic Gaussian marginalization** of the
  77 EFT nuisances that enter linearly, and a **Taylor surrogate** of the marginal
  likelihood that turns a multi-day chain into minutes.
- **External probes** — DESI DR2 BAO, a fiducial-centered Gaussian CMB Fisher block built
  from the Planck/ACT `candl` likelihoods, and a BBN prior on $\omega_b$.

## The two-layer split

This is the **inference layer**. It depends on a companion **model layer**, installed
separately ({doc}`installation` walks through both):

| Repo | Package | Role |
|---|---|---|
| this one | `jaxptpolypol` | parameters, covariance, Fisher, priors, marginal likelihood, samplers, plotting |
| [`ps_1loop_jax-for-pfs`](https://github.com/MinhMPA) | `ps_1loop_jax` | 1-loop $P_\ell$, tree $B_0$, flat $\nu\Lambda$CDM background |

The pipeline runs left to right: the emulator predicts $P_{\rm lin}(k)$, `ps_1loop_jax`
turns it into $P_\ell(k)$ multipoles, and `jaxptpolypol` handles covariances, Fisher
matrices, priors, and posteriors.

## A first forecast

The smallest end-to-end pipeline — one redshift bin, power-spectrum multipoles, AP on —
is a theory closure, its Jacobian, a Gaussian covariance, and the contraction of the two.
`...` marks values you supply; {doc}`usage` gives this walkthrough in full.

```python
import jax
jax.config.update("jax_enable_x64", True)      # required: every entry point assumes float64

import jax.numpy as jnp
from jaxptpolypol.covariance import gaussian_covariance
from jaxptpolypol.inference import fisher_matrix, sigma_from_fisher
from jaxptpolypol.model import CosmoEmulator, PS1LoopModel
from jaxptpolypol.params import CosmoParams, FullShapeSurveyParams, pack_pk_params
from jaxptpolypol.theory import compute_fiducial_distances, make_pk_ell_fn

cosmo = CosmoParams({
    "ombh2": 0.02242, "omch2": 0.11933, "logA": 3.047, "ns": 0.9665, "h": 0.6766,
    "z": 1.1, "A_b": 3.13, "eta_b": 0.603, "logT_AGN": 7.8,
})
survey = FullShapeSurveyParams(shared=..., pk=...)   # bias, counterterms, stochasticity
V_survey = 1.09 * 1000.0**3                          # (Mpc/h)^3

# Fiducial distances are computed once, OUTSIDE jit, and become closure constants.
Hz_fid, DAz_fid = compute_fiducial_distances(cosmo, (1.1,))

# All static configuration is captured in the closure, so jax.jit needs no static_argnames.
pk_fn = make_pk_ell_fn(
    ells=(0, 2, 4),
    pklin_emulator=CosmoEmulator(probe="custom_log", emulator_path=...),
    ps1loop_model=PS1LoopModel(do_irres=True),
    cosmo_keys=cosmo.param_keys,
    cosmo_sizes=cosmo.param_sizes,
    survey_keys=survey.pk_param_keys,
    ap=True, Hz_fid=Hz_fid, DAz_fid=DAz_fid,
)
jitted_pk_fn = jax.jit(pk_fn)

packed = pack_pk_params(cosmo, [survey])
k = jnp.linspace(5e-3, 0.25, 50)

pk_ell = jitted_pk_fn(packed, k=k).reshape(3, k.shape[0])
jac = jax.jacfwd(jitted_pk_fn, argnums=0)(packed, k=k)    # differentiable end to end
cov = gaussian_covariance(V_survey, k, float(k[1] - k[0]), *pk_ell)

print(sigma_from_fisher(fisher_matrix(cov, jac)))         # 1-sigma forecast per parameter
```

{doc}`usage` develops this into four complete workflows, up to a joint PFS + BAO + CMB +
BBN posterior; {doc}`theory` explains what the objects mean and where the approximations
are.

```{toctree}
:maxdepth: 2
:caption: Contents

installation
theory
usage
testing
api/index
```
