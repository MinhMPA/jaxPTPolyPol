# jaxPTPolyPol

[![Documentation Status](https://readthedocs.org/projects/jaxptpolypol/badge/?version=latest)](https://jaxptpolypol.readthedocs.io/en/latest/)

Fisher forecasts and Bayesian inference for the **galaxy power spectrum and bispectrum
multipoles** in perturbation theory, written in JAX — end-to-end differentiable, from the
linear-$P(k)$ emulator through the EFT model to the posterior.

The package targets PFS-like spectroscopic surveys and combines them with DESI BAO,
Planck CMB, and BBN information in a single joint likelihood.

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

## Repository layout

This is the **inference layer**. It depends on a companion **model layer**:

| Repo | Package | Role |
|---|---|---|
| this one | `jaxptpolypol` | parameters, covariance, Fisher, priors, marginal likelihood, samplers, plotting |
| [`ps_1loop_jax-for-pfs`](https://github.com/MinhMPA) | `ps_1loop_jax` | 1-loop $P_\ell$, tree $B_0$, flat $\nu\Lambda$CDM background |

### Modules (`src/jaxptpolypol/`)

| Module | Responsibility |
|---|---|
| `params` | `CosmoParams` / `SurveyParams` JAX pytrees; packed single- and multi-bin vectors |
| `model` | wrappers around the linear-$P(k)$ emulator and `ps_1loop_jax` |
| `theory` | the factory closures — `make_pk_ell_fn`, `make_bk0_fn`, `make_joint_pk_bk_fn`, triangle construction, AP distances |
| `covariance` | Gaussian $P$/$B$/joint covariances, incl. the bispectrum $D_{\ell_1\ell_2\ell_3}$ geometry |
| `inference` | `fisher_matrix`, Schur marginalization, Gaussian prior Fishers, derived-parameter projection |
| `marginal_likelihood`, `marginalization` | analytic Gaussian marginalization over the linear nuisance block |
| `marginal_taylor` | Taylor-surrogate marginal posterior (templates + whitening) |
| `sampler` | NUTS and RWMH drivers, whitening transforms, chain utilities |
| `desi_priors`, `priors` | the DESI DR1-reanalysis EFT prior spec and its runtime rescalings |
| `bao`, `cmb`, `cmb_mcmc_utils` | DESI BAO observables; `candl`/`clipy` CMB likelihood wrappers |
| `joint_forecast` | Gaussian external blocks (CMB, BBN) composed onto the PFS posterior |
| `derived` | $(\Omega_m, H_0, \sigma_8)$ maps for derived-space projection |
| `chain_analysis`, `plotting` | convergence diagnostics, corner and contour plots |

### Examples (`example/`)

`fisher/` holds Fisher forecasts (single-bin prototypes → multi-bin $P{+}B$ → joint
PFS+BAO+CMB, for $\Lambda$CDM and $\nu\Lambda$CDM); `mcmc/` holds the production posterior
notebooks and the scripts that build their cached artifacts; `*_benchmark/` holds
performance comparisons. Notebooks are committed **with their outputs**, which serve as
the reference results and are used for regression checking.

## Installation

```bash
# model layer (companion repo)
git clone <ps_1loop_jax-for-pfs>  &&  cd ps_1loop_jax-for-pfs  &&  pip install -e .

# inference layer (this repo)
cd jaxPTPolyPol  &&  pip install -e ".[full]"
```

The `full` extra pulls `cosmopower-jax`, `ps_1loop_jax`, `quadax`, `matplotlib`,
`numdifftools`, and the CMB stack (`candl-like`, `clipy-like`). The Planck `.clik` data
and `candl_data` are **not** installable from PyPI — see the comments in `pyproject.toml`.
They are needed only to *rebuild* the cached CMB Fisher block; the notebooks that consume
it do not import `candl`.

Always enable 64-bit precision:

```python
import jax; jax.config.update("jax_enable_x64", True)
```

## Quick start

```python
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from jaxptpolypol.theory import make_pk_ell_fn
from jaxptpolypol.inference import fisher_matrix

# All static configuration (emulators, fiducial cosmology, AP, bins) is captured in the
# closure, so the returned function is directly jit/grad-able in the parameters.
pk_fn = make_pk_ell_fn(ells=(0, 2, 4), pklin_emulator=..., ps1loop_model=..., ap=True, ...)
jac = jax.jacfwd(jax.jit(pk_fn))(packed_params, k=k)
F = fisher_matrix(cov, jac)
```

## Tests

```bash
pytest tests/ -q          # 255 passed, 15 deselected
```

The companion repo carries its own suite (`pytest tests/ -v` in `ps_1loop_jax-for-pfs`).
The example notebooks act as integration tests: their committed outputs are the expected
results, and several carry hard tripwires (exact `log_post` values, `chi2(fid) < 1e-10`)
that fail loudly if the pipeline drifts.

## Further reading

- **`docs/source/theory.md`** (the Theory page of the documentation) — what the linear
  vs sampled parameter blocks are, how the analytic marginalization is defined, the c1
  counterterm treatment, and the CMB/prior methodology choices with their rationale.
- **`docs/design/perbin-compile-measurements.md`** — the running measurement record:
  benchmarks, validation gates, forecast results, and the evidence behind each decision.
- **`docs/design/desi-convention-map.md`** — the parameter-convention map between this
  code and the DESI DR1-reanalysis EFT prior specification.

