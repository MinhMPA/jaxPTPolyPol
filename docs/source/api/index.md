# API reference

Generated reference for every module in the `jaxptpolypol` package, listed
alphabetically. Read together they trace the pipeline: {doc}`params` and {doc}`model`
feed the closures in {doc}`theory`, those feed {doc}`covariance` and {doc}`inference`,
and {doc}`marginal_likelihood`, {doc}`marginal_taylor`, {doc}`sampler`, and the
external-probe modules ({doc}`bao`, {doc}`cmb`, {doc}`joint_forecast`) build the
posterior on top of them.

```{toctree}
:maxdepth: 1

bao
chain_analysis
cmb
cmb_mcmc_utils
covariance
derived
desi_priors
inference
joint_forecast
marginal_likelihood
marginal_taylor
marginalization
model
params
plotting
priors
sampler
theory
```
