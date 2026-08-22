# Usage

Four worked workflows, in increasing order of scope: a single-bin Fisher forecast, a
multi-bin $P{+}B$ Fisher forecast with nuisance marginalization, an MCMC on the Taylor
surrogate, and a joint fit that adds BAO, the cached CMB block, and BBN before projecting
to derived parameters.

Each walkthrough mirrors a committed notebook under `example/`, named at the top of the
section. The notebooks are the executable version of these recipes — they carry the full
plotting, diagnostics, and assertion cells that are elided here. The code below is trimmed
to the load-bearing calls; `...` marks values you supply.

Before anything else, in every session:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

See {doc}`installation` for setup, {doc}`theory` for what the objects mean, and the
{doc}`api/index` for exact signatures.

## 1. A single-bin Fisher forecast

**Notebook:** `example/fisher/fisher_prototype_AP.ipynb`

The smallest useful pipeline: one redshift bin, power-spectrum multipoles only, AP
distortion on. Build the parameters, build the theory closure, differentiate it, form the
Gaussian covariance, and contract the two into a Fisher matrix.

```python
import jax
import jax.numpy as jnp
import numpy as np

from jaxptpolypol.model import CosmoEmulator, PS1LoopModel
from jaxptpolypol.params import CosmoParams, FullShapeSurveyParams, pack_pk_params
from jaxptpolypol.theory import make_pk_ell_fn, compute_fiducial_distances
from jaxptpolypol.covariance import gaussian_covariance
from jaxptpolypol.inference import fisher_matrix, marginalize_fisher, sigma_from_fisher

pklin_emulator = CosmoEmulator(probe="custom_log", emulator_path="...")
ps1loop_model = PS1LoopModel(do_irres=True)

cosmo_dict = {
    "ombh2": 0.02242, "omch2": 0.11933, "logA": 3.047,
    "ns": 0.9665, "h": 0.6766, "z": 1.1,
    "A_b": 3.13, "eta_b": 0.603, "logT_AGN": 7.8,
}
cosmo = CosmoParams(cosmo_dict)

knl, ndens, V = 1.02, 9.75e-4, 1.09 * 1000.0**3
z = cosmo_dict["z"]

# Bias fitting functions, Eqs. (3.6)-(3.14) of arXiv:1907.06666.
b1 = 0.9 + 0.4 * z
b2 = -0.704 - 0.208 * z + 0.183 * z**2 - 0.00771 * z**3
bG2 = -(2.0 / 7.0) * (b1 - 1.0)
bGamma3 = (23.0 / 42.0) * (b1 - 1.0)

# Counterterm fiducials scale with the growth factor; D+(z) from ps_1loop_jax.
from ps_1loop_jax import background as bg
Dplus = float(bg.growth_factor(
    cosmo_dict["ombh2"], cosmo_dict["omch2"], cosmo_dict["h"], z, mnu=0.06))

survey = FullShapeSurveyParams(
    shared={
        "bias": {"b1": b1, "b2": b2, "bG2": bG2, "bGamma3": bGamma3},
        "stoch": {"P_shot": 1.0},
        "k_nl": knl,
        "ndens": ndens,
    },
    pk={
        "ctr": {
            "c0": 25.0 * Dplus**2,
            "c2": 25.0 * Dplus**2,
            "c4": Dplus**2,
            "cfog": knl**-4,
        },
        "stoch": {"a0": 0.0, "a2": 0.0},
    },
)
pk_survey_keys = survey.pk_param_keys
packed_params = pack_pk_params(cosmo, [survey])

# Fiducial distances are computed once, OUTSIDE jit, and become closure constants.
Hz_fid, DAz_fid = compute_fiducial_distances(cosmo, (cosmo_dict["z"],))

pk_fn = make_pk_ell_fn(
    ells=(0, 2, 4),
    pklin_emulator=pklin_emulator,
    ps1loop_model=ps1loop_model,
    cosmo_keys=cosmo.param_keys,
    cosmo_sizes=cosmo.param_sizes,
    survey_keys=pk_survey_keys,
    ap=True,
    Hz_fid=Hz_fid,
    DAz_fid=DAz_fid,
    n_gl=16,
)
jitted_pk_fn = jax.jit(pk_fn)

k = jnp.linspace(5e-3, 0.25, 50)
pk_ell = jitted_pk_fn(packed_params, k=k).reshape(3, k.shape[0])
jac = jax.jacfwd(jitted_pk_fn, argnums=0)(packed_params, k=k)

cov = gaussian_covariance(V, k, float(k[1] - k[0]), *pk_ell)
fisher = fisher_matrix(cov, jac)
```

Points worth noting:

- `make_pk_ell_fn` is a **factory**. Everything static is captured in the closure, so
  `pk_fn` takes only the packed parameter vector and `k`, and needs no `static_argnames`.
- The returned data vector is flat, of length `len(ells) * len(k)`; reshape it to
  `(len(ells), len(k))` when you want the individual multipoles.
- `jax.jacfwd` on the jitted closure gives an exact analytic Jacobian. The notebook
  cross-checks it against a `numdifftools` finite-difference Jacobian, which is worth
  repeating whenever you change the theory configuration.

To get constraints, drop the parameters you are holding fixed and marginalize over the
rest:

```python
fixed_cosmo_names = {"z", "A_b", "eta_b", "logT_AGN"}
fixed_survey_keys = {("shared", "k_nl", None), ("shared", "ndens", None)}

fixed_idx = [i for i, name in enumerate(cosmo.param_keys) if name in fixed_cosmo_names]
fixed_idx += [
    len(cosmo.param_keys) + i
    for i, key in enumerate(pk_survey_keys) if key in fixed_survey_keys
]
varied_idx = [i for i in range(fisher.shape[0]) if i not in fixed_idx]

F_marg = marginalize_fisher(fisher, varied_idx)
sigmas = sigma_from_fisher(F_marg)
```

`k_nl` and `ndens` are survey *configuration* carried inside the parameter pytree for
convenience, not parameters to constrain — always fix them. `marginalize_fisher` keeps the
listed indices and Schur-complements the rest away.

## 2. A multi-bin $P{+}B$ Fisher forecast

**Notebook:** `example/fisher/fisher_multibin_LCDM_AP_PB.ipynb`

Seven redshift bins, power spectrum and bispectrum jointly, DESI EFT priors, and a BAO
block added on top. The structure is the same as above; what changes is that the survey
parameters become a list (one `FullShapeSurveyParams` per bin) and the covariance and
theory factories take bin-wise tuples.

```python
from jaxptpolypol.model import BispectrumTreeModel
from jaxptpolypol.params import pack_joint_params
from jaxptpolypol.theory import (
    build_bispectrum_triangles_from_k_grid,
    make_joint_pk_bk_fn,
    make_gaussian_joint_covariance_fn,
)
from jaxptpolypol.inference import (
    fisher_matrix, gaussian_prior_fisher, build_prior_sigmas,
    fixed_and_varied_indices, marginalize_fisher,
)
from jaxptpolypol.bao import load_desi_dr2, bao_fisher_matrix, add_bao_to_fullshape_fisher
from jaxptpolypol.desi_priors import (
    load_desi_prior_spec, make_lcdm_rescaling_fns, build_prior_sigmas_from_desi_spec,
)

z_bins   = (0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.2)
V_bins   = tuple(v * 1000.0**3 for v in (0.59, 0.79, 0.96, 1.09, 1.19, 2.58, 2.71))
knl_bins = (0.52, 0.65, 0.82, 1.02, 1.29, 1.82, 2.88)
n_zbins  = len(z_bins)

bispectrum_model = BispectrumTreeModel(do_AP=True, k_nl_rsd=0.45)

surveys = [...]                      # one FullShapeSurveyParams per bin
joint_survey_keys = surveys[0].joint_param_keys
packed_params = pack_joint_params(cosmo, surveys)

Hz_fid, DAz_fid = compute_fiducial_distances(cosmo, z_bins)

k = jnp.linspace(0.02, 0.20, 37)
dk = float(k[1] - k[0])
triangles, triangle_dk = build_bispectrum_triangles_from_k_grid(
    k, k_min=0.02, k_max=0.08, dk=dk,
)

joint_theory_kwargs = dict(
    pklin_emulator=pklin_emulator,
    ps1loop_model=ps1loop_model,
    bispectrum_model=bispectrum_model,
    cosmo_keys=cosmo.param_keys,
    cosmo_sizes=cosmo.param_sizes,
    survey_keys=joint_survey_keys,
    ap=True,
    z_bins=z_bins,
    Hz_fid=Hz_fid,
    DAz_fid=DAz_fid,
    n_gl=16,
    num_mu=65,
    num_phi=65,
    background_mode="direct",
)
joint_fn = make_joint_pk_bk_fn(**joint_theory_kwargs)
jitted_joint_fn = jax.jit(joint_fn)

joint_cov_fn = make_gaussian_joint_covariance_fn(
    pklin_emulator=pklin_emulator,
    ps1loop_model=ps1loop_model,
    cosmo_keys=cosmo.param_keys,
    cosmo_sizes=cosmo.param_sizes,
    survey_keys=joint_survey_keys,
    ap=True,
    z_bins=z_bins,
    Hz_fid=Hz_fid,
    DAz_fid=DAz_fid,
    bb_power_model="kaiser",
    n_gl=16,
    background_mode="direct",
)

jac = jax.jacfwd(jitted_joint_fn, argnums=0)(packed_params, k=k, triangles=triangles)
cov = joint_cov_fn(
    packed_params, V_survey=V_bins, k=k, dk=dk,
    triangles=triangles, triangle_dk=triangle_dk,
)
```

Each bin contributes a block of length `3 * len(k) + len(triangles)`, and the blocks are
stacked in bin order — the layout every downstream per-bin routine assumes.
`joint_fn.layout` records it as a string on the unjitted closure.

### Adding priors and marginalizing

The EFT and stochastic parameters need priors ({doc}`theory` explains why). Load the DESI
specification and turn it into per-bin width dictionaries:

```python
desi_spec = load_desi_prior_spec()          # desi_dr1_reanalysis_2511_20757

s8_keys = ("ombh2", "omch2", "logA", "ns", "h")
_, _, sigma8_ref_bins = make_lcdm_rescaling_fns(
    pklin_emulator=pklin_emulator,
    cosmo_keys=s8_keys,
    cosmo_sizes=(1,) * len(s8_keys),
    z_bins=z_bins,
    fid_cosmo_native=jnp.array([cosmo_dict[key] for key in s8_keys]),
    mnu_fixed=0.06,
    fixed_cosmo_extras={
        "A_b": cosmo_dict["A_b"],
        "eta_b": cosmo_dict["eta_b"],
        "logT_AGN": cosmo_dict["logT_AGN"],
    },
)
survey_sigma_dicts, sampled_sigma_bins = build_prior_sigmas_from_desi_spec(
    desi_spec, knl_bins=knl_bins, sigma8_ref_bins=sigma8_ref_bins,
)

desi_survey_priors = []
for b in range(n_zbins):
    entry = dict(survey_sigma_dicts[b])
    entry[("shared", "bias", "b2")] = sampled_sigma_bins[b]["b2"]
    entry[("shared", "bias", "bG2")] = sampled_sigma_bins[b]["bG2"]
    desi_survey_priors.append(entry)

prior_sigmas = build_prior_sigmas(
    cosmo_keys=cosmo.param_keys,
    cosmo_sizes=cosmo.param_sizes,
    survey_keys=joint_survey_keys,
    n_bins=n_zbins,
    cosmo_priors={"ombh2": 0.00055, "ns": 0.042},   # BBN + n_s,10
    survey_priors=desi_survey_priors,
)

n_cosmo_params = sum(cosmo.param_sizes)
bao_dr2 = load_desi_dr2("all", data_dir="../../ext_data/bao_data/desi_bao_dr2")
F_bao = bao_fisher_matrix(
    bao_dr2, packed_params[:n_cosmo_params],
    cosmo_keys=cosmo.param_keys, cosmo_sizes=cosmo.param_sizes,
)

F = fisher_matrix(cov, jac)
F = add_bao_to_fullshape_fisher(F, F_bao, n_cosmo=n_cosmo_params)
F = F + gaussian_prior_fisher(packed_params.shape[0], prior_sigmas)

fixed_idx, varied_idx = fixed_and_varied_indices(
    n_cosmo_params,
    len(joint_survey_keys),
    n_zbins,
    [i for i, name in enumerate(cosmo.param_keys)
     if name in {"z", "A_b", "eta_b", "logT_AGN"}],
    [i for i, key in enumerate(joint_survey_keys)
     if key in {("shared", "k_nl", None), ("shared", "ndens", None)}],
)
F_marg = marginalize_fisher(F, varied_idx)
```

`build_prior_sigmas` maps named parameters to packed-vector indices;
`gaussian_prior_fisher` turns that mapping into a diagonal prior Fisher matrix, which is
simply added. `fixed_and_varied_indices` does the index bookkeeping for the multi-bin
layout so you do not have to.

Two cautions. The prior widths here are the fiducial (layer-2 factors equal to one) DESI
widths — correct for a Fisher matrix evaluated at the fiducial, but not the full
$\theta_{\rm NL}$-dependent prior the MCMC uses. And the counterterm trio $c_0, c_2, c_4$
contributes only its *marginal* widths on this path; the off-diagonal correlations of the
rotated prior live in the MCMC-side $\Sigma_p$.

## 3. MCMC on the Taylor surrogate

**Notebook:** `example/mcmc/mcmc_joint_PFS_BAO_BBN_ns_LCDM.ipynb`

Same model, full posterior. The 77 linear nuisances are marginalized analytically and the
remaining 26 are sampled. Two objects make this practical: the per-bin factorization of
the marginal likelihood, and the cached Taylor surrogate.

### Partition the parameter vector

```python
from jaxptpolypol import (
    split_marginal_indices, bin_lin_slices, make_marginal_log_posterior_perbin,
)
from jaxptpolypol.desi_priors import make_desi_prior_fns
from jaxptpolypol.sampler import make_cholesky_transform, make_full_params_fn

split = split_marginal_indices(
    n_cosmo_params=n_cosmo_params,
    survey_keys=joint_survey_keys,
    n_bins=n_zbins,
    fixed_cosmo=[5, 6, 7, 8],                       # z, A_b, eta_b, logT_AGN
    fixed_survey_keys={("shared", "k_nl", None), ("shared", "ndens", None)},
)
# split.n_nl == 26 sampled, split.n_lin == 77 marginalized (11 per bin)
fid_nl = packed_params[jnp.array(split.nl_idx)]
```

### Build the per-bin marginal posterior

The covariance is block diagonal across bins and each bin's theory depends only on its own
$\theta_{\rm lin}$, so the dense $77 \times 77$ marginalization factorises into a sum of
seven small ones — the same posterior value, an order of magnitude faster and at a third
of the peak memory. BAO carries no $\theta_{\rm lin}$ dependence and enters once as a
plain $-\tfrac12 r^{\mathsf T} C^{-1} r$ term through `extra_theory_fn`.

```python
from functools import partial
from jax.scipy.linalg import inv
from jaxptpolypol.theory import make_joint_pk_bk_bin_fn

block_len = 3 * len(k) + len(triangles)
bin_theory_fns = [
    partial(make_joint_pk_bk_bin_fn(bin_index=b, **joint_theory_kwargs),
            k=k, triangles=triangles)
    for b in range(n_zbins)
]
pb_fid = jitted_joint_fn(packed_params, k=k, triangles=triangles)
bin_slices  = [slice(b * block_len, (b + 1) * block_len) for b in range(n_zbins)]
bin_data    = [pb_fid[s] for s in bin_slices]
bin_cov_invs = [inv(cov[s, s]) for s in bin_slices]
bin_lin_idx = [split.lin_idx[s] for s in bin_lin_slices(split, n_zbins)]

prior_mean_fn, prior_sigma_fn, log_prior_nl_desi = make_desi_prior_fns(
    desi_spec,
    split=split,
    knl_bins=knl_bins,
    sigma8_bins_fn=sigma8_bins_fn,
    a_ap_bins_fn=a_ap_bins_fn,
    sigma8_ref_bins=sigma8_ref_bins,
    marginal_means="fiducial",                       # or "spec" for the Table-I means
    fiducial_lin_means=packed_params[jnp.array(split.lin_idx)],
)

to_whitened, to_physical = make_cholesky_transform(center=fid_nl, cov=cov_nl_prior)
full_params_fn = make_full_params_fn(packed_params, split.nl_idx)

log_post = make_marginal_log_posterior_perbin(
    bin_theory_fns=bin_theory_fns,
    bin_data=bin_data,
    bin_cov_invs=bin_cov_invs,
    bin_lin_idx=bin_lin_idx,
    extra_theory_fn=lambda p: bao_theory_fn(p[:n_cosmo_params]),
    extra_data=bao_fid,
    extra_cov_inv=inv(jnp.asarray(bao_dr2.cov)),
    prior_mean_fn=prior_mean_fn,
    prior_sigma_fn=prior_sigma_fn,
    log_prior_nl_fn=log_prior_nl,
    to_physical=to_physical,
    full_params_fn=full_params_fn,
    include_logdet=True,
)

x0 = jnp.zeros(split.n_nl)      # the whitened fiducial
print(float(log_post(x0)))      # record this: it is a wiring tripwire
```

Names carried over from earlier steps: `cov` and `packed_params` from walkthrough 2;
`bao_theory_fn` from `jaxptpolypol.bao.make_bao_theory_fn` and `bao_fid` its value at the
fiducial cosmology; `log_prior_nl` the sum of `log_prior_nl_desi` and the sampled-cosmology
Gaussian prior from `jaxptpolypol.sampler.make_gaussian_log_prior`; `cov_nl_prior` the
inverse of `F_nl_prior`, the prior-included Fisher matrix Schur-complemented down to
$\theta_{\rm NL}$.

`make_cholesky_transform` supplies the whitening. Use the *full* Cholesky factor of the
Fisher-derived covariance, not a diagonal rescaling: the cosmology block is strongly
correlated, and diagonal whitening leaves the target anisotropic enough that isotropic
random-walk proposals are rejected essentially always. Whitening is sampling geometry
only — an affine reparameterisation of the target — so it need not use the same prior
specification as the target itself.

`sigma8_bins_fn`, `a_ap_bins_fn`, and `sigma8_ref_bins` all come from
`make_lcdm_rescaling_fns` as shown in walkthrough 2; the first two are the runtime layer-2
rescalings.

### Swap in the surrogate and sample

```python
from jaxptpolypol import (
    build_taylor_templates, save_taylor_templates,
    load_taylor_templates, make_marginal_log_posterior_taylor,
)
from jaxptpolypol.sampler import run_nuts

# Built once (expensive), then cached. See example/mcmc/scripts/build_taylor_templates_lcdm.py.
tt = build_taylor_templates(
    bin_theory_fns=bin_theory_fns,
    bin_lin_idx=bin_lin_idx,
    full_params_fn=full_params_fn,
    theta0=fid_nl,
    order2_m0=True,
)
save_taylor_templates(tt, "cache/taylor_templates_lcdm.npz", meta=META)

tt = load_taylor_templates("cache/taylor_templates_lcdm.npz", expect_meta=META)
log_post_surr = make_marginal_log_posterior_taylor(
    tt,
    bin_data=bin_data,
    bin_cov_invs=bin_cov_invs,
    extra_theory_fn=lambda p: bao_theory_fn(p[:n_cosmo_params]),
    extra_data=bao_fid,
    extra_cov_inv=inv(jnp.asarray(bao_dr2.cov)),
    prior_mean_fn=prior_mean_fn,
    prior_sigma_fn=prior_sigma_fn,
    log_prior_nl_fn=log_prior_nl,
    to_physical=to_physical,
    full_params_fn=full_params_fn,
    include_logdet=True,
)

# The surrogate is exact at its expansion point: assert it before trusting the chain.
assert abs(float(log_post_surr(x0)) - float(log_post(x0))) < 1e-6

samples_w, diagnostics = run_nuts(
    jax.random.key(20260806), log_post_surr, x0,
    num_warmup=1000, num_samples=5000, num_chains=4, max_tree_depth=10,
)
```

`META` is the theory-configuration stamp, defined once in
`example/mcmc/scripts/stream_common.py` so the build script and every consuming notebook
share a single source of truth.

`run_nuts` returns post-warmup samples with shape `(chains, draws, n_nl)` in *whitened*
coordinates; map them back with `jaxptpolypol.sampler.samples_to_physical`. Pass
`expect_meta` to `load_taylor_templates` every time — it is the guard that refuses a cache
built against a different theory configuration.

For a smoke run, sample the exact per-bin `log_post` with `run_rwmh_python` instead: it is
the ground-truth path and needs no cached artifact, at the cost of seconds per step.

## 4. A joint fit: full shape + BAO + CMB + BBN

**Notebook:** `example/mcmc/mcmc_joint_PFS_BAO_CMB_BBN_LCDM.ipynb`
(and `..._nuLCDM.ipynb` for the $\Sigma m_\nu$ case)

The external probes are fiducial-centered Gaussian blocks composed onto the full-shape
posterior. The sampled vector is extended by $\tau$, which the CMB block alone constrains:
$\theta = [\theta_{\rm NL}\,(26) \mid \tau]$, 27 dimensions.

The CMB Fisher block is a cached artifact, loaded by `load_cmb_fisher_block` from
`example/mcmc/scripts/stream_common.py` — a production script, not part of the installed
package, so the notebooks put `scripts/` on `sys.path`. Nothing here imports `candl`.

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path("scripts").resolve()))
from stream_common import BBN_SIGMA_MOSSA, SHARED_KEYS_CMB_LCDM, load_cmb_fisher_block

from jaxptpolypol.joint_forecast import (
    embed_fisher, make_forecast_joint_log_post, make_gaussian_fisher_loglike,
)

CMB_BLOCK = load_cmb_fisher_block("lcdm")     # hard cosmology / basis / config-hash guards
assert CMB_BLOCK["shared_keys"] == SHARED_KEYS_CMB_LCDM   # ('ombh2',...,'h','tau')

N_NL, TAU_FID = split.n_nl, 0.0561
N_TOT = N_NL + 1
SHARED_IDX_MAP = [0, 1, 2, 3, 4, N_NL]        # shared-basis rows inside theta
assert max(SHARED_IDX_MAP) < N_TOT            # JAX clamps out-of-bounds gathers SILENTLY

theta0 = jnp.concatenate([fid_nl, jnp.array([TAU_FID])])

cmb_loglike = make_gaussian_fisher_loglike(
    CMB_BLOCK["F_shared"], CMB_BLOCK["fid_shared"], SHARED_IDX_MAP,
)
bbn_loglike = make_gaussian_fisher_loglike(
    jnp.array([[1.0 / BBN_SIGMA_MOSSA**2]]), jnp.array([0.02242]), [0],
)

log_post_joint = make_forecast_joint_log_post(
    lambda theta_nl: log_post_surr(to_whitened(theta_nl)),
    n_pfs=N_NL,
    extra_loglike_fns=(cmb_loglike, bbn_loglike),
)
```

Both external blocks are centred on the fiducial, so both contribute **exactly zero** at
`theta0` and `log_post_joint(theta0)` equals the full-shape value at `x0`. Assert that
identity: a failure means the index map or the artifact centre is wrong.

Note what is *removed* here. The sampled-cosmology priors `{"ombh2": 0.00055, "ns": 0.042}`
used in walkthrough 2 are dropped, because $\omega_b$ is now carried by an explicit BBN
likelihood term and $n_s$ by the CMB block. Keeping both would double-count. Because a
fiducial-centered Gaussian prior contributes zero at the fiducial, the log-posterior value
will not reveal the mistake — check the prior-entry *count* instead.

### Whitening the extended vector

$\tau$ carries no full-shape information at all, so without the CMB entry its row of the
whitening Fisher would be identically zero and the Cholesky would fail. Summing the three
sources in physical coordinates is what makes the geometry positive definite:

```python
F_white = embed_fisher(F_nl_prior, list(range(N_NL)), N_TOT)
F_white = F_white + embed_fisher(CMB_BLOCK["F_shared"], SHARED_IDX_MAP, N_TOT)
F_white = F_white + embed_fisher(jnp.array([[1.0 / BBN_SIGMA_MOSSA**2]]), [0], N_TOT)
assert float(jnp.min(jnp.linalg.eigvalsh(F_white))) > 0.0

to_whitened_j, to_physical_j = make_cholesky_transform(center=theta0, cov=inv(F_white))
samples_w, diagnostics = run_nuts(
    jax.random.key(20260806),
    lambda w: log_post_joint(to_physical_j(w)),
    jnp.zeros(N_TOT),
    num_warmup=1000, num_samples=5000, num_chains=4,
)
```

### Projecting to derived parameters

Finish by mapping both the chain and the comparison Fisher matrix into
$(\Omega_m, \sigma_8, H_0)$ through the *same* function, so the two are directly
comparable. Native parameters that are absent from the map — $\tau$, the bias block, the
whole linear block — are marginalized simply by not entering it, while their correlations
still widen the projected covariance through $F^{-1}$.

```python
from jaxptpolypol.derived import make_derived_projection_fn, format_derived_comparison_rows
from jaxptpolypol.inference import project_fisher_to_derived
from jaxptpolypol.cmb_mcmc_utils import chunked_map

derived_fn, DERIVED_NAMES = make_derived_projection_fn(
    cosmo.param_keys,
    cosmo.param_sizes,
    pklin_emulator=pklin_emulator,
    fiducial_native=jnp.asarray(packed_params[:n_cosmo_params]),
    source_indices=[0, 1, 2, 3, 4],          # positions in the comparison basis ...
    native_indices=list(cosmo_varied_global),  # ... -> slots of the native cosmo vector
    mnu_fixed=0.06,
    sigma8_redshift=0.0,
)
assert DERIVED_NAMES == ("Omega_m", "sigma8", "H0")

derived_flat = chunked_map(cosmo_flat, jax.jit(jax.vmap(derived_fn)), chunk_size=20_000)
F_derived, derived_fid, derived_jac, cov_derived = project_fisher_to_derived(
    F_cmp, jnp.asarray(fiducial_shared), derived_fn,
)
rows, mean, sigma, pulls = format_derived_comparison_rows(
    DERIVED_NAMES, derived_fid, derived_flat, np.sqrt(np.diag(cov_derived)),
)
print("\n".join(rows))
```

`cosmo_flat` is the flattened chain restricted to its cosmology columns, `F_cmp` the
inline comparison Fisher matrix built in the same shared basis, `fiducial_shared` its
expansion centre, and `cosmo_varied_global` the positions of the varied cosmological
parameters in the native packed vector.

`chunked_map` exists because the projection runs the emulator forward for every draw;
chunking keeps peak memory bounded on a chain of hundreds of thousands of samples.

Read the resulting table with {doc}`theory`'s methodology rule in hand: widths and
correlations are the quantitative gates, and mean-level agreement should be checked
against the mode, not the raw chain mean.
