"""
jaxptpolypol
============

Modeling and analysis of power spectrum and bispectrum multipoles
in perturbation theory with JAX.

Submodules
----------
params      : Cosmological and survey parameter containers (JAX pytrees).
model       : Wrappers around emulators and theory codes.
theory      : Theory prediction factory (power spectrum multipoles with AP).
covariance  : Gaussian (and future non-Gaussian) covariance matrices.
inference   : Fisher matrix, log-likelihood, and related utilities.
bao         : BAO likelihood, data loading, and Fisher forecast utilities.
plotting    : Triangle plots and 1-d marginals.
sampler     : BlackJAX NUTS sampling with parameter whitening.
chain_analysis : Chain diagnostics, summaries, and plotting for packed samples.
"""

from . import params
from . import model
from . import theory
from . import covariance
from . import inference
from . import bao
from . import plotting
from . import sampler
from . import chain_analysis
from . import priors
from . import derived
from . import marginalization
from . import cmb
from . import cmb_mcmc_utils

from .marginal_likelihood import (
    LIN_SURVEY_KEYS,
    MarginalSplit,
    bin_lin_slices,
    gaussian_marginal_loglike,
    make_constant_prior_fns,
    make_marginal_log_posterior,
    make_marginal_log_posterior_perbin,
    make_marginal_log_posterior_scan,
    make_marginal_templates,
    split_marginal_indices,
)
from .marginal_taylor import (
    TaylorTemplates,
    build_taylor_templates,
    importance_reweight,
    load_taylor_templates,
    make_marginal_log_posterior_taylor,
    reweighted_moments,
    save_taylor_templates,
)
from .desi_priors import (
    DESI_F_FID,
    DesiPriorSpec,
    SpecValidationError,
    build_prior_sigmas_from_desi_spec,
    ctr_rotation_matrices,
    load_desi_prior_spec,
    make_desi_prior_fns,
    make_lcdm_rescaling_fns,
)
