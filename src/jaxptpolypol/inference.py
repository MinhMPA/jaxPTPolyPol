"""
Inference utilities: Fisher matrix, log-likelihood, and marginalization.

These functions are designed to be usable with both Fisher forecasts and
MCMC/nested sampling workflows.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve, inv

__all__ = [
    "fisher_matrix",
    "gaussian_prior_fisher",
    "build_prior_sigmas",
    "log_likelihood_gaussian",
    "marginalize_fisher",
    "sigma_from_fisher",
]


# ---------------------------------------------------------------------------
# Fisher matrix
# ---------------------------------------------------------------------------
@jax.jit
def fisher_matrix(cov: jnp.ndarray, jac: jnp.ndarray) -> jnp.ndarray:
    r"""Fisher information matrix via Cholesky decomposition.

    .. math::
        F_{ij} = J^T C^{-1} J

    Parameters
    ----------
    cov : array, shape (n_data, n_data)
        Data covariance matrix (must be positive-definite).
    jac : array, shape (n_data, n_params)
        Jacobian :math:`\partial \mathbf{d} / \partial \boldsymbol{\theta}`
        evaluated at the fiducial parameters.

    Returns
    -------
    F : array, shape (n_params, n_params)
    """
    L, lower = cho_factor(cov, lower=True)
    return jac.T @ cho_solve((L, lower), jac)


# ---------------------------------------------------------------------------
# Gaussian priors
# ---------------------------------------------------------------------------
def gaussian_prior_fisher(
    n_params: int,
    prior_sigmas: dict[int, float],
) -> jnp.ndarray:
    r"""Build a diagonal Fisher matrix from Gaussian priors.

    For each parameter *i* with a Gaussian prior of width
    :math:`\sigma_i`, the prior Fisher contribution is
    :math:`F_{ii}^{\rm prior} = 1/\sigma_i^2`.

    Parameters
    ----------
    n_params : int
        Total number of parameters in the packed vector.
    prior_sigmas : dict[int, float]
        Mapping from parameter index to prior width :math:`\sigma`.
        Parameters not in the dict receive no prior (flat).

    Returns
    -------
    F_prior : array, shape (n_params, n_params)
        Diagonal matrix to be added to the data Fisher matrix.
    """
    diag = jnp.zeros(n_params)
    for idx, sigma in prior_sigmas.items():
        diag = diag.at[idx].set(1.0 / sigma**2)
    return jnp.diag(diag)


def build_prior_sigmas(
    cosmo_keys: tuple[str, ...],
    cosmo_sizes: tuple[int, ...],
    survey_keys: tuple,
    n_bins: int = 1,
    cosmo_priors: dict[str, float] | None = None,
    survey_priors: (
        dict[tuple[str, str | None], float]
        | list[dict[tuple[str, str | None], float]]
        | None
    ) = None,
) -> dict[int, float]:
    r"""Map named parameter priors to ``{index: sigma}`` for the packed vector.

    Parameters
    ----------
    cosmo_keys : tuple of str
        Cosmological parameter names (from ``CosmoParams.param_keys``).
    cosmo_sizes : tuple of int
        Size of each cosmological parameter (from ``CosmoParams.param_sizes``).
    survey_keys : tuple of (str, str | None)
        Flat survey parameter keys (from ``SurveyParams.param_keys``).
    n_bins : int
        Number of redshift bins.
    cosmo_priors : dict, optional
        ``{param_name: sigma}`` for cosmological parameters, e.g.
        ``{'ombh2': 0.001}``.  Default *None* (no cosmo priors).
    survey_priors : dict or list of dict, optional
        If a **dict**, the same priors are applied to every bin, e.g.
        ``{('bias', 'b1'): 0.5}``.

        If a **list** of length *n_bins*, each element is a dict of
        priors for that bin, allowing different priors per bin, e.g.
        ``[{('bias', 'b1'): 0.3}, {('bias', 'b1'): 0.5}, ...]``.
        Use an empty dict ``{}`` for bins with no survey priors.

        Default *None* (no survey priors).

    Returns
    -------
    prior_sigmas : dict[int, float]
        Ready to pass to :func:`gaussian_prior_fisher`.
    """
    result: dict[int, float] = {}

    # --- Cosmo priors ---
    if cosmo_priors is not None:
        offset = 0
        for key, size in zip(cosmo_keys, cosmo_sizes):
            if key in cosmo_priors:
                for j in range(size):
                    result[offset + j] = cosmo_priors[key]
            offset += size

    # --- Survey priors (per bin) ---
    if survey_priors is not None:
        n_cosmo = sum(cosmo_sizes)
        n_survey = len(survey_keys)

        # Normalize to a list of per-bin dicts
        if isinstance(survey_priors, dict):
            survey_priors_list = [survey_priors] * n_bins
        else:
            if len(survey_priors) != n_bins:
                raise ValueError(
                    f"survey_priors list length ({len(survey_priors)}) "
                    f"!= n_bins ({n_bins})"
                )
            survey_priors_list = survey_priors

        for b in range(n_bins):
            bin_offset = n_cosmo + b * n_survey
            priors_b = survey_priors_list[b]
            for s, skey in enumerate(survey_keys):
                if skey in priors_b:
                    result[bin_offset + s] = priors_b[skey]

    return result


# ---------------------------------------------------------------------------
# Gaussian log-likelihood (for MCMC)
# ---------------------------------------------------------------------------
def log_likelihood_gaussian(
    data: jnp.ndarray,
    theory: jnp.ndarray,
    cov: jnp.ndarray,
) -> jnp.ndarray:
    r"""Gaussian log-likelihood (up to a constant).

    .. math::
        \ln \mathcal{L} = -\frac{1}{2}\,
        (\mathbf{d} - \boldsymbol{\mu})^T\,
        C^{-1}\,
        (\mathbf{d} - \boldsymbol{\mu})

    Parameters
    ----------
    data : array, shape (n_data,)
        Observed data vector.
    theory : array, shape (n_data,)
        Theory prediction at the current parameter values.
    cov : array, shape (n_data, n_data)
        Data covariance matrix.

    Returns
    -------
    logL : scalar
    """
    residual = data - theory
    L, lower = cho_factor(cov, lower=True)
    return -0.5 * residual @ cho_solve((L, lower), residual)


def log_likelihood_gaussian_precomp(
    data: jnp.ndarray,
    theory: jnp.ndarray,
    cov_inv: jnp.ndarray,
) -> jnp.ndarray:
    r"""Gaussian log-likelihood with pre-computed inverse covariance.

    Avoids repeated Cholesky decomposition when the covariance is fixed
    (e.g. in an MCMC chain with a fixed data covariance).

    Parameters
    ----------
    data : array, shape (n_data,)
    theory : array, shape (n_data,)
    cov_inv : array, shape (n_data, n_data)
        Precomputed :math:`C^{-1}`.

    Returns
    -------
    logL : scalar
    """
    residual = data - theory
    return -0.5 * residual @ cov_inv @ residual


# ---------------------------------------------------------------------------
# Marginalization helpers
# ---------------------------------------------------------------------------
def marginalize_fisher(
    fisher: jnp.ndarray,
    varied_idx: list[int] | jnp.ndarray,
) -> jnp.ndarray:
    """Extract the sub-block of a Fisher matrix for varied parameters.

    Fixed parameters are removed (equivalent to marginalizing over them
    by inversion of the sub-block).

    Parameters
    ----------
    fisher : array, shape (n_total, n_total)
    varied_idx : sequence of int
        Indices of the parameters to *keep*.

    Returns
    -------
    F_marg : array, shape (n_varied, n_varied)
    """
    idx = jnp.array(varied_idx)
    return fisher[jnp.ix_(idx, idx)]


def sigma_from_fisher(
    fisher: jnp.ndarray,
    param_idx: int | list[int] | None = None,
) -> jnp.ndarray:
    r"""Marginalized 1-:math:`\sigma` constraints from a Fisher matrix.

    Parameters
    ----------
    fisher : array, shape (n, n)
        Fisher matrix (already restricted to varied parameters).
    param_idx : int, list of int, or None
        If ``None``, return :math:`\sigma` for all parameters.
        Otherwise return only the requested indices.

    Returns
    -------
    sigma : array
    """
    cov = inv(fisher)
    sigma = jnp.sqrt(jnp.diag(cov))
    if param_idx is None:
        return sigma
    return sigma[jnp.array(param_idx)]


def fixed_and_varied_indices(
    n_cosmo: int,
    n_survey_per_bin: int,
    n_bins: int,
    fixed_cosmo_idx: list[int],
    fixed_survey_offsets: list[int],
) -> tuple[list[int], list[int]]:
    """Compute fixed and varied parameter indices for a multi-bin layout.

    Parameters
    ----------
    n_cosmo : int
        Number of shared cosmological parameters.
    n_survey_per_bin : int
        Number of survey parameters per redshift bin.
    n_bins : int
        Number of redshift bins.
    fixed_cosmo_idx : list of int
        Indices within the cosmo block to fix (e.g. ``[5, 6, 7, 8]``
        for z, A_b, eta_b, logT_AGN).
    fixed_survey_offsets : list of int
        Offsets within each survey block to fix (e.g. ``[11, 12]``
        for k_nl, ndens).

    Returns
    -------
    fixed_idx : list of int
    varied_idx : list of int
    """
    n_total = n_cosmo + n_bins * n_survey_per_bin
    fixed_idx = list(fixed_cosmo_idx)
    for b in range(n_bins):
        bin_offset = n_cosmo + b * n_survey_per_bin
        for s in fixed_survey_offsets:
            fixed_idx.append(bin_offset + s)
    varied_idx = [i for i in range(n_total) if i not in fixed_idx]
    return fixed_idx, varied_idx
