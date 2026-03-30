"""
Inference utilities: Fisher matrix, log-likelihood, and marginalization.

These functions are designed to be usable with both Fisher forecasts and
MCMC/nested sampling workflows.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cho_factor, cho_solve, inv

__all__ = [
    "fisher_diagnostics",
    "fisher_matrix",
    "format_fisher_diagnostics",
    "gaussian_prior_fisher",
    "build_prior_sigmas",
    "log_likelihood_gaussian",
    "marginalized_fisher_block",
    "marginalize_fisher",
    "project_fisher_to_derived",
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


def fisher_diagnostics(
    fisher: jnp.ndarray,
    param_names: Sequence[str] | None = None,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-14,
    max_modes: int = 5,
    max_params_per_mode: int = 5,
) -> dict:
    """Diagnose singular or ill-conditioned Fisher directions.

    The report is based on the eigendecomposition of the symmetrized Fisher
    matrix. Small or negative eigenvalues identify unconstrained or
    numerically unstable linear combinations of parameters.

    Parameters
    ----------
    fisher : array, shape (n, n)
        Fisher matrix for the parameter block of interest.
    param_names : sequence of str, optional
        Human-readable parameter names matching the Fisher ordering.
        Defaults to ``("param[0]", ..., "param[n-1]")``.
    rtol, atol : float
        Relative and absolute tolerances used to classify weak modes.
        A mode is treated as weak when ``eigval <= max(atol, rtol * max_eig)``.
    max_modes : int
        Number of weakest eigenmodes to report.
    max_params_per_mode : int
        Number of dominant parameter loadings to keep per eigenmode.

    Returns
    -------
    diagnostics : dict
        Structured report with eigenvalues, weakest modes, null-space
        participation, pseudo-inverse marginal variances, and conditioning
        metadata.
    """
    fisher_np = np.asarray(fisher, dtype=float)
    if fisher_np.ndim != 2 or fisher_np.shape[0] != fisher_np.shape[1]:
        raise ValueError(
            f"fisher must be a square matrix, got shape {fisher_np.shape}"
        )
    if not np.all(np.isfinite(fisher_np)):
        raise ValueError("fisher contains NaN or Inf entries")

    n_params = int(fisher_np.shape[0])
    if param_names is None:
        names = tuple(f"param[{i}]" for i in range(n_params))
    else:
        names = tuple(str(name) for name in param_names)
        if len(names) != n_params:
            raise ValueError(
                f"param_names length ({len(names)}) must match "
                f"fisher size ({n_params})"
            )

    fisher_sym = 0.5 * (fisher_np + fisher_np.T)
    eigvals, eigvecs = np.linalg.eigh(fisher_sym)
    eigvals = np.asarray(eigvals, dtype=float)
    eigvecs = np.asarray(eigvecs, dtype=float)

    max_eig = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
    weak_tol = max(float(atol), float(rtol) * max(max_eig, 1.0))
    weak_mask = eigvals <= weak_tol
    n_weak = int(np.count_nonzero(weak_mask))
    rank = int(n_params - n_weak)

    positive_mask = eigvals > weak_tol
    if positive_mask.any() and n_weak == 0:
        positive = eigvals[positive_mask]
        condition_number = float(positive[-1] / positive[0])
    elif positive_mask.any():
        condition_number = float(np.inf)
    else:
        condition_number = float(np.inf)

    pseudo_cov = np.linalg.pinv(fisher_sym, rcond=rtol, hermitian=True)
    marginal_variances = np.asarray(np.diag(pseudo_cov), dtype=float)

    n_mode_report = min(int(max_modes), n_params)
    weakest_modes: list[dict] = []
    for mode_idx in range(n_mode_report):
        vec = eigvecs[:, mode_idx]
        order = np.argsort(np.abs(vec))[::-1]
        components = [
            {
                "index": int(param_idx),
                "name": names[param_idx],
                "loading": float(vec[param_idx]),
                "abs_loading": float(abs(vec[param_idx])),
            }
            for param_idx in order[: max_params_per_mode]
        ]
        weakest_modes.append(
            {
                "mode": int(mode_idx),
                "eigenvalue": float(eigvals[mode_idx]),
                "is_weak": bool(eigvals[mode_idx] <= weak_tol),
                "components": components,
            }
        )

    if n_weak > 0:
        participation = np.sum(eigvecs[:, weak_mask] ** 2, axis=1)
    elif n_params > 0:
        participation = eigvecs[:, 0] ** 2
    else:
        participation = np.array([], dtype=float)

    order = np.argsort(participation)[::-1]
    nullspace_participation = [
        {
            "index": int(param_idx),
            "name": names[param_idx],
            "participation": float(participation[param_idx]),
        }
        for param_idx in order
    ]

    variance_order = np.argsort(marginal_variances)[::-1]
    pseudo_variance_ranking = [
        {
            "index": int(param_idx),
            "name": names[param_idx],
            "variance": float(marginal_variances[param_idx]),
        }
        for param_idx in variance_order
    ]

    return {
        "n_params": n_params,
        "param_names": names,
        "symmetry_error": float(
            np.linalg.norm(fisher_np - fisher_np.T)
            / max(np.linalg.norm(fisher_np), 1e-30)
        ),
        "rank": rank,
        "n_weak_modes": n_weak,
        "weak_tolerance": float(weak_tol),
        "condition_number": condition_number,
        "eigenvalues": eigvals,
        "weakest_modes": weakest_modes,
        "nullspace_participation": nullspace_participation,
        "pseudo_covariance": pseudo_cov,
        "marginal_variances_pinv": marginal_variances,
        "pseudo_variance_ranking": pseudo_variance_ranking,
    }


def format_fisher_diagnostics(
    diagnostics: dict,
    *,
    max_modes: int = 3,
    max_params_per_mode: int = 4,
    participation_threshold: float = 0.05,
) -> str:
    """Format :func:`fisher_diagnostics` output for notebook inspection."""
    lines = [
        (
            "rank="
            f"{diagnostics['rank']}/{diagnostics['n_params']}, "
            f"weak_modes={diagnostics['n_weak_modes']}, "
            f"cond={diagnostics['condition_number']:.3e}, "
            f"sym_err={diagnostics['symmetry_error']:.3e}, "
            f"weak_tol={diagnostics['weak_tolerance']:.3e}"
        )
    ]

    eigvals = np.asarray(diagnostics["eigenvalues"], dtype=float)
    if eigvals.size:
        n_show = min(int(max_modes), eigvals.size)
        eig_str = ", ".join(f"{eigvals[i]:.3e}" for i in range(n_show))
        lines.append(f"smallest eigenvalues: {eig_str}")

    if diagnostics["nullspace_participation"]:
        leading = [
            item
            for item in diagnostics["nullspace_participation"]
            if item["participation"] >= participation_threshold
        ]
        if not leading:
            leading = diagnostics["nullspace_participation"][:max_params_per_mode]
        joined = ", ".join(
            f"{item['name']} ({item['participation']:.2f})"
            for item in leading[:max_params_per_mode]
        )
        lines.append(f"null-space participation: {joined}")

    for mode in diagnostics["weakest_modes"][:max_modes]:
        comps = ", ".join(
            f"{comp['name']} ({comp['loading']:+.3f})"
            for comp in mode["components"][:max_params_per_mode]
        )
        flag = "weak" if mode["is_weak"] else "smallest"
        lines.append(
            f"mode {mode['mode']} [{flag}] eig={mode['eigenvalue']:.3e}: {comps}"
        )

    return "\n".join(lines)


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
        dict[tuple, float]
        | list[dict[tuple, float]]
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
    survey_keys : tuple
        Flat survey parameter keys for the packed survey block. Keys can be
        the legacy ``(group, key)`` form or role-aware
        ``(section, group, key)`` form.
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


def _covariance_from_fisher(fisher, *, rcond: float = 1e-12) -> np.ndarray:
    """Return a finite covariance matrix from a Fisher matrix."""
    fisher_np = np.asarray(fisher, dtype=float)
    if fisher_np.ndim != 2 or fisher_np.shape[0] != fisher_np.shape[1]:
        raise ValueError(
            f"fisher must be a square matrix, got shape {fisher_np.shape}"
        )
    if not np.all(np.isfinite(fisher_np)):
        raise ValueError("fisher contains NaN or Inf entries")

    fisher_np = 0.5 * (fisher_np + fisher_np.T)
    try:
        cov = np.linalg.inv(fisher_np)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(fisher_np, rcond=rcond, hermitian=True)
    cov = np.asarray(0.5 * (cov + cov.T), dtype=float)
    if not np.all(np.isfinite(cov)):
        raise ValueError("could not compute a finite covariance from fisher")
    return cov


def project_fisher_to_derived(
    fisher,
    fiducial_params,
    derived_fn,
    *,
    rcond: float = 1e-12,
):
    r"""Project a Fisher matrix from a native basis to a derived basis.

    Parameters
    ----------
    fisher : array_like, shape (n_native, n_native)
        Fisher matrix in the native parameter basis. This should already be the
        native block of interest, e.g. the marginalized cosmology block.
    fiducial_params : array_like, shape (n_native,)
        Fiducial native-parameter values matching ``fisher``.
    derived_fn : callable
        Differentiable map ``derived_fn(theta_native) -> theta_derived``.
    rcond : float, optional
        Relative cutoff used for pseudo-inverse fallbacks.

    Returns
    -------
    fisher_derived : ndarray, shape (n_derived, n_derived)
    fiducial_derived : ndarray, shape (n_derived,)
    jacobian : ndarray, shape (n_derived, n_native)
    covariance_derived : ndarray, shape (n_derived, n_derived)
    """
    native = jnp.ravel(jnp.asarray(fiducial_params, dtype=jnp.float64))
    fiducial_derived = np.asarray(jnp.ravel(derived_fn(native)), dtype=float)
    jacobian = np.asarray(jax.jacfwd(derived_fn)(native), dtype=float)
    if jacobian.ndim == 1:
        jacobian = jacobian[None, :]

    if jacobian.shape[1] != native.size:
        raise ValueError(
            f"derived Jacobian has incompatible shape {jacobian.shape} for "
            f"{native.size} native parameters"
        )

    covariance_native = _covariance_from_fisher(fisher, rcond=rcond)
    covariance_derived = jacobian @ covariance_native @ jacobian.T
    covariance_derived = np.asarray(
        0.5 * (covariance_derived + covariance_derived.T),
        dtype=float,
    )
    if not np.all(np.isfinite(covariance_derived)):
        raise ValueError("projected covariance contains NaN or Inf entries")

    try:
        fisher_derived = np.linalg.inv(covariance_derived)
    except np.linalg.LinAlgError:
        fisher_derived = np.linalg.pinv(
            covariance_derived,
            rcond=rcond,
            hermitian=True,
        )
    fisher_derived = np.asarray(
        0.5 * (fisher_derived + fisher_derived.T),
        dtype=float,
    )
    if not np.all(np.isfinite(fisher_derived)):
        raise ValueError("projected Fisher contains NaN or Inf entries")

    return fisher_derived, fiducial_derived, jacobian, covariance_derived


def marginalized_fisher_block(
    fisher,
    keep_idx,
    *,
    rcond: float = 1e-12,
):
    """Return the Fisher block after marginalizing over all other parameters.

    Parameters
    ----------
    fisher : array_like, shape (n, n)
        Fisher matrix in the full basis.
    keep_idx : sequence of int
        Parameter indices to retain after marginalizing over the complement.
    rcond : float, optional
        Relative cutoff used for pseudo-inverse fallbacks.
    """
    keep = np.asarray(keep_idx, dtype=int)
    covariance = _covariance_from_fisher(fisher, rcond=rcond)
    covariance_keep = covariance[np.ix_(keep, keep)]
    covariance_keep = np.asarray(
        0.5 * (covariance_keep + covariance_keep.T),
        dtype=float,
    )

    try:
        fisher_keep = np.linalg.inv(covariance_keep)
    except np.linalg.LinAlgError:
        fisher_keep = np.linalg.pinv(
            covariance_keep,
            rcond=rcond,
            hermitian=True,
        )
    fisher_keep = np.asarray(0.5 * (fisher_keep + fisher_keep.T), dtype=float)
    if not np.all(np.isfinite(fisher_keep)):
        raise ValueError("could not compute a finite marginalized Fisher block")
    return fisher_keep


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
