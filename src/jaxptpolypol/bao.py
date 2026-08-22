"""
BAO likelihood and Fisher forecast utilities.

Implements BAO distance observables (DM/rs, DH/rs, DV/rs) as differentiable
JAX functions, data loading from cobaya/bao_data format, and a Fisher
forecast pipeline for BAO-only or joint BAO + full-shape analyses.

The central function :func:`make_bao_theory_fn` returns a closure

    ``bao_fn(params) -> jnp.ndarray``

that is ready to be JIT-compiled and differentiated, following the same
pattern as :func:`theory.make_pk_ell_fn`.

Data loading
------------
Use :func:`load_bao_data` to read cobaya-format measurement + covariance
files (e.g. DESI DR2 from ``CobayaSampler/bao_data``).

Fisher forecast
---------------
Use :func:`bao_fisher_matrix` for a BAO-only Fisher matrix, or add the
result to a full-shape Fisher matrix for a joint analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .params import CosmoParams
from .inference import fisher_matrix, gaussian_prior_fisher

__all__ = [
    "BAODataPoint",
    "BAOData",
    "load_bao_data",
    "load_desi_2024",
    "load_desi_dr2",
    "make_bao_theory_fn",
    "bao_fisher_matrix",
    "add_bao_to_fullshape_fisher",
    "DESI_2024_TRACERS",
    "DESI_DR2_TRACERS",
]

# ---------------------------------------------------------------------------
# Observable registry
# ---------------------------------------------------------------------------
# Maps observable name (as it appears in data files) to the corresponding
# function in ps_1loop_jax.background.  Each function has signature
#   fn(omega_b, omega_cdm, h, z, mnu) -> scalar
_OBSERVABLE_FUNCTIONS: dict[str, str] = {
    "DM_over_rs": "dm_over_rs",
    "DH_over_rs": "dh_over_rs",
    "DV_over_rs": "dv_over_rs",
}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
class BAODataPoint(NamedTuple):
    """A single BAO measurement."""
    z: float
    value: float
    observable: str  # "DM_over_rs", "DH_over_rs", or "DV_over_rs"


class BAOData(NamedTuple):
    """Loaded BAO dataset with measurements and covariance."""
    data_points: tuple[BAODataPoint, ...]
    data_vector: jnp.ndarray   # shape (n_data,)
    cov: jnp.ndarray           # shape (n_data, n_data)
    rs_fid: float              # fiducial sound horizon used in data


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_bao_data(
    measurements_file: str | Path,
    cov_file: str | Path,
    rs_fid: float = 1.0,
) -> BAOData:
    r"""Load BAO measurements and covariance from cobaya-format text files.

    Parameters
    ----------
    measurements_file : path-like
        Path to the measurements file.  Expected format::

            # [z] [value at z] [quantity]
            0.295  7.9417  DV_over_rs
            0.510 13.5876  DM_over_rs
            ...

        Lines starting with ``#`` are ignored.
    cov_file : path-like
        Path to the covariance matrix file (space-separated, NxN).
    rs_fid : float
        Fiducial sound horizon [Mpc] with which the data values are stored.
        If ``rs_fid = 1`` (cobaya convention for DESI DR2), the data values
        are in absolute Mpc and the theory must also produce absolute
        distances divided by the *true* :math:`r_s`.

    Returns
    -------
    BAOData
        Named tuple with ``data_points``, ``data_vector``, ``cov``, and
        ``rs_fid``.

    Notes
    -----
    The data ordering in the returned arrays matches the row ordering of
    the measurements file, which must be consistent with the covariance
    matrix.
    """
    measurements_file = Path(measurements_file)
    cov_file = Path(cov_file)

    # --- Read measurements ---
    data_points: list[BAODataPoint] = []
    with open(measurements_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            z = float(parts[0])
            value = float(parts[1])
            obs = parts[2]
            if obs not in _OBSERVABLE_FUNCTIONS:
                raise ValueError(
                    f"Unknown observable '{obs}' in {measurements_file}. "
                    f"Supported: {list(_OBSERVABLE_FUNCTIONS.keys())}"
                )
            data_points.append(BAODataPoint(z=z, value=value, observable=obs))

    # --- Read covariance ---
    cov_np = np.atleast_2d(np.loadtxt(cov_file))
    n = len(data_points)
    if cov_np.shape != (n, n):
        raise ValueError(
            f"Covariance shape {cov_np.shape} doesn't match "
            f"{n} data points from {measurements_file}"
        )

    data_vector = jnp.array([dp.value for dp in data_points])
    cov = jnp.array(cov_np)

    return BAOData(
        data_points=tuple(data_points),
        data_vector=data_vector,
        cov=cov,
        rs_fid=rs_fid,
    )


# ---------------------------------------------------------------------------
# Theory prediction factory
# ---------------------------------------------------------------------------
def make_bao_theory_fn(
    bao_data: BAOData,
    *,
    cosmo_keys: tuple[str, ...],
    cosmo_sizes: tuple[int, ...],
    mnu_fixed: float = 0.06,
    rd_fit: str = "aubourg2014",
):
    r"""Build a differentiable BAO theory-prediction function.

    Returns a closure ``bao_fn(params) -> jnp.ndarray`` that computes
    the BAO observables (DM/rs, DH/rs, DV/rs) at each redshift bin,
    ordered to match the data vector in *bao_data*.

    Parameters
    ----------
    bao_data : BAOData
        Loaded BAO dataset (defines which observables at which redshifts).
    cosmo_keys : tuple of str
        Cosmological parameter names, e.g.
        ``('h', 'omega_b', 'omega_cdm', 'mnu', 'n_s', 'logA')``.
    cosmo_sizes : tuple of int
        Size of each cosmological parameter.
    mnu_fixed : float
        Neutrino mass [eV] used when ``'mnu'`` is not among the varied
        cosmological parameters.  Default 0.06.
    rd_fit : str
        Sound-horizon fitting function backend passed to
        ``bg.dm_over_rs`` / ``bg.dh_over_rs`` / ``bg.dv_over_rs``.
        Default ``"aubourg2014"``.

    Returns
    -------
    bao_fn : callable
        ``bao_fn(params) -> jnp.ndarray`` of shape ``(n_data,)``.
        The input *params* is a 1-d array of cosmological parameters
        (no survey parameters needed for BAO).
    """
    from ps_1loop_jax import background as bg

    has_mnu = "mnu" in cosmo_keys
    n_cosmo = sum(cosmo_sizes)

    # Pre-extract the observable info as static data
    n_data = len(bao_data.data_points)
    redshifts = tuple(dp.z for dp in bao_data.data_points)
    obs_names = tuple(dp.observable for dp in bao_data.data_points)
    rs_fid = bao_data.rs_fid

    # Pre-compute static index maps for the vectorised closure.
    # unique_z: sorted unique redshifts; z_index[i] maps data point i
    # to its position in unique_z.  needs_dm[i] / needs_dh[i] flags
    # which observable each data point requires.
    unique_z = sorted(set(redshifts))
    z_index = tuple(unique_z.index(z) for z in redshifts)
    needs_dm = tuple(obs in ("DM_over_rs", "DV_over_rs") for obs in obs_names)
    needs_dh = tuple(obs in ("DH_over_rs", "DV_over_rs") for obs in obs_names)
    is_dv    = tuple(obs == "DV_over_rs" for obs in obs_names)
    n_unique = len(unique_z)

    # Static redshift array for the single chi() call
    _z_max = max(unique_z)

    def bao_fn(params):
        cosmo_obj = CosmoParams.from_array(
            params[:n_cosmo], cosmo_keys, cosmo_sizes
        )
        h = cosmo_obj.h[0]
        omb = cosmo_obj.omega_b[0]
        omc = cosmo_obj.omega_cdm[0]
        mnu = cosmo_obj.mnu[0] if has_mnu else mnu_fixed

        # --- Compute rd once ---
        rd = bg.sound_horizon_drag(omb, omc, h, mnu, fit=rd_fit)

        # --- Compute chi(z) on a single shared grid [0, z_max] ---
        n_grid = 512
        z_grid = jnp.linspace(0.0, _z_max, n_grid)
        chi_grid = bg.chi(omb, omc, h, z_grid, mnu)  # shape (n_grid,)

        # Interpolate DM = chi(z) at each unique redshift
        z_unique_arr = jnp.array(unique_z)
        dm_unique = jnp.interp(z_unique_arr, z_grid, chi_grid)

        # DH = c / H(z) at each unique redshift
        dh_unique = bg.C_KMS / bg.Hz(omb, omc, h, z_unique_arr, mnu)

        # --- Assemble the data vector ---
        predictions = []
        for i in range(n_data):
            zi = z_index[i]
            if is_dv[i]:
                # DV = (z * DM^2 * DH)^{1/3}
                pred_i = jnp.cbrt(
                    unique_z[zi] * dm_unique[zi] ** 2 * dh_unique[zi]
                ) / rd
            elif needs_dm[i]:
                pred_i = dm_unique[zi] / rd
            else:
                pred_i = dh_unique[zi] / rd

            # Rescale if data uses a fiducial sound horizon != 1
            if rs_fid != 1.0:
                pred_i = pred_i * rd / rs_fid
            predictions.append(pred_i)

        return jnp.array(predictions)

    # Attach metadata
    bao_fn.n_data = n_data
    bao_fn.redshifts = redshifts
    bao_fn.obs_names = obs_names
    bao_fn.n_cosmo = n_cosmo
    bao_fn.has_mnu = has_mnu

    return bao_fn


# ---------------------------------------------------------------------------
# Fisher matrix for BAO
# ---------------------------------------------------------------------------
def bao_fisher_matrix(
    bao_data: BAOData,
    fiducial_params: jnp.ndarray,
    *,
    cosmo_keys: tuple[str, ...],
    cosmo_sizes: tuple[int, ...],
    mnu_fixed: float = 0.06,
    rd_fit: str = "aubourg2014",
    cosmo_priors: dict[str, float] | None = None,
) -> jnp.ndarray:
    r"""Compute the BAO Fisher matrix at fiducial cosmology.

    This is a convenience function that builds the BAO theory function,
    computes the Jacobian via ``jax.jacfwd``, and returns the Fisher
    matrix :math:`F = J^T C^{-1} J`.

    Parameters
    ----------
    bao_data : BAOData
        Loaded BAO dataset.
    fiducial_params : array, shape ``(n_cosmo,)``
        Fiducial cosmological parameter values (packed array).
    cosmo_keys : tuple of str
        Cosmological parameter names.
    cosmo_sizes : tuple of int
        Size of each cosmological parameter.
    mnu_fixed : float
        Neutrino mass [eV] when ``'mnu'`` is not varied.
    cosmo_priors : dict, optional
        ``{param_name: sigma}`` for Gaussian priors on cosmological
        parameters.

    Returns
    -------
    F_bao : array, shape ``(n_cosmo, n_cosmo)``
        BAO Fisher matrix.  Can be added to a full-shape Fisher matrix
        (after ensuring consistent parameter ordering) for a joint
        analysis.

    Examples
    --------
    >>> bao = load_bao_data("desi_mean.txt", "desi_cov.txt", rs_fid=1.0)
    >>> F = bao_fisher_matrix(bao, fid_params, cosmo_keys=keys,
    ...                       cosmo_sizes=sizes)
    """
    bao_fn = make_bao_theory_fn(
        bao_data,
        cosmo_keys=cosmo_keys,
        cosmo_sizes=cosmo_sizes,
        mnu_fixed=mnu_fixed,
        rd_fit=rd_fit,
    )

    # Jacobian: d(observables) / d(cosmo_params)
    jac = jax.jacfwd(jax.jit(bao_fn))(fiducial_params)

    # Fisher = J^T C^{-1} J
    F_bao = fisher_matrix(bao_data.cov, jac)

    # Add priors if specified
    if cosmo_priors is not None:
        from .inference import build_prior_sigmas

        n_cosmo = sum(cosmo_sizes)
        prior_sigmas = {}
        offset = 0
        for key, size in zip(cosmo_keys, cosmo_sizes):
            if key in cosmo_priors:
                for j in range(size):
                    prior_sigmas[offset + j] = cosmo_priors[key]
            offset += size

        F_bao = F_bao + gaussian_prior_fisher(n_cosmo, prior_sigmas)

    return F_bao


# ---------------------------------------------------------------------------
# Joint full-shape + BAO Fisher
# ---------------------------------------------------------------------------
def add_bao_to_fullshape_fisher(
    F_fullshape: jnp.ndarray,
    F_bao: jnp.ndarray,
    n_cosmo: int,
) -> jnp.ndarray:
    r"""Add a BAO Fisher matrix to a full-shape Fisher matrix.

    The full-shape Fisher matrix has shape ``(n_cosmo + n_survey, ...)``
    where the first ``n_cosmo`` rows/columns correspond to cosmological
    parameters.  The BAO Fisher matrix has shape ``(n_cosmo, n_cosmo)``
    and only constrains cosmological parameters.

    This function embeds the BAO Fisher in the larger parameter space
    and adds it to the full-shape Fisher.

    Parameters
    ----------
    F_fullshape : array, shape ``(n_total, n_total)``
        Full-shape Fisher matrix.
    F_bao : array, shape ``(n_cosmo, n_cosmo)``
        BAO-only Fisher matrix (cosmological parameters only).
    n_cosmo : int
        Number of cosmological parameters.

    Returns
    -------
    F_joint : array, shape ``(n_total, n_total)``
        Combined Fisher matrix.
    """
    n_total = F_fullshape.shape[0]
    F_bao_embedded = jnp.zeros((n_total, n_total))
    F_bao_embedded = F_bao_embedded.at[:n_cosmo, :n_cosmo].set(F_bao)
    return F_fullshape + F_bao_embedded


# ---------------------------------------------------------------------------
# Preset DESI DR2 data configurations
# ---------------------------------------------------------------------------
# These provide convenient access to the standard DESI DR2 BAO datasets
# as distributed in CobayaSampler/bao_data.

DESI_2024_TRACERS = {
    "all": {
        "mean": "desi_2024_gaussian_bao_ALL_GCcomb_mean.txt",
        "cov": "desi_2024_gaussian_bao_ALL_GCcomb_cov.txt",
        "rs_fid": 1.0,
    },
    "bgs": {
        "mean": "desi_2024_gaussian_bao_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4_mean.txt",
        "cov": "desi_2024_gaussian_bao_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4_cov.txt",
        "rs_fid": 1.0,
    },
    "lrg1": {
        "mean": "desi_2024_gaussian_bao_LRG_GCcomb_z0.4-0.6_mean.txt",
        "cov": "desi_2024_gaussian_bao_LRG_GCcomb_z0.4-0.6_cov.txt",
        "rs_fid": 1.0,
    },
    "lrg2": {
        "mean": "desi_2024_gaussian_bao_LRG_GCcomb_z0.6-0.8_mean.txt",
        "cov": "desi_2024_gaussian_bao_LRG_GCcomb_z0.6-0.8_cov.txt",
        "rs_fid": 1.0,
    },
    "lrg_elg": {
        "mean": "desi_2024_gaussian_bao_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1_mean.txt",
        "cov": "desi_2024_gaussian_bao_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1_cov.txt",
        "rs_fid": 1.0,
    },
    "elg": {
        "mean": "desi_2024_gaussian_bao_ELG_LOPnotqso_GCcomb_z1.1-1.6_mean.txt",
        "cov": "desi_2024_gaussian_bao_ELG_LOPnotqso_GCcomb_z1.1-1.6_cov.txt",
        "rs_fid": 1.0,
    },
    "qso": {
        "mean": "desi_2024_gaussian_bao_QSO_GCcomb_z0.8-2.1_mean.txt",
        "cov": "desi_2024_gaussian_bao_QSO_GCcomb_z0.8-2.1_cov.txt",
        "rs_fid": 1.0,
    },
    "lya": {
        "mean": "desi_2024_gaussian_bao_Lya_GCcomb_mean.txt",
        "cov": "desi_2024_gaussian_bao_Lya_GCcomb_cov.txt",
        "rs_fid": 1.0,
    },
}

DESI_DR2_TRACERS = {
    "all": {
        "mean": "desi_gaussian_bao_ALL_GCcomb_mean.txt",
        "cov": "desi_gaussian_bao_ALL_GCcomb_cov.txt",
        "rs_fid": 1.0,
    },
    "bgs": {
        "mean": "desi_gaussian_bao_BGS_BRIGHT-21.35_GCcomb_mean.txt",
        "cov": "desi_gaussian_bao_BGS_BRIGHT-21.35_GCcomb_cov.txt",
        "rs_fid": 1.0,
    },
    "lrg1": {
        "mean": "desi_gaussian_bao_LRG_GCcomb_z0.4-0.6_mean.txt",
        "cov": "desi_gaussian_bao_LRG_GCcomb_z0.4-0.6_cov.txt",
        "rs_fid": 1.0,
    },
    "lrg2": {
        "mean": "desi_gaussian_bao_LRG_GCcomb_z0.6-0.8_mean.txt",
        "cov": "desi_gaussian_bao_LRG_GCcomb_z0.6-0.8_cov.txt",
        "rs_fid": 1.0,
    },
    "lrg_elg": {
        "mean": "desi_gaussian_bao_LRG+ELG_LOPnotqso_GCcomb_mean.txt",
        "cov": "desi_gaussian_bao_LRG+ELG_LOPnotqso_GCcomb_cov.txt",
        "rs_fid": 1.0,
    },
    "elg": {
        "mean": "desi_gaussian_bao_ELG_LOPnotqso_GCcomb_z1.1-1.6_mean.txt",
        "cov": "desi_gaussian_bao_ELG_LOPnotqso_GCcomb_z1.1-1.6_cov.txt",
        "rs_fid": 1.0,
    },
    "qso": {
        "mean": "desi_gaussian_bao_QSO_GCcomb_mean.txt",
        "cov": "desi_gaussian_bao_QSO_GCcomb_cov.txt",
        "rs_fid": 1.0,
    },
    "lya": {
        "mean": "desi_gaussian_bao_Lya_GCcomb_mean.txt",
        "cov": "desi_gaussian_bao_Lya_GCcomb_cov.txt",
        "rs_fid": 1.0,
    },
}


def _load_desi(
    tracers_dict: dict,
    tracer: str,
    data_dir: str | Path,
) -> BAOData:
    """Internal helper for loading DESI data."""
    if tracer not in tracers_dict:
        raise ValueError(
            f"Unknown tracer '{tracer}'. "
            f"Available: {list(tracers_dict.keys())}"
        )
    info = tracers_dict[tracer]
    data_dir = Path(data_dir)
    return load_bao_data(
        data_dir / info["mean"],
        data_dir / info["cov"],
        rs_fid=info["rs_fid"],
    )


def load_desi_2024(
    tracer: str = "all",
    data_dir: str | Path = ".",
) -> BAOData:
    r"""Load a DESI 2024 (DR1) BAO dataset by tracer name.

    Parameters
    ----------
    tracer : str
        One of: ``'all'``, ``'bgs'``, ``'lrg1'``, ``'lrg2'``,
        ``'lrg_elg'``, ``'elg'``, ``'qso'``, ``'lya'``.
    data_dir : path-like
        Directory containing the DESI 2024 data files.

    Returns
    -------
    BAOData
    """
    return _load_desi(DESI_2024_TRACERS, tracer, data_dir)


def load_desi_dr2(
    tracer: str = "all",
    data_dir: str | Path = ".",
) -> BAOData:
    r"""Load a DESI DR2 BAO dataset by tracer name.

    Parameters
    ----------
    tracer : str
        One of: ``'all'``, ``'bgs'``, ``'lrg1'``, ``'lrg2'``,
        ``'lrg_elg'``, ``'elg'``, ``'qso'``, ``'lya'``.
    data_dir : path-like
        Directory containing the DESI DR2 data files (from
        ``CobayaSampler/bao_data/desi_bao_dr2/``).

    Returns
    -------
    BAOData
    """
    return _load_desi(DESI_DR2_TRACERS, tracer, data_dir)
