"""
Covariance matrices for power spectrum multipoles.

Currently implements the Gaussian covariance for (P0, P2, P4);
see Eq. (B.1)-(B.2) of `arXiv:1907.06666 <https://arxiv.org/abs/1907.06666>`_.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag

__all__ = [
    "gaussian_covariance",
    "gaussian_covariance_multibin",
]


def _build_block_cov(cov_blocks: jnp.ndarray) -> jnp.ndarray:
    """Assemble a block-diagonal matrix from per-k covariance blocks.

    Parameters
    ----------
    cov_blocks : array, shape (n_k, n_ell, n_ell)

    Returns
    -------
    cov : array, shape (n_k * n_ell, n_k * n_ell)
    """
    return block_diag(*[cov_blocks[i] for i in range(cov_blocks.shape[0])])


@jax.jit
def gaussian_covariance(
    V_survey: float,
    k: jnp.ndarray,
    dk: float,
    P0: jnp.ndarray,
    P2: jnp.ndarray,
    P4: jnp.ndarray,
) -> jnp.ndarray:
    r"""Gaussian covariance for power spectrum multipoles (P0, P2, P4).

    Parameters
    ----------
    V_survey : float
        Survey volume in [Mpc/h]^3.
    k : array, shape (n_k,)
        Fourier wavenumber grid in [h/Mpc].
    dk : float
        Bin width in [h/Mpc].
    P0, P2, P4 : array, shape (n_k,)
        Power spectrum monopole, quadrupole, and hexadecapole.

    Returns
    -------
    cov : array, shape (n_k * 3, n_k * 3)
        Block-diagonal covariance matrix ordered as [P0, P2, P4].
    """
    N_k = V_survey * (k**2) * dk / (4.0 * jnp.pi**2)

    C00 = P0**2 + (1.0 / 5.0) * P2**2 + (1.0 / 9.0) * P4**2
    C02 = (
        2.0 * P0 * P2
        + (2.0 / 7.0) * P2**2
        + (4.0 / 7.0) * P2 * P4
        + (100.0 / 693.0) * P4**2
    )
    C04 = (
        (18.0 / 35.0) * P2**2
        + 2.0 * P0 * P4
        + (40.0 / 77.0) * P2 * P4
        + (162.0 / 1001.0) * P4**2
    )
    C22 = (
        5.0 * P0**2
        + (20.0 / 7.0) * P0 * P2
        + (20.0 / 7.0) * P0 * P4
        + (15.0 / 7.0) * P2**2
        + (120.0 / 77.0) * P2 * P4
        + (8945.0 / 9009.0) * P4**2
    )
    C24 = (
        (36.0 / 7.0) * P0 * P2
        + (200.0 / 77.0) * P0 * P4
        + (108.0 / 77.0) * P2**2
        + (3578.0 / 1001.0) * P2 * P4
        + (900.0 / 1001.0) * P4**2
    )
    C44 = (
        9.0 * P0**2
        + (360.0 / 77.0) * P0 * P2
        + (2916.0 / 1001.0) * P0 * P4
        + (16101.0 / 5005.0) * P2**2
        + (3240.0 / 1001.0) * P2 * P4
        + (42849.0 / 17017.0) * P4**2
    )

    cov_blocks = (2.0 / N_k[:, None, None]) * jnp.stack(
        [
            jnp.stack([C00, C02, C04], axis=-1),
            jnp.stack([C02, C22, C24], axis=-1),
            jnp.stack([C04, C24, C44], axis=-1),
        ],
        axis=-2,
    )
    return _build_block_cov(cov_blocks)


def gaussian_covariance_multibin(
    V_bins: tuple[float, ...],
    k: jnp.ndarray,
    dk: float,
    pk_all: jnp.ndarray,
) -> jnp.ndarray:
    r"""Block-diagonal Gaussian covariance across multiple redshift bins.

    Parameters
    ----------
    V_bins : tuple of float
        Survey volume per bin in [Mpc/h]^3.
    k : array, shape (n_k,)
        Fourier wavenumber grid.
    dk : float
        Bin width.
    pk_all : array, shape (n_bins, 3, n_k)
        Power spectrum multipoles ``[P0, P2, P4]`` for each bin.

    Returns
    -------
    cov : array, shape (n_bins * 3 * n_k, n_bins * 3 * n_k)
    """
    cov_blocks = []
    for b in range(pk_all.shape[0]):
        P0, P2, P4 = pk_all[b]
        cov_blocks.append(gaussian_covariance(V_bins[b], k, dk, P0, P2, P4))
    return block_diag(*cov_blocks)
