"""
Gaussian covariance blocks for power-spectrum and bispectrum observables.

Currently implements

- ``C_PP`` for the power-spectrum multipoles ``(P0, P2, P4)``
- ``C_BB`` for the bispectrum monopole ``B0``

and provides single-bin and multi-bin assembly helpers with a joint
``[P..., B...]`` layout that can later accommodate a non-zero ``C_PB`` block.

The power-spectrum block ``C_PP`` follows Eqs. (B.1)-(B.2) of
Chudaykin et al. (2019), while the bispectrum-monopole block ``C_BB``
implements Eq. (B.3) of the same appendix after expanding the total power as
``P_tot(k, mu) = Pt0(k) L0(mu) + P2(k) L2(mu) + P4(k) L4(mu)``
and carrying out the monopole angular average analytically in that basis.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag

__all__ = [
    "gaussian_covariance",
    "gaussian_bispectrum_covariance",
    "gaussian_joint_covariance",
    "gaussian_covariance_multibin",
    "gaussian_bispectrum_covariance_multibin",
    "gaussian_joint_covariance_multibin",
]


def _as_k_grid(k: jnp.ndarray) -> jnp.ndarray:
    """Validate and return a 1-d k grid."""
    k = jnp.asarray(k)
    if k.ndim != 1:
        raise ValueError(f"k must be 1-d, got shape {k.shape}")
    return k


def _as_pk_ell(pk_ell: jnp.ndarray, n_k: int) -> jnp.ndarray:
    """Validate the ``[P0, P2, P4]`` multipole array."""
    pk_ell = jnp.asarray(pk_ell)
    if pk_ell.shape != (3, n_k):
        raise ValueError(
            "pk_ell must have shape (3, n_k) ordered as [P0, P2, P4], "
            f"got {pk_ell.shape}"
        )
    return pk_ell


def _broadcast_pk_bin_widths(
    dk: float | jnp.ndarray,
    n_k: int,
    dtype,
) -> jnp.ndarray:
    """Broadcast power-spectrum bin widths to shape ``(n_k,)``."""
    dk_arr = jnp.asarray(dk, dtype=dtype)
    if dk_arr.ndim == 0:
        return jnp.full((n_k,), dk_arr, dtype=dtype)
    if dk_arr.shape == (n_k,):
        return dk_arr
    raise ValueError(
        "dk must be a scalar or an array of shape (n_k,), "
        f"got {dk_arr.shape}"
    )


def _broadcast_triangle_bin_widths(
    triangle_dk: float | jnp.ndarray,
    n_tri: int,
    dtype,
) -> jnp.ndarray:
    """Broadcast bispectrum triangle bin widths to shape ``(n_tri, 3)``."""
    dk_arr = jnp.asarray(triangle_dk, dtype=dtype)
    if dk_arr.ndim == 0:
        return jnp.full((n_tri, 3), dk_arr, dtype=dtype)
    if dk_arr.shape == (3,):
        return jnp.broadcast_to(dk_arr[None, :], (n_tri, 3))
    if dk_arr.shape == (n_tri,):
        return jnp.broadcast_to(dk_arr[:, None], (n_tri, 3))
    if dk_arr.shape == (n_tri, 3):
        return dk_arr
    raise ValueError(
        "triangle_dk must be a scalar, (3,), (n_tri,), or (n_tri, 3), "
        f"got {dk_arr.shape}"
    )


def _assemble_mode_diagonal(cov_blocks: jnp.ndarray) -> jnp.ndarray:
    """Assemble mode-diagonal observable blocks in observable-major order.

    Parameters
    ----------
    cov_blocks : array, shape (n_mode, n_obs_row, n_obs_col)
        Per-mode covariance blocks. The returned matrix is ordered as
        ``[obs_0(all modes), obs_1(all modes), ...]``.
    """
    n_mode, n_obs_row, n_obs_col = cov_blocks.shape
    eye = jnp.eye(n_mode, dtype=cov_blocks.dtype)
    return jnp.einsum("mij,mn->imjn", cov_blocks, eye).reshape(
        n_obs_row * n_mode,
        n_obs_col * n_mode,
    )


def _combine_joint_bin_blocks(
    pp_cov: jnp.ndarray | None,
    bb_cov: jnp.ndarray | None,
    pb_cov: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Combine ``C_PP``, ``C_BB``, and optionally ``C_PB`` for one bin."""
    if pp_cov is None and bb_cov is None:
        raise ValueError("at least one of pp_cov or bb_cov must be provided")

    if pb_cov is not None and (pp_cov is None or bb_cov is None):
        raise ValueError("pb_cov requires both pp_cov and bb_cov")

    if pp_cov is None:
        return bb_cov
    if bb_cov is None:
        return pp_cov
    if pb_cov is None:
        return block_diag(pp_cov, bb_cov)

    if pb_cov.shape != (pp_cov.shape[0], bb_cov.shape[0]):
        raise ValueError(
            "pb_cov must have shape "
            f"({pp_cov.shape[0]}, {bb_cov.shape[0]}), got {pb_cov.shape}"
        )

    top = jnp.concatenate([pp_cov, pb_cov], axis=1)
    bottom = jnp.concatenate([pb_cov.T, bb_cov], axis=1)
    return jnp.concatenate([top, bottom], axis=0)


def _select_bin_arg(arg, b: int, n_bins: int, name: str):
    """Return a shared argument or the b-th element of a per-bin sequence."""
    if isinstance(arg, (list, tuple)):
        if len(arg) != n_bins:
            raise ValueError(
                f"{name} sequence length ({len(arg)}) != n_bins ({n_bins})"
            )
        return arg[b]
    return arg


@jax.jit
def _gaussian_pk_covariance_blocks(
    V_survey: float,
    k: jnp.ndarray,
    dk: jnp.ndarray,
    pk_ell: jnp.ndarray,
) -> jnp.ndarray:
    """Return the per-k ``(3, 3)`` Gaussian covariance blocks."""
    P0, P2, P4 = pk_ell
    N_k = V_survey * (k**2) * dk / (2.0 * jnp.pi**2)

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

    return (2.0 / N_k[:, None, None]) * jnp.stack(
        [
            jnp.stack([C00, C02, C04], axis=-1),
            jnp.stack([C02, C22, C24], axis=-1),
            jnp.stack([C04, C24, C44], axis=-1),
        ],
        axis=-2,
    )


def _triangle_geometry_terms(triangles: jnp.ndarray):
    """Return the trigonometric combinations entering the bispectrum kernel."""
    k1, k2, k3 = triangles.T

    nu12 = jnp.clip((k3**2 - k1**2 - k2**2) / (2.0 * k1 * k2), -1.0, 1.0)
    nu13 = jnp.clip((k2**2 - k1**2 - k3**2) / (2.0 * k1 * k3), -1.0, 1.0)
    nu23 = jnp.clip((k1**2 - k2**2 - k3**2) / (2.0 * k2 * k3), -1.0, 1.0)

    sin12 = jnp.sqrt(jnp.maximum(1.0 - nu12**2, 0.0))
    sin13 = jnp.sqrt(jnp.maximum(1.0 - nu13**2, 0.0))

    c2_12 = 2.0 * nu12**2 - 1.0
    c2_13 = 2.0 * nu13**2 - 1.0
    c2_23 = 2.0 * nu23**2 - 1.0

    s2_12 = 2.0 * nu12 * sin12
    s2_13 = 2.0 * nu13 * sin13

    c4_12 = 2.0 * c2_12**2 - 1.0
    c4_13 = 2.0 * c2_13**2 - 1.0
    c4_23 = 2.0 * c2_23**2 - 1.0

    s4_12 = 2.0 * s2_12 * c2_12
    s4_13 = 2.0 * s2_13 * c2_13

    c2m = c2_12 * c2_13 + s2_12 * s2_13
    c4p2 = c4_12 * c2_13 - s4_12 * s2_13
    c2p4 = c2_12 * c4_13 - s2_12 * s4_13

    return (
        c2_12,
        c2_13,
        c2_23,
        c4_12,
        c4_13,
        c4_23,
        s2_12,
        s2_13,
        c2m,
        c4p2,
        c2p4,
    )


def _bispectrum_d_coefficients(triangles: jnp.ndarray) -> jnp.ndarray:
    """Return ``D_{l1 l2 l3}`` for ``l_i in {0, 2, 4}``.

    The returned tensor has shape ``(n_tri, 3, 3, 3)`` with multipole index
    ordering ``[0, 2, 4]`` along each axis.
    """
    (
        c2_12,
        c2_13,
        c2_23,
        c4_12,
        c4_13,
        c4_23,
        s2_12,
        s2_13,
        c2m,
        c4p2,
        c2p4,
    ) = _triangle_geometry_terms(triangles)

    zeros = jnp.zeros_like(c2_12)
    ones = jnp.ones_like(c2_12)

    D022 = (1.0 + 3.0 * c2_23) / 20.0
    D044 = (9.0 + 20.0 * c2_23 + 35.0 * c4_23) / 576.0
    D202 = (1.0 + 3.0 * c2_13) / 20.0
    D220 = (1.0 + 3.0 * c2_12) / 20.0
    D222 = (-1.0 + 3.0 * c2_12 + 3.0 * c2_13 + 3.0 * c2_23) / 140.0
    D224 = (
        6.0
        + 3.0 * c2_12
        + 10.0 * c2_13
        + 10.0 * c2_23
        + 35.0 * c2p4
    ) / 1120.0
    D242 = (
        6.0
        + 10.0 * c2_12
        + 3.0 * c2_13
        + 10.0 * c2_23
        + 35.0 * c4p2
    ) / 1120.0
    D244 = 5.0 * (
        -9.0
        + 27.0 * c2_12
        + 27.0 * c2_13
        - 8.0 * c2_23
        + 49.0 * c4_23
        + 21.0 * c4p2
        + 21.0 * c2p4
    ) / 22176.0
    D404 = (9.0 + 20.0 * c2_13 + 35.0 * c4_13) / 576.0
    D422 = (
        3.0
        + 5.0 * c2_13
        + c2_12 * (5.0 + 19.0 * c2_13)
        + 16.0 * s2_12 * s2_13
    ) / 560.0
    D424 = 5.0 * (
        -9.0
        + 27.0 * c2_12
        + 21.0 * c2m
        - 8.0 * c2_13
        + 49.0 * c4_13
        + 27.0 * c2_23
        + 21.0 * c2p4
    ) / 22176.0
    D440 = (9.0 + 20.0 * c2_12 + 35.0 * c4_12) / 576.0
    D442 = 5.0 * (
        -8.0 * c2_12
        + 49.0 * c4_12
        + 3.0 * (
            -3.0
            + 7.0 * c2m
            + 9.0 * c2_13
            + 9.0 * c2_23
            + 7.0 * c4p2
        )
    ) / 22176.0
    D444 = 3.0 * (
        81.0
        - 110.0 * c2_12
        + 245.0 * c4_12
        + 350.0 * c2m
        - 110.0 * c2_13
        + 245.0 * c4_13
        - 110.0 * c2_23
        + 245.0 * c4_23
        + 350.0 * c4p2
        + 350.0 * c2p4
    ) / 256256.0

    row_0_0 = jnp.stack([ones, zeros, zeros], axis=1)
    row_0_2 = jnp.stack([zeros, D022, zeros], axis=1)
    row_0_4 = jnp.stack([zeros, zeros, D044], axis=1)

    row_2_0 = jnp.stack([zeros, D202, zeros], axis=1)
    row_2_2 = jnp.stack([D220, D222, D224], axis=1)
    row_2_4 = jnp.stack([zeros, D242, D244], axis=1)

    row_4_0 = jnp.stack([zeros, zeros, D404], axis=1)
    row_4_2 = jnp.stack([zeros, D422, D424], axis=1)
    row_4_4 = jnp.stack([D440, D442, D444], axis=1)

    plane_0 = jnp.stack([row_0_0, row_0_2, row_0_4], axis=1)
    plane_2 = jnp.stack([row_2_0, row_2_2, row_2_4], axis=1)
    plane_4 = jnp.stack([row_4_0, row_4_2, row_4_4], axis=1)

    return jnp.stack([plane_0, plane_2, plane_4], axis=1)


def _interpolate_triangle_edges(
    k: jnp.ndarray,
    pk_ell: jnp.ndarray,
    triangles: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Interpolate ``[P0, P2, P4]`` to each triangle edge."""
    edge_vals = []
    for edge in range(3):
        edge_vals.append(
            jnp.stack(
                [
                    jnp.interp(triangles[:, edge], k, pk_ell[ell])
                    for ell in range(3)
                ],
                axis=1,
            )
    )
    return edge_vals[0], edge_vals[1], edge_vals[2]


def _gaussian_bispectrum_covariance_integral(
    k: jnp.ndarray,
    triangles: jnp.ndarray,
    pk_ell: jnp.ndarray,
) -> jnp.ndarray:
    """Return the monopole angular integral of Eq. (B.3) in multipole form."""
    edge1, edge2, edge3 = _interpolate_triangle_edges(k, pk_ell, triangles)
    d_coeff = _bispectrum_d_coefficients(triangles)
    return jnp.einsum("tabc,ta,tb,tc->t", d_coeff, edge1, edge2, edge3)


@jax.jit
def _gaussian_bispectrum_covariance_diag(
    V_survey: float,
    k: jnp.ndarray,
    triangles: jnp.ndarray,
    triangle_dk: jnp.ndarray,
    pk_ell: jnp.ndarray,
) -> jnp.ndarray:
    """Return the diagonal entries of ``C_BB`` for the bispectrum monopole."""
    cov_integral = _gaussian_bispectrum_covariance_integral(
        k,
        triangles,
        pk_ell,
    )

    k1, k2, k3 = triangles.T
    dk1, dk2, dk3 = triangle_dk.T

    same12 = jnp.isclose(k1, k2, rtol=1e-10, atol=1e-12)
    same13 = jnp.isclose(k1, k3, rtol=1e-10, atol=1e-12)
    same23 = jnp.isclose(k2, k3, rtol=1e-10, atol=1e-12)
    s123 = jnp.where(
        same12 & same13,
        6.0,
        jnp.where(same12 | same13 | same23, 2.0, 1.0),
    )

    prefactor = (8.0 * jnp.pi**4 / V_survey) * s123 / (
        dk1 * dk2 * dk3 * k1 * k2 * k3
    )
    return prefactor * cov_integral


def gaussian_covariance(
    V_survey: float,
    k: jnp.ndarray,
    dk: float | jnp.ndarray,
    P0: jnp.ndarray,
    P2: jnp.ndarray,
    P4: jnp.ndarray,
) -> jnp.ndarray:
    r"""Gaussian covariance for power-spectrum multipoles ``(P0, P2, P4)``.

    Follows Chudaykin et al. (2019), Eqs. (B.1)-(B.2).

    Parameters
    ----------
    V_survey : float
        Survey volume in ``[Mpc/h]^3``.
    k : array, shape ``(n_k,)``
        Fourier wavenumber grid in ``[h/Mpc]``.
    dk : float or array, shape ``(n_k,)``
        Power-spectrum bin width(s) in ``[h/Mpc]``.
    P0, P2, P4 : array, shape ``(n_k,)``
        Power-spectrum monopole, quadrupole, and hexadecapole. If shot noise
        is required, include it in ``P0`` before calling this function.

    Returns
    -------
    cov : array, shape ``(3 * n_k, 3 * n_k)``
        Covariance matrix ordered as
        ``[P0(k_0...k_{N-1}), P2(k_0...k_{N-1}), P4(k_0...k_{N-1})]``.
    """
    k = _as_k_grid(k)
    dtype = jnp.result_type(k, P0, P2, P4)
    k = jnp.asarray(k, dtype=dtype)
    pk_ell = _as_pk_ell(jnp.stack([P0, P2, P4], axis=0), k.shape[0]).astype(dtype)
    dk_arr = _broadcast_pk_bin_widths(dk, k.shape[0], dtype)
    cov_blocks = _gaussian_pk_covariance_blocks(V_survey, k, dk_arr, pk_ell)
    return _assemble_mode_diagonal(cov_blocks)


def gaussian_bispectrum_covariance(
    V_survey: float,
    k: jnp.ndarray,
    triangles: jnp.ndarray,
    triangle_dk: float | jnp.ndarray,
    pk_ell: jnp.ndarray,
) -> jnp.ndarray:
    r"""Gaussian covariance for the bispectrum monopole ``B0``.

    Implements the Gaussian disconnected bispectrum-monopole covariance of
    Chudaykin et al. (2019), Eq. (B.3), after rewriting the total power in the
    ``[Pt0, P2, P4]`` multipole basis and evaluating the monopole angular
    average analytically through ``D_{l1 l2 l3}`` coefficients.

    Parameters
    ----------
    V_survey : float
        Survey volume in ``[Mpc/h]^3``.
    k : array, shape ``(n_k,)``
        Power-spectrum k grid used to define ``pk_ell``.
    triangles : array, shape ``(n_tri, 3)``
        Triangle bin centers ``[k1, k2, k3]`` in ``[h/Mpc]``.
    triangle_dk : float or array
        Triangle bin widths. Accepted shapes are scalar, ``(3,)``,
        ``(n_tri,)``, or ``(n_tri, 3)``.
    pk_ell : array, shape ``(3, n_k)``
        Total power multipoles ordered as ``[Pt0, P2, P4]``. The first entry
        must be the total monopole, e.g. ``Pt0 = P0 + 1/nbar`` if shot noise
        should be included.

    Returns
    -------
    cov : array, shape ``(n_tri, n_tri)``
        Diagonal Gaussian covariance matrix in the supplied triangle order.
    """
    k = _as_k_grid(k)
    triangles = jnp.asarray(triangles)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(
            "triangles must have shape (n_tri, 3), "
            f"got {triangles.shape}"
        )

    dtype = jnp.result_type(k, triangles, pk_ell)
    k = jnp.asarray(k, dtype=dtype)
    triangles = jnp.asarray(triangles, dtype=dtype)
    pk_ell = _as_pk_ell(pk_ell, k.shape[0]).astype(dtype)
    triangle_dk_arr = _broadcast_triangle_bin_widths(
        triangle_dk,
        triangles.shape[0],
        dtype,
    )
    cov_diag = _gaussian_bispectrum_covariance_diag(
        V_survey,
        k,
        triangles,
        triangle_dk_arr,
        pk_ell,
    )
    return jnp.diag(cov_diag)


def gaussian_joint_covariance(
    V_survey: float,
    k: jnp.ndarray,
    dk: float | jnp.ndarray,
    pk_ell: jnp.ndarray,
    *,
    triangles: jnp.ndarray | None = None,
    triangle_dk: float | jnp.ndarray | None = None,
    bb_pk_ell: jnp.ndarray | None = None,
    pb_cov: jnp.ndarray | None = None,
) -> jnp.ndarray:
    r"""Joint Gaussian covariance for one redshift bin.

    The within-bin data-vector ordering is

    ``[P0(all k), P2(all k), P4(all k), B0(all triangles)]``.

    Parameters
    ----------
    V_survey : float
        Survey volume in ``[Mpc/h]^3``.
    k, dk, pk_ell
        Inputs for the power-spectrum block ``C_PP``.
    triangles : array, optional
        Triangle bin centers. If omitted, only ``C_PP`` is returned.
    triangle_dk : float or array, optional
        Triangle bin widths. Defaults to ``dk`` when ``triangles`` is provided.
    bb_pk_ell : array, optional
        Optional bispectrum-covariance power multipoles ordered as
        ``[Pt0, P2, P4]``. If omitted, ``pk_ell`` is reused for ``C_BB``.
        This hook exists so the joint builder can later accommodate distinct
        power-spectrum inputs for ``C_PP``, ``C_BB``, and future ``C_PB``.
    pb_cov : array, optional
        Placeholder for a future ``C_PB`` implementation. If supplied now, it
        is inserted between the power and bispectrum blocks after shape checks.
    """
    k = _as_k_grid(k)
    pk_ell = _as_pk_ell(pk_ell, k.shape[0])
    if bb_pk_ell is not None:
        bb_pk_ell = _as_pk_ell(bb_pk_ell, k.shape[0])

    pp_cov = gaussian_covariance(
        V_survey,
        k,
        dk,
        pk_ell[0],
        pk_ell[1],
        pk_ell[2],
    )

    bb_cov = None
    if triangles is not None:
        bb_cov = gaussian_bispectrum_covariance(
            V_survey,
            k,
            triangles,
            dk if triangle_dk is None else triangle_dk,
            pk_ell if bb_pk_ell is None else bb_pk_ell,
        )

    return _combine_joint_bin_blocks(pp_cov, bb_cov, pb_cov=pb_cov)


def gaussian_covariance_multibin(
    V_bins: tuple[float, ...] | list[float],
    k: jnp.ndarray,
    dk: float | jnp.ndarray,
    pk_all: jnp.ndarray,
) -> jnp.ndarray:
    r"""Block-diagonal ``C_PP`` across multiple redshift bins.

    Parameters
    ----------
    V_bins : sequence of float
        Survey volume per redshift bin in ``[Mpc/h]^3``.
    k : array, shape ``(n_k,)``
        Fourier wavenumber grid.
    dk : float or array, shape ``(n_k,)``
        Power-spectrum bin width(s).
    pk_all : array, shape ``(n_bins, 3, n_k)``
        Power multipoles ``[P0, P2, P4]`` for each bin.

    Returns
    -------
    cov : array, shape ``(n_bins * 3 * n_k, n_bins * 3 * n_k)``
        Block-diagonal covariance ordered by redshift bin, with each per-bin
        block ordered as
        ``[P0(k_0...k_{N-1}), P2(k_0...k_{N-1}), P4(k_0...k_{N-1})]``.
    """
    k = _as_k_grid(k)
    pk_all = jnp.asarray(pk_all)

    n_bins = len(V_bins)
    if pk_all.shape[:2] != (n_bins, 3):
        raise ValueError(
            "pk_all must have shape (n_bins, 3, n_k), "
            f"got {pk_all.shape}"
        )
    if pk_all.shape[2] != k.shape[0]:
        raise ValueError(
            f"pk_all.shape[2] ({pk_all.shape[2]}) != len(k) ({k.shape[0]})"
        )

    cov_blocks = [
        gaussian_covariance(
            V_bins[b],
            k,
            dk,
            pk_all[b, 0],
            pk_all[b, 1],
            pk_all[b, 2],
        )
        for b in range(n_bins)
    ]
    return block_diag(*cov_blocks)


def gaussian_bispectrum_covariance_multibin(
    V_bins: tuple[float, ...] | list[float],
    k: jnp.ndarray,
    triangles: jnp.ndarray | Sequence[jnp.ndarray],
    triangle_dk: float | jnp.ndarray | Sequence[float | jnp.ndarray],
    pk_all: jnp.ndarray,
) -> jnp.ndarray:
    r"""Block-diagonal ``C_BB`` across multiple redshift bins."""
    k = _as_k_grid(k)
    pk_all = jnp.asarray(pk_all)

    n_bins = len(V_bins)
    if pk_all.shape[:2] != (n_bins, 3):
        raise ValueError(
            "pk_all must have shape (n_bins, 3, n_k), "
            f"got {pk_all.shape}"
        )
    if pk_all.shape[2] != k.shape[0]:
        raise ValueError(
            f"pk_all.shape[2] ({pk_all.shape[2]}) != len(k) ({k.shape[0]})"
        )

    cov_blocks = []
    for b in range(n_bins):
        triangles_b = _select_bin_arg(triangles, b, n_bins, "triangles")
        triangle_dk_b = _select_bin_arg(
            triangle_dk,
            b,
            n_bins,
            "triangle_dk",
        )
        cov_blocks.append(
            gaussian_bispectrum_covariance(
                V_bins[b],
                k,
                triangles_b,
                triangle_dk_b,
                pk_all[b],
            )
        )
    return block_diag(*cov_blocks)


def gaussian_joint_covariance_multibin(
    V_bins: tuple[float, ...] | list[float],
    k: jnp.ndarray,
    dk: float | jnp.ndarray,
    pk_all: jnp.ndarray,
    *,
    triangles: jnp.ndarray | Sequence[jnp.ndarray] | None = None,
    triangle_dk: (
        float | jnp.ndarray | Sequence[float | jnp.ndarray] | None
    ) = None,
    bb_pk_all: jnp.ndarray | None = None,
    pb_cov_all: Sequence[jnp.ndarray | None] | None = None,
) -> jnp.ndarray:
    r"""Block-diagonal joint ``[C_PP, C_BB]`` covariance across redshift bins.

    Each redshift bin contributes one block ordered as

    ``[P0(all k), P2(all k), P4(all k), B0(all triangles)]``.

    The optional ``pb_cov_all`` hook is included so a future ``C_PB``
    implementation can reuse the same bin-wise assembly.
    """
    k = _as_k_grid(k)
    pk_all = jnp.asarray(pk_all)
    if bb_pk_all is not None:
        bb_pk_all = jnp.asarray(bb_pk_all)

    n_bins = len(V_bins)
    if pk_all.shape[:2] != (n_bins, 3):
        raise ValueError(
            "pk_all must have shape (n_bins, 3, n_k), "
            f"got {pk_all.shape}"
        )
    if pk_all.shape[2] != k.shape[0]:
        raise ValueError(
            f"pk_all.shape[2] ({pk_all.shape[2]}) != len(k) ({k.shape[0]})"
        )
    if bb_pk_all is not None:
        if bb_pk_all.shape[:2] != (n_bins, 3):
            raise ValueError(
                "bb_pk_all must have shape (n_bins, 3, n_k), "
                f"got {bb_pk_all.shape}"
            )
        if bb_pk_all.shape[2] != k.shape[0]:
            raise ValueError(
                f"bb_pk_all.shape[2] ({bb_pk_all.shape[2]}) != len(k) "
                f"({k.shape[0]})"
            )
    if pb_cov_all is not None and len(pb_cov_all) != n_bins:
        raise ValueError(
            f"pb_cov_all length ({len(pb_cov_all)}) != n_bins ({n_bins})"
        )

    cov_blocks = []
    for b in range(n_bins):
        triangles_b = _select_bin_arg(triangles, b, n_bins, "triangles")
        triangle_dk_b = _select_bin_arg(
            triangle_dk,
            b,
            n_bins,
            "triangle_dk",
        )
        pb_cov_b = None if pb_cov_all is None else pb_cov_all[b]
        bb_pk_b = None if bb_pk_all is None else bb_pk_all[b]
        cov_blocks.append(
            gaussian_joint_covariance(
                V_bins[b],
                k,
                dk,
                pk_all[b],
                triangles=triangles_b,
                triangle_dk=triangle_dk_b,
                bb_pk_ell=bb_pk_b,
                pb_cov=pb_cov_b,
            )
        )
    return block_diag(*cov_blocks)
