"""Joint forecast log-posterior helpers.

Compose the PFS Taylor-surrogate marginal posterior with fiducial-centered
Gaussian external blocks (CMB Fisher block, BBN prior) on an extended sampled
vector theta = concat(theta_NL, [tau]).  The PFS posterior sees only
theta[:n_pfs]; external blocks address the full vector through index maps.
"""
from __future__ import annotations

from typing import Callable, Sequence

import jax.numpy as jnp

__all__ = [
    "make_gaussian_fisher_loglike",
    "make_forecast_joint_log_post",
    "embed_fisher",
]


def make_gaussian_fisher_loglike(fisher, center, index_map) -> Callable:
    """loglike(theta) = -1/2 (theta[index_map] - center)^T F (theta[index_map] - center).

    Fiducial-centered by construction: contributes exactly 0 at theta[index_map]==center,
    preserving the noiseless-forecast chi2(fid)=0 tripwire.
    """
    F = jnp.asarray(fisher, dtype=jnp.float64)
    c = jnp.asarray(center, dtype=jnp.float64)
    idx = jnp.asarray(index_map, dtype=int)
    k = idx.shape[0]
    if F.shape != (k, k):
        raise ValueError(f"fisher shape {F.shape} does not match index_map length {k}")
    if c.shape != (k,):
        raise ValueError(f"center shape {c.shape} does not match index_map length {k}")

    def loglike(theta):
        d = jnp.asarray(theta)[idx] - c
        return -0.5 * d @ F @ d

    return loglike


def make_forecast_joint_log_post(pfs_log_post: Callable, *, n_pfs: int,
                                 extra_loglike_fns: Sequence[Callable] = ()) -> Callable:
    """log_post(theta) = pfs_log_post(theta[:n_pfs]) + sum(fn(theta) for fn in extras)."""
    if n_pfs <= 0:
        raise ValueError(f"n_pfs must be positive, got {n_pfs}")
    fns = tuple(extra_loglike_fns)

    def log_post(theta):
        theta = jnp.asarray(theta)
        total = pfs_log_post(theta[:n_pfs])
        for fn in fns:
            total = total + fn(theta)
        return total

    return log_post


def embed_fisher(F_sub, index_map, n: int):
    """n x n zeros with F_sub added at ix_(index_map, index_map)."""
    idx = jnp.asarray(index_map, dtype=int)
    return jnp.zeros((n, n)).at[jnp.ix_(idx, idx)].add(jnp.asarray(F_sub, dtype=jnp.float64))
