"""
Analytic (Gaussian) marginalization of linear EFT/stochastic parameters.

Implements the marginal likelihood of arXiv:2511.20757 SS II.3 (which defers
to arXiv:2507.13433 and CLASS-PT, arXiv:2004.10607): for a theory vector
exactly linear in the nuisance block theta_lin,

    t(theta_NL, theta_lin) = m0(theta_NL) + M(theta_NL) @ theta_lin,

the Gaussian-prior integral over theta_lin is closed-form:

    -2 ln L = rt^T Cinv rt - b^T A^{-1} b + ln det(A Sigma_p),
    rt = d - m0 - M mu_p,   A = M^T Cinv M + Sigma_p^{-1},   b = M^T Cinv rt.

The c1 bispectrum counterterm is genuinely (but negligibly) quadratic in the
underlying ps_1loop_jax theory; the templates below interrogate the theory
only through its value and slope at theta_lin = 0, so the marginal model is
linear in c1 by construction -- exactly the 2511.20757 model. See CONTEXT.md.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = [
    "gaussian_marginal_loglike",
]


def gaussian_marginal_loglike(data, m0, M, cov_inv, mu_p, sigma_p,
                              *, include_logdet: bool = True):
    """Closed-form Gaussian marginal log-likelihood over the linear block.

    Parameters
    ----------
    data, m0 : (n_data,) — data vector and linear-model offset t(theta_lin=0).
    M : (n_data, n_lin) — template matrix dt/dtheta_lin.
    cov_inv : (n_data, n_data) — inverse data covariance.
    mu_p, sigma_p : (n_lin,) — Gaussian prior means and widths on theta_lin.
        Every entry of sigma_p must be finite and positive (proper prior).
    include_logdet : bool (static)
        If False, drop the ln det(A Sigma_p) term (the "Jeffreys prior"
        best-fit convention of arXiv:2511.20757 SS II.3).

    Returns
    -------
    scalar log-likelihood (constant offset −½ ln det C omitted; it is
    parameter-independent).
    """
    resid = data - m0 - M @ mu_p
    Ci_M = cov_inv @ M                                   # (n_data, n_lin)
    A = M.T @ Ci_M + jnp.diag(1.0 / sigma_p**2)          # (n_lin, n_lin)
    b = Ci_M.T @ resid                                   # (n_lin,)
    chol = jnp.linalg.cholesky(A)
    z = jax.scipy.linalg.cho_solve((chol, True), b)
    out = -0.5 * (resid @ cov_inv @ resid - b @ z)
    if include_logdet:
        logdet_A = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
        logdet_Sp = jnp.sum(jnp.log(sigma_p**2))       # ln det Sigma_p = sum log sigma^2
        out = out - 0.5 * (logdet_A + logdet_Sp)
    return out
