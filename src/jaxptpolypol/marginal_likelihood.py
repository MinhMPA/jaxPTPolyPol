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

from dataclasses import dataclass

import jax
import jax.numpy as jnp

__all__ = [
    "gaussian_marginal_loglike",
    "LIN_SURVEY_KEYS",
    "MarginalSplit",
    "split_marginal_indices",
    "make_marginal_templates",
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


#: The 11 exactly-linear survey parameters per redshift bin, in canonical
#: marginalization order (arXiv:2511.20757 Table I "analytically marginalized";
#: verified linear to the float64 floor 2026-07-14, c1 linearized by template
#: construction -- see module docstring).
LIN_SURVEY_KEYS = (
    ('shared', 'bias', 'bGamma3'),
    ('shared', 'stoch', 'P_shot'),
    ('pk', 'ctr', 'c0'),
    ('pk', 'ctr', 'c2'),
    ('pk', 'ctr', 'c4'),
    ('pk', 'ctr', 'cfog'),
    ('pk', 'stoch', 'a0'),
    ('pk', 'stoch', 'a2'),
    ('bk', 'ctr', 'c1'),
    ('bk', 'stoch', 'B_shot'),
    ('bk', 'stoch', 'A_shot'),
)


@dataclass(frozen=True)
class MarginalSplit:
    """Full-vector index bookkeeping for the sampled/marginalized partition."""
    nl_idx: tuple
    lin_idx: tuple
    lin_keys: tuple
    nl_b1_pos: tuple
    n_lin: int
    n_nl: int


def split_marginal_indices(*, n_cosmo_params, survey_keys, n_bins,
                           fixed_cosmo=(), fixed_survey_keys=frozenset(),
                           lin_survey_keys=LIN_SURVEY_KEYS):
    """Partition the packed vector [cosmo | survey_bin0 | ...] into the
    sampled block theta_NL and the marginalized block theta_lin."""
    survey_keys = tuple(survey_keys)
    fixed_survey_keys = frozenset(fixed_survey_keys)
    missing = [k for k in lin_survey_keys if k not in survey_keys]
    if missing:
        raise ValueError(f"lin_survey_keys not present in survey_keys: {missing}")
    overlap = fixed_survey_keys & set(lin_survey_keys)
    if overlap:
        raise ValueError(f"keys cannot be both fixed and marginalized: {sorted(overlap)}")

    lin_offsets = {k: survey_keys.index(k) for k in lin_survey_keys}
    nl_survey_offsets = [
        i for i, k in enumerate(survey_keys)
        if k not in lin_offsets and k not in fixed_survey_keys
    ]
    fixed_cosmo = set(int(i) for i in fixed_cosmo)

    nl_idx = [i for i in range(n_cosmo_params) if i not in fixed_cosmo]
    n_cosmo_varied = len(nl_idx)
    nl_b1_pos = []
    lin_idx, lin_keys = [], []
    b1_offset = survey_keys.index(('shared', 'bias', 'b1'))
    n_survey = len(survey_keys)
    for b in range(n_bins):
        base = n_cosmo_params + b * n_survey
        for off in nl_survey_offsets:
            if off == b1_offset:
                nl_b1_pos.append(len(nl_idx))
            nl_idx.append(base + off)
        for key in lin_survey_keys:
            lin_idx.append(base + lin_offsets[key])
            lin_keys.append((b, key))

    return MarginalSplit(
        nl_idx=tuple(nl_idx), lin_idx=tuple(lin_idx), lin_keys=tuple(lin_keys),
        nl_b1_pos=tuple(nl_b1_pos), n_lin=len(lin_idx), n_nl=len(nl_idx),
    )


def make_marginal_templates(theory_fn, lin_idx):
    """Build the exact linear templates m0(theta_NL) = t(theta_lin=0) and
    M(theta_NL) = dt/dtheta_lin.

    Uses jax.linearize: one primal trace + n_lin forward tangents. Exact for
    parameters the theory is linear in; for c1 (exactly quadratic in the
    ps_1loop_jax bispectrum) this returns the slope at c1=0, i.e. the
    linearized single-insertion counterterm of arXiv:2511.20757.
    """
    lin_idx_arr = jnp.array(lin_idx)
    n_lin = len(lin_idx)
    eye = jnp.eye(n_lin)

    def templates(full_params):
        base = full_params.at[lin_idx_arr].set(0.0)

        def t_of_lin(lin_values):
            return theory_fn(base.at[lin_idx_arr].set(lin_values))

        m0, jvp = jax.linearize(t_of_lin, jnp.zeros(n_lin))
        M = jax.vmap(jvp)(eye)          # (n_lin, n_data)
        return m0, M.T                  # (n_data, n_lin)

    return templates
