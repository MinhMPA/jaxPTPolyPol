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
    "make_constant_prior_fns",
    "make_marginal_log_posterior",
    "bin_lin_slices",
    "make_marginal_log_posterior_perbin",
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


def make_constant_prior_fns(mu_p, sigma_p):
    """Constant (theta_NL-independent) prior mean/width functions.

    Stream-A configuration: fiducial-centered means, Fisher-consistent widths.
    """
    mu_p = jnp.asarray(mu_p, dtype=jnp.float64)
    sigma_p = jnp.asarray(sigma_p, dtype=jnp.float64)
    return (lambda _theta_nl: mu_p), (lambda _theta_nl: sigma_p)


def make_marginal_log_posterior(theory_fn, data, cov_inv, lin_idx,
                                prior_mean_fn, prior_sigma_fn,
                                log_prior_nl_fn, to_physical, full_params_fn,
                                *, include_logdet: bool = True):
    """JIT-compiled marginal log-posterior in whitened theta_NL space.

    Mirrors sampler.make_log_posterior but integrates the linear block
    analytically at every step. prior_mean_fn / prior_sigma_fn receive the
    *physical* theta_NL vector and return (n_lin,) arrays -- constant for
    Stream A (make_constant_prior_fns), theta_NL-dependent for the
    arXiv:2511.20757 A_AP*A_amp-rescaled priors (Stream B).
    """
    data = jnp.asarray(data, dtype=jnp.float64)
    cov_inv = jnp.asarray(cov_inv, dtype=jnp.float64)
    templates = make_marginal_templates(theory_fn, lin_idx)

    @jax.jit
    def log_posterior(x):
        theta_nl = to_physical(x)
        full = full_params_fn(theta_nl)
        m0, M = templates(full)
        mu_p = prior_mean_fn(theta_nl)
        sigma_p = prior_sigma_fn(theta_nl)
        ll = gaussian_marginal_loglike(
            data, m0, M, cov_inv, mu_p, sigma_p, include_logdet=include_logdet)
        return ll + log_prior_nl_fn(theta_nl)

    return log_posterior


def _contiguous_slices(counts):
    """Consecutive slices of the given lengths, starting at 0."""
    out, start = [], 0
    for n in counts:
        out.append(slice(start, start + n))
        start += n
    return tuple(out)


def bin_lin_slices(split, n_bins):
    """Position of each bin's marginalized parameters inside the prior vectors.

    ``prior_mean_fn`` / ``prior_sigma_fn`` return ``(n_lin,)`` arrays laid out
    in ``split.lin_idx`` order. Because ``split_marginal_indices`` emits that
    order bin-major with an equal number of entries per bin, bin ``b`` owns the
    contiguous chunk ``slice(b * n_per, (b + 1) * n_per)``.

    Raises ``ValueError`` if ``split.lin_keys`` is not bin-major with equal
    counts, so a mis-ordered split cannot be silently mis-sliced.
    """
    n_bins = int(n_bins)
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    lin_keys = tuple(split.lin_keys)
    if len(lin_keys) != split.n_lin:
        raise ValueError(
            f"split.lin_keys has {len(lin_keys)} entries but n_lin={split.n_lin}")
    if split.n_lin % n_bins != 0:
        raise ValueError(
            f"n_lin={split.n_lin} is not an integer multiple of n_bins={n_bins}; "
            "bin_lin_slices requires the same number of marginalized parameters "
            "in every bin")
    n_per = split.n_lin // n_bins
    slices = _contiguous_slices([n_per] * n_bins)
    for b, sl in enumerate(slices):
        found = {key[0] for key in lin_keys[sl]}
        if found != {b}:
            raise ValueError(
                f"split.lin_keys is not bin-major: entries "
                f"[{sl.start}:{sl.stop}) reference bins {sorted(found)}, "
                f"expected only bin {b}")
    return slices


def make_marginal_log_posterior_perbin(*, bin_theory_fns, bin_data, bin_cov_invs,
                                       bin_lin_idx,
                                       extra_theory_fn=None, extra_data=None,
                                       extra_cov_inv=None,
                                       prior_mean_fn, prior_sigma_fn,
                                       log_prior_nl_fn, to_physical,
                                       full_params_fn,
                                       include_logdet: bool = True):
    """Exactly factorized version of :func:`make_marginal_log_posterior`.

    The data covariance is block-diagonal across redshift bins and each bin's
    theory depends only on its own ``theta_lin``, so the dense template matrix
    ``M`` is block-diagonal too. Both quadratic forms and
    ``ln det(A Sigma_p) = sum_b ln det(A_b Sigma_p,b)`` are then block
    separable, and the sum of per-bin marginalizations equals the dense one
    term by term -- same posterior, ~n_bins times smaller XLA graph.

    Parameters
    ----------
    bin_theory_fns : sequence of callables
        ``bin_theory_fns[b](full_params) -> (n_data_b,)``, the single-bin
        theory block (e.g. ``theory.make_joint_pk_bk_bin_fn`` with its static
        arguments pre-bound). Each takes the *full* packed parameter vector.
    bin_data, bin_cov_invs : sequences
        Per-bin data vectors ``(n_data_b,)`` and inverse covariances
        ``(n_data_b, n_data_b)``.
    bin_lin_idx : sequence of index sequences
        ``bin_lin_idx[b]`` holds bin ``b``'s marginalized parameters as
        *full-vector* indices, i.e. ``split.lin_idx[b * n_per:(b + 1) * n_per]``.
    extra_theory_fn, extra_data, extra_cov_inv : optional
        An additional block carrying no ``theta_lin`` dependence (the BAO
        likelihood). Enters as a plain ``-0.5 r^T Cinv r`` added once.
    prior_mean_fn, prior_sigma_fn : callables
        Unchanged Stream-A/B contract: ``fn(theta_nl) -> (n_lin,)`` in
        ``split.lin_idx`` order. Sliced per bin with the same contiguous
        bin-major layout as :func:`bin_lin_slices`.
    log_prior_nl_fn, to_physical, full_params_fn, include_logdet
        As in :func:`make_marginal_log_posterior`.

    Returns
    -------
    jitted ``log_posterior(x)`` in whitened theta_NL space.
    """
    n_bins = len(bin_theory_fns)
    if not (len(bin_data) == len(bin_cov_invs) == len(bin_lin_idx) == n_bins):
        raise ValueError(
            "bin_theory_fns, bin_data, bin_cov_invs and bin_lin_idx must have "
            f"the same length; got {n_bins}, {len(bin_data)}, "
            f"{len(bin_cov_invs)}, {len(bin_lin_idx)}")
    has_extra = extra_theory_fn is not None
    if has_extra and (extra_data is None or extra_cov_inv is None):
        raise ValueError(
            "extra_theory_fn requires both extra_data and extra_cov_inv")

    bin_data = tuple(jnp.asarray(d, dtype=jnp.float64) for d in bin_data)
    bin_cov_invs = tuple(jnp.asarray(c, dtype=jnp.float64) for c in bin_cov_invs)
    bin_templates = tuple(make_marginal_templates(fn, idx)
                          for fn, idx in zip(bin_theory_fns, bin_lin_idx))
    prior_slices = _contiguous_slices([len(idx) for idx in bin_lin_idx])
    if has_extra:
        extra_data = jnp.asarray(extra_data, dtype=jnp.float64)
        extra_cov_inv = jnp.asarray(extra_cov_inv, dtype=jnp.float64)

    @jax.jit
    def log_posterior(x):
        theta_nl = to_physical(x)
        full = full_params_fn(theta_nl)
        mu_p = prior_mean_fn(theta_nl)
        sigma_p = prior_sigma_fn(theta_nl)
        out = log_prior_nl_fn(theta_nl)
        for b in range(n_bins):
            m0, M = bin_templates[b](full)
            sl = prior_slices[b]
            out = out + gaussian_marginal_loglike(
                bin_data[b], m0, M, bin_cov_invs[b], mu_p[sl], sigma_p[sl],
                include_logdet=include_logdet)
        if has_extra:
            resid = extra_data - extra_theory_fn(full)
            out = out - 0.5 * (resid @ extra_cov_inv @ resid)
        return out

    return log_posterior
