"""
Precomputed Taylor expansions of the per-bin marginal linear templates.

Measured motivation
-------------------
The marginal-likelihood posterior (:mod:`jaxptpolypol.marginal_likelihood`)
rebuilds its exact linear templates ``(m0(theta_NL), M(theta_NL))`` from the
theory on *every* evaluation via :func:`make_marginal_templates` (a
``jax.linearize`` pass through the full ps_1loop_jax graph). On the production
7-bin P+B configuration this template reconstruction *is* the per-eval cost:
5.06 s per posterior call (see
``docs/design/perbin-compile-measurements.md``). A gradient-free
Metropolis-Hastings chain therefore spends essentially all of its wall time
re-tracing a graph whose only free inputs are the ~40 slow theta_NL.

Options review (recorded in the plan):

- **F5-a** -- keep reconstructing the exact templates each step. Correct but
  5 s/step; a 10^4-step chain is ~14 h of pure forward evals.
- **F5-b** -- precompute, *once*, a low-order Taylor expansion of each bin's
  ``(m0, M)`` about the theta_NL fiducial ``theta0``, so each subsequent
  evaluation is a few dense tensor contractions (~ms). This module builds the
  F5-b expansion; downstream tasks consume it.

Why ``M(theta_NL)`` must be expanded too (the logdet tilt)
---------------------------------------------------------
The Gaussian marginal ``-2 ln L`` carries a ``ln det(A Sigma_p)`` term with
``A = M^T Cinv M + Sigma_p^{-1}``. That log-determinant "tilt" of the posterior
is entirely a function of how the template matrix ``M(theta_NL)`` *varies* with
the slow parameters -- freezing ``M`` at ``M(theta0)`` would silently drop it.
So the expansion keeps ``M0 = M(theta0)`` **and** its first-order variation
``dM = dM/dtheta_NL``; only ``m0`` (which enters the residual quadratically and
whose curvature genuinely matters) is carried to second order (``H``).

Expansion carried, per bin ``b`` (``n_b`` data points, ``p_b`` linear params,
``d`` slow parameters):

    m0(theta0 + u) ~= m00 + J @ u + 1/2 u^T H u      (H optional, order2_m0)
    M(theta0 + u)  ~= M0 + dM @ u

Differentiation strategy (forward-over-forward, chunked, never reverse)
----------------------------------------------------------------------
``make_marginal_templates`` computes ``M`` with an inner ``jax.linearize``
(forward mode). Every outer derivative below is taken with ``jax.jacfwd`` on top
of that, so the whole build is forward-over-forward and **no reverse-mode tape
is ever materialised** (a reverse tape over the ps_1loop_jax + linearize graph
is what blows up compile memory).

The outer ``jacfwd`` is evaluated in **column chunks** of ``eye(d)`` rather than
one full-width pass: for a chunk ``E`` of shape ``(d, c)`` we evaluate
``jax.jacfwd(lambda s: f(theta0 + E @ s))(zeros(c))`` and paste the ``c``
columns. ``chunk_J`` sizes the first-order (``J``, ``dM``) chunks. The second
derivative ``H`` is a ``jacfwd``-of-``jacfwd``: the inner Jacobian of ``m0`` is
taken full-``d`` (single pass) and the *outer* differentiation is chunked by
``chunk_H`` -- i.e. ``chunk_H`` outer x full-``d`` inner is the default memory
profile, and ``chunk_H`` is the knob a caller lowers if the predicted build
memory is too large. Both levels flow through the same chunkable helper.

Per-bin work is an unrolled **Python loop** over bins (never ``lax.scan``): the
per-bin theory closures are heterogeneous (each bakes in its redshift/statics),
and a scan body over them defeats XLA -- a measured lesson recorded alongside
:func:`make_marginal_log_posterior_scan`.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .marginal_likelihood import (
    _contiguous_slices,
    gaussian_marginal_loglike,
    make_marginal_templates,
)

__all__ = [
    "TaylorTemplates",
    "build_taylor_templates",
    "make_marginal_log_posterior_taylor",
]


@dataclass(frozen=True)
class TaylorTemplates:
    """Precomputed per-bin Taylor tensors of the marginal linear templates.

    All per-bin fields are tuples indexed by bin. Shapes, per bin ``b``:

    - ``bin_m00[b]`` : ``(n_b,)``      -- ``m0(theta0)``.
    - ``bin_J[b]``   : ``(n_b, d)``    -- ``dm0/dtheta_NL`` at ``theta0``.
    - ``bin_H[b]``   : ``(n_b, d, d)`` or ``None`` -- symmetrized
      ``d^2 m0/dtheta_NL^2`` at ``theta0`` (``None`` when ``order2_m0`` is
      ``False``).
    - ``bin_M0[b]``  : ``(n_b, p_b)``    -- ``M(theta0)``.
    - ``bin_dM[b]``  : ``(n_b, p_b, d)`` -- ``dM/dtheta_NL`` at ``theta0``.

    ``theta0`` is the ``(d,)`` physical theta_NL fiducial the expansion is
    centred on. ``order2_m0`` records whether ``H`` was built.
    ``build_diagnostics`` is a plain dict populated by the builder (per-bin
    Hessian symmetry error and the chunk sizes used).
    """

    theta0: jnp.ndarray
    bin_m00: tuple
    bin_J: tuple
    bin_H: tuple
    bin_M0: tuple
    bin_dM: tuple
    order2_m0: bool
    build_diagnostics: dict


def _chunked_jacfwd(fn, x0, chunk):
    """``jax.jacfwd(fn)(x0)`` evaluated in column chunks of ``eye(d)``.

    ``fn`` maps ``(d,)`` to an arbitrary pytree of arrays. The Jacobian is
    assembled by, for each contiguous block of ``chunk`` coordinate directions,
    evaluating ``jax.jacfwd(lambda s: fn(x0 + E @ s))(zeros(c))`` (``E`` the
    matching ``(d, c)`` slice of the identity) and concatenating the resulting
    ``c``-wide tangent axis -- which ``jacfwd`` always appends as the *last*
    axis of every output leaf -- across chunks.

    Forward mode throughout: safe to nest under an outer ``jacfwd`` (the ``H``
    build does exactly this) without ever creating a reverse tape.
    """
    d = x0.shape[0]
    eye = jnp.eye(d, dtype=x0.dtype)
    parts = []
    for start in range(0, d, chunk):
        E = eye[:, start:start + chunk]                       # (d, c)
        c = E.shape[1]
        jac = jax.jacfwd(lambda s, E=E: fn(x0 + E @ s))(
            jnp.zeros(c, dtype=x0.dtype))
        parts.append(jac)
    return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=-1), *parts)


def build_taylor_templates(*, bin_theory_fns, bin_lin_idx, full_params_fn,
                           theta0, order2_m0=True, chunk_J=4, chunk_H=2):
    """Build the per-bin Taylor expansion of ``(m0, M)`` about ``theta0``.

    Parameters
    ----------
    bin_theory_fns : sequence of callables
        ``bin_theory_fns[b](full_params) -> (n_b,)`` -- the single-bin theory
        block with its static configuration pre-bound (as consumed by
        :func:`make_marginal_log_posterior_perbin`).
    bin_lin_idx : sequence of index sequences
        ``bin_lin_idx[b]`` = bin ``b``'s marginalized parameters as
        *full-vector* indices.
    full_params_fn : callable
        ``theta_NL -> full_params`` from
        :func:`sampler.make_full_params_fn(packed, nl_idx)`.
    theta0 : array_like, shape ``(d,)``
        Physical theta_NL fiducial the expansion is centred on.
    order2_m0 : bool
        Build the second-order term ``H`` (curvature of ``m0`` only). Default
        ``True``.
    chunk_J : int
        Column-chunk width for the first-order ``jacfwd`` (``J``, ``dM``).
    chunk_H : int
        Column-chunk width for the *outer* level of the ``H``
        ``jacfwd``-of-``jacfwd`` (the inner Jacobian is taken full-``d``).

    Returns
    -------
    TaylorTemplates
    """
    theta0 = jnp.asarray(theta0, dtype=jnp.float64)
    if theta0.ndim != 1:
        raise ValueError(f"theta0 must be 1-d, got shape {theta0.shape}")
    n_bins = len(bin_theory_fns)
    if len(bin_lin_idx) != n_bins:
        raise ValueError(
            "bin_theory_fns and bin_lin_idx must have the same length; "
            f"got {n_bins} and {len(bin_lin_idx)}")

    bin_m00, bin_J, bin_H, bin_M0, bin_dM = [], [], [], [], []
    sym_errs = []

    for b in range(n_bins):
        templates_b = make_marginal_templates(bin_theory_fns[b], bin_lin_idx[b])

        def f_b(theta_nl, _t=templates_b):
            # theta_NL -> (m0_b (n_b,), M_b (n_b, p_b))
            return _t(full_params_fn(theta_nl))

        # Center (exact).
        m00, M0 = f_b(theta0)
        bin_m00.append(m00)
        bin_M0.append(M0)

        # First order: chunked forward-over-forward jacfwd of the (m0, M) pair.
        J0, dM = _chunked_jacfwd(f_b, theta0, chunk_J)   # (n_b, d), (n_b, p_b, d)
        bin_J.append(J0)
        bin_dM.append(dM)

        # Second order (m0 only): jacfwd-of-jacfwd, full-d inner x chunk_H outer.
        if order2_m0:
            def m0_jac(theta, _f=f_b):
                d = theta.shape[0]
                return _chunked_jacfwd(lambda th: _f(th)[0], theta, d)  # (n_b, d)

            H = _chunked_jacfwd(m0_jac, theta0, chunk_H)          # (n_b, d, d)
            H_T = jnp.transpose(H, (0, 2, 1))
            denom = jnp.max(jnp.abs(H))
            sym_err = float(
                jnp.max(jnp.abs(H - H_T)) / jnp.where(denom > 0.0, denom, 1.0))
            H = 0.5 * (H + H_T)
            bin_H.append(H)
            sym_errs.append(sym_err)
        else:
            bin_H.append(None)

    build_diagnostics = {
        "order2_m0": order2_m0,
        "chunk_J": chunk_J,
        "chunk_H": chunk_H,
        "H_sym_err": tuple(sym_errs),
    }

    return TaylorTemplates(
        theta0=theta0,
        bin_m00=tuple(bin_m00),
        bin_J=tuple(bin_J),
        bin_H=tuple(bin_H),
        bin_M0=tuple(bin_M0),
        bin_dM=tuple(bin_dM),
        order2_m0=order2_m0,
        build_diagnostics=build_diagnostics,
    )


def make_marginal_log_posterior_taylor(tt, *, bin_data, bin_cov_invs,
                                       prior_mean_fn, prior_sigma_fn,
                                       log_prior_nl_fn, to_physical,
                                       extra_theory_fn=None, extra_data=None,
                                       extra_cov_inv=None,
                                       include_logdet: bool = True):
    """Surrogate marginal log-posterior built from precomputed Taylor templates.

    Drop-in replacement for :func:`marginal_likelihood.
    make_marginal_log_posterior_perbin` that consumes a :class:`TaylorTemplates`
    instead of re-tracing the theory: each per-bin ``(m0, M)`` is reconstructed
    from the carried tensors as dense contractions (~ms) rather than a
    ``jax.linearize`` pass through ps_1loop_jax (~s). On models the expansion
    represents exactly (``m0`` quadratic, ``M`` linear in theta_NL) the surrogate
    reproduces the exact per-bin posterior to the float64 floor; otherwise it is
    the F5-b Taylor approximation (see the module docstring).

    Per bin ``b`` with ``delta = to_physical(x) - tt.theta0``:

        m0 = m00 + J @ delta  (+ 1/2 delta^T H delta   if ``tt.order2_m0``)
        M  = M0 + dM @ delta

    then :func:`gaussian_marginal_loglike` on ``(bin_data[b], m0, M,
    bin_cov_invs[b], mu_p[sl_b], sigma_p[sl_b])`` summed over bins.

    Parameters
    ----------
    tt : TaylorTemplates
        Output of :func:`build_taylor_templates`. The per-bin linear-parameter
        counts (``tt.bin_M0[b].shape[1]``) fix the prior-vector tiling.
    bin_data, bin_cov_invs : sequences
        Per-bin data vectors ``(n_b,)`` and inverse covariances ``(n_b, n_b)``.
    prior_mean_fn, prior_sigma_fn : callables
        Same contract as the per-bin builder: ``fn(theta_nl) -> (n_lin,)`` in
        bin-major order, sliced per bin by the contiguous per-bin lin counts.
        A returned width that does not match ``sum_b p_b`` raises ``ValueError``
        naming the offending function (the tiling guard).
    log_prior_nl_fn, to_physical, include_logdet
        As in :func:`marginal_likelihood.make_marginal_log_posterior_perbin`.
    extra_theory_fn, extra_data, extra_cov_inv : optional
        A theta_lin-independent block (the BAO likelihood), all-or-none, added
        once as a plain ``-0.5 r^T Cinv r`` *outside* the bin loop and evaluated
        exactly. Because the surrogate carries no ``full_params_fn``, this block
        is evaluated at the physical theta_NL vector: ``extra_theory_fn`` here
        receives ``theta_nl`` (not the full packed vector as in the per-bin
        form).

    Returns
    -------
    jitted ``log_posterior(x)`` in whitened theta_NL space.
    """
    n_bins = len(tt.bin_M0)
    if not (len(bin_data) == len(bin_cov_invs) == n_bins):
        raise ValueError(
            "bin_data and bin_cov_invs must match the number of bins in tt; "
            f"got {len(bin_data)}, {len(bin_cov_invs)} vs {n_bins} bins")
    has_extra = extra_theory_fn is not None
    if has_extra and (extra_data is None or extra_cov_inv is None):
        raise ValueError(
            "extra_theory_fn requires both extra_data and extra_cov_inv")

    bin_data = tuple(jnp.asarray(d, dtype=jnp.float64) for d in bin_data)
    bin_cov_invs = tuple(jnp.asarray(c, dtype=jnp.float64) for c in bin_cov_invs)
    theta0 = tt.theta0
    order2 = tt.order2_m0
    bin_counts = [int(M0.shape[1]) for M0 in tt.bin_M0]
    prior_slices = _contiguous_slices(bin_counts)
    n_lin_total = prior_slices[-1].stop
    if has_extra:
        extra_data = jnp.asarray(extra_data, dtype=jnp.float64)
        extra_cov_inv = jnp.asarray(extra_cov_inv, dtype=jnp.float64)

    @jax.jit
    def log_posterior(x):
        theta_nl = to_physical(x)
        delta = theta_nl - theta0
        mu_p = prior_mean_fn(theta_nl)
        sigma_p = prior_sigma_fn(theta_nl)
        # The per-bin prior slices must tile the prior vectors exactly (same guard
        # as the per-bin builder): a mis-sized prior width would otherwise give a
        # silently wrong posterior. Shapes are static at trace time -- free.
        for name, vec in (("prior_mean_fn", mu_p), ("prior_sigma_fn", sigma_p)):
            if vec.shape != (n_lin_total,):
                raise ValueError(
                    f"{name} returned shape {vec.shape}, but tt implies "
                    f"{(n_lin_total,)} linear parameters ({bin_counts} per bin)")
        out = log_prior_nl_fn(theta_nl)
        for b in range(n_bins):
            m0 = tt.bin_m00[b] + tt.bin_J[b] @ delta
            if order2:
                m0 = m0 + 0.5 * jnp.einsum(
                    "ijk,j,k->i", tt.bin_H[b], delta, delta)
            M = tt.bin_M0[b] + jnp.einsum("ijk,k->ij", tt.bin_dM[b], delta)
            sl = prior_slices[b]
            out = out + gaussian_marginal_loglike(
                bin_data[b], m0, M, bin_cov_invs[b], mu_p[sl], sigma_p[sl],
                include_logdet=include_logdet)
        if has_extra:
            resid = extra_data - extra_theory_fn(theta_nl)
            out = out - 0.5 * (resid @ extra_cov_inv @ resid)
        return out

    return log_posterior
