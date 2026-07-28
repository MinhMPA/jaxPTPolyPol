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

from .marginal_likelihood import make_marginal_templates

__all__ = ["TaylorTemplates", "build_taylor_templates"]


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
