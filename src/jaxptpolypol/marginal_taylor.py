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

import json
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .marginal_likelihood import (
    _contiguous_slices,
    gaussian_marginal_loglike,
    make_marginal_templates,
)

__all__ = [
    "TaylorTemplates",
    "build_taylor_templates",
    "make_marginal_log_posterior_taylor",
    "save_taylor_templates",
    "load_taylor_templates",
    "importance_reweight",
    "reweighted_moments",
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
        #
        # CRITICAL memory detail: differentiate an m0-ONLY closure (the theory
        # evaluated at theta_lin = 0), NOT f_b(th)[0]. f_b runs the full
        # _linear_templates machinery -- jax.linearize + a p_b-lane vmap that
        # builds M -- and under nested jacfwd tracing there is no jit boundary
        # to dead-code-eliminate the discarded M, so its p_b tangent lanes
        # multiply BOTH jacfwd widths (measured at production: the inner
        # full-d pass floored at ~82 GB regardless of chunk_H). The m0-only
        # closure reproduces m0 bit-identically (m0 == theory at theta_lin=0,
        # the same .at[].set(0.0) op _linear_templates uses) with plain
        # d-wide tangents: ~9 GB inner at production. The toy tests pin H
        # exactly, proving behavior identity.
        if order2_m0:
            lin_idx_arr_b = jnp.array(bin_lin_idx[b])

            def m0_only(theta_nl, _fn=bin_theory_fns[b], _idx=lin_idx_arr_b):
                return _fn(full_params_fn(theta_nl).at[_idx].set(0.0))

            def m0_jac(theta, _f=m0_only):
                d = theta.shape[0]
                return _chunked_jacfwd(_f, theta, d)              # (n_b, d)

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
                                       full_params_fn,
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
    log_prior_nl_fn, to_physical, full_params_fn, include_logdet
        As in :func:`marginal_likelihood.make_marginal_log_posterior_perbin`.
        ``full_params_fn`` is used *only* for the exact extra term (the
        precomputed templates never need it); it restores drop-in contract
        symmetry with the per-bin builder, so the production BAO closure
        ``lambda p: bao_theory_fn(p[:n_cosmo_params])`` works verbatim against
        the FULL packed vector. Without it, slicing theta_NL would silently
        take 5 cosmology + 4 bias entries.
    extra_theory_fn, extra_data, extra_cov_inv : optional
        A theta_lin-independent block (the BAO likelihood), all-or-none, added
        once as a plain ``-0.5 r^T Cinv r`` *outside* the bin loop and evaluated
        exactly on the full packed vector:
        ``resid = extra_data - extra_theory_fn(full_params_fn(theta_nl))`` --
        the same argument convention as the per-bin form.

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
            resid = extra_data - extra_theory_fn(full_params_fn(theta_nl))
            out = out - 0.5 * (resid @ extra_cov_inv @ resid)
        return out

    return log_posterior


# --- npz persistence with a stale-config guard --------------------------------
#
# A production build of the Taylor tensors is a one-off 15-40 min forward-over-
# forward pass (see the module docstring); it must be reusable across sampler
# runs. But the tensors are *only* valid for the exact data configuration they
# were built against -- change n_bins, the k/triangle grid, the GL/mu/phi
# quadrature, or the theta_NL centre, and silently reusing an old ``.npz`` would
# sample a posterior that no longer matches the notebook. ``save`` therefore
# stamps a flat ``meta`` dict of config identifiers into the file, and ``load``
# refuses (loudly, listing every offending key) to hand back templates whose
# stored ``meta`` disagrees with the caller's ``expect_meta``.

_MISSING = "<absent>"


def save_taylor_templates(tt, path, *, meta: dict):
    """Persist ``tt`` to a single ``.npz`` file with a config-identifier stamp.

    Every tensor is written under its own key so the file is inspectable without
    unpickling: ``theta0``; ``m00_b``, ``J_b``, ``M0_b``, ``dM_b`` for each bin
    ``b`` (and ``H_b`` only for bins whose ``bin_H[b]`` is not ``None``); the
    scalars ``order2_m0`` and ``n_bins``; and both the ``meta`` and
    ``build_diagnostics`` dicts JSON-serialised into 0-d string arrays.

    Parameters
    ----------
    tt : TaylorTemplates
        Output of :func:`build_taylor_templates`.
    path : str or path-like
        Destination ``.npz`` file.
    meta : dict
        FLAT dict of config identifiers (e.g. ``n_bins``, ``n_k``, ``n_tri``,
        ``n_gl``, ``num_mu``, ``num_phi``, ``k_min``, ``k_max``, ``x0_hash``).
        Every value must be ``str``/``int``/``float``/``bool``; a non-scalar
        value raises ``TypeError`` (it could not round-trip as a config stamp,
        and would defeat the :func:`load_taylor_templates` guard).
    """
    for key, val in meta.items():
        if not isinstance(val, (str, int, float, bool)):
            raise TypeError(
                "meta must be a flat dict of config identifiers "
                "(str/int/float/bool values); "
                f"key {key!r} has value of type {type(val).__name__}")

    n_bins = len(tt.bin_m00)
    arrays = {
        "theta0": np.asarray(tt.theta0),
        "order2_m0": np.asarray(bool(tt.order2_m0)),
        "n_bins": np.asarray(int(n_bins)),
        "meta": np.asarray(json.dumps(meta)),
        "build_diagnostics": np.asarray(json.dumps(tt.build_diagnostics)),
    }
    for b in range(n_bins):
        arrays[f"m00_{b}"] = np.asarray(tt.bin_m00[b])
        arrays[f"J_{b}"] = np.asarray(tt.bin_J[b])
        arrays[f"M0_{b}"] = np.asarray(tt.bin_M0[b])
        arrays[f"dM_{b}"] = np.asarray(tt.bin_dM[b])
        if tt.bin_H[b] is not None:
            arrays[f"H_{b}"] = np.asarray(tt.bin_H[b])

    np.savez(path, **arrays)


def load_taylor_templates(path, *, expect_meta: dict | None = None):
    """Reconstruct a :class:`TaylorTemplates` from a :func:`save_taylor_templates` file.

    All tensors come back as ``float64`` ``jnp`` arrays; ``bin_H`` entries are
    ``None`` for any bin that carried no ``H`` (order-1 build). The returned
    object's ``build_diagnostics`` is the stored build diagnostics with the
    loaded ``meta`` attached under a ``"meta"`` key.

    Parameters
    ----------
    path : str or path-like
        A ``.npz`` file written by :func:`save_taylor_templates`.
    expect_meta : dict, optional
        If given, the stale-template guard: the stored ``meta`` must equal
        ``expect_meta`` key-for-key. A key present in only one of the two dicts,
        or present in both with differing values, is a mismatch. Any mismatch
        raises ``ValueError`` listing EVERY offending key with its stored and
        expected values, so a regenerated notebook config cannot silently sample
        with tensors built for a different configuration.

    Returns
    -------
    TaylorTemplates
    """
    with np.load(path, allow_pickle=False) as npz:
        stored_meta = json.loads(str(npz["meta"].item()))

        if expect_meta is not None:
            mismatches = []
            for key in sorted(set(stored_meta) | set(expect_meta)):
                sv = stored_meta.get(key, _MISSING)
                ev = expect_meta.get(key, _MISSING)
                if (key not in stored_meta or key not in expect_meta
                        or sv != ev):
                    mismatches.append(
                        f"  {key}: stored={sv!r}, expected={ev!r}")
            if mismatches:
                raise ValueError(
                    "Stale Taylor templates: stored config meta does not match "
                    "expect_meta on the following key(s):\n"
                    + "\n".join(mismatches))

        n_bins = int(npz["n_bins"])
        order2_m0 = bool(npz["order2_m0"])
        theta0 = jnp.asarray(npz["theta0"], dtype=jnp.float64)

        def _get(prefix, b):
            return jnp.asarray(npz[f"{prefix}_{b}"], dtype=jnp.float64)

        bin_m00 = tuple(_get("m00", b) for b in range(n_bins))
        bin_J = tuple(_get("J", b) for b in range(n_bins))
        bin_M0 = tuple(_get("M0", b) for b in range(n_bins))
        bin_dM = tuple(_get("dM", b) for b in range(n_bins))
        bin_H = tuple(
            _get("H", b) if f"H_{b}" in npz.files else None
            for b in range(n_bins))

        build_diagnostics = json.loads(str(npz["build_diagnostics"].item()))

    build_diagnostics = dict(build_diagnostics)
    build_diagnostics["meta"] = stored_meta

    return TaylorTemplates(
        theta0=theta0,
        bin_m00=bin_m00,
        bin_J=bin_J,
        bin_H=bin_H,
        bin_M0=bin_M0,
        bin_dM=bin_dM,
        order2_m0=order2_m0,
        build_diagnostics=build_diagnostics,
    )


# --- Importance reweighting: restore asymptotic exactness post-hoc ------------
#
# The Taylor surrogate is fast (~ms/eval) but only *asymptotically* exact -- on
# a model it does not represent exactly it is a low-order approximation whose
# error grows into the tails. A surrogate MCMC chain therefore targets a
# slightly-wrong posterior. Importance reweighting corrects this after the fact:
# each surrogate sample x_i is re-weighted by w_i propto p_exact(x_i) /
# p_surrogate(x_i), so weighted expectations converge to the EXACT posterior --
# *provided the surrogate covers the exact posterior's support*. When it does
# not (the surrogate is too narrow, under-covering the tails), a handful of
# tail samples carry almost all the weight, the effective sample size collapses,
# and the reweighted answer is unreliable. The returned diagnostics (``ess``,
# ``ess_frac``, ``max_weight``) exist precisely to make that failure visible --
# they are to be reported, never hidden.


def _flatten_samples(samples):
    """Reshape ``(chains, n, d)`` -> ``(chains*n, d)``; pass ``(n, d)`` through.

    Returns a plain ``numpy`` array (this utility runs post-sampling on host
    memory, not inside a jit).
    """
    arr = np.asarray(samples)
    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])
    return arr


def importance_reweight(samples, log_p_exact_fn, log_p_surrogate_fn, *,
                        subsample=None, seed=0):
    """Importance-reweight surrogate samples back onto the exact posterior.

    Given samples drawn against ``log_p_surrogate_fn`` (a surrogate MCMC chain),
    compute normalized importance weights ``w_i propto exp(log_p_exact(x_i) -
    log_p_surrogate(x_i))`` so that ``sum_i w_i f(x_i)`` estimates the EXACT
    posterior expectation of ``f``. Also returns the effective-sample-size
    diagnostics that reveal when this correction cannot be trusted.

    Parameters
    ----------
    samples : array_like
        ``(n, d)`` samples, or ``(chains, n, d)`` which is reshaped to
        ``(chains*n, d)`` (chain axis flattened). ``d`` is the parameter
        dimension; a 1-D problem uses ``d = 1`` (shape ``(n, 1)``).
    log_p_exact_fn, log_p_surrogate_fn : callable
        ``fn(x) -> scalar`` log-posteriors, called on ONE sample ``x`` of shape
        ``(d,)`` at a time. **The exact posterior is evaluated in a plain Python
        loop, one sample per call -- it is deliberately NOT vmapped.** In
        production the exact marginal posterior costs ~5 s and is memory-heavy
        per evaluation (it re-traces the full ps_1loop_jax graph), and its
        contract is single-``x`` calls; batching it with ``vmap`` would blow up
        memory. The (arbitrary, unnormalized) additive constants of the two
        log-posteriors cancel only up to the shared shift removed below, so only
        their *difference* need be meaningful.
    subsample : int, optional
        If given, evaluate both log-posteriors on only ``subsample`` sample
        indices drawn uniformly WITHOUT replacement (via
        ``numpy.random.default_rng(seed)``), returned sorted in ``idx``. This is
        the laptop path: at ~5 s per exact eval, evaluating all ``n`` samples is
        infeasible, so a random subset is reweighted instead. If ``None``, all
        ``n`` samples are used.
    seed : int
        Seed for the subsample RNG (ignored when ``subsample`` is ``None``).

    Returns
    -------
    dict with keys
        ``weights`` : ``(m,)`` normalized weights summing to 1 (``m = subsample``
            or ``n``).
        ``log_w_raw`` : ``(m,)`` UNSHIFTED log-weights
            ``log_p_exact - log_p_surrogate`` on the evaluated set (before the
            log-sum-exp stabilization).
        ``ess`` : Kish effective sample size ``1 / sum_i weights_i**2``.
        ``ess_frac`` : ``ess / m`` in ``(0, 1]``.
        ``max_weight`` : the largest single normalized weight.
        ``idx`` : ``(m,)`` int indices of the evaluated samples into the
            FLATTENED ``(chains*n, d)`` sample stack (``arange(n)`` when not
            subsampling).

    Interpretation contract (READ THIS)
    -----------------------------------
    A **small ``ess_frac``** or a **``max_weight`` approaching 1** means the
    surrogate UNDER-COVERS the exact posterior's tails: a few samples dominate
    the weights, the reweighted moments are driven by those few draws, and the
    reweighted answer is **NOT trustworthy** -- the importance-sampling estimator
    has effectively collapsed to a handful of samples. This is a property of the
    surrogate/exact mismatch, not a bug. Report these diagnostics alongside any
    reweighted result; do not silently trust a reweighted answer with a low
    ``ess_frac``. (A well-covered surrogate gives ``ess_frac`` near 1 and a tiny
    ``max_weight`` ~ ``1/m``.)
    """
    arr = _flatten_samples(samples)
    n = arr.shape[0]

    if subsample is not None:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=int(subsample), replace=False))
    else:
        idx = np.arange(n)

    eval_samples = arr[idx]
    m = idx.shape[0]

    # Plain Python loop, one sample per call -- do NOT vmap (see docstring): the
    # production exact posterior is memory-heavy and contracted for single-x.
    log_p_ex = np.empty(m, dtype=np.float64)
    log_p_sur = np.empty(m, dtype=np.float64)
    for i in range(m):
        x = eval_samples[i]
        log_p_ex[i] = float(log_p_exact_fn(x))
        log_p_sur[i] = float(log_p_surrogate_fn(x))

    log_w_raw = log_p_ex - log_p_sur              # unshifted log-weights
    lw = log_w_raw - log_w_raw.max()              # log-sum-exp stabilization
    w = np.exp(lw)
    weights = w / w.sum()

    sum_w2 = np.sum(weights ** 2)
    ess = float(1.0 / sum_w2)

    return {
        "weights": weights,
        "log_w_raw": log_w_raw,
        "ess": ess,
        "ess_frac": ess / m,
        "max_weight": float(weights.max()),
        "idx": idx,
    }


def reweighted_moments(samples, weights, idx=None):
    """Weighted mean and standard deviation of samples under importance weights.

    Parameters
    ----------
    samples : array_like
        ``(n, d)`` or ``(chains, n, d)`` (flattened as in
        :func:`importance_reweight`).
    weights : array_like
        ``(m,)`` normalized weights (sum to 1) from :func:`importance_reweight`.
    idx : array_like of int, optional
        The evaluated indices ``weights`` correspond to (the ``idx`` returned by
        :func:`importance_reweight`). When given, moments are taken over
        ``samples[idx]``; when ``None``, ``weights`` must align with all ``n``
        flattened samples.

    Returns
    -------
    (mean, std) : each ``(d,)``
        ``mean = sum_i w_i x_i``. ``std`` uses the reliability-weights unbiased
        estimator ``sqrt( sum_i w_i (x_i - mean)**2 / (1 - sum_i w_i**2) )``: for
        *normalized* weights this reduces to the familiar ``1/(N-1)`` correction
        when all weights are equal (``w_i = 1/N`` gives ``1 - sum w_i**2 =
        (N-1)/N``), and it is the standard bias correction for the frequency
        interpretation of importance weights. A near-degenerate weight set
        (``sum w_i**2 -> 1``, i.e. one sample dominates) makes the denominator
        collapse, correctly signalling that ``std`` is ill-determined -- pair
        this with the :func:`importance_reweight` ``ess_frac`` diagnostic.
    """
    arr = _flatten_samples(samples)
    if idx is not None:
        arr = arr[np.asarray(idx)]
    w = np.asarray(weights, dtype=np.float64)

    mean = np.sum(w[:, None] * arr, axis=0)
    var = (np.sum(w[:, None] * (arr - mean) ** 2, axis=0)
           / (1.0 - np.sum(w ** 2)))
    return mean, np.sqrt(var)
