#!/usr/bin/env python3
"""Gauss-Newton (expected) Fisher pieces for the CMB likelihood terms.

Motivation
----------
``build_cmb_fisher_block.py`` builds the CMB Fisher block as the OBSERVED
Hessian ``F = -0.5 (H + H^T)`` of the summed log-likelihood at our fiducial
cosmology. Those likelihoods are fits to REAL Planck/ACT data, and our fiducial
is not their joint maximum, so the residual ``delta = d - m(theta_fid) != 0``.
For a Gaussian band-power likelihood

    log L = -0.5 (d - m(theta))^T S (d - m(theta)),    S = C^{-1}

the exact Hessian is

    -d2 logL = J^T S J  -  sum_a (S delta)_a  d2 m_a / dtheta^2
               \_______/   \_____________________________________/
               Gauss-Newton          residual-curvature term

The second term has no definite sign. Along a near-null direction of ``J^T S J``
(the CMB geometric degeneracy, ~99.7% H0 with a little mnu) it dominates and
tips the observed Hessian negative -- this is exactly the nuLCDM G2 failure
(raw min eig -0.250293, projected -46.2436).

The Gauss-Newton / expected Fisher ``J^T S J`` drops the residual term. It is
the Fisher information of the Gaussian data model, is PSD by construction, and
is the object a *forecast* wants: curvature of the likelihood as a function of
theory, evaluated at the fiducial, with no dependence on where the real data
happen to sit.

Which terms qualify
-------------------
==================  ==========================  ============================
term                likelihood object           Gaussian in band powers?
==================  ==========================  ============================
planck_highl        ``clipy.smica.smica_lkl``   YES -- ``-0.5 d^T siginv d``
                                                (``smica.py`` __call__ L305-312)
planck_lowl_tt      ``clipy.gibbs.gibbs_lkl``   NO -- Blackwell-Rao/Gibbs
                                                cl-to-x spline (``cl2x``)
planck_lowl_ee      ``clipy.simall.simall_lkl`` NO -- tabulated per-ell
                                                probability spline (``probEE``)
planck_lensing      ``clipy.lkl._clik_lensing`` YES -- ``-0.5 d^T siginv d``
                                                (``lkl.py`` __call__ L470-484)
act_dr6_lensing     ``candl.likelihood.LensLike`` YES -- ``gaussian_logl`` with
                                                ``covariance``
==================  ==========================  ============================

So the resulting block is a HYBRID: Gauss-Newton for the three Gaussian
band-power terms, observed Hessian for the two non-Gaussian low-ell terms
(for which ``J^T C^-1 J`` is simply not defined).

What clipy exposes (investigated, documented incl. negative results)
--------------------------------------------------------------------
``clipy.lkl.clik_candl`` forwards ``covariance`` / ``data_bandpowers`` /
``window_functions`` / ``spec_order`` / ``effective_ells`` to ``self._internal``
and raises ``NotImplementedError`` when the concrete internal likelihood does not
define them. Measured on the installed clipy 0.15:

* ``planck_highl``  : ``like.covariance`` -> NotImplementedError, BUT
  ``like._internal.siginv`` is the (2289, 2289) INVERSE covariance
  (``criterion_gauss_mat``), and ``like.data_bandpowers`` == ``_internal.rqh_f``
  (2289,) works. So the Gaussian pieces are available -- just as ``S``, not ``C``.
* ``planck_lowl_tt``: ``covariance``, ``data_bandpowers``, ``spec_order``,
  ``effective_ells``, ``bins_*`` ALL raise NotImplementedError. ``_internal.cov``
  (28, 28) exists but is the covariance of the Gibbs ``x`` variable, not of band
  powers -- unusable for ``J^T C^-1 J``.
* ``planck_lowl_ee``: same NotImplementedError set; ``_internal`` carries only
  ``probEE`` (28, 3000) / ``coeffEE`` (28, 2999, 4) spline tables. No covariance
  exists, by construction.
* ``planck_lensing``: ``like.covariance`` -> NotImplementedError, BUT the clik
  object itself carries ``siginv`` (9, 9), ``pp_hat`` (9,), ``bins`` (9, 2501),
  ``cors``, ``cl_fid``, ``cor0``, ``renorm``, ``ren1``.
* ``act_dr6_lensing``: native candl -- ``covariance`` (10, 10),
  ``covariance_chol_dec``, ``data_bandpowers`` (10,) are all present.

Because ``jaxptpolypol.cmb.make_candl_theory_vector_fn`` routes through
``likelihood.get_model_specs`` / ``bin_model_specs`` -- methods the clipy
wrapper does NOT define -- it only works for ``act_dr6_lensing``. The two clipy
Gaussian terms therefore get model-vector closures reconstructed here, mirroring
the clipy ``__call__`` bodies line for line. :func:`validate_gn_term` re-derives
the term's log-likelihood from ``(data, model, siginv, prior)`` and compares it
against ``likelihood.log_like`` both in VALUE at the fiducial and in HESSIAN over
the whole parameter vector, so a drift in clipy's internals shows up as a loud
failure rather than a silently wrong Fisher.
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from jaxptpolypol.cmb import make_candl_theory_vector_fn

#: Terms whose data model is Gaussian in band powers -> Gauss-Newton applies.
GN_TERMS = ("planck_highl", "planck_lensing", "act_dr6_lensing")

#: Terms with a non-Gaussian likelihood -> observed Hessian is the only option.
HESSIAN_TERMS = ("planck_lowl_tt", "planck_lowl_ee")


def _whitener_from_inv_cov(inv_cov):
    """``x -> A^T x`` with ``A A^T = S = C^-1``, so ``|A^T x|^2 == x^T S x``.

    Working in whitened space makes ``F = W^T W`` PSD to machine precision
    instead of only up to the rounding of an explicit ``J^T S J`` product.
    """
    S = np.asarray(inv_cov, dtype=np.float64)
    S = 0.5 * (S + S.T)
    chol = jnp.asarray(np.linalg.cholesky(S))
    return lambda x: chol.T @ x


def _whitener_from_cov_chol(cov_chol_dec):
    """``x -> L^-1 x`` with ``L L^T = C``, so ``|L^-1 x|^2 == x^T C^-1 x``.

    Used where the likelihood ships the covariance rather than its inverse. For
    ACT DR6 lensing the covariance is badly scaled (band powers ~1e-8), and an
    explicit ``inv(C)`` reproduces candl's chi2 only to ~1.4%; the triangular
    solve reproduces it exactly.
    """
    L = jnp.asarray(np.asarray(cov_chol_dec, dtype=np.float64))
    return lambda x: jax.scipy.linalg.solve_triangular(L, x, lower=True)


def _mapping_from_theta(layout, fixed_cmb_params):
    """Packed theta -> flat {parameter name: traced scalar} dict.

    Mirrors ``jaxptpolypol.cmb._build_parameter_mapping`` (cosmology first, then
    fixed, then sampled nuisances), reproduced here so this module stays
    import-only with respect to ``src/jaxptpolypol``.
    """
    fixed = dict(fixed_cmb_params or {})

    def _scalar_or_array(value):
        arr = jnp.asarray(value)
        return arr.reshape(()) if arr.size == 1 else arr

    def mapping_fn(theta):
        cosmo, sampled = layout.unpack(theta)
        mapping = {name: _scalar_or_array(value)
                   for name, value in cosmo.to_dict().items()}
        for source in (fixed, sampled):
            for name, value in source.items():
                mapping[str(name)] = _scalar_or_array(value)
        return mapping

    return mapping_fn


def _theory_dls(likelihood, mapping, pars_to_theory_specs):
    """``pars_to_theory_specs`` + the pp/kk aliasing done inside ``cmb.py``."""
    theory = dict(pars_to_theory_specs(mapping, int(likelihood.ell_max),
                                       int(likelihood.ell_min)))
    if "pp" in theory and "kk" not in theory:
        theory["kk"] = jnp.asarray(theory["pp"]) * (jnp.pi / 2.0)
    if "kk" in theory and "pp" not in theory:
        theory["pp"] = jnp.asarray(theory["kk"]) * (2.0 / jnp.pi)
    return theory


def _clipy_cls_and_tot(likelihood, mapping, pars_to_theory_specs):
    """Reproduce clipy's ``log_like`` preamble: candl Dl dict -> (cls, tot_dict).

    Mirrors ``clik_candl.log_like`` -> ``_clik_common.__call__``
    (``clipy/lkl.py`` L39-50 and L525-529): normalize the candl ``Dl`` payload to
    the clik ``cls`` array, then merge the hard-coded defaults and apply the
    rename map.
    """
    params_with_dls = dict(mapping)
    params_with_dls["Dl"] = _theory_dls(likelihood, mapping, pars_to_theory_specs)
    cls, nuisance_dict = likelihood.normalize_from_candl(params_with_dls)
    tot_dict = dict(nuisance_dict) | dict(likelihood._default)
    for old, new in likelihood.rename_dict.items():
        tot_dict[old] = tot_dict[new]
        del tot_dict[new]
    return cls, tot_dict, params_with_dls


def _make_planck_highl_pieces(likelihood, *, pars_to_theory_specs, layout,
                              fixed_cmb_params):
    """plik TTTEEE: ``-0.5 (rqh_f - model)^T siginv (rqh_f - model)``.

    Model vector reproduced from ``clipy/smica.py`` ``smica_lkl.__call__``
    (L305-312): calibrate the cls, build the binned model covariance ``rq``,
    flatten and select with ``oo``.
    """
    internal = likelihood._internal
    mapping_fn = _mapping_from_theta(layout, fixed_cmb_params)
    do_bin = internal.bins is not None

    def model_fn(theta):
        mapping = mapping_fn(theta)
        cls, tot_dict, _ = _clipy_cls_and_tot(likelihood, mapping,
                                              pars_to_theory_specs)
        cls = internal._calib(cls, tot_dict)
        rq = internal.get_model_rq(cls, tot_dict, do_bin)
        return rq.flatten()[internal.oo]

    return {
        "model_fn": model_fn,
        "whiten": _whitener_from_inv_cov(internal.siginv),
        "data": jnp.asarray(internal.rqh_f),
        "source": "clipy.smica.smica_lkl: _internal.siginv (2289x2289 "
                  "criterion_gauss_mat), _internal.rqh_f",
    }


def _make_planck_lensing_pieces(likelihood, *, pars_to_theory_specs, layout,
                                fixed_cmb_params):
    """Planck lensing: ``-0.5 (pp_hat - bcls)^T siginv (pp_hat - bcls)``.

    Model vector reproduced from ``clipy/lkl.py`` ``_clik_lensing.__call__``
    (L466-484), including the ``cors`` renormalization correction and the
    ``A_planck`` calibration that enters it.
    """
    mapping_fn = _mapping_from_theta(layout, fixed_cmb_params)
    lmax = np.asarray(likelihood.lmax)
    extra_names = list(likelihood.extra_parameter_names)

    def model_fn(theta):
        mapping = mapping_fn(theta)
        cls, tot_dict, _ = _clipy_cls_and_tot(likelihood, mapping,
                                              pars_to_theory_specs)
        calib = jnp.asarray(1.0)
        if len(extra_names) == 1:
            a = jnp.asarray(tot_dict["A_planck"])
            calib = 1.0 / (a * a)
        vls = jnp.concatenate([cls[i] for i in range(7) if lmax[i] != -1])
        bcls = jnp.dot(likelihood.bins, cls[0] * likelihood._m_llp1_2) \
            - likelihood.cor0
        if likelihood.cors is not None:
            fpars = vls if likelihood.renorm != 0 else likelihood.cl_fid
            fphi = cls[0] if likelihood.ren1 != 0 \
                else likelihood.cl_fid[:lmax[0] + 1]
            curfid = jnp.concatenate([
                fphi * likelihood._m_llp1_2,
                fpars[lmax[0] + 1:] * calib * likelihood._m_llp1,
            ])
            bcls = bcls + likelihood.cors @ curfid
        return bcls

    return {
        "model_fn": model_fn,
        "whiten": _whitener_from_inv_cov(likelihood.siginv),
        "data": jnp.asarray(likelihood.pp_hat),
        "source": "clipy.lkl._clik_lensing: clik.siginv (9x9), clik.pp_hat",
    }


def _make_candl_pieces(likelihood, *, pars_to_theory_specs, layout,
                       fixed_cmb_params):
    """Native candl (ACT DR6 lensing): ``make_candl_theory_vector_fn`` + ``covariance``."""
    model_fn = make_candl_theory_vector_fn(
        likelihood,
        pars_to_theory_specs=pars_to_theory_specs,
        layout=layout,
        fixed_cmb_params=fixed_cmb_params,
    )
    return {
        "model_fn": model_fn,
        "whiten": _whitener_from_cov_chol(likelihood.covariance_chol_dec),
        "data": jnp.asarray(likelihood._data_bandpowers),
        "source": "candl.likelihood.LensLike: covariance_chol_dec (10x10), "
                  "_data_bandpowers",
    }


_BUILDERS = {
    "planck_highl": _make_planck_highl_pieces,
    "planck_lensing": _make_planck_lensing_pieces,
    "act_dr6_lensing": _make_candl_pieces,
}


def make_prior_loglike_fn(likelihood, *, layout, fixed_cmb_params):
    """Closure over the term's INTERNAL nuisance priors only.

    clipy adds ``self.prior(nuisance_dict)`` on top of the Gaussian chi2 inside
    ``_clik_common.__call__``; candl adds ``self.prior_logl(pars)`` inside
    ``log_like``. Under Gauss-Newton the chi2 part is replaced by ``J^T S J`` and
    the prior part has to be re-added explicitly (it is exactly the same
    Gaussian-prior curvature the observed-Hessian build absorbs implicitly).
    """
    mapping_fn = _mapping_from_theta(layout, fixed_cmb_params)
    is_candl = hasattr(likelihood, "prior_logl")

    def prior_loglike_fn(theta):
        mapping = mapping_fn(theta)
        if is_candl:
            # candl's log_like returns -(gaussian_logl + prior_logl) with both
            # pieces defined POSITIVE (chi2/2), hence the sign flip here.
            # With no internal priors candl returns an integer 0, which jax
            # refuses to differentiate -- force float64.
            return -jnp.asarray(likelihood.prior_logl(mapping), dtype=jnp.float64)
        return jnp.asarray(likelihood.prior(mapping), dtype=jnp.float64)

    return prior_loglike_fn


def make_gn_pieces(term_name, likelihood, *, pars_to_theory_specs, layout,
                   fixed_cmb_params=None):
    """Gauss-Newton ingredients for one term, or ``None`` if not Gaussian."""
    builder = _BUILDERS.get(term_name)
    if builder is None:
        return None
    pieces = builder(likelihood, pars_to_theory_specs=pars_to_theory_specs,
                     layout=layout, fixed_cmb_params=fixed_cmb_params)
    pieces["prior_loglike_fn"] = make_prior_loglike_fn(
        likelihood, layout=layout, fixed_cmb_params=fixed_cmb_params)
    return pieces


def gaussian_loglike_from_pieces(pieces):
    """``theta -> -0.5 |W (d - m(theta))|^2`` (no priors)."""
    data, whiten, model_fn = pieces["data"], pieces["whiten"], pieces["model_fn"]

    def gaussian_loglike(theta):
        r = whiten(data - model_fn(theta))
        return -0.5 * (r @ r)

    return gaussian_loglike


def gn_fisher(pieces, theta):
    """``J^T C^-1 J`` (+ the term's internal nuisance-prior curvature).

    Computed as ``W^T W`` with ``W = whiten(J)``, so the Gauss-Newton part is
    PSD to machine precision. The nuisance-prior part is the (Gaussian, hence
    also PSD) curvature the observed-Hessian build gets for free because clipy /
    candl fold the priors into ``log_like``.
    """
    jac = jnp.asarray(jax.jacfwd(pieces["model_fn"])(theta))
    w = pieces["whiten"](jac)
    fisher = w.T @ w
    prior_hess = jnp.asarray(jax.hessian(pieces["prior_loglike_fn"])(theta))
    fisher = fisher - 0.5 * (prior_hess + prior_hess.T)
    return np.asarray(0.5 * (fisher + fisher.T))


def validate_gn_term(term_name, pieces, term_loglike_fn, theta, *,
                     value_rtol=1e-8, hess_rtol=1e-6):
    """Prove the reconstructed Gaussian form IS the term's likelihood.

    Two checks, both against the untouched ``jaxptpolypol`` log-likelihood
    closure the observed-Hessian build uses:

    1. VALUE at the fiducial: ``gaussian + prior == log_like``;
    2. HESSIAN over the whole packed parameter vector: ``d2(gaussian + prior) ==
       d2 log_like``. This is the strong one -- it holds only if the
       reconstructed model vector is the right FUNCTION of theta, not merely the
       right number at one point.

    Returns a dict of the measured discrepancies; raises ``AssertionError`` on
    failure.
    """
    gauss_fn = gaussian_loglike_from_pieces(pieces)
    prior_fn = pieces["prior_loglike_fn"]

    def total_fn(th):
        return gauss_fn(th) + prior_fn(th)

    ref_val = float(term_loglike_fn(theta))
    got_val = float(total_fn(theta))
    val_err = abs(got_val - ref_val) / max(abs(ref_val), 1.0)

    ref_h = np.asarray(jax.jit(jax.hessian(term_loglike_fn))(theta))
    got_h = np.asarray(jax.jit(jax.hessian(total_fn))(theta))
    scale = max(np.abs(ref_h).max(), 1.0)
    hess_err = float(np.abs(got_h - ref_h).max() / scale)

    r = np.asarray(pieces["whiten"](pieces["data"] - pieces["model_fn"](theta)))
    chi2_resid = float(r @ r)

    report = {
        "term": term_name,
        "loglike_ref": ref_val,
        "loglike_reconstructed": got_val,
        "value_rel_err": val_err,
        "hessian_max_rel_err": hess_err,
        "n_data": int(np.asarray(pieces["data"]).shape[0]),
        "chi2_residual_at_fiducial": chi2_resid,
    }
    assert val_err < value_rtol, (
        f"{term_name}: reconstructed Gaussian log-like {got_val!r} != "
        f"candl/clipy log_like {ref_val!r} (rel err {val_err:.3g})")
    assert hess_err < hess_rtol, (
        f"{term_name}: reconstructed Hessian differs from the candl/clipy "
        f"Hessian by {hess_err:.3g} (relative to max |H| = {scale:.3g})")
    return report
