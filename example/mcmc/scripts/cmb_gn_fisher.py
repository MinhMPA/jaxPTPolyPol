#!/usr/bin/env python3
r"""Gauss-Newton (expected) Fisher pieces for the CMB likelihood terms.

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

#: Bumped whenever the Gauss-Newton construction changes in a way that alters
#: the produced Fisher block. Feeds the artifact fingerprint.
GN_ALGORITHM_VERSION = "1.0"


class GNValidationError(RuntimeError):
    """A reconstructed Gaussian form does not reproduce the candl/clipy term.

    A dedicated exception rather than ``assert``: assertions are stripped by
    ``python -O`` / ``PYTHONOPTIMIZE=1``, which would silently disable the only
    check standing between a drifted clipy internal and a wrong Fisher block.
    """


class SharedPriorError(RuntimeError):
    """A likelihood's internal priors could not be inventoried analytically."""


class UnknownCmbTermError(RuntimeError):
    """A term name belongs to neither ``GN_TERMS`` nor ``HESSIAN_TERMS``."""


def _whitener_from_inv_cov(inv_cov):
    """``x -> A^T x`` with ``A A^T = S = C^-1``, so ``|A^T x|^2 == x^T S x``.

    Working in whitened space makes ``F = W^T W`` PSD to machine precision
    instead of only up to the rounding of an explicit ``J^T S J`` product.
    """
    S = np.asarray(inv_cov, dtype=np.float64)
    scale = max(float(np.abs(S).max()), 1.0)
    asym = float(np.abs(S - S.T).max()) / scale
    if asym > 1e-8:
        raise SharedPriorError(
            f"inverse covariance is not symmetric (max |S - S^T| / max |S| = "
            f"{asym:.3g} > 1e-8); silently symmetrizing it would hide a "
            "corrupted or mis-shaped covariance")
    chol = jnp.asarray(np.linalg.cholesky(0.5 * (S + S.T)))
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
    return cls, tot_dict


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
        cls, tot_dict = _clipy_cls_and_tot(likelihood, mapping,
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
        cls, tot_dict = _clipy_cls_and_tot(likelihood, mapping,
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


def make_prior_loglike_fn(likelihood, *, layout, fixed_cmb_params=None):
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
    """Gauss-Newton ingredients for one term, or ``None`` for a low-ell term.

    ``None`` means "this term is on the observed-Hessian path", and it is
    returned ONLY for the names in :data:`HESSIAN_TERMS`. An unrecognised term
    name raises :class:`UnknownCmbTermError` -- previously it fell through to
    ``None`` and silently took the observed-Hessian path, so a typo'd or newly
    added term would have been Fishered by the wrong method without a word.
    """
    builder = _BUILDERS.get(term_name)
    if builder is None:
        if term_name in HESSIAN_TERMS:
            return None
        raise UnknownCmbTermError(
            f"unknown CMB term {term_name!r}: it is in neither GN_TERMS "
            f"{GN_TERMS} nor HESSIAN_TERMS {HESSIAN_TERMS}. Classify it "
            "explicitly -- a new term must not inherit the observed-Hessian "
            "path by default.")
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
                     value_rtol=1e-8, hess_rtol=1e-12,
                     directional_rtol=0.01):
    """Prove the reconstructed Gaussian form IS the term's likelihood.

    Three checks, all against the untouched ``jaxptpolypol`` log-likelihood
    closure the observed-Hessian build uses:

    1. VALUE at the fiducial: ``gaussian + prior == log_like``;
    2. HESSIAN over the whole packed parameter vector: ``d2(gaussian + prior) ==
       d2 log_like``, as a max-abs error relative to ``max |H_ref|``;
    3. DIRECTIONAL, along the minimum-eigenvalue direction ``v`` of the
       reference Fisher ``F_ref = -0.5 (H_ref + H_ref^T)``:
       ``|v^T (F_got - F_ref) v| < directional_rtol * |lambda_min_ref|``.

    Why (3) exists. Check (2) is normalized by ``max |H_ref| ~ 2e8``, so even a
    tight-looking ``hess_rtol`` buys a generous ABSOLUTE budget. The quantity
    this whole module exists to get right is a near-null eigenvalue of order
    1e-1 to 1e-2. A relative-to-the-largest-eigenvalue test is blind to it by
    construction. Check (3) measures the error exactly where it matters, in the
    units it matters in: a fraction of the smallest eigenvalue of the very
    matrix being reconstructed. ``hess_rtol`` is 1e-12 (was 1e-6, which admitted
    absolute errors ~2e2, i.e. ~400x the eigenvalue at stake).

    Returns a dict of the measured discrepancies; raises
    :class:`GNValidationError` on failure. NOT ``assert`` -- assertions are
    stripped under ``python -O`` / ``PYTHONOPTIMIZE=1``, which would turn this
    gate into a no-op exactly when someone runs the build "for speed".
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

    # Fisher convention for the directional check: symmetrization is what makes
    # v^T dF v == -v^T dH v, so the two conventions agree up to sign.
    f_ref = -0.5 * (ref_h + ref_h.T)
    f_got = -0.5 * (got_h + got_h.T)
    eigvals, eigvecs = np.linalg.eigh(f_ref)
    v = eigvecs[:, 0]
    lam_min_ref = float(eigvals[0])
    dir_err = float(abs(v @ (f_got - f_ref) @ v))
    dir_budget = directional_rtol * abs(lam_min_ref)

    r = np.asarray(pieces["whiten"](pieces["data"] - pieces["model_fn"](theta)))
    chi2_resid = float(r @ r)

    report = {
        "term": term_name,
        "loglike_ref": ref_val,
        "loglike_reconstructed": got_val,
        "value_rel_err": val_err,
        "hessian_max_rel_err": hess_err,
        "hessian_max_abs_err": float(np.abs(got_h - ref_h).max()),
        "min_eig_ref": lam_min_ref,
        "directional_abs_err": dir_err,
        "directional_budget": dir_budget,
        "n_data": int(np.asarray(pieces["data"]).shape[0]),
        "chi2_residual_at_fiducial": chi2_resid,
    }
    if not val_err < value_rtol:
        raise GNValidationError(
            f"{term_name}: reconstructed Gaussian log-like {got_val!r} != "
            f"candl/clipy log_like {ref_val!r} (rel err {val_err:.3g} >= "
            f"{value_rtol:.3g})")
    if not hess_err < hess_rtol:
        raise GNValidationError(
            f"{term_name}: reconstructed Hessian differs from the candl/clipy "
            f"Hessian by {hess_err:.3g} relative to max |H| = {scale:.3g} "
            f"(>= {hess_rtol:.3g})")
    if not dir_err < dir_budget:
        raise GNValidationError(
            f"{term_name}: reconstructed Fisher differs from the candl/clipy "
            f"Fisher by {dir_err:.6g} along the reference minimum-eigenvalue "
            f"direction, which is >= {directional_rtol:.3g} of "
            f"|lambda_min_ref| = {abs(lam_min_ref):.6g} (budget "
            f"{dir_budget:.6g}). The near-null direction is exactly what this "
            "block is built to get right.")
    return report


# ---------------------------------------------------------------------------
# Shared internal priors: inventory and duplicate-curvature removal.
# ---------------------------------------------------------------------------

def _enumerate_term_priors(term_name, likelihood, *, layout, fixed_cmb_params):
    """``[(key, prior_loglike_fn(theta))]`` for every internal prior of a term.

    ``key`` is a parameter name, or a tuple of names for clipy's joint priors.
    The returned closures mirror ``clipy._clik_common.prior`` term by term, so
    their sum is bit-for-bit the term's total prior log-likelihood -- which is
    what :func:`inventory_shared_priors` then asserts.

    candl likelihoods with a NON-EMPTY ``priors`` list raise: candl prior
    objects are not enumerated here, and silently skipping them would under-count
    the inventory. (ACT DR6 lens-only has ``priors == []``, and the
    sum-consistency check below proves it.)
    """
    mapping_fn = _mapping_from_theta(layout, fixed_cmb_params)
    clipy_priors = getattr(likelihood, "_prior", None)
    if clipy_priors is None:
        candl_priors = list(getattr(likelihood, "priors", []))
        if candl_priors:
            raise SharedPriorError(
                f"{term_name}: candl likelihood carries {len(candl_priors)} "
                "internal prior object(s), which this inventory cannot read "
                "analytically. Enumerate them explicitly before building.")
        return []

    entries = []
    for key, prior_fn in clipy_priors.items():
        names = key if isinstance(key, tuple) else (key,)

        def one_prior_loglike(theta, _names=names, _key=key, _fn=prior_fn):
            mapping = mapping_fn(theta)
            if isinstance(_key, tuple):
                value = jnp.array([mapping[n] for n in _names])
            else:
                value = jnp.asarray(mapping[_names[0]])
            return jnp.asarray(_fn(value), dtype=jnp.float64).reshape(())

        entries.append((key, one_prior_loglike))
    return entries


def _packed_names(layout):
    """Packed-vector index -> parameter name (requires scalar cosmo entries)."""
    if any(int(s) != 1 for s in layout.cosmo_sizes):
        raise SharedPriorError(
            f"non-scalar cosmology entries in the layout ({layout.cosmo_sizes}); "
            "the shared-prior inventory addresses parameters by packed index "
            "and needs a one-slot-per-name layout")
    names = list(layout.cosmo_keys) + [""] * (
        layout.nuisance_offset - layout.n_cosmo)
    names += list(layout.cmb_nuisance_names)
    return names


def _prior_curvature(prior_loglike_fn, theta):
    """``-0.5 (H + H^T)`` of a prior log-likelihood over the packed vector."""
    h = np.asarray(jax.hessian(prior_loglike_fn)(theta))
    return -0.5 * (h + h.T)


def inventory_shared_priors(likelihoods, *, layout, theta_fid,
                            fixed_cmb_params=None, prior_gap_rtol=1e-8,
                            rtol=1e-10):
    """Locate every internal prior and flag the ones counted more than once.

    Each of the four Planck ``.clik`` likelihoods is loaded with
    ``all_priors=True``, so each one folds the SAME Gaussian ``A_planck``
    calibration prior into its own ``log_like``. Summing the five per-term
    Fisher blocks therefore counts that one prior four times. This function
    finds such priors; :func:`duplicate_prior_curvature` builds the correction.

    Widths are never hardcoded: each prior's curvature is obtained by
    differentiating the likelihood object's OWN prior callable, so it is the
    effective curvature actually entering the sum. Two hard checks make that
    trustworthy:

    * COMPLETENESS -- for every term, the sum of the enumerated per-prior
      curvatures must equal the curvature of the term's full prior
      log-likelihood. A prior this function failed to enumerate cannot hide.
    * GAUSSIANITY -- each prior's curvature must be independent of where it is
      evaluated, checked by re-evaluating at a point perturbed along that
      prior's OWN parameters. A non-quadratic prior has no single width and is
      refused rather than linearized.

    Both tolerances are RELATIVE, keyword-only, and scale-free: each measured
    gap is divided by the corresponding curvature span before comparison.
    ``prior_gap_rtol`` bounds the COMPLETENESS residual (enumerated sum vs the
    term's total prior curvature); ``rtol`` bounds the GAUSSIANITY drift and the
    term-to-term width agreement. ``prior_gap_rtol`` was called ``atol`` until
    2026-08-08, which misdescribed it -- nothing has ever passed it explicitly.

    Returns ``{parameter_name: {"sigma", "curvature", "count", "terms",
    "packed_index"}}`` for the parameters whose prior appears in more than one
    term. Raises :class:`SharedPriorError` on anything it cannot resolve.
    """
    names = _packed_names(layout)
    theta_fid = np.asarray(theta_fid, dtype=np.float64)
    per_key = {}

    for term_name, likelihood in likelihoods.items():
        entries = _enumerate_term_priors(
            term_name, likelihood, layout=layout,
            fixed_cmb_params=fixed_cmb_params)
        total = np.zeros((theta_fid.size, theta_fid.size))
        for key, fn in entries:
            curv = _prior_curvature(fn, jnp.asarray(theta_fid))
            key_names = key if isinstance(key, tuple) else (key,)
            try:
                idx = [names.index(n) for n in key_names]
            except ValueError as exc:
                raise SharedPriorError(
                    f"{term_name}: prior parameter {key!r} is not in the "
                    "packed layout, so its curvature cannot be located") from exc
            # Gaussianity: perturb along this prior's OWN parameters and require
            # the curvature to be unchanged.
            probe = theta_fid.copy()
            for i in idx:
                probe[i] += 0.05 * max(abs(theta_fid[i]), 1.0)
            curv_probe = _prior_curvature(fn, jnp.asarray(probe))
            drift = float(np.abs(curv_probe - curv).max())
            span = max(float(np.abs(curv).max()), 1.0)
            if drift / span > rtol:
                raise SharedPriorError(
                    f"{term_name}: prior on {key!r} is not Gaussian -- its "
                    f"curvature moves by {drift:.6g} (relative {drift / span:.3g}"
                    f" > {rtol:.3g}) when its own parameters are perturbed. A "
                    "non-quadratic prior has no single width to subtract.")
            total += curv
            per_key.setdefault(key, []).append((term_name, curv))

        reference = _prior_curvature(
            make_prior_loglike_fn(likelihood, layout=layout,
                                  fixed_cmb_params=fixed_cmb_params),
            jnp.asarray(theta_fid))
        span = max(float(np.abs(reference).max()), 1.0)
        gap = float(np.abs(total - reference).max())
        if gap / span > prior_gap_rtol:
            raise SharedPriorError(
                f"{term_name}: the enumerated per-prior curvatures do not sum "
                f"to the term's total prior curvature (max gap {gap:.6g}, "
                f"relative {gap / span:.3g} > {prior_gap_rtol:.3g}). Some "
                "internal prior was not located; refusing to deduplicate "
                "against an incomplete inventory.")

    inventory = {}
    for key, hits in per_key.items():
        if len(hits) < 2:
            continue
        if isinstance(key, tuple):
            raise SharedPriorError(
                f"joint prior {key!r} is shared by {len(hits)} terms "
                f"({[t for t, _ in hits]}); a joint prior has no scalar width "
                "and this inventory refuses to deduplicate it blindly.")
        base = hits[0][1]
        for term_name, curv in hits[1:]:
            gap = float(np.abs(curv - base).max())
            span = max(float(np.abs(base).max()), 1.0)
            if gap / span > rtol:
                raise SharedPriorError(
                    f"prior on {key!r} differs between {hits[0][0]!r} and "
                    f"{term_name!r} (max curvature gap {gap:.6g}, relative "
                    f"{gap / span:.3g} > {rtol:.3g}). These are not the same "
                    "prior; deduplicating them would delete real information.")
        index = names.index(key)
        support = base.copy()
        support[index, index] = 0.0
        if float(np.abs(support).max()) > 0.0:
            raise SharedPriorError(
                f"prior on {key!r} has curvature outside its own diagonal "
                "entry; it couples parameters and cannot be subtracted as a "
                "scalar width.")
        curvature = float(base[index, index])
        if not curvature > 0.0:
            raise SharedPriorError(
                f"prior on {key!r} has non-positive curvature {curvature!r}; "
                "no width can be derived from it.")
        inventory[key] = {
            "sigma": float(1.0 / np.sqrt(curvature)),
            "curvature": curvature,
            "count": len(hits),
            "terms": [t for t, _ in hits],
            "packed_index": int(index),
        }
    return inventory


def duplicate_prior_curvature(inventory, size):
    """``sum_p (count_p - 1) * curvature_p * e_p e_p^T`` -- what to subtract.

    Subtracting AFTER summation is exact, not an approximation: a Gaussian
    prior's Hessian is a constant matrix, so each of the ``count`` copies
    contributes exactly ``curvature * e_p e_p^T`` to the summed block --
    explicitly for the Gauss-Newton terms (which add the prior Hessian by hand)
    and identically for the observed-Hessian terms (whose Hessian contains that
    same constant). Removing ``count - 1`` copies restores single counting with
    no residual.
    """
    correction = np.zeros((int(size), int(size)))
    for entry in inventory.values():
        i = entry["packed_index"]
        correction[i, i] += (entry["count"] - 1) * entry["curvature"]
    return correction
