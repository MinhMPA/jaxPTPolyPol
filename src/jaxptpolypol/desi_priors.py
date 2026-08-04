"""DESI DR1-reanalysis (arXiv:2511.20757 Table I) prior spec machinery.

Layer-1 (constant coefficient-convention map) values are stored mapped AND
validated: each entry carries the verbatim paper value, the map factor (+
affine offset), and the our-convention value; loading raises unless they
reconcile. Layer-2 (theta_NL-dependent A_AP * A_amp rescaling, Table I
footnote) is applied at runtime by make_desi_prior_fns. Convention-map
provenance: docs/design/desi-convention-map.md. See CONTEXT.md
"Stream-B decisions (grill session 2026-07-30)".

Branch stream-b-sigmap (Amendment 1, 2026-07-31): the c0/c2/c4 counterterm
rows carry an optional ``ctr_rotation: "multipole_to_tilde"`` token. The
paper's diagonal priors live in the CLASS-PT per-multipole basis (Eqs
2.21-2.23) while our code coefficients are the mu-space tilde basis (Eq
2.15); the exact per-bin f-dependent correlated prior is assembled at runtime
from L(f) (map section 3.1). The trio must carry the token all-or-none.
"""
from __future__ import annotations

import importlib.resources
from dataclasses import dataclass

import yaml

__all__ = [
    "DesiPriorSpec", "MarginalRow", "SampledRow", "SpecValidationError",
    "load_desi_prior_spec",
]

_RESCALE_TOKENS = ("none", "A_AP", "A_AP*A_amp", "A_AP*A_amp^2")
_FACTOR_FORMULAS = (None, "knl_over_0p45_sq")
_MEAN_FORMULAS = (None, "coevolution_bGamma3")
_CTR_ROTATIONS = (None, "multipole_to_tilde")
_SAMPLED_RESCALE = ("none", "sigma8_sq")
_B1_MEASURES = ("raw", "b1sigma8")
_PHASES = ("forecast", "real_data", "nulcdm")
_CTR_TRIO = (("pk", "ctr", "c0"), ("pk", "ctr", "c2"), ("pk", "ctr", "c4"))
_RECONCILE_RTOL = 1e-12


class SpecValidationError(ValueError):
    """Raised when a spec entry fails load-time reconciliation."""


@dataclass(frozen=True)
class MarginalRow:
    paper_mean: float | None
    paper_sigma: float
    paper_units: str
    paper_variable: str
    factor: float
    offset: float
    mean: float | None
    sigma: float
    rescale: str
    factor_formula: str | None
    mean_formula: str | None
    ctr_rotation: str | None = None


@dataclass(frozen=True)
class SampledRow:
    kind: str
    paper_mean: float | None = None
    paper_sigma: float | None = None
    paper_variable: str | None = None
    rescale: str = "none"
    #: b1 only -- which coordinate the flat prior is flat IN. "raw": flat in
    #: raw b1 (project default; differs from the paper's measure by the
    #: cosmology-dependent weight prod_b sigma8(z_b) -- see CONTEXT.md
    #: deviation 3). "b1sigma8": flat in y = b1*sigma8(z) on
    #: [paper_lower, paper_upper], the Table-I measure (adds the Jacobian
    #: sum_b log sigma8 and the bounds to log_prior_nl_fn).
    measure: str = "raw"
    paper_lower: float | None = None
    paper_upper: float | None = None


@dataclass(frozen=True)
class DesiPriorSpec:
    metadata: dict
    marginalized: dict
    sampled: dict


def _close(a, b):
    return abs(a - b) <= _RECONCILE_RTOL * max(1.0, abs(a), abs(b))


def _validate_row(key, row):
    if row.rescale not in _RESCALE_TOKENS:
        raise SpecValidationError(
            f"{key}: unknown rescale token {row.rescale!r}")
    if row.factor_formula not in _FACTOR_FORMULAS:
        raise SpecValidationError(
            f"{key}: unknown factor_formula {row.factor_formula!r}")
    if row.mean_formula not in _MEAN_FORMULAS:
        raise SpecValidationError(
            f"{key}: unknown mean_formula {row.mean_formula!r}")
    if row.ctr_rotation not in _CTR_ROTATIONS:
        raise SpecValidationError(
            f"{key}: unknown ctr_rotation {row.ctr_rotation!r}")
    if row.paper_sigma is None or row.paper_sigma <= 0.0:
        raise SpecValidationError(f"{key}: paper_sigma must be positive")
    if not _close(row.sigma, row.paper_sigma * abs(row.factor)):
        raise SpecValidationError(
            f"{key}: sigma {row.sigma} != paper_sigma*|factor| "
            f"{row.paper_sigma * abs(row.factor)}")
    if row.mean_formula is None:
        if row.paper_mean is None or row.mean is None:
            raise SpecValidationError(
                f"{key}: numeric mean required when mean_formula is null")
        if not _close(row.mean, row.paper_mean * row.factor + row.offset):
            raise SpecValidationError(
                f"{key}: mean {row.mean} != paper_mean*factor+offset "
                f"{row.paper_mean * row.factor + row.offset}")
    else:
        if row.mean is not None or row.paper_mean is not None:
            raise SpecValidationError(
                f"{key}: mean/paper_mean must be null with mean_formula")


def load_desi_prior_spec(name_or_path="desi_dr1_reanalysis_2511_20757",
                         phase="forecast"):
    """Load and validate a DESI prior spec (packaged name or explicit path)."""
    from .marginal_likelihood import LIN_SURVEY_KEYS

    path = str(name_or_path)
    if not path.endswith((".yaml", ".yml")):
        ref = importlib.resources.files("jaxptpolypol.data") / "priors" / f"{path}.yaml"
        raw = yaml.safe_load(ref.read_text())
    else:
        with open(path) as fh:
            raw = yaml.safe_load(fh)

    marginalized = {}
    for dotted, entry in raw["marginalized"].items():
        key = tuple(dotted.split("."))
        if len(key) != 3:
            raise SpecValidationError(f"bad marginalized key {dotted!r}")
        row = MarginalRow(**entry)
        _validate_row(dotted, row)
        marginalized[key] = row

    expected = set(LIN_SURVEY_KEYS)
    got = set(marginalized)
    if got != expected:
        missing = sorted(".".join(k) for k in expected - got)
        extra = sorted(".".join(k) for k in got - expected)
        raise SpecValidationError(
            f"marginalized keys mismatch: missing={missing} extra={extra}")

    trio_flags = [marginalized[k].ctr_rotation for k in _CTR_TRIO]
    if any(v is not None for v in trio_flags) and not all(
            v is not None for v in trio_flags):
        raise SpecValidationError(
            "ctr_rotation must be set on all of pk.ctr.c0/c2/c4 or none; "
            f"got {dict(zip(('c0', 'c2', 'c4'), trio_flags))}")
    if all(v is not None for v in trio_flags):
        # Cov-mode (make_desi_prior_fns) reuses the c0-slot rescale R_ctr for the
        # whole 3x3 ctr block, so the trio must share one rescale token.
        trio_rescales = {".".join(k): marginalized[k].rescale for k in _CTR_TRIO}
        if len(set(trio_rescales.values())) != 1:
            raise SpecValidationError(
                "ctr_rotation trio pk.ctr.c0/c2/c4 must share one rescale token "
                "(cov-mode reuses a single R_ctr for the whole ctr block); got "
                f"{trio_rescales}")

    sampled = {}
    for name, entry in raw["sampled"].items():
        row = SampledRow(**entry)
        if row.kind not in ("flat", "gaussian"):
            raise SpecValidationError(f"sampled {name}: bad kind {row.kind!r}")
        if row.rescale not in _SAMPLED_RESCALE:
            raise SpecValidationError(
                f"sampled {name}: unknown rescale {row.rescale!r}")
        if row.measure not in _B1_MEASURES:
            raise SpecValidationError(
                f"sampled {name}: unknown measure {row.measure!r} "
                f"(allowed: {_B1_MEASURES})")
        if name != "b1" and row.measure != "raw":
            raise SpecValidationError(
                f"sampled {name}: 'measure' applies to only the b1 row")
        if row.measure == "b1sigma8":
            if row.paper_lower is None or row.paper_upper is None:
                raise SpecValidationError(
                    "sampled b1: measure=b1sigma8 requires numeric "
                    "paper_lower and paper_upper")
            if not row.paper_lower < row.paper_upper:
                raise SpecValidationError(
                    "sampled b1: paper_lower must be < paper_upper")
        if row.kind == "gaussian" and (row.paper_sigma is None
                                       or row.paper_sigma <= 0.0):
            raise SpecValidationError(
                f"sampled {name}: gaussian needs positive paper_sigma")
        sampled[name] = row
    for required in ("b1", "b2", "bG2"):
        if required not in sampled:
            raise SpecValidationError(f"sampled block missing {required!r}")

    if phase not in _PHASES:
        raise SpecValidationError(f"unknown phase {phase!r} (allowed: {_PHASES})")
    if phase != "forecast" and sampled["b1"].measure == "raw":
        raise SpecValidationError(
            f"phase={phase!r} requires the Table-I b1 measure: set the spec's "
            "b1 row to measure: b1sigma8 (raw-b1 flat differs from "
            "arXiv:2511.20757 by the prod_b sigma8(z_b) prior weight, which "
            "lands on Sum m_nu in nuLCDM -- see CONTEXT.md deviation 3)")

    return DesiPriorSpec(metadata=raw.get("metadata", {}),
                         marginalized=marginalized, sampled=sampled)


# =============================================================================
# theta_NL-dependent prior functions (Task 5sigma: base factory + cov-mode)
# =============================================================================

import jax.numpy as jnp  # noqa: E402

from .marginal_likelihood import LIN_SURVEY_KEYS  # noqa: E402

__all__ += ["make_desi_prior_fns", "make_lcdm_rescaling_fns",
            "DESI_F_FID", "ctr_rotation_matrices", "rotate_taylor_templates"]

_LOG2PI = 1.8378770664093453

#: Fiducial linear growth rate f(z_b) at the production fiducial
#: (ombh2=0.02242, omch2=0.11933, h=0.6766, mnu=0.06) for
#: z_bins = (0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.2). From the growth ODE; see
#: docs/design/desi-convention-map.md section 3.1 (the L(f) rotation table).
#: The tuple GOVERNS (cross-branch identity of L is what the equivalence test
#: needs); growth_rate_approx reproduces it within ~6e-4 (checked in tests).
DESI_F_FID = (0.8155, 0.8579, 0.8893, 0.9126, 0.9301, 0.9489, 0.9649)


def ctr_rotation_matrices(f_bins):
    """Stacked upper-triangular counterterm basis maps ``L(f)``, one per bin.

    ``L(f) = [[1, -f/3, 3 f^2/35], [0, 1, -6 f/7], [0, 0, 1]]`` sends the
    paper's per-multipole ``(c0, c2, c4)`` prior basis to our mu-space tilde
    basis (``c_tilde = L(f) . c_paper``); see
    docs/design/desi-convention-map.md section 3.1. Returns shape
    ``(n_bins, 3, 3)`` float64.
    """
    f = jnp.asarray(f_bins, dtype=jnp.float64)
    if f.ndim != 1:
        raise ValueError(f"f_bins must be 1-d, got shape {f.shape}")
    n = f.shape[0]
    L = jnp.zeros((n, 3, 3), dtype=jnp.float64)
    L = L.at[:, 0, 0].set(1.0)
    L = L.at[:, 0, 1].set(-f / 3.0)
    L = L.at[:, 0, 2].set(3.0 * f**2 / 35.0)
    L = L.at[:, 1, 1].set(1.0)
    L = L.at[:, 1, 2].set(-6.0 * f / 7.0)
    L = L.at[:, 2, 2].set(1.0)
    return L


# ctr slot positions (c0, c2, c4) within a bin's 11 theta_lin entries, per
# LIN_SURVEY_KEYS (bGamma3, P_shot, c0, c2, c4, cfog, a0, a2, c1, B_shot, A_shot).
_CTR_COLS = (2, 3, 4)


def rotate_taylor_templates(tt, L_bins):
    """Right-multiply the ctr columns of a :class:`TaylorTemplates` by ``L_bins``.

    Returns a NEW :class:`~jaxptpolypol.marginal_taylor.TaylorTemplates` in which,
    for each bin ``b``, the ctr columns ``(2, 3, 4)`` of ``bin_M0[b]`` (shape
    ``(n_b, p_b)``) and of ``bin_dM[b]`` (shape ``(n_b, p_b, d)``) are
    right-multiplied by ``L_bins[b]`` (``(3, 3)``):

        M0'[:, cols]      = M0[:, cols] @ L_b
        dM'[:, cols, k]   = dM[:, cols, k] @ L_b     for every theta_NL index k

    i.e. the transform acts on axis 1 of ``bin_dM`` -- the theta_lin (linear
    parameter) axis, the axis matched to ``M0``'s columns in
    ``M = M0 + einsum('ijk,k->ij', dM, delta)`` -- NOT the theta_NL axis (axis 2).
    Because the reconstruction is linear in ``delta``, this makes the surrogate
    template ``M'(delta) @ theta`` equal ``M(delta) @ theta'`` with
    ``theta'[cols] = L_b @ theta[cols]`` (an exact linear reparameterization of
    the ctr slots). ``m0``/``J``/``H``/``theta0`` are passed through untouched --
    they live at ``theta_lin = 0``, which ``L`` fixes.
    """
    from .marginal_taylor import TaylorTemplates

    L_bins = jnp.asarray(L_bins, dtype=jnp.float64)
    n_bins = len(tt.bin_M0)
    if L_bins.shape != (n_bins, 3, 3):
        raise ValueError(
            f"L_bins must have shape {(n_bins, 3, 3)}, got {L_bins.shape}")
    cols = jnp.asarray(_CTR_COLS)

    new_M0, new_dM = [], []
    for b in range(n_bins):
        L_b = L_bins[b]                                    # (3, 3)
        M0 = tt.bin_M0[b]                                  # (n_b, p_b)
        new_M0.append(M0.at[:, cols].set(M0[:, cols] @ L_b))
        dM = tt.bin_dM[b]                                  # (n_b, p_b, d)
        # dM[:, cols, :] is (n_b, 3, d); right-multiply the ctr (col) axis by L_b:
        # out[i, o, k] = sum_a dM[i, cols[a], k] * L_b[a, o].
        rotated = jnp.einsum("iak,ao->iok", dM[:, cols, :], L_b)
        new_dM.append(dM.at[:, cols, :].set(rotated))

    return TaylorTemplates(
        theta0=tt.theta0,
        bin_m00=tt.bin_m00,
        bin_J=tt.bin_J,
        bin_H=tt.bin_H,
        bin_M0=tuple(new_M0),
        bin_dM=tuple(new_dM),
        order2_m0=tt.order2_m0,
        build_diagnostics=tt.build_diagnostics,
    )


def _rescale_power(token):
    return {"none": (0, 0), "A_AP": (1, 0),
            "A_AP*A_amp": (1, 1), "A_AP*A_amp^2": (1, 2)}[token]


def make_desi_prior_fns(spec, *, split, knl_bins, sigma8_bins_fn,
                        a_ap_bins_fn, sigma8_ref_bins, f_bins=DESI_F_FID,
                        lin_keys=LIN_SURVEY_KEYS, sampled_marginal_priors=()):
    """Build (prior_mean_fn, prior_sigma_fn, log_prior_nl_fn) from a spec.

    All three receive the physical theta_NL vector. Layer-2 rescaling divides
    by R_b(theta) per the Table-I footnote convention (a prior on the rescaled
    variable x*R => a prior on the raw coefficient x with mean m/R, width s/R).

    Two modes, selected by the spec's counterterm trio:

    - **Diagonal mode** (c0/c2/c4 carry no ``ctr_rotation`` token): the base
      behaviour. ``prior_mean_fn`` and ``prior_sigma_fn`` return
      ``(n_bins*len(lin_keys),)`` bin-major arrays laid out per ``lin_keys``.

    - **Cov mode** (c0/c2/c4 carry ``ctr_rotation == "multipole_to_tilde"``):
      the paper's diagonal per-multipole priors are exactly rotated into our
      mu-space tilde basis by the per-bin f-dependent map ``L(f)``
      (docs/design/desi-convention-map.md section 3.1). ``prior_sigma_fn`` then
      returns a stacked per-bin prior **covariance**
      ``(n_bins, len(lin_keys), len(lin_keys))``: ``diag(sigma_bj(theta)**2)``
      with the counterterm ``(c0..c0+3, c0..c0+3)`` block overwritten by
      ``L_b . diag(sigma_paper_b**2) . L_b^T`` where
      ``sigma_paper_b = row.paper_sigma / R_b(theta)`` (their rescale is
      ``A_AP*A_amp`` => ``R_b = a_ap_b . a_amp_b``) and ``L_b = L(f_bins[b])``.
      ``prior_mean_fn`` stays ``(n_bins*len(lin_keys),)`` but its counterterm
      entries per bin become ``L_b . (paper_mean_c0, c2, c4) / R_b``.
      ``f_bins`` (default :data:`DESI_F_FID`) must then have length ``n_bins``.
      The block is consumed by :func:`gaussian_marginal_loglike`'s ndim==2 branch
      via the per-bin path of the perbin/Taylor builders.

    ``lin_keys`` / ``sampled_marginal_priors`` (Tier-3 c1-sampled support)
    --------------------------------------------------------------------
    ``lin_keys`` (default :data:`LIN_SURVEY_KEYS`, the 11 marginalized rows)
    selects WHICH marginalized rows the prior_mean_fn / prior_sigma_fn cover. The
    c1-sampled analysis passes the 10-key variant ``LIN_SURVEY_KEYS`` minus
    ``('bk','ctr','c1')`` so the marginalized block loses its c1 row (the returned
    blocks are ``(n_bins, 10, 10)``); the counterterm trio still sits at slots
    ``c0..c0+2`` because c1 (slot 8) is AFTER it, so cov-mode is unaffected. The
    ``lin_keys`` MUST be a subset of ``spec.marginalized`` (which still validates
    all 11 rows), and cov-mode requires the c0/c2/c4 trio to remain in ``lin_keys``.

    ``sampled_marginal_priors`` = sequence of ``(key, positions)``: for each
    ``key`` (a ``spec.marginalized`` row REMOVED from ``lin_keys`` and now sampled
    in theta_NL), ``log_prior_nl_fn`` gains a per-bin Gaussian
    ``N(mean_b, width_b^2)`` with the SAME layer-2 machinery the marginalized row
    would have carried -- ``mean_b = row.mean / R_b`` (numeric only;
    ``mean_formula`` keys are rejected) and
    ``width_b = row.sigma * f_knl_b / R_b``, with ``R_b`` from ``row.rescale`` and
    ``f_knl_b = (knl_b/paper_knl)^2`` when ``row.factor_formula`` is the knl form.
    ``positions`` is the length-``n_bins`` sequence of theta_NL indices where that
    sampled parameter lives per bin (e.g. c1 at ``split.nl_b1_pos[b] + 3`` -- but
    only for the c1-sampled split; compute them from ``split.nl_idx`` if in
    doubt, and note they are bounds-checked against ``split.n_nl``). Both the
    mean and the width match ``_per_bin_arrays``, so this is exactly the prior
    the marginalized path integrates analytically and a sampled-c1 chain and a
    marginalized-c1 chain carry an equivalent c1 prior.
    """
    n_bins = len(split.nl_b1_pos)
    lin_keys = tuple(lin_keys)
    missing = [k for k in lin_keys if k not in spec.marginalized]
    if missing:
        raise ValueError(f"lin_keys not present in spec.marginalized: {missing}")
    n_lin_keys = len(lin_keys)
    # The prior rows are laid out purely per lin_keys while the templates are
    # laid out per split.lin_idx, so an equal-length but different (or merely
    # reordered) key list would silently mis-assign every row. split.lin_keys is
    # bin-major, so its first n_lin_keys entries are bin 0's key order.
    split_lin_keys = tuple(k for _b, k in split.lin_keys[:n_lin_keys])
    if (split.n_lin != n_bins * n_lin_keys or split_lin_keys != lin_keys):
        raise ValueError(
            "lin_keys disagrees with the split's marginalized block: split has "
            f"n_lin={split.n_lin} ({split_lin_keys}) per bin, lin_keys has "
            f"{n_lin_keys} ({lin_keys}). Pass the SAME key tuple to "
            "split_marginal_indices(lin_survey_keys=...) and here.")
    knl_arr = jnp.asarray(knl_bins, dtype=jnp.float64)
    if knl_arr.shape != (n_bins,):
        raise ValueError(f"knl_bins must have length {n_bins}")
    sigma8_ref = jnp.asarray(sigma8_ref_bins, dtype=jnp.float64)
    paper_knl = float(spec.metadata.get("paper_knl", 0.45))
    rows = [spec.marginalized[k] for k in lin_keys]

    base_mean = jnp.array([0.0 if r.mean is None else r.mean for r in rows])
    base_sigma = jnp.array([r.sigma for r in rows])
    fac_knl = jnp.array([r.factor_formula == "knl_over_0p45_sq" for r in rows])
    coevo = jnp.array([r.mean_formula == "coevolution_bGamma3" for r in rows])
    coevo_factor = jnp.array([r.factor for r in rows])
    coevo_offset = jnp.array([r.offset for r in rows])
    ap_pow = jnp.array([_rescale_power(r.rescale)[0] for r in rows])
    amp_pow = jnp.array([_rescale_power(r.rescale)[1] for r in rows])
    b1_pos = jnp.asarray(split.nl_b1_pos)

    def _per_bin_arrays(theta_nl):
        """Return ``(R, mean_bin, sig_bin)``, each ``(n_bins, len(lin_keys))``."""
        theta_nl = jnp.asarray(theta_nl, dtype=jnp.float64)
        a_ap = a_ap_bins_fn(theta_nl)                       # (n_bins,)
        a_amp = sigma8_bins_fn(theta_nl) ** 2 / sigma8_ref ** 2
        R = (a_ap[:, None] ** ap_pow[None, :]
             * a_amp[:, None] ** amp_pow[None, :])          # (n_bins, 11)
        f_bin = jnp.where(fac_knl[None, :],
                          (knl_arr[:, None] / paper_knl) ** 2, 1.0)
        b1 = theta_nl[b1_pos]                               # (n_bins,)
        coevo_mean = ((23.0 / 42.0) * (b1[:, None] - 1.0) * coevo_factor[None, :]
                      + coevo_offset[None, :])
        mean = jnp.where(coevo[None, :], coevo_mean, base_mean[None, :])
        return R, mean / R, base_sigma[None, :] * f_bin / R

    cov_mode = all(spec.marginalized[k].ctr_rotation is not None
                   for k in _CTR_TRIO)

    if cov_mode:
        f_arr = jnp.asarray(f_bins, dtype=jnp.float64)
        if f_arr.shape != (n_bins,):
            raise ValueError(
                f"f_bins must have length {n_bins} for cov-mode "
                f"(ctr_rotation) priors; got shape {f_arr.shape}")
        missing_ctr = [k for k in _CTR_TRIO if k not in lin_keys]
        if missing_ctr:
            raise ValueError(
                "cov-mode (ctr_rotation) requires the c0/c2/c4 trio in lin_keys; "
                f"missing {missing_ctr}")
        L_bins = ctr_rotation_matrices(f_arr)               # (n_bins, 3, 3)
        c0 = lin_keys.index(("pk", "ctr", "c0"))            # == 2 (c1 is after)
        ctr = slice(c0, c0 + 3)
        # The rotated L.diag.L^T block is written at slots c0..c0+2, so the trio
        # must be CONTIGUOUS and in (c0, c2, c4) order -- membership alone would
        # let a reordered lin_keys rotate the block onto the wrong parameters.
        if tuple(lin_keys[ctr]) != _CTR_TRIO:
            raise ValueError(
                "cov-mode (ctr_rotation) requires the c0/c2/c4 trio to be "
                f"contiguous and in order in lin_keys; got {lin_keys[ctr]} at "
                f"slots {c0}..{c0 + 2}")
        ctr_rows = [spec.marginalized[k] for k in _CTR_TRIO]
        paper_sigma_ctr = jnp.array([r.paper_sigma for r in ctr_rows])   # (3,)
        paper_mean_ctr = jnp.array(
            [0.0 if r.paper_mean is None else r.paper_mean for r in ctr_rows])
        eye = jnp.eye(n_lin_keys, dtype=jnp.float64)

        def prior_mean_fn(theta_nl):
            R, mean_bin, _ = _per_bin_arrays(theta_nl)
            R_ctr = R[:, c0]                                 # (n_bins,) scalar/bin
            m_scaled = paper_mean_ctr[None, :] / R_ctr[:, None]    # (n_bins, 3)
            ctr_mean = jnp.einsum("bij,bj->bi", L_bins, m_scaled)  # (n_bins, 3)
            return mean_bin.at[:, ctr].set(ctr_mean).reshape(-1)

        def prior_sigma_fn(theta_nl):
            R, _, sig_bin = _per_bin_arrays(theta_nl)
            R_ctr = R[:, c0]                                 # (n_bins,)
            blocks = (sig_bin ** 2)[:, :, None] * eye[None, :, :]  # (n_bins,11,11)
            s_scaled = paper_sigma_ctr[None, :] / R_ctr[:, None]   # (n_bins, 3)
            ctr_block = jnp.einsum(
                "bij,bj,bkj->bik", L_bins, s_scaled ** 2, L_bins)  # (n_bins,3,3)
            return blocks.at[:, ctr, ctr].set(ctr_block)          # (n_bins,11,11)
    else:
        def prior_mean_fn(theta_nl):
            return _per_bin_arrays(theta_nl)[1].reshape(-1)

        def prior_sigma_fn(theta_nl):
            return _per_bin_arrays(theta_nl)[2].reshape(-1)

    gaussian_sampled = [(nm, spec.sampled[nm]) for nm in ("b2", "bG2")
                        if spec.sampled[nm].kind == "gaussian"]
    offsets = {"b2": 1, "bG2": 2}

    # Sampled-marginal (e.g. c1) priors: a marginalized row that is now SAMPLED
    # keeps the same width machinery it would have carried as a marginalized
    # entry -- width_b = row.sigma * f_knl_b / R_b, R_b from row.rescale.
    sampled_extra = []
    for key, positions in sampled_marginal_priors:
        row = spec.marginalized[key]
        if row.mean_formula is not None:
            raise ValueError(
                f"sampled_marginal_priors key {key} has mean_formula "
                f"{row.mean_formula!r}; only numeric-mean rows are supported")
        if key in lin_keys:
            raise ValueError(
                f"sampled_marginal_priors key {key} is still in lin_keys, so it "
                "is marginalized and has no theta_NL slot; drop it from lin_keys "
                "(and from the split's lin_survey_keys) first")
        pos_list = [int(p) for p in positions]
        if len(pos_list) != n_bins:
            raise ValueError(
                f"sampled_marginal_priors positions for {key} must have length "
                f"{n_bins} (one theta_NL index per bin); got {len(pos_list)}")
        # jnp gathers CLAMP out-of-range indices, so an off-by-one would apply
        # this prior to a different parameter with no diagnostic.
        if min(pos_list) < 0 or max(pos_list) >= split.n_nl:
            raise ValueError(
                f"sampled_marginal_priors positions for {key} out of range for "
                f"theta_NL (n_nl={split.n_nl}): {pos_list}")
        pos_arr = jnp.asarray(pos_list)
        ap_p, amp_p = _rescale_power(row.rescale)
        knl_factor = ((knl_arr / paper_knl) ** 2
                      if row.factor_formula == "knl_over_0p45_sq"
                      else jnp.ones(n_bins, dtype=jnp.float64))
        sampled_extra.append((pos_arr, float(0.0 if row.mean is None else row.mean),
                              float(row.sigma), int(ap_p), int(amp_p), knl_factor))

    def log_prior_nl_fn(theta_nl):
        theta_nl = jnp.asarray(theta_nl, dtype=jnp.float64)
        s8 = sigma8_bins_fn(theta_nl)                       # (n_bins,)
        total = 0.0
        for nm, row in gaussian_sampled:
            pos = b1_pos + offsets[nm]
            width = (row.paper_sigma / s8 ** 2 if row.rescale == "sigma8_sq"
                     else jnp.full_like(s8, row.paper_sigma))
            x = theta_nl[pos] - (row.paper_mean or 0.0)
            total = total + jnp.sum(
                -0.5 * (x / width) ** 2 - jnp.log(width) - 0.5 * _LOG2PI)
        if sampled_extra:
            a_ap = a_ap_bins_fn(theta_nl)                   # (n_bins,)
            a_amp = s8 ** 2 / sigma8_ref ** 2               # (n_bins,)
            for pos_arr, mean, base_sig, ap_p, amp_p, knl_factor in sampled_extra:
                R = a_ap ** ap_p * a_amp ** amp_p           # (n_bins,)
                width = base_sig * knl_factor / R           # (n_bins,)
                # Layer-2 divides BOTH the width and the mean by R (same as the
                # marginalized path's _per_bin_arrays), so the two treatments
                # describe the same prior on the raw coefficient.
                x = theta_nl[pos_arr] - mean / R            # (n_bins,)
                total = total + jnp.sum(
                    -0.5 * (x / width) ** 2 - jnp.log(width) - 0.5 * _LOG2PI)
        return total

    return prior_mean_fn, prior_sigma_fn, log_prior_nl_fn


def make_lcdm_rescaling_fns(*, pklin_emulator, cosmo_keys, cosmo_sizes,
                            z_bins, fid_cosmo_native, mnu_fixed=0.06,
                            fixed_cosmo_extras=None):
    """Build (sigma8_bins_fn, a_ap_bins_fn, sigma8_ref_bins) closures.

    theta_NL[: n_cosmo] is interpreted as the native cosmology vector (all
    cosmology parameters sampled, the production LCDM layout). sigma8_ref_bins
    is sigma8(z_b) at fid_cosmo_native, so A_amp(fid) = 1 exactly. Mirrors the
    emulator wiring of make_lcdm_derived_params_fn (derived.py); background from
    ps_1loop_jax (physical units).

    ``fixed_cosmo_extras`` (mapping ``name -> value``, optional) supplies fixed
    emulator inputs that are NOT in the sampled ``cosmo_keys`` basis but that the
    linear-Pk emulator requires -- e.g. the baryon-feedback nuisances
    ``{'A_b': ..., 'eta_b': ..., 'logT_AGN': ...}`` when only the LCDM core
    (ombh2, omch2, logA, ns, h) is varied. They are injected as constants into
    the sigma8 emulator call, so sigma8 (hence the A_amp rescaling) carries no
    spurious derivative w.r.t. them.
    """
    from ps_1loop_jax import background as bg

    from .derived import _emulator_input_dict, sigma8_from_linear_pk
    from .params import CosmoParams

    n_cosmo = int(sum(cosmo_sizes))
    has_mnu = "mnu" in cosmo_keys
    z_arr = tuple(float(z) for z in z_bins)
    emulator_modes = jnp.asarray(pklin_emulator.modes, dtype=jnp.float64)
    emulator_parameters = getattr(pklin_emulator, "parameters", None)
    if emulator_parameters is not None:
        emulator_parameters = tuple(emulator_parameters)

    def _cosmo_obj(theta_nl):
        vec = jnp.asarray(theta_nl, dtype=jnp.float64)[:n_cosmo]
        return CosmoParams.from_array(vec, cosmo_keys, cosmo_sizes)

    def _bg_args(cosmo_obj):
        h = cosmo_obj.h[0]
        omb = cosmo_obj.omega_b[0]
        omc = cosmo_obj.omega_cdm[0]
        mnu = (cosmo_obj.mnu[0] if has_mnu
               else jnp.asarray(mnu_fixed, dtype=jnp.float64))
        return omb, omc, h, mnu

    def sigma8_bins_fn(theta_nl):
        cosmo_obj = _cosmo_obj(theta_nl)
        h = cosmo_obj.h[0]
        vals = []
        for z in z_arr:
            emulator_input = _emulator_input_dict(
                cosmo_obj, emulator_parameters=emulator_parameters,
                sigma8_redshift=z, extra_cosmo=fixed_cosmo_extras)
            pklin = jnp.ravel(jnp.asarray(
                pklin_emulator.predict(emulator_input), dtype=jnp.float64))
            vals.append(sigma8_from_linear_pk(emulator_modes / h, pklin))
        return jnp.stack(vals)

    fid_obj = CosmoParams.from_array(
        jnp.asarray(fid_cosmo_native, dtype=jnp.float64), cosmo_keys, cosmo_sizes)
    omb_f, omc_f, h_f, mnu_f = _bg_args(fid_obj)
    Hz_fid = jnp.array([float(bg.Hz(omb_f, omc_f, h_f, z, mnu_f)) for z in z_arr])
    DA_fid = jnp.array([float(bg.angular_diameter_distance(
        omb_f, omc_f, h_f, z, mnu_f)) for z in z_arr])
    H0_fid = 100.0 * float(h_f)

    def a_ap_bins_fn(theta_nl):
        omb, omc, h, mnu = _bg_args(_cosmo_obj(theta_nl))
        Hz = jnp.stack([bg.Hz(omb, omc, h, z, mnu) for z in z_arr])
        DA = jnp.stack([bg.angular_diameter_distance(omb, omc, h, z, mnu)
                        for z in z_arr])
        return (H0_fid / (100.0 * h)) ** 3 * (Hz / Hz_fid) * (DA_fid / DA) ** 2

    sigma8_ref_bins = sigma8_bins_fn(
        jnp.asarray(fid_cosmo_native, dtype=jnp.float64))
    return sigma8_bins_fn, a_ap_bins_fn, sigma8_ref_bins


# =============================================================================
# Task 6sigma: Fisher-side fiducial widths from the spec (ctr marginal widths)
# =============================================================================

__all__ += ["build_prior_sigmas_from_desi_spec"]


def build_prior_sigmas_from_desi_spec(spec, *, knl_bins, sigma8_ref_bins,
                                      f_bins=DESI_F_FID, return_ctr_blocks=False):
    """Fiducial (R = 1) per-bin prior widths for the Fisher side.

    Returns ``(survey_sigma_dicts, sampled_sigma_bins)``: one dict per bin for
    the marginalized survey block (mapping ``(section, group, key) -> sigma``,
    consumable by :func:`inference.build_prior_sigmas` as a per-bin list), and
    one dict per bin for the sampled bias block (``b2``/``bG2`` raw widths at
    ``sigma8_ref``; the flat ``b1`` is omitted = no prior).

    Branch stream-b-sigmap (Amendment 1). The counterterm trio ``pk.ctr.c0/c2/c4``
    -- the rows whose ``ctr_rotation == "multipole_to_tilde"`` -- carries a
    *correlated* prior in our mu-space tilde basis,
    ``L_b . diag(paper_sigma**2) . L_b^T`` with
    ``L_b = ctr_rotation_matrices(f_bins)[b]`` (map section 3.1). Legacy Fisher
    consumers (:func:`inference.build_prior_sigmas`) are DIAGONAL-ONLY, so the
    width emitted for each ctr row is the MARGINAL width of that correlated
    prior: ``sqrt(diag(L_b . diag(paper_sigma**2) . L_b^T))``, with
    ``paper_sigma`` read from the three rows' ``paper_sigma`` in order
    ``(c0, c2, c4)`` (NOT the mapped ``row.sigma``; consistent with the cov-mode
    factory). The off-diagonal correlations are NOT representable in these
    diagonal dicts -- they live in the gate's Hessian-Fisher
    (:func:`make_desi_prior_fns` cov-mode + :func:`gaussian_marginal_loglike`).

    Non-ctr rows are unchanged from the base contract: ``sigma * (knl/paper_knl)**2``
    for a0/a2 (``factor_formula == "knl_over_0p45_sq"``), the mapped paper
    ``sigma`` otherwise. ``f_bins`` (default :data:`DESI_F_FID`, one entry per
    bin) is consumed only when the trio carries the token, and is added for
    signature parity with :func:`make_desi_prior_fns`.

    With ``return_ctr_blocks=True`` a third element is returned:
    ``ctr_cov_blocks``, the stacked ``(n_bins, 3, 3)`` fiducial (R = 1)
    counterterm covariance ``L_b . diag(paper_sigma**2) . L_b^T`` (``None`` if the
    spec's trio is token-less), for consumers that can use the full correlated
    block rather than its marginal widths.
    """
    paper_knl = float(spec.metadata.get("paper_knl", 0.45))
    knl_bins = tuple(float(k) for k in knl_bins)
    sigma8_ref_bins = tuple(float(s) for s in sigma8_ref_bins)
    n_bins = len(knl_bins)
    if len(sigma8_ref_bins) != n_bins:
        raise ValueError(
            f"knl_bins ({n_bins}) and sigma8_ref_bins "
            f"({len(sigma8_ref_bins)}) must have equal length")

    cov_mode = all(spec.marginalized[k].ctr_rotation is not None
                   for k in _CTR_TRIO)
    ctr_cov_blocks = None
    if cov_mode:
        f_arr = jnp.asarray(f_bins, dtype=jnp.float64)
        if f_arr.shape != (n_bins,):
            raise ValueError(
                f"f_bins must have length {n_bins} for cov-mode (ctr_rotation) "
                f"priors; got shape {f_arr.shape}")
        L_bins = ctr_rotation_matrices(f_arr)                    # (n_bins, 3, 3)
        paper_sigma_ctr = jnp.array(
            [spec.marginalized[k].paper_sigma for k in _CTR_TRIO])   # (3,)
        ctr_cov_blocks = jnp.einsum(
            "bij,j,bkj->bik", L_bins, paper_sigma_ctr ** 2, L_bins)  # (nb,3,3)

    survey_sigma_dicts = []
    sampled_sigma_bins = []
    for b, (knl, s8_ref) in enumerate(zip(knl_bins, sigma8_ref_bins)):
        d = {}
        for key, row in spec.marginalized.items():
            if cov_mode and key in _CTR_TRIO:
                continue                        # ctr marginal widths added below
            f_bin = ((knl / paper_knl) ** 2
                     if row.factor_formula == "knl_over_0p45_sq" else 1.0)
            d[key] = row.sigma * f_bin
        if cov_mode:
            marg = jnp.sqrt(jnp.diagonal(ctr_cov_blocks[b]))     # (3,)
            for key, width in zip(_CTR_TRIO, marg):
                d[key] = float(width)
        survey_sigma_dicts.append(d)

        sb = {}
        for nm in ("b2", "bG2"):
            row = spec.sampled[nm]
            if row.kind == "gaussian":
                sb[nm] = (row.paper_sigma / s8_ref ** 2
                          if row.rescale == "sigma8_sq" else row.paper_sigma)
        sampled_sigma_bins.append(sb)

    if return_ctr_blocks:
        return survey_sigma_dicts, sampled_sigma_bins, ctr_cov_blocks
    return survey_sigma_dicts, sampled_sigma_bins
