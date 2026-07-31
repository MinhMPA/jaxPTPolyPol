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


def load_desi_prior_spec(name_or_path="desi_dr1_reanalysis_2511_20757"):
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

    sampled = {}
    for name, entry in raw["sampled"].items():
        row = SampledRow(**entry)
        if row.kind not in ("flat", "gaussian"):
            raise SpecValidationError(f"sampled {name}: bad kind {row.kind!r}")
        if row.rescale not in _SAMPLED_RESCALE:
            raise SpecValidationError(
                f"sampled {name}: unknown rescale {row.rescale!r}")
        if row.kind == "gaussian" and (row.paper_sigma is None
                                       or row.paper_sigma <= 0.0):
            raise SpecValidationError(
                f"sampled {name}: gaussian needs positive paper_sigma")
        sampled[name] = row
    for required in ("b1", "b2", "bG2"):
        if required not in sampled:
            raise SpecValidationError(f"sampled block missing {required!r}")

    return DesiPriorSpec(metadata=raw.get("metadata", {}),
                         marginalized=marginalized, sampled=sampled)
