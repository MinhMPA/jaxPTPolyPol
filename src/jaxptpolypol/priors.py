"""
Packaged Gaussian prior specifications for full-shape analyses.

The packaged YAML specs store prior metadata together with survey / nuisance
Gaussian priors in the role-aware ``shared`` / ``pk`` / ``bk`` taxonomy used by
``FullShapeSurveyParams``.

These helpers are intentionally thin wrappers around the existing prior-index
plumbing in :mod:`jaxptpolypol.inference`:

- Fisher forecasts still consume ``{packed_index: sigma}``
- sampling workflows still consume ``[(index_in_varied, mean, sigma), ...]``

At present the packaged Eq. (12) spec is already expressed in this codebase's
parameter convention, so the ``scale`` field is treated as validated metadata
rather than an active rescaling rule.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from importlib.resources import files
from pathlib import Path

import yaml

from .inference import build_prior_sigmas

__all__ = [
    "DEFAULT_PRIOR_SPEC",
    "build_prior_entries_from_spec",
    "build_prior_sigmas_from_spec",
    "load_prior_spec",
    "resolve_survey_prior_spec",
]

DEFAULT_PRIOR_SPEC = "eft_eq12_2405_02252"

_OBSERVABLE_SECTIONS = {
    "pk": ("shared", "pk"),
    "bk": ("shared", "bk"),
    "joint": ("shared", "pk", "bk"),
}
_SUPPORTED_SCALES = {"unit", "mpc_over_h_sq"}


def _normalize_observable(observable: str) -> tuple[str, ...]:
    try:
        return _OBSERVABLE_SECTIONS[observable]
    except KeyError as exc:
        raise ValueError(
            "observable must be 'pk', 'bk', or 'joint', "
            f"got {observable!r}"
        ) from exc


def _load_yaml_text(name_or_path: str | Path) -> str:
    path = Path(name_or_path).expanduser()
    looks_like_path = path.suffix in {".yaml", ".yml"} or path.parent != Path(".")
    if path.is_file():
        return path.read_text(encoding="utf-8")
    if looks_like_path:
        raise FileNotFoundError(f"prior spec file not found: {path}")

    filename = path.name or DEFAULT_PRIOR_SPEC
    if not filename.endswith((".yaml", ".yml")):
        filename = f"{filename}.yaml"
    packaged = files("jaxptpolypol").joinpath("data").joinpath("priors").joinpath(filename)
    if not packaged.is_file():
        raise FileNotFoundError(f"packaged prior spec not found: {filename}")
    return packaged.read_text(encoding="utf-8")


def _require_mapping(obj, name: str) -> Mapping:
    if not isinstance(obj, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(obj).__name__}")
    return obj


def _coerce_prior_spec(spec: str | Path | Mapping | None) -> Mapping:
    if spec is None:
        return load_prior_spec(DEFAULT_PRIOR_SPEC)
    if isinstance(spec, Mapping):
        return spec
    return load_prior_spec(spec)


def _matched_prior_for_key(
    survey_key: tuple,
    resolved: Mapping[tuple[str, str, str], dict[str, float | str]],
) -> dict[str, float | str] | None:
    if survey_key in resolved:
        return dict(resolved[survey_key])
    if len(survey_key) == 2:
        shared_key = ("shared", survey_key[0], survey_key[1])
        prior = resolved.get(shared_key)
        if prior is not None:
            return dict(prior)
    return None


def load_prior_spec(name_or_path: str | Path = DEFAULT_PRIOR_SPEC) -> dict:
    """Load a packaged or filesystem prior YAML spec.

    Parameters
    ----------
    name_or_path : str or pathlib.Path, optional
        Packaged prior-spec name (with or without ``.yaml``) or a path to a
        YAML file on disk.

    Returns
    -------
    spec : dict
        Parsed YAML mapping.
    """
    text = _load_yaml_text(name_or_path)
    spec = yaml.safe_load(text)
    spec = _require_mapping(spec, "prior spec")
    _require_mapping(spec.get("survey_priors", {}), "prior spec['survey_priors']")
    return deepcopy(dict(spec))


def resolve_survey_prior_spec(
    spec: str | Path | Mapping | None = None,
    *,
    observable: str = "joint",
) -> dict[tuple[str, str, str], dict[str, float | str]]:
    """Resolve a packaged survey-prior spec for one observable layout.

    The returned mapping is keyed by role-aware survey keys such as
    ``('pk', 'ctr', 'c0')`` and each value contains ``mean``, ``sigma``, and
    ``scale`` fields.
    """
    spec_map = _require_mapping(_coerce_prior_spec(spec), "prior spec")
    survey_priors = _require_mapping(
        spec_map.get("survey_priors", {}),
        "prior spec['survey_priors']",
    )

    resolved: dict[tuple[str, str, str], dict[str, float | str]] = {}
    for section in _normalize_observable(observable):
        section_map = _require_mapping(
            survey_priors.get(section, {}),
            f"prior spec['survey_priors']['{section}']",
        )
        for group, group_map in section_map.items():
            group_map = _require_mapping(
                group_map,
                f"prior spec['survey_priors']['{section}']['{group}']",
            )
            for key, prior in group_map.items():
                prior = _require_mapping(
                    prior,
                    f"prior spec['survey_priors']['{section}']['{group}']['{key}']",
                )
                if "mean" not in prior or "sigma" not in prior:
                    raise KeyError(
                        "each prior entry must define 'mean' and 'sigma': "
                        f"{section}.{group}.{key}"
                    )
                sigma = float(prior["sigma"])
                if sigma <= 0.0:
                    raise ValueError(
                        f"prior sigma must be positive for {section}.{group}.{key}"
                    )
                scale = str(prior.get("scale", "unit"))
                if scale not in _SUPPORTED_SCALES:
                    raise ValueError(
                        f"unsupported prior scale {scale!r} for "
                        f"{section}.{group}.{key}"
                    )
                resolved[(str(section), str(group), str(key))] = {
                    "mean": float(prior["mean"]),
                    "sigma": sigma,
                    "scale": scale,
                }

    return resolved


def build_prior_sigmas_from_spec(
    *,
    cosmo_keys: tuple[str, ...],
    cosmo_sizes: tuple[int, ...],
    survey_keys: tuple,
    n_bins: int = 1,
    observable: str = "joint",
    spec: str | Path | Mapping | None = None,
    cosmo_priors: Mapping[str, float] | None = None,
) -> dict[int, float]:
    """Build packed Fisher prior widths from a packaged prior spec."""
    resolved = resolve_survey_prior_spec(spec, observable=observable)
    survey_priors = {}
    for survey_key in survey_keys:
        prior = _matched_prior_for_key(survey_key, resolved)
        if prior is not None:
            survey_priors[survey_key] = float(prior["sigma"])

    return build_prior_sigmas(
        cosmo_keys=cosmo_keys,
        cosmo_sizes=cosmo_sizes,
        survey_keys=survey_keys,
        n_bins=n_bins,
        cosmo_priors=dict(cosmo_priors) if cosmo_priors is not None else None,
        survey_priors=survey_priors or None,
    )


def build_prior_entries_from_spec(
    varied_idx: Sequence[int],
    *,
    cosmo_keys: tuple[str, ...],
    cosmo_sizes: tuple[int, ...],
    survey_keys: tuple,
    n_bins: int = 1,
    observable: str = "joint",
    spec: str | Path | Mapping | None = None,
    cosmo_prior_entries: Mapping[str, tuple[float, float]] | None = None,
) -> list[tuple[int, float, float]]:
    """Build ``(index_in_varied, mean, sigma)`` Gaussian prior entries."""
    resolved = resolve_survey_prior_spec(spec, observable=observable)
    varied_lookup = {int(full_idx): i for i, full_idx in enumerate(varied_idx)}
    entries: list[tuple[int, float, float]] = []

    offset = 0
    if cosmo_prior_entries is not None:
        for key, size in zip(cosmo_keys, cosmo_sizes):
            if key in cosmo_prior_entries:
                mean, sigma = cosmo_prior_entries[key]
                sigma = float(sigma)
                if sigma <= 0.0:
                    raise ValueError(f"prior sigma must be positive for {key}")
                mean = float(mean)
                for j in range(size):
                    varied_pos = varied_lookup.get(offset + j)
                    if varied_pos is not None:
                        entries.append((varied_pos, mean, sigma))
            offset += size
    else:
        offset = sum(cosmo_sizes)

    n_cosmo = sum(cosmo_sizes)
    n_survey = len(survey_keys)
    for b in range(n_bins):
        bin_offset = n_cosmo + b * n_survey
        for s, survey_key in enumerate(survey_keys):
            prior = _matched_prior_for_key(survey_key, resolved)
            if prior is None:
                continue
            varied_pos = varied_lookup.get(bin_offset + s)
            if varied_pos is None:
                continue
            entries.append(
                (
                    varied_pos,
                    float(prior["mean"]),
                    float(prior["sigma"]),
                )
            )

    entries.sort(key=lambda item: item[0])
    return entries
