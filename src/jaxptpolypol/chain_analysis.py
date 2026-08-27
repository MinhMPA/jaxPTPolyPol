"""
Chain analysis utilities for packed parameter vectors used in jaxPTPolyPol.

This module is designed for sampler outputs built from the packed vectors used
throughout the package:

- full-shape ``Pk + bias/nuisance + cosmo`` parameter vectors
- full-shape ``Bk + bias/nuisance + cosmo`` parameter vectors
- ``BAO + cosmo`` parameter vectors

The key abstraction is :class:`PackedParameterSpec`, which knows how to map a
flat packed vector into named scalar components suitable for summaries,
diagnostics, ArviZ conversion, and plotting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_LATEX_LABELS",
    "PackedParameterSpec",
    "make_fullshape_spec",
    "make_bao_spec",
    "packed_samples_to_dict",
    "packed_sample_labels",
    "scalar_summary",
    "scalar_summary_packed",
    "to_inference_data",
    "to_inference_data_packed",
    "diagnostics_summary",
    "ess_by_variable",
    "slice_draws",
    "discard_burnin",
    "validate_tuned_step_size",
    "lag1_autocorr_per_chain",
    "mean_abs_jump_per_chain",
    "effective_sample_size",
    "ess_per_chain",
    "plot_trace",
    "plot_trace_packed",
    "plot_corner",
    "plot_corner_packed",
    "plot_credible_contours",
    "credible_intervals",
]


DEFAULT_SAMPLE_STATS_MAP = {
    "is_divergent": "diverging",
    "num_integration_steps": "n_steps",
    "num_trajectory_expansions": "tree_depth",
    "logdensity": "lp",
}

DEFAULT_LATEX_LABELS = {
    "Omega_m": r"$\Omega_m$",
    "omega_m": r"$\Omega_m$",
    "h": r"$h$",
    "H0": r"$H_0$",
    "omega_b": r"$\omega_b$",
    "ombh2": r"$\omega_b h^2$",
    "omega_cdm": r"$\omega_c$",
    "omch2": r"$\omega_c h^2$",
    "n_s": r"$n_s$",
    "ns": r"$n_s$",
    "sigma8": r"$\sigma_8$",
    "sigma_8": r"$\sigma_8$",
    "As": r"$A_s$",
    "logA": r"$\ln(10^{10} A_s)$",
    "ln10^{10}A_s": r"$\ln(10^{10} A_s)$",
    "mnu": r"$\sum m_\nu$",
    "z": r"$z$",
    "w0_fld": r"$w_0$",
    "wa_fld": r"$w_a$",
    "bias.b1": r"$b_1$",
    "bias.b2": r"$b_2$",
    "bias.bs2": r"$b_{K^2}$",
    "bias.bs": r"$b_s$",
    "bias.b3nl": r"$b_{3\mathrm{nl}}$",
    "bias.bn2": r"$b_{\nabla^2\delta}$",
    "bias.b_lapl": r"$b_{\nabla^2\delta}$",
    "bias.bdelta": r"$b_\delta$",
    "bias.bdelta2": r"$b_{\delta^2}$",
    "ctr.c0": r"$c_0$",
    "ctr.c2": r"$c_2$",
    "ctr.c4": r"$c_4$",
    "stoch.Pshot": r"$P_{\rm shot}$",
    "stoch.alpha0": r"$\alpha_0$",
    "stoch.alpha2": r"$\alpha_2$",
    "stoch.alpha4": r"$\alpha_4$",
}


def _require_arviz():
    try:
        import arviz as az
    except Exception as exc:  # pragma: no cover - exercised when arviz missing
        raise ImportError(
            "arviz is required for chain diagnostics/InferenceData conversion. "
            "Install with `pip install arviz`."
        ) from exc
    return az


def _require_blackjax_effective_sample_size():
    try:
        import blackjax
        import jax.numpy as jnp
    except Exception as exc:  # pragma: no cover - exercised when blackjax/jax missing
        raise ImportError(
            "blackjax and jax are required for ESS diagnostics."
        ) from exc
    return blackjax.diagnostics.effective_sample_size, jnp


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - exercised when matplotlib missing
        raise ImportError(
            "matplotlib is required for plotting. Install with `pip install matplotlib`."
        ) from exc
    return plt


def _as_mapping(obj: Mapping[str, Any] | Any, *, name: str) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    if hasattr(obj, "_asdict"):
        return dict(obj._asdict())
    fields = getattr(obj, "_fields", None)
    if fields is not None:
        return {field: getattr(obj, field) for field in fields}
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    raise TypeError(
        f"{name} must be a mapping or a namedtuple/dataclass-like object; "
        f"got {type(obj)!r}."
    )


def _normalize_axis(axis: int, ndim: int, *, name: str) -> int:
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"{name}={axis} is out of bounds for ndim={ndim}.")
    return axis


def _as_chain_draw(
    values: Any,
    *,
    chain_axis: int | None,
    draw_axis: int,
    var_name: str,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 0:
        raise ValueError(f"Variable {var_name!r} must have at least one dimension.")

    draw_axis = _normalize_axis(draw_axis, arr.ndim, name="draw_axis")

    if chain_axis is None:
        perm = [draw_axis] + [ax for ax in range(arr.ndim) if ax != draw_axis]
        arr = np.transpose(arr, axes=perm)
        return arr[np.newaxis, ...]

    if arr.ndim < 2:
        raise ValueError(
            f"Variable {var_name!r} must have at least 2 dimensions when chain_axis is set."
        )

    chain_axis = _normalize_axis(chain_axis, arr.ndim, name="chain_axis")
    if chain_axis == draw_axis:
        raise ValueError(f"Variable {var_name!r}: chain_axis and draw_axis cannot be equal.")

    perm = [chain_axis, draw_axis] + [
        ax for ax in range(arr.ndim) if ax not in (chain_axis, draw_axis)
    ]
    return np.transpose(arr, axes=perm)


def _as_scalar_chain_draw(
    values: Any,
    *,
    chain_axis: int | None,
    draw_axis: int,
    var_name: str,
) -> np.ndarray:
    arr = _as_chain_draw(
        values,
        chain_axis=chain_axis,
        draw_axis=draw_axis,
        var_name=var_name,
    )
    if arr.ndim != 2:
        raise ValueError(
            f"Variable {var_name!r} must be scalar-like after chain/draw mapping. "
            f"Got shape {arr.shape}."
        )
    return np.asarray(arr, dtype=float)


def _prepare_group(
    group: Mapping[str, Any],
    *,
    chain_axis: int | None,
    draw_axis: int,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, values in group.items():
        out[name] = _as_chain_draw(
            values,
            chain_axis=chain_axis,
            draw_axis=draw_axis,
            var_name=name,
        )
    return out


def _fallback_tree_map(function, tree):
    if isinstance(tree, Mapping):
        return {key: _fallback_tree_map(function, value) for key, value in tree.items()}
    if dataclasses.is_dataclass(tree):
        return dataclasses.replace(
            tree,
            **{
                field.name: _fallback_tree_map(function, getattr(tree, field.name))
                for field in dataclasses.fields(tree)
            },
        )
    if hasattr(tree, "_fields") and hasattr(tree, "_asdict"):
        values = {key: _fallback_tree_map(function, value) for key, value in tree._asdict().items()}
        return type(tree)(**values)
    if isinstance(tree, tuple):
        return tuple(_fallback_tree_map(function, value) for value in tree)
    if isinstance(tree, list):
        return [_fallback_tree_map(function, value) for value in tree]
    return function(tree)


def _tree_map(function, tree):
    try:  # pragma: no cover - used when jax is available
        from jax import tree_util

        return tree_util.tree_map(function, tree)
    except Exception:
        return _fallback_tree_map(function, tree)


def _var_label(name: str, var_labels: Mapping[str, str] | None = None) -> str:
    if var_labels is not None and name in var_labels:
        return str(var_labels[name])
    return DEFAULT_LATEX_LABELS.get(name, name)


def _reduce_components(values: np.ndarray, how: str) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    if how == "mean":
        return float(np.mean(arr))
    if how == "min":
        return float(np.min(arr))
    if how == "median":
        return float(np.median(arr))
    raise ValueError(f"Unknown component_reduce={how!r}; expected 'mean', 'min', or 'median'.")


def _survey_base_name(
    survey_key: tuple[str, str | None] | tuple[str, str, str | None],
) -> str:
    if len(survey_key) == 2:
        group, key = survey_key
    elif len(survey_key) == 3:
        _, group, key = survey_key
    else:  # pragma: no cover - defensive guard
        raise ValueError(f"unsupported survey key format: {survey_key!r}")
    return group if key is None else f"{group}.{key}"


def _format_component_names(base_name: str, size: int) -> list[str]:
    if size == 1:
        return [base_name]
    return [f"{base_name}[{i}]" for i in range(size)]


def _append_bin_suffix(name: str, bin_label: str | None, n_bins: int) -> str:
    if n_bins == 1:
        return name
    if bin_label is None:
        return name
    return f"{name}@{bin_label}"


def _latex_with_bin(label: str, bin_label: str | None, n_bins: int) -> str:
    if n_bins == 1 or bin_label is None:
        return label
    if label.startswith("$") and label.endswith("$"):
        return label[:-1] + rf"^{{({bin_label})}}$"
    return f"{label} ({bin_label})"


@dataclasses.dataclass(frozen=True)
class PackedParameterSpec:
    """
    Static description of a packed parameter vector.

    Parameters
    ----------
    analysis_kind
        Human-readable label such as ``"pk"``, ``"bk"``, or ``"bao"``.
    cosmo_keys, cosmo_sizes
        Static cosmological parameter metadata.
    survey_keys
        Flat survey/nuisance keys. Empty for BAO-only analyses.
    n_bins
        Number of survey bins for full-shape analyses.
    varied_idx
        Optional subset of packed indices actually present in the samples.
        Use this for chains that sample only a Fisher-selected varied block.
    bin_labels
        Optional labels for survey bins. Defaults to ``("bin0", "bin1", ...)``.
    """

    analysis_kind: str
    cosmo_keys: tuple[str, ...]
    cosmo_sizes: tuple[int, ...]
    survey_keys: tuple[
        tuple[str, str | None] | tuple[str, str, str | None], ...
    ] = ()
    n_bins: int = 1
    varied_idx: tuple[int, ...] | None = None
    bin_labels: tuple[str, ...] | None = None

    def __post_init__(self):
        if len(self.cosmo_keys) != len(self.cosmo_sizes):
            raise ValueError("cosmo_keys and cosmo_sizes must have the same length.")
        if self.n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {self.n_bins}.")
        if self.bin_labels is not None and len(self.bin_labels) != self.n_bins:
            raise ValueError(
                f"bin_labels length ({len(self.bin_labels)}) must match n_bins ({self.n_bins})."
            )
        if self.varied_idx is not None:
            n_total = self.n_total
            for idx in self.varied_idx:
                if idx < 0 or idx >= n_total:
                    raise ValueError(
                        f"varied_idx contains out-of-range index {idx} for n_total={n_total}."
                    )

    @property
    def n_cosmo(self) -> int:
        return int(sum(self.cosmo_sizes))

    @property
    def n_survey_per_bin(self) -> int:
        return int(len(self.survey_keys))

    @property
    def n_total(self) -> int:
        return self.n_cosmo + self.n_bins * self.n_survey_per_bin

    @property
    def resolved_bin_labels(self) -> tuple[str, ...]:
        if self.bin_labels is not None:
            return self.bin_labels
        return tuple(f"bin{i}" for i in range(self.n_bins))

    def full_param_names(self) -> tuple[str, ...]:
        names: list[str] = []

        for key, size in zip(self.cosmo_keys, self.cosmo_sizes):
            names.extend(_format_component_names(key, int(size)))

        if self.survey_keys:
            for bin_label in self.resolved_bin_labels:
                for survey_key in self.survey_keys:
                    base = _survey_base_name(survey_key)
                    names.append(_append_bin_suffix(base, bin_label, self.n_bins))

        return tuple(names)

    def full_param_labels(self) -> dict[str, str]:
        labels: dict[str, str] = {}

        for key, size in zip(self.cosmo_keys, self.cosmo_sizes):
            base_label = DEFAULT_LATEX_LABELS.get(key, key)
            for i, name in enumerate(_format_component_names(key, int(size))):
                labels[name] = base_label if size == 1 else f"{base_label}[{i}]"

        if self.survey_keys:
            for bin_label in self.resolved_bin_labels:
                for survey_key in self.survey_keys:
                    base_name = _survey_base_name(survey_key)
                    name = _append_bin_suffix(base_name, bin_label, self.n_bins)
                    base_label = DEFAULT_LATEX_LABELS.get(base_name, base_name)
                    labels[name] = _latex_with_bin(base_label, bin_label, self.n_bins)

        return labels

    def varied_param_names(self) -> tuple[str, ...]:
        names = self.full_param_names()
        if self.varied_idx is None:
            return names
        return tuple(names[i] for i in self.varied_idx)

    def varied_param_labels(self) -> dict[str, str]:
        labels = self.full_param_labels()
        names = self.varied_param_names()
        return {name: labels[name] for name in names}


def make_fullshape_spec(
    *,
    cosmo_keys: tuple[str, ...],
    cosmo_sizes: tuple[int, ...],
    survey_keys: tuple[
        tuple[str, str | None] | tuple[str, str, str | None], ...
    ],
    n_bins: int = 1,
    varied_idx: Sequence[int] | None = None,
    bin_labels: Sequence[str] | None = None,
    analysis_kind: str = "pk",
) -> PackedParameterSpec:
    """Build a packed-parameter spec for full-shape analyses."""
    return PackedParameterSpec(
        analysis_kind=analysis_kind,
        cosmo_keys=tuple(cosmo_keys),
        cosmo_sizes=tuple(int(size) for size in cosmo_sizes),
        survey_keys=tuple(survey_keys),
        n_bins=int(n_bins),
        varied_idx=None if varied_idx is None else tuple(int(i) for i in varied_idx),
        bin_labels=None if bin_labels is None else tuple(str(label) for label in bin_labels),
    )


def make_bao_spec(
    *,
    cosmo_keys: tuple[str, ...],
    cosmo_sizes: tuple[int, ...],
    varied_idx: Sequence[int] | None = None,
    analysis_kind: str = "bao",
) -> PackedParameterSpec:
    """Build a packed-parameter spec for BAO-only chains."""
    return PackedParameterSpec(
        analysis_kind=analysis_kind,
        cosmo_keys=tuple(cosmo_keys),
        cosmo_sizes=tuple(int(size) for size in cosmo_sizes),
        survey_keys=(),
        n_bins=1,
        varied_idx=None if varied_idx is None else tuple(int(i) for i in varied_idx),
        bin_labels=None,
    )


def packed_sample_labels(spec: PackedParameterSpec) -> dict[str, str]:
    """Return ``{var_name: latex_label}`` for the sampled parameter block."""
    return spec.varied_param_labels()


def packed_samples_to_dict(
    samples: Any,
    spec: PackedParameterSpec,
    *,
    chain_axis: int | None = 0,
    draw_axis: int = 1,
    param_axis: int = -1,
) -> dict[str, np.ndarray]:
    """
    Convert packed chain samples into ``{name: array(chain, draw)}``.

    Parameters
    ----------
    samples
        Packed samples with a parameter axis, typically shaped
        ``(chain, draw, n_params)`` or ``(draw, n_params)``.
    spec
        Packed vector specification.
    chain_axis, draw_axis, param_axis
        Axis positions in the input array.
    """
    arr = np.asarray(samples)
    if arr.ndim < 2:
        raise ValueError(
            f"samples must have at least 2 dimensions including the parameter axis; got {arr.shape}."
        )

    param_axis = _normalize_axis(param_axis, arr.ndim, name="param_axis")
    arr = np.moveaxis(arr, param_axis, -1)
    n_sampled = arr.shape[-1]
    names = spec.varied_param_names()

    if n_sampled != len(names):
        raise ValueError(
            f"samples parameter axis has length {n_sampled}, but spec expects {len(names)} "
            f"sampled parameters."
        )

    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        out[name] = _as_scalar_chain_draw(
            arr[..., i],
            chain_axis=chain_axis,
            draw_axis=draw_axis,
            var_name=name,
        )
    return out


def scalar_summary(
    samples: Mapping[str, Any] | Any,
    *,
    var_names: Sequence[str] | None = None,
    chain_axis: int | None = None,
    draw_axis: int = 0,
    burnin: int = 0,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
) -> dict[str, dict[str, float]]:
    """Compute summary statistics for scalar posterior variables."""
    sample_map = _as_mapping(samples, name="samples")
    if var_names is None:
        keys = list(sample_map)
    else:
        keys = list(var_names)

    qs = tuple(float(q) for q in quantiles)
    for q in qs:
        if q < 0.0 or q > 1.0:
            raise ValueError(f"quantiles must be in [0, 1], got {q!r}.")

    out: dict[str, dict[str, float]] = {}
    for name in keys:
        if name not in sample_map:
            raise KeyError(f"Variable {name!r} not found in samples.")
        arr = _as_scalar_chain_draw(
            sample_map[name],
            chain_axis=chain_axis,
            draw_axis=draw_axis,
            var_name=name,
        )
        if burnin < 0 or burnin >= arr.shape[1]:
            raise ValueError(
                f"burnin must satisfy 0 <= burnin < draws ({arr.shape[1]}), got {burnin}."
            )
        flat = arr[:, burnin:].reshape(-1)
        row = {
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
        }
        for q in qs:
            row[f"q{int(round(100 * q)):02d}"] = float(np.quantile(flat, q))
        out[name] = row
    return out


def scalar_summary_packed(
    samples: Any,
    spec: PackedParameterSpec,
    *,
    chain_axis: int | None = 0,
    draw_axis: int = 1,
    param_axis: int = -1,
    burnin: int = 0,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
) -> dict[str, dict[str, float]]:
    """Compute scalar posterior summaries for packed chain samples."""
    named = packed_samples_to_dict(
        samples,
        spec,
        chain_axis=chain_axis,
        draw_axis=draw_axis,
        param_axis=param_axis,
    )
    return scalar_summary(
        named,
        chain_axis=0,
        draw_axis=1,
        burnin=burnin,
        quantiles=quantiles,
    )


def to_inference_data(
    posterior: Mapping[str, Any] | Any,
    infos: Mapping[str, Any] | Any | None = None,
    *,
    chain_axis: int | None = None,
    draw_axis: int = 0,
    sample_stats_map: Mapping[str, str] | None = None,
    coords: Mapping[str, Sequence[Any]] | None = None,
    dims: Mapping[str, Sequence[str]] | None = None,
    **from_dict_kwargs: Any,
):
    """Convert sampler outputs into ArviZ ``InferenceData``."""
    az = _require_arviz()

    posterior_map = _as_mapping(posterior, name="posterior")
    posterior_group = _prepare_group(
        posterior_map,
        chain_axis=chain_axis,
        draw_axis=draw_axis,
    )

    sample_stats_group = None
    if infos is not None:
        info_map = _as_mapping(infos, name="infos")
        rename = dict(DEFAULT_SAMPLE_STATS_MAP)
        if sample_stats_map is not None:
            rename.update(sample_stats_map)
        mapped_infos = {rename.get(k, k): v for k, v in info_map.items()}
        sample_stats_group = _prepare_group(
            mapped_infos,
            chain_axis=chain_axis,
            draw_axis=draw_axis,
        )

    return az.from_dict(
        posterior=posterior_group,
        sample_stats=sample_stats_group,
        coords=coords,
        dims=dims,
        **from_dict_kwargs,
    )


def to_inference_data_packed(
    samples: Any,
    spec: PackedParameterSpec,
    infos: Mapping[str, Any] | Any | None = None,
    *,
    chain_axis: int | None = 0,
    draw_axis: int = 1,
    param_axis: int = -1,
    sample_stats_map: Mapping[str, str] | None = None,
    coords: Mapping[str, Sequence[Any]] | None = None,
    dims: Mapping[str, Sequence[str]] | None = None,
    **from_dict_kwargs: Any,
):
    """Convert packed samples plus optional diagnostics into ``InferenceData``."""
    posterior = packed_samples_to_dict(
        samples,
        spec,
        chain_axis=chain_axis,
        draw_axis=draw_axis,
        param_axis=param_axis,
    )
    return to_inference_data(
        posterior,
        infos=infos,
        chain_axis=0,
        draw_axis=1,
        sample_stats_map=sample_stats_map,
        coords=coords,
        dims=dims,
        **from_dict_kwargs,
    )


def diagnostics_summary(
    inference_data: Any,
    *,
    var_names: Sequence[str] | None = None,
    round_to: int = 2,
):
    """Return ArviZ diagnostics summary (ESS, R-hat, MCSE, etc.)."""
    az = _require_arviz()
    return az.summary(
        inference_data,
        var_names=var_names,
        kind="diagnostics",
        round_to=round_to,
    )


def ess_by_variable(
    inference_data: Any,
    *,
    var_names: Sequence[str] | None = None,
    method: str = "bulk",
    component_reduce: str = "mean",
) -> dict[str, float]:
    """Compute ArviZ ESS by variable and reduce non-scalar components."""
    az = _require_arviz()
    ess_ds = az.ess(inference_data, var_names=var_names, method=method)
    return {
        name: _reduce_components(np.asarray(data_array.values), component_reduce)
        for name, data_array in ess_ds.data_vars.items()
    }


def slice_draws(
    tree: Any,
    *,
    start: int | None = None,
    stop: int | None = None,
    step: int | None = None,
    draw_axis: int = 1,
):
    """Slice the draw axis of a sampler tree or array-like object."""

    def _slice_leaf(leaf):
        arr = np.asarray(leaf)
        if arr.ndim == 0:
            return leaf
        axis = _normalize_axis(draw_axis, arr.ndim, name="draw_axis")
        index = [slice(None)] * arr.ndim
        index[axis] = slice(start, stop, step)
        return arr[tuple(index)]

    return _tree_map(_slice_leaf, tree)


def discard_burnin(tree: Any, burnin: int, *, draw_axis: int = 1):
    """Drop the first ``burnin`` draws from every leaf in a sampler tree."""
    if burnin < 0:
        raise ValueError(f"burnin must be non-negative, got {burnin}.")
    return slice_draws(tree, start=burnin, draw_axis=draw_axis)


def validate_tuned_step_size(step_size: Any, *, label: str = "step_size") -> np.ndarray:
    """Validate tuned step sizes and return them as a NumPy array."""
    step = np.asarray(step_size, dtype=float)
    if np.any(~np.isfinite(step)):
        raise RuntimeError(f"{label}: non-finite tuned step_size encountered: {step.tolist()}")
    if np.any(step <= 0.0):
        raise RuntimeError(
            f"{label}: tuned step_size contains non-positive entries: {step.tolist()}."
        )
    return step


def lag1_autocorr_per_chain(
    values: Any,
    *,
    chain_axis: int | None = 0,
    draw_axis: int = 1,
    var_name: str = "values",
) -> np.ndarray:
    """Compute lag-1 autocorrelation for each chain of a scalar trace."""
    arr = _as_scalar_chain_draw(
        values,
        chain_axis=chain_axis,
        draw_axis=draw_axis,
        var_name=var_name,
    )
    out = []
    for chain_id in range(arr.shape[0]):
        x = arr[chain_id]
        if x.size < 2:
            out.append(np.nan)
            continue
        x0 = x[:-1] - x[:-1].mean()
        x1 = x[1:] - x[1:].mean()
        den = np.sqrt((x0 * x0).sum() * (x1 * x1).sum())
        out.append(float((x0 * x1).sum() / den) if den > 0 else np.nan)
    return np.asarray(out, dtype=float)


def mean_abs_jump_per_chain(
    values: Any,
    *,
    chain_axis: int | None = 0,
    draw_axis: int = 1,
    var_name: str = "values",
) -> np.ndarray:
    """Compute the mean absolute jump size for each chain of a scalar trace."""
    arr = _as_scalar_chain_draw(
        values,
        chain_axis=chain_axis,
        draw_axis=draw_axis,
        var_name=var_name,
    )
    if arr.shape[1] < 2:
        return np.full(arr.shape[0], np.nan, dtype=float)
    return np.mean(np.abs(np.diff(arr, axis=1)), axis=1)


def effective_sample_size(
    values: Any,
    *,
    chain_axis: int | None = 0,
    draw_axis: int = 1,
    var_name: str = "values",
    variance_tol: float = 1e-16,
) -> float:
    """Compute multi-chain effective sample size for a scalar trace."""
    arr = _as_scalar_chain_draw(
        values,
        chain_axis=chain_axis,
        draw_axis=draw_axis,
        var_name=var_name,
    )
    if arr.shape[1] < 4 or not np.all(np.isfinite(arr)):
        return float(np.nan)
    if np.all(np.var(arr, axis=1) <= variance_tol):
        return float(np.nan)

    ess_fn, jnp = _require_blackjax_effective_sample_size()
    ess = float(ess_fn(jnp.asarray(arr), chain_axis=0, sample_axis=1))
    n_total = arr.shape[0] * arr.shape[1]
    return float(np.clip(ess, 1.0, n_total)) if np.isfinite(ess) else float(np.nan)


def ess_per_chain(
    values: Any,
    *,
    chain_axis: int | None = 0,
    draw_axis: int = 1,
    var_name: str = "values",
    variance_tol: float = 1e-16,
) -> np.ndarray:
    """Compute single-chain ESS for each chain of a scalar trace."""
    arr = _as_scalar_chain_draw(
        values,
        chain_axis=chain_axis,
        draw_axis=draw_axis,
        var_name=var_name,
    )
    ess_fn, jnp = _require_blackjax_effective_sample_size()

    vals = []
    for chain_id in range(arr.shape[0]):
        x = np.asarray(arr[chain_id], dtype=float)
        if x.size < 4 or not np.all(np.isfinite(x)) or np.var(x) <= variance_tol:
            vals.append(np.nan)
            continue
        ess = float(ess_fn(jnp.asarray(x)[None, :], chain_axis=0, sample_axis=1))
        vals.append(float(np.clip(ess, 1.0, x.size)) if np.isfinite(ess) else np.nan)
    return np.asarray(vals, dtype=float)


def plot_trace(
    samples: Mapping[str, Any] | Any,
    *,
    var_names: Sequence[str],
    truths: Mapping[str, float] | None = None,
    var_labels: Mapping[str, str] | None = None,
    chain_axis: int | None = None,
    draw_axis: int = 0,
    burnin: int = 0,
    figsize: tuple[float, float] | None = None,
    alpha: float = 0.8,
):
    """Plot trace lines for scalar variables and return ``(fig, axes)``."""
    plt = _require_matplotlib()
    sample_map = _as_mapping(samples, name="samples")
    if not var_names:
        raise ValueError("var_names must be non-empty.")

    if figsize is None:
        figsize = (9.0, 2.2 * len(var_names))

    fig, axes = plt.subplots(len(var_names), 1, sharex=True, figsize=figsize)
    axes = np.atleast_1d(axes)

    for i, name in enumerate(var_names):
        if name not in sample_map:
            raise KeyError(f"Variable {name!r} not found in samples.")
        arr = _as_scalar_chain_draw(
            sample_map[name],
            chain_axis=chain_axis,
            draw_axis=draw_axis,
            var_name=name,
        )
        if burnin < 0 or burnin >= arr.shape[1]:
            raise ValueError(
                f"burnin must satisfy 0 <= burnin < draws ({arr.shape[1]}), got {burnin}."
            )
        arr = arr[:, burnin:]
        ax = axes[i]
        for chain_id in range(arr.shape[0]):
            ax.plot(arr[chain_id], lw=1.0, alpha=alpha, label=f"chain {chain_id}")
        if truths is not None and name in truths:
            ax.axhline(float(truths[name]), color="k", ls="--", lw=1.0, alpha=0.7)
        ax.set_ylabel(_var_label(name, var_labels))
        if arr.shape[0] <= 4:
            ax.legend(loc="upper right", fontsize=8, frameon=False)

    axes[-1].set_xlabel("draw")
    fig.tight_layout()
    return fig, axes


def plot_trace_packed(
    samples: Any,
    spec: PackedParameterSpec,
    *,
    var_names: Sequence[str] | None = None,
    truths: Mapping[str, float] | None = None,
    chain_axis: int | None = 0,
    draw_axis: int = 1,
    param_axis: int = -1,
    burnin: int = 0,
    figsize: tuple[float, float] | None = None,
    alpha: float = 0.8,
):
    """Plot traces directly from packed chain samples."""
    named = packed_samples_to_dict(
        samples,
        spec,
        chain_axis=chain_axis,
        draw_axis=draw_axis,
        param_axis=param_axis,
    )
    if var_names is None:
        var_names = list(named)
    return plot_trace(
        named,
        var_names=var_names,
        truths=truths,
        var_labels=spec.varied_param_labels(),
        chain_axis=0,
        draw_axis=1,
        burnin=burnin,
        figsize=figsize,
        alpha=alpha,
    )


def plot_corner(
    samples: Mapping[str, Any] | Any,
    *,
    var_names: Sequence[str],
    truths: Mapping[str, float] | None = None,
    var_labels: Mapping[str, str] | None = None,
    chain_axis: int | None = None,
    draw_axis: int = 0,
    burnin: int = 0,
    bins: int = 40,
    alpha: float = 0.18,
    chain_alpha: float = 0.35,
    combined_color: str = "tab:blue",
    chain_colors: Sequence[str] | None = None,
    show_legend: bool = True,
    max_points: int = 20_000,
    max_points_per_chain: int | None = None,
    random_seed: int = 0,
):
    """
    Plot a simple corner figure for scalar variables and return ``(fig, axes)``.

    The off-diagonal panels show scatter overlays; diagonal panels show 1-d
    histograms for the pooled samples plus optional per-chain outlines.
    """
    plt = _require_matplotlib()
    sample_map = _as_mapping(samples, name="samples")
    if not var_names:
        raise ValueError("var_names must be non-empty.")
    if max_points <= 0:
        raise ValueError(f"max_points must be positive, got {max_points!r}.")

    scalar: dict[str, np.ndarray] = {}
    for name in var_names:
        if name not in sample_map:
            raise KeyError(f"Variable {name!r} not found in samples.")
        arr = _as_scalar_chain_draw(
            sample_map[name],
            chain_axis=chain_axis,
            draw_axis=draw_axis,
            var_name=name,
        )
        if burnin < 0 or burnin >= arr.shape[1]:
            raise ValueError(
                f"burnin must satisfy 0 <= burnin < draws ({arr.shape[1]}), got {burnin}."
            )
        scalar[name] = arr[:, burnin:]

    nvar = len(var_names)
    n_chains = scalar[var_names[0]].shape[0]
    rng = np.random.default_rng(random_seed)

    def _subsample_rows(data: np.ndarray, n_max: int) -> np.ndarray:
        if data.shape[0] <= n_max:
            return data
        idx = rng.choice(data.shape[0], size=n_max, replace=False)
        return data[idx]

    combined_data = np.column_stack([scalar[name].reshape(-1) for name in var_names])
    combined_data = _subsample_rows(combined_data, max_points)

    if max_points_per_chain is None:
        max_points_per_chain = max(1, max_points // max(1, n_chains))

    if chain_colors is None:
        cmap = plt.get_cmap("tab10")
        chain_colors = [cmap(i % cmap.N) for i in range(n_chains)]
    else:
        chain_colors = list(chain_colors)
        if len(chain_colors) < n_chains:
            raise ValueError(
                f"chain_colors must have at least {n_chains} entries, got {len(chain_colors)}."
            )

    chain_data: list[np.ndarray] = []
    for c in range(n_chains):
        chain_matrix = np.column_stack([scalar[name][c].reshape(-1) for name in var_names])
        chain_data.append(_subsample_rows(chain_matrix, max_points_per_chain))

    fig, axes = plt.subplots(nvar, nvar, figsize=(2.4 * nvar, 2.4 * nvar))
    for i in range(nvar):
        for j in range(nvar):
            ax = axes[i, j]
            if i < j:
                ax.axis("off")
                continue

            if i == j:
                ax.hist(
                    combined_data[:, j],
                    bins=bins,
                    density=True,
                    color=combined_color,
                    alpha=0.75,
                    label=("combined" if i == 0 else None),
                )
                for c in range(n_chains):
                    ax.hist(
                        chain_data[c][:, j],
                        bins=bins,
                        density=True,
                        histtype="step",
                        color=chain_colors[c],
                        lw=1.2,
                        alpha=chain_alpha,
                        label=(f"chain {c}" if i == 0 else None),
                    )
                if truths is not None and var_names[j] in truths:
                    ax.axvline(float(truths[var_names[j]]), color="k", ls="--", lw=1.0)
            else:
                ax.scatter(
                    combined_data[:, j],
                    combined_data[:, i],
                    s=4,
                    alpha=alpha,
                    color=combined_color,
                    rasterized=True,
                )
                for c in range(n_chains):
                    ax.scatter(
                        chain_data[c][:, j],
                        chain_data[c][:, i],
                        s=4,
                        alpha=min(alpha, chain_alpha),
                        color=chain_colors[c],
                        rasterized=True,
                    )
                if truths is not None:
                    if var_names[j] in truths:
                        ax.axvline(float(truths[var_names[j]]), color="k", ls="--", lw=0.9)
                    if var_names[i] in truths:
                        ax.axhline(float(truths[var_names[i]]), color="k", ls="--", lw=0.9)

            if i == nvar - 1:
                ax.set_xlabel(_var_label(var_names[j], var_labels))
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(_var_label(var_names[i], var_labels))
            elif i != j:
                ax.set_yticklabels([])

    if show_legend:
        axes[0, 0].legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig, axes


def plot_corner_packed(
    samples: Any,
    spec: PackedParameterSpec,
    *,
    var_names: Sequence[str] | None = None,
    truths: Mapping[str, float] | None = None,
    chain_axis: int | None = 0,
    draw_axis: int = 1,
    param_axis: int = -1,
    burnin: int = 0,
    bins: int = 40,
    alpha: float = 0.18,
    chain_alpha: float = 0.35,
    combined_color: str = "tab:blue",
    chain_colors: Sequence[str] | None = None,
    show_legend: bool = True,
    max_points: int = 20_000,
    max_points_per_chain: int | None = None,
    random_seed: int = 0,
):
    """Plot a corner figure directly from packed chain samples."""
    named = packed_samples_to_dict(
        samples,
        spec,
        chain_axis=chain_axis,
        draw_axis=draw_axis,
        param_axis=param_axis,
    )
    if var_names is None:
        var_names = list(named)
    return plot_corner(
        named,
        var_names=var_names,
        truths=truths,
        var_labels=spec.varied_param_labels(),
        chain_axis=0,
        draw_axis=1,
        burnin=burnin,
        bins=bins,
        alpha=alpha,
        chain_alpha=chain_alpha,
        combined_color=combined_color,
        chain_colors=chain_colors,
        show_legend=show_legend,
        max_points=max_points,
        max_points_per_chain=max_points_per_chain,
        random_seed=random_seed,
    )


def _validate_levels(levels) -> list[float]:
    """Ascending, de-duplicated, strictly-in-(0,1) probability masses."""
    values = [float(v) for v in levels]
    if not values:
        raise ValueError("levels must be non-empty.")
    for v in values:
        if not 0.0 < v < 1.0:
            raise ValueError(
                f"levels must be strictly between 0 and 1, got {v!r}.")
    # ArviZ requires ascending hdi_probs and raises otherwise.
    return sorted(set(values))


def plot_credible_contours(
    x,
    y,
    *,
    ax=None,
    levels=(0.68, 0.95),
    colors=None,
    linestyles=("-", "--"),
    linewidths=1.6,
    fill=False,
    fill_alpha=0.25,
    **contour_kwargs,
):
    """Draw 2-D credible-region contours for a pair of chain variables.

    Contours enclose ``levels`` of the JOINT (2-D) posterior mass, computed by
    ArviZ's highest-density KDE (``arviz.plot_kde(..., hdi_probs=...)``). This
    is the same convention as ``plotting.plot_contours(..., level_kind=
    "mass2d")``, so the two may be overlaid directly. It is NOT the same as
    that function's DEFAULT, which draws 1-sigma/2-sigma ellipses enclosing
    39.35%/86.47% -- overlaying these contours on default ellipses makes the
    chain look ~1.5x wider than the forecast for no physical reason.

    Parameters
    ----------
    x, y : array-like
        Samples for the two parameters. Either ``(chain, draw)`` or already
        flattened; both are pooled before the KDE. Shapes must match.
    ax : matplotlib Axes, optional
        Target axis; defaults to the current axis.
    levels : sequence of float, optional
        Probability masses to enclose. Default ``(0.68, 0.95)``. Sorted
        ascending internally (ArviZ requires it).
    colors : str or sequence, optional
        Contour colour(s), passed through to matplotlib.
    linestyles, linewidths : optional
        Per-level line styling.
    fill : bool, optional
        If True, also shade the regions (``contourf``) at ``fill_alpha``.
    **contour_kwargs
        Forwarded to ``arviz.plot_kde``'s ``contour_kwargs``.

    Returns
    -------
    matplotlib Axes
        The axis drawn on, for chaining.
    """
    az = _require_arviz()
    plt = _require_matplotlib()

    probs = _validate_levels(levels)
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.shape != ya.shape:
        raise ValueError(
            f"x and y must have the same shape, got {xa.shape} and {ya.shape}.")

    if ax is None:
        ax = plt.gca()

    kwargs = dict(contour_kwargs)
    if colors is not None:
        kwargs["colors"] = colors
    if linestyles is not None:
        kwargs["linestyles"] = list(linestyles)[: len(probs)]
    if linewidths is not None:
        kwargs["linewidths"] = linewidths

    az.plot_kde(
        xa.reshape(-1),
        ya.reshape(-1),
        contour=True,
        hdi_probs=probs,
        contour_kwargs=kwargs,
        contourf_kwargs={"alpha": fill_alpha if fill else 0.0},
        ax=ax,
    )
    return ax


def credible_intervals(
    samples: Mapping[str, Any] | Any,
    *,
    var_names: Sequence[str],
    levels=(0.68, 0.95),
    chain_axis: int | None = 0,
    draw_axis: int = 1,
) -> dict[str, dict[float, tuple[float, float]]]:
    """Highest-density credible intervals per variable, per level.

    These are 1-D intervals -- the corner plot's DIAGONAL panels. Note the
    asymmetry with the off-diagonals: the 68% 1-D interval is the 1-sigma
    interval, whereas the 68% 2-D contour sits at 1.5096 sigma.

    ``chain_axis``/``draw_axis`` default to ``0``/``1``, i.e. each variable's
    samples are expected as ``(n_chains, n_draws)`` -- the convention used
    throughout this module's ``_packed`` helpers. Chain and draw are pooled
    into one flat array before calling ``arviz.hdi`` (a bare 2-D array is
    read by ArviZ as ``(draw, shape)``, not ``(chain, draw)``).

    Returns
    -------
    dict
        ``{var_name: {level: (lower, upper)}}`` with plain floats.
    """
    az = _require_arviz()
    sample_map = _as_mapping(samples, name="samples")
    probs = _validate_levels(levels)

    out: dict[str, dict[float, tuple[float, float]]] = {}
    for name in var_names:
        if name not in sample_map:
            raise KeyError(f"Variable {name!r} not found in samples.")
        arr = _as_scalar_chain_draw(
            sample_map[name],
            chain_axis=chain_axis,
            draw_axis=draw_axis,
            var_name=name,
        )
        per_level: dict[float, tuple[float, float]] = {}
        for p in probs:
            # az.hdi interprets a bare 2-D array as (draw, shape), not
            # (chain, draw) -- pool chain and draw into one flat 1-D array
            # so the interval is computed over all combined draws.
            bounds = np.asarray(az.hdi(arr.reshape(-1), hdi_prob=p)).reshape(-1)
            per_level[p] = (float(bounds[0]), float(bounds[1]))
        out[name] = per_level
    return out
