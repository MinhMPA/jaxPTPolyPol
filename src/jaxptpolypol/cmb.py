"""
`candl`-based CMB likelihood adapters.

This module provides a single CMB interface for `jaxptpolypol` built around
the differentiable `candl` likelihood framework. The main entry points are:

1. `load_candl_likelihood(...)` to construct a native `candl` or wrapped
   `clipy` likelihood object.
2. `make_candl_pars_to_theory_specs_fn(...)` to build a theory callable that
   returns `Dl` spectra for the likelihood.
3. `make_candl_loglike_fn(...)` and `make_candl_theory_vector_fn(...)` to
   expose packed-parameter closures compatible with the existing Fisher and
   sampler tooling in this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .params import CosmoParams

__all__ = [
    "PLANCK_TAU_PRIOR_SIGMA",
    "add_tau_prior_to_fisher",
    "assert_zero_gradient",
    "CandlParameterLayout",
    "build_candl_parameter_layout",
    "get_candl_default_parameters",
    "get_candl_parameter_names",
    "load_candl_likelihood",
    "make_candl_loglike_fn",
    "make_candl_pars_to_theory_specs_fn",
    "make_candl_theory_vector_fn",
    "make_joint_loglike_fn",
]


_DATA_SELECTION_UNSET = object()
_TCMB_UK = 2.7255e6
_TCMB_UK_SQ = _TCMB_UK**2


def _ordered_unique(names: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return tuple(ordered)


def _as_scalar_or_array(value: Any) -> jnp.ndarray:
    arr = jnp.asarray(value)
    return arr.reshape(()) if arr.size == 1 else arr


def _resolve_dataset_reference(data_set_file: Any) -> Any:
    if not isinstance(data_set_file, str):
        return data_set_file
    if "/" in data_set_file or data_set_file.endswith((".yaml", ".clik")):
        return data_set_file
    if data_set_file.count(".") != 1:
        return data_set_file

    module_name, attr_name = data_set_file.split(".", 1)
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return data_set_file
    return getattr(module, attr_name, data_set_file)


def _require_candl():
    try:
        return importlib.import_module("candl")
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "candl is required for the CMB interface. Install `candl-like` and "
            "a compatible dataset package such as `candl_data`."
        ) from exc


def _clear_like_priors(
    likelihood: Any,
    *,
    clear_internal_priors: bool,
    clear_specific_priors: str | Sequence[str] | None,
) -> Any:
    if not hasattr(likelihood, "priors"):
        return likelihood

    if clear_internal_priors:
        likelihood.priors = []
        return likelihood

    if not clear_specific_priors:
        return likelihood

    ignored = (
        (clear_specific_priors,)
        if isinstance(clear_specific_priors, str)
        else tuple(clear_specific_priors)
    )
    filtered = []
    for prior in likelihood.priors:
        par_names = tuple(getattr(prior, "par_names", ()))
        if any(name in par_names for name in ignored):
            continue
        filtered.append(prior)
    likelihood.priors = filtered
    return likelihood


def get_candl_parameter_names(
    likelihood: Any,
    *,
    cosmo_keys: Sequence[str] = (),
    include_prior_params: bool = True,
) -> tuple[str, ...]:
    """
    Return ordered non-cosmology scalar parameter names required by a `candl` likelihood.

    Parameters
    ----------
    likelihood
        Native `candl` or supported wrapper likelihood object.
    cosmo_keys
        Cosmological parameter names already supplied by the shared packed
        cosmology block. These are excluded from the returned tuple.
    include_prior_params
        If *True*, include non-cosmology parameters that only enter through
        internal priors in addition to the nuisance/data-model parameters.
    """
    excluded = {str(name) for name in cosmo_keys}

    names = list(getattr(likelihood, "required_nuisance_parameters", ()) or ())
    if include_prior_params:
        names.extend(getattr(likelihood, "required_prior_parameters", ()) or ())

    ordered = _ordered_unique(str(name) for name in names)
    return tuple(name for name in ordered if name not in excluded)


def get_candl_default_parameters(
    likelihood: Any,
    *,
    include_cosmology: bool = False,
) -> dict[str, Any]:
    """
    Return default scalar parameters exposed by a `candl` or `clipy` likelihood.

    The returned mapping excludes the `Dl` theory payload. By default it keeps
    only nuisance / prior-only scalar parameters so it can seed fixed or
    sampled nuisance blocks in Fisher and sampling workflows.
    """
    default_par = getattr(likelihood, "default_par", None)
    if not hasattr(default_par, "items"):
        return {}

    allowed = None
    if not include_cosmology:
        allowed = set(get_candl_parameter_names(likelihood, include_prior_params=True))

    result: dict[str, Any] = {}
    for name, value in default_par.items():
        if name == "Dl":
            continue
        if allowed is not None and name not in allowed:
            continue
        result[str(name)] = _as_scalar_or_array(value)
    return result


# Planck 2018 TT,TE,EE+lowE+lensing constraint on the reionization optical
# depth (arXiv:1807.06209, Table 2). Used as a Gaussian prior in Hessian-based
# CMB Fishers because clipy's simall low-ell EE likelihood has an identically
# zero JAX gradient/Hessian (integer table lookup); see
# ~/candl/clipy/fix_simall_grad.md for the upstream fix. Pair every use with
# `assert_zero_gradient` on the simall term so the prior self-retires when
# that fix lands.
PLANCK_TAU_PRIOR_SIGMA = 0.0073


def add_tau_prior_to_fisher(fisher, param_order, tau_prior_sigma):
    """Return a copy of `fisher` with a Gaussian tau prior added on the diagonal.

    `param_order` names the packed parameter axes. If `tau_prior_sigma` is
    None the input is returned unchanged (as an ndarray) so callers can
    switch the prior off without a separate code path.
    """
    fisher = np.asarray(fisher)
    if tau_prior_sigma is None:
        return fisher
    if tau_prior_sigma <= 0:
        raise ValueError("tau_prior_sigma must be positive")
    tau_index = tuple(param_order).index("tau")
    out = np.array(fisher, copy=True)
    out[tau_index, tau_index] += 1.0 / tau_prior_sigma**2
    return out


def assert_zero_gradient(loglike_fn, theta, *, atol=1e-12, name="loglike"):
    """Raise if `loglike_fn` has a nonzero JAX gradient at `theta`.

    Tripwire for the tau-prior workaround: it must fail loudly the moment
    the clipy simall likelihood becomes differentiable, because then adding
    PLANCK_TAU_PRIOR_SIGMA on top would double-count the lowE information.
    """
    grad = jax.grad(loglike_fn)(theta)
    max_abs = float(jnp.max(jnp.abs(jnp.asarray(grad))))
    if max_abs > atol:
        raise RuntimeError(
            f"{name} has a nonzero gradient (max |g| = {max_abs:.3e}): "
            "the simall zero-gradient assumption no longer holds; remove the "
            "Gaussian tau prior to avoid double-counting lowE information "
            "(see ~/candl/clipy/fix_simall_grad.md)."
        )


@dataclass(frozen=True)
class CandlParameterLayout:
    """Static packed-parameter layout for shared cosmology plus CMB nuisances."""

    cosmo_keys: tuple[str, ...]
    cosmo_sizes: tuple[int, ...]
    cmb_nuisance_names: tuple[str, ...] = ()
    cmb_nuisance_offset: int | None = None

    @property
    def n_cosmo(self) -> int:
        return int(sum(self.cosmo_sizes))

    @property
    def n_cmb_nuisance(self) -> int:
        return len(self.cmb_nuisance_names)

    @property
    def size(self) -> int:
        return max(self.n_cosmo, self.nuisance_offset + self.n_cmb_nuisance)

    @property
    def nuisance_offset(self) -> int:
        return self.n_cosmo if self.cmb_nuisance_offset is None else int(self.cmb_nuisance_offset)

    def split(self, params: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
        arr = jnp.asarray(params)
        offset = self.nuisance_offset
        return arr[: self.n_cosmo], arr[offset : offset + self.n_cmb_nuisance]

    def unpack(self, params: Any) -> tuple[CosmoParams, dict[str, jnp.ndarray]]:
        cosmo_values, nuisance_values = self.split(params)
        cosmo = CosmoParams.from_array(cosmo_values, self.cosmo_keys, self.cosmo_sizes)
        nuisance = {
            name: _as_scalar_or_array(value)
            for name, value in zip(self.cmb_nuisance_names, nuisance_values)
        }
        return cosmo, nuisance

    def pack(
        self,
        cosmo: CosmoParams,
        cmb_nuisance: Mapping[str, Any] | None = None,
    ) -> jnp.ndarray:
        if tuple(cosmo.param_keys) != tuple(self.cosmo_keys):
            raise ValueError(
                "CosmoParams key order "
                f"{tuple(cosmo.param_keys)} does not match layout.cosmo_keys "
                f"{tuple(self.cosmo_keys)}; pack() fills the cosmology block "
                "by insertion order, so a mismatch scrambles parameter values."
            )
        nuisance = dict(cmb_nuisance or {})
        packed = jnp.zeros(self.size, dtype=jnp.asarray(cosmo.to_array()).dtype)
        packed = packed.at[: self.n_cosmo].set(cosmo.to_array())
        if self.cmb_nuisance_names:
            offset = self.nuisance_offset
            packed = packed.at[offset : offset + self.n_cmb_nuisance].set(
                jnp.asarray(
                    [
                        _as_scalar_or_array(nuisance[name])
                        for name in self.cmb_nuisance_names
                    ]
                )
            )
        return packed


def build_candl_parameter_layout(
    likelihood: Any,
    *,
    cosmo_keys: Sequence[str],
    cosmo_sizes: Sequence[int],
    sampled_cmb_params: Sequence[str] | None = None,
    include_prior_params: bool = True,
    cmb_nuisance_offset: int | None = None,
) -> CandlParameterLayout:
    """
    Build a packed layout for a shared cosmology block plus sampled CMB scalars.
    """
    inferred = get_candl_parameter_names(
        likelihood,
        cosmo_keys=tuple(cosmo_keys),
        include_prior_params=include_prior_params,
    )
    sampled = tuple(sampled_cmb_params) if sampled_cmb_params is not None else inferred
    missing = [name for name in sampled if name not in inferred]
    if missing:
        raise ValueError(
            "sampled_cmb_params contains names not required by the likelihood: "
            f"{missing!r}"
        )
    return CandlParameterLayout(
        cosmo_keys=tuple(cosmo_keys),
        cosmo_sizes=tuple(int(size) for size in cosmo_sizes),
        cmb_nuisance_names=sampled,
        cmb_nuisance_offset=cmb_nuisance_offset,
    )


def load_candl_likelihood(
    data_set_file: Any,
    *,
    lensing: bool = False,
    wrapper: str | None = None,
    variant: str | None = None,
    feedback: bool = True,
    add_logdet: bool = False,
    data_selection: Any = _DATA_SELECTION_UNSET,
    clear_internal_priors: bool = False,
    clear_specific_priors: str | Sequence[str] | None = None,
    additional_args: Mapping[str, Any] | None = None,
) -> Any:
    """
    Load a `candl` likelihood or supported wrapper likelihood.

    Parameters mirror the native `candl` initialization path closely so this
    adapter can cover the full set of likelihood families and dataset sources
    exposed by the installed `candl` ecosystem.

    For `candl` index files, `variant=None` follows the upstream default in the
    index YAML. In particular, `candl_data.ACT_DR6_Lens` currently resolves to
    the same concrete dataset as `candl_data.ACT_DR6_Lens_only`; use
    `variant="use_CMB"` or `candl_data.ACT_DR6_Lens_and_CMB` for the combined
    ACT DR6 lensing plus primary-CMB likelihood.
    """
    candl = _require_candl()
    data_set_file = _resolve_dataset_reference(data_set_file)
    extra = dict(additional_args or {})

    wrapper_name = wrapper
    if wrapper_name is None and isinstance(data_set_file, str) and data_set_file.endswith(".clik"):
        wrapper_name = "clipy"

    if wrapper_name is not None:
        if wrapper_name != "clipy":
            raise ValueError(f"unsupported candl wrapper {wrapper_name!r}")
        try:
            clipy = importlib.import_module("clipy")
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "clipy is required for wrapper='clipy'. Install `clipy-like` and "
                "the corresponding Planck data files."
            ) from exc
        likelihood = clipy.clik_candl(data_set_file, **extra)
        if not hasattr(likelihood, "required_prior_parameters"):
            prior_names: list[str] = []
            prior_map = getattr(likelihood, "_prior", {})
            for name, value in prior_map.items():
                if isinstance(value, (list, tuple)):
                    prior_names.extend(str(item) for item in value)
                elif isinstance(value, str):
                    prior_names.append(str(value))
            likelihood.required_prior_parameters = _ordered_unique(prior_names)
        return likelihood

    init_args: dict[str, Any] = {
        "variant": variant,
        "feedback": feedback,
        "add_logdet": add_logdet,
        **extra,
    }
    if data_selection is not _DATA_SELECTION_UNSET:
        init_args["data_selection"] = data_selection

    likelihood_cls = candl.LensLike if lensing else candl.Like
    likelihood = likelihood_cls(data_set_file, **init_args)
    return _clear_like_priors(
        likelihood,
        clear_internal_priors=clear_internal_priors,
        clear_specific_priors=clear_specific_priors,
    )


def make_candl_pars_to_theory_specs_fn(
    *,
    emulator_filenames: Mapping[str, str] | None = None,
    pars_to_theory_specs: Callable[[Mapping[str, Any], int, int], Mapping[str, Any]] | None = None,
):
    """
    Return a `candl`-compatible theory callable producing `Dl` spectra.

    By default this wraps `candl.interface.get_CosmoPowerJAX_pars_to_theory_specs_func`.
    Pass `pars_to_theory_specs` directly to use a custom theory backend while
    keeping the packed-vector likelihood API in this module unchanged.
    """
    if pars_to_theory_specs is not None and emulator_filenames is not None:
        raise ValueError(
            "pass either `pars_to_theory_specs` or `emulator_filenames`, not both"
        )
    if pars_to_theory_specs is not None:
        def wrapped_pars_to_theory_specs(
            pars: Mapping[str, Any],
            ell_max: int,
            ell_min: int,
        ) -> Mapping[str, Any]:
            theory = dict(pars_to_theory_specs(pars, ell_max, ell_min))
            if "pp" in theory and "kk" not in theory:
                theory["kk"] = jnp.asarray(theory["pp"]) * (jnp.pi / 2.0)
            if "kk" in theory and "pp" not in theory:
                theory["pp"] = jnp.asarray(theory["kk"]) * (2.0 / jnp.pi)
            return theory

        return wrapped_pars_to_theory_specs
    if emulator_filenames is None:
        raise ValueError(
            "either `pars_to_theory_specs` or `emulator_filenames` must be provided"
        )

    _require_candl()
    try:
        cpj_module = importlib.import_module("cosmopower_jax.cosmopower_jax")
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "cosmopower-jax is required to build the default `candl` theory "
            "backend. Install `cosmopower-jax` or pass `pars_to_theory_specs` "
            "directly."
        ) from exc

    cpj_cls = getattr(cpj_module, "CosmoPowerJAX")
    cp_emulators: dict[str, Any] = {}
    cp_parameter_names: dict[str, tuple[str, ...]] = {}
    cp_apply_log_after_predict: dict[str, bool] = {}
    cp_outputs_are_dls: dict[str, bool] = {}
    for spec_type, filename in dict(emulator_filenames).items():
        try:
            metadata = np.load(filename, allow_pickle=True)
            uses_pca = any(
                key in metadata
                for key in ("n_pcas", "pca_transform_matrix", "pca_matrix")
            )
        except Exception:
            uses_pca = spec_type == "TE"

        probe = "custom_pca" if uses_pca else "custom_log"
        try:
            emulator = cpj_cls(probe=probe, filename=filename)
        except Exception:
            emulator = cpj_cls(probe=probe, filepath=filename)
        cp_emulators[spec_type] = emulator
        cp_apply_log_after_predict[spec_type] = bool(
            uses_pca and str(spec_type).lower() in {"pp", "kk"}
        )
        cp_outputs_are_dls[spec_type] = bool(
            uses_pca and str(spec_type).lower() in {"pp", "kk"}
        )
        cp_parameter_names[spec_type] = tuple(
            "H0" if str(name) == "h" else str(name)
            for name in emulator.parameters
        )

    def pars_to_theory_specs_with_aliases(
        pars: Mapping[str, Any],
        ell_max: int,
        ell_min: int,
    ) -> Mapping[str, Any]:
        theory: dict[str, Any] = {"ell": np.arange(ell_min, ell_max + 1)}
        for spec_type, emulator in cp_emulators.items():
            cp_pars = cp_parameter_names[spec_type]
            pars_for_cp = jnp.zeros(emulator.n_parameters)
            for index, par_name in enumerate(cp_pars):
                value = pars[par_name]
                if par_name == "H0":
                    value = jnp.asarray(value) / 100.0
                pars_for_cp = pars_for_cp.at[index].set(jnp.asarray(value))

            n_ell = ell_max - ell_min + 1
            theory_start_ix = int(max(int(emulator.modes[0]), ell_min) - int(emulator.modes[0]))
            theory_stop_ix = int(min(int(emulator.modes[-1]), ell_max) + 1 - int(emulator.modes[0]))
            like_start_ix = int(max(int(emulator.modes[0]), ell_min) - ell_min)
            like_stop_ix = int(min(int(emulator.modes[-1]), ell_max) + 1 - ell_min)

            prediction = emulator.predict(pars_for_cp).ravel()
            if cp_apply_log_after_predict[spec_type]:
                prediction = 10**prediction

            if cp_outputs_are_dls[spec_type] and str(spec_type).lower() == "pp":
                lensing_weight = emulator.modes * (emulator.modes + 1)
                pp_dl = prediction / lensing_weight
                kk_dl = prediction * (jnp.pi / 2.0)

                theory["pp"] = jnp.zeros(n_ell, dtype=pp_dl.dtype)
                theory["pp"] = theory["pp"].at[
                    like_start_ix:like_stop_ix
                ].set(pp_dl[theory_start_ix:theory_stop_ix])

                theory["kk"] = jnp.zeros(n_ell, dtype=kk_dl.dtype)
                theory["kk"] = theory["kk"].at[
                    like_start_ix:like_stop_ix
                ].set(kk_dl[theory_start_ix:theory_stop_ix])
                continue

            if str(spec_type).lower() in {"tt", "te", "ee", "bb"}:
                this_dl = prediction * _TCMB_UK_SQ
            elif cp_outputs_are_dls[spec_type]:
                this_dl = prediction
            else:
                this_dl = (
                    prediction
                    * emulator.modes
                    * (emulator.modes + 1)
                    / (2 * jnp.pi)
                )
            theory[spec_type] = jnp.zeros(n_ell, dtype=this_dl.dtype)
            theory[spec_type] = theory[spec_type].at[
                like_start_ix:like_stop_ix
            ].set(this_dl[theory_start_ix:theory_stop_ix])
        if "pp" in theory and "kk" not in theory:
            theory["kk"] = jnp.asarray(theory["pp"]) * (jnp.pi / 2.0)
        if "kk" in theory and "pp" not in theory:
            theory["pp"] = jnp.asarray(theory["kk"]) * (2.0 / jnp.pi)
        return theory

    return pars_to_theory_specs_with_aliases


def _build_parameter_mapping(
    cosmo: CosmoParams,
    *,
    sampled_cmb_params: Mapping[str, Any] | None = None,
    fixed_cmb_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    mapping = {
        name: _as_scalar_or_array(value)
        for name, value in cosmo.to_dict().items()
    }
    for source in (fixed_cmb_params or {}, sampled_cmb_params or {}):
        for name, value in source.items():
            mapping[str(name)] = _as_scalar_or_array(value)
    return mapping


def _build_theory_dls(
    likelihood: Any,
    parameter_mapping: Mapping[str, Any],
    pars_to_theory_specs: Callable[[Mapping[str, Any], int, int], Mapping[str, Any]],
) -> Mapping[str, Any]:
    theory = dict(
        pars_to_theory_specs(
            parameter_mapping,
            int(likelihood.ell_max),
            int(likelihood.ell_min),
        )
    )
    if "pp" in theory and "kk" not in theory:
        theory["kk"] = jnp.asarray(theory["pp"]) * (jnp.pi / 2.0)
    if "kk" in theory and "pp" not in theory:
        theory["pp"] = jnp.asarray(theory["kk"]) * (2.0 / jnp.pi)
    return theory


def _build_model_vector(
    likelihood: Any,
    parameter_mapping: Mapping[str, Any],
    pars_to_theory_specs: Callable[[Mapping[str, Any], int, int], Mapping[str, Any]],
) -> jnp.ndarray:
    dls = _build_theory_dls(likelihood, parameter_mapping, pars_to_theory_specs)
    params_with_dls = dict(parameter_mapping)
    params_with_dls["Dl"] = dls
    model_unbinned = likelihood.get_model_specs(params_with_dls)
    model_binned = model_unbinned if _is_lensing_like(likelihood) else likelihood.bin_model_specs(model_unbinned)
    return jnp.asarray(model_binned)


def _is_lensing_like(likelihood: Any) -> bool:
    cls_name = type(likelihood).__name__
    if cls_name == "LensLike":
        return True
    try:
        candl = importlib.import_module("candl")
    except Exception:  # pragma: no cover - optional dependency
        return False
    return isinstance(likelihood, getattr(candl, "LensLike", ()))


def make_candl_loglike_fn(
    likelihood: Any,
    *,
    pars_to_theory_specs: Callable[[Mapping[str, Any], int, int], Mapping[str, Any]],
    layout: CandlParameterLayout | None = None,
    cosmo_keys: Sequence[str] | None = None,
    cosmo_sizes: Sequence[int] | None = None,
    sampled_cmb_params: Sequence[str] = (),
    cmb_nuisance_offset: int | None = None,
    fixed_cmb_params: Mapping[str, Any] | None = None,
    jit_compile: bool = False,
) -> Callable[[Any], jnp.ndarray]:
    """
    Build a packed-parameter `candl` log-likelihood closure.
    """
    if layout is None:
        if cosmo_keys is None or cosmo_sizes is None:
            raise ValueError(
                "either `layout` or both `cosmo_keys` and `cosmo_sizes` are required"
            )
        layout = CandlParameterLayout(
            cosmo_keys=tuple(cosmo_keys),
            cosmo_sizes=tuple(int(size) for size in cosmo_sizes),
            cmb_nuisance_names=tuple(sampled_cmb_params),
            cmb_nuisance_offset=cmb_nuisance_offset,
        )

    fixed_params = dict(fixed_cmb_params or {})

    def loglike_fn(params: Any) -> jnp.ndarray:
        cosmo, sampled = layout.unpack(params)
        mapping = _build_parameter_mapping(
            cosmo,
            sampled_cmb_params=sampled,
            fixed_cmb_params=fixed_params,
        )
        dls = _build_theory_dls(likelihood, mapping, pars_to_theory_specs)
        params_with_dls = dict(mapping)
        params_with_dls["Dl"] = dls
        return jnp.asarray(likelihood.log_like(params_with_dls))

    return jax.jit(loglike_fn) if jit_compile else loglike_fn


def make_candl_theory_vector_fn(
    likelihood: Any,
    *,
    pars_to_theory_specs: Callable[[Mapping[str, Any], int, int], Mapping[str, Any]],
    layout: CandlParameterLayout | None = None,
    cosmo_keys: Sequence[str] | None = None,
    cosmo_sizes: Sequence[int] | None = None,
    sampled_cmb_params: Sequence[str] = (),
    cmb_nuisance_offset: int | None = None,
    fixed_cmb_params: Mapping[str, Any] | None = None,
    jit_compile: bool = False,
) -> Callable[[Any], jnp.ndarray]:
    """
    Build a packed-parameter closure returning the transformed/binned CMB theory vector.
    """
    if layout is None:
        if cosmo_keys is None or cosmo_sizes is None:
            raise ValueError(
                "either `layout` or both `cosmo_keys` and `cosmo_sizes` are required"
            )
        layout = CandlParameterLayout(
            cosmo_keys=tuple(cosmo_keys),
            cosmo_sizes=tuple(int(size) for size in cosmo_sizes),
            cmb_nuisance_names=tuple(sampled_cmb_params),
            cmb_nuisance_offset=cmb_nuisance_offset,
        )

    fixed_params = dict(fixed_cmb_params or {})

    def theory_vector_fn(params: Any) -> jnp.ndarray:
        cosmo, sampled = layout.unpack(params)
        mapping = _build_parameter_mapping(
            cosmo,
            sampled_cmb_params=sampled,
            fixed_cmb_params=fixed_params,
        )
        return _build_model_vector(likelihood, mapping, pars_to_theory_specs)

    return jax.jit(theory_vector_fn) if jit_compile else theory_vector_fn


def make_joint_loglike_fn(
    *,
    lss_loglike_fn: Callable[[Any], Any] | None = None,
    bao_loglike_fn: Callable[[Any], Any] | None = None,
    cmb_loglike_fn: Callable[[Any], Any] | None = None,
    extra_loglike_fns: Sequence[Callable[[Any], Any]] = (),
    log_prior_fn: Callable[[Any], Any] | None = None,
) -> Callable[[Any], jnp.ndarray]:
    """
    Sum optional probe log-likelihood terms into one scalar closure.
    """
    terms = [fn for fn in (lss_loglike_fn, bao_loglike_fn, cmb_loglike_fn) if fn is not None]
    terms.extend(extra_loglike_fns)

    def joint_loglike_fn(params: Any) -> jnp.ndarray:
        total = jnp.asarray(0.0)
        for fn in terms:
            total = total + jnp.asarray(fn(params))
        if log_prior_fn is not None:
            total = total + jnp.asarray(log_prior_fn(params))
        return total

    return joint_loglike_fn
