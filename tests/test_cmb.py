from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from jaxptpolypol.cmb import (
    CandlParameterLayout,
    build_candl_parameter_layout,
    get_candl_parameter_names,
    load_candl_likelihood,
    make_candl_loglike_fn,
    make_candl_pars_to_theory_specs_fn,
    make_candl_theory_vector_fn,
    make_joint_loglike_fn,
)
from jaxptpolypol.params import CosmoParams


class FakePrior:
    def __init__(self, par_names):
        self.par_names = tuple(par_names)


class FakeLikelihood:
    def __init__(self, data_set_file=None, **kwargs):
        self.data_set_file = data_set_file
        self.kwargs = kwargs
        self.ell_min = 2
        self.ell_max = 4
        self.required_nuisance_parameters = ["A_planck", "dust_amp"]
        self.required_prior_parameters = ["tau", "A_planck", "calib"]
        self.priors = [
            FakePrior(("A_planck", "calib")),
            FakePrior(("tau",)),
        ]

    def log_like(self, params):
        theory = params["Dl"]["TT"]
        return -(
            jnp.sum(theory)
            + params.get("A_planck", 0.0)
            + params.get("dust_amp", 0.0)
            + params.get("calib", 0.0)
        )

    def get_model_specs(self, params):
        return params["Dl"]["TT"] + params.get("A_planck", 0.0)

    def bin_model_specs(self, model_specs):
        return jnp.asarray([jnp.sum(model_specs), jnp.max(model_specs)])


class FakeLensLikelihood(FakeLikelihood):
    pass


def _fake_theory_fn(pars, ell_max, ell_min):
    assert ell_min == 2
    assert ell_max == 4
    amp = jnp.asarray(pars["A_s"])
    return {"TT": jnp.asarray([amp, 2.0 * amp, 3.0 * amp])}


def test_get_candl_parameter_names_excludes_cosmology():
    like = FakeLikelihood()

    names = get_candl_parameter_names(
        like,
        cosmo_keys=("A_s", "tau"),
        include_prior_params=True,
    )

    assert names == ("A_planck", "dust_amp", "calib")


def test_build_candl_parameter_layout_infers_sampled_block():
    like = FakeLikelihood()

    layout = build_candl_parameter_layout(
        like,
        cosmo_keys=("A_s", "tau"),
        cosmo_sizes=(1, 1),
    )

    assert layout.n_cosmo == 2
    assert layout.cmb_nuisance_names == ("A_planck", "dust_amp", "calib")


def test_candl_parameter_layout_pack_unpack_roundtrip():
    layout = CandlParameterLayout(
        cosmo_keys=("A_s", "tau"),
        cosmo_sizes=(1, 1),
        cmb_nuisance_names=("A_planck", "calib"),
        cmb_nuisance_offset=4,
    )
    cosmo = CosmoParams({"A_s": 2.1e-9, "tau": 0.055})

    packed = layout.pack(cosmo, {"A_planck": 1.0, "calib": 0.3})
    unpacked_cosmo, unpacked_nuisance = layout.unpack(packed)

    assert packed.shape == (6,)
    assert np.allclose(np.asarray(packed[2:4]), 0.0)
    np.testing.assert_allclose(np.asarray(unpacked_cosmo.to_array()), np.asarray(cosmo.to_array()))
    assert set(unpacked_nuisance) == {"A_planck", "calib"}
    assert np.isclose(float(unpacked_nuisance["A_planck"]), 1.0)
    assert np.isclose(float(unpacked_nuisance["calib"]), 0.3)


def test_make_candl_loglike_fn_unpacks_packed_parameters():
    like = FakeLikelihood()
    layout = CandlParameterLayout(
        cosmo_keys=("A_s",),
        cosmo_sizes=(1,),
        cmb_nuisance_names=("A_planck",),
        cmb_nuisance_offset=2,
    )
    fn = make_candl_loglike_fn(
        like,
        pars_to_theory_specs=_fake_theory_fn,
        layout=layout,
        fixed_cmb_params={"dust_amp": 0.25, "calib": 0.1},
    )

    value = float(fn(jnp.asarray([1.0, 99.0, 0.5])))

    expected = -(1.0 + 2.0 + 3.0 + 0.5 + 0.25 + 0.1)
    assert np.isclose(value, expected)


def test_make_candl_loglike_fn_supports_gradients():
    like = FakeLikelihood()
    fn = make_candl_loglike_fn(
        like,
        pars_to_theory_specs=_fake_theory_fn,
        cosmo_keys=("A_s",),
        cosmo_sizes=(1,),
        sampled_cmb_params=("A_planck",),
        fixed_cmb_params={"calib": 0.0, "dust_amp": 0.0},
    )

    grad = jax.grad(lambda x: fn(jnp.asarray([x, 0.0])))(1.0)

    assert np.isclose(float(grad), -6.0)


def test_make_candl_loglike_fn_jit_matches_nonjit():
    like = FakeLikelihood()
    fn = make_candl_loglike_fn(
        like,
        pars_to_theory_specs=_fake_theory_fn,
        cosmo_keys=("A_s",),
        cosmo_sizes=(1,),
        sampled_cmb_params=("A_planck",),
        fixed_cmb_params={"calib": 0.1, "dust_amp": 0.2},
    )
    fn_jit = make_candl_loglike_fn(
        like,
        pars_to_theory_specs=_fake_theory_fn,
        cosmo_keys=("A_s",),
        cosmo_sizes=(1,),
        sampled_cmb_params=("A_planck",),
        fixed_cmb_params={"calib": 0.1, "dust_amp": 0.2},
        jit_compile=True,
    )

    theta = jnp.asarray([1.5, 0.7])
    np.testing.assert_allclose(np.asarray(fn_jit(theta)), np.asarray(fn(theta)))

    grad = jax.grad(lambda x: fn_jit(jnp.asarray([x, 0.0])))(1.0)
    assert np.isclose(float(grad), -6.0)


def test_make_candl_theory_vector_fn_returns_binned_vector():
    like = FakeLikelihood()
    fn = make_candl_theory_vector_fn(
        like,
        pars_to_theory_specs=_fake_theory_fn,
        cosmo_keys=("A_s",),
        cosmo_sizes=(1,),
        sampled_cmb_params=("A_planck",),
    )

    theory = np.asarray(fn(jnp.asarray([2.0, 0.5])))

    np.testing.assert_allclose(theory, np.asarray([13.5, 6.5]))


def test_make_candl_theory_vector_fn_jit_matches_nonjit():
    like = FakeLikelihood()
    fn = make_candl_theory_vector_fn(
        like,
        pars_to_theory_specs=_fake_theory_fn,
        cosmo_keys=("A_s",),
        cosmo_sizes=(1,),
        sampled_cmb_params=("A_planck",),
    )
    fn_jit = make_candl_theory_vector_fn(
        like,
        pars_to_theory_specs=_fake_theory_fn,
        cosmo_keys=("A_s",),
        cosmo_sizes=(1,),
        sampled_cmb_params=("A_planck",),
        jit_compile=True,
    )

    theta = jnp.asarray([2.0, 0.5])
    np.testing.assert_allclose(np.asarray(fn_jit(theta)), np.asarray(fn(theta)))


def test_make_joint_loglike_fn_adds_all_terms():
    fn = make_joint_loglike_fn(
        lss_loglike_fn=lambda p: p[0],
        bao_loglike_fn=lambda p: 2.0 * p[1],
        cmb_loglike_fn=lambda p: -1.0,
        extra_loglike_fns=(lambda p: p[0] + p[1],),
        log_prior_fn=lambda p: -0.5,
    )

    value = float(fn(jnp.asarray([2.0, 3.0])))

    assert np.isclose(value, 2.0 + 6.0 - 1.0 + 5.0 - 0.5)


def test_make_candl_pars_to_theory_specs_fn_loads_cosmopower_models(monkeypatch):
    class FakeCPJ:
        def __init__(self, probe, filename=None, filepath=None):
            self.path = filename or filepath
            self.parameters = (
                np.asarray(["ombh2", "h"])
                if "pp" in self.path
                else np.asarray(["ombh2", "tau", "h"])
            )
            self.n_parameters = len(self.parameters)
            self.modes = jnp.asarray([2, 3, 4])

        def predict(self, pars_for_cp):
            total = jnp.sum(pars_for_cp)
            return jnp.asarray([total, total + 1.0, total + 2.0])

    def fake_import(name):
        if name == "candl":
            return SimpleNamespace()
        if name == "cosmopower_jax.cosmopower_jax":
            return SimpleNamespace(CosmoPowerJAX=FakeCPJ)
        raise ImportError(name)

    monkeypatch.setattr("jaxptpolypol.cmb.importlib.import_module", fake_import)

    fn = make_candl_pars_to_theory_specs_fn(
        emulator_filenames={"TT": "tt_model", "pp": "pp_model"}
    )
    result = fn({"ombh2": 0.02, "tau": 0.05, "H0": 70.0}, 4, 2)

    assert set(result) == {"ell", "TT", "pp", "kk"}
    np.testing.assert_array_equal(np.asarray(result["ell"]), np.asarray([2, 3, 4]))
    np.testing.assert_allclose(
        np.asarray(result["kk"]),
        np.asarray(result["pp"]) * (np.pi / 2.0),
    )

def test_make_candl_pars_to_theory_specs_fn_adds_pp_kk_alias(monkeypatch):
    class FakeCPJ:
        def __init__(self, probe, filename=None, filepath=None):
            self.parameters = np.asarray(["ombh2", "h"])
            self.n_parameters = 2
            self.modes = jnp.asarray([2, 3, 4])

        def predict(self, pars_for_cp):
            return jnp.asarray([1.0, 2.0, 3.0])

    def fake_import(name):
        if name == "candl":
            return SimpleNamespace()
        if name == "cosmopower_jax.cosmopower_jax":
            return SimpleNamespace(CosmoPowerJAX=FakeCPJ)
        raise ImportError(name)

    monkeypatch.setattr("jaxptpolypol.cmb.importlib.import_module", fake_import)

    fn = make_candl_pars_to_theory_specs_fn(emulator_filenames={"pp": "pp_model"})
    result = fn({"ombh2": 0.02, "H0": 70.0}, 4, 2)

    np.testing.assert_allclose(
        np.asarray(result["kk"]),
        np.asarray(result["pp"]) * (np.pi / 2.0),
    )


def test_load_candl_likelihood_native_path_resolves_shortcuts_and_clears_priors(monkeypatch):
    fake_candl = SimpleNamespace(Like=FakeLikelihood, LensLike=FakeLensLikelihood)
    fake_dataset_module = SimpleNamespace(TEST_DATA="resolved_dataset.yaml")

    def fake_import(name):
        if name == "candl":
            return fake_candl
        if name == "candl_data":
            return fake_dataset_module
        raise ImportError(name)

    monkeypatch.setattr("jaxptpolypol.cmb.importlib.import_module", fake_import)

    like = load_candl_likelihood(
        "candl_data.TEST_DATA",
        clear_specific_priors=("A_planck",),
    )

    assert like.data_set_file == "resolved_dataset.yaml"
    assert len(like.priors) == 1
    assert like.priors[0].par_names == ("tau",)


def test_load_candl_likelihood_supports_lensing_class(monkeypatch):
    fake_candl = SimpleNamespace(Like=FakeLikelihood, LensLike=FakeLensLikelihood)

    def fake_import(name):
        if name == "candl":
            return fake_candl
        raise ImportError(name)

    monkeypatch.setattr("jaxptpolypol.cmb.importlib.import_module", fake_import)

    like = load_candl_likelihood("dataset.yaml", lensing=True)

    assert isinstance(like, FakeLensLikelihood)


def test_load_candl_likelihood_supports_clipy_wrapper(monkeypatch):
    wrapper_like = SimpleNamespace(_prior={"gauss": ["tau", "calib"], "top_hat": "A_planck"})

    def fake_import(name):
        if name == "candl":
            return SimpleNamespace(Like=FakeLikelihood, LensLike=FakeLensLikelihood)
        if name == "clipy":
            return SimpleNamespace(clik_candl=lambda path, **kwargs: wrapper_like)
        raise ImportError(name)

    monkeypatch.setattr("jaxptpolypol.cmb.importlib.import_module", fake_import)

    like = load_candl_likelihood("plik.clik", wrapper="clipy")

    assert like is wrapper_like
    assert like.required_prior_parameters == ("tau", "calib", "A_planck")
