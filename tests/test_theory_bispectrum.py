import numpy as np
import pytest
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

pytest.importorskip("ps_1loop_jax")
from ps_1loop_jax import background as bg

from jaxptpolypol.covariance import (
    gaussian_bispectrum_covariance,
    gaussian_joint_covariance_multibin,
)
from ps_1loop_jax.bs_tree import BispectrumTree

from jaxptpolypol.model import BispectrumTreeModel

# Validation (including the round-off closure tolerance) now lives upstream in
# ps_1loop_jax.bs_tree; these tests pin that contract through the bare class.
if not hasattr(BispectrumTree, "_validate_triangle_eager"):  # pragma: no cover
    pytest.skip(
        "ps_1loop_jax predates the upstream jit-safe validation fix "
        "(ps_1loop_jax-for-pfs PR #5); update it to run these tests.",
        allow_module_level=True,
    )
_validate_triangle_eager = BispectrumTree()._validate_triangle_eager
from jaxptpolypol.params import (
    CosmoParams,
    FullShapeSurveyParams,
    SurveyParams,
    pack_pk_params,
    pack_multibin_params,
    pack_params,
)
from jaxptpolypol.theory import (
    build_bispectrum_triangles_from_k_grid,
    compute_fiducial_distances,
    kaiser_power_multipoles,
    make_bk0_fn,
    make_gaussian_bk0_covariance_fn,
    make_gaussian_joint_covariance_fn,
    make_joint_pk_bk_fn,
    make_pk_ell_fn,
)

MNU_FIXED = 0.06


class FakeEmulator:
    def __init__(self, modes):
        self.modes = jnp.asarray(modes, dtype=float)

    def predict(self, cosmo_dict):
        z = jnp.asarray(cosmo_dict["z"], dtype=float)[0]
        return (1.0 + z) * (1.0 + self.modes)


class FakePS1LoopModel:
    def _base(self, k, pk_data, params):
        p_lin = jnp.interp(jnp.asarray(k, dtype=float), pk_data["k"], pk_data["pk"])
        return p_lin + 0.5 * params["bias"]["b1"] + params["f"]

    def get_pk_ell(self, k, ell, pk_data, params, num=256):
        return self._base(k, pk_data, params) + float(ell)

    def get_pk_ell_ref(self, k, ell, alpha_perp, alpha_para, pk_data, params, num=256):
        return self._base(k, pk_data, params) + float(ell) + alpha_perp - alpha_para

    def get_pkmu(self, k, mu, pk_data, params):
        base = self._base(k, pk_data, params)[:, None]
        return base * (1.0 + mu[None, :] ** 2)

    def get_pkmu_ref(self, k, mu, alpha_perp, alpha_para, pk_data, params):
        base = self._base(k, pk_data, params)[:, None]
        return (base + alpha_perp - alpha_para) * (1.0 + mu[None, :] ** 2)


class FakeBispectrumModel:
    def get_bk0(
        self,
        k1,
        k2,
        k3,
        pk_data,
        params,
        *,
        alpha_perp=None,
        alpha_para=None,
        num_mu=65,
        num_phi=65,
    ):
        del alpha_perp, alpha_para, num_mu, num_phi
        p1 = jnp.interp(jnp.asarray(k1, dtype=float), pk_data["k"], pk_data["pk"])
        return p1 + params["bias"]["b1"] * (k1 + 2.0 * k2 + 3.0 * k3) + params["f"]


def _analytic_kaiser_pk(p_lin, b1, f, shot_noise):
    return np.stack(
        [
            (b1**2 + (2.0 / 3.0) * b1 * f + (1.0 / 5.0) * f**2) * p_lin + shot_noise,
            ((4.0 / 3.0) * b1 * f + (4.0 / 7.0) * f**2) * p_lin,
            (8.0 / 35.0) * f**2 * p_lin,
        ],
        axis=0,
    )


def test_build_bispectrum_triangles_from_k_grid_matches_explicit_triangles():
    k = jnp.array([0.01, 0.03, 0.05, 0.07, 0.09])

    triangles, triangle_dk = build_bispectrum_triangles_from_k_grid(
        k,
        k_min=0.02,
        k_max=0.08,
    )

    expected = jnp.array(
        [
            [0.07, 0.07, 0.07],
            [0.07, 0.07, 0.05],
            [0.07, 0.07, 0.03],
            [0.07, 0.05, 0.05],
            [0.07, 0.05, 0.03],
            [0.05, 0.05, 0.05],
            [0.05, 0.05, 0.03],
            [0.05, 0.03, 0.03],
            [0.03, 0.03, 0.03],
        ]
    )

    np.testing.assert_allclose(np.array(triangles), np.array(expected))
    assert float(triangle_dk) == pytest.approx(0.02)


def test_build_bispectrum_triangles_from_k_grid_gathers_edge_widths():
    k = jnp.array([0.01, 0.03, 0.05, 0.07, 0.09])
    dk = jnp.array([0.10, 0.20, 0.30, 0.40, 0.50])

    triangles, triangle_dk = build_bispectrum_triangles_from_k_grid(
        k,
        k_min=0.02,
        k_max=0.08,
        dk=dk,
    )

    np.testing.assert_allclose(np.array(triangles[0]), np.array([0.07, 0.07, 0.07]))
    np.testing.assert_allclose(np.array(triangle_dk[0]), np.array([0.40, 0.40, 0.40]))
    np.testing.assert_allclose(np.array(triangle_dk[-1]), np.array([0.20, 0.20, 0.20]))


def test_validate_triangle_eager_accepts_roundoff_level_closure_on_broadcast_arrays():
    triangles = jnp.array(
        [
            [0.07500000000000001, 0.04, 0.034999999999999996],
            [0.07, 0.034999999999999996, 0.034999999999999996],
            [0.060000000000000005, 0.03, 0.03],
        ],
        dtype=float,
    )

    k1, k2, k3 = triangles.T

    _validate_triangle_eager(k1[:, None, None], k2[:, None, None], k3[:, None, None])


def test_validate_triangle_eager_rejects_genuinely_open_triangle():
    with pytest.raises(ValueError, match="triangle inequality"):
        _validate_triangle_eager(0.08, 0.03, 0.03)


def test_bispectrum_triangles_from_updated_pk_grid_pass_validator():
    k = jnp.linspace(0.02, 0.20, 37)
    dk = float(k[1] - k[0])
    triangles, _ = build_bispectrum_triangles_from_k_grid(
        k,
        k_min=0.02,
        k_max=0.08,
        dk=dk,
    )

    k1, k2, k3 = triangles.T

    _validate_triangle_eager(k1[:, None, None], k2[:, None, None], k3[:, None, None])


def test_bispectrum_tree_model_preserves_k_nl_rsd_through_pytree_roundtrip():
    model = BispectrumTreeModel(do_AP=True, k_nl_rsd=0.47)
    children, aux_data = model.tree_flatten()
    rebuilt = BispectrumTreeModel.tree_unflatten(aux_data, children)

    assert rebuilt.k_nl_rsd == pytest.approx(0.47)
    assert rebuilt.do_AP is True


def test_make_joint_pk_bk_fn_jit_accepts_traced_triangles_with_real_bispectrum():
    emulator = FakeEmulator([0.07, 0.14, 0.21, 0.28])
    ps_model = FakePS1LoopModel()
    bk_model = BispectrumTreeModel(do_AP=True)
    cosmo = CosmoParams({"h": 0.7, "omega_b": 0.022, "omega_cdm": 0.12, "z": 0.8})
    survey = SurveyParams(
        {
            "bias": {"b1": 2.0, "b2": 0.3, "bG2": 0.1, "bGamma3": 0.0},
            "stoch": {"P_shot": 1.0, "B_shot": 0.0},
            "k_nl": 0.3,
            "ndens": 2.0,
        }
    )
    params = pack_params(cosmo, survey)
    k = jnp.array([0.1, 0.2, 0.3])
    triangles = jnp.array([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1]])
    Hz_fid, DAz_fid = compute_fiducial_distances(cosmo, (0.8,), mnu=MNU_FIXED)

    joint_fn = make_joint_pk_bk_fn(
        pklin_emulator=emulator,
        ps1loop_model=ps_model,
        bispectrum_model=bk_model,
        cosmo_keys=cosmo.param_keys,
        cosmo_sizes=cosmo.param_sizes,
        survey_keys=survey.param_keys,
        ap=True,
        Hz_fid=Hz_fid,
        DAz_fid=DAz_fid,
        num_mu=5,
        num_phi=5,
    )

    eager = np.array(joint_fn(params, k=k, triangles=triangles))
    compiled = np.array(jax.jit(joint_fn)(params, k=k, triangles=triangles))

    np.testing.assert_allclose(compiled, eager, rtol=1e-12, atol=1e-12)


def test_kaiser_power_multipoles_matches_closed_form_without_ap():
    k = jnp.array([0.1, 0.2, 0.3])
    pk_data = {"k": jnp.array([0.1, 0.2, 0.3]), "pk": jnp.array([2.0, 3.0, 5.0])}
    b1 = 1.8
    f = 0.7
    shot_noise = 0.4

    got = np.array(
        kaiser_power_multipoles(
            k,
            pk_data,
            b1=b1,
            f=f,
            shot_noise=shot_noise,
        )
    )
    expected = _analytic_kaiser_pk(np.array(pk_data["pk"]), b1, f, shot_noise)

    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_make_pk_ell_fn_accepts_role_aware_pk_layout():
    emulator = FakeEmulator([0.1, 0.2, 0.3])
    ps_model = FakePS1LoopModel()
    cosmo = CosmoParams({"h": 1.0, "omega_b": 0.022, "omega_cdm": 0.12, "z": 0.8})
    survey = FullShapeSurveyParams(
        shared={
            "bias": {"b1": 2.0, "b2": 0.0, "bG2": 0.0, "bGamma3": 0.0},
            "stoch": {"P_shot": 1.0},
            "k_nl": 0.3,
            "ndens": 2.0,
        },
        pk={
            "ctr": {"c0": 0.1, "c2": 0.0, "c4": 0.0, "cfog": 0.0},
            "stoch": {"a0": 0.0, "a2": 0.0},
        },
        bk={
            "ctr": {"c1": 9.0},
            "stoch": {"B_shot": 8.0, "A_shot": 7.0},
        },
    )
    params = pack_pk_params(cosmo, [survey])
    k = jnp.array([0.1, 0.2, 0.3])

    pk_fn = make_pk_ell_fn(
        ells=(0, 2, 4),
        pklin_emulator=emulator,
        ps1loop_model=ps_model,
        cosmo_keys=cosmo.param_keys,
        cosmo_sizes=cosmo.param_sizes,
        survey_keys=survey.pk_param_keys,
        ap=False,
    )
    got = np.array(pk_fn(params, k=k))

    f_growth = float(
        bg.growth_rate_approx(
            float(cosmo.omega_b[0]),
            float(cosmo.omega_cdm[0]),
            float(cosmo.h[0]),
            float(cosmo.z[0]),
            MNU_FIXED,
        )
    )
    pk_data = {"k": emulator.modes / cosmo.h[0], "pk": emulator.predict(cosmo.to_dict())}
    theory_params = {"h": cosmo.h[0], "f": f_growth, **survey.to_model_dict("pk")}
    expected = np.concatenate(
        [
            np.array(ps_model.get_pk_ell(k, ell, pk_data, theory_params))
            for ell in (0, 2, 4)
        ]
    )

    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_make_gaussian_bk0_covariance_fn_defaults_to_kaiser_power():
    emulator = FakeEmulator([0.1, 0.2, 0.3])
    cosmo = CosmoParams({"h": 1.0, "omega_b": 0.022, "omega_cdm": 0.12, "z": 0.8})
    survey = SurveyParams(
        {
            "bias": {"b1": 2.0, "b2": 0.0, "bG2": 0.0, "bGamma3": 0.0},
            "stoch": {"P_shot": 1.5},
            "k_nl": 0.3,
            "ndens": 3.0,
        }
    )
    params = pack_params(cosmo, survey)
    k = jnp.array([0.1, 0.2, 0.3])
    triangles = jnp.array([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1]])
    volume = 1000.0
    triangle_dk = 0.01

    bb_cov_fn = make_gaussian_bk0_covariance_fn(
        pklin_emulator=emulator,
        cosmo_keys=cosmo.param_keys,
        cosmo_sizes=cosmo.param_sizes,
        survey_keys=survey.param_keys,
        ap=False,
    )
    got = np.array(
        bb_cov_fn(
            params,
            V_survey=volume,
            k=k,
            triangles=triangles,
            triangle_dk=triangle_dk,
        )
    )

    f_growth = float(
        bg.growth_rate_approx(
            float(cosmo.omega_b[0]),
            float(cosmo.omega_cdm[0]),
            float(cosmo.h[0]),
            float(cosmo.z[0]),
            MNU_FIXED,
        )
    )
    p_lin = np.array(emulator.predict(cosmo.to_dict()))
    expected_pk = _analytic_kaiser_pk(
        p_lin,
        float(survey.get("bias", "b1")),
        f_growth,
        float(survey.get("stoch", "P_shot") / survey.get("ndens")),
    )
    expected = np.array(
        gaussian_bispectrum_covariance(
            volume,
            k,
            triangles,
            triangle_dk,
            jnp.asarray(expected_pk),
        )
    )

    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_make_joint_pk_bk_fn_matches_per_bin_block_layout():
    emulator = FakeEmulator([0.1, 0.2, 0.3])
    ps_model = FakePS1LoopModel()
    bk_model = FakeBispectrumModel()
    cosmo = CosmoParams({"h": 1.0, "omega_b": 0.022, "omega_cdm": 0.12, "z": 0.8})
    survey = SurveyParams(
        {
            "bias": {"b1": 2.0, "b2": 0.0, "bG2": 0.0, "bGamma3": 0.0},
            "stoch": {"P_shot": 1.0},
            "k_nl": 0.3,
            "ndens": 2.0,
        }
    )
    params = pack_params(cosmo, survey)
    k = jnp.array([0.1, 0.2, 0.3])
    triangles = jnp.array([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1]])

    joint_fn = make_joint_pk_bk_fn(
        pklin_emulator=emulator,
        ps1loop_model=ps_model,
        bispectrum_model=bk_model,
        cosmo_keys=cosmo.param_keys,
        cosmo_sizes=cosmo.param_sizes,
        survey_keys=survey.param_keys,
        ap=False,
    )
    got = np.array(joint_fn(params, k=k, triangles=triangles))

    f_growth = float(
        bg.growth_rate_approx(
            float(cosmo.omega_b[0]),
            float(cosmo.omega_cdm[0]),
            float(cosmo.h[0]),
            float(cosmo.z[0]),
            MNU_FIXED,
        )
    )
    pk_data = {"k": emulator.modes / cosmo.h[0], "pk": emulator.predict(cosmo.to_dict())}
    theory_params = {"h": cosmo.h[0], "f": f_growth, **survey.to_dict()}
    pk_block = np.concatenate(
        [
            np.array(ps_model.get_pk_ell(k, ell, pk_data, theory_params))
            for ell in (0, 2, 4)
        ]
    )
    bk_block = np.array(
        bk_model.get_bk0(
            triangles[:, 0],
            triangles[:, 1],
            triangles[:, 2],
            pk_data,
            theory_params,
        )
    )
    expected = np.concatenate([pk_block, bk_block])

    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_make_gaussian_joint_covariance_fn_multibin_uses_kaiser_for_bb():
    emulator = FakeEmulator([0.1, 0.2, 0.3])
    ps_model = FakePS1LoopModel()
    cosmo = CosmoParams({"h": 1.0, "omega_b": 0.022, "omega_cdm": 0.12})
    surveys = [
        SurveyParams(
            {
                "bias": {"b1": 1.6, "b2": 0.0, "bG2": 0.0, "bGamma3": 0.0},
                "stoch": {"P_shot": 1.0},
                "k_nl": 0.3,
                "ndens": 2.0,
            }
        ),
        SurveyParams(
            {
                "bias": {"b1": 2.1, "b2": 0.0, "bG2": 0.0, "bGamma3": 0.0},
                "stoch": {"P_shot": 1.5},
                "k_nl": 0.3,
                "ndens": 3.0,
            }
        ),
    ]
    params = pack_multibin_params(cosmo, surveys)
    z_bins = (0.7, 1.0)
    k = jnp.array([0.1, 0.2, 0.3])
    dk = 0.01
    triangles = [
        jnp.array([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1]]),
        jnp.array([[0.2, 0.2, 0.2]]),
    ]
    triangle_dk = [0.01, 0.02]
    volumes = [800.0, 1200.0]

    joint_cov_fn = make_gaussian_joint_covariance_fn(
        pklin_emulator=emulator,
        ps1loop_model=ps_model,
        cosmo_keys=cosmo.param_keys,
        cosmo_sizes=cosmo.param_sizes,
        survey_keys=surveys[0].param_keys,
        z_bins=z_bins,
        ap=False,
    )
    got = np.array(
        joint_cov_fn(
            params,
            V_survey=volumes,
            k=k,
            dk=dk,
            triangles=triangles,
            triangle_dk=triangle_dk,
        )
    )

    pp_blocks = []
    bb_blocks = []
    for z, survey in zip(z_bins, surveys):
        f_growth = float(
            bg.growth_rate_approx(
                float(cosmo.omega_b[0]),
                float(cosmo.omega_cdm[0]),
                float(cosmo.h[0]),
                z,
                MNU_FIXED,
            )
        )
        pk_data = {
            "k": np.array(emulator.modes / cosmo.h[0]),
            "pk": np.array(
                emulator.predict(
                    {
                        **cosmo.to_dict(),
                        "z": jnp.atleast_1d(z),
                    }
                )
            ),
        }
        theory_params = {"h": cosmo.h[0], "f": f_growth, **survey.to_dict()}
        pp_blocks.append(
            np.stack(
                [
                    np.array(ps_model.get_pk_ell(k, ell, pk_data, theory_params))
                    for ell in (0, 2, 4)
                ],
                axis=0,
            )
        )
        bb_blocks.append(
            _analytic_kaiser_pk(
                pk_data["pk"],
                float(survey.get("bias", "b1")),
                f_growth,
                float(survey.get("stoch", "P_shot") / survey.get("ndens")),
            )
        )

    expected = np.array(
        gaussian_joint_covariance_multibin(
            volumes,
            k,
            dk,
            jnp.asarray(np.stack(pp_blocks, axis=0)),
            triangles=triangles,
            triangle_dk=triangle_dk,
            bb_pk_all=jnp.asarray(np.stack(bb_blocks, axis=0)),
        )
    )

    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_bispectrum_irres_default_on_and_values_pinned():
    """Characterization pin for the do_irres flip (2026-08-23, commit 7304e6a).

    The flip moved the joint-vector B entries by up to 4.8e-3 and broke ZERO
    numerical tests -- nothing in the suite pinned B tightly enough to notice
    IR resummation switching on. This test closes that gap: it pins (a) the
    default itself, (b) three tree-B values under do_irres=True on a synthetic
    wiggly spectrum (values recorded at the flip; rtol 1e-8 is ~1000x tighter
    than the on/off separation of ~1e-5 here), and (c) that the flag is not
    inert. A silent revert of the default, or any change to the IR-resummed
    bispectrum path, fails loudly."""
    k = np.geomspace(1e-4, 10.0, 1024)
    pk_data = {"k": k, "pk": k**-1.5 * (1.0 + 0.08 * np.sin(18.0 * k))}
    params = {"f": 0.55, "h": 0.6766,
              "bias": {"b1": 1.9, "b2": -0.3, "bG2": 0.1}, "ctr": {"c1": 0.0}}
    triangles = [(0.05, 0.05, 0.05), (0.05, 0.08, 0.10), (0.06, 0.06, 0.10)]
    pinned_on = [1.1260991996e+05, 6.0287000959e+04, 7.8958027655e+04]

    model = BispectrumTreeModel(k_nl_rsd=0.45)      # rely on the default
    assert model.do_irres is True                    # (a) the default itself
    model_off = BispectrumTreeModel(do_irres=False, k_nl_rsd=0.45)
    for tri, pin in zip(triangles, pinned_on):
        on = float(model.model.get_bk_tree(*tri, 0.3, 0.7, pk_data, params))
        off = float(model_off.model.get_bk_tree(*tri, 0.3, 0.7, pk_data, params))
        assert on == pytest.approx(pin, rel=1e-8)    # (b) pinned values
        assert on != off                             # (c) the flag acts


def test_model_wrappers_are_registered_pytrees():
    """All three wrappers must stay registered JAX pytrees, as the module
    docstring promises ("passed through jax.jit boundaries as static
    arguments").

    Regression guard: a shim-removal edit once silently deleted
    CosmoEmulator's @register_pytree_node_class decorator and the entire
    suite still passed, because nothing exercised the registration.

    Discriminator, chosen so no instance has to be constructed (CosmoEmulator
    needs emulator files, PS1LoopModel builds PT matrices): an UNREGISTERED
    class flattens to itself as a single opaque leaf, whereas a REGISTERED one
    makes JAX dispatch into the class's own tree_flatten -- which, on a bare
    object.__new__ instance, raises from inside that method. Either outcome
    other than "flattened to a bare leaf" proves registration.
    """
    import jax
    from jaxptpolypol.model import CosmoEmulator, PS1LoopModel, BispectrumTreeModel

    for cls in (CosmoEmulator, PS1LoopModel, BispectrumTreeModel):
        obj = object.__new__(cls)
        try:
            leaves, _ = jax.tree_util.tree_flatten(obj)
        except Exception:
            continue  # JAX dispatched into cls.tree_flatten => registered
        assert leaves != [obj], (
            f"{cls.__name__} flattened as an opaque leaf -- it is NOT a "
            "registered pytree (missing @jax.tree_util.register_pytree_node_class?)"
        )


def test_missing_upstream_capability_raises_actionable_importerror(monkeypatch):
    """The guard must fail fast at construction, not deep inside a run.

    Against an older ps_1loop_jax the symptoms are a ValueError on grid-edge
    triangles or a TracerArrayConversionError under jit -- both surfacing only
    after a notebook has spent minutes building emulators. Simulate the old
    package by emptying FEATURES and require a named, actionable error.
    """
    import ps_1loop_jax
    from jaxptpolypol import model as model_mod

    monkeypatch.setattr(ps_1loop_jax, "FEATURES", frozenset(), raising=False)
    with pytest.raises(ImportError, match="bs_tree_jit_safe_validation"):
        model_mod._require_jit_safe_bs_tree()


def test_capability_guard_passes_against_current_upstream():
    from jaxptpolypol import model as model_mod

    model_mod._require_jit_safe_bs_tree()  # must not raise
