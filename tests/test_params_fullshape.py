import numpy as np

from jaxptpolypol.chain_analysis import make_fullshape_spec
from jaxptpolypol.inference import build_prior_sigmas
from jaxptpolypol.params import (
    CosmoParams,
    FullShapeSurveyParams,
    pack_joint_params,
    pack_pk_params,
    unpack_fullshape_params,
)


def _make_fullshape_survey(c1=0.3, bshot=0.4, ashot=0.5):
    return FullShapeSurveyParams(
        shared={
            "bias": {"b1": 2.0, "b2": 0.1},
            "stoch": {"P_shot": 1.5},
            "k_nl": 0.3,
            "ndens": 4.0,
        },
        pk={
            "ctr": {"c0": 0.01, "c2": 0.02, "cfog": 0.03},
            "stoch": {"a0": 0.04, "a2": 0.05},
        },
        bk={
            "ctr": {"c1": c1},
            "stoch": {"B_shot": bshot, "A_shot": ashot},
        },
    )


def test_fullshape_survey_params_exposes_role_aware_layouts():
    survey = _make_fullshape_survey()

    pk_keys = survey.pk_param_keys
    joint_keys = survey.joint_param_keys

    assert all(section in {"shared", "pk"} for section, _, _ in pk_keys)
    assert any(section == "bk" for section, _, _ in joint_keys)

    pk_model = survey.to_model_dict("pk")
    bk_model = survey.to_model_dict("bk")

    assert "c1" not in pk_model.get("ctr", {})
    assert "A_shot" not in pk_model.get("stoch", {})
    assert "c0" not in bk_model.get("ctr", {})
    assert "a0" not in bk_model.get("stoch", {})
    assert pk_model["bias"]["b1"] == 2.0
    assert bk_model["ctr"]["c1"] == 0.3


def test_pack_pk_params_excludes_bk_only_parameters_and_names():
    cosmo = CosmoParams({"h": 0.7, "omega_b": 0.022, "omega_cdm": 0.12, "z": 0.8})
    surveys = [_make_fullshape_survey(), _make_fullshape_survey(c1=0.6)]

    packed = pack_pk_params(cosmo, surveys)
    _, unpacked = unpack_fullshape_params(
        packed,
        cosmo.param_keys,
        cosmo.param_sizes,
        surveys[0].pk_param_keys,
        n_bins=2,
    )

    assert packed.shape[0] == len(cosmo.to_array()) + 2 * len(surveys[0].pk_param_keys)
    assert "c1" not in unpacked[0].to_model_dict("pk").get("ctr", {})
    assert unpacked[1].to_model_dict("pk")["ctr"]["c0"] == 0.01

    spec = make_fullshape_spec(
        cosmo_keys=cosmo.param_keys,
        cosmo_sizes=cosmo.param_sizes,
        survey_keys=surveys[0].pk_param_keys,
        n_bins=2,
        analysis_kind="pk",
    )
    names = spec.full_param_names()

    assert "ctr.c1@bin0" not in names
    assert "stoch.A_shot@bin1" not in names
    assert "ctr.c0@bin0" in names
    assert "stoch.a2@bin1" in names


def test_build_prior_sigmas_supports_role_aware_survey_keys():
    cosmo = CosmoParams({"h": 0.7, "omega_b": 0.022, "omega_cdm": 0.12})
    survey = _make_fullshape_survey()

    prior_sigmas = build_prior_sigmas(
        cosmo_keys=cosmo.param_keys,
        cosmo_sizes=cosmo.param_sizes,
        survey_keys=survey.joint_param_keys,
        n_bins=2,
        survey_priors=[
            {("bk", "ctr", "c1"): 0.7},
            {("pk", "ctr", "cfog"): 1.2},
        ],
    )

    n_cosmo = len(cosmo.to_array())
    n_survey = len(survey.joint_param_keys)
    idx_c1_bin0 = n_cosmo + survey.joint_param_keys.index(("bk", "ctr", "c1"))
    idx_cfog_bin1 = (
        n_cosmo
        + n_survey
        + survey.joint_param_keys.index(("pk", "ctr", "cfog"))
    )

    assert prior_sigmas[idx_c1_bin0] == 0.7
    assert prior_sigmas[idx_cfog_bin1] == 1.2


def test_pack_joint_params_roundtrip_preserves_bk_only_values():
    cosmo = CosmoParams({"h": 0.7, "omega_b": 0.022, "omega_cdm": 0.12})
    surveys = [_make_fullshape_survey(), _make_fullshape_survey(c1=0.9, ashot=0.8)]

    packed = pack_joint_params(cosmo, surveys)
    _, unpacked = unpack_fullshape_params(
        packed,
        cosmo.param_keys,
        cosmo.param_sizes,
        surveys[0].joint_param_keys,
        n_bins=2,
    )

    np.testing.assert_allclose(
        np.array(unpacked[0].bk.get("ctr", "c1")),
        np.array(0.3),
    )
    np.testing.assert_allclose(
        np.array(unpacked[1].bk.get("stoch", "A_shot")),
        np.array(0.8),
    )
