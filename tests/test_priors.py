import numpy as np

from jaxptpolypol.params import CosmoParams, FullShapeSurveyParams
from jaxptpolypol.priors import (
    build_prior_entries_from_spec,
    build_prior_sigmas_from_spec,
    load_prior_spec,
    resolve_survey_prior_spec,
)


def _make_fullshape_survey():
    return FullShapeSurveyParams(
        shared={
            "bias": {"b1": 2.0, "b2": 0.1, "bG2": 0.02, "bGamma3": 0.03},
            "stoch": {"P_shot": 1.5},
            "k_nl": 0.3,
            "ndens": 4.0e-4,
        },
        pk={
            "ctr": {"c0": 0.01, "c2": 0.02, "c4": 0.03, "cfog": 0.04},
            "stoch": {"a0": 0.05, "a2": 0.06},
        },
        bk={
            "ctr": {"c1": 0.07},
            "stoch": {"B_shot": 0.8, "A_shot": 1.2},
        },
    )


def test_load_prior_spec_reads_packaged_yaml():
    spec = load_prior_spec()

    assert spec["metadata"]["source"] == "arXiv:2405.02252"
    assert spec["metadata"]["equation"] == 12
    assert "survey_priors" in spec


def test_resolve_survey_prior_spec_uses_code_convention_means():
    resolved = resolve_survey_prior_spec(observable="joint")

    assert resolved[("shared", "stoch", "P_shot")]["mean"] == 1.0
    assert resolved[("bk", "stoch", "B_shot")]["mean"] == 1.0
    assert resolved[("bk", "stoch", "A_shot")]["mean"] == 1.0
    assert resolved[("pk", "stoch", "a0")]["mean"] == 0.0
    assert resolved[("pk", "ctr", "c0")]["sigma"] == 1.0


def test_build_prior_sigmas_from_spec_matches_pk_and_joint_layouts():
    cosmo = CosmoParams({"h": 0.7, "ombh2": 0.0224, "ns": 0.9649})
    survey = _make_fullshape_survey()

    pk_sigmas = build_prior_sigmas_from_spec(
        cosmo_keys=cosmo.param_keys,
        cosmo_sizes=cosmo.param_sizes,
        survey_keys=survey.pk_param_keys,
        n_bins=2,
        observable="pk",
        cosmo_priors={"ombh2": 5.5e-4, "ns": 0.042},
    )
    joint_sigmas = build_prior_sigmas_from_spec(
        cosmo_keys=cosmo.param_keys,
        cosmo_sizes=cosmo.param_sizes,
        survey_keys=survey.joint_param_keys,
        n_bins=2,
        observable="joint",
    )

    n_cosmo = len(cosmo.to_array())
    pk_keys = survey.pk_param_keys
    joint_keys = survey.joint_param_keys

    idx_c0_bin0 = n_cosmo + pk_keys.index(("pk", "ctr", "c0"))
    idx_c1_bin0 = n_cosmo + joint_keys.index(("bk", "ctr", "c1"))
    idx_bshot_bin1 = (
        n_cosmo + len(joint_keys) + joint_keys.index(("bk", "stoch", "B_shot"))
    )

    assert pk_sigmas[1] == 5.5e-4
    assert pk_sigmas[2] == 0.042
    assert pk_sigmas[idx_c0_bin0] == 1.0
    assert idx_c1_bin0 in joint_sigmas
    assert joint_sigmas[idx_bshot_bin1] == 1.0


def test_build_prior_entries_from_spec_respects_varied_idx():
    cosmo = CosmoParams({"h": 0.7, "ombh2": 0.0224, "ns": 0.9649})
    survey = _make_fullshape_survey()
    n_cosmo = len(cosmo.to_array())
    joint_keys = survey.joint_param_keys

    full_idx_bshot = n_cosmo + joint_keys.index(("bk", "stoch", "B_shot"))
    full_idx_ashot = n_cosmo + joint_keys.index(("bk", "stoch", "A_shot"))
    varied_idx = [1, full_idx_bshot, full_idx_ashot]

    entries = build_prior_entries_from_spec(
        varied_idx,
        cosmo_keys=cosmo.param_keys,
        cosmo_sizes=cosmo.param_sizes,
        survey_keys=joint_keys,
        n_bins=1,
        observable="joint",
        cosmo_prior_entries={"ombh2": (0.0224, 5.5e-4)},
    )

    assert entries == [
        (0, 0.0224, 5.5e-4),
        (1, 1.0, 1.0),
        (2, 1.0, 1.0),
    ]


def test_build_prior_sigmas_from_spec_matches_legacy_shared_keys():
    spec = load_prior_spec()
    prior = resolve_survey_prior_spec(spec, observable="pk")

    legacy_key = ("stoch", "P_shot")
    matched = prior[("shared",) + legacy_key]

    assert np.isclose(matched["mean"], 1.0)
    assert np.isclose(matched["sigma"], 1.0)
