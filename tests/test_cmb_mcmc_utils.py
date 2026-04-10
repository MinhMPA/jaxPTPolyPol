from __future__ import annotations

import numpy as np

from jaxptpolypol.cmb_mcmc_utils import (
    COMBINATION_CONFIGS,
    artifact_path_for_selector,
    chunked_map,
    load_run_artifact,
    save_run_artifact,
    selector_to_label,
)


def test_artifact_path_for_selector_uses_stable_name(tmp_path):
    path = artifact_path_for_selector("planck_primary", tmp_path)

    assert path.name == "cmb_bao_bbn_LCDM_planck_primary.npz"
    assert selector_to_label("planck_primary") == "Planck 2018 TTTEEE + lowT + lowE"


def test_save_and_load_run_artifact_roundtrip(tmp_path):
    path = tmp_path / "run.npz"
    saved = save_run_artifact(
        path,
        metadata={
            "selector": "bao_bbn",
            "label": "BAO + BBN",
            "sampled_cosmo_keys": ["100theta", "ombh2"],
            "settings": {"num_samples": 10},
        },
        flat_samples=np.arange(6.0).reshape(3, 2),
        flat_log_post=np.asarray([-3.0, -2.0, -1.0]),
        whitening_scales=np.asarray([0.1, 0.2]),
        fid_native=np.asarray([67.0, 0.022, 0.12, 3.0, 0.96, 0.05]),
        acceptance_rate=np.asarray([[0.8, 0.9]]),
        num_integration_steps=np.asarray([[10, 12]]),
        is_divergent=np.asarray([[False, True]]),
    )

    loaded = load_run_artifact(saved)

    assert loaded["metadata"]["selector"] == "bao_bbn"
    np.testing.assert_allclose(loaded["flat_samples"], np.arange(6.0).reshape(3, 2))
    np.testing.assert_allclose(loaded["whitening_scales"], np.asarray([0.1, 0.2]))
    np.testing.assert_array_equal(loaded["is_divergent"], np.asarray([[False, True]]))


def test_chunked_map_matches_full_batch():
    values = np.arange(12.0).reshape(6, 2)
    result = chunked_map(values, lambda x: x[:, :1] + 2.0 * x[:, 1:2], chunk_size=2)

    np.testing.assert_allclose(result, values[:, :1] + 2.0 * values[:, 1:2])


def test_act_lensing_bbn_uses_h0_basis_only():
    assert COMBINATION_CONFIGS["act_lensing_bbn"]["sampled_cosmo_keys"] == (
        "H0",
        "ombh2",
        "omch2",
        "logA",
        "ns",
    )
    assert COMBINATION_CONFIGS["act_lensing_bao"]["sampled_cosmo_keys"][0] == "100theta"
    assert COMBINATION_CONFIGS["planck_lensing_bao"]["sampled_cosmo_keys"][0] == "100theta"


def test_planck_primary_uses_full_planck2018_primary_stack():
    assert COMBINATION_CONFIGS["planck_primary"]["cmb_term_names"] == (
        "planck_highl",
        "planck_lowl_tt",
        "planck_lowl_ee",
    )
