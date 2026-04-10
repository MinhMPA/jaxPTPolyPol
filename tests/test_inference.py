import numpy as np

from jaxptpolypol.inference import (
    fisher_diagnostics,
    format_fisher_diagnostics,
    marginalized_fisher_block,
)


def test_fisher_diagnostics_identifies_single_unconstrained_parameter():
    fisher = np.diag([4.0, 0.0, 1.0])
    names = ("omega_b", "bias.b2_bin0", "h")

    diagnostics = fisher_diagnostics(fisher, param_names=names)

    assert diagnostics["rank"] == 2
    assert diagnostics["n_weak_modes"] == 1
    assert np.isinf(diagnostics["condition_number"])
    assert diagnostics["weakest_modes"][0]["components"][0]["name"] == "bias.b2_bin0"
    assert diagnostics["nullspace_participation"][0]["name"] == "bias.b2_bin0"
    assert diagnostics["nullspace_participation"][0]["participation"] == 1.0


def test_fisher_diagnostics_identifies_degenerate_parameter_pair():
    fisher = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
        ]
    )
    names = ("omega_b", "omega_cdm", "h")

    diagnostics = fisher_diagnostics(fisher, param_names=names)

    top_names = {
        item["name"] for item in diagnostics["weakest_modes"][0]["components"][:2]
    }
    assert diagnostics["rank"] == 2
    assert diagnostics["n_weak_modes"] == 1
    assert top_names == {"omega_b", "omega_cdm"}

    report = format_fisher_diagnostics(diagnostics)
    assert "omega_b" in report
    assert "omega_cdm" in report


def test_marginalized_fisher_block_differs_from_plain_subblock():
    fisher = np.array(
        [
            [2.0, 1.0],
            [1.0, 2.0],
        ]
    )

    fisher_keep = marginalized_fisher_block(fisher, [0])

    np.testing.assert_allclose(fisher_keep, np.array([[1.5]]))
    assert not np.isclose(fisher_keep[0, 0], fisher[0, 0])
