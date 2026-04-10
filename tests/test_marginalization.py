import numpy as np
import jax
import jax.numpy as jnp

from jaxptpolypol.inference import marginalized_fisher_block
from jaxptpolypol.marginalization import (
    H0_spec,
    make_lcdm_corner_specs,
    native_spec,
    project_case_to_specs,
)


jax.config.update("jax_enable_x64", True)


class FakeLinearEmulator:
    def __init__(self):
        self.parameters = ("ombh2", "omch2", "h", "logA", "ns", "z")
        self.modes = jnp.geomspace(1.0e-4, 10.0, 512)

    def predict(self, cosmo_dict):
        ombh2 = jnp.asarray(cosmo_dict["ombh2"], dtype=jnp.float64)[0]
        omch2 = jnp.asarray(cosmo_dict["omch2"], dtype=jnp.float64)[0]
        h = jnp.asarray(cosmo_dict["h"], dtype=jnp.float64)[0]
        logA = jnp.asarray(cosmo_dict["logA"], dtype=jnp.float64)[0]
        ns = jnp.asarray(cosmo_dict["ns"], dtype=jnp.float64)[0]
        z = jnp.asarray(cosmo_dict["z"], dtype=jnp.float64)[0]
        amplitude = jnp.exp(logA - 3.0) * (1.0 + 5.0 * ombh2 + 2.0 * omch2 + 0.1 * h)
        tilt = jnp.power(self.modes / 0.2, ns - 1.0)
        growth = 1.0 / (1.0 + z) ** 2
        cutoff = jnp.exp(-self.modes / 5.0)
        return amplitude * tilt * cutoff * growth


class FakeLinearEmulatorWithBaryons(FakeLinearEmulator):
    def __init__(self):
        super().__init__()
        self.parameters = (
            "ombh2",
            "omch2",
            "h",
            "logA",
            "ns",
            "A_b",
            "eta_b",
            "logT_AGN",
            "z",
        )

    def predict(self, cosmo_dict):
        base = super().predict(cosmo_dict)
        A_b = jnp.asarray(cosmo_dict["A_b"], dtype=jnp.float64)[0]
        eta_b = jnp.asarray(cosmo_dict["eta_b"], dtype=jnp.float64)[0]
        logT_AGN = jnp.asarray(cosmo_dict["logT_AGN"], dtype=jnp.float64)[0]
        modifier = 1.0 + 0.01 * A_b + 0.02 * eta_b + 0.001 * logT_AGN
        return modifier * base


def test_project_case_to_specs_native_only_matches_marginalized_block():
    fisher = np.array(
        [
            [5.0, 1.0, 0.5, 0.2],
            [1.0, 4.0, 0.3, 0.1],
            [0.5, 0.3, 3.0, 0.2],
            [0.2, 0.1, 0.2, 2.0],
        ]
    )
    packed_varied = np.array([0.0224, 0.12, 0.67, 1.5])
    native_idx = (0, 1, 2)
    native_names = ("ombh2", "omch2", "h")

    result = project_case_to_specs(
        fisher,
        packed_varied,
        native_idx,
        native_names,
        [native_spec("h"), native_spec("ombh2")],
    )

    expected = marginalized_fisher_block(fisher, [2, 0])

    assert result.plot_names == ("h", "ombh2")
    assert result.plot_labels == (r"$h$", r"$\omega_b h^2$")
    np.testing.assert_allclose(result.fisher_plot, expected)
    np.testing.assert_allclose(result.fid_plot, packed_varied[[2, 0]])


def test_project_case_to_specs_uses_minimal_native_dependencies():
    fisher = np.diag([5.0, 4.0, 3.0, 2.0])
    packed_varied = np.array([0.0224, 0.12, 0.67, 1.5])
    native_idx = (0, 1, 2)
    native_names = ("ombh2", "omch2", "h")

    result = project_case_to_specs(
        fisher,
        packed_varied,
        native_idx,
        native_names,
        [H0_spec()],
    )

    assert result.native_keep_names == ("h",)
    assert result.native_keep_idx == (2,)
    np.testing.assert_allclose(result.fid_plot, np.array([67.0]))


def test_project_case_to_specs_supports_lcdm_transformed_outputs():
    emulator = FakeLinearEmulator()
    fisher = np.diag([2.0e6, 1.5e6, 1.0e4, 25.0, 2.5e3, 5.0])
    packed_varied = np.array([0.0224, 0.12, 0.67, 3.044, 0.965, 1.0])
    native_idx = (0, 1, 2, 3, 4)
    native_names = ("ombh2", "omch2", "h", "logA", "ns")

    result = project_case_to_specs(
        fisher,
        packed_varied,
        native_idx,
        native_names,
        make_lcdm_corner_specs(native_names, pklin_emulator=emulator, mnu_fixed=0.06),
    )

    assert result.plot_names == ("Omega_m", "H0", "sigma8")
    assert result.fisher_plot.shape == (3, 3)
    assert result.fid_plot.shape == (3,)
    assert np.all(np.isfinite(result.fisher_plot))
    assert np.all(np.isfinite(result.fid_plot))


def test_project_case_to_specs_supports_mixed_transformed_and_native_outputs():
    emulator = FakeLinearEmulator()
    fisher = np.diag([2.0e6, 1.5e6, 1.0e4, 25.0, 2.5e3, 300.0, 5.0])
    packed_varied = np.array([0.0224, 0.12, 0.67, 3.044, 0.965, 0.06, 1.0])
    native_idx = (0, 1, 2, 3, 4, 5)
    native_names = ("ombh2", "omch2", "h", "logA", "ns", "mnu")

    specs = list(
        make_lcdm_corner_specs(
            native_names,
            pklin_emulator=emulator,
            mnu_name="mnu",
            mnu_fixed=0.06,
        )
    ) + ["mnu"]
    result = project_case_to_specs(
        fisher,
        packed_varied,
        native_idx,
        native_names,
        specs,
    )

    assert result.plot_names == ("Omega_m", "H0", "sigma8", "mnu")
    assert result.plot_labels[-1] == r"$\sum m_\nu$"
    assert result.native_keep_names == native_names
    assert result.fisher_plot.shape == (4, 4)
    assert np.isclose(result.fid_plot[-1], 0.06)


def test_make_lcdm_corner_specs_accepts_fixed_cosmology_inputs_for_sigma8():
    emulator = FakeLinearEmulatorWithBaryons()
    fisher = np.diag([2.0e6, 1.5e6, 1.0e4, 25.0, 2.5e3, 300.0, 5.0])
    packed_varied = np.array([0.0224, 0.12, 0.67, 3.044, 0.965, 0.06, 1.0])
    native_idx = (0, 1, 2, 3, 4, 5)
    native_names = ("ombh2", "omch2", "h", "logA", "ns", "mnu")

    specs = list(
        make_lcdm_corner_specs(
            native_names,
            pklin_emulator=emulator,
            mnu_name="mnu",
            mnu_fixed=0.06,
            fixed_cosmo_values={
                "ombh2": 0.0224,
                "omch2": 0.12,
                "h": 0.67,
                "logA": 3.044,
                "ns": 0.965,
                "mnu": 0.06,
                "A_b": 3.13,
                "eta_b": 0.603,
                "logT_AGN": 7.8,
            },
        )
    ) + ["mnu"]
    result = project_case_to_specs(
        fisher,
        packed_varied,
        native_idx,
        native_names,
        specs,
    )

    assert result.plot_names == ("Omega_m", "H0", "sigma8", "mnu")
    assert result.native_keep_names == native_names
    assert np.all(np.isfinite(result.fid_plot))
    assert np.all(np.isfinite(result.fisher_plot))
