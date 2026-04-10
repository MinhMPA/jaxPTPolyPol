import numpy as np
import jax
import jax.numpy as jnp

from jaxptpolypol.derived import make_lcdm_derived_params_fn
from jaxptpolypol.inference import project_fisher_to_derived


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


def test_project_fisher_to_derived_matches_linear_covariance_propagation():
    fisher = np.array([[4.0, 1.0], [1.0, 3.0]])
    fid = jnp.array([1.0, 2.0], dtype=jnp.float64)

    def derived_fn(theta):
        return jnp.array([theta[0] + 2.0 * theta[1], theta[0] - theta[1]])

    fisher_derived, fid_derived, jacobian, cov_derived = project_fisher_to_derived(
        fisher,
        fid,
        derived_fn,
    )

    cov_native = np.linalg.inv(fisher)
    jac_expected = np.array([[1.0, 2.0], [1.0, -1.0]])
    cov_expected = jac_expected @ cov_native @ jac_expected.T

    np.testing.assert_allclose(fid_derived, np.array([5.0, -1.0]))
    np.testing.assert_allclose(jacobian, jac_expected)
    np.testing.assert_allclose(cov_derived, cov_expected)
    np.testing.assert_allclose(fisher_derived, np.linalg.inv(cov_expected))


def test_make_lcdm_derived_params_fn_returns_finite_omega_m_h0_sigma8():
    emulator = FakeLinearEmulator()
    derived_fn = make_lcdm_derived_params_fn(
        ("ombh2", "omch2", "h", "logA", "ns"),
        (1, 1, 1, 1, 1),
        pklin_emulator=emulator,
        mnu_fixed=0.06,
    )
    native = jnp.array([0.0224, 0.12, 0.67, 3.044, 0.965], dtype=jnp.float64)

    derived = np.asarray(derived_fn(native), dtype=float)
    jacobian = np.asarray(jax.jacfwd(derived_fn)(native), dtype=float)

    expected_omega_m = (0.0224 + 0.12 + 0.06 / 93.14) / (0.67**2)
    assert np.isclose(derived[0], expected_omega_m)
    assert np.isclose(derived[1], 67.0)
    assert np.isfinite(derived[2])
    assert derived[2] > 0.0
    assert jacobian.shape == (3, 5)
    assert np.all(np.isfinite(jacobian))


def test_project_fisher_to_derived_accepts_lcdm_sigma8_mapping():
    emulator = FakeLinearEmulator()
    derived_fn = make_lcdm_derived_params_fn(
        ("ombh2", "omch2", "h", "logA", "ns"),
        (1, 1, 1, 1, 1),
        pklin_emulator=emulator,
    )
    fisher_native = np.diag([2.0e6, 1.5e6, 1.0e4, 25.0, 2.5e3])
    fid_native = jnp.array([0.0224, 0.12, 0.67, 3.044, 0.965], dtype=jnp.float64)

    fisher_derived, fid_derived, jacobian, cov_derived = project_fisher_to_derived(
        fisher_native,
        fid_native,
        derived_fn,
    )

    assert fisher_derived.shape == (3, 3)
    assert fid_derived.shape == (3,)
    assert jacobian.shape == (3, 5)
    assert cov_derived.shape == (3, 3)
    assert np.all(np.isfinite(fisher_derived))
    assert np.all(np.isfinite(cov_derived))
