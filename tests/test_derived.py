import numpy as np
import pytest

import jax
import jax.numpy as jnp

from jaxptpolypol.derived import (
    format_derived_comparison_rows,
    make_derived_projection_fn,
    make_lcdm_derived_params_fn,
)
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


# ---------------------------------------------------------------------------
# make_derived_projection_fn: the wrapper the four joint MCMC notebooks used to
# each carry a private copy of (~120 lines x 4).
# ---------------------------------------------------------------------------

LCDM_KEYS = ("ombh2", "omch2", "h", "logA", "ns")
NULCDM_KEYS = ("ombh2", "omch2", "h", "logA", "ns", "mnu")
LCDM_FID = jnp.array([0.02242, 0.11933, 0.6766, 3.047, 0.9665],
                     dtype=jnp.float64)
NULCDM_FID = jnp.array([0.02242, 0.11933, 0.6766, 3.047, 0.9665, 0.06],
                       dtype=jnp.float64)


def _projection(keys, fid, source_indices, native_indices):
    return make_derived_projection_fn(
        keys, (1,) * len(keys),
        pklin_emulator=FakeLinearEmulator(),
        fiducial_native=fid,
        source_indices=source_indices,
        native_indices=native_indices,
    )


def test_derived_projection_axis_order_lcdm():
    """LCDM reports (Omega_m, sigma8, H0) -- the library core returns
    (Omega_m, H0, sigma8), so the reorder is the thing under test."""
    derived_fn, names = _projection(
        LCDM_KEYS, LCDM_FID, [0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
    assert names == ("Omega_m", "sigma8", "H0")

    theta = jnp.asarray(LCDM_FID)
    out = np.asarray(derived_fn(theta), dtype=float)
    core = make_lcdm_derived_params_fn(
        LCDM_KEYS, (1,) * len(LCDM_KEYS), pklin_emulator=FakeLinearEmulator())
    core_out = np.asarray(core(theta), dtype=float)
    # (Omega_m, H0, sigma8) -> (Omega_m, sigma8, H0), bit for bit.
    assert out.tolist() == [core_out[0], core_out[2], core_out[1]]


def test_derived_projection_axis_order_nulcdm_puts_mnu_first():
    derived_fn, names = _projection(
        NULCDM_KEYS, NULCDM_FID, [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])
    assert names == ("mnu", "Omega_m", "sigma8", "H0")
    out = np.asarray(derived_fn(NULCDM_FID), dtype=float)
    assert out.shape == (4,)
    assert out[0] == float(NULCDM_FID[5])


def test_derived_projection_exact_identities_lcdm():
    """H0 == 100 h and Omega_m == (ombh2 + omch2 + mnu_fixed/93.14) / h^2,
    to the last bit -- these are the notebook's build-time assertions."""
    derived_fn, names = _projection(
        LCDM_KEYS, LCDM_FID, [0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
    theta = jnp.array([0.0224, 0.12, 0.67, 3.044, 0.965], dtype=jnp.float64)
    out = np.asarray(derived_fn(theta), dtype=float)

    h = 0.67
    assert out[names.index("H0")] == 100.0 * h
    expected_om = (0.0224 + 0.12 + 0.06 / 93.14) / h**2
    assert out[names.index("Omega_m")] == expected_om


def test_derived_projection_exact_identities_nulcdm_uses_sampled_mnu():
    """The nuLCDM Omega_m identity must use the SAMPLED mnu, not mnu_fixed."""
    derived_fn, names = _projection(
        NULCDM_KEYS, NULCDM_FID, [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])
    theta = jnp.array([0.0224, 0.12, 0.67, 3.044, 0.965, 0.21],
                      dtype=jnp.float64)
    out = np.asarray(derived_fn(theta), dtype=float)

    assert out[names.index("H0")] == 67.0
    assert out[names.index("Omega_m")] == (0.0224 + 0.12 + 0.21 / 93.14) / 0.67**2
    assert out[names.index("mnu")] == 0.21


def test_derived_projection_mnu_column_is_a_pure_passthrough():
    """mnu is an identity coordinate: its Jacobian row is e_mnu exactly, and it
    is the ONLY derived entry whose value equals a sampled entry bitwise."""
    derived_fn, names = _projection(
        NULCDM_KEYS, NULCDM_FID, [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])
    theta = jnp.array([0.0224, 0.12, 0.67, 3.044, 0.965, 0.13],
                      dtype=jnp.float64)
    jac = np.asarray(jax.jacfwd(derived_fn)(theta), dtype=float)
    row = jac[names.index("mnu")]
    assert row.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    assert float(derived_fn(theta)[names.index("mnu")]) == 0.13


def test_derived_projection_scatters_into_native_slots_and_freezes_the_rest():
    """A comparison basis that is a PERMUTED SUBSET of the native basis.

    Native slots not written keep their fiducial value (so they carry no
    derivative), and the source/native index pairing is honoured -- the failure
    mode this guards is a silently transposed map.
    """
    # comparison basis = (ns, ombh2); native slots 4 and 0. h/omch2/logA frozen.
    derived_fn, names = _projection(
        LCDM_KEYS, LCDM_FID, [0, 1], [4, 0])
    theta = jnp.array([0.99, 0.03], dtype=jnp.float64)
    out = np.asarray(derived_fn(theta), dtype=float)

    h_fid = float(LCDM_FID[2])
    assert out[names.index("H0")] == 100.0 * h_fid          # h stayed fiducial
    assert out[names.index("Omega_m")] == (
        0.03 + float(LCDM_FID[1]) + 0.06 / 93.14) / h_fid**2  # ombh2 came from theta[1]

    jac = np.asarray(jax.jacfwd(derived_fn)(theta), dtype=float)
    assert jac.shape == (3, 2)
    # Omega_m depends on ombh2 (theta[1]) only, and exactly as 1/h^2.
    assert jac[names.index("Omega_m"), 0] == 0.0
    assert jac[names.index("Omega_m"), 1] == 1.0 / h_fid**2
    # H0 depends on neither, since h is frozen.
    assert jac[names.index("H0")].tolist() == [0.0, 0.0]


def test_derived_projection_is_jit_and_vmap_able():
    derived_fn, names = _projection(
        NULCDM_KEYS, NULCDM_FID, [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])
    batch = jnp.asarray(NULCDM_FID)[None, :] + jnp.array(
        [[0.0] * 6, [1e-4, 1e-3, 1e-3, 1e-2, 1e-3, 1e-2]], dtype=jnp.float64)

    single = np.asarray(jnp.stack([derived_fn(row) for row in batch]))
    batched = np.asarray(jax.jit(jax.vmap(derived_fn))(batch))
    assert batched.shape == (2, len(names))
    # sigma8 goes through a quadrature whose summation order XLA is free to
    # change under vmap+jit, so it agrees to rounding, not bitwise...
    np.testing.assert_allclose(batched, single, rtol=1e-14, atol=0.0)
    # ...while the two analytic coordinates, and the mnu passthrough in
    # particular, must be EXACT under batching -- the notebooks assert the
    # projected mnu column is bit-identical to the sampled one.
    for name in ("mnu", "H0"):
        col = names.index(name)
        np.testing.assert_array_equal(batched[:, col], single[:, col])
    np.testing.assert_array_equal(batched[:, names.index("mnu")],
                                  np.asarray(batch)[:, 5])


def test_derived_projection_rejects_mismatched_index_maps():
    with pytest.raises(ValueError, match="index the same map"):
        _projection(LCDM_KEYS, LCDM_FID, [0, 1], [0])
    with pytest.raises(ValueError, match="outside the 5-slot"):
        _projection(LCDM_KEYS, LCDM_FID, [0], [7])


# ---------------------------------------------------------------------------
# format_derived_comparison_rows
# ---------------------------------------------------------------------------

def test_format_derived_comparison_rows_layout_and_statistics():
    names = ("Omega_m", "sigma8", "H0")
    fid = np.array([0.3111, 0.8102, 67.66])
    fisher_sigma = np.array([0.0067088, 0.024277, 0.35686])
    rng = np.random.default_rng(0)
    samples = fid + rng.normal(size=(5000, 3)) * fisher_sigma

    lines, mean, sigma, pulls = format_derived_comparison_rows(
        names, fid, samples, fisher_sigma)

    assert len(lines) == 1 + len(names) + 1
    assert lines[0].split() == ["param", "fid", "MCMC", "mean", "Fisher",
                                "sig", "MCMC", "sig", "ratio"]
    assert lines[1].split()[0] == "Omega_m"
    assert lines[-1].startswith("residual pulls (sigma_F units): ")
    assert "Omega_m=" in lines[-1] and "H0=" in lines[-1]

    np.testing.assert_array_equal(mean, samples.mean(axis=0))
    np.testing.assert_array_equal(sigma, samples.std(axis=0))
    np.testing.assert_array_equal(pulls, (mean - fid) / fisher_sigma)
    # The ratio column is MCMC/Fisher, not its reciprocal.
    assert lines[1].split()[-1] == f"{sigma[0] / fisher_sigma[0]:.2f}"


def test_format_derived_comparison_rows_reproduces_the_notebook_format():
    """Byte-for-byte against the f-strings the notebooks used to inline."""
    names = ("mnu", "Omega_m")
    fid = np.array([0.06, 0.3111])
    fisher_sigma = np.array([0.038089, 0.0067088])
    samples = np.array([[0.05, 0.31], [0.07, 0.3122]])

    lines, mean, sigma, pulls = format_derived_comparison_rows(
        names, fid, samples, fisher_sigma)

    expected = [f"{'param':>9s} {'fid':>10s} {'MCMC mean':>12s} "
                f"{'Fisher sig':>12s} {'MCMC sig':>12s} {'ratio':>7s}"]
    for i, name in enumerate(names):
        expected.append(
            f"{name:>9s} {fid[i]:10.5g} {mean[i]:12.5g} "
            f"{fisher_sigma[i]:12.5g} {sigma[i]:12.5g} "
            f"{sigma[i] / fisher_sigma[i]:7.2f}")
    expected.append("residual pulls (sigma_F units): "
                    + "  ".join(f"{n}={pulls[i]:+.2f}"
                                for i, n in enumerate(names)))
    assert lines == expected


def test_format_derived_comparison_rows_rejects_shape_mismatches():
    names = ("Omega_m", "sigma8")
    with pytest.raises(ValueError, match="fiducial has shape"):
        format_derived_comparison_rows(
            names, np.zeros(3), np.zeros((4, 2)), np.ones(2))
    with pytest.raises(ValueError, match="columns but there are"):
        format_derived_comparison_rows(
            names, np.zeros(2), np.zeros((4, 3)), np.ones(2))
    with pytest.raises(ValueError, match=r"n_draws, n_derived"):
        format_derived_comparison_rows(
            names, np.zeros(2), np.zeros(2), np.ones(2))
