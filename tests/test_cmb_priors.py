import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from jaxptpolypol.cmb import (
    PLANCK_TAU_PRIOR_SIGMA,
    add_tau_prior_to_fisher,
    assert_zero_gradient,
)


def test_planck_tau_prior_sigma_value():
    assert PLANCK_TAU_PRIOR_SIGMA == 0.0073


def test_add_tau_prior_adds_inverse_variance_at_tau_index():
    fisher = np.eye(3)
    order = ("logA", "tau", "A_planck")
    out = add_tau_prior_to_fisher(fisher, order, 0.1)
    expected = np.eye(3)
    expected[1, 1] += 1.0 / 0.1**2
    np.testing.assert_allclose(out, expected)
    # input must not be mutated
    np.testing.assert_allclose(fisher, np.eye(3))


def test_add_tau_prior_none_is_identity():
    fisher = np.arange(9.0).reshape(3, 3)
    out = add_tau_prior_to_fisher(fisher, ("a", "tau", "b"), None)
    np.testing.assert_allclose(out, fisher)


def test_add_tau_prior_rejects_nonpositive_sigma():
    with pytest.raises(ValueError):
        add_tau_prior_to_fisher(np.eye(2), ("tau", "x"), 0.0)


def test_add_tau_prior_missing_tau_raises():
    with pytest.raises(ValueError):
        add_tau_prior_to_fisher(np.eye(2), ("a", "b"), 0.1)


def test_assert_zero_gradient_passes_for_flat_fn():
    flat = lambda th: jnp.sum(0.0 * th)
    assert_zero_gradient(flat, jnp.ones(4))  # must not raise


def test_assert_zero_gradient_raises_for_sloped_fn():
    sloped = lambda th: jnp.sum(th**2)
    with pytest.raises(RuntimeError, match="gradient"):
        assert_zero_gradient(sloped, jnp.ones(4), name="planck_lowl_ee")
