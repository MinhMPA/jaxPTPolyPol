import jax, jax.numpy as jnp, numpy as np, pytest
jax.config.update("jax_enable_x64", True)

from jaxptpolypol.joint_forecast import (
    make_gaussian_fisher_loglike, make_forecast_joint_log_post, embed_fisher,
)

F2 = jnp.array([[4.0, 1.0], [1.0, 9.0]])
CENTER = jnp.array([0.5, -1.0])
IDX = [1, 3]          # picks theta[1], theta[3]
THETA = jnp.array([9.0, 0.7, 9.0, -0.6])   # d = [0.2, 0.4]

def test_gaussian_fisher_loglike_value():
    ll = make_gaussian_fisher_loglike(F2, CENTER, IDX)
    d = np.array([0.2, 0.4])
    expected = -0.5 * d @ np.asarray(F2) @ d
    assert np.isclose(float(ll(THETA)), expected, rtol=0, atol=1e-14)

def test_gaussian_fisher_loglike_zero_at_center():
    ll = make_gaussian_fisher_loglike(F2, CENTER, IDX)
    theta = jnp.zeros(4).at[jnp.array(IDX)].set(CENTER)
    assert float(ll(theta)) == 0.0

def test_gaussian_fisher_loglike_gradient_hits_only_mapped_indices():
    ll = make_gaussian_fisher_loglike(F2, CENTER, IDX)
    g = np.asarray(jax.grad(ll)(THETA))
    assert g[0] == 0.0 and g[2] == 0.0
    assert g[1] != 0.0 and g[3] != 0.0

def test_gaussian_fisher_loglike_shape_validation():
    with pytest.raises(ValueError):
        make_gaussian_fisher_loglike(F2, jnp.zeros(3), IDX)      # center wrong
    with pytest.raises(ValueError):
        make_gaussian_fisher_loglike(F2, CENTER, [0, 1, 2])      # idx wrong

def test_bbn_center_is_fiducial_not_mossa():
    # E10: the 1-D BBN block used by the notebooks must center on the FIDUCIAL.
    ll = make_gaussian_fisher_loglike(
        jnp.array([[1.0 / 0.00036**2]]), jnp.array([0.02242]), [0])
    assert float(ll(jnp.array([0.02242]))) == 0.0
    pull = -2.0 * float(ll(jnp.array([0.02233])))       # Mossa mean, in chi2
    assert np.isclose(np.sqrt(pull), 0.25, atol=0.01)   # the documented 0.25 sigma

def test_joint_log_post_composition_and_slicing():
    pfs = lambda th: -jnp.sum(th**2)                     # sees ONLY theta[:3]
    extra = make_gaussian_fisher_loglike(F2, CENTER, IDX)
    lp = make_forecast_joint_log_post(pfs, n_pfs=3, extra_loglike_fns=(extra,))
    expected = -float(jnp.sum(THETA[:3]**2)) + float(extra(THETA))
    assert np.isclose(float(lp(THETA)), expected, atol=1e-14)
    g = np.asarray(jax.grad(lp)(THETA))                  # tau-analog theta[3]:
    assert g[3] != 0.0                                   # reached ONLY via extra

def test_joint_log_post_n_pfs_validation():
    with pytest.raises(ValueError):
        make_forecast_joint_log_post(lambda th: th.sum(), n_pfs=0)

def test_embed_fisher():
    F = np.asarray(embed_fisher(F2, IDX, 5))
    expected = np.zeros((5, 5))
    expected[np.ix_(IDX, IDX)] = np.asarray(F2)
    assert np.array_equal(F, expected)
