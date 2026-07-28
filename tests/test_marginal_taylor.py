import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


def _toy_full_setup():
    """Full vector: [t0, t1 | l0, l1]  (2 theta_NL, one 'bin' with 2 lin params).

    theory(p) = m0(t) + M(t) @ l  with
      m0(t) = [t0**2 + t1, 3*t1**2, t0*t1]          (quadratic -> H exact)
      M(t)  = [[1+t0, 2], [t1, -1], [0.5, t0+t1]]   (linear -> dM exact)
    """
    from jaxptpolypol.sampler import make_full_params_fn

    def theory(p):
        t0, t1, l0, l1 = p[0], p[1], p[2], p[3]
        m0 = jnp.array([t0**2 + t1, 3.0 * t1**2, t0 * t1])
        M = jnp.array([[1.0 + t0, 2.0], [t1, -1.0], [0.5, t0 + t1]])
        return m0 + M @ jnp.array([l0, l1])

    packed = jnp.array([0.3, -0.2, 9.9, -9.9])   # junk lin values must not matter
    fpf = make_full_params_fn(packed, (0, 1))
    return theory, fpf, packed


def test_builder_exact_on_representable_toy():
    from jaxptpolypol.marginal_taylor import build_taylor_templates

    theory, fpf, packed = _toy_full_setup()
    theta0 = jnp.array([0.3, -0.2])
    tt = build_taylor_templates(
        bin_theory_fns=[theory], bin_lin_idx=[(2, 3)], full_params_fn=fpf,
        theta0=theta0, order2_m0=True, chunk_J=1, chunk_H=1)

    t0, t1 = float(theta0[0]), float(theta0[1])
    np.testing.assert_allclose(np.asarray(tt.bin_m00[0]),
                               [t0**2 + t1, 3 * t1**2, t0 * t1], atol=1e-14)
    np.testing.assert_allclose(np.asarray(tt.bin_M0[0]),
                               [[1 + t0, 2], [t1, -1], [0.5, t0 + t1]], atol=1e-14)
    np.testing.assert_allclose(np.asarray(tt.bin_J[0]),
                               [[2 * t0, 1], [0, 6 * t1], [t1, t0]], atol=1e-12)
    H = np.zeros((3, 2, 2)); H[0, 0, 0] = 2.0; H[1, 1, 1] = 6.0
    H[2, 0, 1] = H[2, 1, 0] = 1.0
    np.testing.assert_allclose(np.asarray(tt.bin_H[0]), H, atol=1e-12)
    dM = np.zeros((3, 2, 2))
    dM[0, 0, 0] = 1.0; dM[1, 0, 1] = 1.0; dM[2, 1, 0] = 1.0; dM[2, 1, 1] = 1.0
    np.testing.assert_allclose(np.asarray(tt.bin_dM[0]), dM, atol=1e-12)


def test_builder_chunking_invariance():
    """Different chunk sizes must give bit-comparable tensors."""
    from jaxptpolypol.marginal_taylor import build_taylor_templates
    theory, fpf, packed = _toy_full_setup()
    theta0 = jnp.array([0.3, -0.2])
    a = build_taylor_templates(bin_theory_fns=[theory], bin_lin_idx=[(2, 3)],
                               full_params_fn=fpf, theta0=theta0, chunk_J=1, chunk_H=1)
    b = build_taylor_templates(bin_theory_fns=[theory], bin_lin_idx=[(2, 3)],
                               full_params_fn=fpf, theta0=theta0, chunk_J=2, chunk_H=2)
    for x, y in ((a.bin_J[0], b.bin_J[0]), (a.bin_H[0], b.bin_H[0]),
                 (a.bin_dM[0], b.bin_dM[0])):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=1e-13)
