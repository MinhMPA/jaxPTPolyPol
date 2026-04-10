import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jaxptpolypol.covariance import (
    gaussian_bispectrum_covariance,
    gaussian_covariance,
    gaussian_joint_covariance,
)


def _legendre_l2(mu):
    return 0.5 * (3.0 * mu**2 - 1.0)


def _legendre_l4(mu):
    return 0.125 * (35.0 * mu**4 - 30.0 * mu**2 + 3.0)


def _pt(mu, p0, p2, p4):
    return p0 + p2 * _legendre_l2(mu) + p4 * _legendre_l4(mu)


def _symmetry_factor(k1, k2, k3):
    if np.isclose(k1, k2) and np.isclose(k1, k3):
        return 6.0
    if np.isclose(k1, k2) or np.isclose(k1, k3) or np.isclose(k2, k3):
        return 2.0
    return 1.0


def _direct_bispectrum_cov_diag(volume, triangle, triangle_dk, edge_values):
    k1, k2, k3 = triangle
    p01, p21, p41 = edge_values[0]
    p02, p22, p42 = edge_values[1]
    p03, p23, p43 = edge_values[2]

    c12 = (k3**2 - k1**2 - k2**2) / (2.0 * k1 * k2)
    c13 = (k2**2 - k1**2 - k3**2) / (2.0 * k1 * k3)
    s12 = np.sqrt(max(1.0 - c12**2, 0.0))
    s13 = np.sqrt(max(1.0 - c13**2, 0.0))

    mu_nodes, mu_weights = np.polynomial.legendre.leggauss(240)
    phi = np.linspace(0.0, 2.0 * np.pi, 4097)
    cos_phi = np.cos(phi)

    integ_mu = np.zeros_like(mu_nodes)
    for i, mu in enumerate(mu_nodes):
        sin_mu = np.sqrt(max(1.0 - mu**2, 0.0))
        mu1 = mu
        mu2 = mu * c12 - cos_phi * sin_mu * s12
        mu3 = mu * c13 + cos_phi * sin_mu * s13
        integ_phi = (
            _pt(mu1, p01, p21, p41)
            * _pt(mu2, p02, p22, p42)
            * _pt(mu3, p03, p23, p43)
        )
        integ_mu[i] = np.trapz(integ_phi, phi)

    angular_integral = (1.0 / (4.0 * np.pi)) * np.sum(mu_weights * integ_mu)
    prefactor = (
        8.0
        * np.pi**4
        / volume
        * _symmetry_factor(k1, k2, k3)
        / (triangle_dk**3 * k1 * k2 * k3)
    )
    return prefactor * angular_integral


def _b0_legendre_cov_diag(volume, triangle, triangle_dk, edge_values):
    """Reference ``C_BB`` from the Mathematica notebook's Legendre expansion."""
    k1, k2, k3 = triangle
    pt0_1, p2_1, p4_1 = edge_values[0]
    pt0_2, p2_2, p4_2 = edge_values[1]
    pt0_3, p2_3, p4_3 = edge_values[2]

    theta12 = np.arccos(
        np.clip((k3**2 - k1**2 - k2**2) / (2.0 * k1 * k2), -1.0, 1.0)
    )
    theta13 = np.arccos(
        np.clip((k2**2 - k1**2 - k3**2) / (2.0 * k1 * k3), -1.0, 1.0)
    )

    cos = np.cos
    sin = np.sin

    d022 = (1.0 + 3.0 * cos(2.0 * (theta12 + theta13))) / 20.0
    d044 = (
        9.0
        + 20.0 * cos(2.0 * (theta12 + theta13))
        + 35.0 * cos(4.0 * (theta12 + theta13))
    ) / 576.0
    d202 = (1.0 + 3.0 * cos(2.0 * theta13)) / 20.0
    d220 = (1.0 + 3.0 * cos(2.0 * theta12)) / 20.0
    d222 = (
        -1.0
        + 3.0 * cos(2.0 * theta12)
        + 3.0 * cos(2.0 * theta13)
        + 3.0 * cos(2.0 * (theta12 + theta13))
    ) / 140.0
    d224 = (
        6.0
        + 3.0 * cos(2.0 * theta12)
        + 10.0 * cos(2.0 * theta13)
        + 10.0 * cos(2.0 * (theta12 + theta13))
        + 35.0 * cos(2.0 * theta12 + 4.0 * theta13)
    ) / 1120.0
    d242 = (
        6.0
        + 10.0 * cos(2.0 * theta12)
        + 3.0 * cos(2.0 * theta13)
        + 10.0 * cos(2.0 * (theta12 + theta13))
        + 35.0 * cos(4.0 * theta12 + 2.0 * theta13)
    ) / 1120.0
    d244 = 5.0 * (
        -9.0
        + 27.0 * cos(2.0 * theta12)
        + 27.0 * cos(2.0 * theta13)
        - 8.0 * cos(2.0 * (theta12 + theta13))
        + 49.0 * cos(4.0 * (theta12 + theta13))
        + 21.0 * cos(4.0 * theta12 + 2.0 * theta13)
        + 21.0 * cos(2.0 * theta12 + 4.0 * theta13)
    ) / 22176.0
    d404 = (
        9.0
        + 20.0 * cos(2.0 * theta13)
        + 35.0 * cos(4.0 * theta13)
    ) / 576.0
    d422 = (
        3.0
        + 5.0 * cos(2.0 * theta13)
        + cos(2.0 * theta12) * (5.0 + 19.0 * cos(2.0 * theta13))
        + 16.0 * sin(2.0 * theta12) * sin(2.0 * theta13)
    ) / 560.0
    d424 = 5.0 * (
        -9.0
        + 27.0 * cos(2.0 * theta12)
        + 21.0 * cos(2.0 * theta12 - 2.0 * theta13)
        - 8.0 * cos(2.0 * theta13)
        + 49.0 * cos(4.0 * theta13)
        + 27.0 * cos(2.0 * (theta12 + theta13))
        + 21.0 * cos(2.0 * theta12 + 4.0 * theta13)
    ) / 22176.0
    d440 = (
        9.0
        + 20.0 * cos(2.0 * theta12)
        + 35.0 * cos(4.0 * theta12)
    ) / 576.0
    d442 = 5.0 * (
        -8.0 * cos(2.0 * theta12)
        + 49.0 * cos(4.0 * theta12)
        + 3.0
        * (
            -3.0
            + 7.0 * cos(2.0 * theta12 - 2.0 * theta13)
            + 9.0 * cos(2.0 * theta13)
            + 9.0 * cos(2.0 * (theta12 + theta13))
            + 7.0 * cos(4.0 * theta12 + 2.0 * theta13)
        )
    ) / 22176.0
    d444 = 3.0 * (
        81.0
        - 110.0 * cos(2.0 * theta12)
        + 245.0 * cos(4.0 * theta12)
        + 350.0 * cos(2.0 * theta12 - 2.0 * theta13)
        - 110.0 * cos(2.0 * theta13)
        + 245.0 * cos(4.0 * theta13)
        - 110.0 * cos(2.0 * (theta12 + theta13))
        + 245.0 * cos(4.0 * (theta12 + theta13))
        + 350.0 * cos(4.0 * theta12 + 2.0 * theta13)
        + 350.0 * cos(2.0 * theta12 + 4.0 * theta13)
    ) / 256256.0

    cov_integral = (
        pt0_1 * pt0_2 * pt0_3
        + d220 * p2_1 * p2_2 * pt0_3
        + d202 * p2_1 * pt0_2 * p2_3
        + d022 * pt0_1 * p2_2 * p2_3
        + d440 * p4_1 * p4_2 * pt0_3
        + d404 * p4_1 * pt0_2 * p4_3
        + d044 * pt0_1 * p4_2 * p4_3
        + d222 * p2_1 * p2_2 * p2_3
        + d422 * p4_1 * p2_2 * p2_3
        + d242 * p2_1 * p4_2 * p2_3
        + d224 * p2_1 * p2_2 * p4_3
        + d442 * p4_1 * p4_2 * p2_3
        + d424 * p4_1 * p2_2 * p4_3
        + d244 * p2_1 * p4_2 * p4_3
        + d444 * p4_1 * p4_2 * p4_3
    )

    prefactor = (
        8.0
        * np.pi**4
        / volume
        * _symmetry_factor(k1, k2, k3)
        / (triangle_dk**3 * k1 * k2 * k3)
    )
    return prefactor * cov_integral


def test_gaussian_covariance_matches_observable_major_manual_blocks():
    volume = 50.0
    k = np.array([0.1, 0.2])
    dk = 0.01
    p0 = np.array([2.0, 3.0])
    p2 = np.array([5.0, 7.0])
    p4 = np.array([11.0, 13.0])

    nk = volume * k**2 * dk / (4.0 * np.pi**2)

    c00 = p0**2 + (1.0 / 5.0) * p2**2 + (1.0 / 9.0) * p4**2
    c02 = (
        2.0 * p0 * p2
        + (2.0 / 7.0) * p2**2
        + (4.0 / 7.0) * p2 * p4
        + (100.0 / 693.0) * p4**2
    )
    c04 = (
        (18.0 / 35.0) * p2**2
        + 2.0 * p0 * p4
        + (40.0 / 77.0) * p2 * p4
        + (162.0 / 1001.0) * p4**2
    )
    c22 = (
        5.0 * p0**2
        + (20.0 / 7.0) * p0 * p2
        + (20.0 / 7.0) * p0 * p4
        + (15.0 / 7.0) * p2**2
        + (120.0 / 77.0) * p2 * p4
        + (8945.0 / 9009.0) * p4**2
    )
    c24 = (
        (36.0 / 7.0) * p0 * p2
        + (200.0 / 77.0) * p0 * p4
        + (108.0 / 77.0) * p2**2
        + (3578.0 / 1001.0) * p2 * p4
        + (900.0 / 1001.0) * p4**2
    )
    c44 = (
        9.0 * p0**2
        + (360.0 / 77.0) * p0 * p2
        + (2916.0 / 1001.0) * p0 * p4
        + (16101.0 / 5005.0) * p2**2
        + (3240.0 / 1001.0) * p2 * p4
        + (42849.0 / 17017.0) * p4**2
    )

    manual = np.zeros((6, 6))
    blocks = {
        (0, 0): c00,
        (0, 1): c02,
        (0, 2): c04,
        (1, 0): c02,
        (1, 1): c22,
        (1, 2): c24,
        (2, 0): c04,
        (2, 1): c24,
        (2, 2): c44,
    }
    for i in range(3):
        for j in range(3):
            manual[i * 2 : (i + 1) * 2, j * 2 : (j + 1) * 2] = np.diag(
                2.0 * blocks[(i, j)] / nk
            )

    cov = np.array(
        gaussian_covariance(
            volume,
            jnp.array(k),
            dk,
            jnp.array(p0),
            jnp.array(p2),
            jnp.array(p4),
        )
    )
    np.testing.assert_allclose(cov, manual, rtol=1e-12, atol=1e-12)


def test_gaussian_bispectrum_covariance_matches_direct_angular_integration():
    volume = 10.0
    k = np.array([0.1, 0.2, 0.3])
    triangle_dk = 0.01
    pk = np.array(
        [
            [11.0, 12.0, 13.0],
            [1.0, 1.5, 2.0],
            [0.2, 0.3, 0.4],
        ]
    )
    triangles = np.array(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.1, 0.1],
            [0.3, 0.2, 0.1],
        ]
    )

    cov = np.array(
        gaussian_bispectrum_covariance(
            volume,
            jnp.array(k),
            jnp.array(triangles),
            triangle_dk,
            jnp.array(pk),
        )
    )

    ref_diag = []
    for triangle in triangles:
        edge_values = [
            pk[:, np.where(np.isclose(k, triangle[edge]))[0][0]]
            for edge in range(3)
        ]
        ref_diag.append(
            _direct_bispectrum_cov_diag(
                volume,
                triangle,
                triangle_dk,
                edge_values,
            )
        )
    ref_cov = np.diag(ref_diag)

    np.testing.assert_allclose(cov, ref_cov, rtol=3e-6, atol=0.0)


def test_gaussian_bispectrum_covariance_matches_b0_legendre_decomposition():
    volume = 10.0
    k = np.array([0.1, 0.2, 0.3])
    triangle_dk = 0.01
    pk = np.array(
        [
            [11.0, 12.0, 13.0],
            [1.0, 1.5, 2.0],
            [0.2, 0.3, 0.4],
        ]
    )
    triangles = np.array(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.1, 0.1],
            [0.3, 0.2, 0.1],
        ]
    )

    cov = np.array(
        gaussian_bispectrum_covariance(
            volume,
            jnp.array(k),
            jnp.array(triangles),
            triangle_dk,
            jnp.array(pk),
        )
    )

    ref_diag = []
    for triangle in triangles:
        edge_values = [
            pk[:, np.where(np.isclose(k, triangle[edge]))[0][0]]
            for edge in range(3)
        ]
        ref_diag.append(
            _b0_legendre_cov_diag(
                volume,
                triangle,
                triangle_dk,
                edge_values,
            )
        )
    ref_cov = np.diag(ref_diag)

    np.testing.assert_allclose(cov, ref_cov, rtol=1e-12, atol=1e-12)


def test_gaussian_joint_covariance_stacks_pp_and_bb_blocks():
    volume = 10.0
    k = jnp.array([0.1, 0.2, 0.3])
    dk = 0.01
    pk = jnp.array(
        [
            [11.0, 12.0, 13.0],
            [1.0, 1.5, 2.0],
            [0.2, 0.3, 0.4],
        ]
    )
    triangles = jnp.array([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1]])

    cpp = gaussian_covariance(volume, k, dk, pk[0], pk[1], pk[2])
    cbb = gaussian_bispectrum_covariance(volume, k, triangles, dk, pk)
    joint = gaussian_joint_covariance(volume, k, dk, pk, triangles=triangles)

    n_pp = cpp.shape[0]
    n_bb = cbb.shape[0]
    np.testing.assert_allclose(joint[:n_pp, :n_pp], cpp, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        joint[n_pp : n_pp + n_bb, n_pp : n_pp + n_bb],
        cbb,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(joint[:n_pp, n_pp:], 0.0, rtol=0.0, atol=1e-12)


def test_gaussian_joint_covariance_accepts_distinct_bb_power_input():
    volume = 10.0
    k = jnp.array([0.1, 0.2, 0.3])
    dk = 0.01
    pp_pk = jnp.array(
        [
            [11.0, 12.0, 13.0],
            [1.0, 1.5, 2.0],
            [0.2, 0.3, 0.4],
        ]
    )
    bb_pk = jnp.array(
        [
            [7.0, 8.0, 9.0],
            [0.5, 0.75, 1.0],
            [0.05, 0.08, 0.1],
        ]
    )
    triangles = jnp.array([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1]])

    cpp = gaussian_covariance(volume, k, dk, pp_pk[0], pp_pk[1], pp_pk[2])
    cbb = gaussian_bispectrum_covariance(volume, k, triangles, dk, bb_pk)
    joint = gaussian_joint_covariance(
        volume,
        k,
        dk,
        pp_pk,
        triangles=triangles,
        bb_pk_ell=bb_pk,
    )

    n_pp = cpp.shape[0]
    n_bb = cbb.shape[0]
    np.testing.assert_allclose(joint[:n_pp, :n_pp], cpp, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        joint[n_pp : n_pp + n_bb, n_pp : n_pp + n_bb],
        cbb,
        rtol=1e-12,
        atol=1e-12,
    )
