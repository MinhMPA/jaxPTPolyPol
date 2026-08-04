"""Tests for the desi_dr1_reanalysis_2511_20757 prior spec machinery."""
import jax
jax.config.update("jax_enable_x64", True)

import textwrap
import numpy as np
import pytest

from jaxptpolypol.desi_priors import (
    DesiPriorSpec, SpecValidationError, load_desi_prior_spec,
)
from jaxptpolypol.marginal_likelihood import LIN_SURVEY_KEYS

TOY_YAML = textwrap.dedent("""\
metadata:
  source: "toy"
  paper_knl: 0.45
  production_k_nl_rsd: 0.45
marginalized:
  shared.bias.bGamma3:
    {paper_mean: null, paper_sigma: 1.0, paper_units: "unit",
     paper_variable: "bGamma3*A_AP*A_amp^2", factor: 1.0, offset: 0.0,
     mean: null, sigma: 1.0, rescale: "A_AP*A_amp^2",
     factor_formula: null, mean_formula: "coevolution_bGamma3"}
  shared.stoch.P_shot:
    {paper_mean: 0.0, paper_sigma: 2.0, paper_units: "unit",
     paper_variable: "P_shot*A_AP", factor: 1.0, offset: 1.0,
     mean: 1.0, sigma: 2.0, rescale: "A_AP",
     factor_formula: null, mean_formula: null}
  pk.ctr.c0:
    {paper_mean: 0.0, paper_sigma: 30.0, paper_units: "(Mpc/h)^2",
     paper_variable: "c0*A_AP*A_amp", factor: 1.0, offset: 0.0,
     mean: 0.0, sigma: 30.0, rescale: "A_AP*A_amp",
     factor_formula: null, mean_formula: null, ctr_rotation: "multipole_to_tilde"}
  pk.ctr.c2:
    {paper_mean: 30.0, paper_sigma: 30.0, paper_units: "(Mpc/h)^2",
     paper_variable: "c2*A_AP*A_amp", factor: 0.5, offset: 0.0,
     mean: 15.0, sigma: 15.0, rescale: "A_AP*A_amp",
     factor_formula: null, mean_formula: null, ctr_rotation: "multipole_to_tilde"}
  pk.ctr.c4:
    {paper_mean: 0.0, paper_sigma: 30.0, paper_units: "(Mpc/h)^2",
     paper_variable: "c4*A_AP*A_amp", factor: 1.0, offset: 0.0,
     mean: 0.0, sigma: 30.0, rescale: "A_AP*A_amp",
     factor_formula: null, mean_formula: null, ctr_rotation: "multipole_to_tilde"}
  pk.ctr.cfog:
    {paper_mean: 400.0, paper_sigma: 400.0, paper_units: "(Mpc/h)^4",
     paper_variable: "ctilde*A_AP*A_amp", factor: 1.0, offset: 0.0,
     mean: 400.0, sigma: 400.0, rescale: "A_AP*A_amp",
     factor_formula: null, mean_formula: null}
  pk.stoch.a0:
    {paper_mean: 0.0, paper_sigma: 1.0, paper_units: "unit",
     paper_variable: "a0*A_AP", factor: 1.0, offset: 0.0,
     mean: 0.0, sigma: 1.0, rescale: "A_AP",
     factor_formula: "knl_over_0p45_sq", mean_formula: null}
  pk.stoch.a2:
    {paper_mean: 0.0, paper_sigma: 1.0, paper_units: "unit",
     paper_variable: "a2*A_AP", factor: 1.0, offset: 0.0,
     mean: 0.0, sigma: 1.0, rescale: "A_AP",
     factor_formula: "knl_over_0p45_sq", mean_formula: null}
  bk.ctr.c1:
    {paper_mean: 0.0, paper_sigma: 5.0, paper_units: "(Mpc/h)^2",
     paper_variable: "c1*A_AP*A_amp", factor: 0.2025, offset: 0.0,
     mean: 0.0, sigma: 1.0125, rescale: "A_AP*A_amp",
     factor_formula: null, mean_formula: null}
  bk.stoch.B_shot:
    {paper_mean: 0.0, paper_sigma: 1.0, paper_units: "unit",
     paper_variable: "B_shot*A_AP", factor: 1.0, offset: 1.0,
     mean: 1.0, sigma: 1.0, rescale: "A_AP",
     factor_formula: null, mean_formula: null}
  bk.stoch.A_shot:
    {paper_mean: 0.0, paper_sigma: 1.0, paper_units: "unit",
     paper_variable: "A_shot*A_AP", factor: 1.0, offset: 0.0,
     mean: 0.0, sigma: 1.0, rescale: "A_AP",
     factor_formula: null, mean_formula: null}
sampled:
  b1: {kind: flat}
  b2:
    {kind: gaussian, paper_mean: 0.0, paper_sigma: 5.0,
     paper_variable: "b2*sigma8(z)^2", rescale: "sigma8_sq"}
  bG2:
    {kind: gaussian, paper_mean: 0.0, paper_sigma: 5.0,
     paper_variable: "bG2*sigma8(z)^2", rescale: "sigma8_sq"}
""")


@pytest.fixture
def toy_spec_path(tmp_path):
    p = tmp_path / "toy_spec.yaml"
    p.write_text(TOY_YAML)
    return p


def test_toy_spec_loads_and_reconciles(toy_spec_path):
    spec = load_desi_prior_spec(toy_spec_path)
    assert isinstance(spec, DesiPriorSpec)
    assert set(spec.marginalized) == set(LIN_SURVEY_KEYS)
    row = spec.marginalized[("pk", "ctr", "c2")]
    assert row.mean == pytest.approx(30.0 * 0.5)
    assert row.sigma == pytest.approx(30.0 * 0.5)
    assert spec.marginalized[("shared", "stoch", "P_shot")].offset == 1.0
    assert spec.sampled["b1"].kind == "flat"


def _mutated_spec_path(tmp_path, mutate):
    """Parse TOY_YAML, apply ``mutate(raw_dict)``, dump to a temp file."""
    import yaml as _yaml
    raw = _yaml.safe_load(TOY_YAML)
    mutate(raw)
    p = tmp_path / "mutated.yaml"
    p.write_text(_yaml.safe_dump(raw))
    return p


def test_reconciliation_failure_raises(tmp_path):
    def mutate(raw):
        raw["marginalized"]["pk.ctr.c2"]["sigma"] = 14.0
    with pytest.raises(SpecValidationError, match="sigma"):
        load_desi_prior_spec(_mutated_spec_path(tmp_path, mutate))


def test_mean_reconciliation_includes_offset(tmp_path):
    def mutate(raw):
        raw["marginalized"]["shared.stoch.P_shot"]["mean"] = 0.0
    with pytest.raises(SpecValidationError, match="mean"):
        load_desi_prior_spec(_mutated_spec_path(tmp_path, mutate))


def test_missing_lin_key_raises(tmp_path):
    def mutate(raw):
        del raw["marginalized"]["bk.stoch.A_shot"]
    with pytest.raises(SpecValidationError, match="A_shot"):
        load_desi_prior_spec(_mutated_spec_path(tmp_path, mutate))


def test_unknown_rescale_token_raises(tmp_path):
    def mutate(raw):
        raw["marginalized"]["shared.bias.bGamma3"]["rescale"] = "A_AP^3"
    with pytest.raises(SpecValidationError, match="rescale"):
        load_desi_prior_spec(_mutated_spec_path(tmp_path, mutate))


def test_ctr_rotation_partial_trio_raises(tmp_path):
    """The c0/c2/c4 ctr_rotation token must be all-or-none across the trio."""
    def mutate(raw):
        # Leave only c0 carrying the token; c2 and c4 drop it.
        raw["marginalized"]["pk.ctr.c2"].pop("ctr_rotation", None)
        raw["marginalized"]["pk.ctr.c4"].pop("ctr_rotation", None)
    with pytest.raises(SpecValidationError, match="ctr_rotation"):
        load_desi_prior_spec(_mutated_spec_path(tmp_path, mutate))


def test_real_spec_loads():
    spec = load_desi_prior_spec()
    assert set(spec.marginalized) == set(LIN_SURVEY_KEYS)
    assert spec.metadata["source"] == "arXiv:2511.20757"


def test_real_spec_verbatim_anchor_rows():
    """Rows recorded in CONTEXT.md from the primary PDF; if the reconciled
    map contradicts one of these, the MAP governs -- update CONTEXT.md and
    this test together in the same commit, quoting the paper."""
    spec = load_desi_prior_spec()
    c1 = spec.marginalized[("bk", "ctr", "c1")]
    assert c1.paper_mean == 0.0 and c1.paper_sigma == 5.0
    cfog = spec.marginalized[("pk", "ctr", "cfog")]
    assert cfog.paper_mean == 400.0 and cfog.paper_sigma == 400.0
    c2 = spec.marginalized[("pk", "ctr", "c2")]
    assert c2.paper_mean == 30.0
    for nm in ("b2", "bG2"):
        assert spec.sampled[nm].paper_sigma == 5.0
        assert spec.sampled[nm].rescale == "sigma8_sq"
    bg3 = spec.marginalized[("shared", "bias", "bGamma3")]
    assert bg3.mean_formula == "coevolution_bGamma3"
    for k in (("pk", "stoch", "a0"), ("pk", "stoch", "a2")):
        assert spec.marginalized[k].factor_formula == "knl_over_0p45_sq"


def test_real_spec_ctr_rotation_trio():
    """sigmap branch: the c0/c2/c4 trio carries ctr_rotation all-or-none."""
    spec = load_desi_prior_spec()
    for key in (("pk", "ctr", "c0"), ("pk", "ctr", "c2"), ("pk", "ctr", "c4")):
        assert spec.marginalized[key].ctr_rotation == "multipole_to_tilde"
    # non-ctr rows leave the token null
    assert spec.marginalized[("pk", "ctr", "cfog")].ctr_rotation is None


# =============================================================================
# Task 5sigma: make_desi_prior_fns (cov-mode + diag regression) + helpers
# =============================================================================
import yaml
import jax.numpy as jnp

from jaxptpolypol.desi_priors import (
    DESI_F_FID, ctr_rotation_matrices, make_desi_prior_fns,
)
from jaxptpolypol.marginal_likelihood import split_marginal_indices


N_COSMO = 2
N_BINS = 2
KNL_BINS = (0.52, 0.65)
TOY_F_FID = (0.8, 0.9)                    # transparent 2-bin f for the oracle
SURVEY_KEYS_TOY = tuple(LIN_SURVEY_KEYS) + (
    ('shared', 'bias', 'b1'), ('shared', 'bias', 'b2'),
    ('shared', 'bias', 'bG2'))
_CTR_KEYS = (('pk', 'ctr', 'c0'), ('pk', 'ctr', 'c2'), ('pk', 'ctr', 'c4'))


def _toy_rescaling():
    sigma8_ref = jnp.array([0.6, 0.5])

    def sigma8_bins_fn(theta_nl):
        return sigma8_ref * (1.0 + 0.1 * theta_nl[0])

    def a_ap_bins_fn(theta_nl):
        return jnp.ones(N_BINS) * (1.0 + 0.2 * theta_nl[1])

    return sigma8_ref, sigma8_bins_fn, a_ap_bins_fn


@pytest.fixture
def toy_setup(toy_spec_path):
    """Cov-mode: the toy fixture carries the ctr_rotation token."""
    spec = load_desi_prior_spec(toy_spec_path)
    split = split_marginal_indices(
        n_cosmo_params=N_COSMO, survey_keys=SURVEY_KEYS_TOY, n_bins=N_BINS)
    sigma8_ref, sigma8_bins_fn, a_ap_bins_fn = _toy_rescaling()
    fns = make_desi_prior_fns(
        spec, split=split, knl_bins=KNL_BINS,
        sigma8_bins_fn=sigma8_bins_fn, a_ap_bins_fn=a_ap_bins_fn,
        sigma8_ref_bins=sigma8_ref, f_bins=TOY_F_FID)
    return spec, split, fns


@pytest.fixture
def toyless_setup(tmp_path):
    """Diag-mode regression: same toy fixture with the ctr_rotation popped."""
    raw = yaml.safe_load(TOY_YAML)
    for k in ("pk.ctr.c0", "pk.ctr.c2", "pk.ctr.c4"):
        raw["marginalized"][k].pop("ctr_rotation", None)
    p = tmp_path / "toyless.yaml"
    p.write_text(yaml.safe_dump(raw))
    spec = load_desi_prior_spec(p)
    split = split_marginal_indices(
        n_cosmo_params=N_COSMO, survey_keys=SURVEY_KEYS_TOY, n_bins=N_BINS)
    sigma8_ref, sigma8_bins_fn, a_ap_bins_fn = _toy_rescaling()
    fns = make_desi_prior_fns(               # f_bins defaults (unused in diag mode)
        spec, split=split, knl_bins=KNL_BINS,
        sigma8_bins_fn=sigma8_bins_fn, a_ap_bins_fn=a_ap_bins_fn,
        sigma8_ref_bins=sigma8_ref)
    return spec, split, fns


# --- cov-mode: adapted base-brief toy tests (non-ctr diagonal reads) ----------


def test_prior_fns_shapes_and_fiducial_values(toy_setup):
    spec, split, (mean_fn, sigma_fn, _) = toy_setup
    theta0 = jnp.zeros(split.n_nl)          # fiducial: R = 1 everywhere
    mu, sig = mean_fn(theta0), sigma_fn(theta0)
    n = len(LIN_SURVEY_KEYS)
    assert mu.shape == (N_BINS * n,)
    assert sig.shape == (N_BINS, n, n)      # cov-mode: stacked per-bin blocks
    j_a0 = LIN_SURVEY_KEYS.index(('pk', 'stoch', 'a0'))
    j_ps = LIN_SURVEY_KEYS.index(('shared', 'stoch', 'P_shot'))
    for b, knl in enumerate(KNL_BINS):      # non-ctr widths read off block diag
        assert sig[b][j_a0, j_a0] ** 0.5 == pytest.approx(1.0 * (knl / 0.45) ** 2)
        assert mu[b * n + j_ps] == pytest.approx(1.0)


def test_bGamma3_coevolution_mean(toy_setup):
    spec, split, (mean_fn, _, _) = toy_setup
    theta = jnp.zeros(split.n_nl)
    b1_vals = (1.7, 2.1)
    for pos, v in zip(split.nl_b1_pos, b1_vals):
        theta = theta.at[pos].set(v)
    mu = mean_fn(theta)
    j = LIN_SURVEY_KEYS.index(('shared', 'bias', 'bGamma3'))
    n = len(LIN_SURVEY_KEYS)
    row = spec.marginalized[('shared', 'bias', 'bGamma3')]
    for b, b1 in enumerate(b1_vals):
        expected = (23.0 / 42.0) * (b1 - 1.0) * row.factor + row.offset
        assert mu[b * n + j] == pytest.approx(expected)   # R=1 at theta cosmo=0


def test_layer2_rescaling_divides(toy_setup):
    spec, split, (mean_fn, sigma_fn, _) = toy_setup
    theta = jnp.zeros(split.n_nl).at[1].set(0.5)   # a_ap = 1.1, sigma8 unchanged
    a_ap = 1.0 + 0.2 * 0.5
    sig0 = sigma_fn(jnp.zeros(split.n_nl))
    sig = sigma_fn(theta)
    j_ps = LIN_SURVEY_KEYS.index(('shared', 'stoch', 'P_shot'))  # rescale A_AP
    n = len(LIN_SURVEY_KEYS)
    for b in range(N_BINS):
        assert (sig[b][j_ps, j_ps] ** 0.5
                == pytest.approx(sig0[b][j_ps, j_ps] ** 0.5 / a_ap))
    mu = mean_fn(theta)
    assert mu[j_ps] == pytest.approx(1.0 / a_ap)


def test_layer2_amp_powers(toy_setup):
    spec, split, (mean_fn, sigma_fn, _) = toy_setup
    theta = jnp.zeros(split.n_nl).at[0].set(0.5)   # sigma8 *= 1.05 -> A_amp=1.05**2
    a_amp = 1.05 ** 2
    sig0, sig = sigma_fn(jnp.zeros(split.n_nl)), sigma_fn(theta)
    j_bg3 = LIN_SURVEY_KEYS.index(('shared', 'bias', 'bGamma3'))  # A_AP*A_amp^2
    j_a0 = LIN_SURVEY_KEYS.index(('pk', 'stoch', 'a0'))           # A_AP only
    for b in range(N_BINS):
        assert (sig[b][j_bg3, j_bg3] ** 0.5
                == pytest.approx(sig0[b][j_bg3, j_bg3] ** 0.5 / a_amp ** 2))
        assert (sig[b][j_a0, j_a0] ** 0.5
                == pytest.approx(sig0[b][j_a0, j_a0] ** 0.5))     # A_amp-independent


def test_log_prior_nl_gaussian_and_flat(toy_setup):
    spec, split, (_, _, log_prior_nl) = toy_setup
    theta0 = jnp.zeros(split.n_nl)
    theta_b1 = theta0.at[split.nl_b1_pos[0]].set(3.0)
    assert log_prior_nl(theta_b1) == pytest.approx(float(log_prior_nl(theta0)))
    b2_pos = split.nl_b1_pos[0] + 1
    theta_b2 = theta0.at[b2_pos].set(1.0)
    sigma8_ref = 0.6
    w = 5.0 / sigma8_ref ** 2
    expected_delta = -0.5 * (1.0 / w) ** 2
    got = float(log_prior_nl(theta_b2) - log_prior_nl(theta0))
    assert got == pytest.approx(expected_delta)


def test_prior_fns_are_jit_and_grad_safe(toy_setup):
    spec, split, (mean_fn, sigma_fn, log_prior_nl) = toy_setup
    theta = jnp.full(split.n_nl, 0.1)
    assert jnp.allclose(jax.jit(mean_fn)(theta), mean_fn(theta))
    assert jnp.allclose(jax.jit(sigma_fn)(theta), sigma_fn(theta))
    g = jax.grad(lambda t: jnp.sum(sigma_fn(t)) + log_prior_nl(t))(theta)
    assert jnp.all(jnp.isfinite(g))


# --- cov-mode: analytic L.Sigma.L^T oracle at AND off fiducial ----------------


def test_ctr_cov_mode_block_and_mean_oracle(toy_setup):
    """block(2:5,2:5) == L diag(sigma_paper^2) L^T / R^2 ; mean == L (paper)/R."""
    spec, split, (mean_fn, sigma_fn, _) = toy_setup
    n = len(LIN_SURVEY_KEYS)
    c0 = LIN_SURVEY_KEYS.index(('pk', 'ctr', 'c0'))
    L = np.asarray(ctr_rotation_matrices(jnp.asarray(TOY_F_FID)))
    paper_sigma = np.array([spec.marginalized[k].paper_sigma for k in _CTR_KEYS])
    paper_mean = np.array([spec.marginalized[k].paper_mean for k in _CTR_KEYS])
    theta_fid = jnp.zeros(split.n_nl)
    theta_off = jnp.zeros(split.n_nl).at[0].set(0.4).at[1].set(0.5)
    for theta, (a, d) in ((theta_fid, (0.0, 0.0)), (theta_off, (0.4, 0.5))):
        a_amp = (1.0 + 0.1 * a) ** 2
        a_ap = 1.0 + 0.2 * d
        R = a_ap * a_amp                     # ctr rescale A_AP*A_amp
        blocks = np.asarray(sigma_fn(theta))
        mu = np.asarray(mean_fn(theta))
        assert blocks.shape == (N_BINS, n, n)
        for b in range(N_BINS):
            Lb = L[b]
            exp_block = Lb @ np.diag((paper_sigma / R) ** 2) @ Lb.T
            got_block = blocks[b][c0:c0 + 3, c0:c0 + 3]
            np.testing.assert_allclose(got_block, exp_block, rtol=1e-12, atol=1e-12)
            exp_mean = Lb @ (paper_mean / R)
            got_mean = mu[b * n + c0: b * n + c0 + 3]
            np.testing.assert_allclose(got_mean, exp_mean, rtol=1e-12, atol=1e-12)


# --- diag-mode regression: token-less trio -> base-brief behaviour verbatim ---


def test_diag_mode_tokenless_shapes_and_fiducial(toyless_setup):
    """The base brief's ORIGINAL diagonal assertions, run verbatim."""
    spec, split, (mean_fn, sigma_fn, _) = toyless_setup
    theta0 = jnp.zeros(split.n_nl)
    mu, sig = mean_fn(theta0), sigma_fn(theta0)
    assert mu.shape == sig.shape == (N_BINS * len(LIN_SURVEY_KEYS),)
    j_c2 = LIN_SURVEY_KEYS.index(('pk', 'ctr', 'c2'))
    j_a0 = LIN_SURVEY_KEYS.index(('pk', 'stoch', 'a0'))
    j_ps = LIN_SURVEY_KEYS.index(('shared', 'stoch', 'P_shot'))
    n = len(LIN_SURVEY_KEYS)
    for b, knl in enumerate(KNL_BINS):
        assert sig[b * n + j_c2] == pytest.approx(15.0)
        assert sig[b * n + j_a0] == pytest.approx(1.0 * (knl / 0.45) ** 2)
        assert mu[b * n + j_ps] == pytest.approx(1.0)


def test_diag_mode_tokenless_ctr_uses_plain_diagonal(toyless_setup):
    """In diag mode the ctr rows follow the plain 1/R machinery (base brief)."""
    spec, split, (mean_fn, sigma_fn, _) = toyless_setup
    theta = jnp.zeros(split.n_nl).at[1].set(0.5)   # a_ap = 1.1
    a_ap = 1.0 + 0.2 * 0.5
    sig0, sig = sigma_fn(jnp.zeros(split.n_nl)), sigma_fn(theta)
    j_c0 = LIN_SURVEY_KEYS.index(('pk', 'ctr', 'c0'))    # rescale A_AP*A_amp
    assert sig[j_c0] == pytest.approx(sig0[j_c0] / a_ap)   # A_amp = 1 here


# --- helpers: DESI_F_FID vs growth rate, and L(f) structure -------------------


def test_desi_f_fid_matches_growth_rate_approx():
    """The hardcoded f(z) tuple is reproduced by growth_rate_approx within 2e-3."""
    from ps_1loop_jax import background as bg
    z_bins = (0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.2)
    assert len(DESI_F_FID) == len(z_bins)
    for z, f in zip(z_bins, DESI_F_FID):
        f_approx = float(bg.growth_rate_approx(0.02242, 0.11933, 0.6766, z, 0.06))
        assert abs(f_approx - f) < 2e-3, (z, f, f_approx)


def test_ctr_rotation_matrices_structure():
    f = (0.8, 0.9)
    L = np.asarray(ctr_rotation_matrices(jnp.asarray(f)))
    assert L.shape == (2, 3, 3)
    for b, fb in enumerate(f):
        exp = np.array([[1.0, -fb / 3.0, 3.0 * fb ** 2 / 35.0],
                        [0.0, 1.0, -6.0 * fb / 7.0],
                        [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(L[b], exp, rtol=1e-14, atol=1e-14)


def _synthetic_tt(seed=20260731, n_pts=5, n_bins=2, p=11, d=4, order2=True):
    """Tiny random-float64 TaylorTemplates for the column-identity test."""
    from jaxptpolypol.marginal_taylor import TaylorTemplates
    rng = np.random.default_rng(seed)
    theta0 = jnp.asarray(rng.standard_normal(d))
    bin_m00 = tuple(jnp.asarray(rng.standard_normal(n_pts)) for _ in range(n_bins))
    bin_J = tuple(jnp.asarray(rng.standard_normal((n_pts, d))) for _ in range(n_bins))
    bin_H = (tuple(jnp.asarray(rng.standard_normal((n_pts, d, d)))
                   for _ in range(n_bins)) if order2 else (None,) * n_bins)
    bin_M0 = tuple(jnp.asarray(rng.standard_normal((n_pts, p))) for _ in range(n_bins))
    bin_dM = tuple(jnp.asarray(rng.standard_normal((n_pts, p, d)))
                   for _ in range(n_bins))
    return TaylorTemplates(
        theta0=theta0, bin_m00=bin_m00, bin_J=bin_J, bin_H=bin_H,
        bin_M0=bin_M0, bin_dM=bin_dM, order2_m0=order2,
        build_diagnostics={"origin": "synthetic"})


def test_rotate_taylor_templates_column_identity():
    """M(delta) from ROTATED templates applied to theta equals M(delta) from
    UNROTATED templates applied to theta' with theta'[(2,3,4)] = L_b @ theta[(2,3,4)]
    (exact linear reparameterization of the ctr slots); m0/J/H untouched."""
    from jaxptpolypol.desi_priors import (
        ctr_rotation_matrices, rotate_taylor_templates,
    )
    tt = _synthetic_tt()
    n_bins = len(tt.bin_M0)
    f_bins = (0.8155, 0.9301)
    assert len(f_bins) == n_bins
    L_bins = ctr_rotation_matrices(f_bins)
    rtt = rotate_taylor_templates(tt, L_bins)

    cols = np.array([2, 3, 4])
    rng = np.random.default_rng(4242)
    for b in range(n_bins):
        L_b = np.asarray(L_bins[b])
        p = int(tt.bin_M0[b].shape[1])
        d = int(tt.theta0.shape[0])
        for _ in range(4):
            theta = rng.standard_normal(p)
            delta = jnp.asarray(rng.standard_normal(d))
            M_rot = rtt.bin_M0[b] + jnp.einsum("ijk,k->ij", rtt.bin_dM[b], delta)
            lhs = M_rot @ jnp.asarray(theta)
            theta_p = theta.copy()
            theta_p[cols] = L_b @ theta[cols]
            M_un = tt.bin_M0[b] + jnp.einsum("ijk,k->ij", tt.bin_dM[b], delta)
            rhs = M_un @ jnp.asarray(theta_p)
            assert jnp.allclose(lhs, rhs, rtol=0.0, atol=1e-13)

    # m0/J/H/theta0 pass through untouched (identical arrays).
    assert jnp.array_equal(rtt.theta0, tt.theta0)
    assert rtt.order2_m0 == tt.order2_m0
    for b in range(n_bins):
        assert jnp.array_equal(rtt.bin_m00[b], tt.bin_m00[b])
        assert jnp.array_equal(rtt.bin_J[b], tt.bin_J[b])
        assert jnp.array_equal(rtt.bin_H[b], tt.bin_H[b])
        # only ctr columns (2,3,4) of M0/dM changed; other columns identical.
        other = [j for j in range(int(tt.bin_M0[b].shape[1])) if j not in (2, 3, 4)]
        assert jnp.array_equal(rtt.bin_M0[b][:, other], tt.bin_M0[b][:, other])
        assert jnp.array_equal(rtt.bin_dM[b][:, other, :], tt.bin_dM[b][:, other, :])


# =============================================================================
# Task 6sigma: build_prior_sigmas_from_desi_spec (Fisher-side fiducial widths)
# =============================================================================
from jaxptpolypol.desi_priors import build_prior_sigmas_from_desi_spec


def _ctr_marginal_widths(paper_sigmas, f):
    """Analytic sqrt(diag(L(f) . diag(paper^2) . L(f)^T)) for one bin's f.

    Rows of L(f) = [[1, -f/3, 3f^2/35], [0, 1, -6f/7], [0, 0, 1]] give:
    c0 = sqrt(s0^2 + (f/3)^2 s2^2 + (3f^2/35)^2 s4^2);
    c2 = sqrt(s2^2 + (6f/7)^2 s4^2);  c4 = s4 (bottom row of L).
    """
    s0, s2, s4 = paper_sigmas
    c0 = np.sqrt(s0 ** 2 + (f / 3.0) ** 2 * s2 ** 2
                 + (3.0 * f ** 2 / 35.0) ** 2 * s4 ** 2)
    c2 = np.sqrt(s2 ** 2 + (6.0 * f / 7.0) ** 2 * s4 ** 2)
    c4 = s4
    return c0, c2, c4


def test_fisher_sigmas_from_spec(toy_spec_path):
    """Cov-mode (the toy fixture carries the ctr_rotation token): the ctr trio
    emits the MARGINAL widths of the correlated L.Sigma.L^T prior; non-ctr rows
    follow the base-brief formulas unchanged."""
    spec = load_desi_prior_spec(toy_spec_path)
    sigma8_ref = np.array([0.6, 0.5])
    survey_dicts, sampled_bins = build_prior_sigmas_from_desi_spec(
        spec, knl_bins=KNL_BINS, sigma8_ref_bins=sigma8_ref, f_bins=TOY_F_FID)
    assert len(survey_dicts) == len(KNL_BINS)
    # ctr paper sigmas read off the loaded toy spec. c2's PAPER sigma is 30 even
    # though its MAPPED sigma is 15 (factor 0.5) -- the marginal-width oracle uses
    # PAPER 30, consistent with make_desi_prior_fns cov-mode. So the base brief's
    # c2 == 15.0 assertion is REPLACED here by the marginal width.
    paper_ctr = [spec.marginalized[k].paper_sigma for k in _CTR_KEYS]
    for b, (knl, f) in enumerate(zip(KNL_BINS, TOY_F_FID)):
        c0_w, c2_w, c4_w = _ctr_marginal_widths(paper_ctr, f)
        assert survey_dicts[b][("pk", "ctr", "c4")] == pytest.approx(c4_w)
        assert survey_dicts[b][("pk", "ctr", "c2")] == pytest.approx(c2_w)
        assert survey_dicts[b][("pk", "ctr", "c0")] == pytest.approx(c0_w)
        # non-ctr rows: base-brief behaviour unchanged
        assert survey_dicts[b][("pk", "stoch", "a0")] == pytest.approx(
            (knl / 0.45) ** 2)
        assert survey_dicts[b][("bk", "ctr", "c1")] == pytest.approx(1.0125)
        assert sampled_bins[b]["b2"] == pytest.approx(5.0 / sigma8_ref[b] ** 2)
        assert "b1" not in sampled_bins[b]


def test_fisher_sigmas_real_spec():
    """Real DESI spec: bin-0 ctr marginal width at DESI_F_FID; non-ctr rows via
    the base formulas. c0/c2/c4 paper sigmas are all 30 (map section 3.1 table)."""
    spec = load_desi_prior_spec()
    # 7 bins to match DESI_F_FID; bin 0 knl = 0.52 (only bin 0 is asserted).
    knl_bins = (0.52, 0.65, 0.75, 0.80, 0.85, 0.90, 0.95)
    assert len(knl_bins) == len(DESI_F_FID)
    sigma8_ref = np.full(len(DESI_F_FID), 0.6)
    survey_dicts, sampled_bins = build_prior_sigmas_from_desi_spec(
        spec, knl_bins=knl_bins, sigma8_ref_bins=sigma8_ref)   # f_bins=DESI_F_FID
    f0 = 0.8155
    assert f0 == pytest.approx(DESI_F_FID[0])
    exp_c0 = np.sqrt(30.0 ** 2 + (f0 / 3.0) ** 2 * 30.0 ** 2
                     + (3.0 * f0 ** 2 / 35.0) ** 2 * 30.0 ** 2)
    assert exp_c0 == pytest.approx(31.14, abs=1e-2)
    assert survey_dicts[0][("pk", "ctr", "c0")] == pytest.approx(exp_c0)
    # cfog: non-ctr, non-knl -> mapped paper sigma verbatim
    assert survey_dicts[0][("pk", "ctr", "cfog")] == pytest.approx(400.0)
    # a0: knl_over_0p45_sq -> paper_sigma * (knl_0/0.45)^2
    a0_paper = spec.marginalized[("pk", "stoch", "a0")].paper_sigma
    assert survey_dicts[0][("pk", "stoch", "a0")] == pytest.approx(
        (0.52 / 0.45) ** 2 * a0_paper)


def test_fisher_ctr_blocks_return(toy_spec_path):
    """return_ctr_blocks=True yields the stacked (n_bins, 3, 3) fiducial (R=1)
    ctr covariance L.diag(paper^2).L^T: symmetric, diag == marginal widths^2,
    corr(c2, c4) closed-form for the toy f (paper_c2 == paper_c4)."""
    spec = load_desi_prior_spec(toy_spec_path)
    sigma8_ref = np.array([0.6, 0.5])
    survey_dicts, sampled_bins, blocks = build_prior_sigmas_from_desi_spec(
        spec, knl_bins=KNL_BINS, sigma8_ref_bins=sigma8_ref,
        f_bins=TOY_F_FID, return_ctr_blocks=True)
    blocks = np.asarray(blocks)
    assert blocks.shape == (N_BINS, 3, 3)
    paper_ctr = [spec.marginalized[k].paper_sigma for k in _CTR_KEYS]
    for b, f in enumerate(TOY_F_FID):
        Bb = blocks[b]
        np.testing.assert_allclose(Bb, Bb.T, rtol=1e-14, atol=1e-14)
        c0_w, c2_w, c4_w = _ctr_marginal_widths(paper_ctr, f)
        np.testing.assert_allclose(
            np.diag(Bb), np.array([c0_w, c2_w, c4_w]) ** 2, rtol=1e-12)
        # diag also matches the survey-dict marginal widths (same underlying block)
        for k, w in zip(_CTR_KEYS, (c0_w, c2_w, c4_w)):
            assert survey_dicts[b][k] == pytest.approx(w)
        # corr(c2, c4) = -(6f/7)/sqrt(1 + (6f/7)^2) when paper_c2 == paper_c4
        g = 6.0 * f / 7.0
        exp_corr = -g / np.sqrt(1.0 + g ** 2)
        corr = Bb[1, 2] / np.sqrt(Bb[1, 1] * Bb[2, 2])
        assert corr == pytest.approx(exp_corr, rel=1e-12)
    # default call (no flag) still returns the 2-tuple
    out2 = build_prior_sigmas_from_desi_spec(
        spec, knl_bins=KNL_BINS, sigma8_ref_bins=sigma8_ref, f_bins=TOY_F_FID)
    assert len(out2) == 2


# =============================================================================
# Final-review guards: ctr_rotation trio rescale-consistency + bogus token value
# =============================================================================


def test_ctr_rotation_trio_rescale_mismatch_raises(tmp_path):
    """When the c0/c2/c4 trio carries ctr_rotation, all three rows must share one
    rescale token: cov-mode make_desi_prior_fns reuses the c0-slot R_ctr for the
    whole 3x3 ctr block, so a divergent token would be silently wrong."""
    def mutate(raw):
        # c2 keeps its ctr_rotation (all-or-none passes) but flips its rescale.
        raw["marginalized"]["pk.ctr.c2"]["rescale"] = "A_AP"
    with pytest.raises(SpecValidationError, match="rescale"):
        load_desi_prior_spec(_mutated_spec_path(tmp_path, mutate))


def test_ctr_rotation_bogus_token_value_raises(tmp_path):
    """An invalid ctr_rotation token VALUE on all three trio rows (so the
    all-or-none check passes) is rejected by the per-row value validation."""
    def mutate(raw):
        for k in ("pk.ctr.c0", "pk.ctr.c2", "pk.ctr.c4"):
            raw["marginalized"][k]["ctr_rotation"] = "bogus_rotation"
    with pytest.raises(SpecValidationError, match="ctr_rotation"):
        load_desi_prior_spec(_mutated_spec_path(tmp_path, mutate))


# =============================================================================
# Tier-3 c1-sampled: reduced lin_keys threading + sampled_marginal_priors
# (make_desi_prior_fns extension; TDD). The c1-sampled split marginalizes only
# 10 keys/bin (c1 removed) and samples c1, whose prior arrives via
# sampled_marginal_priors carrying the SAME A_AP*A_amp rescale the marginalized
# c1 row would have carried.
# =============================================================================

_C1_KEY = ("bk", "ctr", "c1")
LIN_KEYS_NO_C1 = tuple(k for k in LIN_SURVEY_KEYS if k != _C1_KEY)


def _split_no_c1():
    return split_marginal_indices(
        n_cosmo_params=N_COSMO, survey_keys=SURVEY_KEYS_TOY, n_bins=N_BINS,
        lin_survey_keys=LIN_KEYS_NO_C1)


def _nl_positions(split, key):
    """Per-bin theta_NL indices of a sampled survey key, from the split itself.

    (In this toy, SURVEY_KEYS_TOY puts the LIN block first, so c1 sits BEFORE
    b1 in each bin's nl sub-block -- unlike production, where b1/b2/bG2 come
    first and c1 lands at nl_b1_pos+3. Computing from the split covers both.)"""
    n_survey = len(SURVEY_KEYS_TOY)
    off = SURVEY_KEYS_TOY.index(key)
    nl_pos = {full: pos for pos, full in enumerate(split.nl_idx)}
    return [nl_pos[N_COSMO + b * n_survey + off] for b in range(N_BINS)]


def _c1_nl_positions(split):
    """Per-bin theta_NL indices of the sampled c1."""
    return _nl_positions(split, _C1_KEY)


def test_reduced_lin_keys_cov_blocks_drop_c1(toy_spec_path):
    """With lin_keys = LIN_SURVEY_KEYS minus c1 (10 keys), the cov-mode
    prior_sigma_fn returns (n_bins, 10, 10) blocks with NO c1 row/column, and
    the ctr trio still lands at slots 2,3,4 (c1 was slot 8, after the trio)."""
    spec = load_desi_prior_spec(toy_spec_path)
    split = _split_no_c1()
    assert split.n_lin == N_BINS * 10          # 10 marginalized per bin
    assert _C1_KEY not in [k for _, k in split.lin_keys]
    sigma8_ref, s8_fn, aap_fn = _toy_rescaling()
    mean_fn, sigma_fn, _ = make_desi_prior_fns(
        spec, split=split, knl_bins=KNL_BINS, sigma8_bins_fn=s8_fn,
        a_ap_bins_fn=aap_fn, sigma8_ref_bins=sigma8_ref, f_bins=TOY_F_FID,
        lin_keys=LIN_KEYS_NO_C1)
    theta0 = jnp.zeros(split.n_nl)
    mu, sig = mean_fn(theta0), sigma_fn(theta0)
    assert mu.shape == (N_BINS * 10,)
    assert sig.shape == (N_BINS, 10, 10)       # cov-mode, c1 row absent
    # ctr trio slot positions unchanged by dropping c1.
    assert LIN_KEYS_NO_C1.index(("pk", "ctr", "c0")) == 2
    # non-ctr diagonal read still matches the base formulas (a0 knl rescale).
    j_a0 = LIN_KEYS_NO_C1.index(("pk", "stoch", "a0"))
    for b, knl in enumerate(KNL_BINS):
        assert sig[b][j_a0, j_a0] ** 0.5 == pytest.approx(1.0 * (knl / 0.45) ** 2)
    # cov-mode ctr 3x3 block matches the L.diag.L^T oracle (c1 removal did not
    # disturb the trio).
    L = np.asarray(ctr_rotation_matrices(jnp.asarray(TOY_F_FID)))
    paper_sigma = np.array([spec.marginalized[k].paper_sigma for k in _CTR_KEYS])
    for b in range(N_BINS):
        exp = L[b] @ np.diag(paper_sigma ** 2) @ L[b].T   # R = 1 at fiducial
        np.testing.assert_allclose(
            np.asarray(sig[b])[2:5, 2:5], exp, rtol=1e-12, atol=1e-12)


def test_reduced_lin_keys_diag_mode_shapes(toyless_setup_no_c1):
    """Diag-mode (token-less trio) reduced lin_keys -> (n_bins*10,) flat vectors."""
    spec, split, (mean_fn, sigma_fn, _) = toyless_setup_no_c1
    theta0 = jnp.zeros(split.n_nl)
    mu, sig = mean_fn(theta0), sigma_fn(theta0)
    assert mu.shape == sig.shape == (N_BINS * 10,)
    assert _C1_KEY not in [k for _, k in split.lin_keys]


@pytest.fixture
def toyless_setup_no_c1(tmp_path):
    """Diag-mode + reduced lin_keys (c1 dropped from both spec-rotation and lin)."""
    raw = yaml.safe_load(TOY_YAML)
    for k in ("pk.ctr.c0", "pk.ctr.c2", "pk.ctr.c4"):
        raw["marginalized"][k].pop("ctr_rotation", None)
    p = tmp_path / "toyless_no_c1.yaml"
    p.write_text(yaml.safe_dump(raw))
    spec = load_desi_prior_spec(p)
    split = _split_no_c1()
    sigma8_ref, s8_fn, aap_fn = _toy_rescaling()
    fns = make_desi_prior_fns(
        spec, split=split, knl_bins=KNL_BINS, sigma8_bins_fn=s8_fn,
        a_ap_bins_fn=aap_fn, sigma8_ref_bins=sigma8_ref,
        lin_keys=LIN_KEYS_NO_C1)
    return spec, split, fns


def test_sampled_marginal_prior_c1_R_division(toy_spec_path):
    """The sampled c1 prior term evaluates N(0, (row.sigma / R_b)^2) with
    R_b = a_ap * a_amp (the c1 marginalized row's A_AP*A_amp rescale). At
    fiducial R=1 -> width 1.0125; off-fiducial the width divides by R."""
    spec = load_desi_prior_spec(toy_spec_path)
    split = _split_no_c1()
    c1_pos = _c1_nl_positions(split)
    sigma8_ref, s8_fn, aap_fn = _toy_rescaling()
    _, _, log_prior_nl = make_desi_prior_fns(
        spec, split=split, knl_bins=KNL_BINS, sigma8_bins_fn=s8_fn,
        a_ap_bins_fn=aap_fn, sigma8_ref_bins=sigma8_ref, f_bins=TOY_F_FID,
        lin_keys=LIN_KEYS_NO_C1,
        sampled_marginal_priors=[(_C1_KEY, c1_pos)])
    sig_c1 = spec.marginalized[_C1_KEY].sigma          # 1.0125 (mapped)
    assert sig_c1 == pytest.approx(1.0125)
    theta0 = jnp.zeros(split.n_nl)
    w = 0.5
    # Fiducial (R = 1): delta == -0.5*(w/1.0125)^2 (the -log(width) offset cancels).
    theta_c1 = theta0.at[c1_pos[0]].set(w)
    d = float(log_prior_nl(theta_c1) - float(log_prior_nl(theta0)))
    assert d == pytest.approx(-0.5 * (w / sig_c1) ** 2)
    # Off-fiducial: theta[1]=0.5 -> a_ap=1.1, sigma8 unchanged -> a_amp=1, R=1.1.
    theta_off = theta0.at[1].set(0.5)
    theta_off_c1 = theta_off.at[c1_pos[0]].set(w)
    d_off = float(log_prior_nl(theta_off_c1)) - float(log_prior_nl(theta_off))
    R = 1.1
    assert d_off == pytest.approx(-0.5 * (w / (sig_c1 / R)) ** 2)


def test_sampled_marginal_prior_absolute_log_normalization(toy_spec_path):
    """ABSOLUTE value of log_prior_nl against the closed form -- the only test
    that pins the ``- log(width) - 0.5*log(2pi)`` normalization.

    Every other assertion on this function takes a DIFFERENCE at fixed non-c1
    coordinates, where that term cancels identically. But ``width_b =
    sigma * f_knl_b / R_b(theta_NL)`` is theta-dependent, so ``-log(width_b)``
    contributes ``+ sum_b log R_b(theta_NL)``: a LIVE gradient in the cosmology
    directions, and precisely the term that must mirror the marginalized path's
    analytic ``-0.5 logdet Sigma_p``. Drop it and the sampled-c1 posterior tilts
    in cosmology relative to the marginalized one -- silently corrupting the
    Tier-3 comparison. Evaluated off-fiducial in BOTH rescaling directions
    (theta[0] -> sigma8/A_amp, theta[1] -> a_ap) so R != 1 for every block.
    """
    spec = load_desi_prior_spec(toy_spec_path)
    split = _split_no_c1()
    c1_pos = _c1_nl_positions(split)
    sigma8_ref, s8_fn, aap_fn = _toy_rescaling()
    _, _, log_prior_nl = make_desi_prior_fns(
        spec, split=split, knl_bins=KNL_BINS, sigma8_bins_fn=s8_fn,
        a_ap_bins_fn=aap_fn, sigma8_ref_bins=sigma8_ref, f_bins=TOY_F_FID,
        lin_keys=LIN_KEYS_NO_C1,
        sampled_marginal_priors=[(_C1_KEY, c1_pos)])

    theta = jnp.zeros(split.n_nl).at[0].set(0.3).at[1].set(-0.4)
    c1_vals = (0.35, -0.2)
    for p, v in zip(c1_pos, c1_vals):
        theta = theta.at[p].set(v)
    b2_pos = [int(p) + 1 for p in split.nl_b1_pos]
    bG2_pos = [int(p) + 2 for p in split.nl_b1_pos]
    b2_vals, bG2_vals = (0.6, -0.9), (0.15, 0.4)
    for p, v in zip(b2_pos, b2_vals):
        theta = theta.at[p].set(v)
    for p, v in zip(bG2_pos, bG2_vals):
        theta = theta.at[p].set(v)

    # Closed form, built independently of the implementation.
    log2pi = np.log(2.0 * np.pi)
    s8 = np.asarray(sigma8_ref) * (1.0 + 0.1 * 0.3)          # (n_bins,)
    a_ap = 1.0 + 0.2 * (-0.4)
    a_amp = s8 ** 2 / np.asarray(sigma8_ref) ** 2
    expected = 0.0
    for vals in (b2_vals, bG2_vals):                          # sigma8_sq rescale
        w = 5.0 / s8 ** 2
        expected += float(np.sum(-0.5 * (np.asarray(vals) / w) ** 2
                                 - np.log(w) - 0.5 * log2pi))
    R = a_ap * a_amp                                          # c1: A_AP*A_amp
    w_c1 = 1.0125 / R
    expected += float(np.sum(-0.5 * (np.asarray(c1_vals) / w_c1) ** 2
                             - np.log(w_c1) - 0.5 * log2pi))

    assert float(log_prior_nl(theta)) == pytest.approx(expected, rel=1e-12)

    # Non-vacuity: the normalization is a LIVE cosmology gradient. Moving only
    # the a_ap direction (theta[1]) with all c1 slots at ZERO -- where the
    # quadratic term vanishes identically -- still changes log_prior_nl by
    # exactly n_bins*log(R), the log-det the marginalized path also carries.
    theta_z = jnp.zeros(split.n_nl)
    d = float(log_prior_nl(theta_z.at[1].set(0.5))) - float(log_prior_nl(theta_z))
    assert d == pytest.approx(N_BINS * np.log(1.1))
    assert abs(d) > 0.1                                       # not a no-op


def test_sampled_marginal_prior_nonzero_mean_uses_mean_over_R(toy_spec_path):
    """A nonzero-mean sampled row is centred at ``mean / R_b``, matching the
    marginalized path (``_per_bin_arrays`` returns ``mean / R``). Using the raw
    mean would describe a prior on no consistent variable at all."""
    spec = load_desi_prior_spec(toy_spec_path)
    bshot = ("bk", "stoch", "B_shot")
    assert spec.marginalized[bshot].mean == 1.0          # nonzero, rescale A_AP
    lin_keys = tuple(k for k in LIN_SURVEY_KEYS if k != bshot)
    split = split_marginal_indices(
        n_cosmo_params=N_COSMO, survey_keys=SURVEY_KEYS_TOY, n_bins=N_BINS,
        lin_survey_keys=lin_keys)
    pos = _nl_positions(split, bshot)
    sigma8_ref, s8_fn, aap_fn = _toy_rescaling()
    _, _, log_prior_nl = make_desi_prior_fns(
        spec, split=split, knl_bins=KNL_BINS, sigma8_bins_fn=s8_fn,
        a_ap_bins_fn=aap_fn, sigma8_ref_bins=sigma8_ref, f_bins=TOY_F_FID,
        lin_keys=lin_keys, sampled_marginal_priors=[(bshot, pos)])

    # theta[1] = 0.5 -> a_ap = 1.1 = R (B_shot rescale is A_AP).
    R = 1.1
    theta = jnp.zeros(split.n_nl).at[1].set(0.5)
    at_peak = theta
    for p in pos:
        at_peak = at_peak.at[p].set(1.0 / R)
    g = jax.grad(log_prior_nl)(at_peak)
    for p in pos:
        assert float(g[p]) == pytest.approx(0.0, abs=1e-12)
    # The RAW mean is NOT the peak (would be, if the /R were missing).
    at_raw = theta
    for p in pos:
        at_raw = at_raw.at[p].set(1.0)
    g_raw = jax.grad(log_prior_nl)(at_raw)
    for p in pos:
        assert abs(float(g_raw[p])) > 0.1


def test_lin_keys_inconsistent_with_split_raises(toy_spec_path):
    """An equal-length but DIFFERENT lin_keys list would silently mis-assign
    every prior row against the templates laid out per ``split.lin_idx``."""
    spec = load_desi_prior_spec(toy_spec_path)
    split = _split_no_c1()                       # split built with c1 dropped
    wrong = tuple(k for k in LIN_SURVEY_KEYS
                  if k != ("pk", "stoch", "a0"))  # same length 10, a0 dropped
    assert len(wrong) == len(LIN_KEYS_NO_C1)
    sigma8_ref, s8_fn, aap_fn = _toy_rescaling()
    with pytest.raises(ValueError, match="split"):
        make_desi_prior_fns(
            spec, split=split, knl_bins=KNL_BINS, sigma8_bins_fn=s8_fn,
            a_ap_bins_fn=aap_fn, sigma8_ref_bins=sigma8_ref, f_bins=TOY_F_FID,
            lin_keys=wrong)


def test_ctr_trio_non_contiguous_raises(toy_spec_path):
    """Cov-mode writes the rotated 3x3 block at ``lin_keys[c0:c0+3]``; a
    reordering that separates the trio must be rejected, not silently rotated
    onto the wrong slots."""
    spec = load_desi_prior_spec(toy_spec_path)
    cfog = ("pk", "ctr", "cfog")
    permuted = list(k for k in LIN_SURVEY_KEYS if k != cfog)
    permuted.insert(permuted.index(("pk", "ctr", "c2")), cfog)  # c0, cfog, c2...
    permuted = tuple(permuted)
    split = split_marginal_indices(
        n_cosmo_params=N_COSMO, survey_keys=SURVEY_KEYS_TOY, n_bins=N_BINS,
        lin_survey_keys=permuted)
    sigma8_ref, s8_fn, aap_fn = _toy_rescaling()
    with pytest.raises(ValueError, match="contiguous"):
        make_desi_prior_fns(
            spec, split=split, knl_bins=KNL_BINS, sigma8_bins_fn=s8_fn,
            a_ap_bins_fn=aap_fn, sigma8_ref_bins=sigma8_ref, f_bins=TOY_F_FID,
            lin_keys=permuted)


def test_sampled_marginal_prior_positions_validated(toy_spec_path):
    """JAX gathers CLAMP out-of-range indices, so an unvalidated position would
    put the prior on the wrong parameter silently; and a key still present in
    lin_keys has no theta_NL slot at all."""
    spec = load_desi_prior_spec(toy_spec_path)
    split = _split_no_c1()
    c1_pos = _c1_nl_positions(split)
    sigma8_ref, s8_fn, aap_fn = _toy_rescaling()
    common = dict(split=split, knl_bins=KNL_BINS, sigma8_bins_fn=s8_fn,
                  a_ap_bins_fn=aap_fn, sigma8_ref_bins=sigma8_ref,
                  f_bins=TOY_F_FID, lin_keys=LIN_KEYS_NO_C1)

    bad = list(c1_pos)
    bad[-1] = split.n_nl                       # one past the end -> clamps
    with pytest.raises(ValueError, match="out of range"):
        make_desi_prior_fns(spec, sampled_marginal_priors=[(_C1_KEY, bad)],
                            **common)
    neg = list(c1_pos)
    neg[0] = -1
    with pytest.raises(ValueError, match="out of range"):
        make_desi_prior_fns(spec, sampled_marginal_priors=[(_C1_KEY, neg)],
                            **common)
    # A key that is STILL marginalized cannot also be sampled.
    common_full = dict(common, lin_keys=LIN_SURVEY_KEYS,
                       split=split_marginal_indices(
                           n_cosmo_params=N_COSMO, survey_keys=SURVEY_KEYS_TOY,
                           n_bins=N_BINS))
    with pytest.raises(ValueError, match="lin_keys"):
        make_desi_prior_fns(spec, sampled_marginal_priors=[(_C1_KEY, c1_pos)],
                            **common_full)


def test_sampled_marginal_prior_jit_and_grad_safe(toy_spec_path):
    spec = load_desi_prior_spec(toy_spec_path)
    split = _split_no_c1()
    c1_pos = _c1_nl_positions(split)
    sigma8_ref, s8_fn, aap_fn = _toy_rescaling()
    _, _, log_prior_nl = make_desi_prior_fns(
        spec, split=split, knl_bins=KNL_BINS, sigma8_bins_fn=s8_fn,
        a_ap_bins_fn=aap_fn, sigma8_ref_bins=sigma8_ref, f_bins=TOY_F_FID,
        lin_keys=LIN_KEYS_NO_C1,
        sampled_marginal_priors=[(_C1_KEY, c1_pos)])
    theta = jnp.full(split.n_nl, 0.1)
    assert float(jax.jit(log_prior_nl)(theta)) == pytest.approx(
        float(log_prior_nl(theta)))
    g = jax.grad(log_prior_nl)(theta)
    assert jnp.all(jnp.isfinite(g))
    # c1 positions have a LIVE gradient (mean 0, x=0.1 -> grad = -x/w^2 != 0).
    for p in c1_pos:
        assert abs(float(g[p])) > 1e-3


# ---------------------------------------------------------------------------
# b1 sigma8 measure (F1) + phase gate (F2): spec/loader layer
# ---------------------------------------------------------------------------

def test_b1_measure_defaults_raw(toy_spec_path):
    spec = load_desi_prior_spec(toy_spec_path)
    assert spec.sampled["b1"].measure == "raw"
    assert spec.sampled["b1"].paper_lower is None


def test_b1_measure_b1sigma8_loads_with_bounds(tmp_path):
    def mutate(raw):
        raw["sampled"]["b1"] = {"kind": "flat", "measure": "b1sigma8",
                                "paper_lower": 0.0, "paper_upper": 3.0,
                                "paper_variable": "b1*sigma8(z)"}
    spec = load_desi_prior_spec(_mutated_spec_path(tmp_path, mutate))
    assert spec.sampled["b1"].measure == "b1sigma8"
    assert spec.sampled["b1"].paper_upper == 3.0


def test_b1_measure_bad_token_raises(tmp_path):
    def mutate(raw):
        raw["sampled"]["b1"] = {"kind": "flat", "measure": "b1_sigma_8"}
    with pytest.raises(SpecValidationError, match="measure"):
        load_desi_prior_spec(_mutated_spec_path(tmp_path, mutate))


def test_b1sigma8_requires_both_bounds(tmp_path):
    def mutate(raw):
        raw["sampled"]["b1"] = {"kind": "flat", "measure": "b1sigma8",
                                "paper_lower": 0.0}
    with pytest.raises(SpecValidationError, match="paper_upper"):
        load_desi_prior_spec(_mutated_spec_path(tmp_path, mutate))


def test_measure_on_non_b1_row_raises(tmp_path):
    def mutate(raw):
        raw["sampled"]["b2"]["measure"] = "b1sigma8"
    with pytest.raises(SpecValidationError, match="only the b1 row"):
        load_desi_prior_spec(_mutated_spec_path(tmp_path, mutate))


def test_phase_gate_blocks_raw_measure(toy_spec_path):
    """F2: real-data / nuLCDM phases refuse the raw measure (CONTEXT.md)."""
    for phase in ("real_data", "nulcdm"):
        with pytest.raises(SpecValidationError, match="measure"):
            load_desi_prior_spec(toy_spec_path, phase=phase)
    load_desi_prior_spec(toy_spec_path, phase="forecast")   # default path OK


def test_phase_gate_passes_with_b1sigma8(tmp_path):
    def mutate(raw):
        raw["sampled"]["b1"] = {"kind": "flat", "measure": "b1sigma8",
                                "paper_lower": 0.0, "paper_upper": 3.0}
    p = _mutated_spec_path(tmp_path, mutate)
    load_desi_prior_spec(p, phase="real_data")               # no raise


def test_unknown_phase_raises(toy_spec_path):
    with pytest.raises(SpecValidationError, match="phase"):
        load_desi_prior_spec(toy_spec_path, phase="production")


def test_real_spec_b1_row_and_deviation_note():
    spec = load_desi_prior_spec()
    b1 = spec.sampled["b1"]
    assert b1.measure == "raw" and b1.paper_lower == 0.0 and b1.paper_upper == 3.0
    devs = " ".join(str(d) for d in spec.metadata.get("deviations", []))
    assert "b1" in devs and "measure" in devs and "sigma8" in devs.lower()


# ---------------------------------------------------------------------------
# b1 sigma8 measure: runtime Jacobian + bounds
# ---------------------------------------------------------------------------

def _fns_for_measure(tmp_path, measure):
    def mutate(raw):
        if measure == "b1sigma8":
            raw["sampled"]["b1"] = {"kind": "flat", "measure": "b1sigma8",
                                    "paper_lower": 0.0, "paper_upper": 3.0}
    spec = load_desi_prior_spec(_mutated_spec_path(tmp_path, mutate))
    split = split_marginal_indices(
        n_cosmo_params=N_COSMO, survey_keys=SURVEY_KEYS_TOY, n_bins=N_BINS)
    sigma8_ref = jnp.array([0.6, 0.5])
    s8_fn = lambda t: sigma8_ref * (1.0 + 0.1 * t[0])
    a_ap_fn = lambda t: jnp.ones(N_BINS) * (1.0 + 0.2 * t[1])
    fns = make_desi_prior_fns(
        spec, split=split, knl_bins=KNL_BINS, sigma8_bins_fn=s8_fn,
        a_ap_bins_fn=a_ap_fn, sigma8_ref_bins=sigma8_ref, f_bins=TOY_F_FID)
    return spec, split, s8_fn, fns


def test_b1sigma8_jacobian_pointwise_identity(tmp_path):
    """log_prior_ON - log_prior_OFF == sum_b log sigma8(z_b; theta), exactly."""
    _, split, s8_fn, (_, _, lp_off) = _fns_for_measure(tmp_path, "raw")
    _, _, _, (_, _, lp_on) = _fns_for_measure(tmp_path, "b1sigma8")
    rng = np.random.default_rng(20260804)
    for _ in range(8):
        theta = jnp.asarray(rng.normal(0.0, 0.3, size=split.n_nl))
        # keep b1*sigma8 inside [0, 3]: set b1 slots to ~1.5
        for p in split.nl_b1_pos:
            theta = theta.at[p].set(1.5)
        expected = float(jnp.sum(jnp.log(s8_fn(theta))))
        got = float(lp_on(theta)) - float(lp_off(theta))
        assert got == pytest.approx(expected, abs=1e-12)


def test_b1sigma8_bounds_give_minus_inf(tmp_path):
    _, split, _, (_, _, lp_on) = _fns_for_measure(tmp_path, "b1sigma8")
    theta = jnp.zeros(split.n_nl)
    ok = theta.at[split.nl_b1_pos[0]].set(1.0)          # y = 0.6 in [0,3]
    bad_hi = theta.at[split.nl_b1_pos[0]].set(6.0)      # y = 3.6 > 3
    bad_lo = theta.at[split.nl_b1_pos[0]].set(-0.5)     # y < 0
    assert np.isfinite(float(lp_on(ok)))
    assert float(lp_on(bad_hi)) == -np.inf
    assert float(lp_on(bad_lo)) == -np.inf
    # Discriminates bounds-on-y from bounds-on-raw-b1: with the toy s8 (0.6 at
    # bin 0, theta=0), b1 = 4.0 gives y = 2.4 in [0,3] but raw 4.0 > 3, so a
    # bounds-on-raw-b1 mutant would (wrongly) return -inf here.
    assert np.isfinite(float(lp_on(theta.at[split.nl_b1_pos[0]].set(4.0))))


def test_b1sigma8_gradient_slope(tmp_path):
    """d(Jacobian)/d theta0 = sum_b d log s8/d theta0 = sum_b 0.1/(1+0.1 t0)."""
    _, split, _, (_, _, lp_off) = _fns_for_measure(tmp_path, "raw")
    _, _, _, (_, _, lp_on) = _fns_for_measure(tmp_path, "b1sigma8")
    theta = jnp.zeros(split.n_nl)
    for p in split.nl_b1_pos:
        theta = theta.at[p].set(1.5)
    diff = lambda t: lp_on(t) - lp_off(t)
    g = jax.grad(diff)(theta)
    assert float(g[0]) == pytest.approx(N_BINS * 0.1, rel=1e-10)
    assert float(g[split.nl_b1_pos[0]]) == pytest.approx(0.0, abs=1e-12)


def test_raw_measure_bitwise_unchanged(tmp_path):
    """Default path must be BIT-identical to the pre-change behavior."""
    _, split, _, (mu_a, sig_a, lp_a) = _fns_for_measure(tmp_path, "raw")
    theta = jnp.full(split.n_nl, 0.2)
    # raw measure adds no term and no bounds:
    assert np.isfinite(float(lp_a(theta.at[split.nl_b1_pos[0]].set(50.0))))


# ---------------------------------------------------------------------------
# b1 sigma8 measure: post-hoc reweighting helper (option D)
# ---------------------------------------------------------------------------
from jaxptpolypol.desi_priors import b1sigma8_log_weights
from jaxptpolypol.marginal_taylor import reweighted_moments


def test_log_weights_match_pointwise_jacobian(tmp_path):
    _, split, s8_fn, _ = _fns_for_measure(tmp_path, "raw")
    rng = np.random.default_rng(4)
    samples = jnp.asarray(rng.normal(0.0, 0.3, size=(16, split.n_nl)))
    lw = b1sigma8_log_weights(samples, s8_fn)
    assert lw.shape == (16,)
    for i in range(16):
        assert float(lw[i]) == pytest.approx(
            float(jnp.sum(jnp.log(s8_fn(samples[i])))), abs=1e-12)


def test_log_weights_bounds(tmp_path):
    _, split, s8_fn, _ = _fns_for_measure(tmp_path, "raw")
    samples = jnp.zeros((2, split.n_nl))
    samples = samples.at[1, split.nl_b1_pos[0]].set(10.0)   # y = 6 > 3
    lw = b1sigma8_log_weights(samples, s8_fn,
                              b1_pos=split.nl_b1_pos, lower=0.0, upper=3.0)
    assert np.isfinite(float(lw[0])) and float(lw[1]) == -np.inf


def test_reweighting_gaussian_tilt_analytic_oracle():
    """Reweighting N(0,1) draws by exp(a*x) must give N(a,1): the exact
    finite-sample check is that reweighted moments match the ANALYTIC
    importance estimate, and at n=200k they must be within MC error of (a, 1)."""
    rng = np.random.default_rng(20260804)
    n, a = 200_000, 0.35
    x = rng.normal(0.0, 1.0, size=(n, 1))
    lw = a * x[:, 0]
    w = np.exp(lw - lw.max()); w /= w.sum()
    mean, std = reweighted_moments(x, w)
    se = 1.0 / np.sqrt(n * float((w.sum() ** 2) / (w ** 2).sum()) / n)  # ~1/sqrt(ESS)
    ess = 1.0 / np.sum(w ** 2)
    assert ess / n > 0.85                       # exp(-a^2) = 0.885 predicted
    assert float(mean[0]) == pytest.approx(a, abs=4.0 / np.sqrt(ess))
    assert float(std[0]) == pytest.approx(1.0, abs=4.0 / np.sqrt(ess))

