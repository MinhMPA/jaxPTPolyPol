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
