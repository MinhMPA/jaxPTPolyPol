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
