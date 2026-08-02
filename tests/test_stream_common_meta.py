"""META-guard semantics of ``example/mcmc/scripts/stream_common.py``.

The Stream drivers load a template npz + a whitening npz against ONE
``expect_meta``. The production builder
(``build_taylor_templates_lcdm.py``, default/marginalized mode) stamps a RICHER
meta than the plain 11-key ``META`` every consumer passes: six extra template
keys and three extra whitening keys (``c1_treatment``, ``prior_spec``,
``cosmo_priors``). These tests pin that a rebuilt-cache stamp still loads
(stored-only keys are informational) while a genuinely changed value on a key
the caller DID specify still hard-fails.
"""
import json
import pathlib
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]
                       / "example" / "mcmc" / "scripts"))

import stream_common  # noqa: E402
from jaxptpolypol.marginal_taylor import (  # noqa: E402
    TaylorTemplates, save_taylor_templates,
)

# Verbatim shapes of the two stamps build_taylor_templates_lcdm.py writes in its
# DEFAULT (marginalized) mode -- lines 243-253 of that script.
TEMPLATE_META_SHAPE = {
    **stream_common.META, "c1_treatment": "marginalized",
    "theory_config_hash": "0" * 64,
    "z_bins": str(stream_common.z_bins), "knl_bins": str(stream_common.knl_bins),
    "n_bar": str(stream_common.n_bar), "V_bins": "(5.9e+08,)",
}
WHITENING_META_SHAPE = {
    **stream_common.META, "c1_treatment": "marginalized",
    "prior_spec": "eft_eq12_2405_02252",
    "cosmo_priors": {"ombh2": 0.00055, "ns": 0.042},
}


def _toy_templates():
    return TaylorTemplates(
        theta0=jnp.zeros(2), bin_m00=(jnp.zeros(3),), bin_J=(jnp.zeros((3, 2)),),
        bin_H=(None,), bin_M0=(jnp.zeros((3, 2)),),
        bin_dM=(jnp.zeros((3, 2, 2)),), order2_m0=False, build_diagnostics={})


def _write_pair(tmp_path, template_meta, whitening_meta):
    tpath = tmp_path / "tt.npz"
    wpath = tmp_path / "wz.npz"
    save_taylor_templates(_toy_templates(), tpath, meta=template_meta)
    np.savez(wpath, meta=np.asarray(json.dumps(whitening_meta)))
    return tpath, wpath


def test_rebuilt_cache_stamp_loads_against_plain_meta(tmp_path):
    """A cache freshly rebuilt by build_taylor_templates_lcdm.py (richer stamps
    on BOTH npz) loads against the plain META the consumers pass, warning about
    the identifiers it did not check."""
    tpath, wpath = _write_pair(tmp_path, TEMPLATE_META_SHAPE,
                               WHITENING_META_SHAPE)
    with pytest.warns(UserWarning, match="theory_config_hash"):
        tt, wz = stream_common.load_templates_and_whitening(tpath, wpath)
    assert tt.order2_m0 is False
    assert json.loads(str(wz["meta"].item()))["prior_spec"] == \
        "eft_eq12_2405_02252"


def test_whitening_value_mismatch_still_exits(tmp_path):
    """A key the caller DID specify, with a differing stored value, still
    hard-fails on the whitening side (templates side matches here)."""
    bad_whitening = {**WHITENING_META_SHAPE, "n_bins": 3}
    tpath, wpath = _write_pair(tmp_path, TEMPLATE_META_SHAPE, bad_whitening)
    with pytest.raises(SystemExit) as exc:
        stream_common.load_templates_and_whitening(tpath, wpath)
    assert "n_bins" in str(exc.value)


def test_template_value_mismatch_still_raises(tmp_path):
    """Same on the templates side: a specified key with a differing value is
    still a stale-template ValueError."""
    bad_templates = {**TEMPLATE_META_SHAPE, "n_k": 99}
    tpath, wpath = _write_pair(tmp_path, bad_templates, WHITENING_META_SHAPE)
    with pytest.raises(ValueError, match="n_k"):
        stream_common.load_templates_and_whitening(tpath, wpath)


def test_legacy_stamp_still_loads_against_treatment_meta(tmp_path):
    """The other direction (unchanged): the on-disk PRE-rebuild artifacts stamp
    the plain META, so a caller that asks for ``meta_for('marginalized')`` gets
    the backward-compat warning on both npz, not an error."""
    tpath, wpath = _write_pair(tmp_path, dict(stream_common.META),
                               dict(stream_common.META))
    with pytest.warns(UserWarning, match="c1_treatment"):
        tt, _ = stream_common.load_templates_and_whitening(
            tpath, wpath, expect_meta=stream_common.meta_for("marginalized"))
    assert tt.order2_m0 is False
