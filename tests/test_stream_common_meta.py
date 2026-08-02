"""META-guard semantics of ``example/mcmc/scripts/stream_common.py``.

The Stream drivers load a template npz + a whitening npz through
:func:`stream_common.load_templates_and_whitening`. Its DEFAULT expectations
NAME the two identifiers that matter -- ``theory_config_hash`` (templates only)
and ``c1_treatment`` (both) -- because a key the expectation does not name is
not checked at all, which would leave the stale-template guard unable to fire on
the very thing it exists to catch.

The rule these tests pin is ENFORCE-IF-PRESENT:

* stored value differs from expected -> hard failure (config drift, or a
  marginalized/sampled mix-up);
* stored stamp LACKS the key -> warning, loads anyway (the committed on-disk
  caches predate these keys);
* stored stamp carries keys the expectation does not name -> warning, loads
  (a rebuilt cache stamps more than any single consumer verifies).

Plus one anti-drift test: ``build_taylor_templates_lcdm.py`` must IMPORT the
shared production constants from ``stream_common`` instead of re-declaring them,
or the hash it stamps and the hash the loaders expect would describe two
different configurations.
"""
import ast
import hashlib
import json
import pathlib
import sys
import warnings

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

_SCRIPTS = (pathlib.Path(__file__).resolve().parents[1]
            / "example" / "mcmc" / "scripts")
sys.path.insert(0, str(_SCRIPTS))

import stream_common  # noqa: E402
from jaxptpolypol.marginal_taylor import (  # noqa: E402
    TaylorTemplates, save_taylor_templates,
)

#: What build_taylor_templates_lcdm.py stamps in its default (marginalized)
#: mode -- it calls these very functions, so this IS the producer's stamp.
TEMPLATE_META_SHAPE = stream_common.template_meta_for("marginalized")
WHITENING_META_SHAPE = {
    **stream_common.meta_for("marginalized"),
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


def test_production_stamp_loads_with_no_warnings(tmp_path):
    """The exact stamps a fresh build writes match the default expectations
    key-for-key -- the guard is silent on a correct cache, so every warning it
    does emit is signal."""
    tpath, wpath = _write_pair(tmp_path, TEMPLATE_META_SHAPE,
                               stream_common.meta_for("marginalized"))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        tt, _ = stream_common.load_templates_and_whitening(tpath, wpath)
    assert not rec, [str(w.message) for w in rec]
    assert tt.order2_m0 is False


def test_rebuilt_cache_extra_keys_still_load(tmp_path):
    """A cache stamping MORE identifiers than the consumer verifies still loads;
    the extras are informational. This must not regress -- otherwise the guard
    would forbid the rebuild it tells you to perform."""
    tpath, wpath = _write_pair(
        tmp_path, {**TEMPLATE_META_SHAPE, "builder_git_sha": "abc123"},
        WHITENING_META_SHAPE)
    with pytest.warns(UserWarning, match="builder_git_sha"):
        tt, wz = stream_common.load_templates_and_whitening(tpath, wpath)
    assert tt.order2_m0 is False
    assert json.loads(str(wz["meta"].item()))["prior_spec"] == \
        "eft_eq12_2405_02252"


def test_legacy_stamp_predating_the_identifiers_still_loads(tmp_path):
    """COMPATIBILITY: the committed on-disk caches stamp only the plain 11-key
    META -- no theory_config_hash, no c1_treatment. Naming those keys by default
    must warn, never error, or the tracked artifacts become unloadable."""
    tpath, wpath = _write_pair(tmp_path, dict(stream_common.META),
                               dict(stream_common.META))
    with pytest.warns(UserWarning, match="theory_config_hash"):
        tt, _ = stream_common.load_templates_and_whitening(tpath, wpath)
    assert tt.order2_m0 is False


def test_theory_config_hash_drift_hard_fails(tmp_path):
    """THE point of the guard: a stamp differing ONLY in theory_config_hash --
    i.e. templates built for a different theory config -- is a hard failure."""
    drifted = {**TEMPLATE_META_SHAPE, "theory_config_hash": "d" * 64}
    tpath, wpath = _write_pair(tmp_path, drifted, WHITENING_META_SHAPE)
    with pytest.raises(ValueError, match="theory_config_hash"):
        stream_common.load_templates_and_whitening(tpath, wpath)


def test_c1_treatment_mixup_hard_fails(tmp_path):
    """A sampled-c1 cache loaded as marginalized (or vice versa) hard-fails on
    both npz rather than silently producing a wrong posterior."""
    tpath, wpath = _write_pair(tmp_path,
                               stream_common.template_meta_for("sampled"),
                               stream_common.meta_for("sampled"))
    with pytest.raises(ValueError, match="c1_treatment"):
        stream_common.load_templates_and_whitening(tpath, wpath)
    # ... and the whitening side alone (templates matching) exits.
    tpath2, wpath2 = _write_pair(
        tmp_path, stream_common.template_meta_for("marginalized"),
        stream_common.meta_for("sampled"))
    with pytest.raises(SystemExit, match="c1_treatment"):
        stream_common.load_templates_and_whitening(tpath2, wpath2)


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


# ---------------------------------------------------------------------------
# Anti-drift: the producer must read the SAME constants the consumers expect.
# ---------------------------------------------------------------------------

#: Production constants that feed THEORY_CONFIG_HASH and/or the theory build.
#: A local copy of any of these in the build script would make the hash guard
#: compare two configurations that can silently diverge.
SHARED_CONST_NAMES = frozenset({
    "FIDUCIAL", "MNU_FIXED", "z_bins", "V_bins", "knl_bins", "n_bar", "n_zbins",
    "K_PK_MIN", "K_PK_MAX", "N_K", "K_BK_MIN", "K_BK_MAX", "K_NL_RSD",
    "NUM_MU", "NUM_PHI", "N_GL", "BACKGROUND_MODE", "PFS_EMULATOR", "META",
})


def _build_script_tree():
    src = (_SCRIPTS / "build_taylor_templates_lcdm.py").read_text()
    return ast.parse(src)


def test_build_script_imports_shared_constants_from_stream_common():
    tree = _build_script_tree()
    imported = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "stream_common"
        for alias in node.names
    }
    missing = sorted(SHARED_CONST_NAMES - imported)
    assert not missing, (
        "build_taylor_templates_lcdm.py must import these from stream_common "
        f"(single source of truth for the hashed config): {missing}")


def test_build_script_does_not_redeclare_shared_constants():
    tree = _build_script_tree()
    assigned = set()
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for tgt in targets:
            for sub in ast.walk(tgt):
                if isinstance(sub, ast.Name):
                    assigned.add(sub.id)
    clashes = sorted(SHARED_CONST_NAMES & assigned)
    assert not clashes, (
        "build_taylor_templates_lcdm.py re-declares shared production "
        f"constants {clashes}; they must come from stream_common only, or the "
        "theory_config_hash it stamps can drift from the one consumers expect.")


def test_theory_config_hash_is_a_sha256_of_the_live_constants():
    """The hash is derived, not hard-coded: perturbing a constant changes it."""
    expect = hashlib.sha256(repr((
        stream_common.V_bins, stream_common.n_bar, stream_common.knl_bins,
        stream_common.z_bins,
        (stream_common.K_PK_MIN, stream_common.K_PK_MAX, stream_common.N_K,
         stream_common.K_BK_MIN, stream_common.K_BK_MAX),
        stream_common.K_NL_RSD)).encode()).hexdigest()
    assert stream_common.THEORY_CONFIG_HASH == expect
    perturbed = hashlib.sha256(repr((
        stream_common.V_bins, stream_common.n_bar, stream_common.knl_bins,
        stream_common.z_bins,
        (stream_common.K_PK_MIN, 0.25, stream_common.N_K,
         stream_common.K_BK_MIN, stream_common.K_BK_MAX),
        stream_common.K_NL_RSD)).encode()).hexdigest()
    assert perturbed != stream_common.THEORY_CONFIG_HASH
