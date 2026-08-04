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


def _hash_config(cfg):
    """The live hashing formula (``repr(sorted(...items()))`` sha256) applied to a
    config-dict COPY, so a test can perturb an entry without touching the module's
    own ``_THEORY_CONFIG``."""
    return hashlib.sha256(repr(sorted(cfg.items())).encode()).hexdigest()


def test_theory_config_hash_is_a_sha256_of_the_live_constants():
    """The hash is derived, not hard-coded: it is the sha256 of the live
    ``_THEORY_CONFIG`` dict (sorted-items repr), and perturbing a hashed entry
    changes it."""
    assert (_hash_config(dict(stream_common._THEORY_CONFIG))
            == stream_common.THEORY_CONFIG_HASH)
    # Perturb a k-grid entry in a COPY (never mutate the module's own dict).
    perturbed = dict(stream_common._THEORY_CONFIG)
    perturbed["k_grid"] = (stream_common.K_PK_MIN, 0.25, stream_common.N_K,
                           stream_common.K_BK_MIN, stream_common.K_BK_MAX)
    assert _hash_config(perturbed) != stream_common.THEORY_CONFIG_HASH


# ---------------------------------------------------------------------------
# Extended coverage: cosmo basis, emulator, fiducials (the nuLCDM-disambiguation
# entries) and the old-hash transition. Additions for the theory-config-hash
# extension (nuLCDM prereq).
# ---------------------------------------------------------------------------


def test_hash_covers_fiducial_and_emulator():
    """The extended hash covers the FIDUCIAL cosmology values and the EMULATOR
    path: perturbing either (in a copy of the live config) changes the hash."""
    base = _hash_config(dict(stream_common._THEORY_CONFIG))
    assert base == stream_common.THEORY_CONFIG_HASH        # reconstruction faithful
    # (i) fiducial: bump h by 1e-3
    cfg_fid = dict(stream_common._THEORY_CONFIG)
    fid = dict(cfg_fid["fiducial"])
    fid["h"] = fid["h"] + 1e-3
    cfg_fid["fiducial"] = tuple(sorted(fid.items()))
    assert _hash_config(cfg_fid) != base
    # (ii) emulator: an mnu network at a different path
    cfg_emu = dict(stream_common._THEORY_CONFIG)
    cfg_emu["emulator"] = cfg_emu["emulator"].replace("lcdm", "mnu")
    assert cfg_emu["emulator"] != stream_common._THEORY_CONFIG["emulator"]
    assert _hash_config(cfg_emu) != base


def test_hash_covers_cosmo_basis_mnu():
    """LCDM vs nuLCDM: two configs differing ONLY in whether ``'mnu'`` is in the
    sampled cosmo basis hash differently -- the whole point of the extension, so a
    nuLCDM template cache can never be confused with the LCDM one."""
    lcdm = dict(stream_common._THEORY_CONFIG)
    shared, fixed, mnu = lcdm["cosmo_basis"]
    assert "mnu" not in shared
    nulcdm = dict(stream_common._THEORY_CONFIG)
    nulcdm["cosmo_basis"] = (shared + ("mnu",), fixed, mnu)
    assert _hash_config(lcdm) != _hash_config(nulcdm)


def test_old_hash_cache_hard_fails_after_extension(tmp_path):
    """TRANSITION (enforce-if-present): a template cache stamped with the
    PRE-extension theory_config_hash (the old 6-tuple sha256) is rejected with a
    hard ``ValueError`` under the new richer hash -- a genuinely stale cache must
    not silently load. This bites ONLY a cache that actually carried the old hash;
    the committed on-disk caches predate the key entirely and take the
    warn-and-load path (see
    ``test_legacy_stamp_predating_the_identifiers_still_loads``), which is why the
    smoke tripwire is unaffected by the hash-value change."""
    old_hash = hashlib.sha256(repr((
        stream_common.V_bins, stream_common.n_bar, stream_common.knl_bins,
        stream_common.z_bins,
        (stream_common.K_PK_MIN, stream_common.K_PK_MAX, stream_common.N_K,
         stream_common.K_BK_MIN, stream_common.K_BK_MAX),
        stream_common.K_NL_RSD)).encode()).hexdigest()
    assert old_hash != stream_common.THEORY_CONFIG_HASH    # extension changed it
    stamped = {**TEMPLATE_META_SHAPE, "theory_config_hash": old_hash}
    tpath, wpath = _write_pair(tmp_path, stamped, WHITENING_META_SHAPE)
    with pytest.raises(ValueError, match="theory_config_hash"):
        stream_common.load_templates_and_whitening(tpath, wpath)


# ---------------------------------------------------------------------------
# nuLCDM config block (Task 1): hash disambiguation, per-cosmology meta stamp
# (LCDM default byte-unchanged), and the hash-input key-set RIDER.
# ---------------------------------------------------------------------------

#: The exact hash-input key set. LOCKED so silently dropping (or renaming) a
#: hashed input fails the suite -- the RIDER from the prereq review. The LCDM and
#: nuLCDM configs share this set by construction (nuLCDM is ``{**_THEORY_CONFIG,
#: ...}`` overriding three values), which is what makes them guard-distinguishable
#: only through the three cosmology-dependent VALUES, never a missing key.
_THEORY_CONFIG_KEYS = frozenset({
    "V_bins", "n_bar", "knl_bins", "z_bins", "k_grid", "k_nl_rsd",
    "cosmo_basis", "emulator", "fiducial", "n_gl", "num_mu", "num_phi",
    "background_mode",
})


def test_theory_config_key_set_is_locked():
    assert set(stream_common._THEORY_CONFIG.keys()) == _THEORY_CONFIG_KEYS


def test_nulcdm_theory_config_key_set_mirrors_lcdm():
    assert set(stream_common.NULCDM_THEORY_CONFIG.keys()) == _THEORY_CONFIG_KEYS
    # ... and only the three cosmology-dependent entries differ in VALUE.
    diff = {k for k in _THEORY_CONFIG_KEYS
            if stream_common._THEORY_CONFIG[k] != stream_common.NULCDM_THEORY_CONFIG[k]}
    assert diff == {"cosmo_basis", "emulator", "fiducial"}


def test_nulcdm_hash_differs_from_lcdm_and_is_live():
    """The nuLCDM template cache can never be confused with the LCDM one, and the
    nuLCDM hash is the live sha256 of NULCDM_THEORY_CONFIG (not hard-coded)."""
    assert (stream_common.NULCDM_THEORY_CONFIG_HASH
            != stream_common.THEORY_CONFIG_HASH)
    assert (_hash_config(dict(stream_common.NULCDM_THEORY_CONFIG))
            == stream_common.NULCDM_THEORY_CONFIG_HASH)
    # The cosmo_basis gains 'mnu'; FIXED_COSMO is reused byte-for-byte.
    shared_nu, fixed_nu, mnu_nu = stream_common.NULCDM_THEORY_CONFIG["cosmo_basis"]
    assert "mnu" in shared_nu
    assert fixed_nu == stream_common.FIXED_COSMO == (5, 6, 7, 8)


def test_lcdm_meta_default_is_byte_identical():
    """REGRESSION: the LCDM default path is byte-identical to cosmology='lcdm'
    and carries NO ``cosmology`` key -- the production stamp/tripwire must not
    shift when the nuLCDM branch is added."""
    default_t = stream_common.template_meta_for("marginalized")
    lcdm_t = stream_common.template_meta_for("marginalized", cosmology="lcdm")
    assert default_t == lcdm_t
    assert "cosmology" not in default_t
    assert default_t["theory_config_hash"] == stream_common.THEORY_CONFIG_HASH
    # meta_for (whitening side) likewise unchanged by default.
    default_w = stream_common.meta_for("marginalized")
    assert default_w == stream_common.meta_for("marginalized", cosmology="lcdm")
    assert "cosmology" not in default_w
    assert default_w == {**stream_common.META, "c1_treatment": "marginalized"}


def test_nulcdm_meta_carries_hash_and_cosmology_key():
    """cosmology='nulcdm' stamps the nuLCDM hash and a cosmology key on both the
    templates side and the whitening side."""
    t = stream_common.template_meta_for("marginalized", cosmology="nulcdm")
    assert t["theory_config_hash"] == stream_common.NULCDM_THEORY_CONFIG_HASH
    assert t["cosmology"] == "nulcdm"
    w = stream_common.meta_for("sampled", cosmology="nulcdm")
    assert w["cosmology"] == "nulcdm"
    assert w["c1_treatment"] == "sampled"


def test_meta_rejects_unknown_cosmology():
    with pytest.raises(ValueError, match="cosmology"):
        stream_common.meta_for("marginalized", cosmology="wcdm")
    with pytest.raises(ValueError, match="cosmology"):
        stream_common.template_meta_for("marginalized", cosmology="wcdm")
