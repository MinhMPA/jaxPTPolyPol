"""Unit tests for ``example/mcmc/scripts/cmb_gn_fisher.py``.

All DATA-FREE: every test drives the module with stub likelihood objects, so no
Planck ``.clik`` file, candl dataset or CMB emulator is touched. What is pinned
here is the machinery the real build depends on but which the build itself can
only exercise against ~2 GB of data:

* the whiteners really implement ``|W x|^2 == x^T C^-1 x``;
* ``gn_fisher`` really is ``J^T C^-1 J`` plus the prior curvature, and PSD;
* ``validate_gn_term`` really BITES on a perturbed model -- including under
  ``PYTHONOPTIMIZE=1``, which strips ``assert`` (the previous enforcement
  mechanism, and the reason this file exists);
* an unclassified term name raises instead of silently taking the
  observed-Hessian path;
* the shared-prior dedupe reproduces the analytic single-count matrix EXACTLY,
  including the packed index it subtracts at.
"""
import os
import pathlib
import subprocess
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

_SCRIPTS = (pathlib.Path(__file__).resolve().parents[1]
            / "example" / "mcmc" / "scripts")
sys.path.insert(0, str(_SCRIPTS))

import cmb_gn_fisher  # noqa: E402
from jaxptpolypol.cmb import CandlParameterLayout  # noqa: E402


# ---------------------------------------------------------------------------
# Stubs.
# ---------------------------------------------------------------------------

def _spd(n, seed):
    """A random symmetric positive-definite matrix."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    return a @ a.T + n * np.eye(n)


def _layout(n_nuisance=2):
    return CandlParameterLayout(
        cosmo_keys=("H0", "ombh2"), cosmo_sizes=(1, 1),
        cmb_nuisance_names=tuple(f"nuis{i}" for i in range(n_nuisance)))


class _GaussianPriorStub:
    """Minimal stand-in for a clipy likelihood carrying Gaussian priors.

    ``_prior`` maps a parameter name to the same kind of callable clipy builds:
    ``x -> -0.5 (x - mean)^2 / sigma^2``.
    """

    def __init__(self, priors):
        self._prior = {
            name: (lambda x, _m=mean, _s=sigma:
                   -0.5 * ((jnp.asarray(x) - _m) / _s) ** 2)
            for name, (mean, sigma) in priors.items()
        }

    def prior(self, mapping):
        total = jnp.asarray(0.0)
        for name, fn in self._prior.items():
            total = total + fn(mapping[name])
        return total


class _NonGaussianPriorStub(_GaussianPriorStub):
    """Same interface, but with a quartic (non-quadratic) prior."""

    def __init__(self):
        super().__init__({})
        self._prior = {"nuis0": lambda x: -0.5 * jnp.asarray(x) ** 4}


# ---------------------------------------------------------------------------
# (a) Whiteners.
# ---------------------------------------------------------------------------

def test_whitener_from_inv_cov_reproduces_the_quadratic_form():
    inv_cov = _spd(6, seed=1)
    whiten = cmb_gn_fisher._whitener_from_inv_cov(inv_cov)
    rng = np.random.default_rng(11)
    for _ in range(5):
        x = rng.normal(size=6)
        r = np.asarray(whiten(jnp.asarray(x)))
        assert np.isclose(r @ r, x @ inv_cov @ x, rtol=1e-12, atol=0.0)


def test_whitener_from_cov_chol_reproduces_the_quadratic_form():
    cov = _spd(6, seed=2)
    whiten = cmb_gn_fisher._whitener_from_cov_chol(np.linalg.cholesky(cov))
    inv_cov = np.linalg.inv(cov)
    rng = np.random.default_rng(12)
    for _ in range(5):
        x = rng.normal(size=6)
        r = np.asarray(whiten(jnp.asarray(x)))
        assert np.isclose(r @ r, x @ inv_cov @ x, rtol=1e-10, atol=0.0)


def test_whitener_from_inv_cov_rejects_an_asymmetric_matrix():
    """Silently symmetrizing would hide a corrupted covariance."""
    inv_cov = _spd(4, seed=3)
    inv_cov[0, 1] += 10.0
    with pytest.raises(cmb_gn_fisher.SharedPriorError, match="not symmetric"):
        cmb_gn_fisher._whitener_from_inv_cov(inv_cov)


# ---------------------------------------------------------------------------
# (b) gn_fisher against the closed form.
# ---------------------------------------------------------------------------

def _linear_gaussian_pieces(seed=4):
    """A stub whose model is exactly linear, so GN is the closed form."""
    rng = np.random.default_rng(seed)
    n_data, n_par = 7, 4
    jac = rng.normal(size=(n_data, n_par))
    offset = rng.normal(size=n_data)
    inv_cov = _spd(n_data, seed=seed + 1)
    prior_curv = np.zeros((n_par, n_par))
    prior_curv[2, 2] = 1.0 / 0.05 ** 2

    def model_fn(theta):
        return jnp.asarray(offset) + jnp.asarray(jac) @ jnp.asarray(theta)

    def prior_loglike_fn(theta):
        return -0.5 * (jnp.asarray(theta)[2] / 0.05) ** 2

    pieces = {
        "model_fn": model_fn,
        "whiten": cmb_gn_fisher._whitener_from_inv_cov(inv_cov),
        "data": jnp.asarray(rng.normal(size=n_data)),
        "prior_loglike_fn": prior_loglike_fn,
        "source": "stub",
    }
    return pieces, jac, inv_cov, prior_curv, n_par


def test_gn_fisher_equals_jt_cinv_j_plus_prior_and_is_psd():
    pieces, jac, inv_cov, prior_curv, n_par = _linear_gaussian_pieces()
    theta = jnp.asarray(np.linspace(0.1, 0.4, n_par))
    got = cmb_gn_fisher.gn_fisher(pieces, theta)
    expected = jac.T @ inv_cov @ jac + prior_curv
    assert np.allclose(got, expected, rtol=1e-10, atol=1e-10)
    assert np.linalg.eigvalsh(got).min() > 0.0


def test_validate_gn_term_accepts_the_exact_reconstruction():
    pieces, _jac, _inv_cov, _pc, n_par = _linear_gaussian_pieces()
    theta = jnp.asarray(np.linspace(0.1, 0.4, n_par))
    gauss = cmb_gn_fisher.gaussian_loglike_from_pieces(pieces)

    def reference(th):
        return gauss(th) + pieces["prior_loglike_fn"](th)

    report = cmb_gn_fisher.validate_gn_term("stub", pieces, reference, theta)
    assert report["value_rel_err"] < 1e-12
    assert report["directional_abs_err"] < report["directional_budget"]


# ---------------------------------------------------------------------------
# (c)/(e) The validation gate must BITE -- and keep biting under -O.
# ---------------------------------------------------------------------------

def _perturbed_reconstruction():
    """Reference likelihood vs a model_fn scaled by (1 + 1e-6)."""
    pieces, _jac, _inv_cov, _pc, n_par = _linear_gaussian_pieces()
    theta = jnp.asarray(np.linspace(0.1, 0.4, n_par))
    exact_model = pieces["model_fn"]
    gauss_ref = cmb_gn_fisher.gaussian_loglike_from_pieces(pieces)

    def reference(th):
        return gauss_ref(th) + pieces["prior_loglike_fn"](th)

    perturbed = dict(pieces)
    perturbed["model_fn"] = lambda th: exact_model(th) * (1.0 + 1e-6)
    return perturbed, reference, theta


def test_validate_gn_term_raises_on_a_perturbed_model():
    perturbed, reference, theta = _perturbed_reconstruction()
    with pytest.raises(cmb_gn_fisher.GNValidationError):
        cmb_gn_fisher.validate_gn_term("stub", perturbed, reference, theta)


def test_validate_gn_term_still_raises_under_pythonoptimize():
    """``assert`` is stripped by -O; this proves the gate is not assert-based.

    Runs the same perturbed-model reproduction in a subprocess with
    ``PYTHONOPTIMIZE=1``. Before the fix the subprocess exited 0 (the assertions
    were compiled away); now it must report that GNValidationError was raised.
    """
    script = """
import sys
import jax
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, %r)
sys.path.insert(0, %r)
import test_cmb_gn_fisher as t
import cmb_gn_fisher
assert_stripped = True
try:
    assert False
except AssertionError:
    assert_stripped = False
if assert_stripped is False:
    print("ASSERTIONS_ACTIVE")
    raise SystemExit(2)
perturbed, reference, theta = t._perturbed_reconstruction()
try:
    cmb_gn_fisher.validate_gn_term("stub", perturbed, reference, theta)
except cmb_gn_fisher.GNValidationError:
    print("RAISED")
    raise SystemExit(0)
print("NOT_RAISED")
raise SystemExit(1)
""" % (str(_SCRIPTS), str(pathlib.Path(__file__).resolve().parent))
    env = dict(os.environ)
    env["PYTHONOPTIMIZE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(pathlib.Path(__file__).resolve().parents[1] / "src"),
         str(_SCRIPTS), env.get("PYTHONPATH", "")]).strip(os.pathsep)
    proc = subprocess.run([sys.executable, "-c", script], env=env,
                          capture_output=True, text=True, timeout=600)
    assert "ASSERTIONS_ACTIVE" not in proc.stdout, (
        "PYTHONOPTIMIZE did not take effect; the test proves nothing.\n"
        f"{proc.stdout}\n{proc.stderr}")
    assert proc.returncode == 0 and "RAISED" in proc.stdout, (
        "validate_gn_term did not raise under PYTHONOPTIMIZE=1 -- enforcement "
        f"is still assert-based.\nstdout: {proc.stdout}\nstderr: {proc.stderr}")


# ---------------------------------------------------------------------------
# (d) Unknown term names must not inherit the observed-Hessian path.
# ---------------------------------------------------------------------------

def test_make_gn_pieces_raises_on_an_unknown_term():
    with pytest.raises(cmb_gn_fisher.UnknownCmbTermError, match="unknown CMB term"):
        cmb_gn_fisher.make_gn_pieces(
            "spt3g_d1_tteete", object(), pars_to_theory_specs=None,
            layout=_layout())


def test_make_gn_pieces_returns_none_only_for_declared_hessian_terms():
    for name in cmb_gn_fisher.HESSIAN_TERMS:
        assert cmb_gn_fisher.make_gn_pieces(
            name, object(), pars_to_theory_specs=None, layout=_layout()) is None


def test_gn_terms_constant_matches_the_builder_table():
    assert tuple(cmb_gn_fisher._BUILDERS) == cmb_gn_fisher.GN_TERMS
    assert not set(cmb_gn_fisher.GN_TERMS) & set(cmb_gn_fisher.HESSIAN_TERMS)


# ---------------------------------------------------------------------------
# Shared-prior inventory and dedupe.
# ---------------------------------------------------------------------------

def _shared_prior_setup():
    """Three terms; two of them carry the SAME Gaussian prior on ``nuis0``."""
    layout = _layout(n_nuisance=2)
    shared = {"nuis0": (1.0, 0.0025)}
    likelihoods = {
        "term_a": _GaussianPriorStub({**shared, "nuis1": (0.0, 0.5)}),
        "term_b": _GaussianPriorStub(shared),
        "term_c": _GaussianPriorStub(shared),
    }
    theta = jnp.asarray(np.array([67.66, 0.02242, 1.0, 0.0]))
    return likelihoods, layout, theta


def test_inventory_finds_the_shared_prior_with_the_right_width_and_index():
    likelihoods, layout, theta = _shared_prior_setup()
    inventory = cmb_gn_fisher.inventory_shared_priors(
        likelihoods, layout=layout, theta_fid=theta)
    assert set(inventory) == {"nuis0"}, "nuis1 is in one term only"
    entry = inventory["nuis0"]
    assert entry["count"] == 3
    assert entry["terms"] == ["term_a", "term_b", "term_c"]
    assert np.isclose(entry["sigma"], 0.0025, rtol=1e-12)
    assert np.isclose(entry["curvature"], 1.0 / 0.0025 ** 2, rtol=1e-12)
    # The packed index is the thing a silent bug would get wrong.
    assert entry["packed_index"] == (
        layout.nuisance_offset + list(layout.cmb_nuisance_names).index("nuis0"))


def test_dedupe_reproduces_the_analytic_single_count_matrix_exactly():
    likelihoods, layout, theta = _shared_prior_setup()
    size = int(theta.shape[0])

    def prior_fisher(likelihood):
        fn = cmb_gn_fisher.make_prior_loglike_fn(likelihood, layout=layout)
        h = np.asarray(jax.hessian(fn)(theta))
        return -0.5 * (h + h.T)

    summed = sum(prior_fisher(lk) for lk in likelihoods.values())
    inventory = cmb_gn_fisher.inventory_shared_priors(
        likelihoods, layout=layout, theta_fid=theta)
    deduped = summed - cmb_gn_fisher.duplicate_prior_curvature(inventory, size)

    # Analytic single count: nuis0 once at 1/0.0025^2, nuis1 once at 1/0.5^2.
    expected = np.zeros((size, size))
    names = list(layout.cosmo_keys) + list(layout.cmb_nuisance_names)
    expected[names.index("nuis0"), names.index("nuis0")] = 1.0 / 0.0025 ** 2
    expected[names.index("nuis1"), names.index("nuis1")] = 1.0 / 0.5 ** 2
    assert np.allclose(deduped, expected, rtol=1e-12, atol=1e-9)
    # And the pre-dedupe sum really was over-counted, or the test is vacuous.
    assert not np.allclose(summed, expected)


def test_inventory_rejects_a_non_gaussian_prior():
    layout = _layout(n_nuisance=2)
    likelihoods = {
        "term_a": _NonGaussianPriorStub(),
        "term_b": _NonGaussianPriorStub(),
    }
    theta = jnp.asarray(np.array([67.66, 0.02242, 0.3, 0.0]))
    with pytest.raises(cmb_gn_fisher.SharedPriorError, match="not Gaussian"):
        cmb_gn_fisher.inventory_shared_priors(
            likelihoods, layout=layout, theta_fid=theta)


def test_inventory_rejects_priors_of_differing_width_on_the_same_parameter():
    layout = _layout(n_nuisance=2)
    likelihoods = {
        "term_a": _GaussianPriorStub({"nuis0": (1.0, 0.0025)}),
        "term_b": _GaussianPriorStub({"nuis0": (1.0, 0.0050)}),
    }
    theta = jnp.asarray(np.array([67.66, 0.02242, 1.0, 0.0]))
    with pytest.raises(cmb_gn_fisher.SharedPriorError, match="not the same"):
        cmb_gn_fisher.inventory_shared_priors(
            likelihoods, layout=layout, theta_fid=theta)


def test_inventory_rejects_uninventoried_candl_priors():
    class _CandlStub:
        priors = ["some prior object"]

        def prior_logl(self, mapping):
            return jnp.asarray(0.0)

    with pytest.raises(cmb_gn_fisher.SharedPriorError, match="cannot read"):
        cmb_gn_fisher.inventory_shared_priors(
            {"act": _CandlStub()}, layout=_layout(),
            theta_fid=jnp.asarray(np.zeros(4)))


# ---------------------------------------------------------------------------
# CMB_CONFIG_HASH pin enforcement in the builder (review round 2).
#
# The loader protects CONSUMERS from a drifted artifact. These tests protect the
# ARTIFACT ITSELF: a build run in a drifted environment must not overwrite a
# good production block with one the loader would reject and then exit 0. That
# was the live behaviour -- the pin comparison only warned, the npz was written,
# and a message-matching bootstrap escape swallowed the loader's refusal.
# ---------------------------------------------------------------------------

import ast  # noqa: E402
import json  # noqa: E402

import build_cmb_fisher_block as _build  # noqa: E402


def test_pin_none_is_the_bootstrap_case(capsys):
    """A content-derived hash cannot be pinned before it is computed."""
    assert _build.enforce_cmb_config_hash_pin(
        "lcdm", "abc123", pinned=None) is True
    assert "pins None" in capsys.readouterr().out


def test_pin_match_is_a_normal_build(capsys):
    assert _build.enforce_cmb_config_hash_pin(
        "lcdm", "abc123", pinned="abc123") is False


def test_pin_mismatch_exits_nonzero_with_an_action_line(capsys):
    with pytest.raises(SystemExit) as excinfo:
        _build.enforce_cmb_config_hash_pin(
            "nulcdm", "computed-now", pinned="pinned-earlier")
    # sys.exit("message") -> the message IS the code, and the process exit
    # status is 1. Assert the non-zero status the way Python defines it.
    assert excinfo.value.code != 0
    assert excinfo.value.code is not None
    out = capsys.readouterr().out
    assert "MISMATCH" in out
    assert "ACTION" in out
    assert "CMB_CONFIG_HASH_NULCDM = 'computed-now'" in out


def test_pin_mismatch_does_not_clobber_an_existing_artifact(tmp_path,
                                                            monkeypatch):
    """The pre-existing production artifact must survive a drifted build."""
    monkeypatch.setattr(_build.stream_common, "CACHE_DIR", tmp_path)
    artifact = _build.cmb_fisher_path("lcdm")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"PREVIOUS-GOOD-ARTIFACT")

    with pytest.raises(SystemExit):
        _build.enforce_cmb_config_hash_pin(
            "lcdm", "computed-now", pinned="pinned-earlier")

    assert artifact.read_bytes() == b"PREVIOUS-GOOD-ARTIFACT"


def test_pin_is_enforced_before_the_artifact_is_written():
    """Ordering invariant: the guard is useless if it runs after ``np.savez``.

    Checked on the parsed source of ``main`` rather than by running a build,
    which would need the full Planck/ACT data. A refactor that moves the write
    above the pin check re-opens the silent-overwrite hole and fails here.
    """
    source = pathlib.Path(_build.__file__).read_text()
    main_fn = next(node for node in ast.walk(ast.parse(source))
                   if isinstance(node, ast.FunctionDef) and node.name == "main")
    pin_lines = [n.lineno for n in ast.walk(main_fn)
                 if isinstance(n, ast.Call)
                 and getattr(n.func, "id", None) == "enforce_cmb_config_hash_pin"]
    save_lines = [n.lineno for n in ast.walk(main_fn)
                  if isinstance(n, ast.Call)
                  and getattr(n.func, "attr", None) == "savez"]
    assert len(pin_lines) == 1, "main() must enforce the pin exactly once"
    assert len(save_lines) == 1, "main() must write the artifact exactly once"
    assert pin_lines[0] < save_lines[0], (
        "enforce_cmb_config_hash_pin must run BEFORE np.savez, or a mismatched "
        "pin overwrites the production artifact before aborting")


def test_loader_selfcheck_escape_requires_both_pin_state_and_the_pin_message():
    """The escape guard must be a CONJUNCTION of pin state AND the pin error.

    Both single-condition forms are bugs, and each was live at some point:

    * message ALONE (``"cmb_config_hash" in str(exc)``) also matched the pin
      MISMATCH refusal -> a drifted build overwrote a good artifact, exit 0;
    * pin state ALONE (``bootstrap_pin``) swallows every OTHER loader failure
      during a bootstrap build -- and the loader checks cosmology / shared_keys
      / theory_config_hash BEFORE the CMB hash, so a structural defect fires
      first and would have exited 0.
    """
    source = pathlib.Path(_build.__file__).read_text()
    fn = next(node for node in ast.walk(ast.parse(source))
              if isinstance(node, ast.FunctionDef)
              and node.name == "verify_artifact_round_trip")
    handlers = [h for h in ast.walk(fn) if isinstance(h, ast.ExceptHandler)]
    guards = [ast.unparse(n.test) for h in handlers for n in ast.walk(h)
              if isinstance(n, ast.If)]
    assert len(guards) == 1, (
        f"expected exactly one escape guard in verify_artifact_round_trip; "
        f"found {guards}")
    guard = guards[0]
    assert "bootstrap_pin" in guard, (
        f"escape guard {guard!r} is not keyed on the PIN STATE. Keying on the "
        "exception message alone also matches the pin-MISMATCH refusal, which "
        "lets a drifted build overwrite a good artifact and exit 0.")
    # The marker may appear as the literal or via the named constant; what must
    # be true is that the guard inspects the exception message for it.
    inspects_message = "str(exc)" in guard and (
        _build._MISSING_PIN_MARKER in guard or "_MISSING_PIN_MARKER" in guard)
    assert inspects_message, (
        f"escape guard {guard!r} is not keyed on the missing-pin MESSAGE. "
        "Keying on the pin state alone swallows every other loader failure "
        "during a bootstrap build -- and the loader checks cosmology, "
        "shared_keys and theory_config_hash BEFORE the CMB hash, so a "
        "structural defect fires first and would exit 0.")


# --- behaviour of the round-trip self-check (round 3) -----------------------

class _LoaderRaises:
    """Stub for ``stream_common.load_cmb_fisher_block`` that always refuses."""

    def __init__(self, message):
        self.message = message

    def __call__(self, cosmology, cache_dir=None):
        raise ValueError(self.message)


def test_bootstrap_escapes_only_the_genuine_missing_pin_error(monkeypatch,
                                                              capsys):
    """Legitimate bootstrap: pin is None, loader says so -> build succeeds."""
    monkeypatch.setattr(
        _build.stream_common, "load_cmb_fisher_block",
        _LoaderRaises("no CMB_CONFIG_HASH is pinned for 'lcdm' in "
                      "stream_common, so no artifact of that cosmology may be "
                      "loaded. Build it and pin the fingerprint it reports"))
    assert _build.verify_artifact_round_trip(
        "lcdm", bootstrap_pin=True, cmb_config_hash="abc123") is False
    out = capsys.readouterr().out
    assert "NOT yet loadable" in out and "ACTION" in out


def test_bootstrap_still_aborts_on_a_structural_loader_failure(monkeypatch):
    """The regression this round fixes.

    A shared-basis change is a prime reason to ``None`` the pin and rebuild, so
    a bootstrap build is exactly when a shared_keys defect is most likely --
    and the loader raises on it BEFORE it ever reaches the pin check.
    """
    monkeypatch.setattr(
        _build.stream_common, "load_cmb_fisher_block",
        _LoaderRaises("artifact shared_keys ['ombh2', 'omch2', 'logA', 'ns', "
                      "'tau', 'h'] != expected ('ombh2', 'omch2', 'logA', "
                      "'ns', 'h', 'tau')"))
    with pytest.raises(SystemExit) as excinfo:
        _build.verify_artifact_round_trip(
            "lcdm", bootstrap_pin=True, cmb_config_hash="abc123")
    assert excinfo.value.code not in (0, None)
    assert "fails the production loader's guards" in str(excinfo.value.code)


def test_bootstrap_aborts_on_a_wrong_cosmology_artifact(monkeypatch):
    """Same class of defect, the loader's very first check."""
    monkeypatch.setattr(
        _build.stream_common, "load_cmb_fisher_block",
        _LoaderRaises("artifact cosmology 'nulcdm' != requested 'lcdm'"))
    with pytest.raises(SystemExit):
        _build.verify_artifact_round_trip(
            "lcdm", bootstrap_pin=True, cmb_config_hash="abc123")


def test_non_bootstrap_aborts_even_on_the_missing_pin_message(monkeypatch):
    """With a pin in place, nothing is survivable."""
    monkeypatch.setattr(
        _build.stream_common, "load_cmb_fisher_block",
        _LoaderRaises("no CMB_CONFIG_HASH is pinned for 'lcdm'"))
    with pytest.raises(SystemExit):
        _build.verify_artifact_round_trip(
            "lcdm", bootstrap_pin=False, cmb_config_hash="abc123")


def test_round_trip_reports_success_when_the_artifact_loads(monkeypatch,
                                                            capsys):
    monkeypatch.setattr(
        _build.stream_common, "load_cmb_fisher_block",
        lambda cosmology, cache_dir=None: {
            "F_shared": np.eye(6), "sigma_tau": 0.007089623562})
    assert _build.verify_artifact_round_trip(
        "lcdm", bootstrap_pin=True, cmb_config_hash="abc123") is True
    assert "round-trip OK" in capsys.readouterr().out


def test_missing_pin_marker_matches_the_real_loader_message(tmp_path,
                                                            monkeypatch):
    """Pin the COUPLING between the loader's wording and the escape marker.

    If ``load_cmb_fisher_block``'s missing-pin message is ever reworded so it no
    longer contains ``_MISSING_PIN_MARKER``, the escape silently stops working
    and every legitimate bootstrap build starts aborting. Verified against the
    real loader with a synthetic artifact -- no candl, no Planck data.
    """
    monkeypatch.setattr(_build.stream_common, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(_build.stream_common, "CMB_CONFIG_HASH_LCDM", None)
    keys = list(_build.stream_common.SHARED_KEYS_CMB_LCDM)
    k = len(keys)
    meta = {"cosmology": "lcdm", "shared_keys": keys,
            "theory_config_hash": _build.stream_common.THEORY_CONFIG_HASH,
            "cmb_config_hash": "whatever"}
    np.savez(_build.cmb_fisher_path("lcdm"),
             F_cmb_shared=np.eye(k), fid_shared=np.zeros(k),
             shared_keys=np.array(keys), F_cmb_native=np.eye(k),
             fid_native=np.zeros(k), native_keys=np.array(keys),
             sigma_tau=np.float64(0.007), meta_json=json.dumps(meta))

    with pytest.raises(ValueError) as excinfo:
        _build.stream_common.load_cmb_fisher_block("lcdm")
    assert _build._MISSING_PIN_MARKER in str(excinfo.value), (
        f"the loader's missing-pin message no longer contains "
        f"{_build._MISSING_PIN_MARKER!r}, so the bootstrap escape in "
        "verify_artifact_round_trip can never fire and every first build will "
        "abort")
    # ...and the escape does fire against the REAL loader.
    assert _build.verify_artifact_round_trip(
        "lcdm", bootstrap_pin=True, cmb_config_hash="abc123") is False
