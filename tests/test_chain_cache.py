import numpy as np
import pytest

from jaxptpolypol.sampler import (
    chain_cache_key,
    chain_cache_path,
    load_chain_cache,
    run_chain_cached,
    save_chain_cache,
)


def _fake_sampler_factory(counter):
    def fake_sampler(seed, n):
        counter["calls"] += 1
        rng = np.random.default_rng(seed)
        samples = rng.normal(size=(2, n, 3))
        diagnostics = {"acceptance_rate": np.full(2, 0.25),
                       "proposal_sigma": np.arange(3.0)}
        return samples, diagnostics
    return fake_sampler


def test_run_chain_cached_samples_once_then_loads(tmp_path):
    """Second call with the same config must NOT re-invoke the sampler and
    must return bit-identical samples and diagnostics."""
    counter = {"calls": 0}
    sampler = _fake_sampler_factory(counter)
    cfg = {"sampler": "fake", "seed": 7, "n": 50, "lp0_2dp": -167.75}
    path = chain_cache_path(tmp_path, "toy", cfg)

    s1, d1 = run_chain_cached(sampler, 7, 50, cache_path=path, cache_config=cfg)
    assert counter["calls"] == 1
    assert path.exists()

    s2, d2 = run_chain_cached(sampler, 7, 50, cache_path=path, cache_config=cfg)
    assert counter["calls"] == 1                      # sampler NOT called again
    assert np.array_equal(s1, s2)
    assert sorted(d1) == sorted(d2)
    for k in d1:
        assert np.array_equal(np.asarray(d1[k]), np.asarray(d2[k]))


def test_config_change_changes_filename_so_never_reuses(tmp_path):
    """A changed fingerprint field gives a DIFFERENT path -> miss -> resample.
    Stale reuse is structurally impossible, not merely guarded."""
    counter = {"calls": 0}
    sampler = _fake_sampler_factory(counter)
    cfg_a = {"sampler": "fake", "seed": 7, "n": 50, "lp0_2dp": -167.75}
    cfg_b = {**cfg_a, "lp0_2dp": -167.74}             # posterior moved
    path_a = chain_cache_path(tmp_path, "toy", cfg_a)
    path_b = chain_cache_path(tmp_path, "toy", cfg_b)
    assert path_a != path_b
    run_chain_cached(sampler, 7, 50, cache_path=path_a, cache_config=cfg_a)
    run_chain_cached(sampler, 7, 50, cache_path=path_b, cache_config=cfg_b)
    assert counter["calls"] == 2


def test_load_missing_returns_none(tmp_path):
    cfg = {"a": 1}
    assert load_chain_cache(tmp_path / "absent.npz", config=cfg) is None


def test_internal_config_mismatch_hard_fails(tmp_path):
    """Bypassing chain_cache_path (hand-built path) with a different config
    must raise, never load."""
    cfg_a = {"a": 1}
    cfg_b = {"a": 2}
    path = tmp_path / "hand_built.npz"
    save_chain_cache(path, np.zeros((1, 2, 3)), config=cfg_a)
    with pytest.raises(RuntimeError, match="different config"):
        load_chain_cache(path, config=cfg_b)


def test_non_scalar_config_value_rejected(tmp_path):
    with pytest.raises(TypeError, match="scalar"):
        chain_cache_key({"bad": np.arange(3)})


def test_key_is_order_insensitive():
    assert chain_cache_key({"a": 1, "b": 2}) == chain_cache_key({"b": 2, "a": 1})
