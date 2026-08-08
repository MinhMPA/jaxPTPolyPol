import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def test_pack_rejects_mismatched_cosmo_key_order():
    from jaxptpolypol.cmb import CandlParameterLayout
    from jaxptpolypol.params import CosmoParams

    layout = CandlParameterLayout(
        cosmo_keys=("H0", "tau"), cosmo_sizes=(1, 1), cmb_nuisance_names=()
    )
    wrong_order = CosmoParams({"tau": 0.05, "H0": 67.0})
    with pytest.raises(ValueError, match="cosmo_keys"):
        layout.pack(wrong_order, {})


def test_pack_accepts_matching_cosmo_key_order():
    from jaxptpolypol.cmb import CandlParameterLayout
    from jaxptpolypol.params import CosmoParams

    layout = CandlParameterLayout(
        cosmo_keys=("H0", "tau"), cosmo_sizes=(1, 1), cmb_nuisance_names=()
    )
    right_order = CosmoParams({"H0": 67.0, "tau": 0.05})
    packed = layout.pack(right_order, {})
    np.testing.assert_allclose(np.asarray(packed), [67.0, 0.05])
