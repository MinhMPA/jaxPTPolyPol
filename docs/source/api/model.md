# `model`

Thin wrappers around the external theory codes: ``CosmoEmulator`` over the
``cosmopower-jax`` linear-``P(k)`` emulator, and ``PS1LoopModel`` / ``BispectrumTreeModel``
over ``ps_1loop_jax``. All three are registered JAX pytrees, so they can cross a
``jax.jit`` boundary as static configuration, and they present the uniform interface that
both the Fisher and MCMC pipelines call.

```{eval-rst}
.. automodule:: jaxptpolypol.model
   :members:
   :undoc-members:
   :show-inheritance:
```
