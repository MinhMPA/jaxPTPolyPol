# Removal of the `bs_tree` jax-safe shims (2026-09-03)

Tracked record of a cross-repo change. This lives here, not in `CONTEXT.md`:
that file is a glossary of domain terms by its own header, and it is
gitignored, so a note there cannot travel with a PR or a fresh clone.

## What was removed

`BispectrumTreeModel.__init__` used to monkey-patch jax-safe replacements onto
its `ps_1loop_jax.bs_tree.BispectrumTree` instance:

| Patched attribute | Why it existed |
|---|---|
| `_validate_triangle` | upstream ran eager NumPy on values that are tracers under `jit`, raising `TracerArrayConversionError` |
| `_get_ndens` | same, on stochastic amplitudes -- and its eager check ran *before* the structural short-circuits, so it fired even with no stochasticity |
| `_needs_ndens` | dangling: upstream had no such method, so the assignment was inert |
| `_get_bk_shot` | `hasattr`-guarded: upstream had no such method, so it never bound |

Their module-level helpers (`_contains_tracer`, `_triangle_closure_tolerance`,
`_validate_triangle_eager`, and the three `*_jax_safe` functions) went with them.
The shims also carried a `10*eps*scale` triangle-closure tolerance, without
which grid-built triangles are rejected: 4 of the 264 production triangles
(`K_BK_MIN..K_BK_MAX` on the `stream_common` grid) close only to ~1.4e-17.

## Why it is safe

`ps_1loop_jax` 0.4.0 adopted all of that behaviour upstream, copying the shim
semantics verbatim including the tolerance and the choice to report an invalid
triangle at *runtime* (via `jax.debug.callback`) rather than at trace time.
Measured before removing anything: `get_bk0` over all 264 production triangles
under the full production config (`do_AP=True`, `do_irres=True`, stochasticity,
AP alphas) is **bitwise identical** with and without the shims, eager and under
`jit`. Triangle acceptance is unchanged; validation only raises or passes, it
never filters, so data-vector length is fixed by the untouched builder in
`theory.py`.

## How the dependency is enforced

Not by a version floor alone. Both repos are used as editable installs, where
`importlib.metadata` reports whatever `pyproject.toml` last said and does not
track edits; upstream's version has also regressed historically (0.3.0 ->
0.1.0 via a merge of `archaeo-pteryx/ps_1loop_jax`, which carried its own
numbering), and there were no tags until v0.4.0.

So the gate is a **capability flag**, checked in
`BispectrumTreeModel.__init__` via `_require_jit_safe_bs_tree()`:

```python
"bs_tree_jit_safe_validation" in ps_1loop_jax.FEATURES
```

`pyproject.toml` additionally pins `ps_1loop_jax>=0.4.0` for non-editable
installs, but the flag is what actually protects a run. Without the guard the
failure is silent until deep into a notebook -- a `ValueError` on grid-edge
triangles, or `TracerArrayConversionError` under `jit`, after minutes spent
building emulators. Every notebook `jax.jit(joint_fn)`s with `triangles` passed
as a keyword argument and no `static_argnames`, so the traced path is the
normal path.

## References

- `ps_1loop_jax-for-pfs` PR #5 (`jit-safe-validation`), tag `v0.4.0`
- `jaxPTPolyPol` PR #2 (`cleanup/drop-bs-tree-shims`)
