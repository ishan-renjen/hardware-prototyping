"""Shims that let the unmodified AlphaFold 2 submodule run on this repo's JAX version.

Import before running AlphaFold's model. Each shim names the AlphaFold code that needs
it, so it can be dropped once the fork carries the fix.
"""
import functools

import jax.numpy as jnp

_clip = jnp.clip


@functools.wraps(_clip)
def _clip_accepting_a_min_a_max(arr=None, /, min=None, max=None, *, a_min=None, a_max=None):
    # JAX removed clip's a_min/a_max keywords; AlphaFold still uses them for the
    # relative position encoding (alphafold/model/modules.py:1982, and
    # modules_multimer.py:581,606).
    return _clip(arr, min=a_min if min is None else min, max=a_max if max is None else max)


jnp.clip = _clip_accepting_a_min_a_max
