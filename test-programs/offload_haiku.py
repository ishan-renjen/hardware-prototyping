"""Haiku adapter for offload: run chosen Haiku modules on a prototype hardware device
without editing them.

def match(context):  # context is Haiku's MethodContext
    if isinstance(context.module, TriBlock) and context.method_name == "__call__":
        return "tri_block"  # program name on the device

def model(z):
    with offload_haiku_modules(match, kind="trimul"):
        return Trunk()(z)

The core offload decorator knows nothing about Haiku. This adapter reduces a module
call to what the core needs: a pure function plus its weights as one dict operand.
The device therefore receives the module's parameters first, in order of their full
Haiku name, followed by the call's array arguments: positional ones in order, then
keyword ones in the order the caller wrote them. Arguments that hold no arrays
(e.g. is_training=False) are not sent; they are fixed into the offloaded function.
Modules that use Haiku state or random numbers are not supported."""

import contextlib
import threading

import haiku as hk
import jax
import numpy as np
from haiku._src import base as hk_base

from offload import offload

_OPERAND = object()  # placeholder for an argument that travels to the device


def _has_arrays(value):
    return any(isinstance(leaf, (jax.Array, np.ndarray)) for leaf in jax.tree.leaves(value))


@contextlib.contextmanager
def _without_custom_getters():
    """Suspends hk.custom_getter contexts that are active around the offloaded call.

    Haiku keeps custom getters in one per-thread stack rather than per transform, so
    a getter installed by the outer model (AlphaFold's layer_stack slices stacked
    weights this way) would also fire inside the private transform below and process
    weights that were already processed once. Haiku has no public way to suspend
    them, hence the private import (also used to re-read parameters below).
    """
    stack = hk_base.param_getter_stack
    suspended = []
    while stack:
        suspended.append(stack.pop())
    try:
        yield
    finally:
        for getter in reversed(suspended):
            stack.push(getter)


def _nest(flat):
    nested = {}
    for full_name, value in flat.items():
        module_name, name = full_name.rsplit("/", 1)
        nested.setdefault(module_name, {})[name] = value
    return nested


def offload_haiku_modules(match, **offload_kwargs):
    """Context manager for use inside an hk.transform-ed function.

    match(context) returns the program name for module calls that should run on the
    device, or None to leave a call alone. Other keyword arguments go to offload().
    """
    wrappers = {}
    reentry = threading.local()  # set while the module runs on behalf of the pure function

    def interceptor(next_fun, args, kwargs, context):
        program = None if getattr(reentry, "active", False) else match(context)
        if program is None:
            return next_fun(*args, **kwargs)

        # Split the call into operands (anything holding arrays) and static values.
        operands = []
        arg_slots = [_OPERAND if _has_arrays(a) else a for a in args]
        kwarg_slots = {k: _OPERAND if _has_arrays(v) else v for k, v in kwargs.items()}
        operands.extend(a for a in args if _has_arrays(a))
        operands.extend(v for v in kwargs.values() if _has_arrays(v))

        # Find out which parameters this call reads by running it once. The run is only
        # traced; its result is unused, so under jit it costs nothing at run time.
        read = {}

        def record(next_getter, value, getter_context):
            read[getter_context.full_name] = getter_context
            return next_getter(value)

        with hk.custom_getter(record):
            next_fun(*args, **kwargs)

        # Fetch those parameters here, as the module would see them (e.g. sliced per
        # layer by a layer_stack getter). Values captured during the run above cannot
        # be used: a module may read its weights inside a nested scan, where they are
        # tracers that must not leave that scan.
        frame_params = hk_base.current_frame().params
        seen = {}
        for full_name, getter_context in read.items():
            bundle_name, name = full_name.rsplit("/", 1)
            value = frame_params[bundle_name][name]
            if hk_base.param_getter_stack:
                value = hk_base.run_getters(hk_base.param_getter_stack, getter_context, value)
            seen[full_name] = value

        def pure(params, *inputs):
            # Call through the module (not orig_method) so its name scope is entered and
            # submodules resolve to the same parameter names as in the outer model.
            def call(*a):
                a = iter(a)
                call_args = [next(a) if slot is _OPERAND else slot for slot in arg_slots]
                call_kwargs = {k: next(a) if slot is _OPERAND else slot for k, slot in kwarg_slots.items()}
                reentry.active = True
                try:
                    return getattr(context.module, context.method_name)(*call_args, **call_kwargs)
                finally:
                    reentry.active = False

            with _without_custom_getters():
                return hk.without_apply_rng(hk.transform(call)).apply(_nest(params), *inputs)

        static = repr(["array" if slot is _OPERAND else slot for slot in arg_slots]) + repr(
            {k: "array" if slot is _OPERAND else slot for k, slot in kwarg_slots.items()})
        key = (program, context.module.module_name, context.method_name, static)
        if key not in wrappers:
            wrappers[key] = offload(program=program, **offload_kwargs)(pure)
        return wrappers[key](seen, *operands)

    return hk.intercept_methods(interceptor)
