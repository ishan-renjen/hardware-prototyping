"""Decorator that runs a JAX function on a prototype hardware device.
@offload(kind="adder", program="add_one")
def add_one(x):
    return x + 1

y = add_one(x)"""

import functools

import jax
import numpy as np

PLATFORM = "prototype"


def _find_device(kind=None, device_id=None, platform=PLATFORM):
    devices = jax.devices(platform)
    if device_id is not None:
        for d in devices:
            if d.id == device_id:
                return d
        raise ValueError(
            f"no {platform} device with id {device_id}; have {[d.id for d in devices]}"
        )
    if kind is not None:
        for d in devices:
            if d.device_kind == kind:
                return d
        raise ValueError(
            f"no {platform} device of kind {kind!r}; have {[d.device_kind for d in devices]}"
        )
    return devices[0]


def _to_host(tree):
    try:
        return jax.tree.map(np.asarray, tree)
    except (
        jax.errors.TracerArrayConversionError,
        jax.errors.ConcretizationTypeError,
    ) as e:
        raise TypeError(
            "@offload functions cannot run inside jax.jit, vmap, or grad - hardware call is a separate thing"
        ) from e


def offload(
    kind=None,
    *,
    device_id=None,
    program=None,
    check=False,
    rtol=1e-5,
    atol=1e-6,
    platform=PLATFORM,
):
    def decorate(fn):
        name = program or fn.__name__

        def body(*args):
            return fn(*args)

        body.__name__ = body.__qualname__ = name
        hw_fn = jax.jit(body)
        ref_fn = jax.jit(fn) if check else None
        state = {}

        @functools.wraps(fn)
        def wrapper(*args):
            if "device" not in state:
                state["device"] = _find_device(kind, device_id, platform)
            hw = state["device"]
            home = jax.devices()[0]

            host_args = _to_host(args)
            outputs = hw_fn(*jax.device_put(host_args, hw))
            result = jax.device_put(_to_host(outputs), home)

            if check:
                expected = ref_fn(*jax.device_put(host_args, home))
                jax.tree.map(
                    lambda got, want: np.testing.assert_allclose(
                        np.asarray(got),
                        np.asarray(want),
                        rtol=rtol,
                        atol=atol,
                        err_msg=f"{name}: hardware result differs from reference",
                    ),
                    result,
                    expected,
                )
            return result

        wrapper.program = f"jit_{name}"
        wrapper.lower = lambda *args: hw_fn.lower(*args)
        return wrapper

    return decorate
