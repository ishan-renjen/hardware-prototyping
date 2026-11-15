"""Decorator that runs a JAX function on a prototype hardware device.
@offload(kind="adder", program="add_one")
def add_one(x):
    return x + 1

y = add_one(x)

The decorated function also works under jax.jit, jax.vmap, jax.lax.map and the
lax control-flow operators. Under vmap the whole batch reaches the device in one
call, and operands that are not batched (e.g. weights with in_axes=None) are sent
once rather than once per sample. The hardware module must be a pure function of
its inputs: JAX may drop, repeat or reorder the call. jax.grad is not supported."""

import functools

import jax
import jax.numpy as jnp
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
    return jax.tree.map(np.asarray, tree)


def offload(
    kind=None,
    *,
    device_id=None,
    program=None,
    check=False,
    rtol=1e-5,
    atol=1e-6,
    platform=PLATFORM,
    trace=None,
):
    """trace, if given, is called after every device call as
    trace(program, args, result, reference): args is the call's argument tree as sent
    (NumPy arrays), result the device's output, reference the CPU result (None
    unless check=True). Under vmap, args and result carry the batch dimensions."""

    def decorate(fn):
        name = program or fn.__name__
        state = {}
        compiled = {}

        def build(treedef, masks):
            # masks[i][j] says whether argument leaf i is batched along batch axis j.
            # Every batch layout compiles to a module with the same name, so the
            # device sees one program whose inputs carry extra leading dimensions.
            key = (treedef, masks)
            if key not in compiled:

                def per_sample(*leaves):
                    return fn(*jax.tree.unflatten(treedef, leaves))

                batched = per_sample
                batch_ndim = len(masks[0]) if masks else 0
                for axis in reversed(range(batch_ndim)):
                    in_axes = tuple(0 if mask[axis] else None for mask in masks)
                    batched = jax.vmap(batched, in_axes=in_axes)

                def body(*leaves):
                    return batched(*leaves)

                body.__name__ = body.__qualname__ = name
                # keep_unused: the device module has a fixed arity, so every
                # argument goes on the wire even if the JAX body ignores it.
                hw_fn = jax.jit(body, keep_unused=True)
                ref_fn = jax.jit(batched) if check else None
                compiled[key] = (hw_fn, ref_fn)
            return compiled[key]

        def run_on_hw(treedef, masks, leaves):
            hw_fn, ref_fn = build(treedef, masks)
            outputs = _to_host(hw_fn(*jax.device_put(leaves, state["device"])))
            expected = None
            if check:
                # Reference on the CPU: a GPU backend may use reduced matmul precision
                # by default (~1e-3 relative error on an A100), which would fail the
                # comparison even when the hardware result is right.
                with jax.default_device(jax.devices("cpu")[0]):
                    expected = ref_fn(*leaves)
                jax.tree.map(
                    lambda got, want: np.testing.assert_allclose(
                        got,
                        np.asarray(want),
                        rtol=rtol,
                        atol=atol,
                        err_msg=f"{name}: hardware result differs from reference",
                    ),
                    outputs,
                    expected,
                )
            if trace is not None:
                args = jax.tree.unflatten(treedef, [np.asarray(leaf) for leaf in leaves])
                trace(name, args, outputs, None if expected is None else _to_host(expected))
            return outputs

        @functools.wraps(fn)
        def wrapper(*args):
            if "device" not in state:
                state["device"] = _find_device(kind, device_id, platform)
            leaves, treedef = jax.tree.flatten(args)

            if not any(isinstance(leaf, jax.core.Tracer) for leaf in leaves):
                unbatched = tuple(() for _ in leaves)
                outputs = run_on_hw(treedef, unbatched, [np.asarray(leaf) for leaf in leaves])
                return jax.device_put(outputs, jax.devices()[0])

            # Being traced (jit, vmap, scan, ...): defer the hardware call to run time.
            core_ndims = [jnp.ndim(leaf) for leaf in leaves]

            def host_call(*traced_leaves):
                arrays = [np.asarray(leaf) for leaf in traced_leaves]
                batch_ndim = arrays[0].ndim - core_ndims[0]
                # vmap_method="expand_dims" gives unbatched operands a size-1 axis.
                batch_shape = np.broadcast_shapes(*(a.shape[:batch_ndim] for a in arrays))
                masks = tuple(
                    tuple(a.shape[axis] == batch_shape[axis] for axis in range(batch_ndim))
                    for a in arrays
                )
                squeezed = [
                    a.reshape(
                        tuple(n for n, keep in zip(a.shape[:batch_ndim], mask) if keep)
                        + a.shape[batch_ndim:]
                    )
                    for a, mask in zip(arrays, masks)
                ]
                return run_on_hw(treedef, masks, squeezed)

            return jax.pure_callback(
                host_call, jax.eval_shape(fn, *args), *leaves, vmap_method="expand_dims"
            )

        def lower(*args):
            leaves, treedef = jax.tree.flatten(args)
            return build(treedef, tuple(() for _ in leaves))[0].lower(*leaves)

        wrapper.program = f"jit_{name}"
        wrapper.lower = lower
        return wrapper

    return decorate
