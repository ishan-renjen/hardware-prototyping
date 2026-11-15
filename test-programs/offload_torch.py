"""torchax adapter for offload: run a PyTorch submodule on a prototype hardware device.

model.block = Offloaded(model.block, kind="trimul", program="tri_block_torch")
weights, jax_func = torchax.extract_jax(model)
y = jax.jit(jax_func)(weights, (x,))

The core offload decorator knows nothing about PyTorch. This adapter reduces a module
to what the core needs: a pure JAX function plus its weights as one dict operand. The
device therefore receives the module's parameters and buffers first, in order of
their PyTorch name, followed by the call's inputs. The model has to run under
torchax; plain PyTorch execution does not reach the device."""

import torch
import torchax
from torchax import interop

from offload import offload


class Offloaded(torch.nn.Module):
    """Drop-in wrapper whose forward runs inner.forward on the device.

    Keyword arguments go to offload()."""

    def __init__(self, inner, **offload_kwargs):
        super().__init__()
        self.inner = inner
        _, jax_func = torchax.extract_jax(inner)  # pure: (weights, args) -> outputs
        self._hw = offload(**offload_kwargs)(lambda weights, *args: jax_func(weights, args))

    def forward(self, *args):
        # Read the weights at call time so the ones the outer model was given are used.
        weights = dict(self.inner.named_parameters())
        weights.update(self.inner.named_buffers())
        return interop.call_jax(self._hw, weights, *args)
