import copy

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torchax
from torchax import interop
from offload import offload
from offload_torch import Offloaded

# Full float32 matmuls if the model itself runs on a GPU (see README).
jax.config.update("jax_default_matmul_precision", "float32")

N, C, BLOCKS = 32, 16, 3

@offload(kind='trimul', program='tri_mul', check=True, rtol=1e-4, atol=1e-4)
def tri_mul(a, b): return jnp.einsum('ikc,jkc->ijc', a, b)

class TriBlock(torch.nn.Module):
    """Two projections, a triangle contraction, an output projection."""
    def __init__(self):
        super().__init__()
        self.left, self.right, self.out = (torch.nn.Linear(C, C) for _ in range(3))
        self.contract = lambda a, b: torch.einsum('ikc,jkc->ijc', a, b)

    def forward(self, z):
        return self.out(self.contract(self.left(z), self.right(z)))

class Trunk(torch.nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, z):
        for block in self.blocks:
            z = z + 0.01 * block(z)
        return z

def contraction_on_device(block):
    # 1. the module calls an offloaded function for its contraction
    block.contract = lambda a, b: interop.call_jax(tri_mul, a, b)
    return block

def whole_block_on_device(block):
    # 2. the unmodified module runs on the device, with its weights sent as operands
    return Offloaded(block, kind='trimul', program='tri_block_torch', check=True, rtol=1e-4, atol=1e-4)

torch.manual_seed(0)
plain = Trunk([TriBlock() for _ in range(BLOCKS)])
z = torch.randn(N, N, C)
with torch.no_grad():
    want = plain(z).numpy()

for label, change in (('contraction on device', contraction_on_device), ('whole block on device', whole_block_on_device)):
    model = Trunk([change(copy.deepcopy(block)) for block in plain.blocks])
    weights, jax_func = torchax.extract_jax(model)
    for mode, apply in (('eager', jax_func), ('jit', jax.jit(jax_func))):
        diff = np.max(np.abs(np.asarray(apply(weights, (jnp.asarray(z.numpy()),))) - want))
        print(f"{label}, {mode}: max abs difference from plain PyTorch {diff:.1e}")
