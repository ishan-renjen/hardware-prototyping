import haiku as hk
import jax
import jax.numpy as jnp
from offload import offload
from offload_haiku import offload_haiku_modules

# Full float32 matmuls if the model itself runs on a GPU (see README).
jax.config.update("jax_default_matmul_precision", "float32")

N, C, BLOCKS = 32, 16, 3

@offload(kind='trimul', program='tri_mul', check=True, rtol=1e-4, atol=1e-4)
def tri_mul(a, b): return jnp.einsum('ikc,jkc->ijc', a, b)

class TriBlock(hk.Module):
    """Two projections, a triangle contraction, an output projection."""
    def __init__(self, contract=None, name='tri_block'):
        super().__init__(name=name)
        self.contract = contract or (lambda a, b: jnp.einsum('ikc,jkc->ijc', a, b))

    def __call__(self, z):
        a = hk.Linear(C, name='left')(z)
        b = hk.Linear(C, name='right')(z)
        return hk.Linear(C, name='out')(self.contract(a, b))

def trunk(z, contract=None):
    return hk.layer_stack(BLOCKS)(lambda z: z + 0.01 * TriBlock(contract)(z))(z)

def is_tri_block(context):
    if isinstance(context.module, TriBlock) and context.method_name == '__call__':
        return 'tri_block'

def trunk_with_block_on_device(z):
    with offload_haiku_modules(is_tri_block, kind='trimul', check=True, rtol=1e-4, atol=1e-4):
        return trunk(z)

plain = hk.without_apply_rng(hk.transform(trunk))
# 1. the module calls an offloaded function for its contraction
contraction = hk.without_apply_rng(hk.transform(lambda z: trunk(z, contract=tri_mul)))
# 2. the unmodified module runs on the device, with its weights sent as operands
whole_block = hk.without_apply_rng(hk.transform(trunk_with_block_on_device))

z = jax.random.normal(jax.random.key(0), (N, N, C))
params = plain.init(jax.random.key(1), z)
want = jax.jit(plain.apply)(params, z)

for label, model in (('contraction on device', contraction), ('whole block on device', whole_block)):
    for mode, apply in (('eager', model.apply), ('jit', jax.jit(model.apply))):
        diff = jnp.max(jnp.abs(apply(params, z) - want))
        print(f"{label}, {mode}: max abs difference from plain Haiku {diff:.1e}")
