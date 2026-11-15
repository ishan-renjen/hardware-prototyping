"""Co-simulates three AlphaFold 2 Evoformer blocks on the 'alphafold' device.

For each block this prints the hardware contract (operand order, shapes, sizes), then
runs AlphaFold's own module with and without the block on the device and compares.
Run with --full to add one run at N_res = 256.
Needs the alphafold submodule; real weights are used if alphafold_params/ is present.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'alphafold'))

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from alphafold.model import config, data, layer_stack, modules, prng
from offload_haiku import offload_haiku_modules

# Full float32 matmuls if the model itself runs on a GPU (see README).
jax.config.update("jax_default_matmul_precision", "float32")

PARAMS_DIR = os.path.join(REPO, 'alphafold_params')
EVOFORMER_SCOPE = 'alphafold/alphafold_iteration/evoformer/'
CHECK = dict(kind='alphafold', check=True, rtol=1e-3, atol=1e-3)

cfg = config.model_config('model_1')
evo = cfg.model.embeddings_and_evoformer.evoformer
gc = cfg.model.global_config
C_MSA = cfg.model.embeddings_and_evoformer.msa_channel
C_PAIR = cfg.model.embeddings_and_evoformer.pair_channel

# Haiku module name -> (program on the device, how to build the module, which input it takes)
BLOCKS = {
    'pair_transition': ('transition', lambda: modules.Transition(
        evo.pair_transition, gc, name='pair_transition'), 'pair'),
    'msa_transition': ('transition', lambda: modules.Transition(
        evo.msa_transition, gc, name='msa_transition'), 'msa'),
    'triangle_multiplication_outgoing': ('triangle_multiplication_outgoing', lambda: modules.TriangleMultiplication(
        evo.triangle_multiplication_outgoing, gc, name='triangle_multiplication_outgoing'), 'pair'),
    'triangle_multiplication_incoming': ('triangle_multiplication_incoming', lambda: modules.TriangleMultiplication(
        evo.triangle_multiplication_incoming, gc, name='triangle_multiplication_incoming'), 'pair'),
    'outer_product_mean': ('outer_product_mean', lambda: modules.OuterProductMean(
        evo.outer_product_mean, gc, num_output_channel=C_PAIR, name='outer_product_mean'), 'msa'),
}

def on_device(context):
    """Chooses which Haiku module calls go to the device."""
    if context.method_name != '__call__':  # __init__ is intercepted too, before the name exists
        return None
    name = context.module.module_name.rsplit('/', 1)[-1]
    if name in BLOCKS:
        return BLOCKS[name][0]

def inputs(kind, n_seq, n_res, seed=0):
    shape = (n_res, n_res, C_PAIR) if kind == 'pair' else (n_seq, n_res, C_MSA)
    act = jax.random.normal(jax.random.key(seed), shape)
    mask = (jax.random.uniform(jax.random.key(seed + 1), shape[:-1]) > 0.1).astype(jnp.float32)
    return act, mask

def block_models(name):
    build = BLOCKS[name][1]
    def plain(act, mask):
        return build()(act, mask, is_training=False)
    def offloaded(act, mask):
        with offload_haiku_modules(on_device, **CHECK):
            return build()(act, mask, is_training=False)
    return hk.without_apply_rng(hk.transform(plain)), hk.without_apply_rng(hk.transform(offloaded))

def randomised(params, seed=2):
    # Fresh AlphaFold initialisation zeroes each block's last layer, so randomise everything.
    leaves, treedef = jax.tree.flatten(params)
    keys = jax.random.split(jax.random.key(seed), len(leaves))
    return jax.tree.unflatten(treedef, [p + 0.1 * jax.random.normal(k, p.shape) for p, k in zip(leaves, keys)])

def real_block_params(real, name, layer):
    """One Evoformer layer's weights for a block, renamed for the block run on its own."""
    scope = EVOFORMER_SCOPE + 'evoformer_iteration/'
    return {module[len(scope):]: {n: v[layer] for n, v in weights.items()}
            for module, weights in real.items()
            if module == scope + name or module.startswith(scope + name + '/')}

def compare(label, plain, offloaded, params, *args):
    want = jax.jit(plain.apply)(params, *args)
    scale = float(jnp.max(jnp.abs(want)))
    for mode, apply in (('eager', offloaded.apply), ('jit', jax.jit(offloaded.apply))):
        diff = float(jnp.max(jnp.abs(apply(params, *args) - want)))
        print(f"  {label}, {mode}: max abs difference from AlphaFold {diff:.1e} (largest value {scale:.1f})")

real = None
if os.path.exists(os.path.join(PARAMS_DIR, 'params', 'params_model_1.npz')):
    real = data.get_model_haiku_params('model_1', PARAMS_DIR)
else:
    print(f"no real weights under {PARAMS_DIR}; running with randomised weights only")

N_SEQ, N_RES = 8, 16
for name, (program, _, kind) in BLOCKS.items():
    plain, offloaded = block_models(name)
    act, mask = inputs(kind, N_SEQ, N_RES)
    params = plain.init(jax.random.key(1), act, mask)

    print(f"\n=== {name} -> program jit_{program} on device kind 'alphafold'")
    flat = {f"{module}/{n}": v for module, weights in params.items() for n, v in weights.items()}
    operands = [(full_name[len(name) + 1:], v.shape) for full_name, v in sorted(flat.items())]
    operands += [('act', 'pair [N_res, N_res, %d]' % C_PAIR if kind == 'pair' else 'MSA [N_seq, N_res, %d]' % C_MSA),
                 ('mask', '[N_res, N_res]' if kind == 'pair' else '[N_seq, N_res]')]
    for i, (operand, shape) in enumerate(operands):
        print(f"  operand {i:2d}: {operand:28s} {shape}")
    out = plain.apply(params, act, mask)
    print(f"  result: float32 {'[N_res, N_res, %d]' % out.shape[-1] if out.shape[0] == N_RES else '[N_seq, N_res, %d]' % out.shape[-1]}")
    weight_bytes = sum(v.size * 4 for v in flat.values())
    full_act, full_mask = ((256, 256, C_PAIR), (256, 256)) if kind == 'pair' else ((508, 256, C_MSA), (508, 256))
    print(f"  request size at N_res=256{'' if kind == 'pair' else ', N_seq=508'}: "
          f"{(weight_bytes + 4 * np.prod(full_act) + 4 * np.prod(full_mask)) / 2**20:.0f} MB "
          f"(weights {weight_bytes / 2**20:.2f} MB)")

    compare('randomised weights', plain, offloaded, randomised(params), act, mask)
    if real is not None:
        compare('real weights (layer 0)', plain, offloaded, real_block_params(real, name, 0), act, mask)

    if '--full' in sys.argv and kind == 'pair':
        big_act, big_mask = inputs(kind, 0, 256)
        weights = real_block_params(real, name, 0) if real is not None else randomised(params)
        want = jax.jit(plain.apply)(weights, big_act, big_mask)
        diff = float(jnp.max(jnp.abs(jax.jit(offloaded.apply)(weights, big_act, big_mask) - want)))
        print(f"  N_res=256, jit: max abs difference from AlphaFold {diff:.1e}")

# The blocks inside AlphaFold's own Evoformer stack: its EvoformerIteration, repeated by
# its layer_stack, with the attention modules left on the host.
LAYERS = 2
def evoformer_stack(msa, pair, msa_mask, pair_mask):
    iteration = modules.EvoformerIteration(evo, gc, is_extra_msa=False, name='evoformer_iteration')
    def layer(x):
        act, safe_key = x
        safe_key, subkey = safe_key.split()
        out = iteration(activations=act, masks={'msa': msa_mask, 'pair': pair_mask},
                        is_training=False, safe_key=subkey)
        return out, safe_key
    out, _ = layer_stack.layer_stack(LAYERS)(layer)(({'msa': msa, 'pair': pair}, prng.SafeKey(hk.next_rng_key())))
    return out

def evoformer_stack_on_device(*args):
    with offload_haiku_modules(on_device, **CHECK):
        return evoformer_stack(*args)

plain, offloaded = hk.transform(evoformer_stack), hk.transform(evoformer_stack_on_device)
msa, msa_mask = inputs('msa', N_SEQ, N_RES)
pair, pair_mask = inputs('pair', N_SEQ, N_RES)
rng = jax.random.key(3)
if real is not None:
    weights = {module[len(EVOFORMER_SCOPE):]: {n: v[:LAYERS] for n, v in w.items()}
               for module, w in real.items() if module.startswith(EVOFORMER_SCOPE + 'evoformer_iteration')}
    label = 'real weights'
else:
    weights = randomised(plain.init(rng, msa, pair, msa_mask, pair_mask))
    label = 'randomised weights'
want = jax.jit(plain.apply)(weights, rng, msa, pair, msa_mask, pair_mask)
got = jax.jit(offloaded.apply)(weights, rng, msa, pair, msa_mask, pair_mask)
print(f"\n=== {LAYERS}-layer Evoformer stack, {label}, five blocks per layer on the device")
for rep in ('msa', 'pair'):
    print(f"  {rep} representation: max abs difference from AlphaFold "
          f"{float(jnp.max(jnp.abs(got[rep] - want[rep]))):.1e} (largest value {float(jnp.max(jnp.abs(want[rep]))):.1f})")
