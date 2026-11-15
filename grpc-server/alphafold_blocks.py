"""NumPy reference models of AlphaFold 2 Evoformer blocks.

Each function is the server-side stand-in for one hardware block and doubles as its
specification: replace the function with the hardware model and the client-side
check (offload(check=True)) will tell you whether it still matches AlphaFold.

Operand order is what the client sends: the block's weights sorted by their Haiku
parameter name, then the block's inputs (act, mask). All tensors are float32.
Class and line references are to alphafold/alphafold/model/modules.py.
"""
import numpy as np


def layer_norm(x, scale, offset, eps=1e-5):
    """hk.LayerNorm over the last axis."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return scale * (x - mean) / np.sqrt(var + np.float32(eps)) + offset


def sigmoid(x):
    with np.errstate(over="ignore"):
        return 1 / (1 + np.exp(-x))


def transition(
    input_layer_norm_offset,  # [c]
    input_layer_norm_scale,  # [c]
    transition1_bias,  # [4c]
    transition1_weights,  # [c, 4c]
    transition2_bias,  # [c]
    transition2_weights,  # [4c, c]
    act,  # [..., c]   pair: [N_res, N_res, 128]   MSA: [N_seq, N_res, 256]
    mask,  # [...]     unused by the computation, sent because the module takes it
):
    """Transition (pair_transition and msa_transition), modules.py:515.

    Every position is independent: LayerNorm -> Linear -> ReLU -> Linear.
    """
    x = layer_norm(act, input_layer_norm_scale, input_layer_norm_offset)
    hidden = np.maximum(x @ transition1_weights + transition1_bias, 0)
    return hidden @ transition2_weights + transition2_bias


def triangle_multiplication(equation):
    """TriangleMultiplication, modules.py:1358.

    equation is "ikc,jkc->ijc" (outgoing edges) or "kjc,kic->ijc" (incoming edges);
    everything else is identical between the two.
    """

    def block(
        center_layer_norm_offset,  # [c]
        center_layer_norm_scale,  # [c]
        gating_linear_bias,  # [c]
        gating_linear_weights,  # [c, c]
        layer_norm_input_offset,  # [c]
        layer_norm_input_scale,  # [c]
        left_gate_bias,  # [c]
        left_gate_weights,  # [c, c]
        left_projection_bias,  # [c]
        left_projection_weights,  # [c, c]
        output_projection_bias,  # [c]
        output_projection_weights,  # [c, c]
        right_gate_bias,  # [c]
        right_gate_weights,  # [c, c]
        right_projection_bias,  # [c]
        right_projection_weights,  # [c, c]
        act,  # [N_res, N_res, c], c = 128
        mask,  # [N_res, N_res]
    ):
        mask = mask[..., None]
        x = layer_norm(act, layer_norm_input_scale, layer_norm_input_offset)
        left = mask * (x @ left_projection_weights + left_projection_bias)
        right = mask * (x @ right_projection_weights + right_projection_bias)
        left = left * sigmoid(x @ left_gate_weights + left_gate_bias)
        right = right * sigmoid(x @ right_gate_weights + right_gate_bias)
        y = np.einsum(equation, left, right, optimize=True)  # the N_res^3 contraction
        y = layer_norm(y, center_layer_norm_scale, center_layer_norm_offset)
        y = y @ output_projection_weights + output_projection_bias
        return y * sigmoid(x @ gating_linear_weights + gating_linear_bias)

    return block


def outer_product_mean(
    layer_norm_input_offset,  # [c_m]
    layer_norm_input_scale,  # [c_m]
    left_projection_bias,  # [32]
    left_projection_weights,  # [c_m, 32]
    output_b,  # [c_z]
    output_w,  # [32, 32, c_z]
    right_projection_bias,  # [32]
    right_projection_weights,  # [c_m, 32]
    act,  # [N_seq, N_res, c_m], c_m = 256
    mask,  # [N_seq, N_res]
):
    """OuterProductMean, modules.py:1600. MSA representation in, pair update out.

    out[i, j, f] = (sum over sequences s and channels c, e of
                    left[s, i, c] * right[s, j, e] * output_w[c, e, f]) + output_b[f],
    divided by the number of sequences in which both i and j are unmasked.
    """
    mask = mask[..., None]
    x = layer_norm(act, layer_norm_input_scale, layer_norm_input_offset)
    left = mask * (x @ left_projection_weights + left_projection_bias)
    right = mask * (x @ right_projection_weights + right_projection_bias)
    outer = np.einsum("sic,sje->ijce", left, right, optimize=True)
    out = np.einsum("ijce,cef->ijf", outer, output_w, optimize=True) + output_b
    norm = np.einsum("abc,adc->bdc", mask, mask)  # [N_res, N_res, 1]
    return out / (np.float32(1e-3) + norm)
