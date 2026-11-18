"""Hardware models of basic arithmetic units: adder, subtractor, multiplier.

Each unit is a server-side stand-in for a piece of hardware sitting behind the
gateway. Integer operands (s32) go through a bit-level model of the datapath:

  * adder       - 32-bit ripple-carry adder built from full adders
  * subtractor  - the same adder computing a + ~b + 1 (two's complement)
  * multiplier  - shift-and-add array multiplier, one partial product per bit
                  of the multiplier, accumulated with the ripple-carry adder

All integer results are the low 32 bits, i.e. they wrap on overflow exactly like
int32 arithmetic in JAX/XLA. Floating-point operands (f32, f16) go to a
behavioral floating-point path that rounds in the operand dtype, so f32 in gives
f32 out.

The models are elementwise and vectorized over the whole tensor: one "cycle" of
the bit-level loop processes the same bit position of every element at once.
Operands broadcast against each other (e.g. a scalar and a vector), matching
what jnp.add/subtract/multiply would accept.

Module names follow the client's convention: a JAX function named `add` jitted
for the device lowers to the HLO module `jit_add`, which is what the gateway
receives in ExecuteRequest.module.
"""
import numpy as np

WIDTH = 32
_MASK = np.uint32(0xFFFFFFFF)
_ONE = np.uint32(1)

INT_DTYPES = (np.dtype(np.int32),)
FLOAT_DTYPES = (np.dtype(np.float32), np.dtype(np.float16))


# ---------------------------------------------------------------- gate level
def full_adder(a, b, cin):
    """One full adder per element. a, b, cin are 0/1 uint32 arrays."""
    axb = a ^ b
    s = axb ^ cin
    cout = (a & b) | (axb & cin)
    return s, cout


def ripple_carry_add(a, b, cin=0):
    """WIDTH-bit ripple-carry adder on uint32 words.

    Returns (sum, carry_out). The carry ripples from bit 0 to bit WIDTH-1, one
    full adder per bit, exactly as in the hardware chain.
    """
    a = np.asarray(a, dtype=np.uint32)
    b = np.asarray(b, dtype=np.uint32)
    a, b = np.broadcast_arrays(a, b)
    carry = np.full(a.shape, cin, dtype=np.uint32)
    total = np.zeros(a.shape, dtype=np.uint32)
    for i in range(WIDTH):
        bit = np.uint32(i)
        s, carry = full_adder((a >> bit) & _ONE, (b >> bit) & _ONE, carry)
        total |= s << bit
    return total, carry


def ripple_carry_subtract(a, b):
    """a - b on the adder: invert b and set carry-in (two's complement)."""
    b_inv = ~np.asarray(b, dtype=np.uint32) & _MASK
    diff, _ = ripple_carry_add(a, b_inv, cin=1)
    return diff


def array_multiply(a, b):
    """Shift-and-add multiplier, keeping the low WIDTH bits of the product.

    For each bit i of the multiplier b, the partial product (a << i) is gated by
    that bit and accumulated through the ripple-carry adder. The low WIDTH bits
    of a two's-complement product are the same for signed and unsigned
    operands, so this is also correct for int32.
    """
    a = np.asarray(a, dtype=np.uint32)
    b = np.asarray(b, dtype=np.uint32)
    a, b = np.broadcast_arrays(a, b)
    acc = np.zeros(a.shape, dtype=np.uint32)
    for i in range(WIDTH):
        bit = np.uint32(i)
        gate = np.uint32(0) - ((b >> bit) & _ONE)  # all ones where bit i of b is set
        partial = (a << bit) & gate
        acc, _ = ripple_carry_add(acc, partial)
    return acc


# ---------------------------------------------------------------- dispatch
def _check_operands(op, a, b):
    if a.dtype != b.dtype:
        raise ValueError(
            f"{op}: operands must share a dtype, got {a.dtype} and {b.dtype}")
    if a.dtype not in INT_DTYPES + FLOAT_DTYPES:
        raise ValueError(f"{op}: unsupported dtype {a.dtype}")


def _int_unit(datapath):
    def run(a, b):
        with np.errstate(over="ignore"):  # wraparound is the hardware behavior
            out = datapath(a.view(np.uint32), b.view(np.uint32))
        return np.asarray(out, dtype=np.uint32).view(np.int32)
    return run


def _unit(op, int_path, float_path):
    int_run = _int_unit(int_path)

    def unit(a, b):
        a, b = np.asarray(a), np.asarray(b)
        _check_operands(op, a, b)
        if a.dtype in INT_DTYPES:
            return int_run(a, b)
        # Behavioral floating-point unit: IEEE result rounded in the operand dtype.
        return np.asarray(float_path(a, b), dtype=a.dtype)

    unit.__name__ = op
    return unit


add = _unit("add", lambda a, b: ripple_carry_add(a, b)[0], np.add)
subtract = _unit("subtract", ripple_carry_subtract, np.subtract)
multiply = _unit("multiply", array_multiply, np.multiply)

# Module name (as lowered by JAX) -> hardware model, per unit.
ADDER_MODULES = {"jit_add": add}
SUBTRACTOR_MODULES = {"jit_subtract": subtract}
MULTIPLIER_MODULES = {"jit_multiply": multiply}
