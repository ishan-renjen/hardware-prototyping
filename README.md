 to set up conda env:
1. conda create -n fpga-jax python=3.12
2. conda activate fpga-jax
3. pip install "jax==0.11.2" "jaxlib==0.11.2"
## Arithmetic units

The gateway (`grpc-server/server.py`) exposes three two-operand hardware models from
`grpc-server/arithmetic_units.py`:

| device id | kind                | module         | s32 datapath                         |
|-----------|---------------------|----------------|--------------------------------------|
| 5         | `ripple_adder`      | `jit_add`      | 32-bit ripple-carry adder            |
| 6         | `ripple_subtractor` | `jit_subtract` | same adder computing `a + ~b + 1`    |
| 7         | `array_multiplier`  | `jit_multiply` | shift-and-add, low 32 bits kept      |

Integer results wrap on overflow like int32 in JAX. f32/f16 operands use a behavioral
floating-point path that rounds in the operand dtype. Both operands must share a dtype;
they broadcast like `jnp.add` and friends.

To target a unit from JAX, the `program` name must match the module minus its `jit_` prefix:

```python
from offload import offload

@offload(kind="ripple_adder", program="add")
def add(x, y): return x + y
```
