import jax.numpy as jnp
from offload import offload

@offload(kind='adder', program='add_one', check=True)
def add_one(x): return x + 1

@offload(kind='scaler', program='times_two', check=True)
def times_two(x): return x * 2

x = jnp.arange(4, dtype=jnp.float32)
print(x)
print(f"adding one: {add_one(x)}") 
print(f"multiply by 2: {times_two(x)}") 
print(f"multiply by 2(add one(x)): {times_two(add_one(x))}")