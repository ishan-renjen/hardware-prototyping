import jax.numpy as jnp
from offload import offload

@offload(kind='adder', program='add_one', check=True)
def add_one(x): return x + 1

@offload(kind='scaler', program='times_two', check=True)
def times_two(x): return x * 2

x = jnp.arange(4, dtype=jnp.float32)
print("running add_one")
print(add_one(x)) 
print("running times_two")
print(times_two(x)) 
print("nested test")
print(times_two(add_one(x)))