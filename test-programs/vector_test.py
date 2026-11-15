import jax
import jax.numpy as jnp
from offload import offload

# One "vector" device hosts every module below; see COMPONENTS in grpc-server/server.py.

@offload(kind='vector', program='vec_add', check=True)
def vec_add(x, y): return x + y

@offload(kind='vector', program='vec_mul', check=True)
def vec_mul(x, y): return x * y

@offload(kind='vector', program='vec_gt', check=True)
def vec_gt(x, y): return x > y

@offload(kind='vector', program='dot', check=True)
def dot(x, y): return jnp.dot(x, y)

@offload(kind='vector', program='axpy', check=True)
def axpy(a, x, y): return a * x + y

@offload(kind='vector', program='vec_sum', check=True)
def vec_sum(x): return jnp.sum(x)

@offload(kind='vector', program='matvec', check=True)
def matvec(w, x): return w @ x

x = jnp.arange(4, dtype=jnp.float32)
y = jnp.full(4, 2.0, dtype=jnp.float32)
print(f"x = {x}, y = {y}")
print(f"vec_add: {vec_add(x, y)}")
print(f"vec_mul: {vec_mul(x, y)}")
print(f"vec_gt: {vec_gt(x, y)}")
print(f"dot: {dot(x, y)}")
print(f"axpy(3, x, y): {axpy(3.0, x, y)}")
print(f"vec_sum: {vec_sum(x)}")

# Batches: each vmapped call below is a single request to the device.
X = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
Y = jnp.ones((3, 4), dtype=jnp.float32)
W = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)
print(f"vmap(dot): {jax.vmap(dot)(X, Y)}")
print(f"vmap(axpy), shared scalar: {jax.vmap(axpy, in_axes=(None, 0, 0))(3.0, X, Y)}")
print(f"vmap(matvec), shared weights: {jax.vmap(matvec, in_axes=(None, 0))(W, X)}")
print(f"nested vmap(dot): {jax.vmap(jax.vmap(dot))(jnp.stack([X, X]), jnp.stack([Y, Y]))}")
print(f"jit(vmap(dot)): {jax.jit(jax.vmap(dot))(X, Y)}")
print(f"jit(sum of two offloaded calls): {jax.jit(lambda a, b: vec_sum(vec_mul(a, b)))(x, y)}")
print(f"lax.map(vec_sum), one request per row: {jax.lax.map(vec_sum, X)}")
