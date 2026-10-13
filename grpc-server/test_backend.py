
import jax
import numpy as np

print(jax.devices('prototype'))

x = np.arange(4, dtype=np.float32)
for d in jax.devices('prototype'):
    y = jax.device_put(x, d)
    print(d, d.device_kind, np.array_equal(np.asarray(y), x))