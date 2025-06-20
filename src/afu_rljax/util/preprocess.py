import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def add_noise(
    x: jnp.ndarray,
    key: jnp.ndarray,
    std: float,
    out_min: float = -np.inf,
    out_max: float = np.inf,
    noise_min: float = -np.inf,
    noise_max: float = np.inf,
) -> jnp.ndarray:
    """
    Add noise to actions.
    """
    noise = jnp.clip(jax.random.normal(key, x.shape), noise_min, noise_max)
    return jnp.clip(x + noise * std, out_min, out_max)
