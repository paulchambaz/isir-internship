# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship.
#
# isir-internship is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import jax
import jax.numpy as jnp


@jax.jit
def square_error_loss(values: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    return jnp.square(values - targets)


@jax.jit
def soft_update(
    target_params: dict[str, any],
    online_params: dict[str, any],
    tau: float,
) -> dict[str, any]:
    return jax.tree.map(
        lambda t, s: (1 - tau) * t + tau * s, target_params, online_params
    )
