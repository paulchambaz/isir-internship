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
def se_loss(value: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.square(value - target)


@jax.jit
def huber_loss(
    value: jnp.ndarray, target: jnp.ndarray, delta: float
) -> jnp.ndarray:
    abs_errors = jnp.abs(value - target)

    return jnp.where(
        abs_errors <= delta,
        0.5 * jnp.square(value - target),
        delta * abs_errors - 0.5 * delta**2,
    )


@jax.jit
def quantile_loss(value: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    batch_size, n_critics, n_quantiles = value.shape

    tau = jnp.linspace(
        1.0 / (2.0 * n_quantiles), 1.0 - 1.0 / (2.0 * n_quantiles), n_quantiles
    )

    value_expanded = jnp.expand_dims(value, axis=-1)

    target_expanded = jnp.expand_dims(target, axis=(1, 2))

    tau_expanded = jnp.expand_dims(tau, axis=(0, 1, 3))

    loss = huber_loss(value=value_expanded, target=target_expanded, delta=1.0)

    weights = jnp.abs(tau_expanded - (value_expanded - target_expanded < 0))
    weighted_loss = weights * loss

    loss_per_quantile = weighted_loss.sum(axis=-1) / target.shape[-1]

    return loss_per_quantile.mean(axis=(1, 2))


@jax.jit
def soft_update(
    target_params: dict[str, any],
    online_params: dict[str, any],
    tau: float,
) -> dict[str, any]:
    return jax.tree.map(
        lambda t, s: (1 - tau) * t + tau * s, target_params, online_params
    )


@jax.jit
def lerp(t: jnp.ndarray | float, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return (1 - t) * a + t * b


@jax.jit
def where(
    pred: jnp.ndarray, case_true: jnp.ndarray, case_false: jnp.ndarray
) -> jnp.ndarray:
    return lerp(pred, case_true, case_false)
