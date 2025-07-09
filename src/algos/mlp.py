# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship.
#
# isir-internship is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import math

import flax.linen as nn
import jax
import jax.numpy as jnp


class MLP(nn.Module):
    """
    Multi-layer perceptron with ReLU activations and orthogonal initialization.
    """

    hidden_dims: list[int]
    output_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the network layers."""

        for hidden_dim in self.hidden_dims:
            x = jax.nn.relu(nn.Dense(hidden_dim)(x))

        return nn.Dense(self.output_dim)(x)
