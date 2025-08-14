import flax.linen as nn
import jax.numpy as jnp

from .mlp import MLP


class PolicyNetwork(nn.Module):
    """
    Policy network that outputs action mean and log standard deviation.
    """

    hidden_dims: list[int]
    action_dim: int

    @nn.compact
    def __call__(self, states: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        output = MLP(self.hidden_dims, 2 * self.action_dim)(states)
        mean, log_std = jnp.split(output, 2, axis=1)
        log_std = jnp.clip(log_std, -10, 2)
        return mean, log_std


class VNetwork(nn.Module):
    """
    V-function network with multiple critic heads for state value
    estimation.
    """

    hidden_dims: list[int]
    num_critics: int

    @nn.compact
    def __call__(self, states: jnp.ndarray) -> list[jnp.ndarray]:
        return jnp.asarray(
            [
                MLP(hidden_dims=self.hidden_dims, output_dim=1)(states)
                for _ in range(self.num_critics)
            ]
        )


class QNetwork(nn.Module):
    """
    Q-function network with multiple critic heads for value estimation.
    """

    hidden_dims: list[int]
    num_critics: int

    @nn.compact
    def __call__(
        self, states: jnp.ndarray, actions: jnp.ndarray
    ) -> list[jnp.ndarray]:
        states_actions = jnp.concatenate([states, actions], axis=1)
        return jnp.asarray(
            [
                MLP(hidden_dims=self.hidden_dims, output_dim=1)(states_actions)
                for _ in range(self.num_critics)
            ]
        )


class ZNetwork(nn.Module):
    """
    Distributional critic network outputting quantiles for multiple critics.
    """

    hidden_dims: list[int]
    n_quantiles: int
    n_critics: int

    @nn.compact
    def __call__(
        self, states: jnp.ndarray, actions: jnp.ndarray
    ) -> jnp.ndarray:
        states_actions = jnp.concatenate([states, actions], axis=1)

        return jnp.stack(
            [
                MLP(
                    hidden_dims=self.hidden_dims,
                    output_dim=self.n_quantiles,
                )(states_actions)
                for _ in range(self.n_critics)
            ],
            axis=1,
        )
