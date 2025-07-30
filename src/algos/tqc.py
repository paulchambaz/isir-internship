# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship.
#
# isir-internship is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import math
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random

from .mlp import MLP
from .replay import ReplayBuffer
from .rl_algo import RLAlgo


class TQC(RLAlgo):
    """
    Truncated Quantile Critics (TQC) algorithm for continuous control tasks.

    TQC uses distributional critics with quantile regression and truncation
    to control overestimation bias while maintaining fine-grained control
    over the bias-variance tradeoff.

    Args:
        state_dim: Dimensionality of the state space
        action_dim: Dimensionality of the action space
        hidden_dims: List of hidden layer sizes for neural networks
        replay_size: Maximum size of the replay buffer
        batch_size: Number of samples per training batch
        critic_lr: Learning rate for Z-function networks
        policy_lr: Learning rate for policy network
        temperature_lr: Learning rate for temperature parameter
        tau: Target network soft update coefficient (0 < tau <= 1)
        gamma: Discount factor for future rewards (0 < gamma <= 1)
        alpha: If None, learn alpha, if float, use fixed alpha value
        n_quantiles: Number of quantiles per critic network
        n_critics: Number of critic networks in ensemble
        quantiles_drop: Number of quantiles to drop from truncated mixture
        seed: Random seed for reproducibility
        state: Optional pre-trained network state dictionary
    """

    __slots__ = [
        "action_dim",
        "batch_size",
        "buffer",
        "gamma",
        "key",
        "learn_temperature",
        "log_alpha",
        "n_critics",
        "n_quantiles",
        "policy_network",
        "policy_opt_state",
        "policy_optimizer",
        "policy_params",
        "quantiles_drop",
        "state_dim",
        "target_entropy",
        "tau",
        "temperature_opt_state",
        "temperature_optimizer",
        "z_network",
        "z_opt_state",
        "z_optimizer",
        "z_params",
        "z_target_params",
    ]

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        replay_size: int,
        batch_size: int,
        critic_lr: float,
        policy_lr: float,
        temperature_lr: float,
        tau: float,
        gamma: float,
        alpha: float | None,
        n_quantiles: int,
        n_critics: int,
        quantiles_drop: int,
        seed: int,
        state: dict | None = None,
    ) -> None:
        self.key = random.PRNGKey(seed)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_quantiles = n_quantiles
        self.n_critics = n_critics
        self.quantiles_drop = quantiles_drop
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.target_entropy = -float(action_dim)

        self.z_network = self.ZNetwork(
            hidden_dims=hidden_dims,
            n_quantiles=n_quantiles,
            n_critics=n_critics,
        )
        self.policy_network = self.PolicyNetwork(
            hidden_dims=hidden_dims, action_dim=action_dim
        )

        self.key, z_key, policy_key = random.split(self.key, 3)

        dummy_state = jnp.zeros((1, state_dim))  # [1, state_dim]
        dummy_action = jnp.zeros((1, action_dim))  # [1, action_dim]

        self.z_params = self.z_network.init(z_key, dummy_state, dummy_action)
        self.z_target_params = self.z_params
        self.policy_params = self.policy_network.init(policy_key, dummy_state)

        self.learn_temperature = alpha is None
        if self.learn_temperature:
            self.log_alpha = jnp.array(0, dtype=jnp.float32)  # []
        else:
            self.log_alpha = jnp.log(jnp.array(alpha, dtype=jnp.float32))  # []

        self.z_optimizer = optax.adam(critic_lr)
        self.policy_optimizer = optax.adam(policy_lr)

        self.z_opt_state = self.z_optimizer.init(self.z_params)
        self.policy_opt_state = self.policy_optimizer.init(self.policy_params)

        if self.learn_temperature:
            self.temperature_optimizer = optax.adam(temperature_lr)
            self.temperature_opt_state = self.temperature_optimizer.init(
                self.log_alpha
            )

        self.buffer = ReplayBuffer(replay_size, state_dim, action_dim)

        if state is not None:
            self.load_from_state(state)

    def select_action(self, state: np.ndarray, evaluation: bool) -> np.ndarray:
        """
        Selects an action in range [-1, 1] for the given state using either
        deterministic or stochastic sampling. Use this method to get actions
        from your trained agent during environment interaction.

        Args:
            state: Current environment state observation
            evaluation: Whether to use deterministic or stochastic
        """
        jax_state = state[None, ...]  # [1, state_dim]

        if evaluation:
            action = self._exploit(
                self.policy_params, jax_state
            )  # [1, action_dim]
        else:
            self.key, action_key = random.split(self.key)
            action = self._explore(
                self.policy_params, jax_state, action_key
            )  # [1, action_dim]

        return np.array(action[0])  # [action_dim]

    @partial(jax.jit, static_argnums=(0,))
    def _exploit(
        self, policy_params: dict[str, any], state: jnp.ndarray
    ) -> jnp.ndarray:
        """Select deterministic action using policy mean."""
        mean, _ = self.policy_network.apply(
            policy_params, state
        )  # [batch_size, action_dim]
        return jnp.tanh(mean)  # [batch_size, action_dim]

    @partial(jax.jit, static_argnums=(0,))
    def _explore(
        self,
        policy_params: dict[str, any],
        state: jnp.ndarray,
        key: jnp.ndarray,
    ) -> jnp.ndarray:
        """Sample stochastic action from policy distribution."""
        mean, log_std = self.policy_network.apply(
            policy_params, state
        )  # [batch_size, action_dim], [batch_size, action_dim]
        noise = random.normal(key, mean.shape)  # [batch_size, action_dim]
        raw_action = mean + noise * jnp.exp(log_std)  # [batch_size, action_dim]
        return jnp.tanh(raw_action)  # [batch_size, action_dim]

    def push_buffer(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Stores a transition tuple in the replay buffer for later training. Call
        this method after each environment step to build your training dataset.

        Args:
            state: Current environment state observation
            action: Action taken in the environment
            reward: Reward received from the environment
            next_state: Next state after taking the action
            done: Whether the episode terminated
        """
        self.buffer.push(state, action, reward, next_state, done)

    def update(self) -> None:
        """
        Performs one training step updating all networks using sampled batch
        from replay buffer. Call this method regularly during training to
        improve the agent's performance.
        """
        if len(self.buffer) < self.batch_size:
            return

        self.key, sample_key = random.split(self.key)
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size, sample_key
        )

        self.key, critic_key = random.split(self.key)
        self.z_params, self.z_opt_state = self._update_critic(
            self.z_params,
            self.z_opt_state,
            self.z_target_params,
            self.policy_params,
            self.log_alpha,
            states,
            actions,
            rewards,
            next_states,
            dones,
            critic_key,
        )

        self.key, policy_key = random.split(self.key)
        self.policy_params, self.policy_opt_state, mean_log_probs = (
            self._update_policy(
                self.policy_params,
                self.policy_opt_state,
                self.z_params,
                self.log_alpha,
                states,
                policy_key,
            )
        )

        if self.learn_temperature:
            self.log_alpha, self.temperature_opt_state = (
                self._update_temperature(
                    self.log_alpha, self.temperature_opt_state, mean_log_probs
                )
            )

        self.z_target_params = self._soft_update(
            self.z_target_params, self.z_params, self.tau
        )

    @partial(jax.jit, static_argnums=(0,))
    def _update_critic(
        self,
        z_params: dict[str, jax.Array],
        z_opt_state: optax.OptState,
        z_target_params: dict[str, jax.Array],
        policy_params: dict[str, jax.Array],
        log_alpha: jnp.ndarray,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        next_states: jnp.ndarray,
        dones: jnp.ndarray,
        key: jnp.ndarray,
    ) -> tuple[dict[str, jax.Array], optax.OptState]:
        """Update Z-network parameters using computed gradients."""

        z_grad = jax.grad(self._compute_z_loss)(
            z_params,
            z_target_params=z_target_params,
            policy_params=policy_params,
            log_alpha=log_alpha,
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            key=key,
        )

        z_updates, new_z_opt_state = self.z_optimizer.update(
            z_grad, z_opt_state
        )
        new_z_params = optax.apply_updates(z_params, z_updates)

        return new_z_params, new_z_opt_state

    @partial(jax.jit, static_argnums=(0,))
    def _compute_z_loss(
        self,
        z_params: dict[str, any],
        z_target_params: dict[str, any],
        policy_params: dict[str, any],
        log_alpha: jnp.ndarray,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        next_states: jnp.ndarray,
        dones: jnp.ndarray,
        key: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute Z-network loss using quantile regression with truncated targets.
        """

        z_targets = self._compute_z_targets(
            z_target_params,
            policy_params,
            log_alpha,
            rewards,
            next_states,
            dones,
            key,
        )  # [batch_size, n_kept_quantiles]
        z_values = self.z_network.apply(
            z_params, states, actions
        )  # [batch_size, n_critics, n_quantiles]

        return self._quantile_regression_loss(z_values, z_targets)  # []

    @partial(jax.jit, static_argnums=(0,))
    def _compute_z_targets(
        self,
        z_target_params: dict[str, any],
        policy_params: dict[str, any],
        log_alpha: jnp.ndarray,
        rewards: jnp.ndarray,
        next_states: jnp.ndarray,
        dones: jnp.ndarray,
        key: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute truncated quantile targets."""
        next_actions, next_log_probs = self._compute_action_and_log_prob(
            policy_params, next_states, key
        )  # [batch_size, action_dim], [batch_size, 1]

        z_next_values = self.z_network.apply(
            z_target_params, next_states, next_actions
        )  # [batch_size, n_critics, n_quantiles]
        truncated_quantiles = self._truncate(
            z_next_values
        )  # [batch_size, n_kept_quantiles]

        alpha = jax.lax.stop_gradient(jnp.exp(log_alpha))  # []
        targets = rewards.reshape(-1, 1) + self.gamma * (
            1.0 - dones.reshape(-1, 1)
        ) * (
            truncated_quantiles - alpha * next_log_probs
        )  # [batch_size, n_kept_quantiles]

        return jax.lax.stop_gradient(targets)

    @partial(jax.jit, static_argnums=(0,))
    def _truncate(self, quantiles: jnp.ndarray) -> jnp.ndarray:
        batch_size = quantiles.shape[0]
        all_quantiles = quantiles.reshape(
            batch_size, -1
        )  # [batch_size, n_critics * n_quantiles]
        sorted_quantiles = jnp.sort(
            all_quantiles, axis=1
        )  # [batch_size, n_critics * n_quantiles]

        n_drop = abs(self.quantiles_drop)
        n_total = self.n_critics * self.n_quantiles
        n_kept = n_total - n_drop

        return jax.lax.cond(
            self.quantiles_drop <= 0,
            lambda x: x[:, n_drop:],
            lambda x: x[:, :n_kept],
            sorted_quantiles,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _quantile_regression_loss(
        self, z_values: jnp.ndarray, z_targets: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute quantile regression with Huber loss."""
        tau = jnp.linspace(
            1 / (2 * self.n_quantiles),
            1 - 1 / (2 * self.n_quantiles),
            self.n_quantiles,
        )  # [n_quantiles]

        z_values_expanded = jnp.expand_dims(
            z_values, axis=-1
        )  # [batch_size, n_critics, n_quantiles, 1]
        z_targets_expanded = jnp.expand_dims(
            z_targets, axis=(1, 2)
        )  # [batch_size, 1, 1, n_kept_quantiles]
        tau_expanded = jnp.expand_dims(
            tau, axis=(0, 1, 3)
        )  # [1, 1, n_quantiles, 1]

        diff = (
            z_targets_expanded - z_values_expanded
        )  # [batch_size, n_critics, n_quantiles, n_kept_quantiles]
        huber_loss = self._huber_loss(
            diff, delta=1.0
        )  # [batch_size, n_critics, n_quantiles, n_kept_quantiles]
        weights = jnp.abs(
            tau_expanded - (diff < 0)
        )  # [batch_size, n_critics, n_quantiles, n_kept_quantiles]

        loss_per_quantile = (weights * huber_loss).sum(axis=-1)

        normalization = self.n_quantiles * z_targets.shape[-1]

        errors = loss_per_quantile / normalization
        return errors.mean()

    @partial(jax.jit, static_argnums=(0,))
    def _huber_loss(self, errors: jnp.ndarray, delta: float) -> jnp.ndarray:
        """Compute Huber loss."""
        abs_errors = jnp.abs(
            errors
        )  # [batch_size, n_critics, n_quantiles, n_kept_quantiles]

        return jnp.where(
            abs_errors <= delta,
            0.5 * jnp.square(errors),
            delta * abs_errors - 0.5 * delta**2,
        )  # [batch_size, n_critics, n_quantiles, n_kept_quantiles]

    @partial(jax.jit, static_argnums=(0,))
    def _update_policy(
        self,
        policy_params: dict[str, jax.Array],
        policy_opt_state: optax.OptState,
        z_params: dict[str, jax.Array],
        log_alpha: jnp.ndarray,
        states: jnp.ndarray,
        key: jnp.ndarray,
    ) -> tuple[dict[str, jax.Array], optax.OptState, jax.Array]:
        """Update policy network parameters using computed gradients."""
        policy_grad, mean_log_probs = jax.grad(
            self._compute_policy_loss, has_aux=True
        )(
            policy_params,
            z_params=z_params,
            log_alpha=log_alpha,
            states=states,
            key=key,
        )

        policy_updates, new_policy_opt_state = self.policy_optimizer.update(
            policy_grad, policy_opt_state
        )
        new_policy_params = optax.apply_updates(policy_params, policy_updates)

        return new_policy_params, new_policy_opt_state, mean_log_probs

    @partial(jax.jit, static_argnums=(0,))
    def _compute_policy_loss(
        self,
        policy_params: dict[str, any],
        z_params: dict[str, any],
        log_alpha: jnp.ndarray,
        states: jnp.ndarray,
        key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute policy loss using mean of all quantiles."""

        actions, log_probs = self._compute_action_and_log_prob(
            policy_params, states, key
        )  # [batch_size, action_dim], [batch_size, 1]

        z_values = self.z_network.apply(
            z_params, states, actions
        )  # [batch_size, n_critics, n_quantiles]
        q_values = z_values.mean(axis=(1, 2))  # [batch_size]

        alpha = jax.lax.stop_gradient(jnp.exp(log_alpha))  # []

        mean_log_probs = log_probs.mean()  # []
        mean_q_values = q_values.mean()  # []

        policy_loss = alpha * mean_log_probs - mean_q_values  # []

        return policy_loss, jax.lax.stop_gradient(mean_log_probs)

    @partial(jax.jit, static_argnums=(0,))
    def _update_temperature(
        self,
        log_alpha: jax.Array,
        temperature_opt_state: optax.OptState,
        mean_log_probs: jax.Array,
    ) -> tuple[jax.Array, optax.OptState]:
        """Update temperature parameter for entropy regularization."""

        temperature_grad = jax.grad(self._compute_temperature_loss)(
            log_alpha, mean_log_probs
        )

        temperature_updates, new_temperature_opt_state = (
            self.temperature_optimizer.update(
                temperature_grad, temperature_opt_state
            )
        )
        new_log_alpha = optax.apply_updates(log_alpha, temperature_updates)

        return new_log_alpha, new_temperature_opt_state

    @partial(jax.jit, static_argnums=(0,))
    def _compute_temperature_loss(
        self,
        log_alpha: jnp.ndarray,
        mean_log_probs: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute temperature loss for automatic entropy tuning."""

        return -log_alpha * (self.target_entropy + mean_log_probs)  # []

    @partial(jax.jit, static_argnums=(0,))
    def _compute_action_and_log_prob(
        self,
        policy_params: dict[str, any],
        states: jnp.ndarray,
        key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample actions and compute log probabilities."""
        means, log_stds = self.policy_network.apply(
            policy_params, states
        )  # [batch_size, action_dim], [batch_size, action_dim]

        noise = random.normal(key, means.shape)  # [batch_size, action_dim]
        raw_actions = means + noise * jnp.exp(
            log_stds
        )  # [batch_size, action_dim]

        actions = jnp.tanh(raw_actions)  # [batch_size, action_dim]

        gaussian_log_probs = -0.5 * (
            jnp.square(noise) + 2 * log_stds + jnp.log(2 * math.pi)
        )  # [batch_size, action_dim]

        tanh_corrections = jnp.log(
            nn.relu(1.0 - jnp.square(actions)) + 1e-6
        )  # [batch_size, action_dim]
        log_probs = (gaussian_log_probs - tanh_corrections).sum(
            axis=1, keepdims=True
        )  # [batch_size, 1]

        return actions, log_probs

    @partial(jax.jit, static_argnums=(0,))
    def _soft_update(
        self,
        target_params: dict[str, any],
        online_params: dict[str, any],
        tau: float,
    ) -> dict[str, any]:
        """Perform soft update of target network parameters."""
        return jax.tree.map(
            lambda t, s: (1 - tau) * t + tau * s, target_params, online_params
        )

    def get_state(self) -> dict[str, any]:
        """
        Returns a complete agent state dictionary including all network
        parameters and optimizer states. Use this method to save your trained
        agent for later use or checkpointing.
        """
        state = {
            "z_params": self.z_params,
            "z_target_params": self.z_target_params,
            "policy_params": self.policy_params,
            "log_alpha": self.log_alpha,
            "z_opt_state": self.z_opt_state,
            "policy_opt_state": self.policy_opt_state,
            "key": self.key,
            "replay_buffer": self.buffer.get_data(),
        }

        if self.learn_temperature:
            state["temperature_opt_state"] = self.temperature_opt_state

        return state

    def load_from_state(self, state: dict[str, any]) -> None:
        """
        Loads complete agent state from a previously saved state dictionary. Use
        this method to restore a trained agent or continue training from a
        checkpoint.

        Args:
            state: Dictionary containing saved agent parameters and states
        """
        self.z_params = state["z_params"]
        self.z_target_params = state["z_target_params"]
        self.policy_params = state["policy_params"]
        self.log_alpha = state["log_alpha"]
        self.z_opt_state = state["z_opt_state"]
        self.policy_opt_state = state["policy_opt_state"]
        self.key = state["key"]
        self.buffer.load_data(state["replay_buffer"])

        if self.learn_temperature and "temperature_opt_state" in state:
            self.temperature_opt_state = state["temperature_opt_state"]

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

    class PolicyNetwork(nn.Module):
        """
        Policy network that outputs action mean and log standard deviation.
        """

        hidden_dims: list[int]
        action_dim: int

        @nn.compact
        def __call__(
            self, states: jnp.ndarray
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            output = MLP(self.hidden_dims, 2 * self.action_dim)(states)
            mean, log_std = jnp.split(output, 2, axis=1)
            log_std = jnp.clip(log_std, -10, 2)
            return mean, log_std
