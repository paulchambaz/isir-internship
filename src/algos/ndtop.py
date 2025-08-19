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

from .networks import PolicyNetwork, QNetwork
from .replay import ReplayBuffer
from .rl_algo import RLAlgo
from .utils import se_loss, soft_update


class NDTOP(RLAlgo):
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
        n_critics: int,
        beta: float,
        seed: int,
        state: dict | None = None,
    ) -> None:
        self.key = random.PRNGKey(seed)

        self.state_dim = state_dim
        self.action_dimm = action_dim
        self.n_critics = n_critics
        self.beta = beta
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.target_entropy = -float(action_dim)

        self.q_network = QNetwork(
            hidden_dims=hidden_dims,
            num_critics=n_critics,
        )
        self.policy_network = PolicyNetwork(
            hidden_dims=hidden_dims, action_dim=action_dim
        )

        self.key, q_key, policy_key = random.split(self.key, 3)

        dummy_state = jnp.zeros((1, state_dim))
        dummy_action = jnp.zeros((1, action_dim))

        self.q_params = self.q_network.init(q_key, dummy_state, dummy_action)
        self.q_target_params = self.q_params
        self.policy_params = self.policy_network.init(policy_key, dummy_state)

        self.learn_temperature = alpha is None
        if self.learn_temperature:
            self.log_alpha = jnp.array(0, dtype=jnp.float32)
        else:
            self.log_alpha = jnp.log(jnp.array(alpha, dtype=jnp.float32))

        self.q_optimizer = optax.adam(critic_lr)
        self.policy_optimizer = optax.adam(policy_lr)

        self.q_opt_state = self.q_optimizer.init(self.q_params)
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
        jax_state = state[None, ...]

        if evaluation:
            action = self._exploit(self.policy_params, jax_state)
        else:
            self.key, action_key = random.split(self.key)
            action = self._explore(self.policy_params, jax_state, action_key)

        return np.array(action[0])

    @partial(jax.jit, static_argnums=(0,))
    def _exploit(
        self, policy_params: dict[str, any], state: jnp.ndarray
    ) -> jnp.ndarray:
        """Select deterministic action using policy mean."""
        mean, _ = self.policy_network.apply(policy_params, state)
        return jnp.tanh(mean)

    @partial(jax.jit, static_argnums=(0,))
    def _explore(
        self,
        policy_params: dict[str, any],
        state: jnp.ndarray,
        key: jnp.ndarray,
    ) -> jnp.ndarray:
        """Sample stochastic action from policy distribution."""
        mean, log_std = self.policy_network.apply(policy_params, state)
        noise = random.normal(key, mean.shape)
        raw_action = mean + noise * jnp.exp(log_std)
        return jnp.tanh(raw_action)

    def evaluate(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Evaluates the Q-values for a given state-action pair using the current
        critic network(s). Use this method to assess how valuable the critic
        considers a specific action in a given state.

        Args:
            state: Current environment state observation
            action: Action to evaluate in the given state
        """
        jax_state = state[None, ...]
        jax_action = action[None, ...]
        value = self._compute_critic(self.q_params, jax_state, jax_action)
        return value.squeeze(-1).squeeze(-1)

    @partial(jax.jit, static_argnums=(0,))
    def _compute_critic(
        self, q_params: dict, states: np.ndarray, actions: np.ndarray
    ) -> np.ndarray:
        q_values = self.q_network.apply(q_params, states, actions)

        mean_q = jnp.mean(q_values, axis=1)
        std_q = jnp.mean(q_values, axis=1)
        belief_q = mean_q + self.beta * std_q

        return jnp.mean(belief_q, axis=0, keepdims=True)

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
        self.q_params, self.q_opt_state = self._update_critic(
            self.q_params,
            self.q_opt_state,
            self.q_target_params,
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
                self.q_params,
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

        self.q_target_params = soft_update(
            self.q_target_params, self.q_params, self.tau
        )

    @partial(jax.jit, static_argnums=(0,))
    def _update_critic(
        self,
        q_params: dict[str, jax.Array],
        q_opt_state: optax.OptState,
        q_target_params: dict[str, jax.Array],
        policy_params: dict[str, jax.Array],
        log_alpha: jnp.ndarray,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        next_states: jnp.ndarray,
        dones: jnp.ndarray,
        key: jnp.ndarray,
    ) -> tuple[dict[str, jax.Array], optax.OptState]:
        """Update Q-network parameters using computed gradients."""

        q_grad = jax.grad(self._compute_critic_loss)(
            q_params,
            q_target_params=q_target_params,
            policy_params=policy_params,
            log_alpha=log_alpha,
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            key=key,
        )

        q_updates, new_q_opt_state = self.q_optimizer.update(
            q_grad, q_opt_state
        )
        new_q_params = optax.apply_updates(q_params, q_updates)

        return new_q_params, new_q_opt_state

    @partial(jax.jit, static_argnums=(0,))
    def _compute_critic_loss(
        self,
        q_params: dict[str, any],
        q_target_params: dict[str, any],
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
        Compute Q-network loss using quantile regression with truncated targets.
        """

        targets = self._compute_targets(
            q_target_params,
            policy_params,
            log_alpha,
            rewards,
            next_states,
            dones,
            key,
        )

        values = self.q_network.apply(q_params, states, actions)
        errors = se_loss(value=values, target=targets)
        return jnp.mean(errors)

    @partial(jax.jit, static_argnums=(0,))
    def _compute_targets(
        self,
        q_target_params: dict[str, any],
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
        )

        q_values = self.q_network.apply(
            q_target_params, next_states, next_actions
        )

        mean_q = jnp.mean(q_values, axis=0)
        std_q = jnp.std(q_values, axis=0)
        belief_q = mean_q + self.beta * std_q

        alpha = jax.lax.stop_gradient(jnp.exp(log_alpha))
        target = rewards + self.gamma * (1.0 - dones) * (
            belief_q - alpha * next_log_probs
        )

        return jax.lax.stop_gradient(target)

    @partial(jax.jit, static_argnums=(0,))
    def _update_policy(
        self,
        policy_params: dict[str, jax.Array],
        policy_opt_state: optax.OptState,
        q_params: dict[str, jax.Array],
        log_alpha: jnp.ndarray,
        states: jnp.ndarray,
        key: jnp.ndarray,
    ) -> tuple[dict[str, jax.Array], optax.OptState, jax.Array]:
        """Update policy network parameters using computed gradients."""
        policy_grad, mean_log_probs = jax.grad(
            self._compute_policy_loss, has_aux=True
        )(
            policy_params,
            q_params=q_params,
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
        q_params: dict[str, any],
        log_alpha: jnp.ndarray,
        states: jnp.ndarray,
        key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute policy loss using mean of all quantiles."""

        actions, log_probs = self._compute_action_and_log_prob(
            policy_params, states, key
        )

        q_values = self.q_network.apply(q_params, states, actions)
        q_values = q_values.mean(axis=(1, 2))

        alpha = jax.lax.stop_gradient(jnp.exp(log_alpha))

        mean_log_probs = log_probs.mean()
        mean_q_values = q_values.mean()

        policy_loss = alpha * mean_log_probs - mean_q_values

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

        return -log_alpha * (self.target_entropy + mean_log_probs)

    @partial(jax.jit, static_argnums=(0,))
    def _compute_action_and_log_prob(
        self,
        policy_params: dict[str, any],
        states: jnp.ndarray,
        key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample actions and compute log probabilities."""
        means, log_stds = self.policy_network.apply(policy_params, states)

        noise = random.normal(key, means.shape)
        raw_actions = means + noise * jnp.exp(log_stds)

        actions = jnp.tanh(raw_actions)

        gaussian_log_probs = -0.5 * (
            jnp.square(noise) + 2 * log_stds + jnp.log(2 * math.pi)
        )

        tanh_corrections = jnp.log(nn.relu(1.0 - jnp.square(actions)) + 1e-6)
        log_probs = (gaussian_log_probs - tanh_corrections).sum(
            axis=1, keepdims=True
        )

        return actions, log_probs

    def get_state(self) -> dict[str, any]:
        """
        Returns a complete agent state dictionary including all network
        parameters and optimizer states.
        """
        state = {
            "q_params": self.q_params,
            "q_target_params": self.q_target_params,
            "policy_params": self.policy_params,
            "log_alpha": self.log_alpha,
            "q_opt_state": self.q_opt_state,
            "policy_opt_state": self.policy_opt_state,
            "key": self.key,
            "replay_buffer": self.buffer.get_data(),
        }

        if self.learn_temperature:
            state["temperature_opt_state"] = self.temperature_opt_state

        return state

    def load_from_state(self, state: dict[str, any]) -> None:
        """
        Loads complete agent state from a previously saved state dictionary.
        """
        self.q_params = state["q_params"]
        self.q_target_params = state["q_target_params"]
        self.policy_params = state["policy_params"]
        self.log_alpha = state["log_alpha"]
        self.q_opt_state = state["q_opt_state"]
        self.policy_opt_state = state["policy_opt_state"]
        self.key = state["key"]
        self.buffer.load_data(state["replay_buffer"])

        if self.learn_temperature and "temperature_opt_state" in state:
            self.temperature_opt_state = state["temperature_opt_state"]
