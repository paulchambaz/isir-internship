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


class SAC(RLAlgo):
    """
    Soft Actor-Critic (SAC) algorithm for continuous control tasks.

    SAC is an off-policy actor-critic method based on the maximum entropy
    reinforcement learning framework. It learns a stochastic policy that
    maximizes both expected return and entropy to improve exploration and
    robustness.

    Args:
        state_dim: Dimensionality of the state space
        action_dim: Dimensionality of the action space
        hidden_dims: List of hidden layer sizes for neural networks
        replay_size: Maximum size of the replay buffer
        batch_size: Number of samples per training batch
        critic_lr: Learning rate for Q-function networks
        policy_lr: Learning rate for policy network
        temperature_lr: Learning rate for temperature parameter
        tau: Target network soft update coefficient (0 < tau <= 1)
        gamma: Discount factor for future rewards (0 < gamma <= 1)
        alpha: If None, learn alpha, if float, use fixed alpha value
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
        "policy_network",
        "policy_opt_state",
        "policy_optimizer",
        "policy_params",
        "q_network",
        "q_opt_state",
        "q_optimizer",
        "q_params",
        "q_target_params",
        "state_dim",
        "target_entropy",
        "tau",
        "temperature_opt_state",
        "temperature_optimizer",
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
        seed: int,
        state: dict | None = None,
    ) -> None:
        self.key = random.PRNGKey(seed)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.target_entropy = -float(action_dim)

        self.q_network = self.QNetwork(hidden_dims=hidden_dims, num_critics=2)
        self.policy_network = self.PolicyNetwork(
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

    @partial(jax.jit, static_argnums=0)
    def _exploit(
        self,
        policy_params: dict[str, any],
        state: jnp.ndarray,
    ) -> jnp.ndarray:
        """Select deterministic action using policy mean."""
        mean, _ = self.policy_network.apply(policy_params, state)
        return jnp.tanh(mean)

    @partial(jax.jit, static_argnums=0)
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

        (
            self.q_params,
            self.q_opt_state,
        ) = self._update_critic(
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
        )

        self.key, policy_key = random.split(self.key)
        (
            self.policy_params,
            self.policy_opt_state,
            mean_log_probs,
        ) = self._update_policy(
            self.policy_params,
            self.policy_opt_state,
            self.q_params,
            self.log_alpha,
            states,
            policy_key,
        )

        if self.learn_temperature:
            (
                self.log_alpha,
                self.temperature_opt_state,
            ) = self._update_temperature(
                self.log_alpha,
                self.temperature_opt_state,
                mean_log_probs,
            )

        self.q_target_params = self._soft_update(
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
    ) -> tuple[dict[str, jax.Array], optax.OptState]:
        """Update Q-network parameters using computed gradients."""

        _, q_grad = jax.value_and_grad(self._compute_q_loss)(
            q_params,
            q_target_params=q_target_params,
            policy_params=policy_params,
            log_alpha=log_alpha,
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
        )

        q_updates, new_q_opt_state = self.q_optimizer.update(
            q_grad, q_opt_state
        )
        new_q_params = optax.apply_updates(q_params, q_updates)

        return new_q_params, new_q_opt_state

    @partial(jax.jit, static_argnums=(0,))
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
            return [
                MLP(hidden_dims=self.hidden_dims, output_dim=1)(states_actions)
                for _ in range(self.num_critics)
            ]

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
