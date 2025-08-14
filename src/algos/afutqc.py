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

from .networks import PolicyNetwork, QNetwork, VNetwork
from .replay import ReplayBuffer
from .rl_algo import RLAlgo
from .utils import lerp, se_loss, soft_update, where


class AFUTQC(RLAlgo):
    """
    Actor-Free critic Updates (AFU) algorithm for continuous control tasks.

    AFU uses a V function for state evaluation with Q=V+A decomposition, where
    A represents advantage/regret values. It employs conditional gradient
    rescaling to maintain V as a tight bound of max Q and conditional
    optimization to enforce constraints on A values, while learning a
    stochastic policy that maximizes expected return and entropy.

    Args:
        state_dim: Dimensionality of the state space
        state_dim: Dimensionality of the state space
        hidden_dims: List of hidden layer sizes for neural networks
        replay_size: Maximum size of the replay buffer
        batch_size: Number of samples per training batch
        critic_lr: Learning rate for Q, V, and A function networks
        policy_lr: Learning rate for policy network
        temperature_lr: Learning rate for temperature parameter
        tau: Target network soft update coefficient (0 < tau <= 1)
        gamma: Discount factor for future rewards (0 < gamma <= 1)
        alpha: If None, learn alpha, if float, use fixed alpha value
        rho: Conditional gradient rescaling coefficient for V-A constraints
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
        "rho",
        "state_dim",
        "target_entropy",
        "tau",
        "temperature_opt_state",
        "temperature_optimizer",
        "v_network",
        "v_opt_state",
        "v_optimizer",
        "v_params",
        "v_target_params",
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
        rho: float,
        n_quantiles: int,
        quantiles_drop: int,
        seed: int,
        state: dict[str, any] | None = None,
    ) -> None:
        self.key = random.PRNGKey(seed)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.tau = tau
        self.rho = rho
        self.gamma = gamma
        self.target_entropy = -float(action_dim)

        self.n_quantiles = n_quantiles
        self.quantiles_drop = quantiles_drop

        self.q_network = QNetwork(
            hidden_dims=hidden_dims, output_dim=self.n_quantiles + 1
        )
        self.v_network = VNetwork(hidden_dims=hidden_dims)
        self.policy_network = PolicyNetwork(
            hidden_dims=hidden_dims, action_dim=action_dim
        )

        self.key, q_key, v_key, policy_key = random.split(self.key, 4)

        dummy_state = jnp.zeros((1, state_dim))
        dummy_action = jnp.zeros((1, action_dim))

        self.q_params = self.q_network.init(q_key, dummy_state, dummy_action)
        self.q_target_params = self.q_params
        self.v_params = self.v_network.init(v_key, dummy_state)
        self.v_target_params = self.v_params
        self.policy_params = self.policy_network.init(policy_key, dummy_state)

        self.learn_temperature = alpha is None
        if self.learn_temperature:
            self.log_alpha = jnp.array(0, dtype=jnp.float32)
        else:
            self.log_alpha = jnp.log(jnp.array(alpha, dtype=jnp.float32))

        self.q_optimizer = optax.adam(critic_lr)
        self.v_optimizer = optax.adam(critic_lr)
        self.policy_optimizer = optax.adam(policy_lr)

        self.q_opt_state = self.q_optimizer.init(self.q_params)
        self.v_opt_state = self.v_optimizer.init(self.v_params)
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

        self.q_params, self.v_params, self.q_opt_state, self.v_opt_state = (
            self._update_critic_and_value(
                self.q_params,
                self.v_params,
                self.q_opt_state,
                self.v_opt_state,
                self.q_target_params,
                self.v_target_params,
                states,
                actions,
                rewards,
                next_states,
                dones,
            )
        )

        self.key, policy_key = random.split(self.key)
        self.policy_params, self.policy_opt_state, mean_log_probs = (
            self._update_policy(
                self.policy_params,
                self.policy_opt_state,
                self.q_params,
                self.v_params,
                self.log_alpha,
                states,
                policy_key,
            )
        )

        if self.learn_temperature:
            self.log_alpha, self.temperature_opt_state = (
                self._update_temperature(
                    self.log_alpha,
                    self.temperature_opt_state,
                    mean_log_probs,
                )
            )

        self.v_target_params = soft_update(
            self.v_target_params, self.v_params, self.tau
        )
        self.q_target_params = soft_update(
            self.q_target_params, self.q_params, self.tau
        )

    @partial(jax.jit, static_argnums=(0,))
    def _update_critic_and_value(
        self,
        q_params: dict[str, jax.Array],
        v_params: dict[str, jax.Array],
        q_opt_state: optax.OptState,
        v_opt_state: optax.OptState,
        q_target_params: dict[str, jax.Array],
        v_target_params: dict[str, jax.Array],
        states: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        next_states: jnp.ndarray,
        dones: jnp.ndarray,
    ) -> tuple[
        dict[str, jax.Array],
        dict[str, jax.Array],
        optax.OptState,
        optax.OptState,
    ]:
        """Update Q and V network parameters using computed gradients."""

        (q_grad, v_grad) = jax.grad(self._compute_critic_loss, argnums=(0, 1))(
            q_params,
            v_params,
            q_target_params,
            v_target_params,
            states,
            actions,
            rewards,
            next_states,
            dones,
        )

        q_updates, new_q_opt_state = self.q_optimizer.update(
            q_grad, q_opt_state
        )
        new_q_params = optax.apply_updates(q_params, q_updates)

        v_updates, new_v_opt_state = self.v_optimizer.update(
            v_grad, v_opt_state
        )
        new_v_params = optax.apply_updates(v_params, v_updates)

        return new_q_params, new_v_params, new_q_opt_state, new_v_opt_state

    @partial(jax.jit, static_argnums=(0,))
    def _compute_critic_loss(
        self,
        q_params: dict[str, any],
        v_params: dict[str, any],
        q_target_params: dict[str, any],
        v_target_params: dict[str, any],
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> jnp.ndarray:
        """Compute combined loss for Q and V networks with V-A constraints."""

        q_values_list = self.q_network.apply(q_params, states, actions)
        q_values = q_values_list[..., 1:]
        a_values = -q_values_list[..., :1]
        v_values = self.v_network.apply(v_params, states)

        q_loss = self._compute_q_loss(
            v_target_params, q_values, rewards, next_states, dones
        )

        va_loss = self._compute_va_loss(
            q_target_params, v_values, a_values, states, actions
        )

        return va_loss + q_loss

    @partial(jax.jit, static_argnums=(0,))
    def _compute_va_loss(
        self,
        q_target_params: dict[str, any],
        v_values: jnp.ndarray,
        a_values: jnp.ndarray,
        states: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> None:
        q_targets_list = self.q_network.apply(q_target_params, states, actions)
        q_quantiles = jax.lax.stop_gradient(q_targets_list[..., 1:])
        q_truncated_quantiles = self._truncate(q_quantiles, self.quantiles_drop)
        q_targets = q_truncated_quantiles.mean(axis=1, keepdims=True)

        # applies downward pressure to value gradient
        upsilon_values = where(
            jax.lax.stop_gradient(v_values + a_values < q_targets),
            # if value is under the target apply the full gradient
            v_values,
            # if value is over the target apply a reduced gradient
            lerp(self.rho, v_values, jax.lax.stop_gradient(v_values)),
        )

        # ensure a is negative
        z_values = where(
            jax.lax.stop_gradient(v_values < q_targets),
            # if value is under the target apply, apply standard loss
            se_loss(value=upsilon_values + a_values, target=q_targets),
            # if value is over over target, then a would become positive
            # after gradient application, set V to Q and A to 0, so that
            # V + A = Q + 0, and A < 0
            se_loss(value=upsilon_values, target=q_targets)
            + se_loss(value=a_values, target=0),
        )

        return z_values.mean()

    @partial(jax.jit, static_argnums=(0,))
    def _truncate(self, quantiles: jnp.ndarray, drop: int) -> jnp.ndarray:
        n_drop = abs(drop)
        n_kept = self.n_quantiles - n_drop

        sorted_quantiles = jnp.sort(quantiles, axis=1)

        return jax.lax.cond(
            drop <= 0,
            lambda q: q[:, n_drop:],
            lambda q: q[:, :n_kept],
            sorted_quantiles,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _compute_q_loss(
        self,
        v_target_params: dict[str, any],
        q_values: jnp.ndarray,
        rewards: jnp.ndarray,
        next_states: jnp.ndarray,
        dones: jnp.ndarray,
    ) -> None:
        v_next_targets = self.v_network.apply(v_target_params, next_states)
        q_targets = jax.lax.stop_gradient(
            rewards + self.gamma * (1.0 - dones) * v_next_targets
        )

        errors = se_loss(value=q_values, target=q_targets)
        return errors.mean()

    @partial(jax.jit, static_argnums=(0,))
    def _update_policy(
        self,
        policy_params: dict[str, jax.Array],
        policy_opt_state: optax.OptState,
        q_params: dict[str, jax.Array],
        v_params: dict[str, jax.Array],
        log_alpha: jnp.ndarray,
        states: jnp.ndarray,
        key: jnp.ndarray,
    ) -> tuple[dict[str, jax.Array], optax.OptState, jax.Array]:
        """Update policy network parameters using computed gradients."""

        (_, mean_log_probs), policy_grad = jax.value_and_grad(
            self._compute_policy_loss, has_aux=True
        )(
            policy_params,
            q_params=q_params,
            v_params=v_params,
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
        v_params: dict[str, any],
        log_alpha: jnp.ndarray,
        states: np.ndarray,
        key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute policy loss combining Q-values and entropy regularization."""

        means, log_std = self.policy_network.apply(policy_params, states)

        noises = random.normal(key, means.shape)
        raw_actions = means + noises * jnp.exp(log_std)
        actions = jnp.tanh(raw_actions)

        gaussian_log_probs = -0.5 * (
            jnp.square(noises) + 2 * log_std + jnp.log(2 * math.pi)
        )
        tanh_corrections = jnp.log(nn.relu(1.0 - jnp.square(actions)) + 1e-6)

        log_probs = (gaussian_log_probs - tanh_corrections).sum(
            axis=1, keepdims=True
        )

        q_values = self.q_network.apply(q_params, states, actions)[-1]

        mean_log_probs = log_probs.mean()
        mean_q_values = q_values.mean()

        alpha = jax.lax.stop_gradient(jnp.exp(log_alpha))
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

    def get_state(self) -> dict[str, any]:
        """
        Returns a complete agent state dictionary including all network
        parameters and optimizer states. Use this method to save your trained
        agent for later use or checkpointing.
        """
        state = {
            "q_params": self.q_params,
            "v_params": self.v_params,
            "v_target_params": self.v_target_params,
            "policy_params": self.policy_params,
            "log_alpha": self.log_alpha,
            "q_opt_state": self.q_opt_state,
            "v_opt_state": self.v_opt_state,
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
        self.q_params = state["q_params"]
        self.v_params = state["v_params"]
        self.v_target_params = state["v_target_params"]
        self.policy_params = state["policy_params"]
        self.log_alpha = state["log_alpha"]
        self.q_opt_state = state["q_opt_state"]
        self.v_opt_state = state["v_opt_state"]
        self.policy_opt_state = state["policy_opt_state"]
        self.key = state["key"]
        self.buffer.load_data(state["replay_buffer"])

        if self.learn_temperature and "temperature_opt_state" in state:
            self.temperature_opt_state = state["temperature_opt_state"]
