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

from .rl_algo import RLAlgo


class MLP(nn.Module):
    hidden_dims: list[int]
    output_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for hidden_dim in self.hidden_dims:
            x = jax.nn.relu(
                nn.Dense(
                    hidden_dim,
                    kernel_init=nn.initializers.orthogonal(scale=math.sqrt(2)),
                )(x)
            )

        return nn.Dense(
            self.output_dim, kernel_init=nn.initializers.orthogonal(scale=1)
        )(x)


class QNetwork(nn.Module):
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


class VNetwork(nn.Module):
    hidden_dims: list[int]
    num_critics: int

    @nn.compact
    def __call__(self, states: jnp.ndarray) -> list[jnp.ndarray]:
        return [
            MLP(hidden_dims=self.hidden_dims, output_dim=1)(states)
            for _ in range(self.num_critics)
        ]


class PolicyNetwork(nn.Module):
    hidden_dims: list[int]
    action_dim: int

    @nn.compact
    def __call__(self, states: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        output = MLP(self.hidden_dims, 2 * self.action_dim)(states)
        mean, log_std = jnp.split(output, 2, axis=1)
        log_std = jnp.clip(log_std, -10, 2)
        return mean, log_std


class ReplayBuffer:
    def __init__(
        self, buffer_size: int, state_dim: int, action_dim: int
    ) -> None:
        self.n = 0
        self.p = 0
        self.buffer_size = buffer_size

        self.state = np.empty((buffer_size, state_dim), dtype=np.float32)
        self.action = np.empty((buffer_size, action_dim), dtype=np.float32)
        self.reward = np.empty((buffer_size, 1), dtype=np.float32)
        self.done = np.empty((buffer_size, 1), dtype=np.float32)
        self.next_state = np.empty((buffer_size, state_dim), dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.state[self.p] = state
        self.action[self.p] = action
        self.reward[self.p] = float(reward)
        self.next_state[self.p] = next_state
        self.done[self.p] = float(done)

        self.p = (self.p + 1) % self.buffer_size
        self.n = min(self.n + 1, self.buffer_size)

    def sample(self, batch_size: int, key: jnp.ndarray) -> tuple:
        idxes = random.randint(
            key, shape=(batch_size,), minval=0, maxval=self.n
        )
        return (
            self.state[idxes],
            self.action[idxes],
            self.reward[idxes],
            self.next_state[idxes],
            self.done[idxes],
        )

    def __len__(self) -> int:
        return self.n

    def get_data(self) -> dict[str, any]:
        """Get buffer data for saving."""
        return {
            "state": self.state[: self.n].copy(),
            "action": self.action[: self.n].copy(),
            "reward": self.reward[: self.n].copy(),
            "next_state": self.next_state[: self.n].copy(),
            "done": self.done[: self.n].copy(),
            "n": self.n,
            "p": self.p,
        }

    def load_data(self, data: dict[str, any]) -> None:
        """Load buffer data from saved state."""
        self.n = data["n"]
        self.p = data["p"]
        self.state[: self.n] = data["state"]
        self.action[: self.n] = data["action"]
        self.reward[: self.n] = data["reward"]
        self.next_state[: self.n] = data["next_state"]
        self.done[: self.n] = data["done"]


class AFU(RLAlgo):
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

        self.q_network = QNetwork(hidden_dims=hidden_dims, num_critics=3)
        self.v_network = VNetwork(hidden_dims=hidden_dims, num_critics=2)
        self.policy_network = PolicyNetwork(
            hidden_dims=hidden_dims, action_dim=action_dim
        )

        self.key, q_key, v_key, policy_key = random.split(self.key, 4)

        dummy_state = jnp.zeros((1, state_dim))
        dummy_action = jnp.zeros((1, action_dim))

        self.q_params = self.q_network.init(q_key, dummy_state, dummy_action)
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
        else:
            self.temperature_optimizer = None
            self.temperature_opt_state = None

        self.buffer = ReplayBuffer(replay_size, state_dim, action_dim)

        if state is not None:
            self.load_from_state(state)

    def select_action(self, state: np.ndarray, evaluation: bool) -> np.ndarray:
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
        mean, _ = self.policy_network.apply(policy_params, state)
        return jnp.tanh(mean)

    @partial(jax.jit, static_argnums=0)
    def _explore(
        self,
        policy_params: dict[str, any],
        state: jnp.ndarray,
        key: jnp.ndarray,
    ) -> jnp.ndarray:
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
        self.buffer.push(state, action, reward, next_state, done)

    def update(self) -> None:
        if len(self.buffer) < self.batch_size:
            return

        self.key, sample_key = random.split(self.key)
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size, sample_key
        )

        (
            self.q_params,
            self.v_params,
            self.q_opt_state,
            self.v_opt_state,
        ) = self._update_critic_and_value(
            self.q_params,
            self.v_params,
            self.q_opt_state,
            self.v_opt_state,
            self.v_target_params,
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
        ) = self._update_actor_policy(
            self.policy_params,
            self.policy_opt_state,
            self.q_params,
            self.v_params,
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

        self.v_target_params = self._soft_update(
            self.v_target_params, self.v_params, self.tau
        )

    @partial(jax.jit, static_argunms=(0,))
    def _update_critic_and_value(
        self,
        q_params: dict[str, jax.Array],
        v_params: dict[str, jax.Array],
        q_opt_state: optax.OptState,
        v_opt_state: optax.OptState,
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
        _, (q_grad, v_grad) = jax.value_and_grad(
            self._compute_critic_loss, argnums=(0, 1), has_aux=False
        )(
            self.q_params,
            self.v_params,
            v_target_params=self.v_target_params,
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

        v_updates, new_v_opt_state = self.v_optimizer.update(
            v_grad, v_opt_state
        )
        new_v_params = optax.apply_updates(v_params, v_updates)

        return new_q_params, new_v_params, new_q_opt_state, new_v_opt_state

    @partial(jax.jit, static_argnums=(0,))
    def _update_actor_policy(
        self,
        policy_params: dict[str, jax.Array],
        policy_opt_state: optax.OptState,
        q_params: dict[str, jax.Array],
        v_params: dict[str, jax.Array],
        log_alpha: jnp.ndarray,
        states: jnp.ndarray,
        key: jnp.ndarray,
    ) -> tuple[dict[str, jax.Array], optax.OptState, jax.Array]:
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
    def _update_temperature(
        self,
        log_alpha: jax.Array,
        temperature_opt_state: optax.OptState,
        mean_log_probs: jax.Array,
    ) -> tuple[jax.Array, optax.OptState]:
        _, temperature_grad = jax.value_and_grad(
            self._compute_temperature_loss
        )(log_alpha, mean_log_probs)

        temperature_updates, new_temperature_opt_state = (
            self.temperature_optimizer.update(
                temperature_grad, temperature_opt_state
            )
        )
        new_log_alpha = optax.apply_updates(log_alpha, temperature_updates)

        return new_log_alpha, new_temperature_opt_state

    @partial(jax.jit, static_argnums=0)
    def _compute_critic_loss(
        self,
        q_params: dict[str, any],
        v_params: dict[str, any],
        v_target_params: dict[str, any],
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> jnp.ndarray:
        v_targets_list = jnp.asarray(
            self.v_network.apply(v_target_params, next_states)
        )
        v_targets = jnp.min(v_targets_list, axis=0)

        q_targets = jax.lax.stop_gradient(
            rewards + self.gamma * (1.0 - dones) * v_targets
        )

        v_values = jnp.asarray(self.v_network.apply(v_params, states))

        q_values_list = self.q_network.apply(q_params, states, actions)
        q_values = jnp.asarray(q_values_list[-1:])

        abs_td = jnp.abs(q_targets - q_values)
        q_loss = (abs_td**2).mean()

        a_values = -jnp.asarray(q_values_list[:-1])

        mix_case = jax.lax.stop_gradient(v_values + a_values < q_targets)
        upsilon_values = (
            1 - self.rho * mix_case
        ) * v_values + self.rho * mix_case * jax.lax.stop_gradient(v_values)

        up_case = jax.lax.stop_gradient(v_values >= q_targets)
        z_values = (
            a_values**2
            + up_case * 2 * a_values * (upsilon_values - q_targets)
            + (upsilon_values - q_targets) ** 2
        )

        va_loss = z_values.mean()

        return va_loss + q_loss

    @partial(jax.jit, static_argnums=0)
    def _compute_policy_loss(
        self,
        policy_params: dict[str, any],
        q_params: dict[str, any],
        v_params: dict[str, any],
        log_alpha: jnp.ndarray,
        states: np.ndarray,
        key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        means, log_std = self.policy_network.apply(policy_params, states)

        noise = jax.random.normal(key, means.shape)
        raw_action = means + noise * jnp.exp(log_std)
        actions = jnp.tanh(raw_action)

        gaussian_log_probs = -0.5 * (
            jnp.square(noise) + 2 * log_std + jnp.log(2 * math.pi)
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

    @partial(jax.jit, static_argnums=0)
    def _compute_temperature_loss(
        self,
        log_alpha: jnp.ndarray,
        mean_log_probs: jnp.ndarray,
    ) -> jnp.ndarray:
        return -log_alpha * (self.target_entropy + mean_log_probs)

    @partial(jax.jit, static_argnums=(0,))
    def _soft_update(
        self,
        target_params: dict[str, any],
        online_params: dict[str, any],
        tau: float,
    ) -> dict[str, any]:
        return jax.tree.map(
            lambda t, s: (1 - tau) * t + tau * s, target_params, online_params
        )

    def get_state(self) -> dict[str, any]:
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
            "replay_buffer": self.buffer.get_data(),  # ADD THIS
        }

        if self.learn_temperature:
            state["temperature_opt_state"] = self.temperature_opt_state

        return state

    def load_from_state(self, state: dict[str, any]) -> None:
        self.q_params = state["q_params"]
        self.v_params = state["v_params"]
        self.v_target_params = state["v_target_params"]
        self.policy_params = state["policy_params"]
        self.log_alpha = state["log_alpha"]
        self.q_opt_state = state["q_opt_state"]
        self.v_opt_state = state["v_opt_state"]
        self.policy_opt_state = state["policy_opt_state"]
        self.key = state["key"]
        self.buffer.load_data(state["replay_buffer"])  # ADD THIS

        if self.learn_temperature and "temperature_opt_state" in state:
            self.temperature_opt_state = state["temperature_opt_state"]
