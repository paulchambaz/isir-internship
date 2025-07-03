# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship.
#
# isir-internship is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import math
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax as optix
from haiku import PRNGSequence
from jax import nn


def build_mlp(
    x: jnp.ndarray,
    num_networks: int,
    hidden_dims: list[int],
    output_dim: int,
) -> list[jnp.ndarray]:
    def _forward(x: jnp.ndarray) -> jnp.ndarray:
        for hidden_dim in hidden_dims:
            x = jax.nn.relu(
                hk.Linear(
                    hidden_dim,
                    w_init=hk.initializers.Orthogonal(scale=math.sqrt(2)),
                )(x)
            )

        return hk.Linear(
            output_dim, w_init=hk.initializers.Orthogonal(scale=1)
        )(x)

    return [_forward(x) for _ in range(num_networks)]


def create_network(
    forward: callable, input_dims: list[int], key: jnp.ndarray
) -> tuple:
    network = hk.without_apply_rng(hk.transform(forward))
    params = network.init(
        key, *[np.zeros((1, dim), dtype=np.float32) for dim in input_dims]
    )
    return network, params


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        state_dim: int,
        action_dim: int,
    ) -> None:
        self._n = 0
        self._p = 0
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
        self.state[self._p] = state
        self.action[self._p] = action
        self.reward[self._p] = float(reward)
        self.next_state[self._p] = next_state
        self.done[self._p] = float(done)

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def sample(
        self, batch_size: int, key: jnp.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idxes = jax.random.randint(
            key=key, shape=(batch_size,), minval=0, maxval=self._n
        )

        return (
            self.state[idxes],
            self.action[idxes],
            self.reward[idxes],
            self.next_state[idxes],
            self.done[idxes],
        )


class AFU:
    __slots__ = [
        "_update_target",
        "action_dim",
        "batch_size",
        "buffer",
        "gamma",
        "log_alpha",
        "policy_network",
        "policy_optim",
        "policy_optim_state",
        "policy_params",
        "q_network",
        "q_optim",
        "q_optim_state",
        "q_params",
        "rho",
        "rng",
        "state_dim",
        "target_entropy",
        "tau",
        "temperature_optim",
        "temperature_optim_state",
        "v_network",
        "v_optim",
        "v_optim_state",
        "v_params",
        "v_target_params",
    ]

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        batch_size: int,
        tau: float,
        lr_actor: float,
        lr_critic: float,
        lr_alpha: float,
        units_actor: list[int],
        units_critic: list[int],
        rho: float,
        gamma: float,
        seed: int,
    ) -> None:
        self.rng = PRNGSequence(seed)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.q_network, self.q_params = AFU.build_q_network(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=units_critic,
            key=next(self.rng),
        )

        self.v_network, self.v_params = AFU.build_v_network(
            state_dim=state_dim,
            hidden_dims=units_critic,
            key=next(self.rng),
        )
        self.v_target_params = self.v_params

        self.policy_network, self.policy_params = AFU.build_policy_network(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=units_critic,
            key=next(self.rng),
        )

        self.log_alpha = jnp.array(np.log(1.0), dtype=jnp.float32)

        q_optim_init, self.q_optim = optix.adam(lr_critic)
        self.q_optim_state = q_optim_init(self.q_params)

        v_optim_init, self.v_optim = optix.adam(lr_critic)
        self.v_optim_state = v_optim_init(self.v_params)

        policy_optim_init, self.policy_optim = optix.adam(lr_actor)
        self.policy_optim_state = policy_optim_init(self.policy_params)

        temperature_optim_init, self.temperature_optim = optix.adam(lr_alpha)
        self.temperature_optim_state = temperature_optim_init(self.log_alpha)

        self.buffer = ReplayBuffer(
            buffer_size=buffer_size, state_dim=state_dim, action_dim=action_dim
        )
        self.batch_size = batch_size

        self.target_entropy = -float(action_dim)
        self.tau = tau
        self.rho = rho
        self.gamma = gamma

        self._update_target = jax.jit(partial(self._soft_update, tau=tau))

    def select_action(self, state: np.ndarray, evaluation: bool) -> np.ndarray:
        jax_state = state[None, ...]
        jax_action = (
            self._select_action(self.policy_params, jax_state)
            if evaluation
            else self._explore(self.policy_params, jax_state, next(self.rng))
        )
        return np.array(jax_action[0])

    @partial(jax.jit, static_argnums=0)
    def _select_action(
        self,
        policy_params: hk.Params,
        state: jnp.ndarray,
    ) -> jnp.ndarray:
        mean, _ = self.policy_network.apply(policy_params, state)
        return jnp.tanh(mean)

    @partial(jax.jit, static_argnums=0)
    def _explore(
        self,
        policy_params: hk.Params,
        state: jnp.ndarray,
        key: jnp.ndarray,
    ) -> jnp.ndarray:
        mean, log_std = self.policy_network.apply(policy_params, state)
        noise = jax.random.normal(key, mean.shape)
        raw_action = mean + noise * jnp.exp(log_std)
        return jnp.tanh(raw_action)

    def update(self) -> None:
        states, actions, reward, next_states, dones = self.buffer.sample(
            self.batch_size, next(self.rng)
        )

        # Update critic and value networks together
        _, grad = jax.value_and_grad(self._compute_critic_loss, argnums=(0, 1))(
            self.q_params,
            self.v_params,
            v_target_params=self.v_target_params,
            states=states,
            actions=actions,
            rewards=reward,
            next_states=next_states,
            dones=dones,
        )

        update, self.q_optim_state = self.q_optim(grad[0], self.q_optim_state)
        self.q_params = optix.apply_updates(self.q_params, update)
        update, self.v_optim_state = self.v_optim(grad[1], self.v_optim_state)
        self.v_params = optix.apply_updates(self.v_params, update)

        # Update actor
        (_, mean_log_probs), grad = jax.value_and_grad(
            self._compute_policy_loss, has_aux=True
        )(
            self.policy_params,
            q_params=self.q_params,
            v_params=self.v_params,
            log_alpha=self.log_alpha,
            states=states,
            key=next(self.rng),
        )

        update, self.policy_optim_state = self.policy_optim(
            grad, self.policy_optim_state
        )
        self.policy_params = optix.apply_updates(self.policy_params, update)

        # Update alpha
        _, grad = jax.value_and_grad(self._compute_temperature_loss)(
            self.log_alpha,
            mean_log_probs=mean_log_probs,
        )
        update, self.temperature_optim_state = self.temperature_optim(
            grad, self.temperature_optim_state
        )
        self.log_alpha = optix.apply_updates(self.log_alpha, update)

        # Update target network.
        self.v_target_params = self._update_target(
            self.v_target_params, self.v_params
        )

    @partial(jax.jit, static_argnums=0)
    def _compute_critic_loss(
        self,
        q_params: hk.Params,
        v_params: hk.Params,
        v_target_params: hk.Params,
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
        policy_params: hk.Params,
        q_params: hk.Params,
        v_params: hk.Params,
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

    @partial(jax.jit, static_argnums=0)
    def _soft_update(
        self,
        target_params: hk.Params,
        online_params: hk.Params,
        tau: float,
    ) -> hk.Params:
        return jax.tree.map(
            lambda t, s: (1 - tau) * t + tau * s, target_params, online_params
        )

    @staticmethod
    def build_q_network(
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        key: jnp.ndarray,
    ) -> tuple:
        def forward(states: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
            return build_mlp(
                x=jnp.concatenate([states, actions], axis=1),
                num_networks=3,
                hidden_dims=hidden_dims,
                output_dim=1,
            )

        return create_network(forward, [state_dim, action_dim], key)

    @staticmethod
    def build_v_network(
        state_dim: int, hidden_dims: list[int], key: jnp.ndarray
    ) -> tuple:
        def forward(states: jnp.ndarray) -> jnp.ndarray:
            return build_mlp(
                x=states,
                num_networks=2,
                hidden_dims=hidden_dims,
                output_dim=1,
            )

        return create_network(forward, [state_dim], key)

    @staticmethod
    def build_policy_network(
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        key: jnp.ndarray,
    ) -> tuple:
        def forward(states: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            output = build_mlp(
                x=states,
                num_networks=1,
                hidden_dims=hidden_dims,
                output_dim=2 * action_dim,
            )[0]

            mean, log_std = jnp.split(output, 2, axis=1)
            log_std = jnp.clip(log_std, -10.0, 2.0)
            return mean, log_std

        return create_network(forward, [state_dim], key)
