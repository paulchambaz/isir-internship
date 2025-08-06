# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship.
#
# isir-internship is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import argparse
import itertools
import math
import pickle
from collections.abc import Callable
from functools import partial
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random
from tqdm import tqdm

from algos.utils import lerp, se_loss, soft_update, where


class ToyMdp:
    def __init__(
        self, gamma: float, sigma: float, a0: float, a1: float, nu: float
    ) -> None:
        self.gamma = gamma
        self.sigma = sigma
        self.a0 = a0
        self.a1 = a1
        self.nu = nu

        actions = np.linspace(-1, 1, 1000000)
        rewards = self._mean_reward(self.a0, self.a1, self.nu, actions)
        self.optimal_action = actions[np.argmax(rewards)]
        self.optimal_reward = np.max(rewards)

    def sample_reward(self, action: float, rng: np.random.Generator) -> float:
        mean_reward = self._mean_reward(
            self.a0, self.a1, self.nu, np.array([action])
        )[0]
        return mean_reward + rng.normal(0, self.sigma)

    def true_q_value(self, action: float) -> float:
        mean_reward = self._mean_reward(
            self.a0, self.a1, self.nu, np.array([action])
        )[0]
        return mean_reward + self.gamma * self.optimal_reward / (1 - self.gamma)

    @staticmethod
    def _mean_reward(
        a0: float, a1: float, nu: float, actions: np.ndarray
    ) -> np.ndarray:
        return (a0 + (a1 - a0) / 2 * (actions + 1)) * np.sin(nu * actions)


class MLP(nn.Module):
    hidden_dims: list[int]
    output_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for hidden_dim in self.hidden_dims:
            x = jax.nn.sigmoid(
                nn.Dense(
                    hidden_dim,
                    kernel_init=jax.nn.initializers.he_uniform(),
                )(x)
            )
        return nn.Dense(
            self.output_dim,
            kernel_init=jax.nn.initializers.he_uniform(),
        )(x)


class QNetwork(nn.Module):
    hidden_dims: list[int]
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
                    output_dim=1,
                )(states_actions).squeeze(-1)
                for _ in range(self.n_critics)
            ],
            axis=1,
        )


class VNetwork(nn.Module):
    hidden_dims: list[int]
    n_critics: int

    @nn.compact
    def __call__(self, states: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack(
            [
                MLP(
                    hidden_dims=self.hidden_dims,
                    output_dim=1,
                )(states).squeeze(-1)
                for _ in range(self.n_critics)
            ],
            axis=1,
        )


class ZNetwork(nn.Module):
    hidden_dims: list[int]
    n_quantiles: int
    n_critics: int

    @nn.compact
    def __call__(self, actions: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack(
            [
                MLP(
                    hidden_dims=self.hidden_dims,
                    output_dim=self.n_quantiles,
                )(actions)
                for _ in range(self.n_critics)
            ],
            axis=1,
        )


class AvgEnsemble:
    def __init__(self, n: int, gamma: float, key: jnp.ndarray) -> None:
        self.gamma = gamma
        self.buffer = None

        self.network = QNetwork(hidden_dims=[50, 50], n_critics=n)
        self.optim = optax.adam(1e-3)

        dummy_state = jnp.zeros((1, 1))
        dummy_action = jnp.zeros((1, 1))

        self.params = self.network.init(key, dummy_state, dummy_action)
        self.opt_state = self.optim.init(self.params)

    def set_buffer(self, dataset: list) -> None:
        self.buffer = dataset

    def update(self) -> None:
        actions, rewards = self.buffer
        targets = self._compute_targets(self.params, rewards)
        self.params, self.opt_state = self._update_network(
            self.params, self.opt_state, actions, targets
        )

    @partial(jax.jit, static_argnums=(0,))
    def _compute_targets(
        self, params: dict, rewards: jnp.ndarray
    ) -> jnp.ndarray:
        grid_actions = jnp.linspace(-1, 1, 2001).reshape(-1, 1)
        grid_q_values = self.network.apply(params, grid_actions).mean(axis=1)
        best_q_value = jnp.max(grid_q_values)
        return rewards + self.gamma * best_q_value

    @partial(jax.jit, static_argnums=(0,))
    def _update_network(
        self,
        params: dict,
        opt_state: optax.OptState,
        actions: jnp.ndarray,
        targets: jnp.ndarray,
    ) -> tuple[dict, optax.OptState]:
        def loss_fn(params: dict) -> jnp.ndarray:
            values = self.network.apply(params, actions)
            targets_expanded = targets[:, None]
            return jnp.mean((values - targets_expanded) ** 2)

        grads = jax.grad(loss_fn)(params)
        updates, new_opt_state = self.optim.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state

    @partial(jax.jit, static_argnums=(0,))
    def _call(self, params: dict, actions: jnp.ndarray) -> jnp.ndarray:
        return self.network.apply(params, actions).mean(axis=1)

    def __call__(self, actions: jnp.ndarray) -> jnp.ndarray:
        return self._call(self.params, actions)


class MinEnsemble:
    def __init__(self, n: int, gamma: float, key: jnp.ndarray) -> None:
        self.gamma = gamma
        self.buffer = None

        self.network = QNetwork(hidden_dims=[50, 50], n_critics=n)
        self.optim = optax.adam(1e-3)

        dummy_state = jnp.zeros((1, 1))
        dummy_action = jnp.zeros((1, 1))

        self.params = self.network.init(key, dummy_state, dummy_action)
        self.opt_state = self.optim.init(self.params)

    def set_buffer(self, dataset: list) -> None:
        self.buffer = dataset

    def update(self) -> None:
        actions, rewards = self.buffer
        targets = self._compute_targets(self.params, rewards)
        self.params, self.opt_state = self._update_network(
            self.params, self.opt_state, actions, targets
        )

    @partial(jax.jit, static_argnums=(0,))
    def _compute_targets(
        self, params: dict, rewards: jnp.ndarray
    ) -> jnp.ndarray:
        grid_actions = jnp.linspace(-1, 1, 2001).reshape(-1, 1)
        grid_q_values = self.network.apply(params, grid_actions).min(axis=1)
        best_q_value = jnp.max(grid_q_values)
        return rewards + self.gamma * best_q_value

    @partial(jax.jit, static_argnums=(0,))
    def _update_network(
        self,
        params: dict,
        opt_state: optax.OptState,
        actions: jnp.ndarray,
        targets: jnp.ndarray,
    ) -> tuple[dict, optax.OptState]:
        def loss_fn(params: dict) -> jnp.ndarray:
            values = self.network.apply(params, actions)
            targets_expanded = targets[:, None]
            return jnp.mean((values - targets_expanded) ** 2)

        grads = jax.grad(loss_fn)(params)
        updates, new_opt_state = self.optim.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state

    @partial(jax.jit, static_argnums=(0,))
    def _call(self, params: dict, actions: jnp.ndarray) -> jnp.ndarray:
        return self.network.apply(params, actions).min(axis=1)

    def __call__(self, actions: jnp.ndarray) -> jnp.ndarray:
        return self._call(self.params, actions)


class TqcEnsemble:
    def __init__(
        self, n: int, num_networks: int, gamma: float, key: jnp.ndarray
    ) -> None:
        self.gamma = gamma
        self.buffer = None

        self.num_quantiles = 25
        self.num_networks = num_networks
        self.kept_per_network = self.num_quantiles - n
        self.total_kept = self.kept_per_network * self.num_networks

        self.network = ZNetwork(
            hidden_dims=[50, 50],
            n_quantiles=self.num_quantiles,
            n_critics=self.num_networks,
        )
        self.optim = optax.adam(1e-3)

        self.tau_levels = jnp.array(
            [
                (2 * m - 1) / (2 * self.num_quantiles)
                for m in range(1, self.num_quantiles + 1)
            ]
        )
        self.all_tau = jnp.tile(self.tau_levels, self.num_networks)

        dummy_state = jnp.zeros((1, 1))
        dummy_action = jnp.zeros((1, 1))

        self.params = self.network.init(key, dummy_state, dummy_action)
        self.opt_state = self.optim.init(self.params)

    def set_buffer(self, dataset: list) -> None:
        self.buffer = dataset

    def update(self) -> None:
        actions, rewards = self.buffer
        targets = self._compute_targets(self.params, rewards)
        self.params, self.opt_state = self._update_network(
            self.params, self.opt_state, actions, targets
        )

    @partial(jax.jit, static_argnums=(0,))
    def _compute_targets(
        self, params: dict, rewards: jnp.ndarray
    ) -> jnp.ndarray:
        grid_actions = jnp.linspace(-1, 1, 2001).reshape(-1, 1)

        grid_quantiles = self.network.apply(params, grid_actions).reshape(
            grid_actions.shape[0], -1
        )
        grid_sorted = jnp.sort(grid_quantiles, axis=-1)
        grid_truncated = grid_sorted[:, : self.total_kept]
        grid_q_values = jnp.mean(grid_truncated, axis=-1)

        best_action_idx = jnp.argmax(grid_q_values)

        next_quantiles = grid_quantiles[best_action_idx].flatten()
        next_sorted = jnp.sort(next_quantiles)
        next_truncated = next_sorted[: self.total_kept]

        return rewards[:, None] + self.gamma * next_truncated[None, :]

    @partial(jax.jit, static_argnums=(0,))
    def _update_network(
        self,
        params: dict,
        opt_state: optax.OptState,
        actions: jnp.ndarray,
        targets: jnp.ndarray,
    ) -> tuple[dict, optax.OptState]:
        def loss_fn(params: dict) -> jnp.ndarray:
            current_quantiles = self.network.apply(params, actions).reshape(
                actions.shape[0], -1
            )

            targets_expanded = targets[:, None, :]
            quantiles_expanded = current_quantiles[:, :, None]

            diff = targets_expanded - quantiles_expanded
            abs_diff = jnp.abs(diff)

            huber = jnp.where(
                abs_diff <= 1.0, 0.5 * diff * diff, abs_diff - 0.5
            )

            indicator = (diff < 0).astype(jnp.float32)
            tau_expanded = self.all_tau[None, :, None]
            weights = jnp.abs(tau_expanded - indicator)

            weighted_loss = weights * huber
            total_loss = jnp.sum(weighted_loss)

            normalization = (
                targets.shape[0] * current_quantiles.shape[1] * targets.shape[1]
            )

            return total_loss / normalization

        grads = jax.grad(loss_fn)(params)
        updates, new_opt_state = self.optim.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state

    @partial(jax.jit, static_argnums=(0,))
    def _call(self, params: dict, actions: jnp.ndarray) -> jnp.ndarray:
        quantiles = self.network.apply(params, actions)
        union_quantiles = quantiles.reshape(actions.shape[0], -1)
        sorted_quantiles = jnp.sort(union_quantiles, axis=-1)
        truncated = sorted_quantiles[:, : self.total_kept]

        return jnp.mean(truncated, axis=-1)

    def __call__(self, actions: jnp.ndarray) -> jnp.ndarray:
        return self._call(self.params, actions)


class TopEnsemble:
    def __init__(self, beta: float, gamma: float, key: jnp.ndarray) -> None:
        self.beta = beta
        self.gamma = gamma
        self.buffer = None

        self.n_critics = 2
        self.num_quantiles = 25

        self.network = ZNetwork(
            hidden_dims=[50, 50],
            n_quantiles=self.num_quantiles,
            n_critics=self.n_critics,
        )
        self.optim = optax.adam(1e-3)

        self.tau_levels = jnp.array(
            [
                (2 * m - 1) / (2 * self.num_quantiles)
                for m in range(1, self.num_quantiles + 1)
            ]
        )
        self.all_tau = jnp.tile(self.tau_levels, self.n_critics)

        dummy_state = jnp.zeros((1, 1))
        dummy_action = jnp.zeros((1, 1))

        self.params = self.network.init(key, dummy_state, dummy_action)
        self.opt_state = self.optim.init(self.params)

    def set_buffer(self, dataset: list) -> None:
        self.buffer = dataset

    def update(self) -> None:
        actions, rewards = self.buffer
        targets = self._compute_targets(self.params, rewards)
        self.params, self.opt_state = self._update_network(
            self.params, self.opt_state, actions, targets
        )

    @partial(jax.jit, static_argnums=(0,))
    def _compute_targets(
        self, params: dict, rewards: jnp.ndarray
    ) -> jnp.ndarray:
        grid_actions = jnp.linspace(-1, 1, 2001).reshape(-1, 1)

        grid_quantiles = self.network.apply(params, grid_actions)

        mean_quantiles = jnp.mean(grid_quantiles, axis=1)
        std_quantiles = jnp.std(grid_quantiles, axis=1)

        belief_quantiles = mean_quantiles + self.beta * std_quantiles

        belief_q_values = jnp.mean(belief_quantiles, axis=1)

        best_action_idx = jnp.argmax(belief_q_values)
        next_quantiles = belief_quantiles[best_action_idx]

        return rewards[:, None] + self.gamma * next_quantiles[None, :]

    @partial(jax.jit, static_argnums=(0,))
    def _update_network(
        self,
        params: dict,
        opt_state: optax.OptState,
        actions: jnp.ndarray,
        targets: jnp.ndarray,
    ) -> tuple[dict, optax.OptState]:
        def loss_fn(params: dict) -> jnp.ndarray:
            current_quantiles = self.network.apply(params, actions)

            targets_expanded = targets[:, None, None, :]
            quantiles_expanded = current_quantiles[:, :, :, None]

            diff = targets_expanded - quantiles_expanded
            abs_diff = jnp.abs(diff)

            huber = jnp.where(
                abs_diff <= 1.0, 0.5 * diff * diff, abs_diff - 0.5
            )

            indicator = (diff < 0).astype(jnp.float32)
            tau_expanded = self.tau_levels[None, None, :, None]
            weights = jnp.abs(tau_expanded - indicator)

            weighted_loss = weights * huber
            total_loss = jnp.sum(weighted_loss)

            normalization = (
                targets.shape[0]
                * self.n_critics
                * self.num_quantiles
                * targets.shape[1]
            )

            return total_loss / normalization

        grads = jax.grad(loss_fn)(params)
        updates, new_opt_states = self.optim.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_states

    @partial(jax.jit, static_argnums=(0,))
    def _call(self, params: dict, actions: jnp.ndarray) -> jnp.ndarray:
        quantiles = self.network.apply(params, actions)

        mean_quantiles = jnp.mean(quantiles, axis=1)
        std_quantiles = jnp.std(quantiles, axis=1)

        belief_quantiles = mean_quantiles + self.beta * std_quantiles

        return jnp.mean(belief_quantiles, axis=1)

    def __call__(self, actions: jnp.ndarray) -> jnp.ndarray:
        return self._call(self.params, actions)


class NDTopEnsemble:
    def __init__(self, beta: float, gamma: float, key: jnp.ndarray) -> None:
        self.beta = beta
        self.gamma = gamma
        self.buffer = None

        self.network = QNetwork(hidden_dims=[50, 50], n_critics=2)
        self.optim = optax.adam(1e-3)

        dummy_state = jnp.zeros((1, 1))
        dummy_action = jnp.zeros((1, 1))

        self.params = self.network.init(key, dummy_state, dummy_action)
        self.opt_state = self.optim.init(self.params)

    def set_buffer(self, dataset: list) -> None:
        self.buffer = dataset

    def update(self) -> None:
        actions, rewards = self.buffer
        targets = self._compute_targets(self.params, rewards)
        self.params, self.opt_state = self._update_network(
            self.params, self.opt_state, actions, targets
        )

    @partial(jax.jit, static_argnums=(0,))
    def _compute_targets(
        self, params: dict, rewards: jnp.ndarray
    ) -> jnp.ndarray:
        grid_actions = jnp.linspace(-1, 1, 2001).reshape(-1, 1)

        grid_q_values = self.network.apply(params, grid_actions)

        mean_q = jnp.mean(grid_q_values, axis=1)
        std_q = jnp.std(grid_q_values, axis=1)
        belief_q = mean_q + self.beta * std_q

        best_q_value = jnp.max(belief_q)

        return rewards + self.gamma * best_q_value

    @partial(jax.jit, static_argnums=(0,))
    def _update_network(
        self,
        params: dict,
        opt_state: optax.OptState,
        actions: jnp.ndarray,
        targets: jnp.ndarray,
    ) -> tuple[dict, optax.OptState]:
        def loss_fn(params: dict) -> jnp.ndarray:
            values = self.network.apply(params, actions)
            targets_expanded = targets[:, None]
            return jnp.mean((values - targets_expanded) ** 2)

        grads = jax.grad(loss_fn)(params)
        updates, new_opt_state = self.optim.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state

    @partial(jax.jit, static_argnums=(0,))
    def _call(self, params: dict, actions: jnp.ndarray) -> jnp.ndarray:
        q_values = self.network.apply(params, actions)

        mean_q = jnp.mean(q_values, axis=1)
        std_q = jnp.std(q_values, axis=1)

        return mean_q + self.beta * std_q

    def __call__(self, actions: jnp.ndarray) -> jnp.ndarray:
        return self._call(self.params, actions)


class AFUEnsemble:
    def __init__(self, rho: float, gamma: float, key: jnp.ndarray) -> None:
        self.rho = rho
        self.gamma = gamma
        self.buffer = None

        self.q_network = QNetwork(hidden_dims=[50, 50], n_critics=3)
        self.v_network = VNetwork(hidden_dims=[50, 50], n_critics=2)

        self.q_optim = optax.adam(1e-3)
        self.v_optim = optax.adam(1e-3)

        q_key, v_key = random.split(key)

        dummy_state = jnp.zeros((1, 1))
        dummy_action = jnp.zeros((1, 1))

        self.q_params = self.q_network.init(q_key, dummy_state, dummy_action)
        self.v_params = self.v_network.init(v_key, dummy_action)
        self.v_target_params = self.v_params

        self.q_opt_state = self.q_optim.init(self.q_params)
        self.v_opt_state = self.v_optim.init(self.v_params)

    def set_buffer(self, dataset: list) -> None:
        self.buffer = dataset

    def update(self) -> None:
        states, actions, rewards, next_states, dones = self.buffer

        targets = self._compute_targets(
            self.v_params, states, actions, rewards, next_states, dones
        )
        self.q_params, self.v_params, self.q_opt_state, self.v_opt_state = (
            self._update_networks(
                self.q_params,
                self.v_params,
                self.q_opt_state,
                self.v_opt_state,
                states,
                actions,
                rewards,
                next_states,
                dones,
                targets,
            )
        )

    @partial(jax.jit, static_argnums=(0,))
    def _compute_targets(
        self,
        v_params: dict,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        next_states: jnp.ndarray,
        dones: jnp.ndarray,
    ) -> jnp.ndarray:
        v_target_list = self.v_network.apply(v_params, next_states)
        v_targets = jnp.min(v_target_list, axis=1)
        return rewards + self.gamma * (1.0 - dones) * v_targets

    @partial(jax.jit, static_argnums=(0,))
    def _update_networks(
        self,
        q_params: dict,
        v_params: dict,
        q_opt_state: optax.OptState,
        v_opt_state: optax.OptState,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        next_states: jnp.ndarray,
        dones: jnp.ndarray,
        targets: jnp.ndarray,
    ) -> tuple[dict, dict, optax.OptState, optax.OptState]:
        def loss_fn(q_params: dict, v_params: dict) -> jnp.ndarray:
            q_targets = jax.lax.stop_gradient(jnp.expand_dims(targets, -1))

            print(f"{q_targets=}")

            v_values = self.v_network.apply(v_params, states).squeeze(0)

            print(f"{v_values=}")

            v_values = jnp.tile(v_values, (actions.shape[0], 1))

            print(f"{v_values=}")

            q_values_list = self.q_network.apply(q_params, actions)

            print(f"{q_values_list=}")

            q_values = q_values_list[:, -1:]

            print(f"{q_values=}")

            a_values = -q_values_list[:, :-1]

            print(f"{a_values=}")

            errors = se_loss(value=q_values, target=q_targets)

            print(f"{errors=}")

            q_loss = errors.mean()

            print(f"{q_loss=}")

            upsilon_values = where(
                jax.lax.stop_gradient(v_values + a_values < q_targets),
                v_values,
                lerp(self.rho, v_values, jax.lax.stop_gradient(v_values)),
            )

            print(f"{upsilon_values=}")

            z_values = where(
                jax.lax.stop_gradient(v_values < q_targets),
                se_loss(value=upsilon_values + a_values, target=q_targets),
                se_loss(value=upsilon_values, target=q_targets)
                + se_loss(value=a_values, target=0),
            )

            print(f"{z_values=}")

            va_loss = z_values.mean()

            print(f"{va_loss=}")

            exit()

            return va_loss + q_loss

        (q_grads, v_grads) = jax.grad(loss_fn, argnums=(0, 1))(
            q_params, v_params
        )
        q_updates, new_q_opt_state = self.q_optim.update(q_grads, q_opt_state)
        new_q_params = optax.apply_updates(q_params, q_updates)

        v_updates, new_v_opt_state = self.v_optim.update(v_grads, v_opt_state)
        new_v_params = optax.apply_updates(v_params, v_updates)

        return new_q_params, new_v_params, new_q_opt_state, new_v_opt_state

    @partial(jax.jit, static_argnums=(0,))
    def _call(
        self, q_params: dict, v_params: dict, actions: jnp.ndarray
    ) -> jnp.ndarray:
        q_values = self.q_network.apply(q_params, actions)
        return q_values[:, -1]

    def __call__(self, actions: jnp.ndarray) -> jnp.ndarray:
        return self._call(self.q_params, self.v_params, actions)


def create_avg(n: int, gamma: float, key: jnp.ndarray) -> AvgEnsemble:
    return AvgEnsemble(n, gamma, key)


def create_min_ensemble(n: int, gamma: float, key: jnp.ndarray) -> MinEnsemble:
    return MinEnsemble(n, gamma, key)


def create_tqc_ensemble(n: int, gamma: float, key: jnp.ndarray) -> TqcEnsemble:
    return TqcEnsemble(n, 2, gamma, key)


def create_ttqc_ensemble(n: int, gamma: float, key: jnp.ndarray) -> TqcEnsemble:
    return TqcEnsemble(n, 1, gamma, key)


def create_ndtop_ensemble(
    beta: float, gamma: float, key: jnp.ndarray
) -> NDTopEnsemble:
    return NDTopEnsemble(beta, gamma, key)


def create_top_ensemble(
    beta: float, gamma: float, key: jnp.ndarray
) -> TopEnsemble:
    return TopEnsemble(beta, gamma, key)


def create_afu_ensemble(
    rho: float, gamma: float, key: jnp.ndarray
) -> AFUEnsemble:
    return AFUEnsemble(rho, gamma, key)


def create_dataset(
    mdp: ToyMdp, buffer_size: int, rng: np.random.Generator
) -> tuple[jnp.ndarray, jnp.ndarray]:
    states = np.zeros((buffer_size, 1))
    actions = np.linspace(-1, 1, buffer_size)
    rewards = [mdp.sample_reward(a, rng) for a in actions]
    next_states = np.zeros((buffer_size, 1))
    dones = np.zeros((buffer_size,))

    states_tensor = jnp.array(states, dtype=jnp.float32)
    actions_tensor = jnp.array(actions, dtype=jnp.float32).reshape(-1, 1)
    rewards_tensor = jnp.array(rewards, dtype=jnp.float32)
    next_states_tensor = jnp.array(next_states, dtype=jnp.float32)
    dones_tensor = jnp.array(dones, dtype=jnp.float32)

    return (
        states_tensor,
        actions_tensor,
        rewards_tensor,
        next_states_tensor,
        dones_tensor,
    )


def robust_mean(data: np.ndarray, p1: float, p2: float) -> float:
    q1, q2 = np.quantile(data, [p1, p2])
    selection = data[(data >= q1) & (data <= q2)]
    mean = np.mean(selection) if len(selection) == 0 else np.mean(data)
    return float(mean)


def compute_bias_variance(
    n: int | float,
    mdp: ToyMdp,
    buffer_size: int,
    total_steps: int,
    eval_freq: int,
    num_seed: int,
    create_ensemble_fn: Callable,
) -> dict:
    eval_actions = jnp.linspace(-1, 1, 2000).reshape(-1, 1)
    true_q = np.array([mdp.true_q_value(a.item()) for a in eval_actions])
    optim_action = mdp.optimal_action

    metadata = {
        "optim_action": float(optim_action),
        "eval_actions": np.array(eval_actions.flatten()).tolist(),
        "true_q": true_q.tolist(),
    }

    step_results = {}
    for seed in range(num_seed):
        rng = np.random.default_rng(seed + int(abs(10 * n)) * num_seed)
        dataset = create_dataset(mdp, buffer_size, rng)

        key = random.PRNGKey(seed + int(abs(10 * n)) * num_seed)
        ensemble = create_ensemble_fn(n, mdp.gamma, key)
        ensemble.set_buffer(dataset)

        for step in range(eval_freq, total_steps + 1, eval_freq):
            for _ in range(eval_freq):
                ensemble.update()

            predicted_q = np.array(ensemble(eval_actions))

            error = predicted_q - true_q
            bias = float(np.mean(error))
            variance = float(np.var(error))

            policy_error = float(np.abs(optim_action - mdp.optimal_action))

            if step not in step_results:
                step_results[step] = []
            step_results[step].append(
                (bias, variance, policy_error, predicted_q.tolist())
            )

    final_results = {}
    for step, results in step_results.items():
        metrics = np.array(
            [
                (bias, variance, policy_error)
                for bias, variance, policy_error, _ in results
            ]
        )
        final_results[step] = tuple(
            robust_mean(metrics[:, i], 0.1, 0.9) for i in range(3)
        )

    return {
        "metadata": metadata,
        "results": final_results,
        "raw_data": step_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute TQC figure")
    parser.add_argument(
        "--cpu", action="store_true", help="Force JAX to use CPU instead of GPU"
    )
    args = parser.parse_args()
    if args.cpu:
        jax.config.update("jax_platform_name", "cpu")

    mdp = ToyMdp(gamma=0.99, sigma=0.25, a0=0.3, a1=0.9, nu=5.0)

    # avg_data = [1]
    # avg_data = [1, 3, 5]
    avg_data = [1, 3, 5, 10, 20, 50]
    min_data = [2, 3, 4, 6, 8, 10]
    tqc_data = [0, 1, 2, 3, 4, 5, 6, 7, 10, 13, 16]
    top_data = [-1.0, -0.7, -0.5, 0.0, 0.5, 1.0]
    # rho_data = [0.1, 0.3, 0.5, 0.7, 0.9]
    rho_data = [0.5]

    experiments = list(
        itertools.chain(
            # [("avg", n, create_avg) for n in avg_data],
            # [("min", n, create_min_ensemble) for n in min_data],
            # [("tqc", n, create_tqc_ensemble) for n in tqc_data],
            # [("ttqc", n, create_ttqc_ensemble) for n in tqc_data],
            # [("ndtop", beta, create_ndtop_ensemble) for beta in top_data],
            # [("top", beta, create_top_ensemble) for beta in top_data],
            [("afu", rho, create_afu_ensemble) for rho in rho_data]
        )
    )

    results = {}

    progress_bar = tqdm(experiments, desc="Running experiments")

    for method, n, create_ensemble_fn in progress_bar:
        progress_bar.set_postfix({"method": method, "n": n})
        results[(method, n)] = compute_bias_variance(
            n=n,
            mdp=mdp,
            buffer_size=5,
            total_steps=20000,
            eval_freq=100,  # 100
            num_seed=1,  # 10
            create_ensemble_fn=create_ensemble_fn,
        )

    # for key, value in results.items():
    #     print(f"\n# {key}\n")
    #     for k, v in value.items():
    #         print(f"{k}\t{v}")

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/tqc_figure_results.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
