# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship.
#
# isir-internship is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import itertools
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


class ToyMdp:
    __slots__ = [
        "a0",
        "a1",
        "gamma",
        "nu",
        "optimal_action",
        "optimal_reward",
        "sigma",
    ]

    def __init__(
        self, gamma: float, sigma: float, a0: float, a1: float, nu: float
    ) -> None:
        self.gamma = gamma
        self.sigma = sigma
        self.a0 = a0
        self.a1 = a1
        self.nu = nu

        actions = np.linspace(-1, 1, 10000)
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
        return a0 + (a1 - a0) / 2 * (actions + 1) * np.sin(nu * actions)


class MLP(nn.Module):
    hidden_dims: list[int]
    output_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for hidden_dim in self.hidden_dims:
            x = jax.nn.relu(
                nn.Dense(
                    hidden_dim,
                    kernel_init=jax.nn.initializers.variance_scaling(
                        scale=1 / 3, mode="fan_in", distribution="uniform"
                    ),
                )(x)
            )
        return nn.Dense(
            self.output_dim,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=1 / 3, mode="fan_in", distribution="uniform"
            ),
        )(x)


class QNetwork(nn.Module):
    hidden_dims: list[int]

    @nn.compact
    def __call__(self, actions: jnp.ndarray) -> jnp.ndarray:
        return MLP(hidden_dims=self.hidden_dims, output_dim=1)(actions).squeeze(
            -1
        )


class ZNetwork(nn.Module):
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


class AvgEnsemble:
    def __init__(self, n: int, gamma: float, key: jnp.ndarray) -> None:
        self.n = n
        self.gamma = gamma
        self.buffer = None

        self.network = QNetwork(hidden_dims=[50, 50])
        self.optim = optax.adam(1e-3)

        dummy_input = jnp.zeros((1, 1))
        keys = jax.random.split(key, n)

        self.params_list = [self.network.init(k, dummy_input) for k in keys]
        self.opt_states = [self.optim.init(p) for p in self.params_list]

    def set_buffer(self, dataset: list) -> None:
        self.buffer = dataset

    def update(self) -> None:
        actions, rewards = self.buffer

        targets = self._compute_targets(self.params_list, rewards)

        for i in range(self.n):
            self.params_list[i], self.opt_states[i] = self._update_network(
                self.params_list[i],
                self.opt_states[i],
                actions,
                targets,
            )

    @partial(jax.jit, static_argnums=(0,))
    def _compute_targets(
        self,
        params_list: list[dict],
        rewards: jnp.ndarray,
    ) -> jnp.ndarray:
        grid_actions = jnp.linspace(-1, 1, 2001).reshape(-1, 1)

        grid_q_values = jnp.stack(
            [self.network.apply(params, grid_actions) for params in params_list]
        ).mean(axis=0)

        best_q_value = jnp.max(grid_q_values)

        return rewards + self.gamma * best_q_value

    @partial(jax.jit, static_argnums=(0,))
    def _update_network(
        self,
        params: dict,
        opt_state: optax.OptState,
        actions: jnp.ndarray,
        targets: jnp.ndarray,
    ) -> None:
        def loss_fn(params: dict) -> jnp.ndarray:
            values = self.network.apply(params, actions)
            return jnp.mean((values - targets) ** 2)

        grads = jax.grad(loss_fn)(params)
        updates, new_opt_state = self.optim.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, actions: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack(
            [self.network.apply(params, actions) for params in self.params_list]
        ).mean(axis=0)


class MinEnsemble:
    def __init__(self, n: int, gamma: float, key: jnp.ndarray) -> None:
        self.n = n
        self.gamma = gamma
        self.buffer = None

        self.network = QNetwork(hidden_dims=[50, 50])
        self.optim = optax.adam(1e-3)

        dummy_input = jnp.zeros((1, 1))
        keys = jax.random.split(key, n)

        self.params_list = [self.network.init(k, dummy_input) for k in keys]
        self.opt_states = [self.optim.init(p) for p in self.params_list]

    def set_buffer(self, dataset: list) -> None:
        self.buffer = dataset

    def update(self) -> None:
        actions, rewards = self.buffer

        targets = self._compute_targets(self.params_list, rewards)

        for i in range(self.n):
            self.params_list[i], self.opt_states[i] = self._update_network(
                self.params_list[i],
                self.opt_states[i],
                actions,
                targets,
            )

    @partial(jax.jit, static_argnums=(0,))
    def _compute_targets(
        self,
        params_list: list[dict],
        rewards: jnp.ndarray,
    ) -> jnp.ndarray:
        grid_actions = jnp.linspace(-1, 1, 2001).reshape(-1, 1)

        grid_q_values = jnp.stack(
            [self.network.apply(params, grid_actions) for params in params_list]
        ).min(axis=0)

        best_q_value = jnp.max(grid_q_values)

        return rewards + self.gamma * best_q_value

    @partial(jax.jit, static_argnums=(0,))
    def _update_network(
        self,
        params: dict,
        opt_state: optax.OptState,
        actions: jnp.ndarray,
        targets: jnp.ndarray,
    ) -> None:
        def loss_fn(params: dict) -> jnp.ndarray:
            values = self.network.apply(params, actions)
            return jnp.mean((values - targets) ** 2)

        grads = jax.grad(loss_fn)(params)
        updates, new_opt_state = self.optim.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, actions: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack(
            [self.network.apply(params, actions) for params in self.params_list]
        ).min(axis=0)


def train_avg_method(
    dataset: tuple[jnp.ndarray, jnp.ndarray],
    n: int,
    iterations: int,
    gamma: float,
    key: jnp.ndarray,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    ensemble = AvgEnsemble(n, gamma, key)
    ensemble.set_buffer(dataset)

    for _ in range(iterations):
        ensemble.update()

    return ensemble


def train_min_method(
    dataset: tuple[jnp.ndarray, jnp.ndarray],
    n: int,
    iterations: int,
    gamma: float,
    key: jnp.ndarray,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    ensemble = MinEnsemble(n, gamma, key)
    ensemble.set_buffer(dataset)

    for _ in range(iterations):
        ensemble.update()

    return ensemble


# def train_tqc_method(
#     dataset: tuple[jnp.ndarray, jnp.ndarray],
#     n: int,
#     iterations: int,
#     gamma: float,
#     key: jnp.ndarray,
# ) -> Callable[[jnp.ndarray], jnp.ndarray]:
#     ensemble = TQCEnsemble(n, gamma, key)
#     ensemble.set_buffer(dataset)
#
#     for _ in range(iterations):
#         ensemble.update()
#
#     return ensemble


def create_dataset(
    mdp: ToyMdp, buffer_size: int, rng: np.random.Generator
) -> tuple[jnp.ndarray, jnp.ndarray]:
    actions = np.linspace(-1, 1, buffer_size)
    rewards = [mdp.sample_reward(a, rng) for a in actions]

    actions_tensor = jnp.array(actions, dtype=jnp.float32).reshape(-1, 1)
    rewards_tensor = jnp.array(rewards, dtype=jnp.float32)

    return actions_tensor, rewards_tensor


def robust_mean(data: np.ndarray, p1: float, p2: float) -> float:
    q1, q2 = np.quantile(data, [p1, p2])
    return np.mean(data[(data >= q1) & (data <= q2)])


def compute_bias_variance(
    n: int,
    mdp: ToyMdp,
    buffer_size: int,
    iterations: int,
    num_seed: int,
    train_fn: Callable[
        [tuple[jnp.ndarray, jnp.ndarray], int, int, float, jnp.ndarray],
        Callable[[jnp.ndarray], jnp.ndarray],
    ],
) -> None:
    errors = []
    optim_actions = []

    for seed in range(num_seed):
        rng = np.random.default_rng(seed)
        dataset = create_dataset(mdp, buffer_size, rng)

        key = random.PRNGKey(seed)
        trained_net = train_fn(dataset, n, iterations, mdp.gamma, key)

        eval_actions = jnp.linspace(-1, 1, 2000).reshape(-1, 1)

        predicted_q = np.array(trained_net(eval_actions))
        true_q = np.array([mdp.true_q_value(a.item()) for a in eval_actions])

        error = predicted_q - true_q
        errors.append(error)

        optim_action = eval_actions[np.argmax(predicted_q)].item()
        optim_actions.append(optim_action)

    errors = np.array(errors)

    bias = robust_mean(np.mean(errors, axis=1), 0.1, 0.9)
    variance = robust_mean(np.var(errors, axis=1), 0.1, 0.9)
    policy_error = robust_mean(
        np.abs(optim_actions - mdp.optimal_action), 0.1, 0.9
    )

    return bias, variance, policy_error


def main() -> None:
    mdp = ToyMdp(gamma=0.99, sigma=0.25, a0=0.3, a1=0.9, nu=5.0)

    avg_data = [1, 3, 5, 10, 20, 50]
    min_data = [2, 3, 4, 6, 8, 10]

    experiments = list(
        itertools.chain(
            [("avg", n, train_avg_method) for n in avg_data],
            [("min", n, train_min_method) for n in min_data],
        )
    )

    results = {}

    for method, n, train_fn in tqdm(experiments, desc="Running experiments"):
        tqdm.write(f"Processing {method} with n={n}")
        results[(method, n)] = compute_bias_variance(
            n=n,
            mdp=mdp,
            buffer_size=50,
            iterations=3000,
            num_seed=1,
            train_fn=train_fn,
        )

    print(results)

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/tqc_figure_results.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
