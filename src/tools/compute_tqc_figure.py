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
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random
from tqdm import tqdm

from algos import MLP


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


class QNetwork(nn.Module):
    hidden_dims: list[int]
    num_critics: int

    @nn.compact
    def __call__(self, actions: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(
            [
                MLP(hidden_dims=self.hidden_dims, output_dim=1)(
                    actions
                ).squeeze(-1)
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


def train_avg_method(
    dataset: tuple[jnp.ndarray, jnp.ndarray],
    n: int,
    iterations: int,
    gamma: float,
    key: jnp.ndarray,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    actions, rewards = dataset

    net = QNetwork(hidden_dims=[50, 50], num_critics=n)
    params = net.init(key, actions[:1])
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    grid_actions = jnp.linspace(-1, 1, 2001).reshape(-1, 1)

    @jax.jit
    def update(
        params: dict[str, jax.Array],
        opt_state: optax.OptState,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
    ) -> tuple[dict[str, jax.Array], optax.OptState]:
        grid_q_values = net.apply(params, grid_actions)
        grid_q_values_avg = jnp.mean(grid_q_values, axis=0)
        best_q_value = jnp.max(grid_q_values_avg)

        targets = jnp.broadcast_to(
            (rewards + gamma * best_q_value)[None, :], (n, len(rewards))
        )

        grads = jax.grad(compute_loss)(params, actions, targets)
        updates, new_opt_states = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_states

    @jax.jit
    def compute_loss(
        params: dict[str, jax.Array], actions: jnp.ndarray, targets: jnp.ndarray
    ) -> jax.Array:
        values = net.apply(params, actions)
        losses = jnp.mean((values - targets) ** 2, axis=1)
        return jnp.mean(losses)

    for _ in range(iterations):
        params, opt_state = update(params, opt_state, actions, rewards)

    @jax.jit
    def ensemble_evaluator(eval_actions: jnp.ndarray) -> jnp.ndarray:
        critic_outputs = net.apply(params, eval_actions)
        return jnp.mean(critic_outputs, axis=0)

    return ensemble_evaluator


def train_min_method(
    dataset: tuple[jnp.ndarray, jnp.ndarray],
    n: int,
    iterations: int,
    gamma: float,
    key: jnp.ndarray,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    actions, rewards = dataset

    net = QNetwork(hidden_dims=[50, 50], num_critics=n)
    params = net.init(key, actions[:1])
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    grid_actions = jnp.linspace(-1, 1, 2001).reshape(-1, 1)

    @jax.jit
    def update(
        params: dict[str, jax.Array],
        opt_state: optax.OptState,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
    ) -> tuple[dict[str, jax.Array], optax.OptState]:
        grid_q_values = net.apply(params, grid_actions)
        grid_q_values_min = jnp.min(grid_q_values, axis=0)
        best_q_value = jnp.max(grid_q_values_min)

        targets = jnp.broadcast_to(
            (rewards + gamma * best_q_value)[None, :], (n, len(rewards))
        )

        grads = jax.grad(compute_loss)(params, actions, targets)
        updates, new_opt_states = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_states

    @jax.jit
    def compute_loss(
        params: dict[str, jax.Array], actions: jnp.ndarray, targets: jnp.ndarray
    ) -> jax.Array:
        values = net.apply(params, actions)
        losses = jnp.mean((values - targets) ** 2, axis=1)
        return jnp.mean(losses)

    for _ in range(iterations):
        params, opt_state = update(params, opt_state, actions, rewards)

    @jax.jit
    def ensemble_evaluator(eval_actions: jnp.ndarray) -> jnp.ndarray:
        critic_outputs = net.apply(params, eval_actions)
        return jnp.min(critic_outputs, axis=0)

    return ensemble_evaluator


def train_tqc_method(
    dataset: tuple[jnp.ndarray, jnp.ndarray],
    n: int,
    iterations: int,
    gamma: float,
    key: jnp.ndarray,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    actions, rewards = dataset

    num_quantiles = 25
    num_critics = 2
    total_kept = num_critics * (num_quantiles - n)

    net = ZNetwork(
        hidden_dims=[50, 50], n_quantiles=num_quantiles, n_critics=num_critics
    )
    params = net.init(key, actions[:1])
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    grid_actions = jnp.linspace(-1, 1, 2001).reshape(-1, 1)

    tau_levels = jnp.array(
        [(2 * m - 1) / (2 * num_quantiles) for m in range(1, num_quantiles + 1)]
    )
    all_tau = jnp.tile(tau_levels, num_critics)

    @jax.jit
    def update(
        params: dict[str, jax.Array],
        opt_state: optax.OptState,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
    ) -> tuple[dict[str, jax.Array], optax.OptState]:
        grid_quantiles = net.apply(params, grid_actions)
        grid_quantiles_flat = grid_quantiles.reshape(
            grid_quantiles.shape[0], -1
        )

        grid_sorted = jnp.sort(grid_quantiles_flat, axis=-1)
        grid_truncated = grid_sorted[:, :total_kept]
        grid_q_values = jnp.mean(grid_truncated, axis=-1)

        best_action_idx = jnp.argmax(grid_q_values)
        next_quantiles = grid_quantiles_flat[best_action_idx]
        next_sorted = jnp.sort(next_quantiles)
        next_truncated = next_sorted[:total_kept]

        targets = jnp.broadcast_to(
            rewards[:, None] + gamma * next_truncated[None, :],
            (len(rewards), total_kept),
        )

        grads = jax.grad(compute_loss)(params, actions, targets, all_tau)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state

    @jax.jit
    def compute_loss(
        params: dict[str, jax.Array],
        actions: jnp.ndarray,
        targets: jnp.ndarray,
        tau_levels: jnp.ndarray,
    ) -> jax.Array:
        current_quantiles = net.apply(params, actions)
        current_quantiles_flat = current_quantiles.reshape(
            current_quantiles.shape[0], -1
        )

        targets_expanded = targets[:, None, :]
        quantiles_expanded = current_quantiles_flat[:, :, None]

        diff = targets_expanded - quantiles_expanded
        abs_diff = jnp.abs(diff)

        huber = jnp.where(
            abs_diff <= 1.0, 0.5 * jnp.square(diff), abs_diff - 0.5
        )

        indicator = (diff < 0).astype(jnp.float32)
        tau_expanded = tau_levels[None, :, None]
        weights = jnp.abs(tau_expanded - indicator)

        weighted_loss = weights * huber
        total_loss = jnp.sum(weighted_loss)
        return total_loss / (
            total_kept * num_quantiles * num_critics * actions.shape[0]
        )

    for _ in range(iterations):
        params, opt_state = update(params, opt_state, actions, rewards)

    @jax.jit
    def ensemble_evaluator(eval_actions: jnp.ndarray) -> jnp.ndarray:
        quantiles = net.apply(params, eval_actions)
        quantiles_flat = quantiles.reshape(quantiles.shape[0], -1)
        sorted_quantiles = jnp.sort(quantiles_flat, axis=-1)
        truncated = sorted_quantiles[:, :total_kept]
        return jnp.mean(truncated, axis=-1)

    return ensemble_evaluator


def create_dataset(
    mdp: ToyMdp, buffer_size: int, seed: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    rng = np.random.default_rng(seed)
    actions = np.linspace(-1, 1, buffer_size)
    rewards = [mdp.sample_reward(a, rng) for a in actions]

    actions_tensor = jnp.array(actions, dtype=jnp.float32).reshape(-1, 1)
    rewards_tensor = jnp.array(rewards, dtype=jnp.float32)

    return actions_tensor, rewards_tensor


def robust_mean(data: np.ndarray, p1: int, p2: int) -> float:
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
        dataset = create_dataset(mdp, buffer_size, seed)

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

    # avg_data = [1, 3, 5, 10, 20, 50]
    avg_data = [1, 3, 5, 10]
    # min_data = [2, 3, 4, 6, 8, 10]
    # min_data = [2, 3, 4]
    # tqc_data = [0, 1, 2, 3, 4, 5, 6, 7, 10, 13, 16]
    # tqc_data = [0, 1, 2, 3, 4]

    experiments = list(
        itertools.chain(
            [("avg", n, train_avg_method) for n in avg_data],
            # [("min", n, train_min_method) for n in min_data],
            # [("tqc", n, train_tqc_method) for n in tqc_data],
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
            num_seed=5,
            train_fn=train_fn,
        )

    print(results)

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/tqc_figure_results.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
