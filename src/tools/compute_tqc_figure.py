# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship.
#
# isir-internship is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import argparse
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
    n_critics: int

    @nn.compact
    def __call__(self, actions: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack(
            [
                MLP(
                    hidden_dims=self.hidden_dims,
                    output_dim=1,
                )(actions).squeeze(-1)
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

        dummy_input = jnp.zeros((1, 1))
        self.params = self.network.init(key, dummy_input)
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

        dummy_input = jnp.zeros((1, 1))
        self.params = self.network.init(key, dummy_input)
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

        dummy_input = jnp.zeros((1, 1))
        self.params = self.network.init(key, dummy_input)
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

        dummy_input = jnp.zeros((1, 1))
        self.params = self.network.init(key, dummy_input)
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

        dummy_input = jnp.zeros((1, 1))
        self.params = self.network.init(key, dummy_input)
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
    n: int | float,
    mdp: ToyMdp,
    buffer_size: int,
    total_steps: int,
    eval_freq: int,
    num_seed: int,
    create_ensemble_fn: Callable,
) -> dict:
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

            eval_actions = jnp.linspace(-1, 1, 2000).reshape(-1, 1)

            predicted_q = np.array(ensemble(eval_actions))
            true_q = np.array(
                [mdp.true_q_value(a.item()) for a in eval_actions]
            )

            error = predicted_q - true_q
            bias = np.mean(error)
            variance = np.var(error)

            optim_action = eval_actions[np.argmax(predicted_q)].item()
            policy_error = np.abs(optim_action - mdp.optimal_action)

            if step not in step_results:
                step_results[step] = []
            step_results[step].append((bias, variance, policy_error))

    final_results = {}
    for step, results in step_results.items():
        res = np.array(results)
        biases = res[:, 0]
        variances = res[:, 1]
        policy_errors = res[:, 2]

        final_results[step] = (
            robust_mean(biases, 0.1, 0.9),
            robust_mean(variances, 0.1, 0.9),
            robust_mean(policy_errors, 0.1, 0.9),
        )

    return final_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute TQC figure")
    parser.add_argument(
        "--cpu", action="store_true", help="Force JAX to use CPU instead of GPU"
    )
    args = parser.parse_args()
    if args.cpu:
        jax.config.update("jax_platform_name", "cpu")

    mdp = ToyMdp(gamma=0.99, sigma=0.25, a0=0.3, a1=0.9, nu=5.0)

    avg_data = [1, 3, 5, 10, 20, 50]
    min_data = [2, 3, 4, 6, 8, 10]
    tqc_data = [0, 1, 2, 3, 4, 5, 6, 7, 10, 13, 16]
    top_data = [-1.5, -1.0, -0.7, -0.5, 0.0, 0.5, 1.0]

    experiments = list(
        itertools.chain(
            [("avg", n, create_avg) for n in avg_data],
            [("min", n, create_min_ensemble) for n in min_data],
            # [("tqc", n, create_tqc_ensemble) for n in tqc_data],
            # [("ttqc", n, create_ttqc_ensemble) for n in tqc_data],
            # [("ndtop", beta, create_ndtop_ensemble) for beta in top_data],
            # [("top", beta, create_top_ensemble) for beta in top_data],
        )
    )

    results = {}

    progress_bar = tqdm(experiments, desc="Running experiments")

    for method, n, create_ensemble_fn in progress_bar:
        progress_bar.set_postfix({"method": method, "n": n})
        # results[(method, n)] = compute_bias_variance(
        #     n=n,
        #     mdp=mdp,
        #     buffer_size=50,
        #     total_steps=10000,
        #     eval_freq=100,
        #     num_seed=10,
        #     create_ensemble_fn=create_ensemble_fn,
        # )
        results[(method, n)] = compute_bias_variance(
            n=n,
            mdp=mdp,
            buffer_size=50,
            total_steps=20000,
            eval_freq=100,
            num_seed=100,
            create_ensemble_fn=create_ensemble_fn,
        )

    for key, value in results.items():
        print(f"\n# {key}\n")
        for k, v in value.items():
            print(f"{k}\t{v}")

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/tqc_figure_results.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
