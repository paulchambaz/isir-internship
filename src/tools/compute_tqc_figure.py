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

import numpy as np
import torch
from torch import nn
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


class QNetwork(nn.Module):
    __slots__ = ["actions"]

    def __init__(self, hidden_dims: list[int]) -> None:
        super().__init__()
        layers = [nn.Linear(1, hidden_dims[0]), nn.ReLU()]
        for in_dim, out_dim in itertools.pairwise(hidden_dims):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.network = nn.Sequential(*layers)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return self.network(action).squeeze(-1)


class ZNetwork(nn.Module):
    __slots__ = ["networks"]

    def __init__(
        self, hidden_dims: list[int], num_quantiles: int, num_networks: int
    ) -> None:
        super().__init__()
        self.networks = nn.ModuleList()
        for _ in range(num_networks):
            layers = [nn.Linear(1, hidden_dims[0]), nn.ReLU()]
            for in_dim, out_dim in itertools.pairwise(hidden_dims):
                layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
            layers.append(nn.Linear(hidden_dims[-1], num_quantiles))
            self.networks.append(nn.Sequential(*layers))

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return torch.cat([network(action) for network in self.networks], dim=-1)


def train_avg_method(
    dataset: tuple[torch.Tensor, torch.Tensor],
    n: int,
    iterations: int,
    gamma: float,
) -> nn.Module:
    actions, rewards = dataset
    networks = [QNetwork([50, 50]) for _ in range(n)]
    optims = [torch.optim.Adam(net.parameters(), lr=1e-3) for net in networks]

    for _ in range(iterations):
        current_q_values = [network(actions) for network in networks]

        with torch.no_grad():
            grid_actions = torch.linspace(-1, 1, 2001).unsqueeze(-1)
            grid_q_values = torch.stack(
                [network(grid_actions) for network in networks]
            ).mean(dim=0)
            best_action_idx = torch.argmax(grid_q_values)
            best_q_value = grid_q_values[best_action_idx]

        targets = rewards + gamma * best_q_value

        for opt, q_vals in zip(optims, current_q_values, strict=True):
            loss = nn.MSELoss()(q_vals, targets.detach())
            opt.zero_grad()
            loss.backward()
            opt.step()

    class MeanEnsembleNetwork:
        def __init__(self, networks: nn.Module) -> None:
            self.networks = networks

        def __call__(self, actions: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                outputs = [net(actions) for net in self.networks]
                return torch.stack(outputs).mean(dim=0)

    return MeanEnsembleNetwork(networks)


def train_min_method(
    dataset: tuple[torch.Tensor, torch.Tensor],
    n: int,
    iterations: int,
    gamma: float,
) -> nn.Module:
    actions, rewards = dataset
    networks = [QNetwork([50, 50]) for _ in range(n)]
    optims = [torch.optim.Adam(net.parameters(), lr=1e-3) for net in networks]

    for _ in range(iterations):
        current_q_values = [network(actions) for network in networks]

        with torch.no_grad():
            grid_actions = torch.linspace(-1, 1, 2001).unsqueeze(-1)
            grid_q_values, _ = torch.stack(
                [network(grid_actions) for network in networks]
            ).min(dim=0)
            best_action_idx = torch.argmax(grid_q_values)
            best_q_value = grid_q_values[best_action_idx]

        targets = rewards + gamma * best_q_value

        for opt, q_vals in zip(optims, current_q_values, strict=True):
            loss = nn.MSELoss()(q_vals, targets.detach())
            opt.zero_grad()
            loss.backward()
            opt.step()

    class MinEnsembleNetwork:
        def __init__(self, networks: nn.Module) -> None:
            self.networks = networks

        def __call__(self, actions: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                outputs = [net(actions) for net in self.networks]
                return torch.stack(outputs).min(dim=0)[0]

    return MinEnsembleNetwork(networks)


def train_tqc_method(
    dataset: tuple[torch.Tensor, torch.Tensor],
    n: int,
    iterations: int,
    gamma: float,
) -> nn.Module:
    actions, rewards = dataset
    num_quantiles = 25
    num_networks = 2
    k = num_quantiles - n
    kN = k * num_networks

    network = ZNetwork([50, 50], num_quantiles, num_networks)
    optim = torch.optim.Adam(network.parameters(), lr=1e-3)

    tau_levels = torch.tensor(
        [(2 * m - 1) / (2 * num_quantiles) for m in range(1, num_quantiles + 1)]
    )
    tau_all = tau_levels.repeat(num_networks)

    for _ in range(iterations):
        current_z_values = network(actions)

        with torch.no_grad():
            grid_actions = torch.linspace(-1, 1, 2001).unsqueeze(-1)
            grid_z_values = network(grid_actions)

            grid_z_sorted, _ = torch.sort(grid_z_values, dim=-1)
            grid_z_truncated = grid_z_sorted[:, :kN]
            grid_q_values = grid_z_truncated.mean(dim=-1)

            best_action_idx = torch.argmax(grid_q_values)

            next_z_all = grid_z_values[best_action_idx]
            next_z_sorted, _ = torch.sort(next_z_all)
            next_z_truncated = next_z_sorted[:kN]

        targets = rewards.unsqueeze(-1) + gamma * next_z_truncated.unsqueeze(0)

        targets_expanded = targets.unsqueeze(1)
        current_z_expanded = current_z_values.unsqueeze(-1)
        diff = targets_expanded - current_z_expanded

        abs_diff = torch.abs(diff)
        huber = torch.where(abs_diff <= 1.0, 0.5 * diff * diff, abs_diff - 0.5)

        indicator = (diff < 0).float()
        tau_expanded = tau_all.unsqueeze(0).unsqueeze(-1)
        weight = torch.abs(tau_expanded - indicator)

        total_loss = torch.sum(weight * huber)

        loss = total_loss / (
            kN * num_quantiles * num_networks * rewards.shape[0]
        )

        optim.zero_grad()
        loss.backward()
        optim.step()

    class TQCEvaluator:
        def __init__(self, network: nn.Module, k: int, num_nets: int) -> None:
            self.network = network
            self.k = k
            self.num_nets = num_nets
            self.kN = k * num_nets

        def __call__(self, actions: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                z_values = self.network(actions)
                z_sorted, _ = torch.sort(z_values, dim=-1)
                z_truncated = z_sorted[:, : self.kN]
                return z_truncated.mean(dim=-1)

    return TQCEvaluator(network, k, num_networks)


def create_dataset(
    mdp: ToyMdp, buffer_size: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    actions = np.linspace(-1, 1, buffer_size)
    rewards = [mdp.sample_reward(a, rng) for a in actions]

    actions_tensor = torch.tensor(actions, dtype=torch.float32).unsqueeze(-1)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

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
        [tuple[torch.Tensor, torch.Tensor], int, int, float], nn.Module
    ],
) -> None:
    errors = []
    optim_actions = []

    for seed in range(num_seed):
        torch.manual_seed(seed)
        dataset = create_dataset(mdp, buffer_size, seed)
        trained_net = train_fn(dataset, n, iterations, mdp.gamma)

        eval_actions = torch.linspace(-1, 1, 2000).unsqueeze(-1)
        predicted_q = trained_net(eval_actions).numpy()
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
    tqc_data = [0, 1, 2, 3, 4, 5, 6, 7, 10, 13, 16]

    experiments = list(
        itertools.chain(
            [("avg", n, train_avg_method) for n in avg_data],
            [("min", n, train_min_method) for n in min_data],
            [("tqc", n, train_tqc_method) for n in tqc_data],
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
            num_seed=10,
            train_fn=train_fn,
        )

    print(results)

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/tqc_figure_results.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
