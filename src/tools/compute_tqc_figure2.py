# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship.
#
# isir-internship is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import itertools
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm


class ToyMdp:
    def __init__(
        self, gamma: float, sigma: float, a0: float, a1: float, nu: float
    ) -> None:
        self.gamma = gamma
        self.sigma = sigma
        self.a0 = a0
        self.a1 = a1
        self.nu = nu

        actions = np.linspace(-1, 1, 10000)
        rewards = self.f(actions)
        self.optimal_action = actions[np.argmax(rewards)]
        self.optimal_reward = np.max(rewards)

    def f(self, actions: np.ndarray) -> np.ndarray:
        return (self.a0 + (self.a1 - self.a0) / 2 * (actions + 1)) * (
            np.sin(self.nu * actions)
        )

    def reward(self, action: float, rng: np.random.Generator) -> float:
        return self.f(action) + rng.normal(0, self.sigma)

    def q_policy(self, action: np.ndarray, policy: np.ndarray) -> np.ndarray:
        return self.f(action) + self.gamma * self.f(policy) / (1 - self.gamma)

    def q_optimal(self, action: float) -> float:
        return self.q_policy(action, self.optimal_action)


class QNetwork(nn.Module):
    def __init__(self, hidden_dims: list[int], n_critics: int) -> None:
        super().__init__()
        self.networks = nn.ModuleList()
        for _ in range(n_critics):
            layers = [nn.Linear(1, hidden_dims[0]), nn.ReLU()]
            for in_dim, out_dim in itertools.pairwise(hidden_dims):
                layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
            layers.append(nn.Linear(hidden_dims[-1], 1))
            self.networks.append(nn.Sequential(*layers))

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [net(action).squeeze(-1) for net in self.networks], axis=1
        )


class ZNetwork(nn.Module):
    def __init__(
        self, hidden_dims: list[int], n_critics: int, n_quantiles: int
    ) -> None:
        super().__init__()
        self.networks = nn.ModuleList()
        for _ in range(n_critics):
            layers = [nn.Linear(1, hidden_dims[0]), nn.ReLU()]
            for in_dim, out_dim in itertools.pairwise(hidden_dims):
                layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
            layers.append(nn.Linear(hidden_dims[-1], n_quantiles))
            self.networks.append(nn.Sequential(*layers))

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return torch.stack([net(action) for net in self.networks], axis=1)


def soft_update(target: nn.Module, current: nn.Module, tau: float) -> None:
    for target_param, main_param in zip(
        target.parameters(), current.parameters(), strict=True
    ):
        target_param.data.copy_(
            tau * main_param.data + (1.0 - tau) * target_param.data
        )


def se_loss(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.square(value - target)


def huber_loss(
    value: torch.Tensor, target: torch.Tensor, delta: float
) -> torch.Tensor:
    abs_errors = torch.abs(value - target)
    return torch.where(
        abs_errors <= delta,
        0.5 * torch.square(value - target),
        delta * abs_errors - 0.5 * delta**2,
    )


def quantile_loss(value: torch.tensor, target: torch.Tensor) -> torch.Tensor:
    batch_size, n_critics, n_quantiles = value.shape

    tau = torch.linspace(
        1.0 / (2.0 * n_quantiles), 1.0 - 1.0 / (2.0 * n_quantiles), n_quantiles
    )

    value = value.unsqueeze(-1)
    target = target.unsqueeze(1).unsqueeze(2)
    tau = tau.unsqueeze(0).unsqueeze(1).unsqueeze(3)

    loss = huber_loss(value, target, delta=1.0)
    weights = torch.abs(tau - (value - target < 0).float())
    weighted_loss = weights * loss

    loss_per_quantile = weighted_loss.sum(axis=-1) / target.shape[-1]

    return loss_per_quantile.mean(axis=(1, 2))


class AvgAgent:
    def __init__(self, n: int, gamma: float, tau: float) -> None:
        self.gamma = gamma
        self.tau = tau

        self.network = QNetwork([50, 50], n)

        self.target_network = QNetwork([50, 50], n)
        self.target_network.load_state_dict(self.network.state_dict())

        self.optim = torch.optim.Adam(self.network.parameters(), lr=1e-3)

    def set_buffer(self, dataset: list) -> None:
        self.buffer = dataset

    def update(self) -> None:
        actions, rewards = self.buffer

        targets = self.compute_targets(rewards)

        def loss_fn(network: nn.Module) -> torch.Tensor:
            values = network(actions)
            errors = se_loss(values, targets)
            return torch.mean(errors)

        loss = loss_fn(self.network)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        soft_update(self.target_network, self.network, self.tau)

    def compute_targets(self, rewards: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            grid_actions = torch.linspace(-1, 1, 2001).reshape(-1, 1)

            q_values_next_list = self.target_network(grid_actions)
            q_values_next = q_values_next_list.mean(axis=1)
            v_values_next = torch.max(q_values_next)

            return rewards + self.gamma * v_values_next

    def __call__(self, actions: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q_values_list = self.network(actions)
            q_values = q_values_list.mean(axis=1)
            return q_values.reshape(-1, 1)


class MinAgent:
    def __init__(self, n: int, gamma: float, tau: float) -> None:
        self.gamma = gamma
        self.tau = tau

        self.network = QNetwork([50, 50], n)

        self.target_network = QNetwork([50, 50], n)
        self.target_network.load_state_dict(self.network.state_dict())

        self.optim = torch.optim.Adam(self.network.parameters(), lr=1e-3)

    def set_buffer(self, dataset: list) -> None:
        self.buffer = dataset

    def update(self) -> None:
        actions, rewards = self.buffer

        targets = self.compute_targets(rewards)

        def loss_fn(network: nn.Module) -> torch.Tensor:
            values = network(actions)
            errors = se_loss(values, targets)
            return torch.mean(errors)

        loss = loss_fn(self.network)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        soft_update(self.target_network, self.network, self.tau)

    def compute_targets(self, rewards: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            grid_actions = torch.linspace(-1, 1, 2001).reshape(-1, 1)

            q_values_next_list = self.target_network(grid_actions)
            q_values_next, _ = q_values_next_list.min(axis=1)
            v_values_next = torch.max(q_values_next)

            return rewards + self.gamma * v_values_next

    def __call__(self, actions: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q_values_list = self.network(actions)
            q_values, _ = q_values_list.min(axis=1)
            return q_values.reshape(-1, 1)


class TqcAgent:
    def __init__(self, n: int, gamma: float, tau: float) -> None:
        self.gamma = gamma
        self.tau = tau

        self.num_network = 2
        self.num_quantiles = 25
        self.total_kept = self.num_network * (self.num_quantiles - n)

        self.network = ZNetwork([50, 50], self.num_network, self.num_quantiles)

        self.target_network = ZNetwork(
            [50, 50], self.num_network, self.num_quantiles
        )
        self.target_network.load_state_dict(self.network.state_dict())

        self.optim = torch.optim.Adam(self.network.parameters(), lr=1e-3)

    def set_buffer(self, dataset: list) -> None:
        self.buffer = dataset

    def update(self) -> None:
        actions, rewards = self.buffer

        targets = self.compute_targets(rewards)

        def loss_fn(network: nn.Module) -> torch.Tensor:
            values = network(actions)
            errors = quantile_loss(values, targets)
            return torch.mean(errors)

        loss = loss_fn(self.network)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        soft_update(self.target_network, self.network, self.tau)

    def compute_targets(self, rewards: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            grid_actions = torch.linspace(-1, 1, 2001).reshape(-1, 1)

            quantiles = self.target_network(grid_actions)

            union_quantiles = quantiles.reshape(grid_actions.shape[0], -1)
            sorted_quantiles, _ = torch.sort(union_quantiles, axis=-1)
            truncated_quantiles = sorted_quantiles[:, : self.total_kept]

            v_values_next, _ = torch.max(truncated_quantiles, axis=0)

            return rewards + self.gamma * v_values_next

    def __call__(self, actions: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            quantiles = self.network(actions)
            q_values = quantiles.mean(axis=(1, 2))
            return q_values.reshape(-1, 1)


def create_dataset(
    mdp: ToyMdp, buffer_size: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    actions = np.linspace(-1, 1, buffer_size)
    rewards = [mdp.reward(a, rng) for a in actions]

    actions_tensor = torch.tensor(actions, dtype=torch.float32).reshape(-1, 1)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32).reshape(-1, 1)

    return actions_tensor, rewards_tensor


def robust_mean(data: np.ndarray, p1: int, p2: int) -> float:
    q1, q2 = np.quantile(data, [p1, p2])
    result = data[(data >= q1) & (data <= q2)]
    return np.mean(result) if len(result) > 0 else np.mean(data)


def compute_bias_variance(
    n: int,
    mdp: ToyMdp,
    tau: float,
    buffer_size: int,
    total_steps: int,
    eval_freq: int,
    num_seed: int,
    create_agent: any,
    progress_bar: tqdm,
    method: str,
) -> None:
    eval_actions = np.linspace(-1, 1, 2000).reshape(-1, 1)
    eval_actions_flat = eval_actions.flatten()
    true_qs = np.array([mdp.q_optimal(a) for a in eval_actions_flat])

    step_results = {}
    for step in range(0, total_steps + 1, eval_freq):
        step_results[step] = {
            "actions": [],
            "optimal_action": [],
            "true_qs": [],
            "predicted_qs": [],
            "policy_action": [],
            "policy_qs": [],
        }

    for seed in range(num_seed):
        torch.manual_seed(seed)
        dataset = create_dataset(mdp, buffer_size, seed)

        agent = create_agent(n, mdp.gamma, tau)
        agent.set_buffer(dataset)

        for step in range(0, total_steps + 1, eval_freq):
            predicted_qs = agent(torch.Tensor(eval_actions)).numpy().flatten()
            policy_action = eval_actions_flat[np.argmax(predicted_qs)]
            policy_qs = mdp.q_policy(eval_actions_flat, policy_action)

            step_results[step]["actions"].append(eval_actions_flat.copy())
            step_results[step]["optimal_action"].append(mdp.optimal_action)
            step_results[step]["true_qs"].append(true_qs.copy())
            step_results[step]["predicted_qs"].append(predicted_qs)
            step_results[step]["policy_action"].append(policy_action)
            step_results[step]["policy_qs"].append(policy_qs)

            progress_bar.set_description(
                f"Running ({method} n={n}), seed={seed + 1:02d}/{num_seed:02d}, step={step:04d}/{total_steps}"
            )

            if step < total_steps:
                for _ in range(eval_freq):
                    agent.update()
                    progress_bar.update(1)

    return step_results


def main() -> None:
    num_seed = 50
    tau = 0.05
    total_steps = 5_000
    buffer_size = 50
    eval_freq = 50

    mdp = ToyMdp(gamma=0.99, sigma=0.25, a0=0.3, a1=0.9, nu=5.0)

    avg_data = [1, 3, 5, 10, 20, 50]
    min_data = [2, 3, 4, 6, 8, 10]
    tqc_data = [0, 1, 2, 3, 4, 5, 6, 7, 10, 13, 16]

    experiments = list(
        itertools.chain(
            [("avg", n, AvgAgent) for n in avg_data],
            [("min", n, MinAgent) for n in min_data],
            [("tqc", n, TqcAgent) for n in tqc_data],
        )
    )

    total_iterations = len(experiments) * num_seed * total_steps
    progress_bar = tqdm(total=total_iterations, desc="Running experiments")

    results = {}
    for method, n, create_agent in experiments:
        results[(method, n)] = compute_bias_variance(
            n=n,
            mdp=mdp,
            tau=tau,
            buffer_size=buffer_size,
            total_steps=total_steps,
            eval_freq=eval_freq,
            num_seed=num_seed,
            create_agent=create_agent,
            progress_bar=progress_bar,
            method=method,
        )

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/tqc_figure_results.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
