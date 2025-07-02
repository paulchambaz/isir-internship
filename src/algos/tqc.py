# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship.
#
# isir-internship is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import itertools

import numpy as np
import torch
import torch.distributions as dist
from torch import nn

from .replay import ReplayBuffer
from .rl_algo import RLAlgo
from .utils import soft_update_target


class TQC(RLAlgo):
    __slots__ = [
        "action_dim",
        "batch_size",
        "gamma",
        "learn_temperature",
        "log_alpha",
        "n_critics",
        "n_quantiles",
        "policy_network",
        "policy_optimizer",
        "quantiles_drop",
        "replay_buffer",
        "state_dim",
        "target_entropy",
        "tau",
        "temperature_optimizer",
        "z_network",
        "z_optimizer",
        "z_target_network",
    ]

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        hidden_dims: list[int],
        replay_size: int,
        batch_size: int,
        critic_lr: float,
        policy_lr: float,
        temperature_lr: float,
        tau: float,
        gamma: float,
        alpha: float | None,
        n_quantiles: int,
        n_critics: int,
        quantiles_drop: int,
        state: dict | None = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_quantiles = n_quantiles
        self.n_critics = n_critics
        self.quantiles_drop = quantiles_drop
        self.learn_temperature = alpha is None

        self.z_network = self.ZNetwork(
            state_dim, hidden_dims, action_dim, n_quantiles, n_critics
        )
        self.z_target_network = self.ZNetwork(
            state_dim, hidden_dims, action_dim, n_quantiles, n_critics
        )
        self.z_target_network.load_state_dict(self.z_network.state_dict())

        self.policy_network = self.PolicyNetwork(
            state_dim, hidden_dims, action_dim
        )

        self.log_alpha = (
            nn.Parameter(torch.zeros(1))
            if self.learn_temperature
            else torch.log(torch.tensor(alpha))
        )

        self.z_optimizer = torch.optim.Adam(
            self.z_network.parameters(), lr=critic_lr
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=policy_lr
        )
        self.temperature_optimizer = (
            torch.optim.Adam([self.log_alpha], lr=temperature_lr)
            if self.learn_temperature
            else None
        )

        self.replay_buffer = ReplayBuffer(replay_size)
        self.batch_size = batch_size

        self.target_entropy = -action_dim
        self.tau = tau
        self.gamma = gamma

        if state is not None:
            self.load_from_state(state)

    def select_action(self, state: np.ndarray, evaluation: bool) -> np.ndarray:
        """
        Selects an action in range [-1, 1] using deterministic mean
        (evaluation=True) or stochastic sampling (evaluation=False).

        Args:
            state: Current environment state observation
            evaluation: Whether to use deterministic (True) or stochastic (False)
             action selection
        """
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        mean, log_std = self.policy_network(state)

        action = (
            mean
            if evaluation
            else dist.Normal(mean, log_std.exp() + 1e-8).rsample()
        )

        return torch.tanh(action).squeeze(0).detach().numpy()

    def push_buffer(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> None:
        """
        Performs one training step updating Q-networks, policy network, and
        temperature using replay buffer samples.
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = (
            self.replay_buffer.sample(self.batch_size)
        )

        z_loss = self._compute_z_loss(
            states, actions, rewards, next_states, dones
        )
        self.z_optimizer.zero_grad()
        z_loss.backward()
        self.z_optimizer.step()

        soft_update_target(self.z_target_network, self.z_network, self.tau)

        policy_loss = self._compute_policy_loss(states)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.learn_temperature:
            temperature_loss = self._compute_temperature_loss(states)
            self.temperature_optimizer.zero_grad()
            temperature_loss.backward()
            self.temperature_optimizer.step()

    def get_state(self) -> dict:
        """
        Returns dictionary containing all agent state for saving or transferring.
        """
        state = {
            "z": self.z_network.state_dict(),
            "z_target": self.z_target_network.state_dict(),
            "policy": self.policy_network.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "z_optimizer": self.z_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "replay_buffer": self.replay_buffer.get_data(),
        }

        if self.learn_temperature:
            state["temperature_optimizer"] = (
                self.temperature_optimizer.state_dict()
            )

        return state

    def load_from_state(self, state: dict) -> None:
        """
        Restores complete agent state from state dictionary.

        Args:
            state: Dictionary of agent state from get_state
        """
        self.z_network.load_state_dict(state["z"])
        self.z_target_network.load_state_dict(state["z_target"])
        self.policy_network.load_state_dict(state["policy"])
        self.log_alpha.data = torch.tensor(state["log_alpha"])
        self.z_optimizer.load_state_dict(state["z_optimizer"])
        self.policy_optimizer.load_state_dict(state["policy_optimizer"])
        self.replay_buffer.load_data(state["replay_buffer"])

        if self.learn_temperature:
            self.temperature_optimizer.load_state_dict(
                state["temperature_optimizer"]
            )

    def _compute_z_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        # pi(s'), log pi (pi(s') | s')
        next_actions, log_probs = self._compute_action_and_log_prob(next_states)

        with torch.no_grad():
            nz_targets = self.z_target_network(next_states, next_actions)
            all_nz_targets = nz_targets.view(nz_targets.shape[0], -1)
            sorted_nz_targets, _ = torch.sort(all_nz_targets, dim=1)
            n_quantiles = self.n_quantiles * (
                self.n_critics - self.quantiles_drop
            )
            truncated_nz_targets = sorted_nz_targets[:, :n_quantiles]

            alpha = self.log_alpha.exp() + 1e-8
            z_targets = rewards.unsqueeze(1) + self.gamma * (
                1.0 - dones.float().unsqueeze(1)
            ) * (truncated_nz_targets - alpha * log_probs.unsqueeze(1))

        z_values = self.z_network(states, actions)

        return self._quantile_huber_loss(z_values, z_targets)

    def _compute_policy_loss(self, states: torch.Tensor) -> torch.Tensor:
        actions, log_probs = self._compute_action_and_log_prob(states)

        z_values = self.z_network(states, actions)
        q_values = z_values.mean(dim=(1, 2))

        alpha = self.log_alpha.exp() + 1e-8
        return torch.mean(alpha * log_probs - q_values)

    def _compute_temperature_loss(self, states: torch.Tensor) -> torch.Tensor:
        _, log_probs = self._compute_action_and_log_prob(states)

        alpha = self.log_alpha.exp() + 1e-8
        return torch.mean(-alpha * (log_probs.detach() + self.target_entropy))

    def _compute_action_and_log_prob(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        #  N(mu_theta (s), sigma_theta (s)^2)
        mean, log_std = self.policy_network(state)
        gaussian = dist.Normal(mean, log_std.exp())

        # a = tanh(u), u ~ N
        raw_action = gaussian.rsample()
        action = torch.tanh(raw_action)

        # log pi(a|s) = log pi(u|s) - sum_i log(1 - a_i)^2
        log_prob_gaussian = gaussian.log_prob(raw_action)
        clamped_action = torch.clamp(action, min=-1.0 + 1e-6, max=1.0 - 1e-6)
        tanh_correction = torch.log1p(-(clamped_action**2))
        log_prob = (log_prob_gaussian - tanh_correction).sum(dim=-1)

        return action, log_prob

    def _quantile_huber_loss(
        self, z_values: torch.Tensor, z_targets: torch.Tensor
    ) -> torch.Tensor:
        current_expanded = z_values.unsqueeze(-1)
        target_expanded = z_targets.unsqueeze(1).unsqueeze(1)

        tau = torch.linspace(
            1 / (2 * self.n_quantiles),
            1 - 1 / (2 * self.n_quantiles),
            self.n_quantiles,
        ).view(1, 1, -1, 1)

        diff = target_expanded - current_expanded
        abs_diff = torch.abs(diff)

        huber = torch.where(abs_diff <= 1.0, 0.5 * diff * diff, abs_diff - 0.5)

        indicator = (diff < 0).float()
        weights = torch.abs(tau - indicator)

        weighted_loss = weights * huber
        return weighted_loss.mean()

    class ZNetwork(nn.Module):
        __slots__ = ["networks"]

        def __init__(
            self,
            state_dim: int,
            hidden_dims: list[int],
            action_dim: int,
            n_quantiles: int,
            n_critics: int,
        ) -> None:
            super().__init__()
            self.networks = nn.ModuleList()

            for _ in range(n_critics):
                layers = [
                    nn.Linear(state_dim + action_dim, hidden_dims[0]),
                    nn.ReLU(),
                ]
                for in_dim, out_dim in itertools.pairwise(hidden_dims):
                    layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
                layers.append(nn.Linear(hidden_dims[-1], n_quantiles))

                self.networks.append(nn.Sequential(*layers))

        def forward(
            self, state: torch.Tensor, action: torch.Tensor
        ) -> torch.Tensor:
            state_action = torch.cat([state, action], dim=1)
            return torch.stack(
                [network(state_action) for network in self.networks], dim=1
            )

    class PolicyNetwork(nn.Module):
        __slots__ = ["action_dim", "model"]

        def __init__(
            self, state_dim: int, hidden_dims: list[int], action_dim: int
        ) -> None:
            super().__init__()
            layers = [nn.Linear(state_dim, hidden_dims[0]), nn.ReLU()]
            for in_dim, out_dim in itertools.pairwise(hidden_dims):
                layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
            layers.append(nn.Linear(hidden_dims[-1], action_dim * 2))

            self.model = nn.Sequential(*layers)
            self.action_dim = action_dim

        def forward(
            self, state: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            output = self.model(state)
            mean, log_std = torch.split(output, self.action_dim, dim=-1)
            log_std = torch.clamp(log_std, -20, 2)
            return mean, log_std
