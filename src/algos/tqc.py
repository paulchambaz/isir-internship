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


class TQC:
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

    __slots__ = [
        "action_dim",
        "batch_size",
        "critic_network",
        "critic_optimizer",
        "critic_target_network",
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

        self.critic_network = self.ZNetwork(
            state, hidden_dims, action_dim, n_quantiles, n_critics
        )
        self.critic_target_network = self.ZNetwork(
            state_dim, hidden_dims, action_dim, n_quantiles, n_critics
        )
        self.critic_target_network.load_state_dict(
            self.critic_network.state_dict()
        )

        self.policy_network = self.PolicyNetwork(
            state_dim, hidden_dims, action_dim
        )

        self.log_alpha = (
            nn.Parameter(torch.zeros(1))
            if self.learn_temperature
            else torch.log(torch.tensor(alpha))
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic_network.parameters(), lr=critic_lr
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
