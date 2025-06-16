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


class QNetwork(nn.Module):
    __slots__ = ["model"]

    def __init__(
        self,
        state_dim: int,
        hidden_dims: list[int],
        action_dim: int,
    ) -> None:
        super().__init__()
        layers = [nn.Linear(state_dim + action_dim, hidden_dims[0]), nn.ReLU()]

        for in_dim, out_dim in itertools.pairwise(hidden_dims):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])

        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.model = nn.Sequential(*layers)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        state_action = torch.cat([state, action], dim=1)
        return self.model(state_action).squeeze(-1)


class PolicyNetwork(nn.Module):
    __slots__ = [
        "backbone",
        "log_std",
        "log_std_head",
        "log_std_max",
        "log_std_min",
        "mean_head",
    ]

    def __init__(
        self,
        state_dim: int,
        hidden_dims: list[int],
        action_dim: int,
    ) -> None:
        super().__init__()
        layers = [nn.Linear(state_dim, hidden_dims[0]), nn.ReLU()]
        for in_dim, out_dim in itertools.pairwise(hidden_dims):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        self.backbone = nn.Sequential(*layers)

        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)

        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(state)
        mean = self.mean_head(features)
        log_std = torch.clamp(
            self.log_std_head(features), self.log_std_min, self.log_std_max
        )

        return mean, log_std


class SAC:
    __slots__ = [
        "action_dim",
        "alpha_optimizer",
        "batch_size",
        "gamma",
        "log_alpha",
        "policy_network",
        "policy_optimizer",
        "q_network1",
        "q_network2",
        "q_optimizer1",
        "q_optimizer2",
        "q_target_network1",
        "q_target_network2",
        "replay_buffer",
        "state_dim",
        "target_entropy",
        "tau",
    ]

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        hidden_dims: list[int],
        replay_size: int,
        batch_size: int,
        q_lr: float,
        policy_lr: float,
        alpha_lr: float,
        tau: float,
        gamma: float,
        weights: dict | None = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.q_network1 = QNetwork(state_dim, hidden_dims, action_dim)
        self.q_network2 = QNetwork(state_dim, hidden_dims, action_dim)

        self.policy_network = PolicyNetwork(state_dim, hidden_dims, action_dim)

        self.q_target_network1 = QNetwork(state_dim, hidden_dims, action_dim)
        self.q_target_network2 = QNetwork(state_dim, hidden_dims, action_dim)

        self.q_target_network1.load_state_dict(self.q_network1.state_dict())
        self.q_target_network2.load_state_dict(self.q_network2.state_dict())

        self.log_alpha = nn.Parameter(torch.zeros(1))

        self.q_optimizer1 = torch.optim.Adam(
            self.q_network1.parameters(), lr=q_lr
        )
        self.q_optimizer2 = torch.optim.Adam(
            self.q_network2.parameters(), lr=q_lr
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=policy_lr
        )
        self.alpha_optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=policy_lr
        )

        self.replay_buffer = ReplayBuffer(replay_size)
        self.batch_size = batch_size

        self.target_entropy = -action_dim
        self.tau = tau
        self.gamma = gamma

        if weights is not None:
            self.load_from_weights(weights)

    def select_action(
        self, state: np.ndarray, *, evaluation: bool = False
    ) -> np.ndarray:
        state = torch.as_tensor(state, dtype=torch.float32)
        mean, log_std = self.policy_network.forward(state)

        if evaluation:
            action = mean
        else:
            std = log_std.exp()
            normal = dist.Normal(mean, std)
            action = normal.rsample()

        return np.tanh(action.detach().numpy())

    def update() -> None:
        pass

    def get_weights(self) -> dict:
        return {
            "q1": self.q_network1.state_dict(),
            "q2": self.q_network2.state_dict(),
            "policy": self.policy_network.state_dict(),
            "log_alpha": self.log_alpha.item(),
        }

    def load_from_weights(self, weights: dict) -> None:
        self.q_network1.load_state_dict(weights["q1"])
        self.q_network2.load_state_dict(weights["q2"])
        self.q_target_network1.load_state_dict(weights["q1"])
        self.q_target_network2.load_state_dict(weights["q2"])
        self.policy_network.load_state_dict(weights["policy"])
        self.log_alpha.data = torch.tensor(weights["log_alpha"])
