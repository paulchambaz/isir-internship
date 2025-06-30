# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship.
#
# isir-internship is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.


import itertools
import math

import numpy as np
import torch
from torch import nn

from .algo import Algo
from .replay import ReplayBuffer
from .utils import soft_update_target


class AFU(Algo):
    """
    Actor-Free critic Updates (AFU) algorithm for continuous control tasks.

    AFU uses a V function for state evaluation in the Bellman TD equation with
    Q=V+A decomposition, where A represents regret of not taking the best
    action. It employs conditional gradient rescaling to ensure V is a tight
    bound of max Q and conditional optimization to enforce negative A values,
    while maintaining a stochastic policy that maximizes both expected return
    and entropy.

    Args:
        action_dim: Dimensionality of the action space
        state_dim: Dimensionality of the state space
        hidden_dims: List of hidden layer sizes for neural networks
        replay_size: Maximum size of the replay buffer
        batch_size: Number of samples per training batch
        critic_lr: Learning rate for Q, V, and A function networks
        policy_lr: Learning rate for policy network
        temperature_lr: Learning rate for temperature parameter
        tau: Target network soft update coefficient (0 < tau <= 1)
        rho: Conditional gradient rescaling coefficient for V-A constraints
        gamma: Discount factor for future rewards (0 < gamma <= 1)
        alpha: If None, learn alpha, if float, use fixed alpha value
        state: Optional pre-trained network state dictionary
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        critic_lr: float,
        policy_lr: float,
        temperature_lr: float,
        tau: float,
        gamma: float,
        rho: float,
        replay_size: int,
        batch_size: int,
        alpha: float | None,
        state: dict | None = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learn_temperature = alpha is None

        self.q_network = self.QNetwork(
            state_dim, action_dim, hidden_dims, num_critics=3
        )

        self.v_network = self.VNetwork(state_dim, hidden_dims, num_critics=2)
        self.v_target_network = self.VNetwork(
            state_dim, hidden_dims, num_critics=2
        )
        self.v_target_network.load_state_dict(self.v_network.state_dict())

        self.policy_network = self.PolicyNetwork(
            state_dim,
            action_dim,
            hidden_dims,
            log_std_min=-10.0,
            log_std_max=2.0,
        )

        self.log_alpha = (
            nn.Parameter(torch.zeros(1))
            if self.learn_temperature
            else torch.log(torch.tensor(alpha))
        )

        self.q_optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=critic_lr
        )
        self.v_optimizer = torch.optim.Adam(
            self.v_network.parameters(), lr=critic_lr
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
        self.rho = rho
        self.gamma = gamma

    def select_action(
        self, state: np.ndarray, *, evaluation: bool
    ) -> np.ndarray:
        """
        Selects an action in range [-1, 1] using deterministic mean
        (evaluation=True) or stochastic sampling (evaluation=False).

        Args:
            state: Current environment state observation
            evaluation: Whether to use deterministic (True) or stochastic (False)
             action selection
        """
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            mean, log_std = self.policy_network(state_tensor)
            if evaluation:
                action = torch.tanh(mean)
            else:
                raw_action = torch.distributions.Normal(
                    mean, log_std.exp()
                ).rsample()
                action = torch.tanh(raw_action)
        return action.squeeze(0).numpy()

    def push_buffer(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,  # noqa: FBT001
    ) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> None:
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = (
            self.replay_buffer.sample(self.batch_size)
        )

        critic_loss = self._compute_critic_loss(
            states, actions, rewards, next_states, dones
        )

        self.v_optimizer.zero_grad()
        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.v_optimizer.step()
        self.q_optimizer.step()

        soft_update_target(self.v_target_network, self.v_network, self.tau)

        policy_loss, temperature_loss = self._compute_policy_loss(states)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.learn_temperature:
            self.temperature_optimizer.zero_grad()
            temperature_loss.backward()
            self.temperature_optimizer.step()

    def get_state(self) -> dict:
        state = {
            "q_network": self.q_network.state_dict(),
            "v_network": self.v_network.state_dict(),
            "v_target_network": self.v_target_network.state_dict(),
            "policy_network": self.policy_network.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "replay_buffer": self.replay_buffer.get_data(),
        }

        state["log_alpha"] = (
            self.log_alpha.item() if self.learn_temperature else self.log_alpha
        )

        if self.learn_temperature:
            state["temperature_optimizer"] = (
                self.temperature_optimizer.state_dict()
            )

        return state

    def load_from_state(self, state: dict) -> None:
        self.q_network.load_state_dict(state["q_network"])
        self.v_network.load_state_dict(state["v_network"])
        self.v_target_network.load_state_dict(state["v_target_network"])
        self.policy_network.load_state_dict(state["policy_network"])
        self.q_optimizer.load_state_dict(state["q_optimizer"])
        self.v_optimizer.load_state_dict(state["v_optimizer"])
        self.policy_optimizer.load_state_dict(state["policy_optimizer"])
        self.replay_buffer.load_data(state["replay_buffer"])

        if self.learn_temperature:
            self.log_alpha.data = torch.tensor(state["log_alpha"])
            self.temperature_optimizer.load_state_dict(
                state["temperature_optimizer"]
            )
        else:
            self.log_alpha = state["log_alpha"]

    def _compute_critic_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_v_values = self.v_target_network(next_states)
            v_targets = torch.min(torch.stack(next_v_values), dim=0)[0]
            q_targets = rewards + self.gamma * (1 - dones.float()) * v_targets

        v_values = self.v_network(states)
        optim_values = torch.stack(v_values)

        q_values = self.q_network(states, actions)
        q_final = q_values[-1]
        optim_advantages = -torch.stack(q_values[:-1])

        indicator = (
            (optim_values + optim_advantages < q_targets.unsqueeze(0))
            .detach()
            .float()
        )
        upsilon_values = (
            1 - self.rho * indicator
        ) * optim_values + self.rho * indicator * optim_values.detach()

        # upsilon_values = (1 - indicator) * (
        #     ((1 - self.rho) * optim_values).detach() + self.rho * optim_values
        # ) + indicator * optim_values

        target_diff = upsilon_values - q_targets.unsqueeze(0)
        z_values = torch.where(
            (optim_values >= q_targets.unsqueeze(0)).detach(),
            (optim_advantages + target_diff) ** 2,
            optim_advantages**2 + target_diff**2,
        )
        va_loss = torch.mean(z_values)

        q_loss = torch.mean((q_targets - q_final) ** 2)

        return va_loss + q_loss

    def _compute_policy_loss(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.policy_network(states)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        raw_action = normal.rsample()
        actions = torch.tanh(raw_action)

        log_prob = normal.log_prob(raw_action) - torch.log(
            torch.relu(1 - actions.pow(2)) + 1e-6
        )
        log_probs = log_prob.sum(dim=-1, keepdim=True)

        q_values = self.q_network(states, actions)
        q_value = q_values[-1]

        alpha = self.log_alpha.exp().detach()
        policy_loss = torch.mean(alpha * log_probs - q_value)

        mean_log_prob = torch.mean(log_probs).detach()
        temperature_loss = -torch.mean(
            self.log_alpha * (self.target_entropy + mean_log_prob)
        )

        return policy_loss, temperature_loss

    class QNetwork(nn.Module):
        __slots__ = ["networks"]

        def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dims: list[int],
            *,
            num_critics: int,
        ) -> None:
            super().__init__()
            self.networks = nn.ModuleList()

            for _ in range(num_critics):
                input_dim = state_dim + action_dim
                layers = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]
                nn.init.orthogonal_(layers[0].weight, gain=math.sqrt(2))

                for in_dim, out_dim in itertools.pairwise(hidden_dims):
                    linear_layer = nn.Linear(in_dim, out_dim)
                    nn.init.orthogonal_(linear_layer.weight, gain=math.sqrt(2))
                    layers.extend([linear_layer, nn.ReLU()])

                layers.append(nn.Linear(hidden_dims[-1], 1))
                self.networks.append(nn.Sequential(*layers))

        def forward(
            self, state: torch.Tensor, action: torch.Tensor
        ) -> list[torch.Tensor]:
            state_action = torch.cat([state, action], dim=1)
            return [
                network(state_action).squeeze(-1) for network in self.networks
            ]

    class VNetwork(nn.Module):
        __slots__ = ["networks"]

        def __init__(
            self, state_dim: int, hidden_dims: list[int], *, num_critics: int
        ) -> None:
            super().__init__()
            self.networks = nn.ModuleList()

            for _ in range(num_critics):
                layers = [nn.Linear(state_dim, hidden_dims[0]), nn.ReLU()]
                nn.init.orthogonal_(layers[0].weight, gain=math.sqrt(2))

                for in_dim, out_dim in itertools.pairwise(hidden_dims):
                    linear_layer = nn.Linear(in_dim, out_dim)
                    nn.init.orthogonal_(linear_layer.weight, gain=math.sqrt(2))
                    layers.extend([linear_layer, nn.ReLU()])

                layers.append(nn.Linear(hidden_dims[-1], 1))
                self.networks.append(nn.Sequential(*layers))

        def forward(self, state: torch.Tensor) -> list[torch.Tensor]:
            return [network(state).squeeze(-1) for network in self.networks]

    class PolicyNetwork(nn.Module):
        __slots__ = ["log_std_max", "log_std_min", "network"]

        def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dims: list[int],
            *,
            log_std_min: float,
            log_std_max: float,
        ) -> None:
            super().__init__()
            self.action_dim = action_dim
            self.log_std_min = log_std_min
            self.log_std_max = log_std_max

            layers = [nn.Linear(state_dim, hidden_dims[0]), nn.ReLU()]
            nn.init.orthogonal_(layers[0].weight, gain=math.sqrt(2))

            for in_dim, out_dim in itertools.pairwise(hidden_dims):
                linear_layer = nn.Linear(in_dim, out_dim)
                nn.init.orthogonal_(linear_layer.weight, gain=math.sqrt(2))
                layers.extend([linear_layer, nn.ReLU()])

            layers.append(nn.Linear(hidden_dims[-1], 2 * action_dim))
            self.network = nn.Sequential(*layers)

        def forward(
            self, state: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            output = self.network(state)
            mean, log_std = torch.split(output, self.action_dim, dim=-1)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            return mean, log_std
