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

from .algo import Algo
from .replay import ReplayBuffer
from .utils import soft_update_target


class SAC(Algo):
    """
    Soft Actor-Critic (SAC) algorithm for continuous control tasks.

    SAC is an off-policy actor-critic method based on the maximum entropy
    reinforcement learning framework. It learns a stochastic policy that maximizes
    both expected return and entropy to improve exploration and robustness.

    Args:
        action_dim: Dimensionality of the action space
        state_dim: Dimensionality of the state space
        hidden_dims: List of hidden layer sizes for neural networks
        replay_size: Maximum size of the replay buffer
        batch_size: Number of samples per training batch
        critic_lr: Learning rate for Q-function networks
        policy_lr: Learning rate for policy network
        temperature_lr: Learning rate for temperature parameter
        tau: Target network soft update coefficient (0 < tau <= 1)
        gamma: Discount factor for future rewards (0 < gamma <= 1)
        alpha: If None, learn alpha, if float, use fixed alpha value
        state: Optional pre-trained network state dictionary
    """

    __slots__ = [
        "action_dim",
        "batch_size",
        "gamma",
        "learn_temperature",
        "log_alpha",
        "policy_network",
        "policy_optimizer",
        "q_network",
        "q_optimizer",
        "q_target_network",
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
        state: dict | None = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learn_temperature = alpha is None

        self.q_network = self.QNetwork(state_dim, hidden_dims, action_dim, 2)

        self.q_target_network = self.QNetwork(
            state_dim, hidden_dims, action_dim, 2
        )
        self.q_target_network.load_state_dict(self.q_network.state_dict())

        self.policy_network = self.PolicyNetwork(
            state_dim, hidden_dims, action_dim
        )

        self.log_alpha = (
            nn.Parameter(torch.zeros(1))
            if self.learn_temperature
            else torch.log(torch.tensor(alpha))
        )

        self.q_optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=critic_lr
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
        done: bool,  # noqa: FBT001
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

        q_loss = self._compute_q_loss(
            states, actions, rewards, next_states, dones
        )
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        soft_update_target(self.q_target_network, self.q_network, self.tau)

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
            "q": self.q_network.state_dict(),
            "q_target": self.q_target_network.state_dict(),
            "policy": self.policy_network.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "q_optimizer": self.q_optimizer.state_dict(),
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
        self.q_network.load_state_dict(state["q"])
        self.q_target_network.load_state_dict(state["q_target"])
        self.policy_network.load_state_dict(state["policy"])
        self.log_alpha.data = torch.tensor(state["log_alpha"])
        self.q_optimizer.load_state_dict(state["q_optimizer"])
        self.policy_optimizer.load_state_dict(state["policy_optimizer"])
        self.replay_buffer.load_data(state["replay_buffer"])

        if self.learn_temperature:
            self.temperature_optimizer.load_state_dict(
                state["temperature_optimizer"]
            )

    def _compute_q_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        # pi(s'), log pi (pi(s') | s')
        next_actions, log_probs = self._compute_action_and_log_prob(next_states)

        # min(Q^target_theta_1 (s', a'), Q^target_theta_2 (s', a'))
        q_targets_list = self.q_target_network(next_states, next_actions)
        q_targets = torch.min(torch.stack(q_targets_list), dim=0)[0]

        # r + gamma (1 - d) (min Q(s', a') - log pi (a' | s'))
        alpha = self.log_alpha.exp() + 1e-8
        targets = rewards + self.gamma * (1.0 - dones.float()) * (
            q_targets - alpha * log_probs.detach()
        )

        # EE [ 1/2 * (Q_theta_i (s, a) - y)^2 ]
        q_values_list = self.q_network(states, actions)
        q_losses = [
            torch.mean(0.5 * (q_values - targets.detach()) ** 2)
            for q_values in q_values_list
        ]

        return sum(q_losses)

    def _compute_policy_loss(self, states: torch.Tensor) -> torch.Tensor:
        # pi(s) log pi (pi(s) | s)
        actions, log_probs = self._compute_action_and_log_prob(states)

        # min(Q_theta_1 (s, a), Q_theta_2 (s, a))
        q_values_list = self.q_network(states, actions)
        q_values = torch.min(torch.stack(q_values_list), dim=0)[0]

        # EE [ alpha log pi (a | s) - Q (s, a) ]
        alpha = self.log_alpha.exp() + 1e-8
        return torch.mean(alpha * log_probs - q_values)

    def _compute_temperature_loss(self, states: torch.Tensor) -> torch.Tensor:
        # log pi (pi(s) | s)
        _, log_probs = self._compute_action_and_log_prob(states)

        # EE [ -alpha (log pi (a | s) + H) ]
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
        eps = torch.finfo(action.dtype).eps
        clamped_action = torch.clamp(action, min=-1.0 + eps, max=1.0 - eps)
        tanh_correction = torch.log1p(-(clamped_action**2))
        log_prob = (log_prob_gaussian - tanh_correction).sum(dim=-1)

        return action, log_prob

    # def _soft_update_targets(self) -> None:
    #     with torch.no_grad():
    #         for target_param, param in zip(
    #             self.q_target_network.parameters(),
    #             self.q_network.parameters(),
    #             strict=True,
    #         ):
    #             target_param.data.copy_(
    #                 self.tau * param + (1 - self.tau) * target_param.data
    #             )

    class QNetwork(nn.Module):
        __slots__ = ["networks"]

        def __init__(
            self,
            state_dim: int,
            hidden_dims: list[int],
            action_dim: int,
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
                layers.append(nn.Linear(hidden_dims[-1], 1))

                self.networks.append(nn.Sequential(*layers))

        def forward(
            self, state: torch.Tensor, action: torch.Tensor
        ) -> list[torch.Tensor]:
            state_action = torch.cat([state, action], dim=1)
            return [
                network(state_action).squeeze(-1) for network in self.networks
            ]

    class PolicyNetwork(nn.Module):
        __slots__ = [
            "action_dim",
            "model",
        ]

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
