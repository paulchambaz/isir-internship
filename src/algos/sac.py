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
        self, state_dim: int, hidden_dims: list[int], action_dim: int
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

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.model(state)
        mean, raw_scale = torch.split(output, self.action_dim, dim=-1)

        log_std = torch.clamp(
            torch.log(torch.softplus(raw_scale) + 1e-5), -20, 2
        )

        return mean, log_std


class SAC:
    """
    Soft Actor-Critic (SAC) algorithm for continuous control tasks.

    SAC is an off-policy actor-critic method based on the maximum entropy
    reinforcement learning framework. It learns a stochastic policy that
    maximizes both expected return and entropy to improve exploration and
    robustness.

    Args:
        action_dim: Dimensionality of the action space
        state_dim: Dimensionality of the state space
        hidden_dims: List of hidden layer sizes for neural networks
        replay_size: Maximum size of the replay buffer
        batch_size: Number of samples per training batch
        q_lr: Learning rate for Q-function networks
        policy_lr: Learning rate for policy network
        alpha_lr: Learning rate for temperature parameter
        tau: Target network soft update coefficient (0 < tau <= 1)
        gamma: Discount factor for future rewards (0 < gamma <= 1)
        weights: Optional pre-trained network weights dictionary
    """

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
        state: dict | None = None,
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
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.replay_buffer = ReplayBuffer(replay_size)
        self.batch_size = batch_size

        self.target_entropy = -action_dim
        self.tau = tau
        self.gamma = gamma

        if state is not None:
            self.load_from_state(state)

    def select_action(
        self, state: np.ndarray, *, evaluation: bool = False
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
            mean if evaluation else dist.Normal(mean, log_std.exp()).rsample()
        )

        return torch.tanh(action).squeeze(0).detach().numpy()

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

        q1_loss = self._compute_q_loss(
            states, actions, rewards, next_states, dones, network_idx=1
        )
        self.q_optimizer1.zero_grad()
        q1_loss.backward()
        self.q_optimizer1.step()

        q2_loss = self._compute_q_loss(
            states, actions, rewards, next_states, dones, network_idx=2
        )
        self.q_optimizer2.zero_grad()
        q2_loss.backward()
        self.q_optimizer2.step()

        policy_loss = self._compute_policy_loss(states)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        alpha_loss = self._compute_alpha_loss(states)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self._soft_update_targets()

    def get_state(self) -> dict:
        """
        Returns dictionary containing all agent state for saving or transferring.
        """
        return {
            "q1": self.q_network1.state_dict(),
            "q2": self.q_network2.state_dict(),
            "q1_target": self.q_target_network1.state_dict(),
            "q2_target": self.q_target_network2.state_dict(),
            "policy": self.policy_network.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "q_optimizer1": self.q_optimizer1.state_dict(),
            "q_optimizer2": self.q_optimizer2.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "replay_buffer": self.replay_buffer.get_data(),
        }

    def load_from_state(self, state: dict) -> None:
        """
        Restores complete agent state from state dictionary.

        Args:
            state: Dictionary of agent state from get_state
        """
        self.q_network1.load_state_dict(state["q1"])
        self.q_network2.load_state_dict(state["q2"])
        self.q_target_network1.load_state_dict(state["q1_target"])
        self.q_target_network2.load_state_dict(state["q2_target"])
        self.policy_network.load_state_dict(state["policy"])
        self.log_alpha.data = torch.tensor(state["log_alpha"])
        self.q_optimizer1.load_state_dict(state["q_optimizer1"])
        self.q_optimizer2.load_state_dict(state["q_optimizer2"])
        self.policy_optimizer.load_state_dict(state["policy_optimizer"])
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        self.replay_buffer.load_data(state["replay_buffer"])

    def _compute_q_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        network_idx: int,
    ) -> torch.Tensor:
        # pi(s'), log pi (pi(s') | s')
        next_actions, log_probs = self._compute_action_and_log_prob(next_states)

        # min(Q^target_theta_1 (s', a'), Q^target_theta_2 (s', a'))
        q1_targets = self.q_target_network1(next_states, next_actions)
        q2_targets = self.q_target_network2(next_states, next_actions)
        q_targets = torch.min(q1_targets, q2_targets)

        # r + gamma (1 - d) (min Q(s', a') - log pi (a' | s'))
        alpha = self.log_alpha.exp()
        targets = rewards + self.gamma * (1.0 - dones.float()) * (
            q_targets - alpha * log_probs
        )

        # EE [ 1/2 * (Q_theta_i (s, a) - y) ]
        q_network = self.q_network1 if network_idx == 1 else self.q_network2
        q_values = q_network(states, actions)
        return torch.mean(0.5 * (q_values - targets.detach()) ** 2)

    def _compute_policy_loss(self, states: torch.Tensor) -> torch.Tensor:
        # pi(s) log pi (pi(s) | s)
        actions, log_probs = self._compute_action_and_log_prob(states)

        # min(Q_theta_1 (s, a), Q_theta_2 (s, a))
        q1_values = self.q_network1(states, actions)
        q2_values = self.q_network2(states, actions)
        q_values = torch.min(q1_values, q2_values)

        # EE [ alpha log pi (a | s) - Q (s, a) ]
        alpha = self.log_alpha.exp()
        return torch.mean(alpha * log_probs - q_values)

    def _compute_alpha_loss(self, states: torch.Tensor) -> torch.Tensor:
        # log pi (pi(s) | s)
        _, log_probs = self._compute_action_and_log_prob(states)

        # EE [ alpha (log pi (a | s) - H) ]
        alpha = self.log_alpha.exp()
        return torch.mean(alpha * (log_probs - self.target_entropy))

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
        tanh_correction = torch.log(1 - action.pow(2) + 1e-6)
        log_prob = (log_prob_gaussian - tanh_correction).sum(dim=-1)

        return action, log_prob

    def _soft_update_targets(self) -> None:
        network_pairs = [
            (self.q_target_network1, self.q_network1),
            (self.q_target_network2, self.q_network2),
        ]

        for target_net, current_net in network_pairs:
            for target_param, param in zip(
                target_net.parameters(), current_net.parameters(), strict=True
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )
