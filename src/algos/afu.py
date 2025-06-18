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


class ANetwork(nn.Module):
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


class VNetwork(nn.Module):
    __slots__ = ["model"]

    def __init__(self, state_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        layers = [nn.Linear(state_dim, hidden_dims[0]), nn.ReLU()]
        for in_dim, out_dim in itertools.pairwise(hidden_dims):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.model = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.model(state).squeeze(-1)


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
        mean, log_std = torch.split(output, self.action_dim, dim=-1)

        log_std = torch.clamp(log_std, -20, 2)

        return mean, log_std


class AFU:
    __slots__ = [
        "a_network1",
        "a_network2",
        "a_optimizer1",
        "a_optimizer2",
        "action_dim",
        "alpha_optimizer",
        "batch_size",
        "gamma",
        "learn_temperature",
        "log_alpha",
        "policy_network",
        "policy_optimizer",
        "q_network",
        "q_optimizer",
        "replay_buffer",
        "rho",
        "state_dim",
        "target_entropy",
        "tau",
        "v_network1",
        "v_network2",
        "v_optimizer1",
        "v_optimizer2",
        "v_target_network1",
        "v_target_network2",
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
        alpha_lr: float,
        tau: float,
        rho: float,
        gamma: float,
        alpha: float | None,
        state: dict | None = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learn_temperature = alpha is None

        self.q_network = QNetwork(state_dim, hidden_dims, action_dim)

        self.v_network1 = VNetwork(state_dim, hidden_dims)
        self.v_network2 = VNetwork(state_dim, hidden_dims)

        self.v_target_network1 = VNetwork(state_dim, hidden_dims)
        self.v_target_network2 = VNetwork(state_dim, hidden_dims)
        self.v_target_network1.load_state_dict(self.v_network1.state_dict())
        self.v_target_network2.load_state_dict(self.v_network2.state_dict())

        self.a_network1 = ANetwork(state_dim, hidden_dims, action_dim)
        self.a_network2 = ANetwork(state_dim, hidden_dims, action_dim)

        self.policy_network = PolicyNetwork(state_dim, hidden_dims, action_dim)

        self.log_alpha = (
            nn.Parameter(torch.zeros(1))
            if self.learn_temperature
            else torch.log(torch.tensor(alpha))
        )

        self.q_optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=critic_lr
        )

        self.v_optimizer1 = torch.optim.Adam(
            self.v_network1.parameters(), lr=critic_lr
        )
        self.v_optimizer2 = torch.optim.Adam(
            self.v_network2.parameters(), lr=critic_lr
        )

        self.a_optimizer1 = torch.optim.Adam(
            self.a_network1.parameters(), lr=critic_lr
        )
        self.a_optimizer2 = torch.optim.Adam(
            self.a_network2.parameters(), lr=critic_lr
        )

        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=policy_lr
        )

        self.alpha_optimizer = (
            torch.optim.Adam([self.log_alpha], lr=alpha_lr)
            if self.learn_temperature
            else None
        )

        self.replay_buffer = ReplayBuffer(replay_size)
        self.batch_size = batch_size

        self.target_entropy = -action_dim
        self.tau = tau
        self.rho = rho
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

    def update(self) -> None:
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

        va1_loss, va2_loss = self._compute_va_loss(
            states, actions, rewards, next_states, dones
        )
        self.v_optimizer1.zero_grad()
        va1_loss.backward()
        self.v_optimizer1.step()
        self.v_optimizer2.zero_grad()
        va2_loss.backward()
        self.v_optimizer2.step()

        va1_loss, va2_loss = self._compute_va_loss(
            states, actions, rewards, next_states, dones
        )
        self.a_optimizer1.zero_grad()
        va1_loss.backward()
        self.a_optimizer1.step()
        self.a_optimizer2.zero_grad()
        va2_loss.backward()
        self.a_optimizer2.step()

        self._soft_update_targets()

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
        # TODO: export everything needed

    def load_from_state(self, state: dict) -> None:
        """
        Restores complete agent state from state dictionary.

        Args:
            state: Dictionary of agent state from get_state
        """
        # TODO: import everything needed

    def _compute_q_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        # min(V^target_phi_1 (s'), V^target_phi_2 (s'))
        v1_targets = self.v_target_network1(next_states)
        v2_targets = self.v_target_network2(next_states)
        v_targets = torch.min(v1_targets, v2_targets)

        # r + gamma (1 - d) (min V(s'))
        targets = rewards + self.gamma * (1 - dones.float()) * v_targets

        # EE [ 1/2 * (Q_psi (s, a) - y)^2 ]
        q_values = self.q_network(states, actions)
        return torch.mean(0.5 * (q_values - targets.detach()) ** 2)

    def _compute_va_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # min(V^target_phi_1 (s'), V^target_phi_2 (s'))
        v1_targets = self.v_target_network1(next_states)
        v2_targets = self.v_target_network2(next_states)
        v_targets = torch.min(v1_targets, v2_targets)

        # r + gamma (1 - d) (min V(s'))
        targets = rewards + self.gamma * (1 - dones.float()) * v_targets

        # Q (s, a)
        q_values = self.q_network(states, actions)

        # V_phi_1 (s), V_phi_2 (s)
        v1_values = self.v_network1(states)
        v2_values = self.v_network2(states)

        # V^nograd_phi_1 (s), V^nograd_phi_2 (s)
        v1_values_nograd = v1_values.detach()
        v2_values_nograd = v2_values.detach()

        # A_xi_1 (s, a), A_xi_2 (s, a)
        a1_values = self.a_network1(states, actions)
        a2_values = self.a_network2(states, actions)

        # rho * I_1, rho * I_2
        rho1 = self.rho * (v1_values + a1_values < q_values).float()
        rho2 = self.rho * (v2_values + a2_values < q_values).float()

        # upsilon_1, upsilon_2
        upsilon1_values = (1 - rho1) * v1_values + rho1 * v1_values_nograd
        upsilon2_values = (1 - rho2) * v2_values + rho2 * v2_values_nograd

        x1 = upsilon1_values - targets
        x2 = upsilon2_values - targets

        y1 = a1_values
        y2 = a2_values

        i1 = (x1 >= 0).float()
        i2 = (x2 >= 0).float()

        return (
            torch.mean((x1 + y1) ** 2 * (1 - i1) + (x1**2 + y1**2) * i1),
            torch.mean((x2 + y2) ** 2 * (1 - i2) + (x2**2 + y2**2) * i2),
        )

    def _compute_policy_loss(self, states: torch.Tensor) -> torch.Tensor:
        # pi(s) log pi (pi(s) | s)
        actions, log_probs = self._compute_action_and_log_prob(states)

        # Q(s, a)
        q_values = self.q_network(states, actions)

        # EE [ alpha log pi (a | s) - Q(s, a) ]
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

    def _soft_update_targets(self) -> None:
        network_pairs = [
            (self.v_target_network1, self.v_network1),
            (self.v_target_network2, self.v_network2),
        ]

        for target_net, current_net in network_pairs:
            for target_param, param in zip(
                target_net.parameters(), current_net.parameters(), strict=True
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )
