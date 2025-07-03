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
import torch.distributions as dist
from torch import nn

from .replay import ReplayBuffer
from .rl_algo import RLAlgo
from .utils import soft_update_target


class AFU(RLAlgo):
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
        "replay_buffer",
        "rho",
        "state_dim",
        "target_entropy",
        "tau",
        "temperature_optimizer",
        "v_network",
        "v_optimizer",
        "v_target_network",
    ]

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

        self.replay_buffer = ReplayBuffer(replay_size, state_dim, action_dim)
        self.batch_size = batch_size

        self.target_entropy = -action_dim
        self.tau = tau
        self.rho = rho
        self.gamma = gamma

    def select_action(self, state: np.ndarray, evaluation: bool) -> np.ndarray:
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
                raw_action = dist.Normal(mean, log_std.exp()).rsample()
                action = torch.tanh(raw_action)
        return action.squeeze(0).numpy()

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

        policy_loss, temperature_loss = self._compute_policy_loss(states)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.learn_temperature:
            self.temperature_optimizer.zero_grad()
            temperature_loss.backward()
            self.temperature_optimizer.step()

        soft_update_target(self.v_target_network, self.v_network, self.tau)

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
        print("=== CRITIC LOSS COMPUTATION - TENSOR SHAPES ===")
        print(f"Input states shape: {states.shape}")
        print(f"Input actions shape: {actions.shape}")
        print(f"Input rewards shape: {rewards.shape}")
        print(f"Input next_states shape: {next_states.shape}")
        print(f"Input dones shape: {dones.shape}")

        print("\n--- TARGET COMPUTATION ---")
        with torch.no_grad():
            next_v_values = self.v_target_network(next_states)
            print(f"next_v_values (list length): {len(next_v_values)}")
            for i, v_val in enumerate(next_v_values):
                print(f"  next_v_values[{i}] shape: {v_val.shape}")

            v_targets_stacked = torch.stack(next_v_values)
            print(f"v_targets_stacked shape: {v_targets_stacked.shape}")

            v_targets = torch.min(v_targets_stacked, dim=0)[0]
            print(f"v_targets shape after min: {v_targets.shape}")

            q_targets = rewards + self.gamma * (1.0 - dones.float()) * v_targets
            print(f"q_targets shape: {q_targets.shape}")
            print(f"rewards shape: {rewards.shape}")
            print(f"dones shape: {dones.shape}")
            print(f"v_targets shape: {v_targets.shape}")

        print("\n--- CURRENT VALUE ESTIMATES ---")
        v_values_list = self.v_network(states)
        print(f"v_values_list (list length): {len(v_values_list)}")
        for i, v_val in enumerate(v_values_list):
            print(f"  v_values_list[{i}] shape: {v_val.shape}")

        v_values = torch.stack(v_values_list)
        print(f"v_values stacked shape: {v_values.shape}")

        q_values_list = self.q_network(states, actions)
        print(f"q_values_list (list length): {len(q_values_list)}")
        for i, q_val in enumerate(q_values_list):
            print(f"  q_values_list[{i}] shape: {q_val.shape}")

        q_final = torch.stack(q_values_list[-1:])
        print(f"q_final shape: {q_final.shape}")

        print("\n--- Q-LOSS COMPUTATION ---")
        abs_td = torch.abs(q_targets - q_final)
        print(f"abs_td shape: {abs_td.shape}")
        print(
            f"Broadcasting: q_targets {q_targets.shape} - q_final {q_final.shape}"
        )

        q_loss = torch.mean(abs_td**2)
        print(f"q_loss (scalar): {q_loss.shape}")

        print("\n--- ADVANTAGE COMPUTATION ---")
        optim_advantages = -torch.stack(q_values_list[:-1])
        print(f"optim_advantages shape: {optim_advantages.shape}")
        print(f"Number of advantage critics: {len(q_values_list[:-1])}")

        print("\n--- CONDITIONAL OPTIMIZATION LOGIC ---")
        condition_tensor = v_values + optim_advantages
        print(f"v_values shape: {v_values.shape}")
        print(f"optim_advantages shape: {optim_advantages.shape}")
        print(
            f"condition_tensor (v_values + optim_advantages) shape: {condition_tensor.shape}"
        )
        print(f"q_targets shape for comparison: {q_targets.shape}")

        mix_case = (condition_tensor < q_targets).detach().float()
        print(f"mix_case shape: {mix_case.shape}")
        print(
            f"Broadcasting check: condition_tensor {condition_tensor.shape} < q_targets {q_targets.shape}"
        )

        print("\n--- UPSILON COMPUTATION ---")
        term1 = (1 - self.rho * mix_case) * v_values
        term2 = self.rho * mix_case * v_values.detach()
        print(f"term1 shape: {term1.shape}")
        print(f"term2 shape: {term2.shape}")

        upsilon_values = term1 + term2
        print(f"upsilon_values shape: {upsilon_values.shape}")

        print("\n--- FINAL LOSS COMPUTATION ---")
        target_diff = upsilon_values - q_targets
        print(f"target_diff shape: {target_diff.shape}")
        print(
            f"Broadcasting: upsilon_values {upsilon_values.shape} - q_targets {q_targets.shape}"
        )

        up_case = (v_values >= q_targets).detach().float()
        print(f"up_case shape: {up_case.shape}")

        z_term1 = optim_advantages**2
        z_term2 = up_case * 2 * optim_advantages * target_diff
        z_term3 = target_diff**2

        print(f"z_term1 (advantages^2) shape: {z_term1.shape}")
        print(f"z_term2 (interaction term) shape: {z_term2.shape}")
        print(f"z_term3 (target_diff^2) shape: {z_term3.shape}")

        z_values = z_term1 + z_term2 + z_term3
        print(f"z_values shape: {z_values.shape}")

        va_loss = torch.mean(z_values)
        print(f"va_loss (scalar): {va_loss.shape}")

        total_loss = va_loss + q_loss
        print(f"total_loss (scalar): {total_loss.shape}")
        print("=== END CRITIC LOSS COMPUTATION ===\n")

        return total_loss

    # def _compute_critic_loss(
    #     self,
    #     states: torch.Tensor,
    #     actions: torch.Tensor,
    #     rewards: torch.Tensor,
    #     next_states: torch.Tensor,
    #     dones: torch.Tensor,
    # ) -> torch.Tensor:
    #     with torch.no_grad():
    #         next_v_values = torch.stack(self.v_target_network(next_states))
    #         v_targets = torch.min(next_v_values, dim=0)[0]
    #         q_targets = rewards + self.gamma * (1.0 - dones.float()) * v_targets
    #
    #     v_values = torch.stack(self.v_network(states))
    #
    #     q_values = torch.stack(self.q_network(states, actions))
    #     q_final = q_values[-1:]
    #
    #     abs_td = torch.abs(q_targets - q_final)
    #     q_loss = torch.mean(abs_td**2)
    #
    #     optim_advantages = -q_values[:-1]
    #
    #     mix_case = (v_values + optim_advantages < q_targets).detach().float()
    #     upsilon_values = (
    #         1 - self.rho * mix_case
    #     ) * v_values + self.rho * mix_case * v_values.detach()
    #
    #     target_diff = upsilon_values - q_targets
    #     up_case = (v_values >= q_targets).detach().float()
    #     z_values = (
    #         optim_advantages**2
    #         + up_case * 2 * optim_advantages * target_diff
    #         + target_diff**2
    #     )
    #
    #     va_loss = torch.mean(z_values)
    #
    #     return va_loss + q_loss

    def _compute_policy_loss(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions, log_probs = self._compute_action_and_log_prob(states)

        q_values = self.q_network(states, actions)
        q_value = q_values[-1]

        alpha = self.log_alpha.exp().detach()
        policy_loss = torch.mean(alpha * log_probs - q_value)

        temperature_loss = torch.mean(
            -self.log_alpha * (log_probs.detach() + self.target_entropy)
        )

        return policy_loss, temperature_loss

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
        tanh_correction = torch.log(torch.relu(1 - action.pow(2)) + 1e-6)
        log_prob = (log_prob_gaussian - tanh_correction).sum(dim=-1)

        return action, log_prob

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

                final_layer = nn.Linear(hidden_dims[-1], 1)
                nn.init.orthogonal_(final_layer.weight)
                layers.append(final_layer)
                self.networks.append(nn.Sequential(*layers))

        def forward(
            self, state: torch.Tensor, action: torch.Tensor
        ) -> list[torch.Tensor]:
            state_action = torch.cat([state, action], dim=1)
            return [network(state_action) for network in self.networks]

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

                final_layer = nn.Linear(hidden_dims[-1], 1)
                nn.init.orthogonal_(final_layer.weight)
                layers.append(final_layer)
                self.networks.append(nn.Sequential(*layers))

        def forward(self, state: torch.Tensor) -> list[torch.Tensor]:
            return [network(state) for network in self.networks]

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

            final_layer = nn.Linear(hidden_dims[-1], 2 * action_dim)
            nn.init.orthogonal_(final_layer.weight)
            layers.append(final_layer)
            self.network = nn.Sequential(*layers)

        def forward(
            self, state: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            output = self.network(state)
            mean, log_std = torch.split(output, self.action_dim, dim=-1)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            return mean, log_std
