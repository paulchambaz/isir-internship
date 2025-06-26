# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship.
#
# isir-internship is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import argparse
import gc
import pickle
from itertools import product
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import algos

POSITION_MIN = -1.2
POSITION_MAX = 0.6
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07
GOAL_POSITION = 0.45


def run_monte_carlo_trajectory(
    policy_net: nn.Module,
    start_position: float,
    start_velocity: float,
    max_steps: int,
    gamma: float,
) -> float:
    env = gym.make("MountainCarContinuous-v0")
    _, _ = env.reset()
    env.unwrapped.state = np.array(
        [start_position, start_velocity], dtype=np.float32
    )

    total_return = 0.0
    discount = 1.0

    state = np.array([start_position, start_velocity], dtype=np.float32)

    for _ in range(max_steps):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            mean, log_std = policy_net(state_tensor)
            raw_action = torch.distributions.Normal(
                mean, log_std.exp()
            ).rsample()
            action = torch.tanh(raw_action).squeeze(0).numpy()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_return += discount * reward
        discount *= gamma

        state = next_state

        if done:
            break

    env.close()

    return total_return


def compute_mc_qvalue(
    policy_net: nn.Module,
    position: float,
    velocity: float,
    n_trajectories: int,
    max_steps: int,
    gamma: float,
) -> float:
    return np.mean(
        [
            run_monte_carlo_trajectory(
                policy_net, position, velocity, max_steps, gamma
            )
            for _ in range(n_trajectories)
        ]
    )


def compute_mc_values_grid(
    policy_net: nn.Module,
    grid_size: int,
    n_trajectories: int,
    max_steps: int,
    gamma: float,
) -> dict:
    mc_values = {}
    positions = np.linspace(POSITION_MIN, POSITION_MAX, grid_size)
    velocities = np.linspace(VELOCITY_MIN, VELOCITY_MAX, grid_size)

    for pos, vel in tqdm(product(positions, velocities)):
        mc_values.setdefault(pos, {})[vel] = compute_mc_qvalue(
            policy_net, pos, vel, n_trajectories, max_steps, gamma
        )

    return mc_values


def load_policy_from_state(state_dict: dict) -> nn.Module:
    policy_net = algos.AFU.PolicyNetwork(
        state_dim=2,
        action_dim=1,
        hidden_dims=[256, 256],
        log_std_min=-10.0,
        log_std_max=2.0,
    )
    policy_net.load_state_dict(state_dict["policy_network"])

    return policy_net


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calcul des Q-values Monte Carlo pour comparaison avec les réseaux"
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Dossier contenant l'historique des états",
    )
    args = parser.parse_args()

    mc_returns = {}

    checkpoint_files = sorted(Path(args.file).glob("agent_history_*.pk"))

    with tqdm(
        total=len(checkpoint_files), desc="Processing files"
    ) as file_pbar:
        for checkpoint_file in checkpoint_files:
            with open(checkpoint_file, "rb") as f:
                partial_history = pickle.load(f)  # noqa: S301

            for step_id, state_dict in partial_history.items():
                policy_net = load_policy_from_state(state_dict)

                mc_returns[step_id] = compute_mc_values_grid(
                    policy_net,
                    grid_size=50,
                    n_trajectories=200,
                    max_steps=1000,
                    gamma=0.999,
                )

            del partial_history
            gc.collect()
            file_pbar.update(1)

    Path("outputs").mkdir(exist_ok=True)
    filename = "outputs/mc_returns_complete.pkl"

    with open(filename, "wb") as f:
        pickle.dump(mc_returns, f)


if __name__ == "__main__":
    main()
