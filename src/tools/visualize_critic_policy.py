# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship.
#
# isir-internship is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import argparse
import pickle
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import algos

POSITION_MIN = -1.2
POSITION_MAX = 0.6
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07

ACTION_MIN = -1.0
ACTION_MAX = 1.0

GOAL_POSITION = 0.45


def measure_trajectories(policy_net, n_episodes):
    env = gym.make("MountainCarContinuous-v0")

    trajectories = []
    n_episodes = 3
    for _ in range(n_episodes):
        trajectory = []
        state, _ = env.reset()

        while True:
            trajectory.append([state[0], state[1]])

            state_tensor = torch.tensor(
                [[state[0], state[1]]], dtype=torch.float32
            )
            mean, _ = policy_net(state_tensor)
            action = float(torch.tanh(mean).item())

            next_state, _, terminated, truncated, _ = env.step([action])
            done = terminated or truncated

            if done:
                break

            state = next_state

        trajectories.append(np.array(trajectory))

    env.close()

    max_len = max(len(traj) for traj in trajectories)
    avg_trajectory = []

    for step in range(max_len):
        positions = []
        velocities = []

        for traj in trajectories:
            if step < len(traj) and traj[step][0] < GOAL_POSITION:
                positions.append(traj[step][0])
                velocities.append(traj[step][1])

        if positions:
            avg_trajectory.append([np.mean(positions), np.mean(velocities)])
        else:
            break

    return avg_trajectory


def measure_v_values(algorithm, q1_net, q2_net, v_net, grid_size):
    positions = np.linspace(POSITION_MIN, POSITION_MAX, grid_size)
    velocities = np.linspace(VELOCITY_MIN, VELOCITY_MAX, grid_size)

    v_values = np.zeros((grid_size, grid_size))

    for i, pos in enumerate(positions):
        for j, vel in enumerate(velocities):
            state = torch.tensor([[pos, vel]], dtype=torch.float32)

            match algorithm:
                case "sac":
                    n_actions = 50

                    actions_batch = torch.linspace(
                        ACTION_MIN, ACTION_MAX, n_actions
                    ).unsqueeze(1)
                    states_batch = state.expand(n_actions, -1)

                    q1_values = q1_net(states_batch, actions_batch)
                    q2_values = q2_net(states_batch, actions_batch)
                    q_values = torch.min(q1_values, q2_values)

                    v_value = float(torch.max(q_values))
                case "afu":
                    with torch.no_grad():
                        v_values_list = v_net(state)
                        v_stacked = torch.stack(v_values_list)
                        v_value = float(torch.min(v_stacked, dim=0)[0])

            v_values[j, i] = v_value

    return v_values


def measure_actions(policy_net, grid_size):
    positions = np.linspace(POSITION_MIN, POSITION_MAX, grid_size)
    velocities = np.linspace(VELOCITY_MIN, VELOCITY_MAX, grid_size)

    actions = np.zeros((grid_size, grid_size))

    for i, pos in enumerate(positions):
        for j, vel in enumerate(velocities):
            state = torch.tensor([[pos, vel]], dtype=torch.float32)

            mean, _ = policy_net(state)
            action = float(torch.tanh(mean).squeeze())
            actions[j, i] = action

    return actions


def display_visualization(i, algorithm, v_values, actions, avg_trajectories):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    extent = [POSITION_MIN, POSITION_MAX, VELOCITY_MIN, VELOCITY_MAX]
    im1 = ax1.imshow(
        v_values,
        extent=extent,
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    ax1.set_title(rf"Value function $V(s)$ - {algorithm.upper()}")
    ax1.set_xlabel("Position")
    ax1.set_ylabel("Velocity")
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(
        actions,
        extent=extent,
        aspect="auto",
        origin="lower",
        cmap="hot",
        vmin=-1,
        vmax=1,
    )
    ax2.set_title(rf"Policy actions $\pi(s)$ - {algorithm.upper()}")
    ax2.set_xlabel("Position")
    ax2.set_ylabel("Velocity")
    plt.colorbar(im2, ax=ax2)

    ax1.axvline(x=GOAL_POSITION, color="black", linestyle="--", linewidth=1)
    ax2.axvline(x=GOAL_POSITION, color="#3498db", linestyle="--", linewidth=1)

    if len(avg_trajectories) > 1:
        avg_trajectory = np.array(avg_trajectories)
        positions_avg = avg_trajectory[::2, 0]
        velocities_avg = avg_trajectory[::2, 1]
        ax1.scatter(positions_avg, velocities_avg, c="black", s=4)
        ax2.scatter(positions_avg, velocities_avg, c="#3498db", s=4)

    path = f"paper/figures/{algorithm}_mountaincar_critic_policy"
    Path(path).mkdir(exist_ok=True)
    plt.savefig(
        f"{path}/{i}.png",
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
    )


def get_figure(i, state_dict, algorithm):
    state_dim = 2
    action_dim = 1
    hidden_dims = [256, 256]
    match algorithm:
        case "sac":
            q1_net = algos.SAC.QNetwork(state_dim, hidden_dims, action_dim)
            q2_net = algos.SAC.QNetwork(state_dim, hidden_dims, action_dim)
            v_net = None
            policy_net = algos.SAC.PolicyNetwork(
                state_dim, hidden_dims, action_dim
            )

            q1_net.load_state_dict(state_dict["q1"])
            q2_net.load_state_dict(state_dict["q2"])
            policy_net.load_state_dict(state_dict["policy"])
        case "afu":
            q1_net = None
            q2_net = None
            v_net = algos.AFU.VNetwork(state_dim, hidden_dims, num_critics=2)
            policy_net = algos.AFU.PolicyNetwork(
                state_dim,
                action_dim,
                hidden_dims,
                log_std_min=-10.0,
                log_std_max=2.0,
            )

            v_net.load_state_dict(state_dict["v_network"])
            policy_net.load_state_dict(state_dict["policy_network"])

    grid_size = 50

    avg_trajectory = measure_trajectories(policy_net, 3)

    match algorithm:
        case "sac":
            v_values = measure_v_values("sac", q1_net, q2_net, v_net, grid_size)
        case "afu":
            v_values = measure_v_values("afu", q1_net, q2_net, v_net, grid_size)

    actions = measure_actions(policy_net, grid_size)

    plt.rcParams["font.size"] = 20
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"

    display_visualization(i, algorithm, v_values, actions, avg_trajectory)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize MountainCar value fonctions and policies"
    )
    parser.add_argument(
        "--algorithm",
        choices=["sac", "afu"],
        required=True,
        help="Algorithm type",
    )
    parser.add_argument(
        "--file", type=str, required=True, help="Path to agent state file"
    )
    args = parser.parse_args()

    with open(args.file, "rb") as f:
        state_history = pickle.load(f)  # noqa: S301

    for i, state_dict in tqdm(state_history.items()):
        get_figure(i, state_dict, args.algorithm)


if __name__ == "__main__":
    main()
