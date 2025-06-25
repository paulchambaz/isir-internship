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
from scipy.spatial import ConvexHull
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


def measure_max_q_values(algorithm, q_net, v_net, policy_net, grid_size):
    positions = np.linspace(POSITION_MIN, POSITION_MAX, grid_size)
    velocities = np.linspace(VELOCITY_MIN, VELOCITY_MAX, grid_size)

    max_q_values = np.zeros((grid_size, grid_size))

    for i, pos in enumerate(positions):
        for j, vel in enumerate(velocities):
            state = torch.tensor([[pos, vel]], dtype=torch.float32)

            n_actions = 50
            actions_batch = torch.linspace(
                ACTION_MIN, ACTION_MAX, n_actions
            ).unsqueeze(1)
            states_batch = state.expand(n_actions, -1)

            if algorithm == "sac":
                q1_values = q_net[0](states_batch, actions_batch)
                q2_values = q_net[1](states_batch, actions_batch)
                q_values = torch.min(q1_values, q2_values)
            else:  # afu
                q_values_list = q_net(states_batch, actions_batch)
                q_values = q_values_list[-1]

            max_q_value = float(torch.max(q_values))
            max_q_values[j, i] = max_q_value

    return max_q_values


def measure_v_plus_max_a(q_net, v_net, grid_size):
    positions = np.linspace(POSITION_MIN, POSITION_MAX, grid_size)
    velocities = np.linspace(VELOCITY_MIN, VELOCITY_MAX, grid_size)

    v_plus_max_a = np.zeros((grid_size, grid_size))

    for i, pos in enumerate(positions):
        for j, vel in enumerate(velocities):
            state = torch.tensor([[pos, vel]], dtype=torch.float32)

            with torch.no_grad():
                v_values_list = v_net(state)
                v_value = float(torch.min(torch.stack(v_values_list), dim=0)[0])

            n_actions = 50
            actions_batch = torch.linspace(
                ACTION_MIN, ACTION_MAX, n_actions
            ).unsqueeze(1)
            states_batch = state.expand(n_actions, -1)

            with torch.no_grad():
                q_values_list = q_net(states_batch, actions_batch)
                q_values = q_values_list[-1]  # Use the final Q network
                max_q = float(torch.max(q_values))
                max_a = max_q - v_value

            v_plus_max_a[j, i] = v_value + max_a

    return v_plus_max_a


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


def get_replay_buffer_convex_hull(replay_positions, replay_velocities):
    points = np.column_stack((replay_positions, replay_velocities))

    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    hull_points = np.vstack([hull_points, hull_points[0]])

    return hull_points[:, 0], hull_points[:, 1]


def extract_replay_buffer_states(state_dict):
    replay_data = state_dict["replay_buffer"]
    states = np.array([transition[0] for transition in replay_data])
    positions = states[:, 0]
    velocities = states[:, 1]
    return positions, velocities


def display_visualization(
    i,
    algorithm,
    v_values,
    actions,
    max_q_values,
    v_plus_max_a_values,
    avg_trajectories,
    replay_positions,
    replay_velocities,
    hull_pos,
    hull_vel,
):
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    axes = axes.flatten()
    extent = [POSITION_MIN, POSITION_MAX, VELOCITY_MIN, VELOCITY_MAX]

    all_values = np.concatenate(
        [
            v_values.flatten(),
            max_q_values.flatten(),
            v_plus_max_a_values.flatten(),
        ]
    )
    vmin_shared = np.min(all_values)
    vmax_shared = np.max(all_values)

    im1 = axes[0].imshow(
        v_values,
        extent=extent,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=vmin_shared,
        vmax=vmax_shared,
    )
    axes[0].set_title(rf"$V(s)$ - {algorithm.upper()}")
    axes[0].set_xlabel("Position")
    axes[0].set_ylabel("Velocity")
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(
        v_values - max_q_values,
        extent=extent,
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    axes[1].set_title(rf"$V(s) - \max_a Q(s,a)$ - {algorithm.upper()}")
    axes[1].set_xlabel("Position")
    axes[1].set_ylabel("Velocity")
    plt.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(
        v_values - v_plus_max_a_values,
        extent=extent,
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    axes[2].set_title(rf"$\max_a A(s,a)$ - {algorithm.upper()}")
    axes[2].set_xlabel("Position")
    axes[2].set_ylabel("Velocity")
    plt.colorbar(im3, ax=axes[2])

    im4 = axes[3].imshow(
        actions,
        extent=extent,
        aspect="auto",
        origin="lower",
        cmap="inferno",
        vmin=-1,
        vmax=1,
    )
    axes[3].set_title(rf"$\pi(s)$ - {algorithm.upper()}")
    axes[3].set_xlabel("Position")
    axes[3].set_ylabel("Velocity")
    plt.colorbar(im4, ax=axes[3])

    axes[4].set_xlim(POSITION_MIN, POSITION_MAX)
    axes[4].set_ylim(VELOCITY_MIN, VELOCITY_MAX)
    axes[4].set_title(rf"Replay Buffer - {algorithm.upper()}")
    axes[4].set_xlabel("Position")
    axes[4].set_ylabel("Velocity")

    colors = np.arange(len(replay_positions))
    scatter = axes[4].scatter(
        replay_positions,
        replay_velocities,
        c=colors,
        cmap="magma",
        s=1,
        alpha=0.7,
    )
    plt.colorbar(scatter, ax=axes[4], label="Recency")

    axes[5].axis("off")

    for j in range(5):
        axes[j].axvline(
            x=GOAL_POSITION, color="black", linestyle="--", linewidth=1
        )

    for j in range(3):
        axes[j].plot(hull_pos, hull_vel, "k--", linewidth=1)

    if len(avg_trajectories) > 1:
        avg_trajectory = np.array(avg_trajectories)
        positions_avg = avg_trajectory[::2, 0]
        velocities_avg = avg_trajectory[::2, 1]
        for j in range(4):
            axes[j].scatter(
                positions_avg,
                velocities_avg,
                c="white",
                s=10,
                edgecolors="black",
                linewidths=0.6,
            )

    path = f"paper/figures/{algorithm}_mountaincar_critic_policy"
    Path(path).mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        f"{path}/{i:05d}.png",
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
    )
    plt.close()


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
            q_net = algos.AFU.QNetwork(
                state_dim, action_dim, hidden_dims, num_critics=3
            )
            q_net.load_state_dict(state_dict["q_network"])
            q_nets = q_net
            v_net = algos.AFU.VNetwork(state_dim, hidden_dims, num_critics=2)
            v_net.load_state_dict(state_dict["v_network"])
            policy_net = algos.AFU.PolicyNetwork(
                state_dim,
                action_dim,
                hidden_dims,
                log_std_min=-10.0,
                log_std_max=2.0,
            )
            policy_net.load_state_dict(state_dict["policy_network"])

    grid_size = 50
    avg_trajectory = measure_trajectories(policy_net, 3)

    match algorithm:
        case "sac":
            v_values = measure_v_values("sac", q1_net, q2_net, v_net, grid_size)
        case "afu":
            v_values = measure_v_values("afu", None, None, v_net, grid_size)
            max_q_values = measure_max_q_values(
                "afu", q_nets, v_net, policy_net, grid_size
            )
            v_plus_max_a_values = measure_v_plus_max_a(q_nets, v_net, grid_size)

    actions = measure_actions(policy_net, grid_size)

    replay_positions, replay_velocities = extract_replay_buffer_states(
        state_dict
    )
    hull_pos, hull_vel = get_replay_buffer_convex_hull(
        replay_positions, replay_velocities
    )

    plt.rcParams["font.size"] = 20
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"

    display_visualization(
        i,
        algorithm,
        v_values,
        actions,
        max_q_values,
        v_plus_max_a_values,
        avg_trajectory,
        replay_positions,
        replay_velocities,
        hull_pos,
        hull_vel,
    )


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
