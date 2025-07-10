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
from pathlib import Path

import flax.linen as nn
import gymnasium as gym
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
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


def measure_trajectories(
    policy_params: dict, policy_network: nn.Module, n_episodes: int
) -> list[list[float]]:
    env = gym.make("MountainCarContinuous-v0")

    trajectories = []
    for _ in range(n_episodes):
        trajectory = []
        state, _ = env.reset()

        while True:
            trajectory.append([state[0], state[1]])

            state_jax = jnp.array([[state[0], state[1]]], dtype=jnp.float32)
            mean, _ = policy_network.apply(policy_params, state_jax)
            action = float(jnp.tanh(mean).squeeze())

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


def measure_all_values_batched(
    q_params: dict,
    v_params: dict,
    policy_params: dict,
    q_network: nn.Module,
    v_network: nn.Module,
    policy_network: nn.Module,
    grid_size: int,
    n_actions: int,
) -> np.ndarray:
    positions = np.linspace(POSITION_MIN, POSITION_MAX, grid_size)
    velocities = np.linspace(VELOCITY_MIN, VELOCITY_MAX, grid_size)
    actions = np.linspace(ACTION_MIN, ACTION_MAX, n_actions)

    pos_grid, vel_grid = np.meshgrid(positions, velocities, indexing="ij")

    states_batch = jnp.array(
        np.stack([pos_grid.flatten(), vel_grid.flatten()], axis=1),
        dtype=jnp.float32,
    )

    v_values_list = v_network.apply(v_params, states_batch)
    v_values_batch = jnp.min(jnp.stack(v_values_list), axis=0)
    v_values = np.array(v_values_batch).reshape(grid_size, grid_size)

    mean_batch, _ = policy_network.apply(policy_params, states_batch)
    actions_batch = jnp.tanh(mean_batch)
    policy_actions = np.array(actions_batch).reshape(grid_size, grid_size)

    states_expanded = jnp.repeat(states_batch, n_actions, axis=0)
    actions_expanded = jnp.tile(actions, grid_size * grid_size).reshape(-1, 1)

    q_values_list = q_network.apply(q_params, states_expanded, actions_expanded)

    q_values_reshaped = [
        q_vals.reshape(grid_size * grid_size, n_actions)
        for q_vals in q_values_list
    ]

    a_values_batch = -(q_values_reshaped[0] + q_values_reshaped[1]) / 2
    q_values_batch = q_values_reshaped[2]

    max_q_values = np.array(jnp.max(q_values_batch, axis=1)).reshape(
        grid_size, grid_size
    )
    max_a_values = np.array(jnp.max(a_values_batch, axis=1)).reshape(
        grid_size, grid_size
    )

    return v_values, policy_actions, max_q_values, max_a_values


def extract_replay_buffer_states(
    state_dict: dict,
) -> tuple[np.ndarray, np.ndarray]:
    replay_data = state_dict["replay_buffer"]
    states = replay_data["state"]
    positions = states[:, 0]
    velocities = states[:, 1]
    return positions, velocities


def get_replay_buffer_convex_hull(
    replay_positions: np.ndarray, replay_velocities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    points = np.column_stack((replay_positions, replay_velocities))

    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    hull_points = np.vstack([hull_points, hull_points[0]])

    return hull_points[:, 0], hull_points[:, 1]


def display_visualization(
    i: int,
    v_values: np.ndarray,
    actions: np.ndarray,
    max_q_values: np.ndarray,
    max_a_values: np.ndarray,
    avg_trajectories: np.ndarray,
    replay_positions: np.ndarray,
    replay_velocities: np.ndarray,
    hull_pos: np.ndarray,
    hull_vel: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    axes = axes.flatten()
    extent = [POSITION_MIN, POSITION_MAX, VELOCITY_MIN, VELOCITY_MAX]

    v_all_values = np.concatenate(
        [
            v_values.flatten(),
            max_q_values.flatten(),
            (v_values + max_a_values).flatten(),
        ]
    )
    vmin_shared = np.min(v_all_values)
    vmax_shared = np.max(v_all_values)

    q_all_values = np.concatenate(
        [
            (v_values - max_q_values).flatten(),
            (v_values - max_q_values + max_a_values).flatten(),
        ]
    )
    qmin_shared = np.min(q_all_values)
    qmax_shared = np.max(q_all_values)

    qabsolute = max(-qmin_shared, qmax_shared)

    im1 = axes[0].imshow(
        v_values,
        extent=extent,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=vmin_shared,
        vmax=vmax_shared,
    )
    axes[0].set_title(r"$V(s)$")
    axes[0].set_xlabel("Position")
    axes[0].set_ylabel("Velocity")
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(
        v_values - max_q_values,
        extent=extent,
        aspect="auto",
        origin="lower",
        cmap="RdBu",
        vmin=-qabsolute,
        vmax=qabsolute,
    )
    axes[1].set_title(r"$V(s) - \max_a Q(s,a)$")
    axes[1].set_xlabel("Position")
    axes[1].set_ylabel("Velocity")
    plt.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(
        v_values - max_q_values + max_a_values,
        extent=extent,
        aspect="auto",
        origin="lower",
        cmap="RdBu",
        vmin=-qabsolute,
        vmax=qabsolute,
    )
    axes[2].set_title(r"$V(s) - \max_a Q(s, a) + \max_a A(s,a)$")
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
    axes[3].set_title(r"$\pi(s)$")
    axes[3].set_xlabel("Position")
    axes[3].set_ylabel("Velocity")
    plt.colorbar(im4, ax=axes[3])

    axes[4].set_xlim(POSITION_MIN, POSITION_MAX)
    axes[4].set_ylim(VELOCITY_MIN, VELOCITY_MAX)
    axes[4].set_title(r"Replay Buffer")
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

    path = "outputs/mountaincar_critic_policy"
    Path(path).mkdir(exist_ok=True)
    plt.tight_layout()
    plt.show()
    # plt.savefig(
    #     f"{path}/{i:05d}.png",
    #     bbox_inches="tight",
    #     pad_inches=0,
    #     dpi=300,
    # )
    plt.close()


def get_figure(i: int, state_dict: dict) -> None:
    action_dim = 1
    hidden_dims = [64, 64]

    q_network = algos.AFU.QNetwork(hidden_dims=hidden_dims, num_critics=3)
    v_network = algos.AFU.VNetwork(hidden_dims=hidden_dims, num_critics=2)
    policy_network = algos.AFU.PolicyNetwork(
        hidden_dims=hidden_dims, action_dim=action_dim
    )

    q_params = state_dict["q_params"]
    v_params = state_dict["v_params"]
    policy_params = state_dict["policy_params"]

    grid_size = 50
    n_actions = 50
    v_values, policy_actions, max_q_values, max_a_values = (
        measure_all_values_batched(
            q_params,
            v_params,
            policy_params,
            q_network,
            v_network,
            policy_network,
            grid_size,
            n_actions,
        )
    )

    n_episodes = 3
    avg_trajectory = measure_trajectories(
        policy_params, policy_network, n_episodes
    )

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
        v_values,
        policy_actions,
        max_q_values,
        max_a_values,
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
        "--file", type=str, required=True, help="Path to agent state file"
    )
    args = parser.parse_args()

    checkpoint_files = sorted(Path(args.file).glob("agent_history_*.pk"))
    total_states = len(checkpoint_files) * 100

    with tqdm(total=total_states) as pbar:
        for checkpoint_file in checkpoint_files:
            with open(checkpoint_file, "rb") as f:
                partial_history = pickle.load(f)  # noqa: S301

            for i, state_dict in partial_history.items():
                get_figure(i, state_dict)
                pbar.update(1)

            del partial_history
            gc.collect()


if __name__ == "__main__":
    main()
