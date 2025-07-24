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
from functools import partial
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from tqdm import tqdm

import algos


def measure_trajectories(
    agent: algos.RLAlgo, goal_position: float, episodes: int
) -> list[list[float]]:
    env = gym.make("MountainCarContinuous-v0")

    trajectories = []
    for _ in range(episodes):
        trajectory = []
        state, _ = env.reset()

        while True:
            trajectory.append(state)

            state_tensor = jnp.array(state)
            action = agent.select_action(state_tensor, evaluation=True)

            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if done:
                break

            state = next_state

        trajectories.append(np.array(trajectory))

    env.close()

    # TODO: difference between mountaincar and pendulum
    # TODO: we should have a clean helper function that does that cleanly
    state_steps = zip(*trajectories, strict=False)
    filtered_steps = [
        [
            (position, velocity)
            for (position, velocity) in states
            if position < goal_position
        ]
        for states in state_steps
    ]
    mean_steps = [np.mean(step, axis=0) for step in filtered_steps]
    return [mean_step.tolist() for mean_step in mean_steps]


@partial(jax.jit, static_argnums=(0,))
def _compute_v_values(
    agent: algos.RLAlgo,
    v_params: dict[str, jax.Array],
    states_batch: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.min(agent.v_network.apply(v_params, states_batch), axis=0)


def measure_v_values(
    agent: algos.RLAlgo, states_grid: jnp.ndarray, grid_size: int
) -> jnp.ndarray:
    v_values = _compute_v_values(agent, agent.v_params, states_grid)
    return v_values.reshape(grid_size, grid_size).T


@partial(jax.jit, static_argnums=(0, 1, 2))
def _compute_qa_values(
    agent: algos.RLAlgo,
    n_states: int,
    n_actions: int,
    q_params: dict[str, jax.Array],
    states_batch: jnp.ndarray,
    actions_batch: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    q_values_list = agent.q_network.apply(q_params, states_batch, actions_batch)

    all_q_values = q_values_list[-1:]

    all_a_values_list = -q_values_list[:-1]
    all_a_values = jnp.min(all_a_values_list, axis=0)

    max_q_per_state = jnp.max(all_q_values.reshape(n_states, n_actions), axis=1)
    max_a_per_state = jnp.max(all_a_values.reshape(n_states, n_actions), axis=1)

    return max_a_per_state, max_q_per_state


@partial(jax.jit, static_argnums=(0,))
def _compute_actions(
    agent: algos.RLAlgo,
    policy_params: dict[str, jax.Array],
    states_batch: jnp.ndarray,
) -> jnp.ndarray:
    mean, _ = agent.policy_network.apply(policy_params, states_batch)
    return jnp.tanh(mean).flatten()


def measure_actions(
    agent: algos.RLAlgo, states_grid: jnp.ndarray, grid_size: int
) -> jnp.ndarray:
    actions = _compute_actions(agent, agent.policy_params, states_grid)
    return actions.reshape(grid_size, grid_size).T


def measure_qa_values(
    agent: algos.RLAlgo,
    states_grid: jnp.ndarray,
    grid_size: int,
    n_actions: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    n_states = grid_size * grid_size

    actions = jnp.linspace(-1.0, 1.0, n_actions)
    all_states_grid = jnp.repeat(states_grid, n_actions, axis=0)
    all_actions_grid = jnp.tile(actions[:, None], (n_states, 1))

    max_a_per_state, max_q_per_state = _compute_qa_values(
        agent,
        n_states,
        n_actions,
        agent.q_params,
        all_states_grid,
        all_actions_grid,
    )
    a_values = max_a_per_state.reshape(grid_size, grid_size).T
    q_values = max_q_per_state.reshape(grid_size, grid_size).T

    return a_values, q_values


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


def create_figure(
    i: int,
    v_values: np.ndarray,
    q_values: np.ndarray,
    a_values: np.ndarray,
    actions: np.ndarray,
    avg_trajectory: np.ndarray,
    replay_positions: np.ndarray,
    replay_velocities: np.ndarray,
    hull_pos: np.ndarray,
    hull_vel: np.ndarray,
    position_min: float,
    position_max: float,
    velocity_min: float,
    velocity_max: float,
    goal_position: float,
) -> None:
    plt.rcParams["font.size"] = 20
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    axes = axes.flatten()
    extent = [position_min, position_max, velocity_min, velocity_max]

    v_all_values = np.concatenate(
        [
            v_values.flatten(),
            q_values.flatten(),
            (v_values + a_values).flatten(),
        ]
    )
    vmin_shared = np.min(v_all_values)
    vmax_shared = np.max(v_all_values)

    q_all_values = np.concatenate(
        [
            (v_values - q_values).flatten(),
            (v_values - q_values + a_values).flatten(),
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
        v_values - q_values,
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
        v_values - q_values + a_values,
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

    axes[4].set_xlim(position_min, position_max)
    axes[4].set_ylim(velocity_min, velocity_max)
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
            x=goal_position, color="black", linestyle="--", linewidth=1
        )

    for j in range(3):
        axes[j].plot(hull_pos, hull_vel, "k--", linewidth=1)

    if len(avg_trajectory) > 1:
        avg_trajectory = np.array(avg_trajectory)
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
    plt.savefig(
        f"{path}/{i:05d}.png",
        bbox_inches="tight",
        dpi=150,
    )
    plt.close()


def get_figure(i: int, state: dict) -> None:
    agent = algos.AFU(
        state_dim=2,
        action_dim=1,
        hidden_dims=[64, 64],
        replay_size=200_000,
        batch_size=256,
        critic_lr=3e-4,
        policy_lr=3e-4,
        temperature_lr=3e-4,
        tau=0.005,
        gamma=0.99,
        alpha=None,
        rho=0.7,
        seed=42,
        state=state,
    )

    grid_size = 50
    n_actions = 50

    position_min = -1.2
    position_max = 0.6
    velocity_min = -0.07
    velocity_max = 0.07
    goal_position = 0.45

    avg_trajectory = measure_trajectories(agent, goal_position, episodes=3)

    positions = np.linspace(position_min, position_max, grid_size)
    velocities = np.linspace(velocity_min, velocity_max, grid_size)
    pos_grid, vel_grid = np.meshgrid(positions, velocities, indexing="ij")
    states_grid = jnp.stack([pos_grid.flatten(), vel_grid.flatten()], axis=1)

    v_values = measure_v_values(agent, states_grid, grid_size)
    a_values, q_values = measure_qa_values(
        agent, states_grid, grid_size, n_actions
    )
    actions = measure_actions(agent, states_grid, grid_size)

    replay_positions, replay_velocities = extract_replay_buffer_states(state)
    hull_pos, hull_vel = get_replay_buffer_convex_hull(
        replay_positions, replay_velocities
    )

    del agent

    create_figure(
        i,
        v_values,
        q_values,
        a_values,
        actions,
        avg_trajectory,
        replay_positions,
        replay_velocities,
        hull_pos,
        hull_vel,
        position_min,
        position_max,
        velocity_min,
        velocity_max,
        goal_position,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize MountainCar value functions and policies"
    )
    parser.add_argument(
        "--dir", type=str, required=True, help="Path to agent state directory"
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Force JAX to use CPU instead of GPU"
    )
    args = parser.parse_args()

    if args.cpu:
        jax.config.update("jax_platform_name", "cpu")

    checkpoint_files = sorted(Path(args.dir).glob("agent_history_*.pk"))
    total_states = len(checkpoint_files) * 100

    with tqdm(total=total_states) as pbar:
        for checkpoint_file in checkpoint_files:
            with open(checkpoint_file, "rb") as f:
                partial_history = pickle.load(f)  # noqa: S301

            for i, state_dict in partial_history.items():
                if i <= 2000:
                    get_figure(i, state_dict)
                pbar.update(1)

            del partial_history
            jax.clear_caches()
            gc.collect()


if __name__ == "__main__":
    main()
