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
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import algos

COLORS = ["#6ca247", "#d66b6a", "#5591e1", "#39a985", "#ad75ca", "#c77c1e"]


@dataclass
class Bounds:
    max_1: 0
    min_1: 0
    max_2: 0
    min_2: 0
    max_3: 0
    min_3: 0


def get_figure(
    i: int, state_dict: dict, position: float, velocity: float, bounds: Bounds
) -> None:
    action_dim = 1
    hidden_dims = [256, 256]

    q_network = algos.AFU.QNetwork(hidden_dims=hidden_dims, num_critics=3)
    v_network = algos.AFU.VNetwork(hidden_dims=hidden_dims, num_critics=2)
    policy_network = algos.AFU.PolicyNetwork(
        hidden_dims=hidden_dims, action_dim=action_dim
    )

    state = jnp.array([[position, velocity]], dtype=jnp.float32)

    q_params = state_dict["q_params"]
    v_params = state_dict["v_params"]
    policy_params = state_dict["policy_params"]

    v_values_list = v_network.apply(v_params, state)
    v_value = float(jnp.min(jnp.stack(v_values_list), dim=0)[0])

    mean, log_std = policy_network.apply(policy_params, state)
    policy_action = float(jnp.tanh(mean).squeeze())

    n_actions = 100
    actions = np.linspace(-1.0, 1.0, n_actions)
    actions_tensor = jnp.tensor(actions, dtype=jnp.float32).unsqueeze(1)
    states_batch = state.tile(state, (n_actions, 1))

    q_values_list = q_network.apply(q_params, states_batch, actions_tensor)
    a_values = -(q_values_list[0] + q_values_list[1]).squeeze() / 2
    q_values = q_values_list[2].squeeze()

    a_values = np.array(a_values)
    q_values = np.array(q_values)

    mode_action_idx = np.argmax(q_values)
    mode_action = actions[mode_action_idx]

    plt.rcParams["font.size"] = 20
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"

    target_max_1 = max(q_values.max(), v_value, (v_value + a_values).max())
    target_min_1 = min(q_values.min(), v_value, (v_value + a_values).min())
    target_max_2 = target_max_1
    target_min_2 = target_min_1
    target_max_3 = a_values.max()
    target_min_3 = a_values.min()

    lerp_factor = 0.5
    if bounds.max_1 == 0 and bounds.min_1 == 0:
        bounds.max_1 = q_values.max()
        bounds.min_1 = q_values.min()
        bounds.max_2 = max(q_values.max(), v_value, (v_value + a_values).max())
        bounds.min_2 = min(q_values.min(), v_value, (v_value + a_values).min())
        bounds.max_3 = a_values.max()
        bounds.min_3 = a_values.min()
    else:
        target_max_1 = q_values.max()
        target_min_1 = q_values.min()
        target_max_2 = max(q_values.max(), v_value, (v_value + a_values).max())
        target_min_2 = min(q_values.min(), v_value, (v_value + a_values).min())
        target_max_3 = a_values.max()
        target_min_3 = a_values.min()

        range_1 = target_max_1 - target_min_1
        range_2 = target_max_2 - target_min_2
        range_3 = target_max_3 - target_min_3

        target_max_1 += range_1 * 0.1
        target_min_1 -= range_1 * 0.1
        target_max_2 += range_2 * 0.1
        target_min_2 -= range_2 * 0.1
        target_max_3 += range_3 * 0.1
        target_min_3 -= range_3 * 0.1

        bounds.max_1 = max(
            bounds.max_1 * (1 - lerp_factor) + target_max_1 * lerp_factor,
            target_max_1,
        )
        bounds.min_1 = min(
            bounds.min_1 * (1 - lerp_factor) + target_min_1 * lerp_factor,
            target_min_1,
        )
        bounds.max_2 = max(
            bounds.max_2 * (1 - lerp_factor) + target_max_2 * lerp_factor,
            target_max_2,
        )
        bounds.min_2 = min(
            bounds.min_2 * (1 - lerp_factor) + target_min_2 * lerp_factor,
            target_min_2,
        )
        bounds.max_3 = max(
            bounds.max_3 * (1 - lerp_factor) + target_max_3 * lerp_factor,
            target_max_3,
        )
        bounds.min_3 = min(
            bounds.min_3 * (1 - lerp_factor) + target_min_3 * lerp_factor,
            target_min_3,
        )

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])
    plt.subplots_adjust(
        left=0.08, right=0.95, top=0.92, bottom=0.1, wspace=0.3, hspace=0.3
    )

    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])

    axes = [ax1, ax2, ax3]

    axes[0].plot(
        actions, q_values, color=COLORS[0], linewidth=4, label=r"$Q(s,a)$"
    )
    axes[0].plot([], [], color=COLORS[1], linewidth=2, label=r"V(s)")
    axes[0].plot([], [], color=COLORS[2], linewidth=2, label=r"V(s) + A(s, a)")
    axes[0].plot([], [], color=COLORS[4], linewidth=2, label=r"$A(s,a)$")
    axes[0].scatter(
        [mode_action],
        [bounds.min_1 + (bounds.max_1 - bounds.min_1) * 0.02],
        color=COLORS[5],
        s=100,
        zorder=5,
        label=r"$a_Q^*(s)$",
    )
    axes[0].scatter(
        [policy_action],
        [bounds.min_1 + (bounds.max_1 - bounds.min_1) * 0.02],
        color=COLORS[3],
        s=100,
        zorder=5,
        label=r"$\pi(s)$",
    )
    axes[0].set_xlabel("Action")
    axes[0].set_ylabel(r"$Q(s, a)$")
    axes[0].set_title("Q-values")
    axes[0].grid(visible=True, alpha=0.25)
    axes[0].legend(loc="upper left")
    axes[0].set_ylim(bounds.min_1, bounds.max_1)

    axes[1].plot(actions, q_values, color=COLORS[0], linewidth=4)
    axes[1].axhline(y=v_value, color=COLORS[1], linewidth=2)
    axes[1].plot(actions, v_value + a_values, color=COLORS[2], linewidth=2)
    axes[1].scatter(
        [mode_action], [bounds.min_2], color=COLORS[5], s=100, zorder=5
    )
    axes[1].scatter(
        [policy_action], [bounds.min_2], color=COLORS[3], s=100, zorder=5
    )
    axes[1].set_xlabel("Action")
    axes[1].set_ylabel("Value")
    axes[1].set_title("Q = V + A")
    axes[1].grid(visible=True, alpha=0.25)
    axes[1].set_ylim(bounds.min_2, bounds.max_2)

    axes[2].plot(actions, a_values, color=COLORS[4], linewidth=2)
    axes[2].axhline(y=0, color="k", linestyle="-", alpha=0.3)
    axes[2].scatter(
        [mode_action], [bounds.min_3], color=COLORS[5], s=100, zorder=5
    )
    axes[2].scatter(
        [policy_action], [bounds.min_3], color=COLORS[3], s=100, zorder=5
    )
    axes[2].set_xlabel("Action")
    axes[2].set_ylabel(r"$A(s, a)$")
    axes[2].set_title("Advantage")
    axes[2].grid(visible=True, alpha=0.25)
    axes[2].set_ylim(bounds.min_3, max(0, bounds.max_3))

    path = "outputs/mountaincar_action_critic"
    Path(path).mkdir(exist_ok=True)
    plt.suptitle(f"{i * 50} steps - state: [{position}, {velocity}]")
    plt.tight_layout()
    plt.savefig(
        f"{path}/{i:05d}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def load_full_agent_history(directory: str) -> None:
    full_agent_history = {}

    checkpoint_files = sorted(
        Path(directory).glob("agent_history_*.pk"),
    )

    for checkpoint_file in checkpoint_files:
        with open(checkpoint_file, "rb") as f:
            partial_history = pickle.load(f)  # noqa: S301

        full_agent_history.update(partial_history)

        del partial_history
        gc.collect()

    return full_agent_history


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize agent (a, Q(s, a)) for a given state"
    )
    parser.add_argument(
        "--file", type=str, required=True, help="Path to agent state file"
    )
    parser.add_argument(
        "--position",
        type=float,
        required=True,
        help="State position to measure the (a, Q(s, a)) at",
    )
    parser.add_argument(
        "--velocity",
        type=float,
        required=True,
        help="State velocity to measure the (a, Q(s, a)) at",
    )
    args = parser.parse_args()

    checkpoint_files = sorted(
        Path(args.file).glob("agent_history_*.pk"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    total_states = len(checkpoint_files) * 100

    bounds = Bounds(
        max_1=0,
        min_1=0,
        max_2=0,
        min_2=0,
        max_3=0,
        min_3=0,
    )

    with tqdm(total=total_states) as pbar:
        for checkpoint_file in checkpoint_files:
            with open(checkpoint_file, "rb") as f:
                partial_history = pickle.load(f)  # noqa: S301

            for i, state_dict in partial_history.items():
                get_figure(i, state_dict, args.position, args.velocity, bounds)
                pbar.update(1)

            del partial_history
            gc.collect()


if __name__ == "__main__":
    main()
