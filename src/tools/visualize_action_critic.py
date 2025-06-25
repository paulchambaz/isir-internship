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

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import algos

ACTION_MIN = -1.0
ACTION_MAX = 1.0

COLORS = ["#6ca247", "#d66b6a", "#5591e1", "#39a985", "#ad75ca", "#c77c1e"]

MAX_1 = 0
MIN_1 = 0
MAX_2 = 0
MIN_2 = 0
MAX_3 = 0
MIN_3 = 0


def get_figure(i, state_dict, position, velocity):
    global MAX_1, MIN_1, MAX_2, MIN_2, MAX_3, MIN_3

    state_dim = 2
    action_dim = 1
    hidden_dims = [256, 256]

    q_net = algos.AFU.QNetwork(
        state_dim, action_dim, hidden_dims, num_critics=3
    )
    q_net.load_state_dict(state_dict["q_network"])

    v_net = algos.AFU.VNetwork(state_dim, hidden_dims, num_critics=2)
    v_net.load_state_dict(state_dict["v_network"])

    policy_net = algos.AFU.PolicyNetwork(
        state_dim, action_dim, hidden_dims, log_std_min=10.0, log_std_max=2.0
    )
    policy_net.load_state_dict(state_dict["policy_network"])

    state = torch.tensor([[position, velocity]], dtype=torch.float32)

    with torch.no_grad():
        v_values_list = v_net(state)
        v_value = float(torch.min(torch.stack(v_values_list), dim=0)[0])
        mean, log_std = policy_net(state)
        policy_action = float(torch.tanh(mean).squeeze())

    n_actions = 100
    actions = np.linspace(ACTION_MIN, ACTION_MAX, n_actions)
    actions_tensor = torch.tensor(actions, dtype=torch.float32).unsqueeze(1)
    states_batch = state.expand(n_actions, -1)

    with torch.no_grad():
        q_values_list = q_net(states_batch, actions_tensor)
        a_values = -(q_values_list[0] + q_values_list[1]).squeeze().numpy() / 2
        q_values = q_values_list[2].squeeze().numpy()

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
    if MAX_1 == 0 and MIN_1 == 0:
        MAX_1 = q_values.max()
        MIN_1 = q_values.min()
        MAX_2 = max(q_values.max(), v_value, (v_value + a_values).max())
        MIN_2 = min(q_values.min(), v_value, (v_value + a_values).min())
        MAX_3 = a_values.max()
        MIN_3 = a_values.min()
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

        MAX_1 = max(
            MAX_1 * (1 - lerp_factor) + target_max_1 * lerp_factor, target_max_1
        )
        MIN_1 = min(
            MIN_1 * (1 - lerp_factor) + target_min_1 * lerp_factor, target_min_1
        )
        MAX_2 = max(
            MAX_2 * (1 - lerp_factor) + target_max_2 * lerp_factor, target_max_2
        )
        MIN_2 = min(
            MIN_2 * (1 - lerp_factor) + target_min_2 * lerp_factor, target_min_2
        )
        MAX_3 = max(
            MAX_3 * (1 - lerp_factor) + target_max_3 * lerp_factor, target_max_3
        )
        MIN_3 = min(
            MIN_3 * (1 - lerp_factor) + target_min_3 * lerp_factor, target_min_3
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
        [MIN_1 + (MAX_1 - MIN_1) * 0.02],
        color=COLORS[5],
        s=100,
        zorder=5,
        label=r"$a_Q^*(s)$",
    )
    axes[0].scatter(
        [policy_action],
        [MIN_1 + (MAX_1 - MIN_1) * 0.02],
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
    axes[0].set_ylim(MIN_1, MAX_1)

    axes[1].plot(actions, q_values, color=COLORS[0], linewidth=4)
    axes[1].axhline(y=v_value, color=COLORS[1], linewidth=2)
    axes[1].plot(actions, v_value + a_values, color=COLORS[2], linewidth=2)
    axes[1].scatter([mode_action], [MIN_2], color=COLORS[5], s=100, zorder=5)
    axes[1].scatter([policy_action], [MIN_2], color=COLORS[3], s=100, zorder=5)
    axes[1].set_xlabel("Action")
    axes[1].set_ylabel("Value")
    axes[1].set_title("Q = V + A")
    axes[1].grid(visible=True, alpha=0.25)
    axes[1].set_ylim(MIN_2, MAX_2)

    axes[2].plot(actions, a_values, color=COLORS[4], linewidth=2)
    axes[2].axhline(y=0, color="k", linestyle="-", alpha=0.3)
    axes[2].scatter([mode_action], [MIN_3], color=COLORS[5], s=100, zorder=5)
    axes[2].scatter([policy_action], [MIN_3], color=COLORS[3], s=100, zorder=5)
    axes[2].set_xlabel("Action")
    axes[2].set_ylabel(r"$A(s, a)$")
    axes[2].set_title("Advantage")
    axes[2].grid(visible=True, alpha=0.25)
    axes[2].set_ylim(MIN_3, max(0, MAX_3))

    path = "outputs/mountaincar_action_critic"
    Path(path).mkdir(exist_ok=True)
    plt.suptitle(f"{i * 1000} steps - state: [{position}, {velocity}]")
    plt.tight_layout()
    plt.savefig(
        f"{path}/{i:05d}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


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

    with open(args.file, "rb") as f:
        state_history = pickle.load(f)  # noqa: S301

    for i, state_dict in tqdm(state_history.items()):
        # if i == max(state_history.keys()):
        get_figure(i, state_dict, args.position, args.velocity)


if __name__ == "__main__":
    main()
