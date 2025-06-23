# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship.
#
# isir-internship is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

import algos

POSITION_MIN = -1.2
POSITION_MAX = 0.6
VELOCITY_MIN = -0.07
VELOCITY_MAX = -0.07

ACTION_MIN = -1.0
ACTION_MAX = 1.0


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
        "--file", type=str, required=True, help="Path to weights file"
    )
    args = parser.parse_args()

    with open(args.file, "rb") as f:
        state_dict = pickle.load(f)  # noqa: S301

    state_dim = 2
    action_dim = 1
    hidden_dims = [256, 256]
    match args.algorithm:
        case "sac":
            q1_net = algos.SAC.QNetwork(state_dim, hidden_dims, action_dim)
            q2_net = algos.SAC.QNetwork(state_dim, hidden_dims, action_dim)
            policy_net = algos.SAC.PolicyNetwork(
                state_dim, hidden_dims, action_dim
            )

            q1_net.load_state_dict(state_dict["q1"])
            q2_net.load_state_dict(state_dict["q2"])
            policy_net.load_state_dict(state_dict["policy"])
        case "afu":
            v1_net = algos.AFU.VNetwork(state_dim, hidden_dims)
            v2_net = algos.AFU.VNetwork(state_dim, hidden_dims)
            policy_net = algos.AFU.AFUPolicyNetwork(
                state_dim, hidden_dims, action_dim
            )

            v1_net.load_state_dict(state_dict["v1"])
            v2_net.load_state_dict(state_dict["v2"])
            policy_net.load_state_dict(state_dict["policy"])

    grid_size = 50
    positions = np.linspace(POSITION_MIN, POSITION_MAX, grid_size)
    velocities = np.linspace(VELOCITY_MIN, VELOCITY_MAX, grid_size)

    v_values = np.zeros((grid_size, grid_size))
    actions = np.zeros((grid_size, grid_size))

    for i, pos in enumerate(positions):
        for j, vel in enumerate(velocities):
            state = torch.tensor([[pos, vel]], dtype=torch.float32)

            match args.algorith:
                case "sac":
                    # TODO: COMPUTE
                    v_value = 0
                case "afu":
                    # TODO: COMPUTE
                    v_value = 0

            v_values[j, i] = v_value

            # TODO: compute
            action = 0
            actions[j, i] = action

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
    extent = [POSITION_MIN, POSITION_MAX, VELOCITY_MIN, VELOCITY_MAX]
    im1 = ax1.imshow(
        v_values,
        extent=extent,
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    ax1.set_title(f"Value function V(s) - {args.algorithm.upper()}")
    ax1.set_xlabel("Position")
    ax1.set_ylabel("Velocity")
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(
        actions,
        extent=extent,
        aspect="auto",
        origin="lower",
        cmap="RdBu",
        vmin=-1,
        vmax=1,
    )
    ax2.set_title(f"Policy actions π(s) - {args.algorithm.upper()}")
    ax2.set_xlabel("Position")
    ax2.set_ylabel("Velocity")
    plt.colorbar(im2, ax=ax2)

    for ax in [ax1, ax2]:
        ax.axvline(x=0.45, color="red", linestyle="--", alpha=0.7, label="Goal")
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
