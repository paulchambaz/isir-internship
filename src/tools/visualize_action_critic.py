# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship.
#
# isir-internship is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import argparse
import pickle

import algos

POSITION_MIN = -1.2
POSITION_MAX = 0.6
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07

ACTION_MIN = -1.0
ACTION_MAX = 1.0

GOAL_POSITION = 0.45


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize agent (a, Q(s, a)) for a given state"
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
    parser.add_argument(
        "--position",
        type=str,
        required=True,
        help="State position to measure the (a, Q(s, a)) at",
    )
    parser.add_argument(
        "--velocity",
        type=str,
        required=True,
        help="State velocity to measure the (a, Q(s, a)) at",
    )
    args = parser.parse_args()

    with open(args.file, "rb") as f:
        state_history = pickle.load(f)  # noqa: S301


if __name__ == "__main__":
    main()
