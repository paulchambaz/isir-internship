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

from tqdm import tqdm


def get_figure(i: int, state_dict: dict) -> None:
    pass


def main() -> None:
    # TODO: update to pendulum
    parser = argparse.ArgumentParser(
        description="Visualize MountainCar value functions and policies"
    )
    parser.add_argument(
        "--dir", type=str, required=True, help="Path to agent state directory"
    )
    args = parser.parse_args()

    checkpoint_files = sorted(Path(args.dir).glob("agent_history_*.pk"))
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
