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
from matplotlib.ticker import FuncFormatter

from .utils import compute_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Test RL algorithms")
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="History file",
    )
    args = parser.parse_args()

    plt.rcParams["font.size"] = 20
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"

    with open(args.file, "rb") as f:
        history = pickle.load(f)  # noqa: S301

    steps = [k * 1000 for k in history]
    stats = [compute_stats(results) for results in history.values()]

    mins, q1s, iqms, q3s, maxs = zip(*stats, strict=True)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(steps, iqms, linewidth=2, color="#3498db", label="SAC")
    ax.fill_between(steps, q1s, q3s, alpha=0.25, color="#3498db")

    ax.set_xlabel("Training Steps")
    plt.gca().xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{int(x / 1000)}k")
    )
    ax.set_ylabel("Episode Return")
    ax.legend()
    ax.grid(visible=True, alpha=0.25)

    Path("paper/figures").mkdir(exist_ok=True)
    plt.savefig(
        "paper/figures/training_curve.svg",
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
    )


if __name__ == "__main__":
    main()
