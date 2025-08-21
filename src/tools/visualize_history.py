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
    parser = argparse.ArgumentParser(
        description="Visualize RL training curves",
        usage="%(prog)s --ALGO1 file1.pk --ALGO2 file2.pk [--ALGO3 file3.pk ...]",
    )

    known_args, unknown_args = parser.parse_known_args()

    algo_files = {}
    i = 0
    while i < len(unknown_args):
        if unknown_args[i].startswith("--") and i + 1 < len(unknown_args):
            algo_name = unknown_args[i][2:]
            file_path = unknown_args[i + 1]

            if not file_path.startswith("--"):
                algo_files[algo_name] = file_path
                i += 2
            else:
                i += 1
        else:
            i += 1

    if not algo_files:
        parser.print_help()
        print(
            "\nExample: python script.py --SAC sac_history.pk --AFU afu_history.pk --AFUTEST test_history.pk"
        )
        return

    colors = ["#d66b6a", "#5591e1", "#6ca247", "#39a985", "#ad75ca", "#c77c1e"]
    plt.rcParams["font.size"] = 20
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"

    fig, ax = plt.subplots(figsize=(6, 6))
    for i, (algo_name, file_path) in enumerate(algo_files.items()):
        with open(file_path, "rb") as f:
            history = pickle.load(f)  # noqa: S301

        steps = [k * 500 for k in history]
        stats = [compute_stats(results) for results in history.values()]
        mins, q1s, iqms, q3s, maxs = zip(*stats, strict=True)

        color = colors[i % len(colors)]

        ax.plot(steps, iqms, linewidth=2, color=color, label=algo_name.upper())
        ax.fill_between(steps, q1s, q3s, alpha=0.25, color=color)

    ax.set_xlabel("Training Steps")
    plt.gca().xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{int(x / 1000)}k")
    )
    ax.axhline(y=0, color="black", linewidth=2, zorder=1)
    ax.set_ylabel("Episode Return")
    ax.legend()
    ax.grid(visible=True, alpha=0.25)

    Path("paper/figures").mkdir(exist_ok=True)
    plt.show()


if __name__ == "__main__":
    main()
