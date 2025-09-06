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
from matplotlib.ticker import FuncFormatter

from .utils import compute_stats

COLORS = {
    # avg
    ("avg", 1): "#000a4e",
    ("avg", 3): "#00347c",
    ("avg", 5): "#205da8",
    ("avg", 10): "#4b87d6",
    ("avg", 20): "#75b3ff",
    ("avg", 50): "#c0e1ff",
    # min
    ("min", 2): "#370001",
    ("min", 3): "#6f0013",
    ("min", 4): "#9d3438",
    ("min", 6): "#cd5f5f",
    ("min", 8): "#fe8a88",
    ("min", 10): "#ff9895",
    # tqc
    ("tqc", 0): "#000500",
    ("tqc", 1): "#021800",
    ("tqc", 2): "#0b2d00",
    ("tqc", 3): "#184400",
    ("tqc", 4): "#285c00",
    ("tqc", 5): "#3f7307",
    ("tqc", 6): "#568b2b",
    ("tqc", 7): "#6da445",
    ("tqc", 10): "#85be5f",
    ("tqc", 13): "#9ed878",
    ("tqc", 16): "#b7f291",
}


SYMBOLS = {
    "avg": "N",
    "min": "N",
    "tqc": "d",
}

TITLES = {
    "avg": r"AVG ($N$ Q-networks)",
    "min": r"MIN ($N$ Q-networks)",
    "tqc": r"TQC $N=2$ $M=25$ ($d$ dropped quantiles)",
}


def visualize(results: dict, current_method: str, ax) -> None:
    for (method, n), step_data in results.items():
        if method != current_method:
            continue

        steps = step_data.keys()

        iqms = []
        q1s = []
        q3s = []
        for _, data in step_data.items():
            policy_action = np.array(data["policy_action"])
            true_qs = np.array(data["true_qs"])
            actions = np.array(data["actions"])

            closest_indices = np.argmin(
                np.abs(actions - policy_action[:, None]), axis=1
            )
            returns = true_qs[np.arange(len(policy_action)), closest_indices]

            _, q1, iqm, q3, _ = compute_stats(returns)

            q1s.append(q1)
            iqms.append(iqm)
            q3s.append(q3)

        ax.fill_between(
            steps,
            q3s,
            q1s,
            color=COLORS[(method, n)],
            alpha=0.25,
            zorder=2,
        )
        ax.plot(
            steps,
            iqms,
            color=COLORS[(method, n)],
            label=rf"{SYMBOLS[current_method]}={n}",
            linewidth=4,
            zorder=4,
        )

    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{int(x / 1000)}k")
    )

    ax.set_title(TITLES[current_method], fontsize=24, pad=20)

    ax.set_xlabel("Training step")

    if current_method == "avg":
        ax.set_ylabel("Returns")

    ax.grid(visible=True, alpha=0.25)
    ax.legend(loc="lower right")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()

    with open(args.file, "rb") as f:
        results = pickle.load(f)  # noqa:S301

    plt.rcParams["font.size"] = 20
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["text.latex.preamble"] = (
        r"\usepackage{amsmath}\usepackage{amssymb}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(21, 8))

    visualize(results, "avg", axes[0])
    visualize(results, "min", axes[1])
    visualize(results, "tqc", axes[2])

    plt.tight_layout()
    plt.savefig(
        "paper/figures/returns_over_time_combined.png",
        bbox_inches="tight",
        dpi=300,
    )
    # plt.show()
    plt.close()


if __name__ == "__main__":
    main()
