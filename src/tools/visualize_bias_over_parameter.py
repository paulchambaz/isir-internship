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

from .utils import compute_stats

COLORS = {
    "avg": "#5591e1",
    "min": "#d76868",
    "tqc": "#6aa142",
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


def visualize(results: dict, current_method: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))

    params = []
    stats_list = []

    for (method, n), step_data in results.items():
        if method != current_method:
            continue

        params.append(n)

        total_errors = []
        for _, data in step_data.items():
            predicted_qs = np.array(data["predicted_qs"])
            policy_qs = np.array(data["policy_qs"])
            errors = predicted_qs - policy_qs
            total_errors.extend(errors.flatten())

        stats_list.append(compute_stats(total_errors))

    color = COLORS[current_method]
    symbol = SYMBOLS[current_method]

    positions = np.arange(len(params))
    iqm_values = []

    ax.axhline(y=0, color="black", linewidth=2, zorder=1)

    for i, stats in enumerate(stats_list):
        min_val, q1, iqm, q3, max_val = stats
        iqm_values.append(iqm)

        box_width = 0.25
        whisker_width = 0.16

        ax.add_patch(
            plt.Rectangle(
                (i - box_width / 2, q1),
                box_width,
                q3 - q1,
                color=color,
                alpha=0.25,
                zorder=2,
            )
        )

        ax.plot(
            [i - box_width / 2, i + box_width / 2],
            [q1, q1],
            color=color,
            linewidth=4,
            zorder=2,
        )
        ax.plot(
            [i - box_width / 2, i + box_width / 2],
            [q3, q3],
            color=color,
            linewidth=4,
            zorder=2,
        )

        ax.plot([i, i], [min_val, q1], color=color, linewidth=3, zorder=2)
        ax.plot([i, i], [q3, max_val], color=color, linewidth=3, zorder=2)

        ax.plot(
            [i - whisker_width / 2, i + whisker_width / 2],
            [min_val, min_val],
            color=color,
            linewidth=3,
            zorder=2,
        )
        ax.plot(
            [i - whisker_width / 2, i + whisker_width / 2],
            [max_val, max_val],
            color=color,
            linewidth=3,
            zorder=2,
        )

        ax.plot(
            i,
            iqm,
            "o",
            markersize=8,
            markeredgecolor=color,
            markerfacecolor=color,
            zorder=2,
        )
        ax.plot(
            [i - box_width / 2, i + box_width / 2],
            [iqm, iqm],
            color=color,
            linewidth=2,
            zorder=2,
        )

    ax.plot(
        positions,
        iqm_values,
        color="gray",
        linestyle=":",
        linewidth=3,
        zorder=1000,
        alpha=0.8,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels([str(param) for param in params])
    ax.set_xlabel(f"${symbol}$")
    ax.set_ylabel("Bias")
    ax.set_title(TITLES[current_method], fontsize=24, pad=20)
    ax.grid(visible=True, axis="y", alpha=0.3)

    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(-3e2, 3e4)

    plt.tight_layout()

    # plt.show()

    plt.savefig(
        f"paper/figures/bias_over_parameter_{current_method}.png",
        bbox_inches="tight",
        dpi=300,
    )

    plt.close()


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

    visualize(results, "avg")
    visualize(results, "min")
    visualize(results, "tqc")


if __name__ == "__main__":
    main()
