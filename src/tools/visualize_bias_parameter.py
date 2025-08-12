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
from tqdm import tqdm

from .utils import compute_stats

algorithm_colors = {
    "avg": "#5591e1",
    "min": "#d76868",
    "tqc": "#6aa142",
    "ttqc": "#00a97f",
    "ndtop": "#a19100",
    "top": "#c87b13",
    "afu": "#a178d5",
}


def whisker_plot(stats_list, params, color, symbol, method, step) -> None:
    plt.figure(figsize=(10, 6))

    positions = np.arange(len(params))
    iqm_values = []

    for i, stats in enumerate(stats_list):
        min_val, q1, iqm, q3, max_val = stats

        iqm_values.append(iqm)

        box_width = 0.15
        whisker_width = 0.08

        plt.gca().add_patch(
            plt.Rectangle(
                (i - box_width / 2, q1),
                box_width,
                q3 - q1,
                color=color,
                alpha=0.2,
            )
        )

        plt.plot(
            [i - box_width / 2, i + box_width / 2],
            [q1, q1],
            color=color,
            linewidth=2,
        )
        plt.plot(
            [i - box_width / 2, i + box_width / 2],
            [q3, q3],
            color=color,
            linewidth=2,
        )

        plt.plot(
            [i, i],
            [min_val, q1],
            color=color,
            linewidth=1.5,
        )
        plt.plot(
            [i, i],
            [q3, max_val],
            color=color,
            linewidth=1.5,
        )

        plt.plot(
            [i - whisker_width / 2, i + whisker_width / 2],
            [min_val, min_val],
            color=color,
            linewidth=1.5,
        )
        plt.plot(
            [i - whisker_width / 2, i + whisker_width / 2],
            [max_val, max_val],
            color=color,
            linewidth=1.5,
        )

        plt.plot(
            i,
            iqm,
            "o",
            markersize=8,
            markeredgecolor=color,
            markerfacecolor=color,
        )
        plt.plot(
            [i - box_width / 2, i + box_width / 2],
            [iqm, iqm],
            color=color,
            linewidth=1,
        )

    plt.plot(
        positions,
        iqm_values,
        color="gray",
        linestyle=":",
        linewidth=2,
        zorder=1000,
        alpha=0.8,
    )

    plt.xticks(positions, [str(param) for param in params])
    plt.xlabel(f"${symbol}$")
    plt.ylabel("Bias")
    plt.title(f"{method.upper()} - Step {step}")
    plt.grid(visible=True, axis="y", alpha=0.3)

    plt.tight_layout()

    plt.show()

    # directory = f"paper/figures/bias_parameter/{method}"
    # Path(directory).mkdir(parents=True, exist_ok=True)
    # plt.savefig(f"{directory}/{step:05d}.png", bbox_inches="tight", dpi=150)
    # plt.close()


def create_bias_parameter_plot(
    results: dict,
    method: str,
    params: list[int | float],
    symbol: float,
    step: int,
) -> None:
    stats_list = []
    for n in params:
        if (method, n) not in results or step not in results[(method, n)][
            "raw_data"
        ]:
            continue

        metadata = results[(method, n)]["metadata"]
        true_q = np.array(metadata["true_q"])

        raw_data = results[(method, n)]["raw_data"][step]
        all_predicted_q = np.array(
            [np.array(predicted_q) for _, _, _, predicted_q in raw_data]
        )
        predicted_q = np.mean(all_predicted_q, axis=0)

        bias = predicted_q - true_q

        stats_list.append(compute_stats(bias))

    whisker_plot(
        stats_list, params, algorithm_colors[method], symbol, method, step
    )


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

    create_bias_parameter_plot(
        results, "afu", [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95], r"\rho", 5000
    )

    exit()
    for i in tqdm(range(100, 5001, 100)):
        create_bias_parameter_plot(
            results, "avg", [1, 3, 5, 10, 20, 50], "N", i
        )
        create_bias_parameter_plot(results, "min", [2, 3, 4, 6, 8, 10], "N", i)
        create_bias_parameter_plot(
            results, "tqc", [1, 2, 3, 4, 6, 10, 14], "d", i
        )
        create_bias_parameter_plot(
            results, "ttqc", [1, 2, 3, 4, 6, 10, 14], "d", i
        )
        create_bias_parameter_plot(
            results, "ndtop", [-1.0, -0.7, -0.5, 0.0, 0.5, 1.0], r"\beta", i
        )
        create_bias_parameter_plot(
            results, "top", [-1.0, -0.7, -0.5, 0.0, 0.5, 1.0], r"\beta", i
        )
        create_bias_parameter_plot(
            results, "afu", [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95], r"\rho", i
        )


if __name__ == "__main__":
    main()
