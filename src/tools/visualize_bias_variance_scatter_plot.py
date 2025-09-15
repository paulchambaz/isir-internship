# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship.
#
# isir-internship is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import argparse
import pickle
from itertools import chain
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from tqdm import tqdm

COLORS = {
    "avg": "#5591e1",
    "min": "#d76868",
    "tqc": "#6aa142",
}


def visualize(results: dict, step: int) -> None:
    processed_results = {}

    for (method, n), step_data in results.items():
        data = step_data[step]

        biases = []
        variances = []
        policy_errors = []

        for seed in range(len(data["predicted_qs"])):
            predicted_qs = data["predicted_qs"][seed]
            policy_qs = data["policy_qs"][seed]
            # true_qs = data["true_qs"][seed]
            policy_action = data["policy_action"][seed]
            optimal_action = data["optimal_action"][seed]

            error = predicted_qs - policy_qs
            # error = predicted_qs - true_qs
            biases.append(error.mean())
            variances.append(error.var())
            policy_errors.append(abs(policy_action - optimal_action))

        bias = np.mean(biases)
        variance = np.mean(variances)
        policy_error = np.mean(np.array(policy_errors))

        processed_results[(method, n)] = (bias, variance, policy_error)

    results = processed_results

    plt.rcParams["font.size"] = 20
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["text.latex.preamble"] = (
        r"\usepackage{amsmath}\usepackage{amssymb}"
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.axvline(x=0, color="black", linewidth=2, zorder=1)

    ax.set_title(f"Training step: {step}", fontsize=24, pad=20)

    ax.grid(visible=True, which="major", alpha=0.6, linewidth=0.8)

    ax.set_xlabel(
        r"$\mathbb{E}[\Delta(a)]$, $\quad a \sim \mathcal{U}(-1,1)$",
        fontsize=30,
    )
    ax.set_ylabel(
        r"$\mathbb{V}\text{ar}[\Delta(a)]$, $\quad a \sim \mathcal{U}(-1,1)$",
        fontsize=30,
    )

    tqc_points = {
        "x": [],
        "y": [],
        "sizes": [],
        "numbers": [],
        "color": COLORS["tqc"],
    }
    min_points = {
        "x": [],
        "y": [],
        "sizes": [],
        "numbers": [],
        "color": COLORS["min"],
    }
    avg_points = {
        "x": [],
        "y": [],
        "sizes": [],
        "numbers": [],
        "color": COLORS["avg"],
    }

    for (method, n), (bias, variance, policy_error) in results.items():
        size = 2 * max(100, min(500, 400 - policy_error * 1000))

        if method == "tqc":
            tqc_points["x"].append(bias)
            tqc_points["y"].append(variance)
            tqc_points["sizes"].append(size)
            tqc_points["numbers"].append(str(n))
        elif method == "min":
            min_points["x"].append(bias)
            min_points["y"].append(variance)
            min_points["sizes"].append(size)
            min_points["numbers"].append(str(n))
        elif method == "avg":
            avg_points["x"].append(bias)
            avg_points["y"].append(variance)
            avg_points["sizes"].append(size)
            avg_points["numbers"].append(str(n))

    for data in [tqc_points, min_points, avg_points]:
        ax.scatter(
            data["x"],
            data["y"],
            c=data["color"],
            s=data["sizes"],
            alpha=0.7,
        )
        for x, y, num in zip(
            data["x"],
            data["y"],
            data["numbers"],
            strict=True,
        ):
            ax.text(
                x,
                y,
                num,
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
            )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS["tqc"],
            markersize=18,
            alpha=0.7,
            label="TQC N=2 M=25 (d dropped quantiles)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS["min"],
            markersize=18,
            alpha=0.7,
            label="MIN (N Q-networks)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS["avg"],
            markersize=18,
            alpha=0.7,
            label="AVG (N Q-networks)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="black",
            markersize=15,
            alpha=0.8,
            label="far from optimal policy",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="black",
            markersize=25,
            alpha=0.8,
            label="close to optimal policy",
        ),
    ]

    legend = ax.legend(
        handles=legend_elements,
        loc="upper left",
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=16,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.9)

    ax.set_xscale("symlog", linthresh=1.8)
    ax.set_xlim(-3e2, 3e4)
    # ax.set_xlim(-1e2, 1e2)

    ax.set_yscale("log")
    ax.set_ylim(4e-3, 2.5e4)

    plt.draw()

    plt.tight_layout()

    # plt.show()

    directory = "paper/figures/bias_variance_scatter_plot"
    Path(directory).mkdir(parents=True, exist_ok=True)
    plt.savefig(
        f"{directory}/{step:05d}.png",
        bbox_inches="tight",
        dpi=100,
    )

    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()

    with open(args.file, "rb") as f:
        results = pickle.load(f)  # noqa:S301

    all_steps = sorted(
        set(chain.from_iterable(values.keys() for values in results.values()))
    )

    for step in tqdm(all_steps):
        visualize(results, step)


if __name__ == "__main__":
    main()
