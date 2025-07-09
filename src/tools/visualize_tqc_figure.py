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
from matplotlib.lines import Line2D


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()

    with open(args.file, "rb") as f:
        results = pickle.load(f)  # noqa: S301

    plt.rcParams["font.size"] = 20
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["text.latex.preamble"] = (
        r"\usepackage{amsmath}\usepackage{amssymb}"
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("log")

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
        "color": "#6ca247",
    }
    min_points = {
        "x": [],
        "y": [],
        "sizes": [],
        "numbers": [],
        "color": "#d66b6a",
    }
    avg_points = {
        "x": [],
        "y": [],
        "sizes": [],
        "numbers": [],
        "color": "#5591e1",
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
            markerfacecolor="#6ca247",
            markersize=18,
            alpha=0.7,
            label="TQC N=2 M=25 (d dropped quantiles)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#d66b6a",
            markersize=18,
            alpha=0.7,
            label="MIN (N Q-networks)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#5591e1",
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

    plt.draw()

    plt.tight_layout()
    plt.show()
    # plt.savefig(
    #     "paper/figures/tqc_figure.svg",
    #     bbox_inches="tight",
    #     dpi=300,
    # )


if __name__ == "__main__":
    main()
