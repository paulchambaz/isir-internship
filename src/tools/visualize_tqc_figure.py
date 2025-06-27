# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship.
#
# isir-internship is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main() -> None:
    plt.rcParams["font.size"] = 20
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["text.latex.preamble"] = (
        r"\usepackage{amsmath}\usepackage{amssymb}"
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("log")
    ax.set_xlim(-2e1, 2e4)
    ax.set_ylim(0.5e-1, 2e1)

    ax.grid(visible=True, which="major", alpha=0.6, linewidth=0.8)

    ax.set_xlabel(
        r"$\mathbb{E}[\Delta(a)]$, $\quad a \sim \mathcal{U}(-1,1)$",
        fontsize=30,
    )
    ax.set_ylabel(
        r"$\mathbb{V}\text{ar}[\Delta(a)]$, $\quad a \sim \mathcal{U}(-1,1)$",
        fontsize=30,
    )

    tqc_data = {
        "x": [1.5, 2.5],
        "y": [0.8, 0.12],
        "sizes": [800, 300],
        "numbers": ["7", "5"],
        "color": "#6ca247",
    }

    min_data = {
        "x": [-1.8, -0.5],
        "y": [0.15, 0.6],
        "sizes": [300, 100],
        "numbers": ["4", "3"],
        "color": "#d66b6a",
    }

    avg_data = {
        "x": [3.5, 4.2],
        "y": [0.9, 0.4],
        "sizes": [350, 120],
        "numbers": ["2", "10"],
        "color": "#5591e1",
    }

    for data in [tqc_data, min_data, avg_data]:
        ax.scatter(
            data["x"], data["y"], c=data["color"], s=data["sizes"], alpha=0.7
        )
        for x, y, num in zip(
            data["x"], data["y"], data["numbers"], strict=True
        ):
            ax.text(x, y, num, ha="center", va="center")

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

    fig.text(
        0.15,
        0.9,
        "d",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="black",
    )
    fig.text(
        0.15,
        0.845,
        "N",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="black",
    )
    fig.text(
        0.15,
        0.79,
        "N",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="black",
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
