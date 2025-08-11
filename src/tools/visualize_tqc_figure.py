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
from matplotlib.lines import Line2D
from tqdm import tqdm


def robust_mean_var(data: np.ndarray, p1: float, p2: float) -> float:
    q1, q2 = np.quantile(data, [p1, p2])
    selection = data[(data >= q1) & (data <= q2)]
    mean = np.mean(selection) if len(selection) == 0 else np.mean(data)
    var = np.var(selection) if len(selection) == 0 else np.var(data)
    return float(mean), float(var)


def create_tqc_visualization(results: dict, step: int) -> None:
    fig, ax = plt.subplots(figsize=(12, 9))

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

    ax.set_title(f"Training step: {step}", fontsize=24, pad=20)

    ax.set_ylim(1e0, 1e5)
    ax.set_xlim(-1e2, 1e2)

    points = {
        "avg": {"c": "#5591e1"},
        "min": {"c": "#d76868"},
        "tqc": {"c": "#6aa142"},
        "ttqc": {"c": "#00a97f"},
        "ndtop": {"c": "#a19100"},
        "top": {"c": "#c87b13"},
        "afu": {"c": "#a178d5"},
    }

    for (method, n), value in results.items():
        metadata = value["metadata"]
        true_q = np.array(metadata["true_q"])

        if step not in value["raw_data"]:
            continue

        raw_data = value["raw_data"][step]
        all_predicted_q = np.array(
            [np.array(predicted_q) for _, _, _, predicted_q in raw_data]
        )

        all_errors = (all_predicted_q - true_q).reshape(-1)

        error_bias, error_variance = robust_mean_var(all_errors, 0.1, 0.9)

        points[method].setdefault("x", []).append(error_bias)
        points[method].setdefault("y", []).append(error_variance)
        points[method].setdefault("n", []).append(str(n))

    for _, data in points.items():
        if "x" not in data:
            continue

        ax.scatter(data["x"], data["y"], c=data["c"], s=300, alpha=0.7)
        for x, y, n in zip(data["x"], data["y"], data["n"], strict=True):
            ax.text(x, y, n, ha="center", va="center", fontsize=16)

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#a178d5",
            markersize=18,
            alpha=0.7,
            label=r"AFU ($\rho$ coefficient)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#c87b13",
            markersize=18,
            alpha=0.7,
            label=r"TOP N=2 M=25 ($\beta$ coefficient)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#a19100",
            markersize=18,
            alpha=0.7,
            label=r"ND-TOP N=2 ($\beta$ coefficient)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#00a97f",
            markersize=18,
            alpha=0.7,
            label="TQC N=1 M=25 (d dropped quantiles)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#6aa142",
            markersize=18,
            alpha=0.7,
            label="TQC N=2 M=25 (d dropped quantiles)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#d76868",
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
    legend.get_frame().set_alpha(0.7)

    plt.tight_layout()

    # plt.show()
    plt.savefig(
        f"paper/figures/tqc_figure/step_{step:05d}.png",
        bbox_inches="tight",
        dpi=150,
    )
    plt.close()


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

    # create_tqc_visualization(results, 25_000)
    for i in tqdm(range(100, 25001, 100)):
        create_tqc_visualization(results, i)


if __name__ == "__main__":
    main()
