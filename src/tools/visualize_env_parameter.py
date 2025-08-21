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

from .utils import compute_stats

COLORS = {
    "msac": "#5591e1",
    "sac": "#d76868",
    "tqc": "#6aa142",
    "ttqc": "#00a97f",
    "top": "#c87b13",
    "ndtop": "#a19100",
    "afu": "#a178d5",
    "tafu": "#be6db7",
}

SYMBOLS = {
    "msac": "N",
    "sac": "N",
    "tqc": "d",
    "ttqc": "d",
    "top": r"\beta",
    "ndtop": r"\beta",
    "afu": r"\rho",
    "tafu": r"\rho",
}


def display_graph(results: dict, visu_method: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))

    params = []
    stats_list = []

    for (method, n), value in sorted(results.items(), key=lambda x: (x[0][0], float(x[0][1]))):
        if method != visu_method:
            continue

        params.append(n)

        total_errors = []
        for _, data in value.items():
            true_qs = data["true_qs"]
            estimated_qs = data["estimated_qs"]

            if method in {"afu", "tafu"}:
                estimated_qs = estimated_qs[:, 0]

            errors = estimated_qs - true_qs
            total_errors.extend(errors)

        stats_list.append(compute_stats(total_errors))

    color = COLORS[visu_method]
    symbol = SYMBOLS[visu_method]
    method = visu_method


    positions = np.arange(len(params))
    iqm_values = []

    ax.axhline(y=0, color="black", linewidth=2, zorder=1)

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
                alpha=0.25,
                zorder=2,
            )
        )

        plt.plot(
            [i - box_width / 2, i + box_width / 2],
            [q1, q1],
            color=color,
            linewidth=4,
                zorder=2,
        )
        plt.plot(
            [i - box_width / 2, i + box_width / 2],
            [q3, q3],
            color=color,
            linewidth=4,
                zorder=2,
        )

        plt.plot(
            [i, i],
            [min_val, q1],
            color=color,
            linewidth=3,
                zorder=2,
        )
        plt.plot(
            [i, i],
            [q3, max_val],
            color=color,
            linewidth=3,
                zorder=2,
        )

        plt.plot(
            [i - whisker_width / 2, i + whisker_width / 2],
            [min_val, min_val],
            color=color,
            linewidth=3,
                zorder=2,
        )
        plt.plot(
            [i - whisker_width / 2, i + whisker_width / 2],
            [max_val, max_val],
            color=color,
            linewidth=3,
                zorder=2,
        )

        plt.plot(
            i,
            iqm,
            "o",
            markersize=8,
            markeredgecolor=color,
            markerfacecolor=color,
                zorder=2,
        )
        plt.plot(
            [i - box_width / 2, i + box_width / 2],
            [iqm, iqm],
            color=color,
            linewidth=2,
                zorder=2,
        )

    plt.plot(
        positions,
        iqm_values,
        color="gray",
        linestyle=":",
        linewidth=3,
        zorder=1000,
        alpha=0.8,
    )

    plt.xticks(positions, [str(param) for param in params])
    plt.xlabel(f"${symbol}$")
    plt.ylabel("Bias")
    plt.title(f"{method.upper()}")
    plt.grid(visible=True, axis="y", alpha=0.3)

    plt.tight_layout()

    directory = "paper/figures"
    Path(directory).mkdir(parents=True, exist_ok=True)
    plt.savefig(
        f"{directory}/env_param_{visu_method}.png", bbox_inches="tight", dpi=200
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

    display_graph(results, "msac")
    display_graph(results, "sac")
    display_graph(results, "tqc")
    display_graph(results, "ttqc")
    display_graph(results, "top")
    display_graph(results, "ndtop")
    display_graph(results, "afu")
    display_graph(results, "tafu")


if __name__ == "__main__":
    main()
