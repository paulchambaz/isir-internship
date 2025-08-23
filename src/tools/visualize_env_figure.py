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
from matplotlib.lines import Line2D
from tqdm import tqdm

COLORS = {
    "tafu": "#be6db7",
    "afu": "#a178d5",
    "msac": "#5591e1",
    "ndtop": "#a19100",
    "sac": "#d76868",
    "top": "#c87b13",
    "ttqc": "#00a97f",
    "tqc": "#6aa142",
}


def display_graph(results: dict, step: int) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))

    for (method, n), value in sorted(
        results.items(), key=lambda x: (x[0][0], float(x[0][1]))
    ):
        data = value[step]
        true_qs = data["true_qs"]
        estimated_qs = data["estimated_qs"]

        if method in {"afu", "tafu"}:
            estimated_qs = estimated_qs[:, 0]

        errors = estimated_qs - true_qs

        bias = errors.mean()
        variance = errors.var()

        ax.scatter([bias], [variance], c=COLORS[method], s=400)
        ax.text(bias, variance, n, ha="center", va="center", fontsize=16)

    ax.set_yscale("log")

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS["msac"],
            markersize=18,
            alpha=0.7,
            label=r"aSAC ($N$ Q-networks)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS["sac"],
            markersize=18,
            alpha=0.7,
            label=r"SAC ($N$ Q-networks)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS["tqc"],
            markersize=18,
            alpha=0.7,
            label=r"TQC $N=2$ $M=25$ ($d$ dropped quantiles)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS["ttqc"],
            markersize=18,
            alpha=0.7,
            label=r"TQC $N=1$ $M=25$ ($d$ dropped quantiles)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS["top"],
            markersize=18,
            alpha=0.7,
            label=r"TOP $N=2$ $M=25$ ($\beta$ coefficient)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS["ndtop"],
            markersize=18,
            alpha=0.7,
            label=r"ND-TOP $N=2$ ($\beta$ coefficient)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS["afu"],
            markersize=18,
            alpha=0.7,
            label=r"AFU $N=2$ ($\rho$ coefficient)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS["tafu"],
            markersize=18,
            alpha=0.7,
            label=r"AFU $N=1$ ($\rho$ coefficient)",
        ),
    ]

    legend = ax.legend(
        handles=legend_elements,
        loc="lower right",
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=16,
    )

    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.7)

    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.7)

    ax.grid(visible=True, which="major", alpha=0.6, linewidth=0.8)

    ax.set_xlabel(
        r"$\mathbb{E}[Q_{\theta} - Q^{\pi}]$",
        fontsize=30,
    )
    ax.set_ylabel(
        r"$\mathbb{V}\text{ar}[Q_{\theta} - Q^{\pi}]$",
        fontsize=30,
    )

    ax.set_ylim(1e-3, 1e7)
    ax.set_xlim(-150, 650)

    ax.set_title(f"Training step: {step}", fontsize=24, pad=20)

    plt.tight_layout()

    directory = "paper/figures/env_figure"
    Path(directory).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{directory}/{step:06d}.png", bbox_inches="tight", dpi=100)
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

    all_steps = sorted(
        set(chain.from_iterable(values.keys() for values in results.values()))
    )

    for step in tqdm(all_steps):
        display_graph(results, step)


if __name__ == "__main__":
    main()
