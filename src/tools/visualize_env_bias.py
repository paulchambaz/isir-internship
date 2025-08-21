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

COLORS = {
    # tafu
    ("tafu", "0.2"): "#be6db7",
    ("tafu", "0.4"): "#ce7cc8",
    ("tafu", "0.6"): "#df8cd8",
    ("tafu", "0.8"): "#f09be8",
    # afu
    ("afu", "0.2"): "#a178d5",
    ("afu", "0.4"): "#b087e6",
    ("afu", "0.6"): "#c097f6",
    ("afu", "0.8"): "#d0a7ff",
    # msac
    ("msac", "1"): "#5591e1",
    ("msac", "3"): "#61a0f4",
    ("msac", "5"): "#70b0ff",
    ("msac", "10"): "#87c0ff",
    # ndtop
    ("ndtop", "-1.0"): "#a19100",
    ("ndtop", "-0.5"): "#b0a11b",
    ("ndtop", "0.0"): "#c0b034",
    ("ndtop", "0.5"): "#d0c047",
    # sac
    ("sac", "2"): "#d76868",
    ("sac", "3"): "#e87876",
    ("sac", "5"): "#fa8785",
    ("sac", "8"): "#ff9895",
    # top
    ("top", "-1.0"): "#c87b13",
    ("top", "-0.5"): "#d88a2c",
    ("top", "0.0"): "#e99a3f",
    ("top", "0.5"): "#faaa51",
    # ttqc
    ("ttqc", "1"): "#00a97f",
    ("ttqc", "2"): "#0db98e",
    ("ttqc", "3"): "#32ca9d",
    ("ttqc", "5"): "#49daac",
    # tqc
    ("tqc", "1"): "#6aa142",
    ("tqc", "2"): "#79b152",
    ("tqc", "3"): "#88c162",
    ("tqc", "5"): "#98d171",
}


def display_graph(results: dict, visu_method: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))

    for (method, n), value in sorted(results.items(), key=lambda x: (x[0][0], float(x[0][1]))):
        if not (
            (method == "msac" and n == "1")
            or (method == "sac" and n == "2")
            or method == visu_method
        ):
            continue

        steps = value.keys()

        iqms = []
        q1s = []
        q3s = []
        for _, data in value.items():
            true_qs = data["true_qs"]
            estimated_qs = data["estimated_qs"]

            if method in {"afu", "tafu"}:
                estimated_qs = estimated_qs[:, 0]

            errors = estimated_qs - true_qs

            _, q1, iqm, q3, _ = compute_stats(errors)

            q1s.append(q1)
            iqms.append(iqm)
            q3s.append(q3)

        ax.axhline(y=0, color="black", linewidth=2, zorder=1)
        ax.fill_between(
            steps,
            q3s,
            q1s,
            color=COLORS[(method, n)],
            alpha=0.2,
            zorder=2,
        )
        ax.plot(
            steps,
            iqms,
            color=COLORS[(method, n)],
            label=f"{method} n={n}",
            linewidth=3,
            zorder=4,
        )

    plt.gca().xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{int(x / 1000)}k")
    )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Bias")

    ax.grid(visible=True, alpha=0.25)
    ax.legend()

    plt.tight_layout()
    directory = "paper/figures"
    Path(directory).mkdir(parents=True, exist_ok=True)
    plt.savefig(
        f"{directory}/env_bias_{visu_method}.png", bbox_inches="tight", dpi=200
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
