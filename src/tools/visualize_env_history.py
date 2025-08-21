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
CONVERT = {
    ("afu n1r0.2"): ("tafu", "0.2"),
    ("afu n1r0.4"): ("tafu", "0.4"),
    ("afu n1r0.6"): ("tafu", "0.6"),
    ("afu n1r0.8"): ("tafu", "0.8"),
    ("afu n2r0.2"): ("afu", "0.2"),
    ("afu n2r0.4"): ("afu", "0.4"),
    ("afu n2r0.6"): ("afu", "0.6"),
    ("afu n2r0.8"): ("afu", "0.8"),
    ("msac n1"): ("msac", "1"),
    ("msac n10"): ("msac", "10"),
    ("msac n3"): ("msac", "3"),
    ("msac n5"): ("msac", "5"),
    ("ndtop n2b-0.5"): ("ndtop", "-0.5"),
    ("ndtop n2b-1.0"): ("ndtop", "-1.0"),
    ("ndtop n2b0.0"): ("ndtop", "0.0"),
    ("ndtop n2b0.5"): ("ndtop", "0.5"),
    ("sac n2"): ("sac", "2"),
    ("sac n3"): ("sac", "3"),
    ("sac n5"): ("sac", "5"),
    ("sac n8"): ("sac", "8"),
    ("top n2m25b-0.5"): ("top", "-0.5"),
    ("top n2m25b-1.0"): ("top", "-1.0"),
    ("top n2m25b0.0"): ("top", "0.0"),
    ("top n2m25b0.5"): ("top", "0.5"),
    ("tqc n1m25d1"): ("ttqc", "1"),
    ("tqc n1m25d2"): ("ttqc", "2"),
    ("tqc n1m25d3"): ("ttqc", "3"),
    ("tqc n1m25d5"): ("ttqc", "5"),
    ("tqc n2m25d1"): ("tqc", "1"),
    ("tqc n2m25d2"): ("tqc", "2"),
    ("tqc n2m25d3"): ("tqc", "3"),
    ("tqc n2m25d5"): ("tqc", "5"),
}


def display_graph(results: dict, visu_method: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))

    for (method, n), value in sorted(
        results.items(), key=lambda x: (x[0][0], float(x[0][1]))
    ):
        if not (
            (method == "msac" and n == "1")
            or (method == "sac" and n == "2")
            or method == visu_method
        ):
            continue

        steps = [k * 500 for k in value]
        stats = [compute_stats(r) for r in value.values()]
        _, q1s, iqms, q3s, _ = zip(*stats, strict=True)

        ax.plot(steps, iqms, linewidth=2)

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
    ax.set_ylabel("Returns")

    ax.grid(visible=True, alpha=0.25)
    ax.legend()

    plt.tight_layout()
    directory = "paper/figures"
    Path(directory).mkdir(parents=True, exist_ok=True)
    plt.savefig(
        f"{directory}/env_returns_{visu_method}.png",
        bbox_inches="tight",
        dpi=200,
    )

    # plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize MountainCar value functions and policies"
    )
    parser.add_argument(
        "--dir", type=str, required=True, help="Path to agent state directory"
    )
    args = parser.parse_args()

    checkpoint_files = sorted(Path(args.dir).glob("*/*/history.pk"))

    results = {}

    for file in checkpoint_files:
        with open(file, "rb") as f:
            data = pickle.load(f)  # noqa: S301

        results[CONVERT[f"{file.parent.parent.name} {file.parent.name}"]] = data

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
