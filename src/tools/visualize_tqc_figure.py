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
from tqdm import tqdm


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

    all_steps = set()
    for step_data in results.values():
        all_steps.update(step_data.keys())
    all_steps = sorted(all_steps)

    # print(all_steps)

    for step in tqdm(all_steps):
        if step != 0000:
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

            ax.set_title(f"Training step: {step}", fontsize=24, pad=20)

            ax.set_ylim(1e-3, 1e5)
            ax.set_xlim(-1e3, 1e5)

            points = {
                "avg": {"color": "#5591e1"},
                "min": {"color": "#d66b6a"},
                "tqc": {"color": "#6ca247"},
                "ttqc": {"color": "#39a985"},
                "ndtop": {"color": "#a19101"},
                "top": {"color": "#c77c1e"},
            }

            for (method, n), step_data in results.items():
                bias, variance, _ = step_data[step]

                # if method == "avg" and n != 1:
                #     continue

                points[method].setdefault("x", []).append(bias)
                points[method].setdefault("y", []).append(variance)
                points[method].setdefault("numbers", []).append(str(n))

            for key, data in points.items():
                if key not in ["avg", "min"]:
                    continue

                if "x" not in data:
                    continue

                ax.scatter(
                    data["x"],
                    data["y"],
                    c=data["color"],
                    s=300,
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
                # Line2D(
                #     [0],
                #     [0],
                #     marker="o",
                #     color="w",
                #     markerfacecolor="#c77c1e",
                #     markersize=18,
                #     alpha=0.7,
                #     label=r"TOP N=2 M=25 ($\beta$ coefficient)",
                # ),
                # Line2D(
                #     [0],
                #     [0],
                #     marker="o",
                #     color="w",
                #     markerfacecolor="#a19101",
                #     markersize=18,
                #     alpha=0.7,
                #     label=r"ND-TOP N=2 ($\beta$ coefficient)",
                # ),
                # Line2D(
                #     [0],
                #     [0],
                #     marker="o",
                #     color="w",
                #     markerfacecolor="#39a985",
                #     markersize=18,
                #     alpha=0.7,
                #     label="TQC N=1 M=25 (d dropped quantiles)",
                # ),
                # Line2D(
                #     [0],
                #     [0],
                #     marker="o",
                #     color="w",
                #     markerfacecolor="#6ca247",
                #     markersize=18,
                #     alpha=0.7,
                #     label="TQC N=2 M=25 (d dropped quantiles)",
                # ),
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

            plt.show()
            # plt.savefig(
            #     f"paper/figures/tqc_figure/step_{step:05d}.svg",
            #     bbox_inches="tight",
            #     dpi=150,
            # )
            plt.close()


if __name__ == "__main__":
    main()
