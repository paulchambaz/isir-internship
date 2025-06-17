# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship.
#
# isir-internship is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import pickle
from pathlib import Path

import matplotlib.pyplot as plt

from .utils import compute_stats


def main() -> None:
    plt.rcParams["font.size"] = 20
    plt.rcParams["text.usetex"] = True

    with open("outputs/history.pk", "rb") as f:
        history = pickle.load(f)  # noqa: S301

    steps = [k * 512 for k in history]
    stats = [compute_stats(results) for results in history.values()]

    mins, q1s, iqms, q3s, maxs = zip(*stats, strict=True)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(steps, iqms, linewidth=2, color="#3498db", label="IQM")

    ax.fill_between(steps, q1s, q3s, alpha=0.3, color="#3498db")

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Episode Return")
    ax.grid(visible=True, alpha=0.1)

    Path("paper/figures").mkdir(exist_ok=True)
    plt.savefig(
        "paper/figures/training_curve.svg",
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
    )


if __name__ == "__main__":
    main()
