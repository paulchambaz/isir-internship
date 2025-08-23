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

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .utils import compute_stats

COLORS = {
    "avg": "#5591e1",
    "min": "#d76868",
    "tqc": "#6aa142",
}

TITLES = {
    "avg": r"AVG ($N$ Q-networks)",
    "min": r"MIN ($N$ Q-networks)",
    "tqc": r"TQC $N=2$ $M=25$ ($d$ dropped quantiles)",
}


def visualize(results: dict, current_method: str, n: int, step: int) -> None:
    if (current_method, n) not in results:
        return

    if step not in results[(current_method, n)]:
        return

    data = results[(current_method, n)][step]

    actions = np.array(data["actions"][0])
    predicted_qs = np.mean(np.array(data["predicted_qs"]), axis=0)
    policy_qs = np.mean(np.array(data["policy_qs"]), axis=0)

    mean_predicted = np.mean(predicted_qs)
    mean_policy = np.mean(policy_qs)
    # bias_correction = mean_policy - mean_predicted
    # predicted_qs_corrected = predicted_qs + bias_correction

    sampled_actions = np.linspace(-1, 1, 50)
    sampled_policy_qs = np.interp(sampled_actions, actions, policy_qs)

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(
        actions,
        policy_qs - mean_policy,
        color="black",
        linewidth=4,
        label=r"$Q^{\pi}(s,a) - \mathbb{E}_{a \in [-1, 1]} [ Q^{\pi}(s,a) ]$",
    )
    ax.plot(
        actions,
        predicted_qs - mean_predicted,
        color=COLORS[current_method],
        linewidth=3,
        label=r"$\hat{Q}^{\pi}(s,a) - \mathbb{E}_{a \in [-1, 1]} [ \hat{Q}^{\pi}(s,a) ]$",
    )

    ax.scatter(
        sampled_actions,
        sampled_policy_qs - mean_policy,
        color="gray",
        s=20,
        zorder=5,
    )

    ax.set_xlabel("Action")
    ax.set_ylabel("Q-value")
    ax.set_title(f"{TITLES[current_method]} - Step {step}", fontsize=24, pad=20)
    ax.grid(visible=True, alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.show()


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

    experiments = list(
        chain(
            [("avg", n) for n in [1, 3, 5, 10, 20, 50]],
            [("min", n) for n in [2, 3, 4, 6, 8, 10]],
            [("tqc", n) for n in [0, 1, 2, 3, 4, 5, 6, 7, 10, 13, 16]],
        )
    )

    for step in tqdm(all_steps):
        for method, n in experiments:
            visualize(results, method, n, step)


if __name__ == "__main__":
    main()
