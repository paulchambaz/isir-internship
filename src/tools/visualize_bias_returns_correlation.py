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
from scipy.optimize import curve_fit

from .utils import compute_stats

COLORS = {
    # avg
    ("avg", 1): "#000a4e",
    ("avg", 3): "#00347c",
    ("avg", 5): "#205da8",
    ("avg", 10): "#4b87d6",
    ("avg", 20): "#75b3ff",
    ("avg", 50): "#c0e1ff",
    # min
    ("min", 2): "#370001",
    ("min", 3): "#6f0013",
    ("min", 4): "#9d3438",
    ("min", 6): "#cd5f5f",
    ("min", 8): "#fe8a88",
    ("min", 10): "#ff9895",
    # tqc
    ("tqc", 0): "#000500",
    ("tqc", 1): "#021800",
    ("tqc", 2): "#0b2d00",
    ("tqc", 3): "#184400",
    ("tqc", 4): "#285c00",
    ("tqc", 5): "#3f7307",
    ("tqc", 6): "#568b2b",
    ("tqc", 7): "#6da445",
    ("tqc", 10): "#85be5f",
    ("tqc", 13): "#9ed878",
    ("tqc", 16): "#b7f291",
}

SYMBOLS = {
    "avg": "N",
    "min": "N",
    "tqc": "d",
}

TITLES = {
    "avg": r"AVG ($N$ Q-networks)",
    "min": r"MIN ($N$ Q-networks)",
    "tqc": r"TQC $N=2$ $M=25$ ($d$ dropped quantiles)",
}


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

    fig, ax = plt.subplots(figsize=(12, 8))

    methods_to_plot = ["tqc"]
    all_bias = []
    all_returns = []

    rng = np.random.default_rng(seed=42)

    for (method, n), step_data in results.items():
        if method not in methods_to_plot:
            continue

        step_bias = []
        step_returns = []

        for _, data in step_data.items():
            predicted_qs = np.array(data["predicted_qs"])
            policy_qs = np.array(data["policy_qs"])
            errors = predicted_qs - policy_qs
            _, _, bias_iqm, _, _ = compute_stats(errors)

            policy_action = np.array(data["policy_action"])
            true_qs = np.array(data["true_qs"])
            actions = np.array(data["actions"])
            closest_indices = np.argmin(
                np.abs(actions - policy_action[:, None]), axis=1
            )
            returns = true_qs[np.arange(len(policy_action)), closest_indices]
            _, _, returns_iqm, _, _ = compute_stats(returns)

            step_bias.append(bias_iqm)
            step_returns.append(returns_iqm)

        ax.scatter(
            step_returns,
            step_bias,
            c=COLORS[(method, n)],
            alpha=0.7,
            s=50,
            zorder=rng.random(),
        )

        all_bias.extend(step_bias)
        all_returns.extend(step_returns)

    all_bias = np.array(all_bias)
    all_returns = np.array(all_returns)

    def linear_func(x: float, a: float, b: float) -> float:
        return a * x + b

    params, _ = curve_fit(linear_func, all_returns, all_bias)
    a, b = params
    fit_line = linear_func(all_returns, a, b)

    ss_tot = np.sum((all_bias - np.mean(all_bias)) ** 2)
    ss_res = np.sum((all_bias - fit_line) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    sorted_indices = np.argsort(all_returns)
    ax.plot(
        all_returns[sorted_indices],
        fit_line[sorted_indices],
        "k-",
        linewidth=2,
    )

    ax.set_xlabel("Returns")
    ax.set_ylabel("Bias")
    ax.set_title(
        r"TQC $N=2$ $M=25$: Returns vs Bias Correlation", fontsize=24, pad=20
    )
    ax.grid(visible=True, alpha=0.25)

    ax.set_ylim(-1e2, 1e2)

    ax.annotate(
        rf"$R^2 = {r_squared:.4f}$",
        xy=(0.85, 0.95),
        xycoords="axes fraction",
    )

    plt.tight_layout()
    plt.savefig(
        "paper/figures/bias_returns_correlation_combined.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
