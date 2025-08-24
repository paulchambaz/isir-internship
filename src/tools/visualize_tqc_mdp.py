# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship.
#
# isir-internship is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import matplotlib.pyplot as plt
import numpy as np

from .compute_tqc_figure2 import ToyMdp


def main() -> None:
    mdp = ToyMdp(gamma=0.99, sigma=0.25, a0=0.3, a1=0.9, nu=5.0)
    actions = np.linspace(-1, 1, 2000).reshape(-1, 1)
    rewards = mdp.f(actions)
    q_values = mdp.q_optimal(actions)

    plt.rcParams["font.size"] = 20
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["text.latex.preamble"] = (
        r"\usepackage{amsmath}\usepackage{amssymb}"
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(
        actions,
        rewards,
        color="black",
        linewidth=3,
        label=r"$\mathbb{E} [ R(s_0, a) ] = f(a)$",
    )

    ax.scatter(
        [mdp.optimal_action],
        [mdp.optimal_reward],
        color="#d76868",
        s=100,
        zorder=5,
        label=r"Optimal action",
    )

    ax.set_xlabel("Action")
    ax.set_ylabel("Expectation of rewards")

    ax.grid(visible=True, alpha=0.25)
    ax.legend()

    # plt.show()

    plt.savefig(
        "paper/figures/mdp_reward.png",
        bbox_inches="tight",
        dpi=300,
    )

    plt.close()

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(
        actions,
        q_values,
        color="black",
        linewidth=3,
        label=r"$Q(s_0, a)$",
    )

    ax.scatter(
        [mdp.optimal_action],
        [mdp.q_optimal(mdp.optimal_action)],
        color="#d76868",
        s=100,
        zorder=5,
        label=r"Optimal action",
    )

    ax.set_xlabel("Action")
    ax.set_ylabel("Expectation of rewards")

    ax.grid(visible=True, alpha=0.25)
    ax.legend()

    # plt.show()

    plt.savefig(
        "paper/figures/mdp_q_values.png",
        bbox_inches="tight",
        dpi=300,
    )

    plt.close()


if __name__ == "__main__":
    main()
