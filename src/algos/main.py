# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship.
#
# isir-internship is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Main module for algos."""

import time

import gymnasium as gym
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy
import sklearn
import torch


def test_fig() -> None:
    plt.rcParams["font.size"] = 20
    plt.rcParams["text.usetex"] = True

    fig, ax = plt.subplots(figsize=(6, 6))

    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)

    ax.plot(x, y, linewidth=2, color="#3498db")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y = \sin(x)$")
    ax.grid(visible=True, alpha=0.1)

    plt.savefig(
        "paper/figures/sin.svg",
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
    )


def test_gymnasium() -> None:
    envs = {
        "ant": "Ant-v5",
        "swimmer": "Swimmer-v5",
        "walker": "Walker2d-v5",
        "half-cheeta": "HalfCheetah-v5",
        "humanoid": "Humanoid-v5",
    }

    env = gym.make(envs["half-cheeta"], render_mode="human")
    state, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        time.sleep(0.016)

    env.close()


def prints() -> None:
    print("Hello, world!")
    print(plt.__name__)
    print(nx.__name__)
    print(np.__name__)
    print(scipy.__name__)
    print(sklearn.__name__)
    print(torch.__name__)
    print(gym.__name__)


def main() -> None:
    prints()
    test_gymnasium()


if __name__ == "__main__":
    main()
