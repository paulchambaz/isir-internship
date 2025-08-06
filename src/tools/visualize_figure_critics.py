import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .utils import compute_stats


def create_q_visualization(
    results: dict, method: str, n: int | float, step: int
) -> None:
    if (method, n) not in results:
        return

    if step not in results[(method, n)]["raw_data"]:
        return

    actions = np.linspace(-1, 1, 50)

    metadata = results[(method, n)]["metadata"]
    eval_actions = np.array(metadata["eval_actions"])
    true_q = np.array(metadata["true_q"])
    optim_action = metadata["optim_action"]

    raw_data = results[(method, n)]["raw_data"][step]

    all_predicted_q = np.array(
        [np.array(predicted_q_list) for _, _, _, predicted_q_list in raw_data]
    )
    predicted_q = np.mean(all_predicted_q, axis=0)

    error = predicted_q - true_q
    max_q = true_q[np.argmin(np.abs(eval_actions - optim_action))]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
    plt.subplots_adjust(hspace=0.3)

    ax1.plot(
        eval_actions,
        true_q,
        color="black",
        linewidth=4,
        label=r"$Q^* (s, a)$",
    )
    ax1.plot(
        eval_actions,
        predicted_q,
        color="#5591e1",
        linewidth=2,
        label=r"$Q_\theta (s, a)$",
    )

    ax1.scatter(
        [optim_action],
        [max_q],
        color="#d66b6a",
        s=100,
        zorder=5,
        label="Optimal action",
    )

    actions_q_values = np.interp(actions, eval_actions, true_q)
    ax1.scatter(
        actions,
        actions_q_values,
        color="#d66b6a",
        s=20,
        zorder=5,
        label="Actions",
    )
    ax1.set_xlabel("Action")
    ax1.set_ylabel("Q-value")
    ax1.grid(visible=True, alpha=0.25)
    ax1.legend(loc="center left")

    ax2.plot(
        eval_actions,
        error,
        color="#5591e1",
        linewidth=2,
    )
    ax2.axhline(
        y=0,
        color="black",
        linewidth=2,
        label=r"$Q^* (s, a)$",
    )
    ax2.scatter(
        [optim_action],
        [0],
        color="#d66b6a",
        s=100,
        zorder=5,
    )
    ax2.scatter(
        actions,
        np.zeros_like(actions),
        color="#d66b6a",
        s=20,
        zorder=5,
    )

    ax2.set_ylim(-10, 10)
    ax2.set_xlabel("Action")
    ax2.set_ylabel("Error")
    # ax2.set_yscale("symlog", linthresh=0.1)
    ax2.grid(visible=True, alpha=0.25)

    plt.suptitle(f"Step {step}", fontsize=24)
    plt.tight_layout()

    directory = "paper/figures/figure_critics/bare"
    # plt.show()
    Path(directory).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{directory}/{step:05d}.png", bbox_inches="tight", dpi=100)
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

    # create_q_visualization(results, "avg", 1, 2000)
    for i in tqdm(range(100, 20001, 100)):
        create_q_visualization(results, "avg", 1, i)


if __name__ == "__main__":
    main()
