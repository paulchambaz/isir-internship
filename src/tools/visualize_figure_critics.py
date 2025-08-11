import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


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

    mean_true_q = np.mean(true_q)
    mean_predicted_q = np.mean(predicted_q)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

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

    ax2.set_ylim(-50, 50)
    ax2.set_xlabel("Action")
    ax2.set_ylabel("Error")
    ax2.grid(visible=True, alpha=0.25)

    ax3.plot(eval_actions, true_q, color="black", linewidth=4)
    ax3.plot(
        eval_actions,
        predicted_q - mean_predicted_q + mean_true_q,
        color="#5591e1",
        linewidth=2,
    )
    ax3.scatter(actions, actions_q_values, color="#d66b6a", s=20, zorder=5)
    ax3.scatter([optim_action], [max_q], color="#d66b6a", s=100, zorder=5)
    ax3.set_xlabel("Action")
    ax3.set_ylabel(r"$Q - \text{Mean} [Q_\theta] + \text{Mean} [Q^*]$")
    ax3.grid(visible=True, alpha=0.25)

    plt.suptitle(rf"{method} - $n={n}$ - Step {step}", fontsize=24)
    plt.tight_layout()

    plt.show()
    # directory = f"paper/figures/figure_critics/{method}_{n}"
    # Path(directory).mkdir(parents=True, exist_ok=True)
    # plt.savefig(f"{directory}/{step:05d}.png", bbox_inches="tight", dpi=100)
    # plt.close()


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
    for i in tqdm(range(100, 25001, 100)):
        if i != 25000:
            continue

        create_q_visualization(results, "avg", 1, i)
        # create_q_visualization(results, "avg", 3, i)
        # create_q_visualization(results, "avg", 5, i)
        # create_q_visualization(results, "avg", 10, i)
        # create_q_visualization(results, "avg", 20, i)
        # create_q_visualization(results, "avg", 50, i)
        #
        # create_q_visualization(results, "min", 2, i)
        # create_q_visualization(results, "min", 3, i)
        # create_q_visualization(results, "min", 4, i)
        # create_q_visualization(results, "min", 6, i)
        # create_q_visualization(results, "min", 8, i)
        # create_q_visualization(results, "min", 10, i)
        #
        # create_q_visualization(results, "ttqc", 1, i)
        # create_q_visualization(results, "ttqc", 2, i)
        # create_q_visualization(results, "ttqc", 3, i)
        # create_q_visualization(results, "ttqc", 4, i)
        # create_q_visualization(results, "ttqc", 6, i)
        # create_q_visualization(results, "ttqc", 10, i)
        # create_q_visualization(results, "ttqc", 14, i)
        #
        # create_q_visualization(results, "tqc", 1, i)
        # create_q_visualization(results, "tqc", 2, i)
        # create_q_visualization(results, "tqc", 3, i)
        # create_q_visualization(results, "tqc", 4, i)
        # create_q_visualization(results, "tqc", 6, i)
        # create_q_visualization(results, "tqc", 10, i)
        # create_q_visualization(results, "tqc", 14, i)
        # create_q_visualization(results, "tqc", 14, i)
        #
        # create_q_visualization(results, "ndtop", -1.0, i)
        # create_q_visualization(results, "ndtop", -0.7, i)
        # create_q_visualization(results, "ndtop", -0.5, i)
        # create_q_visualization(results, "ndtop", 0.0, i)
        # create_q_visualization(results, "ndtop", 0.5, i)
        # create_q_visualization(results, "ndtop", 1.0, i)
        #
        # create_q_visualization(results, "top", -1.0, i)
        # create_q_visualization(results, "top", -0.7, i)
        # create_q_visualization(results, "top", -0.5, i)
        # create_q_visualization(results, "top", 0.0, i)
        # create_q_visualization(results, "top", 0.5, i)
        # create_q_visualization(results, "top", 1.0, i)
        #
        # create_q_visualization(results, "afu", 0.05, i)
        # create_q_visualization(results, "afu", 0.1, i)
        # create_q_visualization(results, "afu", 0.3, i)
        # create_q_visualization(results, "afu", 0.5, i)
        create_q_visualization(results, "afu", 0.7, i)
        # create_q_visualization(results, "afu", 0.9, i)
        # create_q_visualization(results, "afu", 0.95, i)


if __name__ == "__main__":
    main()
