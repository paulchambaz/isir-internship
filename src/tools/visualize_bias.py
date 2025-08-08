import argparse
import pickle
from itertools import chain
from pathlib import Path

import matplotlib.pyplot as plt

COLORS = ["#6ca247", "#d66b6a", "#5591e1", "#39a985", "#ad75ca", "#c77c1e"]


def display_graph(results: dict) -> None:
    if "metadata" in results:
        _ = results["metadata"]
        del results["metadata"]

    all_steps = sorted(
        set(
            chain.from_iterable(
                data["results"].keys() for _, data in results.items()
            )
        )
    )

    plt.figure(figsize=(12, 8))

    for i, ((method, n), data) in enumerate(results.items()):
        filtered_results = [
            (step, data["results"][step][0]) for step in all_steps
        ]
        steps, biases = map(list, zip(*filtered_results, strict=True))

        plt.plot(
            steps, biases, color=COLORS[i], label=f"{method} n={n}", linewidth=2
        )
        plt.axhline(y=0, color="black", linewidth=2)

    plt.xlabel("Training step")
    plt.ylabel("Bias")
    plt.grid(visible=True, alpha=0.25)
    plt.legend(loc="lower right")
    # plt.ylim(-10, 10)
    plt.tight_layout()

    plt.show()
    # directory = "paper/figures"
    # Path(directory).mkdir(parents=True, exist_ok=True)
    # plt.savefig(f"{directory}/bias_time.png", bbox_inches="tight", dpi=100)


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

    display_graph(results)


if __name__ == "__main__":
    main()
