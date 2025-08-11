import argparse
import pickle
from itertools import chain
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .utils import align_center

COLORS = {
    # avg
    ("avg", 1): "#5591e1",
    ("avg", 3): "#61a0f4",
    ("avg", 5): "#70b0ff",
    ("avg", 10): "#87c0ff",
    ("avg", 20): "#a4d1ff",
    ("avg", 50): "#c0e1ff",
    # "#dbf1ff",
    # min
    ("min", 2): "#d76868",
    ("min", 3): "#e87876",
    ("min", 4): "#fa8785",
    ("min", 6): "#ff9895",
    ("min", 8): "#ffb2af",
    ("min", 10): "#ffcbc8",
    # "#ffe3e0",
    # tqc
    ("tqc", 1): "#6aa142",
    ("tqc", 2): "#79b152",
    ("tqc", 3): "#88c162",
    ("tqc", 4): "#98d171",
    ("tqc", 6): "#a7e281",
    ("tqc", 10): "#b7f291",
    ("tqc", 14): "#c7ffa1",
    # ttqc
    ("ttqc", 1): "#00a97f",
    ("ttqc", 2): "#0db98e",
    ("ttqc", 3): "#32ca9d",
    ("ttqc", 4): "#49daac",
    ("ttqc", 6): "#5debbc",
    ("ttqc", 10): "#70fccc",
    ("ttqc", 14): "#a2ffe1",
    # ndtop
    ("ndtop", -1.0): "#a19100",
    ("ndtop", -0.7): "#b0a11b",
    ("ndtop", -0.5): "#c0b034",
    ("ndtop", 0.0): "#d0c047",
    ("ndtop", 0.5): "#e0d159",
    ("ndtop", 1.0): "#f1e16b",
    # "#fff27c",
    # top
    ("top", -1.0): "#c87b13",
    ("top", -0.7): "#d88a2c",
    ("top", -0.5): "#e99a3f",
    ("top", 0.0): "#faaa51",
    ("top", 0.5): "#ffba62",
    ("top", 1.0): "#ffd196",
    # "#ffe7c2",
    # afu
    ("afu", 0.05): "#a178d5",
    ("afu", 0.1): "#b087e6",
    ("afu", 0.3): "#c097f6",
    ("afu", 0.5): "#d0a7ff",
    ("afu", 0.7): "#ddbbff",
    ("afu", 0.9): "#e9d1ff",
    ("afu", 0.95): "#f6e6ff",
}


def stats(data: np.ndarray, p1: float, p2: float) -> float:
    q1, q2 = np.quantile(data, [p1, p2])
    selection = data[(data >= q1) & (data <= q2)]
    mean = np.mean(selection) if len(selection) == 0 else np.mean(data)
    return float(mean), q1, q2


def display_graph(results: dict) -> None:
    if "metadata" in results:
        _ = results["metadata"]
        del results["metadata"]

    steps = sorted(
        set(
            chain.from_iterable(
                data["raw_data"].keys() for _, data in results.items()
            )
        )
    )

    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()
    for (method, n), value in results.items():
        metadata = value["metadata"]
        true_q = np.array(metadata["true_q"])
        mean_true_q = np.mean(true_q)

        corrected_true_q = true_q - mean_true_q

        biases = []
        lows = []
        highs = []
        corrected_biases = []
        for step in steps:
            if step not in value["raw_data"]:
                continue

            raw_data = value["raw_data"][step]
            all_predicted_q = np.array(
                [np.array(predicted_q) for _, _, _, predicted_q in raw_data]
            )
            mean_all_predicted_q = np.mean(
                all_predicted_q, axis=1, keepdims=True
            )
            corrected_all_predicted_q = all_predicted_q - mean_all_predicted_q

            all_errors = (all_predicted_q - true_q).reshape(-1)
            bias, low, high = stats(all_errors, 0.25, 0.75)
            biases.append(bias)
            lows.append(low)
            highs.append(high)

            all_corrected_errors = np.abs(
                corrected_all_predicted_q - corrected_true_q
            ).reshape(-1)
            corrected_bias, _, _ = stats(all_corrected_errors, 0.1, 0.9)
            corrected_biases.append(corrected_bias)

        ax1.axhline(y=0, color="black", linewidth=2)
        ax1.plot(
            steps,
            biases,
            color=COLORS[(method, n)],
            label=f"{method} n={n}",
            linewidth=2,
        )
        ax1.fill_between(
            steps, lows, highs, color=COLORS[(method, n)], alpha=0.2
        )
        ax2.plot(
            steps,
            corrected_biases,
            color=COLORS[(method, n)],
            linestyle="--",
            linewidth=2,
        )

    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Bias")
    ax2.set_ylabel("Corrected Bias")

    ax1.grid(visible=True, alpha=0.25)
    ax1.legend(loc="lower right")

    align_center(ax1, ax2)

    plt.tight_layout()

    plt.show()
    # directory = "paper/figures"
    # Path(directory).mkdir(parents=True, exist_ok=True)
    # plt.savefig(f"{directory}/bias_time.png", bbox_inches="tight", dpi=100)
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

    display_graph(results)


if __name__ == "__main__":
    main()
