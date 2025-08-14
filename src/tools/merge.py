import argparse
import os
import pickle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    total = {}
    for file in args.input:
        with open(file, "rb") as f:
            data = pickle.load(f)  # noqa:S301
        total = total | data

    with open(args.output, "wb") as f:
        pickle.dump(total, f)

    for file in args.input:
        os.remove(file)


if __name__ == "__main__":
    main()
