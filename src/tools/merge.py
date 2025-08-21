# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship.
#
# isir-internship is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import argparse
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


if __name__ == "__main__":
    main()
