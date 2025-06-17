# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship.
#
# isir-internship is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import numpy as np


def compute_stats(results: list) -> tuple[float, float, float, float, float]:
    data = np.array(results)
    min_val = np.min(data)
    max_val = np.max(data)
    q1, q3 = np.percentile(data, [25, 75])
    mask = (data >= q1) & (data <= q3)
    iqm = np.mean(data[mask])
    return float(min_val), float(q1), float(iqm), float(q3), float(max_val)
