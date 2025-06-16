# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship.
#
# isir-internship is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import numpy as np


class SimpleRight:
    __slots__ = [
        "action_dim",
        "direction",
    ]

    def __init__(
        self,
        action_dim: int,
        direction: float,
    ) -> None:
        self.action_dim = action_dim
        self.direction = direction

    def select_action(
        self, state: np.ndarray, *, evaluation: bool = False
    ) -> np.ndarray:
        return np.full(self.action_dim, self.direction)
