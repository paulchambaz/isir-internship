# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship.
#
# isir-internship is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from abc import ABC, abstractmethod

import numpy as np


class RLAlgo(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def select_action(self, state: np.ndarray, evaluation: bool) -> np.ndarray:
        pass

    @abstractmethod
    def push_buffer(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        pass

    @abstractmethod
    def update(self) -> None:
        pass

    @abstractmethod
    def get_state(self) -> dict:
        pass

    @abstractmethod
    def load_from_state(self, state: dict) -> None:
        pass
