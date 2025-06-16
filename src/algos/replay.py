# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship.
#
# isir-internship is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    __slots__ = ["buffer"]

    def __init__(self, max_size: int) -> None:
        self.buffer = deque(maxlen=max_size)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        transitions = random.sample(self.buffer, batch_size)
        return tuple(
            torch.as_tensor(batch) for batch in zip(*transitions, strict=True)
        )

    def __len__(self) -> int:
        return len(self.buffer)
