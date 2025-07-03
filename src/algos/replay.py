# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship.
#
# isir-internship is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.


import numpy as np
import torch


class ReplayBuffer:
    __slots__ = (
        "action",
        "buffer_size",
        "done",
        "n",
        "next_state",
        "p",
        "reward",
        "state",
    )

    def __init__(
        self,
        buffer_size: int,
        state_dim: int,
        action_dim: int,
    ) -> None:
        self.buffer_size = buffer_size
        self.n = 0
        self.p = 0

        self.state = np.empty((buffer_size, state_dim), dtype=np.float32)
        self.action = np.empty((buffer_size, action_dim), dtype=np.float32)
        self.reward = np.empty((buffer_size, 1), dtype=np.float32)
        self.next_state = np.empty((buffer_size, state_dim), dtype=np.float32)
        self.done = np.empty((buffer_size, 1), dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.state[self.p] = state
        self.action[self.p] = action
        self.reward[self.p] = float(reward)
        self.next_state[self.p] = next_state
        self.done[self.p] = float(done)

        self.p = (self.p + 1) % self.buffer_size
        self.n = min(self.n + 1, self.buffer_size)

    def sample(
        self, batch_size: int
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        rng = np.random.default_rng()
        idxes = rng.integers(0, self.n, size=batch_size)

        states = torch.from_numpy(self.state[idxes])
        actions = torch.from_numpy(self.action[idxes])
        rewards = torch.from_numpy(self.reward[idxes])
        next_states = torch.from_numpy(self.next_state[idxes])
        dones = torch.from_numpy(self.done[idxes])

        return states, actions, rewards, next_states, dones

    def get_data(
        self,
    ) -> list[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]:
        data = []
        for i in range(self.n):
            data.append(
                (
                    self.state[i].copy(),
                    self.action[i].copy(),
                    float(self.reward[i]),
                    self.next_state[i].copy(),
                    bool(self.done[i]),
                )
            )
        return data

    def load_data(
        self, data: list[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]
    ) -> None:
        n = min(len(data), self.buffer_size)
        for i in range(n):
            state, action, reward, next_state, done = data[i]
            self.state[i] = state
            self.action[i] = action
            self.reward[i] = reward
            self.next_state[i] = next_state
            self.done[i] = float(done)
        self.n = n
        self.p = n % self.buffer_size

    def __len__(self) -> int:
        return self.n
