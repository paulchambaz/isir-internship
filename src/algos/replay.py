# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship.
#
# isir-internship is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.


import jax
import jax.numpy as jnp
import numpy as np


class ReplayBuffer:
    def __init__(
        self, buffer_size: int, state_dim: int, action_dim: int
    ) -> None:
        """
        Circular buffer for storing and sampling experience transitions in RL
        training. Use this to accumulate experience data and sample random
        batches for learning.

        Args:
            buffer_size: Maximum number of transitions to store
            state_dim: Dimensionality of state observations
            action_dim: Dimensionality of action vectors
        """

        self.n = 0
        self.p = 0
        self.buffer_size = buffer_size

        self.state = np.empty((buffer_size, state_dim), dtype=np.float32)
        self.action = np.empty((buffer_size, action_dim), dtype=np.float32)
        self.reward = np.empty((buffer_size, 1), dtype=np.float32)
        self.done = np.empty((buffer_size, 1), dtype=np.float32)
        self.next_state = np.empty((buffer_size, state_dim), dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Stores a single transition in the buffer, overwriting old data when
        full. Call this method after each environment step to accumulate
        training data.
        """

        self.state[self.p] = state
        self.action[self.p] = action
        self.reward[self.p] = float(reward)
        self.next_state[self.p] = next_state
        self.done[self.p] = float(done)

        self.p = (self.p + 1) % self.buffer_size
        self.n = min(self.n + 1, self.buffer_size)

    def sample(
        self, batch_size: int, key: jnp.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns a random batch of transitions for training as a tuple in the
        order: (states, actions, rewards, next_states, dones). Use this method
        to get training batches during learning updates.

        Args:
            batch_size: Number of transitions to sample
            key: JAX random key for sampling
        """

        idxes = jax.random.randint(
            key, shape=(batch_size,), minval=0, maxval=self.n
        )
        return (
            self.state[idxes],
            self.action[idxes],
            self.reward[idxes],
            self.next_state[idxes],
            self.done[idxes],
        )

    def sample_timed_state_action(
        self, batch_size: int, key: jnp.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns a random batch of transitions with their timesteps within episodes.
        Goes backward from each sampled position to find the episode start.

        Args:
            batch_size: Number of transitions to sample
            key: JAX random key for sampling

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, timesteps)
        """
        idxes = jax.random.randint(
            key, shape=(batch_size,), minval=0, maxval=self.n
        )

        timesteps = np.zeros(batch_size, dtype=np.int32)

        for i, idx in enumerate(idxes):
            current_idx = idx
            t = 0

            while True:
                prev_idx = (current_idx - 1) % self.buffer_size

                if prev_idx == self.p and self.n == self.buffer_size:
                    break

                if self.n < self.buffer_size and prev_idx >= self.n:
                    break

                if self.done[prev_idx, 0] == 1.0:
                    break

                current_idx = prev_idx
                t += 1

                if t >= self.buffer_size:
                    break

            timesteps[i] = t

        return (
            self.state[idxes],
            self.action[idxes],
            timesteps,
        )

    def __len__(self) -> int:
        """Returns the current number of stored transitions."""
        return self.n

    def get_data(self) -> dict[str, any]:
        """Returns buffer data dictionary for saving."""
        return {
            "state": self.state[: self.n].copy(),
            "action": self.action[: self.n].copy(),
            "reward": self.reward[: self.n].copy(),
            "next_state": self.next_state[: self.n].copy(),
            "done": self.done[: self.n].copy(),
            "n": self.n,
            "p": self.p,
        }

    def load_data(self, data: dict[str, any]) -> None:
        """Loads buffer data from a saved state dictionary."""
        self.n = data["n"]
        self.p = data["p"]
        self.state[: self.n] = data["state"]
        self.action[: self.n] = data["action"]
        self.reward[: self.n] = data["reward"]
        self.next_state[: self.n] = data["next_state"]
        self.done[: self.n] = data["done"]
