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
    """Abstract base class defining the interface for RL algorithms."""

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def select_action(self, state: np.ndarray, evaluation: bool) -> np.ndarray:
        """
        Selects an action in range [-1, 1] for the given state using either
        deterministic or stochastic sampling. Use this method to get actions
        from your trained agent during environment interaction.

        Args:
            state: Current environment state observation
            evaluation: Whether to use deterministic or stochastic
        """

    @abstractmethod
    def evaluate(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Evaluates the Q-values for a given state-action pair using the current
        critic network(s). Use this method to assess how valuable the critic
        considers a specific action in a given state.

        Args:
            state: Current environment state observation
            action: Action to evaluate in the given state
        """

    @abstractmethod
    def push_buffer(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Stores a transition tuple in the replay buffer for later training. Call
        this method after each environment step to build your training dataset.

        Args:
            state: Current environment state observation
            action: Action taken in the environment
            reward: Reward received from the environment
            next_state: Next state after taking the action
            done: Whether the episode terminated
        """

    @abstractmethod
    def update(self) -> None:
        """
        Performs one training step updating all networks using sampled batch
        from replay buffer. Call this method regularly during training to
        improve the agent's performance.
        """

    @abstractmethod
    def get_state(self) -> dict[str, any]:
        """
        Returns a complete agent state dictionary including all network
        parameters and optimizer states. Use this method to save your trained
        agent for later use or checkpointing.
        """

    @abstractmethod
    def load_from_state(self, state: dict[str, any]) -> None:
        """
        Loads complete agent state from a previously saved state dictionary. Use
        this method to restore a trained agent or continue training from a
        checkpoint.

        Args:
            state: Dictionary containing saved agent parameters and states
        """
