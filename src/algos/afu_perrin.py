import gymnasium as gym
import numpy as np

from afu_rljax.algorithm import AFU

from .replay import ReplayBuffer


class AFUPerrin:
    __slots__ = [
        "algo",
        "replay_buffer",
    ]

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        hidden_dims: list[int],
        replay_size: int,
        batch_size: int,
        critic_lr: float,
        policy_lr: float,
        temperature_lr: float,
        tau: float,
        rho: float,
        gamma: float,
        alpha: float,
        action_space: gym.Env,
        state_space: gym.Env,
    ) -> None:
        self.algo = AFU(
            action_space=action_space,
            state_space=state_space,
            units_actor=hidden_dims,
            units_critic=hidden_dims,
            buffer_size=replay_size,
            batch_size=batch_size,
            lr_actor=policy_lr,
            lr_critic=critic_lr,
            lr_alpha=temperature_lr,
            tau=tau,
            gradient_reduction=rho,
            gamma=gamma,
            variant="alpha",
            alg="AFU",
            seed=42,
            num_agent_steps=100_000,
        )
        self.replay_buffer = ReplayBuffer(replay_size)

    def select_action(
        self, state: np.ndarray, *, evaluation: bool
    ) -> np.ndarray:
        return (
            self.algo.select_action(state)
            if evaluation
            else self.algo.explore(state)
        )

    def update(self) -> None:
        self.algo.update()

    def get_state(self) -> dict:
        return {}

    def load_from_state(self, state: dict) -> None:
        pass
