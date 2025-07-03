import numpy as np

# from afu_rljax.algorithm import AFU
from .afu2 import AFU
from .rl_algo import RLAlgo


class AFUP(RLAlgo):
    __slots__ = [
        "algo",
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
    ) -> None:
        self.algo = AFU(
            state_dim=state_dim,
            action_dim=action_dim,
            units_actor=hidden_dims,
            units_critic=hidden_dims,
            buffer_size=replay_size,
            batch_size=batch_size,
            lr_actor=policy_lr,
            lr_critic=critic_lr,
            lr_alpha=temperature_lr,
            tau=tau,
            rho=rho,
            gamma=gamma,
            seed=42,
        )

    def select_action(self, state: np.ndarray, evaluation: bool) -> np.ndarray:
        return self.algo.select_action(state, evaluation)

    def push_buffer(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.algo.buffer.push(state, action, reward, next_state, done)

    def update(self) -> None:
        self.algo.update()

    def get_state(self) -> dict:
        return {}

    def load_from_state(self, state: dict) -> None:
        pass
