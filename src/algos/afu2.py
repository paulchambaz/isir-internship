# afu2.py

import math
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax as optix
from haiku import PRNGSequence
from jax import nn


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        state_dim: int,
        action_dim: int,
    ) -> None:
        self._n = 0
        self._p = 0
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
        self.state[self._p] = state
        self.action[self._p] = action
        self.reward[self._p] = float(reward)
        self.next_state[self._p] = next_state
        self.done[self._p] = float(done)

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def sample(
        self, batch_size: int, key: jnp.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idxes = jax.random.randint(
            key=key, shape=(batch_size,), minval=0, maxval=self._n
        )

        return (
            self.state[idxes],
            self.action[idxes],
            self.reward[idxes],
            self.next_state[idxes],
            self.done[idxes],
        )


class AFU:
    __slots__ = [
        "_update_target",
        "action_dim",
        "batch_size",
        "buffer",
        "gamma",
        "log_alpha",
        "policy_network",
        "policy_optim",
        "policy_optim_state",
        "policy_params",
        "q_network",
        "q_optim",
        "q_optim_state",
        "q_params",
        "rho",
        "rng",
        "state_dim",
        "target_entropy",
        "tau",
        "temperature_optim",
        "temperature_optim_state",
        "v_network",
        "v_optim",
        "v_optim_state",
        "v_params",
        "v_target_params",
    ]

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        batch_size: int,
        tau: float,
        lr_actor: float,
        lr_critic: float,
        lr_alpha: float,
        units_actor: list[int],
        units_critic: list[int],
        rho: float,
        gamma: float,
        seed: int,
    ) -> None:
        self.rng = PRNGSequence(seed)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.q_network = hk.without_apply_rng(
            hk.transform(
                lambda s, a: self.QNetwork(
                    num_critics=3, hidden_units=units_critic
                )(s, a)
            )
        )
        self.q_params = self.q_network.init(
            next(self.rng),
            np.zeros((1, state_dim), dtype=np.float32),
            np.zeros((1, action_dim), dtype=np.float32),
        )

        self.v_network = hk.without_apply_rng(
            hk.transform(
                lambda s: self.VNetwork(
                    num_critics=2, hidden_units=units_critic
                )(s)
            )
        )
        self.v_params = self.v_network.init(
            next(self.rng), np.zeros((1, state_dim), dtype=np.float32)
        )
        self.v_target_params = self.v_params

        self.policy_network = hk.without_apply_rng(
            hk.transform(
                lambda s: self.PolicyNetwork(
                    action_dim=action_dim, hidden_units=units_actor
                )(s)
            )
        )
        self.policy_params = self.policy_network.init(
            next(self.rng), np.zeros((1, state_dim), dtype=np.float32)
        )

        self.log_alpha = jnp.array(np.log(1.0), dtype=jnp.float32)

        q_optim_init, self.q_optim = optix.adam(lr_critic)
        self.q_optim_state = q_optim_init(self.q_params)

        v_optim_init, self.v_optim = optix.adam(lr_critic)
        self.v_optim_state = v_optim_init(self.v_params)

        policy_optim_init, self.policy_optim = optix.adam(lr_actor)
        self.policy_optim_state = policy_optim_init(self.policy_params)

        temperature_optim_init, self.temperature_optim = optix.adam(lr_alpha)
        self.temperature_optim_state = temperature_optim_init(self.log_alpha)

        self.buffer = ReplayBuffer(
            buffer_size=buffer_size, state_dim=state_dim, action_dim=action_dim
        )
        self.batch_size = batch_size

        self.target_entropy = -float(action_dim)
        self.tau = tau
        self.rho = rho
        self.gamma = gamma

        self._update_target = jax.jit(partial(self._soft_update, tau=tau))

    def select_action(self, state: np.ndarray, evaluation: bool) -> np.ndarray:
        jax_state = state[None, ...]
        jax_action = (
            self._select_action(self.policy_params, jax_state)
            if evaluation
            else self._explore(self.policy_params, jax_state, next(self.rng))
        )
        return np.array(jax_action[0])

    @partial(jax.jit, static_argnums=0)
    def _select_action(
        self,
        policy_params: hk.Params,
        state: jnp.ndarray,
    ) -> jnp.ndarray:
        mean, _ = self.policy_network.apply(policy_params, state)
        return jnp.tanh(mean)

    @partial(jax.jit, static_argnums=0)
    def _explore(
        self,
        policy_params: hk.Params,
        state: jnp.ndarray,
        key: jnp.ndarray,
    ) -> jnp.ndarray:
        mean, log_std = self.policy_network.apply(policy_params, state)
        action, _ = self._reparameterize_gaussian_and_tanh(
            mean=mean, log_std=log_std, key=key
        )
        return action

    def update(self) -> None:
        state, action, reward, next_state, done = self.buffer.sample(
            self.batch_size, next(self.rng)
        )

        # Update critic and value networks together
        loss_critic, grad = jax.value_and_grad(
            self._compute_critic_loss, argnums=(0, 1)
        )(
            self.q_params,
            self.v_params,
            v_target_params=self.v_target_params,
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            rho=self.rho,
        )

        update, self.q_optim_state = self.q_optim(grad[0], self.q_optim_state)
        self.q_params = optix.apply_updates(self.q_params, update)
        update, self.v_optim_state = self.v_optim(grad[1], self.v_optim_state)
        self.v_params = optix.apply_updates(self.v_params, update)

        # Update actor
        (loss_actor, mean_log_pi), grad = jax.value_and_grad(
            self._compute_policy_loss, has_aux=True
        )(
            self.policy_params,
            q_params=self.q_params,
            v_params=self.v_params,
            log_alpha=self.log_alpha,
            state=state,
            key=next(self.rng),
        )

        update, self.policy_optim_state = self.policy_optim(
            grad, self.policy_optim_state
        )
        self.policy_params = optix.apply_updates(self.policy_params, update)

        # Update alpha
        loss_alpha, grad = jax.value_and_grad(self._compute_temperature_loss)(
            self.log_alpha,
            mean_log_pi=mean_log_pi,
        )
        update, self.temperature_optim_state = self.temperature_optim(
            grad, self.temperature_optim_state
        )
        self.log_alpha = optix.apply_updates(self.log_alpha, update)

        # Update target network.
        self.v_target_params = self._update_target(
            self.v_target_params, self.v_params
        )

    @partial(jax.jit, static_argnums=0)
    def _compute_critic_loss(
        self,
        q_params: hk.Params,
        v_params: hk.Params,
        v_target_params: hk.Params,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        rho: float,
    ) -> jnp.ndarray:
        next_v_values = self.v_network.apply(v_target_params, next_state)
        v_targets = jnp.min(jnp.asarray(next_v_values), axis=0)

        q_targets = jax.lax.stop_gradient(
            reward + self.gamma * (1.0 - done) * v_targets
        )

        v_values = jnp.asarray(self.v_network.apply(v_params, state))

        q_values_list = self.q_network.apply(q_params, state, action)
        q_values = jnp.asarray(q_values_list[-1:])

        abs_td = jnp.abs(q_targets - q_values)
        q_loss = (abs_td**2).mean()

        a_values = -jnp.asarray(q_values_list[:-1])

        mix_case = jax.lax.stop_gradient(v_values + a_values < q_targets)
        upsilon_values = (
            1 - rho * mix_case
        ) * v_values + rho * mix_case * jax.lax.stop_gradient(v_values)

        up_case = jax.lax.stop_gradient(v_values >= q_targets)
        z_values = (
            a_values**2
            + up_case * 2 * a_values * (upsilon_values - q_targets)
            + (upsilon_values - q_targets) ** 2
        )

        va_loss = z_values.mean()

        return va_loss + q_loss

    @partial(jax.jit, static_argnums=0)
    def _compute_policy_loss(
        self,
        policy_params: hk.Params,
        q_params: hk.Params,
        v_params: hk.Params,
        log_alpha: jnp.ndarray,
        state: np.ndarray,
        key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        mean, log_std = self.policy_network.apply(policy_params, state)

        action, log_pi = self._reparameterize_gaussian_and_tanh(
            mean=mean, log_std=log_std, key=key
        )
        q_list = self.q_network.apply(q_params, state, action)

        mean_log_pi = log_pi.mean()

        mean_q = q_list[-1].mean()
        loss = jax.lax.stop_gradient(jnp.exp(log_alpha)) * mean_log_pi - mean_q

        return loss, jax.lax.stop_gradient(mean_log_pi)

    @partial(jax.jit, static_argnums=0)
    def _compute_temperature_loss(
        self,
        log_alpha: jnp.ndarray,
        mean_log_pi: jnp.ndarray,
    ) -> jnp.ndarray:
        return -log_alpha * (self.target_entropy + mean_log_pi)

    @partial(jax.jit, static_argnums=0)
    def _soft_update(
        self,
        target_params: hk.Params,
        online_params: hk.Params,
        tau: float,
    ) -> hk.Params:
        return jax.tree.map(
            lambda t, s: (1 - tau) * t + tau * s, target_params, online_params
        )

    @partial(jax.jit, static_argnums=0)
    def _reparameterize_gaussian_and_tanh(
        self,
        mean: jnp.ndarray,
        log_std: jnp.ndarray,
        key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        std = jnp.exp(log_std)
        noise = jax.random.normal(key, std.shape)
        action = jnp.tanh(mean + noise * std)

        gaussian_log_prob = -0.5 * (
            jnp.square(noise) + 2 * log_std + jnp.log(2 * math.pi)
        )
        tanh_correction = jnp.log(nn.relu(1.0 - jnp.square(action)) + 1e-6)

        log_pi = (gaussian_log_prob - tanh_correction).sum(
            axis=1, keepdims=True
        )

        return action, log_pi

    class QNetwork(hk.Module):
        def __init__(self, num_critics: int, hidden_units: list[int]) -> None:
            super().__init__()
            self.num_critics = num_critics
            self.hidden_units = hidden_units

        def __call__(
            self, state: jnp.ndarray, action: jnp.ndarray
        ) -> list[jnp.ndarray]:
            def _fn(x: jnp.ndarray) -> jnp.ndarray:
                for hidden_dim in self.hidden_units:
                    x = nn.relu(
                        hk.Linear(
                            hidden_dim,
                            w_init=hk.initializers.Orthogonal(scale=np.sqrt(2)),
                        )(x)
                    )

                return hk.Linear(1, w_init=hk.initializers.Orthogonal())(x)

            state_action = jnp.concatenate([state, action], axis=1)
            return [_fn(state_action) for _ in range(self.num_critics)]

    class VNetwork(hk.Module):
        def __init__(self, num_critics: int, hidden_units: list[int]) -> None:
            super().__init__()
            self.num_critics = num_critics
            self.hidden_units = hidden_units

        def __call__(self, state: jnp.ndarray) -> list[jnp.ndarray]:
            def _fn(x: jnp.ndarray) -> jnp.ndarray:
                for hidden_dim in self.hidden_units:
                    x = nn.relu(
                        hk.Linear(
                            hidden_dim,
                            w_init=hk.initializers.Orthogonal(scale=np.sqrt(2)),
                        )(x)
                    )
                return hk.Linear(1, w_init=hk.initializers.Orthogonal())(x)

            return [_fn(state) for _ in range(self.num_critics)]

    class PolicyNetwork(hk.Module):
        def __init__(self, action_dim: int, hidden_units: list[int]) -> None:
            super().__init__()
            self.action_dim = action_dim
            self.hidden_units = hidden_units

        def __call__(
            self, state: jnp.ndarray
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            def _fn(x: jnp.ndarray) -> jnp.ndarray:
                for hidden_dim in self.hidden_units:
                    x = nn.relu(
                        hk.Linear(
                            hidden_dim,
                            w_init=hk.initializers.Orthogonal(scale=np.sqrt(2)),
                        )(x)
                    )
                return hk.Linear(
                    2 * self.action_dim,
                    w_init=hk.initializers.Orthogonal(),
                )(x)

            output = _fn(state)

            mean, log_std = jnp.split(output, 2, axis=1)
            log_std = jnp.clip(log_std, -10.0, 2.0)
            return mean, log_std
