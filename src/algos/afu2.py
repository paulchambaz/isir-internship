# afu2.py

import math
from functools import partial
from typing import Any

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax as optix
from gymnasium import Env
from haiku import PRNGSequence
from jax import nn


@jax.jit
def _soft_update(
    target_params: hk.Params,
    online_params: hk.Params,
    tau: float,
) -> hk.Params:
    return jax.tree.map(
        lambda t, s: (1 - tau) * t + tau * s, target_params, online_params
    )


@jax.jit
def _gaussian_log_prob(
    log_std: jnp.ndarray,
    noise: jnp.ndarray,
) -> jnp.ndarray:
    return -0.5 * (jnp.square(noise) + 2 * log_std + jnp.log(2 * math.pi))


@jax.jit
def _gaussian_and_tanh_log_prob(
    log_std: jnp.ndarray,
    noise: jnp.ndarray,
    action: jnp.ndarray,
) -> jnp.ndarray:
    return _gaussian_log_prob(log_std, noise) - jnp.log(
        nn.relu(1.0 - jnp.square(action)) + 1e-6
    )


@partial(jax.jit, static_argnums=3)
def _reparameterize_gaussian_and_tanh(
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
    key: jnp.ndarray,
    return_log_pi: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    std = jnp.exp(log_std)
    noise = jax.random.normal(key, std.shape)
    action = jnp.tanh(mean + noise * std)
    if return_log_pi:
        return action, _gaussian_and_tanh_log_prob(log_std, noise, action).sum(
            axis=1, keepdims=True
        )
    return action


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        state_space: Env,
        action_space: Env,
        gamma: float,
    ):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.state_shape = state_space.shape

        self.state = np.empty(
            (buffer_size, *state_space.shape), dtype=np.float32
        )
        self.next_state = np.empty(
            (buffer_size, *state_space.shape), dtype=np.float32
        )

        self.action = np.empty(
            (buffer_size, *action_space.shape), dtype=np.float32
        )

        self.reward = np.empty((buffer_size, 1), dtype=np.float32)
        self.done = np.empty((buffer_size, 1), dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        next_state: np.ndarray,
    ) -> None:
        self.state[self._p] = state
        self.action[self._p] = action
        self.reward[self._p] = float(reward)
        self.done[self._p] = float(done)
        self.next_state[self._p] = next_state

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)

        return (
            self.state[idxes],
            self.action[idxes],
            self.reward[idxes],
            self.done[idxes],
            self.next_state[idxes],
        )


class MLP(hk.Module):
    def __init__(
        self,
        output_dim,
        hidden_units,
        hidden_activation,
        output_activation,
        hidden_scale,
        output_scale,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.hidden_kwargs = {
            "w_init": hk.initializers.Orthogonal(scale=hidden_scale)
        }
        self.output_kwargs = {
            "w_init": hk.initializers.Orthogonal(scale=output_scale)
        }

    def __call__(self, x):
        for i, unit in enumerate(self.hidden_units):
            x = hk.Linear(unit, **self.hidden_kwargs)(x)
            x = self.hidden_activation(x)
        x = hk.Linear(self.output_dim, **self.output_kwargs)(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x


class QNetwork(hk.Module):
    def __init__(
        self,
        num_critics,
        hidden_units,
    ):
        super().__init__()
        self.num_critics = num_critics
        self.hidden_units = hidden_units

    def __call__(self, s, a):
        def _fn(x):
            return MLP(
                output_dim=1,
                hidden_units=self.hidden_units,
                hidden_activation=nn.relu,
                output_activation=None,
                hidden_scale=np.sqrt(2),
                output_scale=1.0,
            )(x)

        x = jnp.concatenate([s, a], axis=1)
        return [_fn(x) for _ in range(self.num_critics)]


class VNetwork(hk.Module):
    def __init__(
        self,
        num_critics,
        hidden_units,
    ):
        super().__init__()
        self.num_critics = num_critics
        self.hidden_units = hidden_units

    def __call__(self, x):
        def _fn(x):
            return MLP(
                output_dim=1,
                hidden_units=self.hidden_units,
                hidden_activation=nn.relu,
                output_activation=None,
                hidden_scale=np.sqrt(2),
                output_scale=1.0,
            )(x)

        return [_fn(x) for _ in range(self.num_critics)]


class PolicyNetwork(hk.Module):
    def __init__(
        self,
        action_space,
        hidden_units,
        log_std_min,
        log_std_max,
    ):
        super().__init__()
        self.action_space = action_space
        self.hidden_units = hidden_units
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def __call__(self, x):
        x = MLP(
            output_dim=2 * self.action_space.shape[0],
            hidden_units=self.hidden_units,
            hidden_activation=nn.relu,
            output_activation=None,
            hidden_scale=np.sqrt(2),
            output_scale=1.0,
        )(x)
        mean, log_std = jnp.split(x, 2, axis=1)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std


class AFU:
    name = "AFU"

    def __init__(
        self,
        state_space,
        action_space,
        seed,
        gamma,
        buffer_size,
        batch_size,
        tau,
        lr_actor,
        lr_critic,
        lr_alpha,
        units_actor,
        units_critic,
        gradient_reduction,
        *args,
        **kwargs,
    ):
        self.use_key_critic = False
        self.use_key_actor = True

        self.tau = tau

        np.random.seed(seed)
        self.rng = PRNGSequence(seed)

        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma

        self.buffer = ReplayBuffer(
            buffer_size=buffer_size,
            state_space=state_space,
            action_space=action_space,
            gamma=gamma,
        )

        self.discount = gamma
        self.batch_size = batch_size

        self._update_target = jax.jit(partial(_soft_update, tau=tau))

        # Define fake input for critic.
        self.fake_args_critic = (
            self._fake_state(state_space),
            self._fake_action(action_space),
        )

        # Define fake input for actor.
        if not hasattr(self, "fake_args_actor"):
            self.fake_args_actor = (self._fake_state(state_space),)

        self.fake_args_value = (self._fake_state(state_space),)

        self.gradient_reduction = gradient_reduction

        def fn_critic(s, a):
            return QNetwork(
                num_critics=3,
                hidden_units=units_critic,
            )(s, a)

        def fn_value(s):
            return VNetwork(
                num_critics=2,
                hidden_units=units_critic,
            )(s)

        def fn_actor(s):
            return PolicyNetwork(
                action_space=action_space,
                hidden_units=units_actor,
                log_std_min=-10.0,
                log_std_max=2.0,
            )(s)

        # Critic.
        self.critic = hk.without_apply_rng(hk.transform(fn_critic))
        self.params_critic = self.params_critic_target = self.critic.init(
            next(self.rng), *self.fake_args_critic
        )
        opt_init, self.opt_critic = optix.adam(lr_critic)
        self.opt_state_critic = opt_init(self.params_critic)
        # Value.
        self.value = hk.without_apply_rng(hk.transform(fn_value))
        self.params_value = self.params_value_target = self.value.init(
            next(self.rng), *self.fake_args_value
        )
        opt_init, self.opt_value = optix.adam(lr_critic)
        self.opt_state_value = opt_init(self.params_value)
        # Actor.
        self.actor = hk.without_apply_rng(hk.transform(fn_actor))
        self.params_actor = self.actor.init(
            next(self.rng), *self.fake_args_actor
        )
        opt_init, self.opt_actor = optix.adam(lr_actor)
        self.opt_state_actor = opt_init(self.params_actor)
        # Entropy coefficient.
        if not hasattr(self, "target_entropy"):
            self.target_entropy = -float(self.action_space.shape[0])
        self.log_alpha = jnp.array(np.log(1.0), dtype=jnp.float32)
        opt_init, self.opt_alpha = optix.adam(lr_alpha, b1=0.9)
        self.opt_state_alpha = opt_init(self.log_alpha)

    @partial(jax.jit, static_argnums=0)
    def _select_action(
        self,
        params_actor: hk.Params,
        state: np.ndarray,
    ) -> jnp.ndarray:
        mean, _ = self.actor.apply(params_actor, state)
        return jnp.tanh(mean)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        action = self._select_action(self.params_actor, state[None, ...])
        return np.array(action[0])

    def explore(self, state: np.ndarray) -> np.ndarray:
        key = next(self.rng)
        mean, log_std = self.actor.apply(self.params_actor, state[None, ...])
        action = _reparameterize_gaussian_and_tanh(
            mean=mean, log_std=log_std, key=key, return_log_pi=False
        )
        return np.array(action[0])

    def update(self) -> None:
        batch = self.buffer.sample(self.batch_size)
        state, action, reward, done, next_state = batch

        # Update critic and value networks together
        (loss_critic, critic_aux), grad = jax.value_and_grad(
            self._loss_critic, argnums=(0, 1), has_aux=True
        )(
            self.params_critic,
            self.params_value,
            params_value_target=self.params_value_target,
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state,
            gradient_reduction=self.gradient_reduction,
            **self.kwargs_critic,
        )

        update, self.opt_state_critic = self.opt_critic(
            grad[0], self.opt_state_critic
        )
        self.params_critic = optix.apply_updates(self.params_critic, update)
        update, self.opt_state_value = self.opt_value(
            grad[1], self.opt_state_value
        )
        self.params_value = optix.apply_updates(self.params_value, update)

        # Update actor
        (loss_actor, mean_log_pi), grad = jax.value_and_grad(
            self._loss_actor, has_aux=True
        )(
            self.params_actor,
            params_critic=self.params_critic,
            params_value=self.params_value,
            log_alpha=self.log_alpha,
            state=state,
            action=action,
            **self.kwargs_actor,
        )

        update, self.opt_state_actor = self.opt_actor(
            grad, self.opt_state_actor
        )
        self.params_actor = optix.apply_updates(self.params_actor, update)

        # Update alpha
        (loss_alpha, _), grad = jax.value_and_grad(
            self._loss_alpha, has_aux=True
        )(
            self.log_alpha,
            mean_log_pi=mean_log_pi,
        )
        update, self.opt_state_alpha = self.opt_alpha(
            grad, self.opt_state_alpha
        )
        self.log_alpha = optix.apply_updates(self.log_alpha, update)

        # Update target network.
        self.params_critic_target = self._update_target(
            self.params_critic_target, self.params_critic
        )
        self.params_value_target = self._update_target(
            self.params_value_target, self.params_value
        )

    @partial(jax.jit, static_argnums=0)
    def _loss_critic(
        self,
        params_critic: hk.Params,
        params_value: hk.Params,
        params_value_target: hk.Params,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
        gradient_reduction: float,
    ) -> tuple[jnp.ndarray, dict]:
        next_v_values = self.value.apply(params_value_target, next_state)
        v_targets = jnp.min(jnp.asarray(next_v_values), axis=0)

        q_targets = jax.lax.stop_gradient(
            reward + self.discount * (1.0 - done) * v_targets
        )

        v_values = jnp.asarray(self.value.apply(params_value, state))

        q_values = self.critic.apply(params_critic, state, action)
        q_final = jnp.asarray(q_values[-1:])

        abs_td = jnp.abs(q_targets - q_final)
        q_loss = (abs_td**2).mean()

        rho = gradient_reduction

        optim_advantages = -jnp.asarray(q_values[:-1])

        mix_case = jax.lax.stop_gradient(
            v_values + optim_advantages < q_targets
        )
        upsilon_values = (
            1 - rho * mix_case
        ) * v_values + rho * mix_case * jax.lax.stop_gradient(v_values)

        up_case = jax.lax.stop_gradient(v_values >= q_targets)
        z_values = (
            optim_advantages**2
            + up_case * 2 * optim_advantages * (upsilon_values - q_targets)
            + (upsilon_values - q_targets) ** 2
        )

        va_loss = z_values.mean()

        critic_loss = va_loss + q_loss

        return (critic_loss, {"abs_td": jax.lax.stop_gradient(abs_td)})

    @partial(jax.jit, static_argnums=0)
    def _loss_actor(
        self,
        params_actor: hk.Params,
        params_critic: hk.Params,
        params_value: hk.Params,
        log_alpha: jnp.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        mean, log_std = self.actor.apply(params_actor, state)

        sampled_action, log_pi = _reparameterize_gaussian_and_tanh(
            mean=mean, log_std=log_std, key=key, return_log_pi=True
        )
        q_list = self._calculate_value_list(
            params_critic, state, sampled_action
        )

        mean_log_pi = log_pi.mean()

        mean_q = q_list[-1].mean()
        loss = jax.lax.stop_gradient(jnp.exp(log_alpha)) * mean_log_pi - mean_q

        return loss, jax.lax.stop_gradient(mean_log_pi)

    @partial(jax.jit, static_argnums=0)
    def _loss_alpha(
        self,
        log_alpha: jnp.ndarray,
        mean_log_pi: jnp.ndarray,
    ) -> tuple[jnp.ndarray, Any]:
        return -log_alpha * (self.target_entropy + mean_log_pi), None

    @property
    def kwargs_critic(self):
        return {"key": next(self.rng)} if self.use_key_critic else {}

    @property
    def kwargs_actor(self):
        return {"key": next(self.rng)} if self.use_key_actor else {}

    @partial(jax.jit, static_argnums=0)
    def _calculate_value_list(
        self,
        params_critic: hk.Params,
        state: np.ndarray,
        action: np.ndarray,
    ) -> list[jnp.ndarray]:
        return self.critic.apply(params_critic, state, action)

    @partial(jax.jit, static_argnums=0)
    def _calculate_value(
        self,
        params_critic: hk.Params,
        state: np.ndarray,
        action: np.ndarray,
    ) -> jnp.ndarray:
        return jnp.asarray(
            self._calculate_value_list(params_critic, state, action)
        ).min(axis=0)

    def _fake_state(self, state_space: Env) -> np.ndarray:
        return state_space.sample()[None, ...].astype(np.float32)

    def _fake_action(self, action_space: Env) -> np.ndarray:
        return action_space.sample().astype(np.float32)[None, ...]
