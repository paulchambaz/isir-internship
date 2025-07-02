# afu2.py

import math
from abc import abstractmethod
from functools import partial
from typing import Any

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax as optix
from gymnasium.spaces import Box, Discrete
from haiku import PRNGSequence
from jax import nn


@jax.jit
def soft_update(
    target_params: hk.Params,
    online_params: hk.Params,
    tau: float,
) -> hk.Params:
    return jax.tree.map(
        lambda t, s: (1 - tau) * t + tau * s, target_params, online_params
    )


def fake_state(state_space):
    state = state_space.sample()[None, ...]
    if len(state_space.shape) == 1:
        state = state.astype(np.float32)
    return state


def fake_action(action_space):
    return action_space.sample().astype(np.float32)[None, ...]


@jax.jit
def gaussian_log_prob(
    log_std: jnp.ndarray,
    noise: jnp.ndarray,
) -> jnp.ndarray:
    return -0.5 * (jnp.square(noise) + 2 * log_std + jnp.log(2 * math.pi))


@jax.jit
def gaussian_and_tanh_log_prob(
    log_std: jnp.ndarray,
    noise: jnp.ndarray,
    action: jnp.ndarray,
) -> jnp.ndarray:
    return gaussian_log_prob(log_std, noise) - jnp.log(
        nn.relu(1.0 - jnp.square(action)) + 1e-6
    )


@partial(jax.jit, static_argnums=3)
def reparameterize_gaussian_and_tanh(
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
    key: jnp.ndarray,
    return_log_pi: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    std = jnp.exp(log_std)
    noise = jax.random.normal(key, std.shape)
    action = jnp.tanh(mean + noise * std)
    if return_log_pi:
        return action, gaussian_and_tanh_log_prob(log_std, noise, action).sum(
            axis=1, keepdims=True
        )
    return action


class ReplayBuffer:
    def __init__(
        self,
        buffer_size,
        state_space,
        action_space,
        gamma,
    ):
        assert len(state_space.shape) in (1, 3)

        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.state_shape = state_space.shape
        self.use_image = len(self.state_shape) == 3

        if self.use_image:
            # Store images as a list of LazyFrames, which uses 4 times less memory.
            self.state = [None] * buffer_size
            self.next_state = [None] * buffer_size
        else:
            self.state = np.empty(
                (buffer_size, *state_space.shape), dtype=np.float32
            )
            self.next_state = np.empty(
                (buffer_size, *state_space.shape), dtype=np.float32
            )

        if type(action_space) == Box:
            self.action = np.empty(
                (buffer_size, *action_space.shape), dtype=np.float32
            )
        elif type(action_space) == Discrete:
            self.action = np.empty((buffer_size, 1), dtype=np.int32)
        else:
            NotImplementedError

        self.reward = np.empty((buffer_size, 1), dtype=np.float32)
        self.done = np.empty((buffer_size, 1), dtype=np.float32)

    def append(
        self, state, action, reward, done, next_state, episode_done=None
    ):
        self._append(state, action, reward, done, next_state)

    def _append(self, state, action, reward, done, next_state):
        self.state[self._p] = state
        self.action[self._p] = action
        self.reward[self._p] = float(reward)
        self.done[self._p] = float(done)
        self.next_state[self._p] = next_state

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def _sample_idx(self, batch_size):
        return np.random.randint(low=0, high=self._n, size=batch_size)

    def _sample(self, idxes):
        if self.use_image:
            state = np.empty((len(idxes), *self.state_shape), dtype=np.uint8)
            next_state = state.copy()
            for i, idx in enumerate(idxes):
                state[i, ...] = self.state[idx]
                next_state[i, ...] = self.next_state[idx]
        else:
            state = self.state[idxes]
            next_state = self.next_state[idxes]

        return (
            state,
            self.action[idxes],
            self.reward[idxes],
            self.done[idxes],
            next_state,
        )

    def sample(self, batch_size):
        idxes = self._sample_idx(batch_size)
        batch = self._sample(idxes)
        # Use fake weight to use the same interface with PER.
        weight = np.ones((), dtype=np.float32)
        return weight, batch


class OffPolicyActorCritic:
    def __init__(
        self,
        num_agent_steps,
        state_space,
        action_space,
        seed,
        gamma,
        buffer_size,
        batch_size,
        tau=None,
    ):
        if not hasattr(self, "use_key_critic"):
            self.use_key_critic = False
        if not hasattr(self, "use_key_actor"):
            self.use_key_actor = False

        np.random.seed(seed)
        self.rng = PRNGSequence(seed)

        self.agent_step = 0
        self.episode_step = 0
        self.learning_step = 0
        self.num_agent_steps = num_agent_steps
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.discrete_action = False if type(action_space) == Box else True

        if not hasattr(self, "buffer"):
            self.buffer = ReplayBuffer(
                buffer_size=buffer_size,
                state_space=state_space,
                action_space=action_space,
                gamma=gamma,
            )

        self.discount = gamma
        self.batch_size = batch_size
        self.cumulated_reward = 0.0
        self.step_list = []

        self._update_target = jax.jit(partial(soft_update, tau=tau))

        # Define fake input for critic.
        if not hasattr(self, "fake_args_critic"):
            if self.discrete_action:
                self.fake_args_critic = (fake_state(state_space),)
            else:
                self.fake_args_critic = (
                    fake_state(state_space),
                    fake_action(action_space),
                )

        # Define fake input for actor.
        if not hasattr(self, "fake_args_actor"):
            self.fake_args_actor = (fake_state(state_space),)

    @property
    def kwargs_critic(self):
        return {"key": next(self.rng)} if self.use_key_critic else {}

    @property
    def kwargs_actor(self):
        return {"key": next(self.rng)} if self.use_key_actor else {}

    def select_action(self, state):
        action = self._select_action(self.params_actor, state[None, ...])
        return np.array(action[0])

    def explore(self, state):
        action = self._explore(
            self.params_actor, state[None, ...], next(self.rng)
        )
        return np.array(action[0])

    @abstractmethod
    def _sample_action(self, params_actor, state, *args, **kwargs):
        pass

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

    @partial(jax.jit, static_argnums=0)
    def _calculate_loss_critic_and_abs_td(
        self,
        value_list: list[jnp.ndarray],
        target: jnp.ndarray,
        weight: np.ndarray,
    ) -> jnp.ndarray:
        abs_td = jnp.abs(target - value_list[0])
        loss_critic = (jnp.square(abs_td) * weight).mean()
        for value in value_list[1:]:
            loss_critic += (jnp.square(target - value) * weight).mean()
        return loss_critic, jax.lax.stop_gradient(abs_td)

    def is_update(self):
        return True

    def __str__(self):
        return self.name


class MLP(hk.Module):
    def __init__(
        self,
        output_dim,
        hidden_units,
        hidden_activation=nn.relu,
        output_activation=None,
        hidden_scale=1.0,
        output_scale=1.0,
    ):
        super(MLP, self).__init__()
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
        x_input = x
        for i, unit in enumerate(self.hidden_units):
            x = hk.Linear(unit, **self.hidden_kwargs)(x)
            x = self.hidden_activation(x)
        x = hk.Linear(self.output_dim, **self.output_kwargs)(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x


class ContinuousQFunction(hk.Module):
    def __init__(
        self,
        num_critics,
        hidden_units,
    ):
        super(ContinuousQFunction, self).__init__()
        self.num_critics = num_critics
        self.hidden_units = hidden_units

    def __call__(self, s, a):
        def _fn(x):
            return MLP(
                1,
                self.hidden_units,
                hidden_activation=nn.relu,
                hidden_scale=np.sqrt(2),
            )(x)

        x = jnp.concatenate([s, a], axis=1)
        # Return list even if num_critics == 1 for simple implementation.
        return [_fn(x) for _ in range(self.num_critics)]


class ContinuousVFunction(hk.Module):
    def __init__(
        self,
        num_critics=1,
        hidden_units=(64, 64),
    ):
        super(ContinuousVFunction, self).__init__()
        self.num_critics = num_critics
        self.hidden_units = hidden_units

    def __call__(self, x):
        def _fn(x):
            return MLP(
                1,
                self.hidden_units,
                # hidden_activation=jnp.tanh,
                hidden_activation=nn.relu,
                hidden_scale=np.sqrt(2),
            )(x)

        if self.num_critics == 1:
            return _fn(x)
        return [_fn(x) for _ in range(self.num_critics)]


class StateDependentGaussianPolicyExtra(hk.Module):
    def __init__(
        self,
        action_space,
        hidden_units=(256, 256),
        log_std_min=-10.0,
        log_std_max=2.0,
        clip_log_std=True,
    ):
        super(StateDependentGaussianPolicyExtra, self).__init__()
        self.action_space = action_space
        self.hidden_units = hidden_units
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.clip_log_std = clip_log_std

    def __call__(self, x):
        x = MLP(
            2 * self.action_space.shape[0],
            self.hidden_units,
            hidden_activation=nn.relu,
            hidden_scale=np.sqrt(2),
        )(x)
        mean, log_std = jnp.split(x, 2, axis=1)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std


class AFU(OffPolicyActorCritic):
    name = "AFU"

    def __init__(
        self,
        num_agent_steps,
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
        if not hasattr(self, "use_key_critic"):
            self.use_key_critic = False
            # self.use_key_critic = True
        if not hasattr(self, "use_key_actor"):
            self.use_key_actor = True

        self.info_dict = {}

        self.tau = tau
        self.target_update_period = 1 if tau < 1 else int(tau)

        super(AFU, self).__init__(
            num_agent_steps=num_agent_steps,
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            gamma=gamma,
            buffer_size=buffer_size,
            batch_size=batch_size,
            tau=tau if tau < 1 else 1,
            *args,
            **kwargs,
        )

        self.fake_args_value = (fake_state(state_space),)

        self.gradient_reduction = gradient_reduction

        def fn_critic(s, a):
            return ContinuousQFunction(
                num_critics=3,
                hidden_units=units_critic,
            )(s, a)

        def fn_value(s):
            return ContinuousVFunction(
                num_critics=2,
                hidden_units=units_critic,
            )(s)

        def fn_actor(s):
            return StateDependentGaussianPolicyExtra(
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

    @partial(jax.jit, static_argnums=0)
    def _explore(
        self,
        params_actor: hk.Params,
        state: np.ndarray,
        key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        mean, log_std = self.actor.apply(params_actor, state)
        return reparameterize_gaussian_and_tanh(mean, log_std, key, False)

    def select_action(self, state):
        action = self._select_action(self.params_actor, state[None, ...])
        return np.array(action[0])

    def explore(self, state):
        action = self._explore(
            self.params_actor, state[None, ...], next(self.rng)
        )
        return np.array(action[0])

    def update(self):
        self.learning_step += 1
        weight, batch = self.buffer.sample(self.batch_size)
        state, action, reward, done, next_state = batch

        @partial(jax.jit, static_argnums=(0, 1, 4, 7))
        def optimize_two_models(
            fn_loss: Any,
            opt: Any,
            opt_state: Any,
            params_to_update: hk.Params,
            opt2: Any,
            opt_state2: Any,
            params_to_update2: hk.Params,
            *args,
            **kwargs,
        ) -> tuple[Any, hk.Params, Any, hk.Params, jnp.ndarray, Any]:
            (loss, aux), grad = jax.value_and_grad(
                fn_loss, argnums=(0, 1), has_aux=True
            )(
                params_to_update,
                params_to_update2,
                *args,
                **kwargs,
            )
            update, opt_state = opt(grad[0], opt_state)
            params_to_update = optix.apply_updates(params_to_update, update)
            update2, opt_state2 = opt2(grad[1], opt_state2)
            params_to_update2 = optix.apply_updates(params_to_update2, update2)
            return (
                opt_state,
                params_to_update,
                opt_state2,
                params_to_update2,
                loss,
                aux,
            )

        # Update critic.
        (
            self.opt_state_critic,
            self.params_critic,
            self.opt_state_value,
            self.params_value,
            loss_critic,
            critic_aux,
        ) = optimize_two_models(
            self._loss_critic,
            self.opt_critic,
            self.opt_state_critic,
            self.params_critic,
            self.opt_value,
            self.opt_state_value,
            self.params_value,
            params_value_target=self.params_value_target,
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state,
            weight=weight,
            gradient_reduction=self.gradient_reduction,
            **self.kwargs_critic,
        )

        @partial(jax.jit, static_argnums=(0, 1, 4))
        def optimize(
            fn_loss: Any,
            opt: Any,
            opt_state: Any,
            params_to_update: hk.Params,
            *args,
            **kwargs,
        ) -> tuple[Any, hk.Params, jnp.ndarray, Any]:
            (loss, aux), grad = jax.value_and_grad(fn_loss, has_aux=True)(
                params_to_update,
                *args,
                **kwargs,
            )
            update, opt_state = opt(grad, opt_state)
            params_to_update = optix.apply_updates(params_to_update, update)
            return opt_state, params_to_update, loss, aux

        # Update actor.
        self.opt_state_actor, self.params_actor, loss_actor, mean_log_pi = (
            optimize(
                self._loss_actor,
                self.opt_actor,
                self.opt_state_actor,
                self.params_actor,
                params_critic=self.params_critic,
                params_value=self.params_value,
                log_alpha=self.log_alpha,
                state=state,
                action=action,
                **self.kwargs_actor,
            )
        )

        # Update alpha.
        self.opt_state_alpha, self.log_alpha, loss_alpha, _ = optimize(
            self._loss_alpha,
            self.opt_alpha,
            self.opt_state_alpha,
            self.log_alpha,
            mean_log_pi=mean_log_pi,
        )

        # Update target network.
        if not self.learning_step % self.target_update_period:
            self.params_critic_target = self._update_target(
                self.params_critic_target, self.params_critic
            )
            self.params_value_target = self._update_target(
                self.params_value_target, self.params_value
            )

        self.info_dict["log_alpha"] = self.log_alpha
        self.info_dict["entropy"] = -mean_log_pi
        self.info_dict["loss/critic"] = loss_critic
        self.info_dict["loss/actor"] = loss_actor
        self.info_dict["loss/alpha"] = loss_alpha

    @partial(jax.jit, static_argnums=0)
    def _sample_action(
        self,
        params_actor: hk.Params,
        state: np.ndarray,
        key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        mean, log_std = self.actor.apply(params_actor, state)
        return reparameterize_gaussian_and_tanh(mean, log_std, key, True)

    @partial(jax.jit, static_argnums=0)
    def _calculate_log_pi(
        self,
        action: np.ndarray,
        log_pi: np.ndarray,
    ) -> jnp.ndarray:
        return log_pi

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
        weight: float | np.ndarray | list[jnp.ndarray],
        gradient_reduction: float,
        *args,
        **kwargs,
    ) -> tuple[jnp.ndarray, dict]:
        target_optim_values = jnp.min(
            jnp.asarray(self.value.apply(params_value_target, next_state)),
            axis=0,
        )
        target_q = jax.lax.stop_gradient(
            reward + (1.0 - done) * self.discount * target_optim_values
        )

        optim_values = jnp.asarray(self.value.apply(params_value, state))

        critic = self._calculate_value_list(params_critic, state, action)
        q_values = jnp.asarray(critic[-1:])
        grad_red = gradient_reduction

        optim_advantages = -jnp.asarray(critic[:-1])
        mix_case = jax.lax.stop_gradient(
            optim_values + optim_advantages < target_q
        )
        mix_gd_optim_values = optim_values * (
            1 - mix_case * grad_red
        ) + jax.lax.stop_gradient(optim_values) * (mix_case * grad_red)
        up_case = jax.lax.stop_gradient(target_q <= optim_values)

        loss_critic = (
            optim_advantages**2
            + up_case * 2 * optim_advantages * (mix_gd_optim_values - target_q)
            + (mix_gd_optim_values - target_q) ** 2
        ).mean()

        abs_td = jnp.abs(target_q - q_values)
        loss_critic += (abs_td**2).mean()
        loss_critic *= weight
        return (loss_critic, {"abs_td": jax.lax.stop_gradient(abs_td)})

    @partial(jax.jit, static_argnums=0)
    def _loss_actor(
        self,
        params_actor: hk.Params,
        params_critic: hk.Params,
        params_value: hk.Params,
        log_alpha: jnp.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        *args,
        **kwargs,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        mean, log_std = self.actor.apply(params_actor, state)
        sampled_action, log_pi = reparameterize_gaussian_and_tanh(
            mean, log_std, kwargs["key"], True
        )
        q_list = self._calculate_value_list(
            params_critic, state, sampled_action
        )
        mean_log_pi = self._calculate_log_pi(sampled_action, log_pi).mean()

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

    def _calculate_target(
        self,
        params_critic_target,
        reward,
        done,
        next_state,
        next_action,
        *args,
        **kwargs,
    ):
        pass

    def calculate_value(
        self,
        params_critic: hk.Params,
        state: np.ndarray,
        action: np.ndarray,
    ) -> jnp.ndarray:
        critic = self._calculate_value_list(params_critic, state, action)
        return jnp.min(jnp.asarray(critic[:-1]), axis=0)

    def show_stuff(self):
        for key in self.info_dict:
            print(key, self.info_dict[key])
