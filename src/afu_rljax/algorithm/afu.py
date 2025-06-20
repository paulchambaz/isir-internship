from functools import partial
from typing import Any

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax as optix

from afu_rljax.algorithm.base_class import OffPolicyActorCritic
from afu_rljax.network import (
    ContinuousQFunction,
    ContinuousVFunction,
    StateDependentGaussianPolicyExtra,
)
from afu_rljax.util import (
    clip_gradient_norm,
    fake_state,
    optimize,
    reparameterize_gaussian_and_tanh,
)


@partial(jax.jit, static_argnums=(0, 1, 4, 7))
def optimize_two_models(
    fn_loss: Any,
    opt: Any,
    opt_state: Any,
    params_to_update: hk.Params,
    opt2: Any,
    opt_state2: Any,
    params_to_update2: hk.Params,
    max_grad_norm: float | None,
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
    if max_grad_norm is not None:
        grad = clip_gradient_norm(grad, max_grad_norm)
    update, opt_state = opt(grad[0], opt_state)
    params_to_update = optix.apply_updates(params_to_update, update)
    update2, opt_state2 = opt2(grad[1], opt_state2)
    params_to_update2 = optix.apply_updates(params_to_update2, update2)
    return opt_state, params_to_update, opt_state2, params_to_update2, loss, aux


class AFU(OffPolicyActorCritic):
    name = "AFU"

    def __init__(
        self,
        num_agent_steps,
        state_space,
        action_space,
        seed,
        max_grad_norm=None,
        gamma=0.99,
        nstep=1,
        num_critics=2,
        buffer_size=10**6,
        use_per=False,
        batch_size=256,
        start_steps=10000,
        update_interval=1,
        tau=1e-2,
        fn_actor=None,
        fn_critic=None,
        fn_value=None,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        units_actor=(256, 256),
        units_critic=(256, 256),
        log_std_min=-10.0,
        log_std_max=2.0,
        d2rl=False,
        init_alpha=1.0,
        adam_b1_alpha=0.9,
        gradient_reduction=0.8,
        alg="AFU",
        ablation=False,
        variant="alpha",
        hyperparam=1.0,  # used in IQL/SQL/EQL baselines, not in AFU
        xpag=False,
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
            max_grad_norm=max_grad_norm,
            gamma=gamma,
            nstep=nstep,
            num_critics=num_critics,
            buffer_size=buffer_size,
            use_per=use_per,
            batch_size=batch_size,
            start_steps=start_steps,
            update_interval=update_interval,
            tau=tau if tau < 1 else 1,
            *args,
            **kwargs,
        )
        if d2rl:
            self.name += "-D2RL"

        self.fake_args_value = (fake_state(state_space),)

        self.num_critics = num_critics
        self.gradient_reduction = gradient_reduction
        self.xpag = xpag
        self.alg = alg
        self.ablation = ablation
        self.variant = variant
        self.hyperparam = hyperparam
        assert self.alg in ["IQL", "SQL", "EQL", "AFU"], (
            'alg must be "AFU", "IQL", "SQL" or "EQL"'
        )
        if self.alg == "AFU":
            delta = num_critics
        else:
            delta = 0

        if fn_critic is None:

            def fn_critic(s, a):
                return ContinuousQFunction(
                    num_critics=delta + 1,
                    hidden_units=units_critic,
                    d2rl=d2rl,
                )(s, a)

        if fn_value is None:

            def fn_value(s):
                return ContinuousVFunction(
                    num_critics=num_critics,
                    hidden_units=units_critic,
                )(s)

        if fn_actor is None:

            def fn_actor(s):
                return StateDependentGaussianPolicyExtra(
                    action_space=action_space,
                    hidden_units=units_actor,
                    log_std_min=log_std_min,
                    log_std_max=log_std_max,
                    d2rl=d2rl,
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
        self.log_alpha = jnp.array(np.log(init_alpha), dtype=jnp.float32)
        opt_init, self.opt_alpha = optix.adam(lr_alpha, b1=adam_b1_alpha)
        self.opt_state_alpha = opt_init(self.log_alpha)

    @partial(jax.jit, static_argnums=0)
    def _select_action(
        self,
        params_actor: hk.Params,
        state: np.ndarray,
    ) -> jnp.ndarray:
        mean, _, _ = self.actor.apply(params_actor, state)
        return jnp.tanh(mean)

    @partial(jax.jit, static_argnums=0)
    def _explore(
        self,
        params_actor: hk.Params,
        state: np.ndarray,
        key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        mean, log_std, _ = self.actor.apply(params_actor, state)
        return reparameterize_gaussian_and_tanh(mean, log_std, key, False)

    def select_action(self, state):
        if self.xpag:
            action = self._select_action(self.params_actor, state)
            return action
        action = self._select_action(self.params_actor, state[None, ...])
        return np.array(action[0])

    def explore(self, state):
        if self.xpag:
            action = self._explore(self.params_actor, state, next(self.rng))
            return action
        action = self._explore(
            self.params_actor, state[None, ...], next(self.rng)
        )
        return np.array(action[0])

    def update(self, writer=None):
        self.learning_step += 1
        weight, batch = self.buffer.sample(self.batch_size)
        state, action, reward, done, next_state = batch

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
            self.max_grad_norm,
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

        # Update priority.
        if self.use_per:
            self.buffer.update_priority(critic_aux["abs_td"])

        # Update actor.
        self.opt_state_actor, self.params_actor, loss_actor, mean_log_pi = (
            optimize(
                self._loss_actor,
                self.opt_actor,
                self.opt_state_actor,
                self.params_actor,
                self.max_grad_norm,
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
            None,
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

        if writer and self.learning_step % 1000 == 0:
            writer.add_scalar("loss/critic", loss_critic, self.learning_step)
            writer.add_scalar("loss/actor", loss_actor, self.learning_step)
            writer.add_scalar("loss/alpha", loss_alpha, self.learning_step)
            writer.add_scalar(
                "stat/alpha", jnp.exp(self.log_alpha), self.learning_step
            )
            writer.add_scalar("stat/entropy", -mean_log_pi, self.learning_step)

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
        mean, log_std, _ = self.actor.apply(params_actor, state)
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
        if self.alg == "AFU":
            target_optim_values = jnp.min(
                jnp.asarray(self.value.apply(params_value_target, next_state)),
                axis=0,
            )
            target_q = jax.lax.stop_gradient(
                reward + (1.0 - done) * self.discount * target_optim_values
            )
            # target_q += 0.5 * jax.random.normal(kwargs["key"], shape=target_q.shape)

            optim_values = jnp.asarray(self.value.apply(params_value, state))

            critic = self._calculate_value_list(params_critic, state, action)
            q_values = jnp.asarray(critic[-1:])
            grad_red = gradient_reduction
            if not self.ablation:
                optim_advantages = -jnp.asarray(critic[:-1])
                up_case = jax.lax.stop_gradient(target_q <= optim_values)
                no_mix_case = jax.lax.stop_gradient(
                    target_q <= optim_values + optim_advantages
                    # ) * up_case
                )

                # mix_case = jax.lax.stop_gradient(
                #     (optim_values < target_q - optim_advantages
                #      ) * (optim_values >= target_q)
                # )
                # no_mix_case = 1 - mix_case

                mix_gd_optim_values = (1 - no_mix_case) * (
                    jax.lax.stop_gradient((1 - grad_red) * optim_values)
                    + grad_red * optim_values
                ) + no_mix_case * optim_values
                loss_critic = (
                    optim_advantages**2
                    + up_case
                    * 2
                    * optim_advantages
                    * (mix_gd_optim_values - target_q)
                    + (mix_gd_optim_values - target_q) ** 2
                ).mean()
            else:
                optim_advantages = -jnp.abs(jnp.asarray(critic[:-1]))
                no_mix_case = jax.lax.stop_gradient(
                    target_q <= optim_values + optim_advantages
                )
                mix_gd_optim_values = (1 - no_mix_case) * (
                    jax.lax.stop_gradient((1 - grad_red) * optim_values)
                    + grad_red * optim_values
                ) + no_mix_case * optim_values
                loss_critic = (
                    (mix_gd_optim_values + optim_advantages - target_q) ** 2
                ).mean()
            abs_td = jnp.abs(target_q - q_values)
            loss_critic += (abs_td**2).mean()
            loss_critic *= weight
            return (loss_critic, {"abs_td": jax.lax.stop_gradient(abs_td)})
        if self.alg == "IQL":
            target_optim_values = jnp.min(
                jnp.asarray(self.value.apply(params_value_target, next_state)),
                axis=0,
            )
            target_q = jax.lax.stop_gradient(
                reward + (1.0 - done) * self.discount * target_optim_values
            )
            optim_values = jnp.asarray(self.value.apply(params_value, state))
            q_values = jnp.asarray(
                self.critic.apply(params_critic, state, action)
            )
            expectile_param = self.hyperparam

            def loss_expectile(diff, expectile=0.8):
                wgt = jnp.where(diff > 0, expectile, (1 - expectile))
                return wgt * (diff**2)

            loss_critic = loss_expectile(
                target_q - optim_values, expectile_param
            ).mean()
            abs_td = jnp.abs(target_q - q_values)
            loss_critic += (abs_td**2).mean()
            loss_critic *= weight
            return (loss_critic, {"abs_td": jax.lax.stop_gradient(abs_td)})
        if self.alg == "SQL":
            target_optim_values = jnp.min(
                jnp.asarray(self.value.apply(params_value_target, next_state)),
                axis=0,
            )
            target_q = jax.lax.stop_gradient(
                reward + (1.0 - done) * self.discount * target_optim_values
            )
            optim_values = jnp.asarray(self.value.apply(params_value, state))
            q_values = jnp.asarray(
                self.critic.apply(params_critic, state, action)
            )
            alpha = self.hyperparam
            sp_term = (target_q - optim_values) / (2 * alpha) + 1.0
            sp_weight = jnp.where(sp_term > 0, 1.0, 0.0)
            loss_critic = (
                sp_weight * (sp_term**2) + optim_values / alpha
            ).mean()
            abs_td = jnp.abs(target_q - q_values)
            loss_critic += (abs_td**2).mean()
            loss_critic *= weight
            return (loss_critic, {"abs_td": jax.lax.stop_gradient(abs_td)})
        # self.alg == "EQL"
        target_optim_values = jnp.min(
            jnp.asarray(self.value.apply(params_value_target, next_state)),
            axis=0,
        )
        target_q = jax.lax.stop_gradient(
            reward + (1.0 - done) * self.discount * target_optim_values
        )
        optim_values = jnp.asarray(self.value.apply(params_value, state))
        q_values = jnp.asarray(self.critic.apply(params_critic, state, action))
        alpha = self.hyperparam
        sp_term = (target_q - optim_values) / alpha
        sp_term = jnp.minimum(sp_term, 5.0)
        max_sp_term = jnp.max(sp_term, axis=0)
        max_sp_term = jnp.where(max_sp_term < -1.0, -1.0, max_sp_term)
        max_sp_term = jax.lax.stop_gradient(max_sp_term)
        loss_critic = (
            jnp.exp(sp_term - max_sp_term)
            + jnp.exp(-max_sp_term) * optim_values / alpha
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
        mean, log_std, extra_loc = self.actor.apply(params_actor, state)
        extra_loc = jnp.clip(extra_loc, -1.0, 1.0)
        sampled_action, log_pi = reparameterize_gaussian_and_tanh(
            mean, log_std, kwargs["key"], True
        )
        q_list = self._calculate_value_list(
            params_critic, state, sampled_action
        )
        mean_log_pi = self._calculate_log_pi(sampled_action, log_pi).mean()

        if self.variant == "alpha":
            mean_q = q_list[-1].mean()
            loss = (
                jax.lax.stop_gradient(jnp.exp(log_alpha)) * mean_log_pi - mean_q
            )
        else:  # self.variant == "beta":
            optim_values = jnp.min(
                jnp.asarray(self.value.apply(params_value, state)), axis=0
            )
            static_q_list = self._calculate_value_list(
                params_critic, state, action
            )
            q_vals = jnp.vstack((q_list[-1], static_q_list[-1]))
            full_optim_values = jnp.vstack((optim_values, optim_values))
            full_action = jnp.vstack(
                (jax.lax.stop_gradient(sampled_action), action)
            )
            activation = jax.lax.stop_gradient(q_vals >= full_optim_values)
            repeat_extra_loc = jnp.vstack((extra_loc, extra_loc))
            loss = jnp.square(
                activation
                * (repeat_extra_loc - jax.lax.stop_gradient(full_action))
            ).sum() / (jnp.sum(activation) + 1e-8)

            def compute_values(action_vec):
                q_l = self._calculate_value_list(
                    params_critic, state, action_vec
                )
                return q_l[-1].mean()

            act_grad = jax.grad(compute_values)(sampled_action)
            dot_products = ((extra_loc - sampled_action) * act_grad).sum(
                -1, keepdims=True
            )
            q = q_list[-1]
            correction_term = -0.5 * jnp.square(
                jax.lax.stop_gradient(extra_loc)
                - jax.lax.stop_gradient(
                    (dot_products < 0.0) * (q < optim_values)
                )
                * sampled_action
            ).sum(-1, keepdims=True)
            correction_term_renormalized = (
                correction_term
                * jax.lax.stop_gradient(
                    jnp.abs(dot_products)
                    / (
                        jnp.square(extra_loc - sampled_action).sum(
                            -1, keepdims=True
                        )
                        + 1e-8
                    )
                )
            )

            mean_q = q.mean() + correction_term_renormalized.sum()
            loss += (
                jax.lax.stop_gradient(jnp.exp(log_alpha)) * mean_log_pi - mean_q
            )

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
