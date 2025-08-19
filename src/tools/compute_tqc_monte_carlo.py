# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship.
#
# isir-internship is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import argparse
import pickle
from collections import defaultdict

import gymnasium as gym
import jax
import numpy as np
from jax import random
from tqdm import tqdm

import algos

from .utils import compute_stats


def get_stats(data: list) -> str:
    min_val, q1, iqm, q3, max_val = compute_stats(data)
    return f"[{min_val:.1f}|{q1:.1f}|{iqm:.1f}|{q3:.1f}|{max_val:.1f}]"


def monte_carlo_estimate(
    agent: algos.RLAlgo,
    env: gym.Env,
    state: np.ndarray,
    action: np.ndarray,
    timestep: int,
    n: int,
) -> float:
    returns = []
    for _ in range(n):
        _, _ = env.reset()
        env.unwrapped.state = state
        env._elapsed_steps = timestep

        total_return = 0.0
        discount = 1.0

        while True:
            action = agent.select_action(state, evaluation=False)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_return += discount * reward
            discount *= agent.gamma

            state = next_state

            if done:
                break

        returns.append(total_return)

    return np.mean(np.array(returns))


def main() -> None:
    seed = 42

    envs = {
        "mountaincar": {
            "name": "MountainCarContinuous-v0",
            "kwargs": {},
            "steps": 100_000,
        },
        "pendulum": {
            "name": "Pendulum-v1",
            "kwargs": {},
            "steps": 200_000,
        },
        "lunarlander": {
            "name": "LunarLander-v3",
            "kwargs": {"continuous": True},
            "steps": 200_000,
        },
        "swimmer": {
            "name": "Swimmer-v5",
            "kwargs": {},
            "steps": 400_000,
        },
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--gpu", action="store_true")

    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--n", type=str, required=True)
    args = parser.parse_args()

    if not args.gpu:
        jax.config.update("jax_platform_name", "cpu")

    m = d = b = r = None
    match args.method:
        case "msac":
            algo = "sac"
            n = int(args.n)
        case "sac":
            algo = "sac"
            n = int(args.n)
        case "ttqc":
            algo = "tqc"
            n = 1
            m = 25
            d = int(args.n)
        case "tqc":
            algo = "tqc"
            n = 2
            m = 25
            d = int(args.n)
        case "top":
            algo = "top"
            n = 2
            m = 25
            b = float(args.n)
        case "ndtop":
            algo = "ndtop"
            n = 2
            b = float(args.n)
        case "afu":
            algo = "afu"
            n = 2
            r = float(args.n)
        case "tafu":
            algo = "afu"
            n = 1
            r = float(args.n)

    env = envs[args.env]
    test_env = gym.make(env["name"], **env["kwargs"])

    with open(args.file, "rb") as f:
        data = pickle.load(f)  # noqa: S301

    eval_size = 50
    mc_total = 100

    results = defaultdict(dict)

    progress = tqdm(data.items())

    for k, value in data.items():
        step = k * 500

        action_dim = test_env.action_space.shape[0]
        state_dim = test_env.observation_space.shape[0]
        hidden_dims = [64, 64]
        replay_size = 400_000
        batch_size = 256
        lr = 3e-4
        tau = 0.005
        gamma = 0.9999
        alpha = None
        seed = 42
        rho = r if r else 0.7
        n_critics = n if n else 2
        n_quantiles = m if m else 25
        quantiles_drop = -d if d else -2
        beta = b if b else -1.0

        match algo:
            case "sac":
                agent = algos.SAC(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    hidden_dims=hidden_dims,
                    replay_size=replay_size,
                    batch_size=batch_size,
                    critic_lr=lr,
                    policy_lr=lr,
                    temperature_lr=lr,
                    tau=tau,
                    gamma=gamma,
                    alpha=alpha,
                    n_critics=n_critics,
                    seed=seed,
                    state=value,
                )
            case "msac":
                agent = algos.MSAC(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    hidden_dims=hidden_dims,
                    replay_size=replay_size,
                    batch_size=batch_size,
                    critic_lr=lr,
                    policy_lr=lr,
                    temperature_lr=lr,
                    tau=tau,
                    gamma=gamma,
                    alpha=alpha,
                    n_critics=n_critics,
                    seed=seed,
                    state=value,
                )
            case "afu":
                agent = algos.AFU(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    hidden_dims=hidden_dims,
                    replay_size=replay_size,
                    batch_size=batch_size,
                    critic_lr=lr,
                    policy_lr=lr,
                    temperature_lr=lr,
                    tau=tau,
                    gamma=gamma,
                    alpha=alpha,
                    rho=rho,
                    n_critics=n_critics,
                    seed=seed,
                    state=value,
                )
            case "afutqc":
                agent = algos.AFUTQC(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    hidden_dims=hidden_dims,
                    replay_size=replay_size,
                    batch_size=batch_size,
                    critic_lr=lr,
                    policy_lr=lr,
                    temperature_lr=lr,
                    tau=tau,
                    gamma=gamma,
                    alpha=alpha,
                    rho=rho,
                    n_quantiles=n_quantiles,
                    n_critics=n_critics,
                    quantiles_drop=quantiles_drop,
                    seed=seed,
                    state=value,
                )
            case "afup":
                agent = algos.AFUP(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    hidden_dims=hidden_dims,
                    replay_size=replay_size,
                    batch_size=batch_size,
                    critic_lr=lr,
                    policy_lr=lr,
                    temperature_lr=lr,
                    tau=tau,
                    gamma=gamma,
                    alpha=alpha,
                    rho=rho,
                    seed=seed,
                    state=value,
                )
            case "tqc":
                agent = algos.TQC(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    hidden_dims=hidden_dims,
                    replay_size=replay_size,
                    batch_size=batch_size,
                    critic_lr=lr,
                    policy_lr=lr,
                    temperature_lr=lr,
                    tau=tau,
                    gamma=gamma,
                    alpha=alpha,
                    n_quantiles=n_quantiles,
                    n_critics=n_critics,
                    quantiles_drop=quantiles_drop,
                    seed=seed,
                    state=value,
                )
            case "top":
                agent = algos.TOP(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    hidden_dims=hidden_dims,
                    replay_size=replay_size,
                    batch_size=batch_size,
                    critic_lr=lr,
                    policy_lr=lr,
                    temperature_lr=lr,
                    tau=tau,
                    gamma=gamma,
                    alpha=alpha,
                    n_quantiles=n_quantiles,
                    n_critics=n_critics,
                    beta=beta,
                    seed=seed,
                    state=value,
                )
            case "ndtop":
                agent = algos.NDTOP(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    hidden_dims=hidden_dims,
                    replay_size=replay_size,
                    batch_size=batch_size,
                    critic_lr=lr,
                    policy_lr=lr,
                    temperature_lr=lr,
                    tau=tau,
                    gamma=gamma,
                    alpha=alpha,
                    n_critics=n_critics,
                    beta=beta,
                    seed=seed,
                    state=value,
                )
            case _:
                return

        key = random.PRNGKey(seed)
        key, sample_key = random.split(key)
        states, actions, timesteps = agent.buffer.sample_timed_state_action(
            eval_size, sample_key
        )

        true_qs = []
        estimated_qs = []

        for state, action, timestep in zip(
            states, actions, timesteps, strict=True
        ):
            estimated_qs.append(agent.evaluate(state, action))
            true_qs.append(
                monte_carlo_estimate(
                    agent, test_env, state, action, timestep, mc_total
                )
            )

        true_qs = np.array(true_qs, dtype=np.float32)
        estimated_qs = np.array(estimated_qs, dtype=np.float32)

        results[(args.method, args.n)][step] = {
            "states": states,
            "actions": actions,
            "true_qs": true_qs,
            "estimated_qs": estimated_qs,
        }

        errors = estimated_qs - true_qs

        progress.update(1)
        progress.set_postfix({"bias": get_stats(errors)})

    with open("outputs/random_test.pk", "wb") as f:
        pickle.dump(results)


if __name__ == "__main__":
    main()
