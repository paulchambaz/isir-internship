# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship.
#
# isir-internship is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import argparse
import gc
import pickle
from collections import defaultdict
from pathlib import Path

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


def get_args(args: argparse) -> tuple[str, int, int, int, float, float]:
    m = d = b = r = None
    match args.method:
        case "msac":
            algo = "msac"
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

    return algo, n, m, d, b, r


def get_agent(
    algo: str,
    state: dict,
    env: gym.Env,
    n: int,
    m: int,
    d: int,
    b: float,
    r: float,
    gamma: float,
) -> algos.RLAlgo:
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    hidden_dims = [64, 64]
    replay_size = 400_000
    batch_size = 256
    lr = 3e-4
    tau = 0.005
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
                state=state,
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
                state=state,
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
                state=state,
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
                state=state,
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
                state=state,
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
                state=state,
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
                state=state,
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
                state=state,
            )

    return agent


def compute_results(
    agent: algos.RLAlgo,
    test_env: gym.Env,
    eval_size: int,
    mc_total: int,
    seed: int,
    progress: tqdm,
) -> dict:
    key = random.PRNGKey(seed)
    key, sample_key = random.split(key)
    states, actions, timesteps = agent.buffer.sample_timed_state_action(
        eval_size, sample_key
    )

    true_qs = []
    estimated_qs = []

    for state, action, timestep in zip(states, actions, timesteps, strict=True):
        estimated_qs.append(agent.evaluate(state, action))
        true_qs.append(
            monte_carlo_estimate(
                agent, test_env, state, action, timestep, mc_total
            )
        )
        progress.update(1)

    true_qs = np.array(true_qs, dtype=np.float32)
    estimated_qs = np.array(estimated_qs, dtype=np.float32)

    return {
        "states": states,
        "actions": actions,
        "true_qs": true_qs,
        "estimated_qs": estimated_qs,
    }


def main() -> None:
    envs = {
        "mountaincar": {
            "name": "MountainCarContinuous-v0",
            "kwargs": {},
            "steps": 100_000,
            "gamma": 0.99,
        },
        "pendulum": {
            "name": "Pendulum-v1",
            "kwargs": {},
            "steps": 200_000,
            "gamma": 0.99,
        },
        "lunarlander": {
            "name": "LunarLander-v3",
            "kwargs": {"continuous": True},
            "steps": 200_000,
            "gamma": 0.99,
        },
        "swimmer": {
            "name": "Swimmer-v5",
            "kwargs": {},
            "steps": 400_000,
            "gamma": 0.9999,
        },
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--gpu", action="store_true")

    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--n", type=str, required=True)
    args = parser.parse_args()

    if not args.gpu:
        jax.config.update("jax_platform_name", "cpu")

    algo, n, m, d, b, r = get_args(args)

    env = envs[args.env]
    test_env = gym.make(env["name"], **env["kwargs"])

    seed = 42

    eval_size = 20
    mc_total = 50

    files = sorted(Path(args.dir).glob("agent_history_*.pk"))
    total_steps = len(files) * 100

    results = defaultdict(dict)
    progress = tqdm(total=total_steps * eval_size)

    for file in files:
        with open(file, "rb") as f:
            data = pickle.load(f)  # noqa: S301

        for k, value in data.items():
            step = k * 500

            progress.set_description(
                f"Running ({args.method} n={args.n}), step={step}/{total_steps * 500}"
            )

            agent = get_agent(
                algo, value, test_env, n, m, d, b, r, env["gamma"]
            )
            results[(args.method, args.n)][step] = compute_results(
                agent, test_env, eval_size, mc_total, seed, progress
            )

        del data
        jax.clear_caches()
        gc.collect()

    progress.close()

    with open(f"{args.dir}/simulations.pk", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
