# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship.
#
# isir-internship is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import argparse
import logging
import pickle
from pathlib import Path

import gymnasium as gym
from tqdm import tqdm

import algos

from .utils import compute_stats


def get_stats(data: list) -> str:
    min_val, q1, iqm, q3, max_val = compute_stats(data)
    return f"[{min_val:.1f}|{q1:.1f}|{iqm:.1f}|{q3:.1f}|{max_val:.1f}]"


def train(
    agent: algos.SAC,
    train_env: gym.Env,
    test_env: gym.Env,
    steps: int,
    evaluation_frequency: int,
) -> algos.SAC:
    training_steps = 0
    history = {}

    goal_reached_count = 0

    progress = tqdm(range(steps))

    while training_steps < steps:
        state, _ = train_env.reset()

        while True:
            action = agent.select_action(state, evaluation=False)

            next_state, reward, terminated, truncated, _ = train_env.step(
                action
            )
            done = terminated or truncated

            if reward > 50:
                goal_reached_count += 1

            agent.replay_buffer.push(state, action, reward, next_state, done)

            if training_steps > 5000:
                agent.update()

            state = next_state
            training_steps += 1
            progress.update(1)

            if training_steps % evaluation_frequency == 0:
                results = test(agent, test_env, 10)
                history[training_steps // evaluation_frequency] = results
                progress.set_postfix(
                    {"eval": get_stats(results), "goals": goal_reached_count}
                )

            if done or training_steps >= steps:
                break

    return agent, history


def test(agent: algos.SAC, env: gym.Env, n: int) -> list:
    results = []

    for _ in range(n):
        state, _ = env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(state, evaluation=True)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward
            state = next_state

            if done:
                break

        results.append(total_reward)

    return results


def main() -> None:
    envs = {
        "mountaincar": "MountainCarContinuous-v0",
        "pendulum": "Pendulum-v1",
    }

    parser = argparse.ArgumentParser(description="Test RL algorithms")
    parser.add_argument(
        "--env",
        type=str,
        choices=envs.keys(),
        required=True,
        help="Environment",
    )

    parser.add_argument(
        "--steps",
        type=int,
        required=False,
        default=200_000,
        help="Max number of steps for the experiment",
    )
    args = parser.parse_args()

    env_name = envs[args.env]
    train_env = gym.make(env_name)
    test_env = gym.make(env_name)

    agent = algos.SAC(
        action_dim=train_env.action_space.shape[0],
        state_dim=train_env.observation_space.shape[0],
        hidden_dims=[256, 256],
        replay_size=200_000,
        batch_size=256,
        q_lr=3e-4,
        policy_lr=3e-4,
        alpha_lr=3e-4,
        tau=0.005,
        gamma=0.99,
    )

    trained_agent, history = train(agent, train_env, test_env, args.steps, 1000)

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/agent.pt", "wb") as f:
        pickle.dump(trained_agent.get_state(), f)
    with open("outputs/history.pk", "wb") as f:
        pickle.dump(history, f)

    train_env.close()
    test_env.close()


if __name__ == "__main__":
    main()
