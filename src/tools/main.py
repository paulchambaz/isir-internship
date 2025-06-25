# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship.
#
# isir-internship is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import argparse
import copy
import pickle
from pathlib import Path

import gymnasium as gym
import numpy as np
from tqdm import tqdm

import algos

from .utils import compute_stats


def train(
    agent: algos.AFU,
    train_env: gym.Env,
    test_env: gym.Env,
    steps: int,
    warmup: int,
    train_freq: int,
    gradient_steps: int,
    test_freq: int,
    count: int,
) -> tuple[algos.AFU, dict]:
    history = {}
    agent_history = {}

    agent_state = agent.get_state()
    best_agent_state = agent_state
    best_iqm = -float("inf")

    for i in range(count):
        training_steps = 0

        agent.load_from_state(agent_state)

        progress = tqdm(range(steps), desc=f"Run {i + 1}/{count}")

        while training_steps < steps:
            state, _ = train_env.reset()

            while True:
                action = agent.select_action(state, evaluation=False)

                next_state, reward, terminated, truncated, _ = train_env.step(
                    action
                )
                done = terminated or truncated

                agent.push_buffer(state, action, reward, next_state, done)

                if training_steps > warmup and training_steps % train_freq == 0:
                    for _ in range(gradient_steps):
                        agent.update()

                state = next_state
                training_steps += 1
                progress.update(1)

                if training_steps > warmup and training_steps % test_freq == 0:
                    results = test(agent, test_env, 10)
                    result_id = training_steps // test_freq
                    history.setdefault(result_id, []).extend(results)
                    agent_history[result_id] = copy.deepcopy(agent.get_state())
                    progress.set_postfix({"eval": get_stats(results)})

                if done or training_steps >= steps:
                    break

        final_evaluation = test(agent, test_env, 100)
        _, _, iqm, _, _ = compute_stats(final_evaluation)

        print(get_stats(final_evaluation))

        if iqm > best_iqm:
            best_iqm = iqm
            best_agent_state = copy.deepcopy(agent.get_state())

    agent.load_from_state(best_agent_state)

    return agent, history, agent_history


def get_stats(data: list) -> str:
    min_val, q1, iqm, q3, max_val = compute_stats(data)
    return f"[{min_val:.1f}|{q1:.1f}|{iqm:.1f}|{q3:.1f}|{max_val:.1f}]"


def expert_mountaincar(env: gym.Env, count: int) -> None:
    transitions = []

    for turning_point in range(count):
        state, _ = env.reset()
        steps = 0

        while True:
            action = np.array(
                [-1] if steps < 12 + turning_point else [1], dtype=np.float32
            )

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            transitions.append((state, action, reward, next_state, done))

            state = next_state

            if done:
                break

            steps += 1

    return transitions


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
        default=100_000,
        help="Max number of steps for the experiment",
    )
    parser.add_argument(
        "--runs",
        type=int,
        required=False,
        default=1,
        help="Number of runs for the experiment",
    )
    args = parser.parse_args()

    env_name = envs[args.env]
    train_env = gym.make(env_name)
    test_env = gym.make(env_name)

    # agent = algos.SAC(
    #     action_dim=train_env.action_space.shape[0],
    #     state_dim=train_env.observation_space.shape[0],
    #     hidden_dims=[256, 256],
    #     replay_size=200_000,
    #     batch_size=256,
    #     critic_lr=3e-4,
    #     policy_lr=3e-4,
    #     temperature_lr=3e-4,
    #     tau=0.005,
    #     gamma=0.999,
    #     alpha=None,
    # )

    agent = algos.AFU(
        action_dim=train_env.action_space.shape[0],
        state_dim=train_env.observation_space.shape[0],
        hidden_dims=[256, 256],
        replay_size=200_000,
        batch_size=256,
        critic_lr=3e-4,
        policy_lr=3e-4,
        temperature_lr=3e-4,
        tau=0.005,
        rho=0.7,
        gamma=0.999,
        alpha=None,
    )

    # agent = algos.AFUPerrin(
    #     action_dim=train_env.action_space.shape[0],
    #     state_dim=train_env.observation_space.shape[0],
    #     hidden_dims=[256, 256],
    #     replay_size=200_000,
    #     batch_size=256,
    #     critic_lr=3e-4,
    #     policy_lr=3e-4,
    #     temperature_lr=3e-4,
    #     tau=0.005,
    #     rho=0.3,
    #     gamma=0.999,
    #     alpha=None,
    #     action_space=train_env.action_space,
    #     state_space=train_env.observation_space,
    # )

    if args.env == "mountaincar":
        expert_transitions = expert_mountaincar(train_env, count=20)
        for state, action, reward, next_state, done in expert_transitions:
            agent.push_buffer(state, action, reward, next_state, done)

    trained_agent, history, agent_history = train(
        agent=agent,
        train_env=train_env,
        test_env=test_env,
        steps=args.steps,
        warmup=10000,
        train_freq=32,
        gradient_steps=32,
        test_freq=50,
        count=args.runs,
    )

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/agent.pt", "wb") as f:
        pickle.dump(trained_agent.get_state(), f)
    with open("outputs/history.pk", "wb") as f:
        pickle.dump(history, f)
    with open("outputs/agent_history.pk", "wb") as f:
        pickle.dump(agent_history, f)

    train_env.close()
    test_env.close()


if __name__ == "__main__":
    main()
