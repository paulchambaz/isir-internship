# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship.
#
# isir-internship is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import gymnasium as gym
from tqdm import tqdm

import algos


def train(env: gym.Env, agent: algos.SAC, steps: int) -> algos.SAC:
    training_steps: int = 0

    progress = tqdm(range(steps))

    while training_steps < steps:
        state, _ = env.reset()

        while True:
            action = agent.select_action(state, evaluation=False)
            print(action)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            training_steps += 1
            progress.update(1)

            if done or training_steps >= steps:
                break

    return agent


def test(env: gym.Env, agent: algos.SAC, n: int) -> list:
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
    env = gym.make("MountainCarContinuous-v0")

    agent = algos.SAC(
        action_dim=env.action_space.shape[0],
        state_dim=env.observation_space.shape[0],
        hidden_dims=[256, 256],
        replay_size=200_000,
        batch_size=256,
        q_lr=3e-4,
        policy_lr=3e-4,
        alpha_lr=3e-4,
        tau=0.01,
        gamma=0.99,
    )

    # agent = algos.SimpleRight(
    #     action_dim=env.action_space.shape[0], direction=-1
    # )

    trained_agent = train(env, agent, 20)
    results = test(env, trained_agent, 10)

    print(results)

    env.close()


if __name__ == "__main__":
    main()
