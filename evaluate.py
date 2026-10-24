import os
import csv

import numpy as np
from stable_baselines3 import DQN, PPO

from privacy_env import PrivacyEnv
from baselines import baselines


SEEDS = [11, 22, 33, 44, 55]
TEST_EPISODES = 100


def evaluate_policy(policy_name, policy_fn, seed, writer):
    env = PrivacyEnv()

    for episode in range(TEST_EPISODES):
        obs, _ = env.reset(seed=seed * 1000 + episode)
        episode_return = 0.0
        service_utility = 0.0
        privacy_cost = 0.0
        overdraw = False
        exact_count = 0
        city_count = 0
        refuse_count = 0
        done = False
        episode_length = 0

        while not done:
            action = int(policy_fn(obs))
            obs, reward, terminated, truncated, info = env.step(action)

            episode_return += reward
            service_utility += info["utility"]
            privacy_cost += info["privacy_cost"]
            overdraw = overdraw or info["overdraw"]
            episode_length += 1

            if action == 0:
                exact_count += 1
            elif action == 1:
                city_count += 1
            else:
                refuse_count += 1

            done = terminated or truncated

        writer.writerow({
            "policy": policy_name,
            "seed": seed,
            "episode": episode,
            "episode_return": episode_return,
            "service_utility": service_utility,
            "privacy_cost": privacy_cost,
            "remaining_budget": info["remaining_budget"],
            "overdraw": int(overdraw),
            "episode_length": episode_length,
            "exact_count": exact_count,
            "city_count": city_count,
            "refuse_count": refuse_count,
        })

    env.close()


def evaluate_model(algorithm, seed, writer):
    model_path = f"models/{algorithm.lower()}_seed{seed}.zip"
    model_class = DQN if algorithm == "DQN" else PPO
    model = model_class.load(model_path)

    def policy_fn(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    evaluate_policy(f"{algorithm}_seed{seed}", policy_fn, seed, writer)


def main():
    os.makedirs("results", exist_ok=True)
    output_path = "results/evaluation_episodes.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "policy", "seed", "episode", "episode_return",
            "service_utility", "privacy_cost", "remaining_budget",
            "overdraw", "episode_length", "exact_count",
            "city_count", "refuse_count",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for seed in SEEDS:
            for name, policy in baselines.items():
                evaluate_policy(name, policy, seed, writer)

            for algorithm in ["DQN", "PPO"]:
                evaluate_model(algorithm, seed, writer)

    print(f"Evaluation results saved to {output_path}")


if __name__ == "__main__":
    main()