
import csv
import os

from stable_baselines3 import DQN, PPO

from privacy_env import PrivacyEnv
from baselines import baselines, random_policy


SEEDS = [11, 22, 33, 44, 55]
TEST_EPISODES = 100
TEST_SEED = 90000
RANDOM_POLICY_SEED = 120000


def evaluate_policy(policy_name, policy_fn, episodes, writer, training_seed="baseline"):
    env = PrivacyEnv()

    for episode in range(episodes):
        obs, _ = env.reset(seed=TEST_SEED + episode)
        rng_seed = RANDOM_POLICY_SEED + episode
        episode_rng = __import__("numpy").random.default_rng(rng_seed)

        episode_return = 0.0
        service_utility = 0.0
        privacy_cost = 0.0
        overdraw = False
        exact_count = 0
        city_count = 0
        refuse_count = 0
        episode_length = 0
        done = False

        while not done:
            if policy_fn is random_policy:
                action = policy_fn(obs, episode_rng)
            else:
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
            "training_seed": training_seed,
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

    evaluate_policy(
        algorithm,
        policy_fn,
        TEST_EPISODES,
        writer,
        training_seed=seed,
    )


def main():
    os.makedirs("results", exist_ok=True)
    output_path = "results/evaluation_episodes.csv"

    fieldnames = [
        "policy",
        "training_seed",
        "episode",
        "episode_return",
        "service_utility",
        "privacy_cost",
        "remaining_budget",
        "overdraw",
        "episode_length",
        "exact_count",
        "city_count",
        "refuse_count",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for name, policy in baselines.items():
            evaluate_policy(name, policy, TEST_EPISODES, writer)

        for algorithm in ["DQN", "PPO"]:
            for seed in SEEDS:
                evaluate_model(algorithm, seed, writer)
                print(f"Evaluated {algorithm}, training seed {seed}")

    print(f"Evaluation results saved to {output_path}")


if __name__ == "__main__":
    main()
