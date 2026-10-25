
import numpy as np
from privacy_env import PrivacyEnv


def always_exact_policy(obs):
    return 0


def always_city_policy(obs):
    return 1


def always_refuse_policy(obs):
    return 2


def random_policy(obs, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return int(rng.choice([0, 1, 2]))


def threshold_policy(obs, tau=0.7):
    urgency, required_resolution, budget_fraction, time_fraction = obs
    budget = budget_fraction * 150.0

    if required_resolution == 1 and urgency >= tau and budget >= 10.0:
        return 0

    if budget >= 2.0:
        return 1

    return 2

def pacing_policy(obs, tau=0.7, max_budget=150.0, max_steps=50):
    urgency, required_resolution, budget_fraction, time_fraction = obs
    # budget left after reserving a city answer for every remaining step
    slack = budget_fraction * max_budget - 2.0 * round(time_fraction * max_steps)
    if required_resolution == 1 and urgency >= tau and slack >= 8.0:
        return 0
    return 1


def evaluate_baseline(policy_fn, episodes=100, seed=0):
    env = PrivacyEnv()
    total_rewards = []
    budgets_left = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        rng = np.random.default_rng(seed + episode)
        episode_reward = 0.0
        done = False

        while not done:
            if policy_fn is random_policy:
                action = policy_fn(obs, rng)
            else:
                action = policy_fn(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)
        budgets_left.append(info["remaining_budget"])

    env.close()
    return float(np.mean(total_rewards)), float(np.mean(budgets_left))


baselines = {
    "Always Exact": always_exact_policy,
    "Always City": always_city_policy,
    "Always Refuse": always_refuse_policy,
    "Random": random_policy,
    "Threshold": threshold_policy,
    "Threshold+Pacing": pacing_policy,
}


if __name__ == "__main__":
    print("Evaluating baselines over 100 episodes:")

    for name, policy in baselines.items():
        avg_reward, avg_budget = evaluate_baseline(policy, episodes=100)
        print(
            f"{name:15} | Average Reward: {avg_reward:8.2f} "
            f"| Remaining Budget: {avg_budget:6.2f}"
        )
