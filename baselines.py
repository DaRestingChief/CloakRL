import numpy as np
from privacy_env import PrivacyEnv

# Baseline 1: Always discloses exact location
def always_exact_policy(obs):
    return 0

# Baseline 2: Always discloses city-level location
def always_city_policy(obs):
    return 1

# Baseline 3: Always refuses location requests
def always_refuse_policy(obs):
    return 2

# Baseline 4: Random action selection
def random_policy(obs):
    return np.random.choice([0, 1, 2])

def evaluate_baseline(policy_fn, episodes=100):
    env = PrivacyEnv()
    total_rewards = []
    budgets_left = []
    
    for _ in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
        total_rewards.append(episode_reward)
        budgets_left.append(obs[2]) # Budget remaining is at index 2
        
    return np.mean(total_rewards), np.mean(budgets_left)


baselines = {
    "Always Exact": always_exact_policy,
    "Always City": always_city_policy,
    "Always Refuse": always_refuse_policy,
    "Random": random_policy,
}

if __name__ == "__main__":
    print("Evaluating Baselines over 100 episodes:")
    for name, policy in baselines.items():
        avg_reward, avg_budget = evaluate_baseline(policy)
        print(f"{name:15} | Avg Reward: {avg_reward:7.2f} | Remaining Budget: {avg_budget:6.2f}")
