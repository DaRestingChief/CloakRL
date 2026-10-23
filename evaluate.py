import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN, PPO
from privacy_env import PrivacyEnv
from baselines import evaluate_baseline, baselines

def evaluate_agent(model, episodes=100):
    env = PrivacyEnv()
    total_rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            episode_reward += reward
            done = terminated or truncated
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

if __name__ == "__main__":
    print("Evaluating models over 100 episodes...")
    results = {}
    
    for name, policy in baselines.items():
        avg_reward, _ = evaluate_baseline(policy, episodes=100)
        results[name] = avg_reward
        
    dqn_model = DQN.load("models/dqn_cloakrl")
    results["CloakRL (DQN)"] = evaluate_agent(dqn_model)
    
    ppo_model = PPO.load("models/ppo_cloakrl")
    results["CloakRL (PPO)"] = evaluate_agent(ppo_model)
        
    names = list(results.keys())
    rewards = list(results.values())
    
    plt.figure(figsize=(11, 6))
    colors = ['lightgray'] * 4 + ['royalblue', 'forestgreen']
    bars = plt.bar(names, rewards, color=colors, edgecolor='black')
    
    plt.axhline(0, color='black', linewidth=1.2)
    plt.title("CloakRL vs Baselines: DQN vs PPO Performance", fontsize=14)
    plt.ylabel("Average Cumulative Utility (Reward)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        offset = 5 if yval >= 0 else -15
        plt.text(bar.get_x() + bar.get_width()/2, yval + offset, round(yval, 1), ha='center', va='bottom', fontweight='bold')
        
    plt.savefig("evaluation_plot.png", bbox_inches='tight')
    print("Graph saved as evaluation_plot.png")