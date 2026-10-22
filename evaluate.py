import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from privacy_env import PrivacyEnv
from baselines import evaluate_baseline, baselines

def evaluate_dqn(model_path, episodes=100):
    env = PrivacyEnv()
    model = DQN.load(model_path)
    total_rewards = []
    
    for _ in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            episode_reward += reward
            done = terminated or truncated
        total_rewards.append(episode_reward)
        
    return np.mean(total_rewards)

if __name__ == "__main__":
    print("Evaluating models over 100 episodes...")
    results = {}
    
    # 1. Evaluate Baselines
    for name, policy in baselines.items():
        avg_reward, _ = evaluate_baseline(policy, episodes=100)
        results[name] = avg_reward
        
    # 2. Evaluate RL Agent
    try:
        dqn_reward = evaluate_dqn("models/dqn_cloakrl")
        results["CloakRL (DQN)"] = dqn_reward
    except FileNotFoundError:
        print("Model file not found. Ensure train.py has been run first.")
        exit(1)
        
    # 3. Print results
    print("-" * 35)
    for name, reward in results.items():
        print(f"{name:15} | Avg Reward: {reward:7.2f}")
    print("-" * 35)
        
    # 4. Generate Plot
    names = list(results.keys())
    rewards = list(results.values())
    
    plt.figure(figsize=(10, 6))
    colors = ['lightgray', 'lightgray', 'lightgray', 'lightgray', 'royalblue']
    bars = plt.bar(names, rewards, color=colors, edgecolor='black')
    
    plt.axhline(0, color='black', linewidth=1.2)
    plt.title("CloakRL vs Static Baselines: Privacy-Utility Trade-off", fontsize=14)
    plt.ylabel("Average Cumulative Utility (Reward)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        offset = 5 if yval >= 0 else -15
        plt.text(bar.get_x() + bar.get_width()/2, yval + offset, round(yval, 1), 
                 ha='center', va='bottom', fontweight='bold')
        
    plt.savefig("evaluation_plot.png", bbox_inches='tight')
    print("Graph saved successfully as evaluation_plot.png")