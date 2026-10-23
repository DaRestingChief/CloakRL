import os
from stable_baselines3 import DQN, PPO
from privacy_env import PrivacyEnv

def train():
    env = PrivacyEnv()
    os.makedirs("models", exist_ok=True)
    
    # Train DQN
    print("Training DQN...")
    dqn_model = DQN("MlpPolicy", env, learning_rate=1e-3, buffer_size=10000, learning_starts=1000, batch_size=64, gamma=0.99, exploration_fraction=0.2, exploration_final_eps=0.05, target_update_interval=500, verbose=0)
    dqn_model.learn(total_timesteps=30000)
    dqn_model.save(os.path.join("models", "dqn_cloakrl"))
    
    # Train PPO
    print("Training PPO...")
    ppo_model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=64, gamma=0.99, verbose=0)
    ppo_model.learn(total_timesteps=30000)
    ppo_model.save(os.path.join("models", "ppo_cloakrl"))
    
    print("Both models successfully trained and saved.")

if __name__ == "__main__":
    train()