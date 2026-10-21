import os
from stable_baselines3 import DQN
from privacy_env import PrivacyEnv

def train():
    env = PrivacyEnv()
    
    # DQN agent setup
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        target_update_interval=500,
        verbose=1,
    )
    
    total_timesteps = 30000
    model.learn(total_timesteps=total_timesteps)
    
    # Save model weights
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "dqn_cloakrl")
    model.save(save_path)
    print(f"Model saved: {save_path}.zip")

if __name__ == "__main__":
    train()