import os
import random

import numpy as np
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from privacy_env import PrivacyEnv


SEEDS = [11, 22, 33, 44, 55]
TOTAL_TIMESTEPS = 30000


def train_model(algorithm, seed):
    os.makedirs("models", exist_ok=True)
    os.makedirs(f"logs/{algorithm.lower()}", exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_env = Monitor(PrivacyEnv())
    train_env.reset(seed=seed)

    eval_env = Monitor(PrivacyEnv())
    eval_env.reset(seed=seed + 1000)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"models/{algorithm.lower()}_seed{seed}_best",
        log_path=f"logs/{algorithm.lower()}/seed{seed}",
        eval_freq=2000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=0,
    )

    if algorithm == "DQN":
        model = DQN(
            "MlpPolicy",
            train_env,
            learning_rate=1e-3,
            buffer_size=10000,
            learning_starts=1000,
            batch_size=64,
            gamma=0.99,
            exploration_fraction=0.2,
            exploration_final_eps=0.05,
            target_update_interval=500,
            seed=seed,
            verbose=0,
        )
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            seed=seed,
            verbose=0,
        )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
    )

    model.save(f"models/{algorithm.lower()}_seed{seed}")

    train_env.close()
    eval_env.close()

    print(f"{algorithm} seed {seed} training complete")


def train():
    for algorithm in ["DQN", "PPO"]:
        for seed in SEEDS:
            train_model(algorithm, seed)

    print("All training runs completed.")


if __name__ == "__main__":
    train()