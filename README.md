# CloakRL: Adaptive Location Privacy

CloakRL is a Reinforcement Learning project that treats mobile location privacy as a sequential Markov Decision Process (MDP). AI agents manage a finite privacy budget across heterogeneous app requests, dynamically selecting location disclosure precision to optimize the privacy-utility trade-off.

## Project Structure
* `privacy_env.py`: Custom Gymnasium environment defining the state, action spaces, and MDP dynamics.
* `baselines.py`: Static heuristic policies (Always Exact, Always City, Always Refuse, Random) for benchmarking.
* `train.py`: Training pipeline for both Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) algorithms.
* `evaluate.py`: Benchmarking script that generates multi-algorithm performance comparisons.

## Features
* **Custom Gymnasium Environment:** Implements a continuous observation space and discrete action space to simulate real-world app location requests.
* **Comparative RL Analysis:** Utilizes Stable-Baselines3 to train and compare two distinct agents (`DQN` and `PPO`) against each other using an `MlpPolicy`.
* **Heuristic Baselines:** Includes static policies (Always Exact, Always City, Always Refuse, Random) to benchmark the learned RL performance.

## Environment Dynamics (MDP)
* **Observation Space (Continuous):** An array of `[app_urgency, required_resolution, remaining_privacy_budget]`.
* **Action Space (Discrete):** 
  * `0`: Provide Exact Location (High utility, High privacy cost)
  * `1`: Provide City-Level Location (Moderate utility, Low privacy cost)
  * `2`: Refuse Request (Negative utility penalty, Zero privacy cost)
* **Reward Function:** The agent is rewarded based on how well the action satisfies the app's urgency and required resolution, minus the privacy cost. Exhausting the privacy budget results in a severe early-termination penalty.

## Installation
Ensure you have Python 3.8+ installed, then install the required dependencies:
```bash
pip install -r requirements.txt