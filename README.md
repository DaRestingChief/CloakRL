# CloakRL: Adaptive Location Privacy

CloakRL is a Reinforcement Learning project that treats mobile location privacy as a sequential Markov Decision Process (MDP). A Deep Q-Network (DQN) agent manages a finite privacy budget across heterogeneous app requests, dynamically selecting location disclosure precision to optimize the privacy-utility trade-off.

## Project Structure
* `privacy_env.py`: Custom Gymnasium environment defining the state, action spaces, and MDP dynamics.
* `baselines.py`: Static heuristic policies (Always Exact, Always City, Always Refuse, Random) for benchmarking.
* `train.py`: Training pipeline using Stable-Baselines3.
* `evaluate.py`: Benchmarking script that generates performance comparisons.

## Features
* **Custom Gymnasium Environment:** Implements a continuous observation space and discrete action space to simulate real-world app location requests.
* **Deep RL Agent:** Utilizes Stable-Baselines3's `DQN` with an `MlpPolicy` to learn adaptive privacy strategies over thousands of timesteps.
* **Heuristic Baselines:** Includes static policies (Always Exact, Always City, Always Refuse, Random) to benchmark the RL agent's learned performance.

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