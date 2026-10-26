
# CloakRL: Adaptive Location Privacy

CloakRL is a Reinforcement Learning project that models mobile location privacy as a sequential Markov Decision Process (MDP). An agent manages a finite privacy budget across app requests and chooses whether to share an exact location, share a city-level location, or refuse a request. The goal is to balance service utility against privacy cost over time.

## Project Structure

- `privacy_env.py`: Custom Gymnasium environment defining the state, action space, reward function, and environment dynamics.
- `baselines.py`: Baseline policies including Always Exact, Always City, Always Refuse, Random, and Threshold.
- `optimal_dp.py`: Dynamic programming baseline for comparison with the learned policies.
- `train.py`: Training pipeline for Deep Q-Network (DQN) and Proximal Policy Optimization (PPO).
- `evaluate.py`: Evaluates trained models and baseline policies over multiple episodes and training seeds.
- `plots.py`: Generates graphs for training progress, episode returns, service utility, privacy costs, remaining budget, and action distributions.
- `test_privacy_env.py`: Tests core environment behavior.
- `results/`: Stores evaluation results.
- `figures/`: Stores generated plots.
- `models/`: Stores trained models and validation checkpoints.
- `logs/`: Stores training and validation logs.

## Features

- **Custom Gymnasium Environment:** Simulates sequential app location requests with a finite privacy budget.
- **Multiple RL Algorithms:** Trains and compares DQN and PPO using Stable-Baselines3 and an MLP policy.
- **Baseline Comparison:** Compares learned agents with fixed, random, threshold-based, and dynamic-programming policies.
- **Multi-Seed Experiments:** Uses five training seeds per algorithm to examine variation across runs.
- **Performance Analysis:** Tracks episode return, service utility, privacy cost, remaining budget, episode length, and action distribution.
- **Visualizations:** Generates graphs to examine training behavior and compare policy performance.

## Environment Dynamics (MDP)

### Observation Space

The observation is a four-dimensional vector:

`[app_urgency, required_resolution, budget_fraction, time_fraction]`

- `app_urgency`: Urgency of the current app request, between 0 and 1.
- `required_resolution`: Indicates whether the request requires city-level or exact location.
- `budget_fraction`: Remaining privacy budget divided by the maximum budget.
- `time_fraction`: Fraction of the episode horizon remaining.

### Action Space

The action space is discrete:

- `0`: Provide exact location — high disclosure cost.
- `1`: Provide city-level location — lower disclosure cost.
- `2`: Refuse the request — no privacy expenditure.

### Reward and Budget

The environment assigns service utility based on request urgency, required resolution, and the selected action. For affordable disclosures, the reward is the service utility minus a weighted privacy cost.

The default environment starts with a privacy budget of 100 and uses a maximum episode horizon of 50 requests. Exact-location disclosure costs 10 budget units, while city-level disclosure costs 2. Refusal has no privacy cost.

An unaffordable disclosure triggers a penalty and terminates the episode. Episodes can also terminate when the budget is exhausted or end when the request horizon is reached.

## Algorithms and Evaluation

CloakRL trains DQN and PPO agents and compares them with the baseline policies. The configured training setup uses five random seeds per algorithm and 30,000 timesteps per run.

Evaluation results are recorded in `results/evaluation_episodes.csv`. The plotting script generates figures from the available evaluation and training logs.

The experiments examine whether learned policies balance service utility and privacy expenditure effectively. Results are compared against the baselines rather than assuming that either RL algorithm is always superior.

## Installation

Install the project dependencies using a compatible Python environment:

```bash
pip install -r requirements.txt
```

## Running the Project

Run the environment tests:

```bash
python test_privacy_env.py
```

Train the DQN and PPO models:

```bash
python train.py
```

Evaluate the trained models and baselines:

```bash
python evaluate.py
```

Generate the plots:

```bash
python plots.py
```

Run training before evaluation so that the expected trained model files are available.

## Limitations

CloakRL is evaluated in a simulated environment with synthetic app requests and utility values. Its results demonstrate behavior within this environment and do not, by themselves, establish real-world location privacy guarantees. Performance depends on the reward design, environment dynamics, training configuration, and evaluation setup.
