
from privacy_env import PrivacyEnv
import numpy as np


def test_reset():
    env = PrivacyEnv()
    obs, _ = env.reset(seed=42)

    assert obs.shape == (5,)
    assert env.observation_space.contains(obs)
    assert env.current_budget == 150.0      # was 100.0
    assert env.current_step == 0
    print("Reset test passed")


def test_city_action():
    env = PrivacyEnv()
    env.reset(seed=42)
    env.current_budget = 10.0

    obs, reward, terminated, truncated, info = env.step(1)

    assert env.current_budget == 8.0
    assert info["privacy_cost"] == 2.0
    assert not info["overdraw"]
    assert np.isfinite(reward)
    print("City action test passed")


def test_exact_action():
    env = PrivacyEnv()
    env.reset(seed=42)
    env.current_budget = 10.0

    obs, reward, terminated, truncated, info = env.step(0)

    assert env.current_budget == 0.0
    assert info["privacy_cost"] == 10.0
    assert terminated
    assert not info["overdraw"]
    print("Exact action test passed")


def test_unaffordable_action():
    env = PrivacyEnv()
    env.reset(seed=42)
    env.current_budget = 5.0

    obs, reward, terminated, truncated, info = env.step(0)

    assert reward == -50.0
    assert env.current_budget == 5.0
    assert info["overdraw"]
    assert terminated
    print("Unaffordable action test passed")


def test_refusal():
    env = PrivacyEnv()
    env.reset(seed=42)
    initial_budget = env.current_budget

    obs, reward, terminated, truncated, info = env.step(2)

    assert env.current_budget == initial_budget
    assert info["privacy_cost"] == 0.0
    assert not info["overdraw"]
    print("Refusal test passed")


if __name__ == "__main__":
    test_reset()
    test_city_action()
    test_exact_action()
    test_unaffordable_action()
    test_refusal()
    print("\nAll environment tests passed!")
