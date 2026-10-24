
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_PATH = "results/evaluation_episodes.csv"
LOG_ROOT = "logs"


def plot_reward_comparison(df):
    summary = df.groupby("policy")["episode_return"].agg(
        ["mean", "std"]
    ).sort_values("mean", ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(
        summary.index,
        summary["mean"],
        yerr=summary["std"].fillna(0),
        capsize=4,
    )
    plt.ylabel("Mean episode return")
    plt.title("CloakRL vs Baselines: Episode Return")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig("figures/reward_comparison.png", dpi=300)
    plt.close()


def plot_action_distribution(df):
    policies = df.groupby("policy")[
        ["exact_count", "city_count", "refuse_count"]
    ].sum()

    totals = policies.sum(axis=1).replace(0, 1)
    percentages = policies.div(totals, axis=0) * 100

    percentages.columns = ["Exact", "City-level", "Refuse"]

    percentages.plot(
        kind="bar",
        stacked=True,
        figsize=(12, 6),
    )
    plt.ylabel("Percentage of actions")
    plt.title("Action Distribution by Policy")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig("figures/action_distribution.png", dpi=300)
    plt.close()


def plot_privacy_metrics(df):
    summary = df.groupby("policy").agg(
        privacy_cost=("privacy_cost", "mean"),
        remaining_budget=("remaining_budget", "mean"),
        overdraw_rate=("overdraw", "mean"),
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    summary["privacy_cost"].plot(kind="bar", ax=axes[0])
    axes[0].set_title("Average Privacy Cost")
    axes[0].set_ylabel("Privacy cost")

    summary["remaining_budget"].plot(kind="bar", ax=axes[1])
    axes[1].set_title("Average Remaining Budget")
    axes[1].set_ylabel("Budget")

    (summary["overdraw_rate"] * 100).plot(kind="bar", ax=axes[2])
    axes[2].set_title("Overdraw Rate")
    axes[2].set_ylabel("Episodes (%)")

    for ax in axes:
        ax.tick_params(axis="x", rotation=70)

    plt.tight_layout()
    plt.savefig("figures/privacy_metrics.png", dpi=300)
    plt.close()


def plot_utility_comparison(df):
    summary = df.groupby("policy")["service_utility"].mean()
    summary = summary.sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(summary.index, summary.values)
    plt.ylabel("Mean service utility")
    plt.title("Service Utility by Policy")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig("figures/service_utility.png", dpi=300)
    plt.close()


def plot_training_curves():
    fig, ax = plt.subplots(figsize=(10, 6))
    found_logs = False

    for algorithm in ["dqn", "ppo"]:
        seed_returns = []

        for seed in [11, 22, 33, 44, 55]:
            path = f"{LOG_ROOT}/{algorithm}/seed{seed}/evaluations.npz"

            if not os.path.exists(path):
                continue

            data = np.load(path)
            timesteps = data["timesteps"]
            rewards = data["results"].mean(axis=1)

            ax.plot(
                timesteps,
                rewards,
                alpha=0.25,
                label=f"{algorithm.upper()} seed {seed}",
            )
            seed_returns.append((timesteps, rewards))
            found_logs = True

        if len(seed_returns) > 1:
            common_steps = seed_returns[0][0]

            if all(
                np.array_equal(steps, common_steps)
                for steps, _ in seed_returns
            ):
                values = np.array([rewards for _, rewards in seed_returns])
                mean = values.mean(axis=0)
                std = values.std(axis=0)

                ax.plot(
                    common_steps,
                    mean,
                    linewidth=2.5,
                    label=f"{algorithm.upper()} mean",
                )
                ax.fill_between(
                    common_steps,
                    mean - std,
                    mean + std,
                    alpha=0.15,
                )

    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Mean validation episode reward")
    ax.set_title("Training Progress Across Random Seeds")

    if found_logs:
        ax.legend(fontsize=7, ncol=2)
    else:
        ax.text(
            0.5,
            0.5,
            "Training logs not found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    plt.tight_layout()
    plt.savefig("figures/training_curves.png", dpi=300)
    plt.close()


def main():
    os.makedirs("figures", exist_ok=True)

    if not os.path.exists(RESULTS_PATH):
        print(f"Results file not found: {RESULTS_PATH}")
        print("Run training and evaluation before generating plots.")
        return

    df = pd.read_csv(RESULTS_PATH)

    if df.empty:
        print("The evaluation results file is empty.")
        return

    plot_reward_comparison(df)
    plot_action_distribution(df)
    plot_privacy_metrics(df)
    plot_utility_comparison(df)
    plot_training_curves()

    print("Graphs saved in the figures/ directory.")


if __name__ == "__main__":
    main()
