import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def load_monitor(path):
    rewards, lengths = [], []
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            rewards.append(float(row[0]))
            lengths.append(int(row[1]))
    episodes = np.arange(len(rewards))
    return episodes, np.array(rewards), np.array(lengths)


def moving_average(data, window=100):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def main():
    csv_path = "logs/monitor.csv"
    if not os.path.exists(csv_path):
        print(f"Monitor log not found: {csv_path}")
        print("Run train.py first.")
        return

    episodes, rewards, _ = load_monitor(csv_path)
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, alpha=0.3, label="Episode Reward")

    if len(rewards) >= 100:
        ma = moving_average(rewards, window=100)
        plt.plot(episodes[99:], ma, color="red", linewidth=2, label="100-episode Moving Avg")

    plt.axhline(y=150, color="green", linestyle="--", linewidth=1, label="Target (150)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("LunarLander-v2 Training Rewards")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/rewards.png", dpi=150)
    print("Saved reward plot to plots/rewards.png")


if __name__ == "__main__":
    main()
