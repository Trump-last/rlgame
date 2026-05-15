import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import os

from centered_reward_wrapper import CenteredLandingWrapper


class EarlyStoppingOnMeanReward(BaseCallback):
    def __init__(self, n_episodes=10, target_reward=200, verbose=0):
        super().__init__(verbose)
        self.n_episodes = n_episodes
        self.target_reward = target_reward
        self.episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                if len(self.episode_rewards) >= self.n_episodes:
                    mean_reward = sum(self.episode_rewards[-self.n_episodes :]) / self.n_episodes
                    if mean_reward >= self.target_reward:
                        print(
                            f"\n[Phase 2] Early stopping at step {self.num_timesteps}: "
                            f"mean reward = {mean_reward:.1f} over last {self.n_episodes} episodes >= {self.target_reward}"
                        )
                        return False
        return True


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    phase1_path = "models/ppo_lunarlander_phase1.zip"
    if not os.path.exists(phase1_path):
        print(f"Phase 1 model not found: {phase1_path}")
        print("Run train_phase1.py first.")
        return

    env = gym.make("LunarLander-v3")
    env = CenteredLandingWrapper(env, penalty_coeff=0.03)
    env = Monitor(env, "logs/monitor")

    model = PPO.load(phase1_path, env=env)
    print(f"Loaded phase 1 model from {phase1_path}")
    print("Starting phase 2 fine-tuning with centered landing penalty (coeff=0.03)...")

    callback = EarlyStoppingOnMeanReward(n_episodes=10, target_reward=200)
    model.learn(total_timesteps=1_000_000, callback=callback)
    model.save("models/ppo_lunarlander")
    print("Phase 2 complete. Model saved to models/ppo_lunarlander.zip")

    env.close()


if __name__ == "__main__":
    main()
