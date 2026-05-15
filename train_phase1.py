import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import os


class EarlyStoppingOnMeanReward(BaseCallback):
    def __init__(self, n_episodes=10, target_reward=150, verbose=0):
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
                            f"\n[Phase 1] Early stopping at step {self.num_timesteps}: "
                            f"mean reward = {mean_reward:.1f} over last {self.n_episodes} episodes >= {self.target_reward}"
                        )
                        return False
        return True


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = gym.make("LunarLander-v3")
    env = Monitor(env, "logs/monitor")

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=2.5e-4,
        gae_lambda=0.95,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
    )

    callback = EarlyStoppingOnMeanReward(n_episodes=10, target_reward=150)
    model.learn(total_timesteps=1_000_000, callback=callback)
    model.save("models/ppo_lunarlander_phase1")
    print("Phase 1 complete. Model saved to models/ppo_lunarlander_phase1.zip")

    env.close()


if __name__ == "__main__":
    main()
