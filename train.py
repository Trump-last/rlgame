import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = gym.make("LunarLander-v2")
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

    model.learn(total_timesteps=1_000_000)
    model.save("models/ppo_lunarlander")
    print("Model saved to models/ppo_lunarlander.zip")

    env.close()


if __name__ == "__main__":
    main()
