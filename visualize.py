import os

import gymnasium as gym
import imageio
import numpy as np
from stable_baselines3 import PPO


def record_episode(env, policy, max_steps=1000):
    frames = []
    obs, _ = env.reset()
    total_reward = 0.0

    for _ in range(max_steps):
        frame = env.render()
        frames.append(frame)

        if hasattr(policy, "predict"):
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action = policy(obs)

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    return frames, total_reward


def save_gif(frames, path, fps=30):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)
    print(f"Saved GIF: {path} ({len(frames)} frames)")


def main():
    env = gym.make("LunarLander-v3", render_mode="rgb_array")

    # Untrained (random) policy
    def random_policy(obs):
        return env.action_space.sample()

    print("Recording untrained policy...")
    frames, reward = record_episode(env, random_policy)
    save_gif(frames, "gifs/untrained.gif")
    print(f"Untrained total reward: {reward:.2f}")

    # Trained policy
    model_path = "models/ppo_lunarlander.zip"
    if os.path.exists(model_path):
        print("\nRecording trained policy...")
        model = PPO.load(model_path, env=env, device="cpu")
        frames, reward = record_episode(env, model)
        save_gif(frames, "gifs/trained.gif")
        print(f"Trained total reward: {reward:.2f}")
    else:
        print(f"\nTrained model not found: {model_path}")
        print("Run train_phase2.py first.")

    env.close()


if __name__ == "__main__":
    main()
