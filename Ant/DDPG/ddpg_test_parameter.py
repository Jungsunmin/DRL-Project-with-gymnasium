import gymnasium as gym
import torch
import numpy as np
import os
BASE_DIR = "./Ant/DDPG"
ACTOR_PATH = os.path.join(BASE_DIR, "actor.pth")
from DDPG_exec import Actor, DEVICE  # 이미 정의된 Actor 클래스를 재사용

def test_agent(model_path=ACTOR_PATH, episodes=5, render=True):
    env = gym.make("Ant-v4", render_mode="human" if render else None)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # --- Load trained actor ---
    actor = Actor(obs_dim, act_dim).to(DEVICE)
    actor.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print(f"Observation dim: {obs_dim}, Action dim: {act_dim}")
    actor.eval()
    print(f"Loaded trained actor from: {model_path}")

    def act(obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            action = actor(obs_t).cpu().numpy()[0]
        return np.clip(action, -1.0, 1.0)

    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=0)
        total_reward = 0.0

        for step in range(1000):  # Ant-v4 max_episode_steps
            action = act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if done:
                break

        print(f"[Test Episode {ep}] Reward: {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    test_agent(episodes=5, render=True)