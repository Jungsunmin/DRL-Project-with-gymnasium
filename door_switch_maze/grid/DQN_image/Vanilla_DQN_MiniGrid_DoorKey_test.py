# ===============================================================
# Test Script for Baseline DQN on MiniGrid-DoorKey-8x8-v0
# Loads trained weights and evaluates performance
# ===============================================================

import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
import torch
import torch.nn as nn
import numpy as np
import time
import os

# --- Model Architecture (must match training) ---
class QNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    def forward(self, x):
        return self.net(x)

def normalize_obs(obs):
    # Normalize by 255.0 as in the initial implementation
    return np.asarray(obs, dtype=np.float32) / 255.0

def test_dqn(num_episodes=10, render=True):
    # Setup environment
    render_mode = "human" if render else None
    env = gym.make("MiniGrid-DoorKey-8x8-v0", render_mode=render_mode)
    env = FlatObsWrapper(env)
    
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Path handling to load relative to script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "dqn_model_weights.pth")
    
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Load model
    model = QNetwork(input_dim, action_dim).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Model weights loaded successfully from: {model_path}")
    except FileNotFoundError:
        print(f"Error: '{model_path}' not found. Please run the training script first.")
        return

    success_count = 0
    total_rewards = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        done, total_reward = False, 0
        
        while not done:
            if render:
                env.render()
                time.sleep(0.05)
            
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                q_vals = model(state_tensor)
                action = int(torch.argmax(q_vals).item())
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

        total_rewards.append(total_reward)
        if total_reward > 0:
            success_count += 1
        
        print(f"Episode {ep+1}: Reward = {total_reward:.2f}")

    print("\n" + "="*30)
    print(f"Test Results over {num_episodes} episodes:")
    print(f"Success Rate: {success_count/num_episodes * 100:.1f}%")
    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    print("="*30)
    
    env.close()

if __name__ == "__main__":
    # Set render=False if you don't want to see the GUI
    test_dqn(num_episodes=100, render=True)
