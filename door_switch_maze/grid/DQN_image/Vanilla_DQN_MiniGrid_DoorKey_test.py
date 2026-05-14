# ===============================================================
# Test Script for Baseline DQN on MiniGrid-DoorKey-8x8-v0
# Loads trained weights and evaluates performance
# ===============================================================

import gymnasium as gym
import minigrid
import torch
import torch.nn as nn
import numpy as np
import time

# --- Preprocessing Function (must match training) ---
def preprocess_obs(obs):
    img = obs["image"]
    return img.flatten().astype(np.float32) / 255.0

# --- Model Architecture (must match training) ---
class DQN(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    def forward(self, x):
        return self.net(x)

def test_dqn(num_episodes=10, render=True):
    # Setup environment
    render_mode = "human" if render else None
    env = gym.make("MiniGrid-DoorKey-8x8-v0", render_mode=render_mode)
    
    input_dim = 7 * 7 * 3
    action_dim = env.action_space.n
    
    # Load model
    model = DQN(input_dim, action_dim)
    try:
        model.load_state_dict(torch.load("dqn_model_weights.pth"))
        model.eval()
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print("Error: 'dqn_model_weights.pth' not found. Please run the training script first.")
        return

    success_count = 0
    total_rewards = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        state = torch.FloatTensor(preprocess_obs(obs)).unsqueeze(0)
        done, total_reward = False, 0
        
        while not done:
            if render:
                env.render()
                time.sleep(0.05)
            
            with torch.no_grad():
                q_vals = model(state)
                action = int(torch.argmax(q_vals).item())
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = torch.FloatTensor(preprocess_obs(next_obs)).unsqueeze(0)

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
    test_dqn(num_episodes=10, render=False)
