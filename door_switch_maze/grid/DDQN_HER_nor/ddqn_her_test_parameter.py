import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
import torch
import numpy as np
from model import QNetwork
import time
import os

def normalize_obs(obs):
    # Use only the image observation, flatten it, and normalize it to 0~1.
    return np.asarray(obs, dtype=np.float32).flatten() / 255.0

def normalize_goal(goal, env):
    goal = np.asarray(goal, dtype=np.float32)
    scale = np.array([env.unwrapped.width - 1, env.unwrapped.height - 1], dtype=np.float32)
    return goal / scale

def test_ddqn_her(env_id="MiniGrid-DoorKey-8x8-v0", model_path=None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if model_path is None:
        model_path = os.path.join(current_dir, "ddqn_her_model.pth")
    
    env = gym.make(env_id, render_mode="human")
    env = ImgObsWrapper(env)
    
    obs_dim = int(np.prod(env.observation_space.shape))
    state_size = obs_dim + 2
    action_size = env.action_space.n
    
    model = QNetwork(state_size, action_size)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print(f"Model loaded from {model_path}")
    else:
        print("Model file not found. Running with random weights.")
    
    state, _ = env.reset()
    state = normalize_obs(state)
    actual_goal_raw = np.array([env.unwrapped.width-2, env.unwrapped.height-2], dtype=np.float32)
    actual_goal = normalize_goal(actual_goal_raw, env)
    
    done = False
    total_reward = 0
    step_count = 0
    
    while not done and step_count < 250:
        combined_state = np.hstack([state, actual_goal])
        state_tensor = torch.from_numpy(combined_state).float().unsqueeze(0)
        with torch.no_grad():
            action_values = model(state_tensor)
        action = np.argmax(action_values.data.numpy())
        
        state, reward, terminated, truncated, _ = env.step(action)
        state = normalize_obs(state)
        done = terminated or truncated
        total_reward += reward
        step_count += 1
        env.render()
        time.sleep(0.1)
        
    print(f"Test finished. Total reward: {total_reward:.4f}, Steps: {step_count}")
    env.close()

if __name__ == "__main__":
    test_ddqn_her()
