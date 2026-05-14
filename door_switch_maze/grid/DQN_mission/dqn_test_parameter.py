import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
import torch
import numpy as np
from model import QNetwork
import time

def normalize_obs(obs):
    # Use only the image observation, flatten it, and normalize it to 0~1.
    return np.asarray(obs, dtype=np.float32).flatten() / 255.0

def test_dqn(env_id="MiniGrid-DoorKey-8x8-v0", model_path="dqn_model.pth"):
    env = gym.make(env_id, render_mode="human")
    env = ImgObsWrapper(env)
    
    state_size = int(np.prod(env.observation_space.shape))
    action_size = env.action_space.n
    
    model = QNetwork(state_size, action_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    state, _ = env.reset()
    state = normalize_obs(state)
    done = False
    total_reward = 0
    
    while not done:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_values = model(state_tensor)
        action = np.argmax(action_values.data.numpy())
        
        state, reward, terminated, truncated, _ = env.step(action)
        state = normalize_obs(state)
        done = terminated or truncated
        total_reward += reward
        env.render()
        time.sleep(0.1)
        
    print(f"Test finished. Total reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    test_dqn()
