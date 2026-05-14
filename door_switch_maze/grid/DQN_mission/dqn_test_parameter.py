import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
import torch
import numpy as np
from model import QNetwork
import time

def test_dqn(env_id="MiniGrid-DoorKey-8x8-v0", model_path="dqn_model.pth"):
    env = gym.make(env_id, render_mode="human")
    env = FlatObsWrapper(env)
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    model = QNetwork(state_size, action_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_values = model(state_tensor)
        action = np.argmax(action_values.data.numpy())
        
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        env.render()
        time.sleep(0.1)
        
    print(f"Test finished. Total reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    test_dqn()
