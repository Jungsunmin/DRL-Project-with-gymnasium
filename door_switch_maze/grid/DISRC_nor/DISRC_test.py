import gymnasium as gym
import minigrid
import torch
import numpy as np
import time
import os
from DISRC_MiniGrid_DoorKey import DISRCStateEncoder, DISRC_DQN, preprocess_obs, DEVICE

def test_disrc(env_id="MiniGrid-DoorKey-8x8-v0"):
    # Initialize environment with human rendering
    env = gym.make(env_id, render_mode="human")
    
    input_dim = 7 * 7 * 3
    action_dim = env.action_space.n
    
    # Initialize networks
    encoder = DISRCStateEncoder(input_dim=input_dim, encoded_dim=64).to(DEVICE)
    model = DISRC_DQN(input_size=64, action_size=action_dim).to(DEVICE)
    
    # Load saved models
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "disrc_model.pth")
    encoder_path = os.path.join(current_dir, "disrc_encoder.pth")
    
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        encoder.load_state_dict(torch.load(encoder_path, map_location=DEVICE))
        print(f"✅ Loaded models from {model_path} and {encoder_path}")
    else:
        print(f"❌ Error: Model files not found at {model_path} or {encoder_path}")
        return

    model.eval()
    encoder.eval()
    
    print(f"\n🚀 Starting Evaluation...")
    obs, _ = env.reset()
    state_arr = preprocess_obs(obs)
    state_tensor = torch.FloatTensor(state_arr).unsqueeze(0).to(DEVICE)
    done = False
    total_reward = 0
    
    while not done:
        with torch.no_grad():
            encoded_state = encoder(state_tensor)
            q_values = model(encoded_state)
            action = int(torch.argmax(q_values).item())
        
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # MiniGrid rendering
        env.render()
        time.sleep(0.1)

        next_state_arr = preprocess_obs(next_obs)
        state_tensor = torch.FloatTensor(next_state_arr).unsqueeze(0).to(DEVICE)

    print(f"==========================================")
    print(f"🏁 Test Finished. Total Reward: {total_reward:.2f}")
    print(f"==========================================")
    env.close()

if __name__ == "__main__":
    test_disrc()
