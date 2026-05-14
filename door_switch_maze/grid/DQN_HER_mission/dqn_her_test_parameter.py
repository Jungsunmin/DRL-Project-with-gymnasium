import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
import torch
import numpy as np
from model import QNetwork
import time
import os

def normalize_obs(obs):
    # Normalize by 255.0 for project-wide consistency
    return np.asarray(obs, dtype=np.float32) / 255.0

def normalize_goal(goal, env):
    goal = np.asarray(goal, dtype=np.float32)
    scale = np.array([env.unwrapped.width - 1, env.unwrapped.height - 1], dtype=np.float32)
    return goal / scale

def test_dqn_her(env_id="MiniGrid-DoorKey-8x8-v0", model_path=None):
    # 스크립트 파일의 위치를 기준으로 모델 경로 설정
    if model_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "dqn_her_model.pth")
    
    # 렌더링 모드 설정
    env = gym.make(env_id, render_mode="human")
    env = FlatObsWrapper(env)
    
    # HER 모델은 상태(obs)와 목표(goal, 2차원 좌표)를 합쳐서 입력받습니다.
    obs_dim = env.observation_space.shape[0]
    goal_dim = 2
    state_size = obs_dim + goal_dim
    action_size = env.action_space.n
    
    # 모델 로드
    model = QNetwork(state_size, action_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # 초기 상태 및 고정된 최종 목표 설정
    state, _ = env.reset()
    state = normalize_obs(state)
    actual_goal_raw = np.array([env.unwrapped.width-2, env.unwrapped.height-2], dtype=np.float32)
    actual_goal = normalize_goal(actual_goal_raw, env)
    
    done = False
    total_reward = 0
    step_count = 0
    
    print(f"Simulation started using model: {model_path}")
    while not done and step_count < 200:
        # 상태와 목표 결합
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
    test_dqn_her()
