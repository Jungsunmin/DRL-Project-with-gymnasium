# ===============================================================
# Baseline DQN on MiniGrid-DoorKey-8x8-v0
# Implements a Deep Q-Network for the DoorKey environment
# ===============================================================

import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import deque

# --- Device Configuration ---
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"==========================================")
print(f"📌 Using Device: {DEVICE}")
print(f"==========================================")

# --- Reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- Model Architecture ---
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

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, state_dim, size=100000, device=DEVICE):
        self.device = device
        self.obs_buf = np.zeros((size, state_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, state_dim), dtype=np.float32)
        self.acts_buf = np.zeros(size, dtype=np.int64)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=128):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (torch.as_tensor(self.obs_buf[idxs], device=self.device),
                torch.as_tensor(self.acts_buf[idxs], device=self.device).unsqueeze(-1),
                torch.as_tensor(self.rews_buf[idxs], device=self.device).unsqueeze(-1),
                torch.as_tensor(self.next_obs_buf[idxs], device=self.device),
                torch.as_tensor(self.done_buf[idxs], device=self.device).unsqueeze(-1))

# --- DQN Agent ---
class DQNAgent:
    def __init__(self, state_size, action_size, lr=1e-4, gamma=0.99, tau=1e-3, device=DEVICE):
        self.q_local = QNetwork(state_size, action_size).to(device)
        self.q_target = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_local.parameters(), lr=lr)
        self.q_target.load_state_dict(self.q_local.state_dict())
        self.gamma, self.tau, self.device, self.action_size = gamma, tau, device, action_size

    def act(self, state, eps=0.):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.q_local.eval()
        with torch.no_grad(): action_values = self.q_local(state_tensor)
        self.q_local.train()
        if random.random() > eps: return np.argmax(action_values.cpu().data.numpy())
        else: return random.choice(np.arange(self.action_size))

    def update(self, replay, batch_size=128):
        states, actions, rewards, next_states, dones = replay.sample(batch_size)
        Q_targets_next = self.q_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.q_local(states).gather(1, actions)
        loss = nn.functional.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        for tp, lp in zip(self.q_target.parameters(), self.q_local.parameters()):
            tp.data.copy_(self.tau * lp.data + (1.0 - self.tau) * tp.data)
        return loss.item()

# --- Training ---
def train_dqn(env, agent, replay, num_episodes=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.999, batch_size=128):
    scores, losses = [], []
    eps = eps_start
    current_dir = os.path.dirname(os.path.abspath(__file__))

    for ep in range(num_episodes):
        state, _ = env.reset(seed=SEED)
        score = 0
        for t in range(500):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay.store(state, action, reward, next_state, done)
            state, score = next_state, score + reward
            if replay.size > batch_size:
                loss = agent.update(replay, batch_size)
                losses.append(loss)
            if done: break
        
        scores.append(score)
        # Linear epsilon decay: proportional to total episodes
        eps = max(eps_end, eps_start - (eps_start - eps_end) * (ep / num_episodes))
        
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{num_episodes}, Score: {score:.4f}, Avg Score: {np.mean(scores[-50:]):.4f}, Eps: {eps:.4f}")
            # Save results
            pd.DataFrame({"episode": range(1, len(scores)+1), "score": scores}).to_excel(os.path.join(current_dir, "episode_rewards.xlsx"), index=False)
            torch.save(agent.q_local.state_dict(), os.path.join(current_dir, "dqn_model_weights.pth"))
            plt.figure(figsize=(10,5)); plt.plot(scores); plt.savefig(os.path.join(current_dir, "scores.png")); plt.close()

    return scores

if __name__ == "__main__":
    env = gym.make("MiniGrid-DoorKey-8x8-v0")
    env = FlatObsWrapper(env)
    
    agent = DQNAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)
    replay = ReplayBuffer(state_dim=env.observation_space.shape[0])
    
    train_dqn(env, agent, replay, num_episodes=1000)
    print("Training Complete.")
