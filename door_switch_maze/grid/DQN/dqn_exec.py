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
from model import QNetwork

# GPU selection (supporting MPS for Mac, CUDA for Nvidia, or CPU)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

class ReplayBuffer:
    def __init__(self, obs_dim, size=100000, device=DEVICE):
        self.device = device
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros(size, dtype=np.int64)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_size = size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=128):
        idxs = np.random.randint(0, self.size, size=batch_size)
        obs = torch.as_tensor(self.obs_buf[idxs], dtype=torch.float32, device=self.device)
        acts = torch.as_tensor(self.acts_buf[idxs], dtype=torch.int64, device=self.device).unsqueeze(-1)
        rews = torch.as_tensor(self.rews_buf[idxs], dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_obs = torch.as_tensor(self.next_obs_buf[idxs], dtype=torch.float32, device=self.device)
        done = torch.as_tensor(self.done_buf[idxs], dtype=torch.float32, device=self.device).unsqueeze(-1)
        return obs, acts, rews, next_obs, done

class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        lr=1e-4,
        gamma=0.99,
        tau=1e-3,
        device=DEVICE
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.q_local = QNetwork(state_size, action_size).to(device)
        self.q_target = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_local.parameters(), lr=lr)

        self.q_target.load_state_dict(self.q_local.state_dict())

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.q_local.eval()
        with torch.no_grad():
            action_values = self.q_local(state)
        self.q_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def update(self, replay, batch_size=128):
        states, actions, rewards, next_states, dones = replay.sample(batch_size)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.q_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.q_local(states).gather(1, actions)

        # Compute loss
        loss = nn.functional.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network
        self.soft_update(self.q_local, self.q_target)
        
        return loss.item()

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

def train_dqn(env, agent, replay, episodes=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, batch_size=128):
    scores = []
    eps = eps_start
    
    for ep in range(episodes):
        state, _ = env.reset()
        score = 0
        for t in range(500): # max steps per episode
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay.store(state, action, reward, next_state, done)
            state = next_state
            score += reward
            
            if replay.size > batch_size:
                agent.update(replay, batch_size)
            
            if done:
                break
        
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)
        
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{episodes}, Average Score: {np.mean(scores[-10:]):.4f}, Epsilon: {eps:.4f}")
            
            # Save results progressively
            df = pd.DataFrame({"episode": range(1, len(scores)+1), "score": scores})
            df.to_excel("episode_rewards.xlsx", index=False)

    return scores

if __name__ == "__main__":
    env = gym.make("MiniGrid-DoorKey-8x8-v0")
    env = FlatObsWrapper(env)
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    replay = ReplayBuffer(state_size, size=100000)
    
    print(f"Starting training on {DEVICE}...")
    scores = train_dqn(env, agent, replay, episodes=500)
    
    # Save model
    torch.save(agent.q_local.state_dict(), "dqn_model.pth")
    print("Training finished. Model saved.")
    
    # Plotting
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('scores.png')
