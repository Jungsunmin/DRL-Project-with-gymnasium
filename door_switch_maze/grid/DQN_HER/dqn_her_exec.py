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

# GPU 가속 설정
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

class HERReplayBuffer:
    def __init__(self, obs_dim, goal_dim, size=100000, device=DEVICE):
        self.device = device
        self.obs_buf = np.zeros((size, obs_dim + goal_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim + goal_dim), dtype=np.float32)
        self.acts_buf = np.zeros(size, dtype=np.int64)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_size = size

    def store(self, obs, goal, act, rew, next_obs, next_goal, done):
        self.obs_buf[self.ptr] = np.hstack([obs, goal])
        self.next_obs_buf[self.ptr] = np.hstack([next_obs, next_goal])
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
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
    def __init__(self, state_size, action_size, lr=1e-4, gamma=0.99, tau=1e-3, device=DEVICE):
        self.q_local = QNetwork(state_size, action_size).to(device)
        self.q_target = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_local.parameters(), lr=lr)
        self.q_target.load_state_dict(self.q_local.state_dict())
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.action_size = action_size

    def act(self, state, goal, eps=0.):
        combined_state = np.hstack([state, goal])
        state_tensor = torch.from_numpy(combined_state).float().unsqueeze(0).to(self.device)
        self.q_local.eval()
        with torch.no_grad():
            action_values = self.q_local(state_tensor)
        self.q_local.train()
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def update(self, replay, batch_size=128):
        states, actions, rewards, next_states, dones = replay.sample(batch_size)
        Q_targets_next = self.q_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.q_local(states).gather(1, actions)
        loss = nn.functional.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        for target_param, local_param in zip(self.q_target.parameters(), self.q_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        return loss.item()

def get_agent_pos(env):
    return np.array(env.unwrapped.agent_pos, dtype=np.float32)

def train_dqn_her_future(env, agent, replay, episodes=500, k_future=4, batch_size=128):
    scores = []
    eps = 1.0
    eps_decay = 0.995
    eps_min = 0.01
    actual_goal = np.array([env.unwrapped.width-2, env.unwrapped.height-2], dtype=np.float32)

    for ep in range(episodes):
        state, _ = env.reset()
        episode_experience = []
        score = 0
        for t in range(500):
            action = agent.act(state, actual_goal, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_experience.append({
                's': state, 'a': action, 'r': reward, 's_next': next_state, 
                'd': done, 'pos': get_agent_pos(env)
            })
            state = next_state
            score += reward
            if replay.size > batch_size:
                agent.update(replay, batch_size)
            if done: break

        for exp in episode_experience:
            replay.store(exp['s'], actual_goal, exp['a'], exp['r'], exp['s_next'], actual_goal, exp['d'])
        for t in range(len(episode_experience)):
            if t < len(episode_experience) - 1:
                future_indices = random.sample(range(t, len(episode_experience)), min(k_future, len(episode_experience) - t))
                for f_idx in future_indices:
                    future_goal = episode_experience[f_idx]['pos']
                    dist = np.linalg.norm(episode_experience[t]['pos'] - future_goal)
                    new_reward = 1.0 if dist < 0.1 else 0.0
                    new_done = True if new_reward == 1.0 else episode_experience[t]['d']
                    replay.store(episode_experience[t]['s'], future_goal, episode_experience[t]['a'], new_reward, episode_experience[t]['s_next'], future_goal, new_done)

        scores.append(score)
        eps = max(eps_min, eps * eps_decay)
        print(f"Episode {ep+1}/{episodes}, Score: {score:.4f}, Epsilon: {eps:.4f}")
        if (ep + 1) % 10 == 0:
            pd.DataFrame({"episode": range(1, len(scores)+1), "score": scores}).to_excel("episode_rewards_her.xlsx", index=False)
            plt.figure(figsize=(10,5)); plt.plot(scores); plt.savefig('scores_her.png'); plt.close()
    return scores

if __name__ == "__main__":
    env = gym.make("MiniGrid-DoorKey-8x8-v0"); env = FlatObsWrapper(env)
    agent = DQNAgent(state_size=env.observation_space.shape[0] + 2, action_size=env.action_space.n)
    replay = HERReplayBuffer(env.observation_space.shape[0], 2)
    train_dqn_her_future(env, agent, replay, episodes=500)
    torch.save(agent.q_local.state_dict(), "dqn_her_model.pth")
