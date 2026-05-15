import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
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

print(f"==========================================")
print(f"📌 Using Device: {DEVICE}")
print(f"==========================================")

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

def normalize_obs(obs):
    # Use only the image observation, flatten it, and normalize it to 0~1.
    return np.asarray(obs, dtype=np.float32).flatten() / 255.0

def normalize_goal(goal, env):
    goal = np.asarray(goal, dtype=np.float32)
    scale = np.array([env.unwrapped.width - 1, env.unwrapped.height - 1], dtype=np.float32)
    return goal / scale

def get_agent_pos(env):
    return np.array(env.unwrapped.agent_pos, dtype=np.float32)

def train_dqn_her_future(env, agent, replay, episodes=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.999, k_future=4, batch_size=128, start_scores=None):
    scores = start_scores if start_scores is not None else []
    eps = eps_start
    
    start_ep = len(scores)
    total_episodes = start_ep + episodes
    actual_goal_raw = np.array([env.unwrapped.width-2, env.unwrapped.height-2], dtype=np.float32)
    actual_goal = normalize_goal(actual_goal_raw, env)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    for ep in range(start_ep, total_episodes):
        state, _ = env.reset()
        state = normalize_obs(state)
        episode_experience = []
        score = 0
        for t in range(500):
            action = agent.act(state, actual_goal, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = normalize_obs(next_state)
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
                    future_goal_raw = episode_experience[f_idx]['pos']
                    dist = np.linalg.norm(episode_experience[t]['pos'] - future_goal_raw)
                    new_reward = 1.0 if dist < 0.1 else 0.0
                    new_done = True if new_reward == 1.0 else episode_experience[t]['d']
                    future_goal = normalize_goal(future_goal_raw, env)
                    replay.store(episode_experience[t]['s'], future_goal, episode_experience[t]['a'], new_reward, episode_experience[t]['s_next'], future_goal, new_done)

        scores.append(score)
        # Linear epsilon decay: proportional to total episodes
        eps = max(eps_end, eps_start - (eps_start - eps_end) * (ep / total_episodes))
        
        progress = (ep + 1) / total_episodes * 100
        remaining = 100 - progress
        print(f"Episode {ep+1}/{total_episodes}, Score: {score:.4f}, Epsilon: {eps:.4f}, Progress: {progress:.2f}%, Remaining: {remaining:.2f}%")
        
        if (ep + 1) % 10 == 0:
            excel_path = os.path.join(current_dir, "episode_rewards_her.xlsx")
            image_path = os.path.join(current_dir, "scores_her.png")
            pd.DataFrame({"episode": range(1, len(scores)+1), "score": scores}).to_excel(excel_path, index=False)
            
            # Plotting with Smoothing
            plt.figure(figsize=(10,5))
            plt.plot(scores, alpha=0.3, color='blue', label='Raw Reward')
            if len(scores) >= 10:
                smooth_scores = pd.Series(scores).rolling(window=10).mean()
                plt.plot(smooth_scores, color='orange', linewidth=2, label='Smoothed Reward (MA 10)')
            plt.title('DQN HER Training Scores')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            plt.legend()
            plt.savefig(image_path)
            plt.close()
            
            model_path = os.path.join(current_dir, "dqn_her_model.pth")
            torch.save(agent.q_local.state_dict(), model_path)
            
    return scores

if __name__ == "__main__":
    env = gym.make("MiniGrid-DoorKey-8x8-v0"); env = ImgObsWrapper(env)
    
    obs_dim = int(np.prod(env.observation_space.shape))
    agent = DQNAgent(state_size=obs_dim + 2, action_size=env.action_space.n)
    replay = HERReplayBuffer(obs_dim, 2)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "dqn_her_model.pth")
    excel_path = os.path.join(current_dir, "episode_rewards_her.xlsx")

    start_scores = []

    print(f"Starting training on {DEVICE} from scratch for 1000 episodes with initial eps=1.0...")
    train_dqn_her_future(env, agent, replay, episodes=1000, eps_start=1.0, start_scores=start_scores)
    
    torch.save(agent.q_local.state_dict(), model_path)
    print(f"Final model saved to: {model_path}")
