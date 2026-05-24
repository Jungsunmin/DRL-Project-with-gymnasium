import argparse
import os
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import minigrid  # noqa: F401
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from minigrid.wrappers import ImgObsWrapper

# GPU selection (supporting MPS for Mac, CUDA for Nvidia, or CPU)
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

print("==========================================")
print(f"📌 Using Device: {DEVICE}")
print("==========================================")


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
        )

    def forward(self, x):
        return self.net(x)


class HERReplayBuffer:
    """Goal-conditioned replay buffer (obs + goal concat)."""

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


def compute_recent_weighted_reward(recent_rewards, window_size=5, decay=0.7):
    """
    최근 env 보상에 decay 가중 평균을 적용합니다.
    HER relabel transition의 replay reward로 사용합니다.
    """
    weighted_sum = 0.0
    weight_sum = 0.0
    weight = 1.0

    for reward in reversed(recent_rewards[-window_size:]):
        weighted_sum += weight * float(reward)
        weight_sum += weight
        weight *= decay

    if weight_sum <= 0.0:
        return 0.0

    return weighted_sum / weight_sum


class DDQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        lr=1e-4,
        gamma=0.99,
        tau=1e-3,
        device=DEVICE,
    ):
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.q_local = QNetwork(state_size, action_size).to(device)
        self.q_target = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_local.parameters(), lr=lr)
        self.q_target.load_state_dict(self.q_local.state_dict())

    def act(self, state, goal, eps=0.0):
        combined_state = np.hstack([state, goal])
        state_tensor = torch.from_numpy(combined_state).float().unsqueeze(0).to(self.device)

        self.q_local.eval()
        with torch.no_grad():
            action_values = self.q_local(state_tensor)
        self.q_local.train()

        if random.random() > eps:
            return int(np.argmax(action_values.cpu().data.numpy()))
        return int(random.choice(np.arange(self.action_size)))

    def update(self, replay, batch_size=128):
        states, actions, rewards, next_states, dones = replay.sample(batch_size)

        # Double DQN: local network에서 next action 선택
        self.q_local.eval()
        with torch.no_grad():
            next_actions = self.q_local(next_states).max(1)[1].unsqueeze(1)
        self.q_local.train()

        # Target network에서 해당 action의 Q값 평가
        q_targets_next = self.q_target(next_states).gather(1, next_actions).detach()
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        q_expected = self.q_local(states).gather(1, actions)

        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.q_local, self.q_target)
        return float(loss.item())

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


def normalize_obs(obs):
    return np.asarray(obs, dtype=np.float32).flatten() / 255.0


def normalize_goal(goal, env):
    goal = np.asarray(goal, dtype=np.float32)
    scale = np.array([env.unwrapped.width - 1, env.unwrapped.height - 1], dtype=np.float32)
    return goal / scale


def get_agent_pos(env):
    return np.array(env.unwrapped.agent_pos, dtype=np.float32)


def train_ddqn_her_reward_shaping(
    env,
    agent,
    replay,
    episodes=1000,
    eps_start=1.0,
    eps_end=0.01,
    batch_size=128,
    start_scores=None,
    k_future=4,
    reward_window=5,
    reward_decay=0.7,
):
    scores = start_scores if start_scores is not None else []
    her_weighted_reward_sums = []
    eps = eps_start

    start_ep = len(scores)
    total_episodes = start_ep + episodes
    current_dir = os.path.dirname(os.path.abspath(__file__))

    actual_goal_raw = np.array([env.unwrapped.width - 2, env.unwrapped.height - 2], dtype=np.float32)
    actual_goal = normalize_goal(actual_goal_raw, env)

    for ep in range(start_ep, total_episodes):
        state, _ = env.reset()
        state = normalize_obs(state)

        score = 0.0
        her_weighted_sum = 0.0
        recent_rewards = []
        episode_experience = []

        for _ in range(500):
            action = agent.act(state, actual_goal, eps)

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = normalize_obs(next_state)
            done = terminated or truncated

            recent_rewards.append(float(reward))
            weighted_r = compute_recent_weighted_reward(
                recent_rewards,
                window_size=reward_window,
                decay=reward_decay,
            )

            episode_experience.append({
                "s": state,
                "a": action,
                "r": float(reward),
                "s_next": next_state,
                "d": done,
                "pos": get_agent_pos(env),
                "weighted_r": weighted_r,
            })

            state = next_state
            score += float(reward)

            if replay.size > batch_size:
                agent.update(replay, batch_size)

            if done:
                break

        # Actual goal transition: 환경 보상 그대로 replay 저장
        for exp in episode_experience:
            replay.store(
                exp["s"],
                actual_goal,
                exp["a"],
                exp["r"],
                exp["s_next"],
                actual_goal,
                exp["d"],
            )

        # HER future goal selection: relabel transition에는 shaped reward 사용
        for t in range(len(episode_experience)):
            if t < len(episode_experience) - 1:
                future_indices = random.sample(
                    range(t, len(episode_experience)),
                    min(k_future, len(episode_experience) - t),
                )
                for f_idx in future_indices:
                    future_goal_raw = episode_experience[f_idx]["pos"]
                    dist = np.linalg.norm(episode_experience[t]["pos"] - future_goal_raw)
                    reached = dist < 0.1
                    new_done = True if reached else episode_experience[t]["d"]
                    future_goal = normalize_goal(future_goal_raw, env)
                    shaped_reward = episode_experience[t]["weighted_r"]
                    her_weighted_sum += float(shaped_reward)

                    replay.store(
                        episode_experience[t]["s"],
                        future_goal,
                        episode_experience[t]["a"],
                        shaped_reward,
                        episode_experience[t]["s_next"],
                        future_goal,
                        new_done,
                    )

        scores.append(score)
        her_weighted_reward_sums.append(her_weighted_sum)

        eps = max(
            eps_end,
            eps_start - (eps_start - eps_end) * (ep / max(1, total_episodes)),
        )

        progress = (ep + 1) / total_episodes * 100
        remaining = 100 - progress
        print(
            f"Episode {ep + 1}/{total_episodes}, "
            f"Env Score: {score:.4f}, "
            f"HER Weighted Reward Sum: {her_weighted_sum:.4f}, "
            f"Average Score: {np.mean(scores[-10:]):.4f}, "
            f"Epsilon: {eps:.4f}, "
            f"Progress: {progress:.2f}%, Remaining: {remaining:.2f}%"
        )

        if (ep + 1) % 10 == 0:
            df = pd.DataFrame({
                "episode": range(1, len(scores) + 1),
                "env_score": scores,
                "her_weighted_reward_sum": her_weighted_reward_sums,
            })
            df.to_excel(
                os.path.join(current_dir, "episode_rewards_ddqn_her_reward.xlsx"),
                index=False,
            )

            torch.save(
                agent.q_local.state_dict(),
                os.path.join(current_dir, "ddqn_her_reward_model.pth"),
            )

            plt.figure(figsize=(10, 5))
            plt.plot(scores, alpha=0.3, color="blue", label="Raw Environment Reward")
            if len(scores) >= 10:
                smooth_scores = pd.Series(scores).rolling(window=10).mean()
                plt.plot(
                    smooth_scores,
                    color="orange",
                    linewidth=2,
                    label="Smoothed Reward (MA 10)",
                )
            plt.title("DDQN HER Reward Shaping Training Scores")
            plt.xlabel("Episode")
            plt.ylabel("Score")
            plt.legend()
            plt.savefig(os.path.join(current_dir, "scores_ddqn_her_reward.png"))
            plt.close()

    df = pd.DataFrame({
        "episode": range(1, len(scores) + 1),
        "env_score": scores,
        "her_weighted_reward_sum": her_weighted_reward_sums,
    })
    df.to_excel(
        os.path.join(current_dir, "episode_rewards_ddqn_her_reward.xlsx"),
        index=False,
    )

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DDQN + HER(future) with reward shaping on HER transitions only."
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.01)
    parser.add_argument("--k_future", type=int, default=4)
    parser.add_argument("--reward_window", type=int, default=5)
    parser.add_argument("--reward_decay", type=float, default=0.7)
    args = parser.parse_args()

    env = gym.make("MiniGrid-DoorKey-8x8-v0")
    env = ImgObsWrapper(env)

    obs_dim = int(np.prod(env.observation_space.shape))
    goal_dim = 2
    action_dim = int(env.action_space.n)

    agent = DDQNAgent(state_size=obs_dim + goal_dim, action_size=action_dim)
    replay = HERReplayBuffer(obs_dim=obs_dim, goal_dim=goal_dim, size=100000)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "ddqn_her_reward_model.pth")

    print(
        f"Starting DDQN+HER reward shaping on {DEVICE} "
        f"for {args.episodes} episodes (reward_window={args.reward_window})..."
    )

    scores = train_ddqn_her_reward_shaping(
        env=env,
        agent=agent,
        replay=replay,
        episodes=args.episodes,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        start_scores=[],
        k_future=args.k_future,
        reward_window=args.reward_window,
        reward_decay=args.reward_decay,
    )

    torch.save(agent.q_local.state_dict(), model_path)
    print(f"Training finished. Model saved to {model_path}")
