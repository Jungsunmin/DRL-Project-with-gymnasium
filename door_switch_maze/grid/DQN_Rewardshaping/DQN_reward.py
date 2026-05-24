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

# GPU selection (supporting MPS for Mac, CUDA for Nvidia, or CPU)
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

print(f"==========================================")
print(f"📌 Using Device: {DEVICE}")
print(f"==========================================")


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )

    def forward(self, x):
        return self.net(x)


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

        obs = torch.as_tensor(
            self.obs_buf[idxs],
            dtype=torch.float32,
            device=self.device
        )

        acts = torch.as_tensor(
            self.acts_buf[idxs],
            dtype=torch.int64,
            device=self.device
        ).unsqueeze(-1)

        rews = torch.as_tensor(
            self.rews_buf[idxs],
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(-1)

        next_obs = torch.as_tensor(
            self.next_obs_buf[idxs],
            dtype=torch.float32,
            device=self.device
        )

        done = torch.as_tensor(
            self.done_buf[idxs],
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(-1)

        return obs, acts, rews, next_obs, done


def compute_recent_weighted_reward(recent_rewards, window_size=10, decay=0.9):
    """
    Computes a weighted sum of the current reward and previous rewards.

    The most recent reward gets the largest weight. Older rewards up to
    `window_size` steps back get exponentially smaller weights.

    Example with decay=0.9:
        r_t * 1.0 + r_{t-1} * 0.9 + r_{t-2} * 0.9^2 + ...
    """
    weighted_reward = 0.0
    weight = 1.0

    for reward in reversed(recent_rewards[-window_size:]):
        weighted_reward += weight * float(reward)
        weight *= decay

    return weighted_reward


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

        # Get max predicted Q values for next states from target model
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
        for target_param, local_param in zip(
            target_model.parameters(),
            local_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data
                + (1.0 - self.tau) * target_param.data
            )


def normalize_obs(obs):
    # Use only the image observation, flatten it, and normalize it to 0~1.
    return np.asarray(obs, dtype=np.float32).flatten() / 255.0


def train_dqn(
    env,
    agent,
    replay,
    episodes=1000,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.999,
    batch_size=128,
    start_scores=None,
    reward_window=10,
    reward_decay=0.9
):
    scores = start_scores if start_scores is not None else []
    eps = eps_start
    weighted_reward_sums = []

    start_ep = len(scores)
    total_episodes = start_ep + episodes

    current_dir = os.path.dirname(os.path.abspath(__file__))

    for ep in range(start_ep, total_episodes):
        state, _ = env.reset()
        state = normalize_obs(state)

        score = 0.0
        weighted_reward_sum = 0.0
        recent_rewards = []

        for t in range(500):  # max steps per episode
            action = agent.act(state, eps)

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = normalize_obs(next_state)
            done = terminated or truncated

            recent_rewards.append(float(reward))
            weighted_reward = compute_recent_weighted_reward(
                recent_rewards,
                window_size=reward_window,
                decay=reward_decay
            )

            replay.store(state, action, weighted_reward, next_state, done)

            state = next_state
            score += float(reward)
            weighted_reward_sum += float(weighted_reward)

            if replay.size > batch_size:
                agent.update(replay, batch_size)

            if done:
                break

        scores.append(score)
        weighted_reward_sums.append(weighted_reward_sum)

        # Linear epsilon decay: proportional to total episodes
        eps = max(
            eps_end,
            eps_start - (eps_start - eps_end) * (ep / total_episodes)
        )

        progress = (ep + 1) / total_episodes * 100
        remaining = 100 - progress

        print(
            f"Episode {ep+1}/{total_episodes}, "
            f"Env Score: {score:.4f}, "
            f"Weighted Reward Sum: {weighted_reward_sum:.4f}, "
            f"Average Score: {np.mean(scores[-10:]):.4f}, "
            f"Epsilon: {eps:.4f}, "
            f"Progress: {progress:.2f}%, Remaining: {remaining:.2f}%"
        )

        if (ep + 1) % 10 == 0:
            # Save results progressively
            df = pd.DataFrame({
                "episode": range(1, len(scores) + 1),
                "env_score": scores,
                "weighted_reward_sum": weighted_reward_sums
            })

            df.to_excel(
                os.path.join(current_dir, "episode_rewards.xlsx"),
                index=False
            )

            # Save model periodically
            torch.save(
                agent.q_local.state_dict(),
                os.path.join(current_dir, "dqn_model.pth")
            )

            # Plotting with Smoothing
            plt.figure(figsize=(10, 5))
            plt.plot(
                scores,
                alpha=0.3,
                color="blue",
                label="Raw Environment Reward"
            )

            if len(scores) >= 10:
                smooth_scores = pd.Series(scores).rolling(window=10).mean()
                plt.plot(
                    smooth_scores,
                    color="orange",
                    linewidth=2,
                    label="Smoothed Reward (MA 10)"
                )

            plt.title("DQN Training Scores")
            plt.xlabel("Episode")
            plt.ylabel("Score")
            plt.legend()
            plt.savefig(os.path.join(current_dir, "scores.png"))
            plt.close()

    df = pd.DataFrame({
        "episode": range(1, len(scores) + 1),
        "env_score": scores,
        "weighted_reward_sum": weighted_reward_sums
    })

    df.to_excel(
        os.path.join(current_dir, "episode_rewards.xlsx"),
        index=False
    )

    return scores


if __name__ == "__main__":
    env = gym.make("MiniGrid-DoorKey-8x8-v0")
    env = ImgObsWrapper(env)

    state_size = int(np.prod(env.observation_space.shape))
    action_size = env.action_space.n

    agent = DQNAgent(state_size=state_size, action_size=action_size)
    replay = ReplayBuffer(state_size, size=100000)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "dqn_model.pth")
    excel_path = os.path.join(current_dir, "episode_rewards.xlsx")

    start_scores = []

    print(
        f"Starting training on {DEVICE} from scratch for 1000 episodes "
        f"with initial eps=1.0..."
    )

    scores = train_dqn(
        env,
        agent,
        replay,
        episodes=1000,
        eps_start=1.0,
        start_scores=start_scores,
        reward_window=10,
        reward_decay=0.9
    )

    # Final save
    torch.save(agent.q_local.state_dict(), model_path)

    print("Training finished. Model saved.")