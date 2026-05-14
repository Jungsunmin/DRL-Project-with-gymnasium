# ===============================================================
# Baseline DQN on MiniGrid-DoorKey-8x8-v0
# Implements a Deep Q-Network for the DoorKey environment
# Authors: Yash Kini, Shiv Davay, Shreya Polavarapu
# ===============================================================

# --- Imports ---
import gymnasium as gym
import minigrid  # Required for MiniGrid environments
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# ===============================================================
# Reproducibility
# Ensures consistent results across runs
# ===============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ===============================================================
# Preprocessing Function
# Flattens and normalizes MiniGrid image observations
# ===============================================================
def preprocess_obs(obs):
    img = obs["image"]  # Shape: (7, 7, 3)
    return img.flatten().astype(np.float32) / 255.0  # Normalize to [0, 1]

# ===============================================================
# Vanilla DQN Model Architecture
# Maps preprocessed state to Q-values for each action
# ===============================================================
class DQN(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),   # Optional normalization for stability
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, action_dim)  # Output layer for Q-values
        )
    def forward(self, x):
        return self.net(x)

# ===============================================================
# Epsilon-Greedy Action Selection
# ===============================================================
def epsilon_greedy(model, state, epsilon, action_space):
    """With probability epsilon, take a random action; otherwise, pick the best one."""
    if random.random() < epsilon:
        return random.randrange(action_space)
    with torch.no_grad():
        q_vals = model(state)
        return int(torch.argmax(q_vals).item())

# Soft update to slowly blend target network toward current network
def soft_update(target, source, tau=0.005):
    for tp, p in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

# ===============================================================
# Training Loop
# Core DQN training logic using experience replay and target network
# ===============================================================
def train_dqn(env, model, target_model, optimizer,
              num_episodes=700, gamma=0.99,
              epsilon_start=1.0, epsilon_min=0.1, batch_size=128):
    replay_buffer = deque(maxlen=50000)
    all_rewards, losses = [], []
    epsilon = epsilon_start

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=SEED)
        state = torch.FloatTensor(preprocess_obs(obs)).unsqueeze(0)
        done, total_reward = False, 0

        while not done:
            # Choose action and step through environment
            action = epsilon_greedy(model, state, epsilon, env.action_space.n)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            next_state = torch.FloatTensor(preprocess_obs(next_obs)).unsqueeze(0)
            # Store transition
            replay_buffer.append((state.detach(), action, reward, next_state.detach(), float(done)))
            state = next_state

            # Only train once buffer has enough data
            if len(replay_buffer) >= 5000:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards_b, next_states, dones = zip(*batch)

                # Batch formatting
                states = torch.cat(states)
                next_states = torch.cat(next_states)
                actions = torch.LongTensor(actions)
                rewards_b = torch.FloatTensor(rewards_b)
                dones = torch.FloatTensor(dones)

                # Compute targets using Double DQN logic
                with torch.no_grad():
                    next_q = model(next_states)
                    next_act = torch.argmax(next_q, dim=1)
                    next_q_target = target_model(next_states)
                    target_vals = next_q_target.gather(1, next_act.view(-1, 1)).squeeze(1)
                    targets = rewards_b + gamma * target_vals * (1 - dones)

                # Compute current Q-values and loss
                q_vals = model(states).gather(1, actions.view(-1, 1)).squeeze(1)
                loss = (q_vals - targets).pow(2).mean()

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
                optimizer.step()
                losses.append(loss.item())

                # Slowly update target network
                soft_update(target_model, model, tau=0.005)

        all_rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon * 0.995)  # Decay epsilon over time

        # Logging
        if ep % 10 == 0:
            mean_reward = np.mean(all_rewards[-50:]) if len(all_rewards) >= 50 else np.mean(all_rewards)
            mean_loss = np.mean(losses[-100:]) if losses else 0.0
            print(f"[Ep {ep:03d}] Reward: {total_reward:.2f} | Mean(50): {mean_reward:.2f} | "
                  f"Eps: {epsilon:.3f} | Loss: {mean_loss:.4f}")

    return all_rewards, losses

# ===============================================================
# Main
# Sets up environment, trains model, evaluates, and visualizes
# ===============================================================
def main():
    env = gym.make("MiniGrid-DoorKey-8x8-v0")  # Key-door sparse reward task
    input_dim = 7 * 7 * 3  # Flattened image size
    action_dim = env.action_space.n

    # Instantiate model and target model
    model = DQN(input_dim, action_dim)
    target_model = DQN(input_dim, action_dim)
    target_model.load_state_dict(model.state_dict())  # Synchronize weights

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train model
    rewards, losses = train_dqn(env, model, target_model, optimizer)

    # --- Evaluation Metrics ---
    final_window = 50
    mean_final_reward = np.mean(rewards[-final_window:])
    success_threshold = 0.8
    episodes_to_threshold = next((i+1 for i, r in enumerate(rewards) if r >= success_threshold), len(rewards))
    loss_variance = float(np.var(losses)) if losses else 0.0
    reward_std = float(np.std(rewards))
    auc_reward = float(np.trapezoid(rewards, dx=1))

    # Output metrics
    print("===== BASELINE DQN METRICS =====")
    print(f"Mean reward in final {final_window} episodes: {mean_final_reward:.2f}")
    print(f"Episodes to first success (>{success_threshold}): {episodes_to_threshold}")
    print(f"Loss variance: {loss_variance:.6f}")
    print(f"Reward standard deviation: {reward_std:.2f}")
    print(f"AUC: {auc_reward:.2f}")
    print("================================")

    # --- Plotting ---
    def smooth(data, w=0.9):
        """Exponential moving average for smoothing curves."""
        sm, last = [], data[0]
        for x in data:
            last = last * w + (1 - w) * x
            sm.append(last)
        return sm

    plt.figure(figsize=(12, 5))

    # Reward plot
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Raw Reward", alpha=0.4)
    plt.plot(smooth(rewards), label="Smoothed Reward")
    plt.title("Baseline DQN: Rewards Over Time")
    plt.xlabel("Episode"); plt.ylabel("Total Reward"); plt.legend(); plt.grid(alpha=0.3)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(losses, label="Mini-Batch Loss")
    plt.title("Baseline DQN: Loss During Training")
    plt.xlabel("Training Step"); plt.ylabel("Loss"); plt.legend(); plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("Vanilla_DQN_MiniGrid_DoorKey_results.png", dpi=300, bbox_inches='tight')
    
    # --- Save Model Weights ---
    torch.save(model.state_dict(), "dqn_model_weights.pth")
    print("Model weights saved as 'dqn_model_weights.pth'")
    
    plt.show()
    print("Plot saved as 'Vanilla_DQN_MiniGrid_DoorKey_results.png'")

# Entrypoint
if __name__ == "__main__":
    main()
