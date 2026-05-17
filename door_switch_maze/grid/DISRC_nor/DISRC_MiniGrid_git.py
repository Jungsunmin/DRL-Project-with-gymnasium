# ================================================================
# DISRC DQN on MiniGrid-DoorKey-8x8-v0
# Implements a Deep Q-Network with Deep Intrinsic Surprise-Regularized Control
# Authors: Yash Kini, Shiv Davay, Shreya Polavarapu
# ================================================================

# Imports
import gymnasium as gym
import minigrid  # Required to register MiniGrid environments
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

# ================================================================
# Reproducibility
# Set seeds for reproducibility across NumPy, random, and PyTorch
# ================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ================================================================
# Preprocessing for MiniGrid
# Converts MiniGrid's image observations into flat, normalized vectors
# ================================================================
def preprocess_obs(obs):
    """
    Extract and flatten the image part of the observation, normalizing to [0,1].
    MiniGrid observations contain both an image and a mission string.
    """
    img = obs['image']  # Shape: (7, 7, 3)
    flat = img.flatten().astype(np.float32) / 255.0
    return flat  # Shape: (147,)

# ================================================================
# Utility Functions
# Includes tensor formatting and weight initialization
# ================================================================
def ensure_2d_tensor(tensor):
    """Ensures tensor is 2D for batch processing."""
    if isinstance(tensor, np.ndarray):
        tensor = torch.FloatTensor(tensor)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

def init_weights(m):
    """Applies Xavier initialization to linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

# ================================================================
# DISRC Controller
# Computes surprise-based intrinsic bonus for each encoded state
# ================================================================
class DISRCController:
    def __init__(self, state_dim, alpha=0.03, beta_start=0.04):
        self.alpha = alpha  # Moving average rate
        self.beta_start = beta_start  # Initial strength of surprise scaling
        self.setpoint = np.zeros(state_dim, dtype=np.float32)  # Latent running average

    def compute_bonus(self, encoded_state, episode_ratio):
        """
        Computes a deviation-based bonus by comparing the current encoded
        state to a running latent setpoint.
        """
        s_np = encoded_state.detach().cpu().numpy().squeeze()
        s_norm = s_np / (np.linalg.norm(s_np) + 1e-8)
        sp_norm = self.setpoint / (np.linalg.norm(self.setpoint) + 1e-8)
        deviation = np.linalg.norm(s_norm - sp_norm)
        beta = self.beta_start * (1.0 - episode_ratio ** 1.2)  # Decays over time
        bonus = -beta * deviation
        self.setpoint = (1.0 - self.alpha) * self.setpoint + self.alpha * s_np
        return bonus

# ================================================================
# Neural Network Architectures
# Encoder maps raw input to latent space; DQN predicts Q-values
# ================================================================
class DISRCStateEncoder(nn.Module):
    def __init__(self, input_dim, encoded_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, encoded_dim)
        )
    def forward(self, x):
        return self.encoder(x)

class DISRC_DQN(nn.Module):
    def __init__(self, input_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )
    def forward(self, x):
        return self.net(x)

# ================================================================
# Action Selection and Target Update
# Epsilon-greedy policy and soft target model updates
# ================================================================
def epsilon_greedy(model, state, epsilon, action_space):
    """Chooses a random action with probability epsilon, else greedy."""
    if random.random() < epsilon:
        return random.randrange(action_space)
    with torch.no_grad():
        q_values = model(state)
        return int(torch.argmax(q_values).item())

def soft_update(target, source, tau=0.005):
    """Smooth update of target network parameters."""
    for tp, p in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

# ================================================================
# Training Loop
# Full training logic including experience replay and DISRC shaping
# ================================================================
def train_dqn(env, model, target_model, encoder,
              encoder_optimizer, model_optimizer,
              disrc_controller,
              num_episodes=700, gamma=0.99,
              epsilon_start=1.0, epsilon_min=0.1,
              batch_size=128):
    replay_buffer = deque(maxlen=50000)
    all_rewards, losses = [], []
    epsilon = epsilon_start
    reward_norm = 1.0  # For normalizing rewards

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=SEED)
        state_arr = preprocess_obs(obs)
        state_tensor = torch.FloatTensor(state_arr).unsqueeze(0)
        done, total_reward = False, 0

        while not done:
            encoded_state = encoder(state_tensor)
            action = epsilon_greedy(model, encoded_state, epsilon, env.action_space.n)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            next_state_arr = preprocess_obs(next_obs)
            next_tensor = torch.FloatTensor(next_state_arr).unsqueeze(0)

            # Reward shaping with DISRC bonus
            reward_norm = 0.99 * reward_norm + 0.01 * abs(reward)
            normed_reward = reward / (reward_norm + 1e-8)
            bonus = disrc_controller.compute_bonus(encoded_state, episode / num_episodes)
            shaped_reward = np.clip(normed_reward + 0.2 * bonus, -1.0, 1.0)

            replay_buffer.append((state_tensor.detach(), action, shaped_reward,
                                  next_tensor.detach(), float(done)))
            state_tensor = next_tensor

            # Training step
            if len(replay_buffer) >= 5000:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards_b, next_states, dones = zip(*batch)
                states = torch.cat([ensure_2d_tensor(s) for s in states])
                next_states = torch.cat([ensure_2d_tensor(ns) for ns in next_states])
                actions = torch.LongTensor(actions)
                rewards_b = torch.FloatTensor(rewards_b)
                dones = torch.FloatTensor(dones)

                s_enc = encoder(states)
                ns_enc = encoder(next_states)

                with torch.no_grad():
                    next_q = model(ns_enc)
                    next_act = torch.argmax(next_q, dim=1)
                    next_q_target = target_model(ns_enc)
                    target_vals = next_q_target.gather(1, next_act.view(-1, 1)).squeeze(1)
                    targets = rewards_b + gamma * target_vals * (1 - dones)

                q_vals = model(s_enc).gather(1, actions.view(-1, 1)).squeeze(1)
                loss = (q_vals - targets).pow(2).mean()

                # Backpropagation
                model_optimizer.zero_grad()
                encoder_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
                model_optimizer.step()
                encoder_optimizer.step()
                losses.append(loss.item())

                soft_update(target_model, model, tau=0.005)

        all_rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon * 0.995)

        # Logging
        if episode % 10 == 0:
            mean_reward = np.mean(all_rewards[-50:]) if len(all_rewards) >= 50 else np.mean(all_rewards)
            mean_loss = np.mean(losses[-100:]) if losses else 0.0
            print(f"[Ep {episode:03d}] Reward: {total_reward:.2f} | Mean(50): {mean_reward:.2f} | "
                  f"Eps: {epsilon:.3f} | Loss: {mean_loss:.4f}")

    return all_rewards, losses

# ================================================================
# Main Execution
# Instantiates models, runs training, logs metrics, and plots results
# ================================================================
def main():
    env = gym.make("MiniGrid-DoorKey-8x8-v0")  # Sparse-reward environment
    input_dim = 7 * 7 * 3  # Flattened observation
    action_dim = env.action_space.n

    # Initialize networks
    encoder = DISRCStateEncoder(input_dim=input_dim, encoded_dim=64)
    model = DISRC_DQN(input_size=64, action_size=action_dim)
    target_model = DISRC_DQN(input_size=64, action_size=action_dim)
    target_model.load_state_dict(model.state_dict())  # Sync weights

    # Initialize weights and optimizers
    encoder.apply(init_weights)
    model.apply(init_weights)
    target_model.apply(init_weights)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=3e-4)
    model_optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Initialize DISRC controller
    disrc_controller = DISRCController(state_dim=64, alpha=0.03, beta_start=0.04)

    # Train
    rewards, losses = train_dqn(env, model, target_model, encoder,
                                encoder_optimizer, model_optimizer,
                                disrc_controller)

    # Evaluation Metrics
    final_window = 50
    mean_final_reward = np.mean(rewards[-final_window:])
    success_threshold = 0.8
    episodes_to_threshold = next((i+1 for i, r in enumerate(rewards) if r >= success_threshold), len(rewards))
    loss_variance = float(np.var(losses)) if losses else 0.0
    reward_std = float(np.std(rewards))
    auc_reward = float(np.trapezoid(rewards, dx=1))

    # Log metrics
    print("===== METRICS =====")
    print(f"Mean reward in final {final_window} episodes: {mean_final_reward:.2f}")
    print(f"Episodes to first success (>{success_threshold}): {episodes_to_threshold}")
    print(f"Loss variance: {loss_variance:.6f}")
    print(f"Reward standard deviation: {reward_std:.2f}")
    print(f"AUC: {auc_reward:.2f}")
    print("===================")

    # Plot reward and loss curves
    def smooth(data, w=0.9):
        sm, last = [], data[0]
        for x in data:
            last = last * w + (1 - w) * x
            sm.append(last)
        return sm

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Raw Reward", alpha=0.4)
    plt.plot(smooth(rewards), label="Smoothed Reward")
    plt.title("Episode Rewards Over Time")
    plt.xlabel("Episode"); plt.ylabel("Total Reward"); plt.legend(); plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(losses, label="Mini-Batch Loss")
    plt.title("Mini-Batch Loss During Training")
    plt.xlabel("Training Step"); plt.ylabel("Loss"); plt.legend(); plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("DISRC_MiniGrid_DoorKey_results.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Plot saved as 'DISRC_MiniGrid_DoorKey_results.png'")

# Entrypoint
if __name__ == "__main__":
    main()
