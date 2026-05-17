# ================================================================
# RUDDER-LSTM DQN on MiniGrid-DoorKey-8x8-v0
# Implements a Deep Q-Network with an LSTM return predictor for
# RUDDER-style reward redistribution.
# ================================================================

import gymnasium as gym
import minigrid
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import os
import argparse
import pandas as pd

# ================================================================
# Device Selection
# Supporting MPS for Mac, CUDA for Nvidia, or CPU
# ================================================================
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"==========================================")
print(f"📌 Using Device: {DEVICE}")
print(f"==========================================")

# ================================================================
# Preprocessing for MiniGrid
# Converts MiniGrid's image observations into flat, normalized vectors
# ================================================================
def preprocess_obs(obs):
    img = obs["image"]  # Shape: (7, 7, 3)
    flat = img.flatten().astype(np.float32) / 255.0
    return flat  # Shape: (147,)

# ================================================================
# Utility Functions
# ================================================================
def ensure_2d_tensor(tensor):
    if isinstance(tensor, np.ndarray):
        tensor = torch.FloatTensor(tensor).to(DEVICE)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

def compute_discounted_return(rewards, gamma=0.99):
    total = 0.0
    discount = 1.0
    for reward in rewards:
        total += discount * float(reward)
        discount *= gamma
    return total

def one_hot_action(action, action_dim):
    action_vec = torch.zeros(action_dim, dtype=torch.float32, device=DEVICE)
    action_vec[int(action)] = 1.0
    return action_vec

# ================================================================
# Neural Network Architectures
# ================================================================
class RUDDERStateEncoder(nn.Module):
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

class RUDDER_DQN(nn.Module):
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

class LSTMReturnPredictor(nn.Module):
    """
    Predicts the final episode return at every prefix of a trajectory.
    Input at each time step is [encoded_state, one_hot_action].
    Output is predicted final return g_t for each prefix t.
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        pred_returns = self.fc(out).squeeze(-1)
        return pred_returns

# ================================================================
# Action Selection and Target Update
# ================================================================
def epsilon_greedy(model, state, epsilon, action_space):
    if random.random() < epsilon:
        return random.randrange(action_space)
    with torch.no_grad():
        q_values = model(state)
        return int(torch.argmax(q_values).item())

def soft_update(target, source, tau=0.005):
    for tp, p in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

# ================================================================
# RUDDER Reward Redistribution
# ================================================================
def build_return_predictor_sequence(encoder, states, actions, action_dim):
    seq_items = []
    with torch.no_grad():
        for state_tensor, action in zip(states, actions):
            encoded_state = encoder(state_tensor.to(DEVICE)).squeeze(0)
            action_vec = one_hot_action(action, action_dim)
            seq_items.append(torch.cat([encoded_state, action_vec], dim=0))
    return torch.stack(seq_items, dim=0).unsqueeze(0).to(DEVICE)

def train_return_predictor(return_predictor, return_optimizer, encoder,
                           trajectory, action_dim, gamma=0.99):
    states = [item[0] for item in trajectory]
    actions = [item[1] for item in trajectory]
    rewards = [item[2] for item in trajectory]

    if len(states) == 0:
        return 0.0

    episode_return = compute_discounted_return(rewards, gamma=gamma)
    x_seq = build_return_predictor_sequence(encoder, states, actions, action_dim)
    target = torch.full(
        (1, x_seq.size(1)),
        float(episode_return),
        dtype=torch.float32,
        device=DEVICE
    )

    pred = return_predictor(x_seq)
    loss = F.mse_loss(pred, target)

    return_optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(return_predictor.parameters(), max_norm=1.0)
    return_optimizer.step()

    return float(loss.item())

def compute_redistributed_rewards(return_predictor, encoder, trajectory,
                                  action_dim, warmup=False,
                                  clip_min=-1.0, clip_max=1.0):
    original_rewards = [float(item[2]) for item in trajectory]

    if warmup or len(trajectory) == 0:
        return original_rewards

    states = [item[0] for item in trajectory]
    actions = [item[1] for item in trajectory]
    x_seq = build_return_predictor_sequence(encoder, states, actions, action_dim)

    with torch.no_grad():
        pred_returns = return_predictor(x_seq).squeeze(0)
        redistributed = torch.zeros_like(pred_returns)
        redistributed[0] = pred_returns[0]
        if pred_returns.numel() > 1:
            redistributed[1:] = pred_returns[1:] - pred_returns[:-1]
        redistributed = torch.clamp(redistributed, clip_min, clip_max)

    return redistributed.detach().cpu().numpy().astype(np.float32).tolist()

# ================================================================
# Training Loop
# ================================================================
def train_dqn(env, model, target_model, encoder, return_predictor,
              encoder_optimizer, model_optimizer, return_optimizer,
              num_episodes=1000, gamma=0.99,
              epsilon_start=1.0, epsilon_min=0.05,
              batch_size=128, replay_start_size=5000,
              rudder_warmup_episodes=100):
    replay_buffer = deque(maxlen=50000)
    all_rewards, dqn_losses, return_losses = [], [], []
    epsilon = epsilon_start
    current_dir = os.path.dirname(os.path.abspath(__file__))

    for episode in range(num_episodes):
        obs, _ = env.reset()
        state_arr = preprocess_obs(obs)
        state_tensor = torch.FloatTensor(state_arr).unsqueeze(0).to(DEVICE)
        done, total_reward = False, 0.0
        episode_trajectory = []

        while not done:
            encoded_state = encoder(state_tensor)
            action = epsilon_greedy(model, encoded_state, epsilon, env.action_space.n)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)

            next_state_arr = preprocess_obs(next_obs)
            next_tensor = torch.FloatTensor(next_state_arr).unsqueeze(0).to(DEVICE)

            episode_trajectory.append((
                state_tensor.detach(),
                int(action),
                float(reward),
                next_tensor.detach(),
                float(done)
            ))

            state_tensor = next_tensor

            if len(replay_buffer) >= replay_start_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards_b, next_states, dones = zip(*batch)
                states = torch.cat([ensure_2d_tensor(s) for s in states]).to(DEVICE)
                next_states = torch.cat([ensure_2d_tensor(ns) for ns in next_states]).to(DEVICE)
                actions = torch.LongTensor(actions).to(DEVICE)
                rewards_b = torch.FloatTensor(rewards_b).to(DEVICE)
                dones = torch.FloatTensor(dones).to(DEVICE)

                s_enc = encoder(states)
                ns_enc = encoder(next_states)

                with torch.no_grad():
                    next_q = model(ns_enc)
                    next_act = torch.argmax(next_q, dim=1)
                    next_q_target = target_model(ns_enc)
                    target_vals = next_q_target.gather(1, next_act.view(-1, 1)).squeeze(1)
                    targets = rewards_b + gamma * target_vals * (1 - dones)

                q_vals = model(s_enc).gather(1, actions.view(-1, 1)).squeeze(1)
                dqn_loss = (q_vals - targets).pow(2).mean()

                model_optimizer.zero_grad()
                encoder_optimizer.zero_grad()
                dqn_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=0.3)
                model_optimizer.step()
                encoder_optimizer.step()
                dqn_losses.append(float(dqn_loss.item()))

                soft_update(target_model, model, tau=0.005)

        return_loss = train_return_predictor(
            return_predictor=return_predictor,
            return_optimizer=return_optimizer,
            encoder=encoder,
            trajectory=episode_trajectory,
            action_dim=env.action_space.n,
            gamma=gamma
        )
        return_losses.append(return_loss)

        use_warmup = episode < rudder_warmup_episodes
        redistributed_rewards = compute_redistributed_rewards(
            return_predictor=return_predictor,
            encoder=encoder,
            trajectory=episode_trajectory,
            action_dim=env.action_space.n,
            warmup=use_warmup,
            clip_min=-1.0,
            clip_max=1.0
        )

        for transition, redistributed_reward in zip(episode_trajectory, redistributed_rewards):
            state, action, original_reward, next_state, done_flag = transition
            replay_buffer.append((state, action, float(redistributed_reward), next_state, done_flag))

        all_rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon_start - (epsilon_start - epsilon_min) * (episode / num_episodes))

        if episode % 10 == 0:
            mean_reward = np.mean(all_rewards[-50:]) if len(all_rewards) >= 50 else np.mean(all_rewards)
            mean_dqn_loss = np.mean(dqn_losses[-100:]) if dqn_losses else 0.0
            mean_return_loss = np.mean(return_losses[-50:]) if return_losses else 0.0
            mode = "ENV" if use_warmup else "RUDDER"
            print(f"[Ep {episode:03d}] Reward: {total_reward:.2f} | Mean(50): {mean_reward:.2f} | "
                  f"Eps: {epsilon:.3f} | DQN Loss: {mean_dqn_loss:.4f} | "
                  f"Return Loss: {mean_return_loss:.4f} | RewardMode: {mode}")

        if (episode + 1) % 50 == 0:
            df = pd.DataFrame({"episode": range(1, len(all_rewards)+1), "score": all_rewards})
            df.to_excel(os.path.join(current_dir, "episode_rewards_rudder_lstm.xlsx"), index=False)

            torch.save(model.state_dict(), os.path.join(current_dir, "rudder_lstm_model.pth"))
            torch.save(encoder.state_dict(), os.path.join(current_dir, "rudder_lstm_encoder.pth"))
            torch.save(return_predictor.state_dict(), os.path.join(current_dir, "rudder_lstm_return_predictor.pth"))

            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(all_rewards, alpha=0.3, color="blue", label="Raw Environment Reward")
            if len(all_rewards) >= 10:
                smooth_scores = pd.Series(all_rewards).rolling(window=10).mean()
                plt.plot(smooth_scores, color="orange", linewidth=2, label="Smoothed Reward (MA 10)")
            plt.title("RUDDER-LSTM Training Scores")
            plt.xlabel("Episode")
            plt.ylabel("Score")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(dqn_losses, label="DQN Mini-Batch Loss")
            if return_losses:
                plt.plot(return_losses, label="LSTM Return Predictor Loss", alpha=0.7)
            plt.title("Losses During Training")
            plt.xlabel("Training Step / Episode")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(current_dir, "scores_rudder_lstm.png"))
            plt.close()

    df = pd.DataFrame({"episode": range(1, len(all_rewards) + 1), "score": all_rewards})
    df.to_excel(os.path.join(current_dir, "episode_rewards_rudder_lstm.xlsx"), index=False)

    return all_rewards, dqn_losses, return_losses

# ================================================================
# Main Execution
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--epsilon_min", type=float, default=0.05)
    args = parser.parse_args()

    env = gym.make("MiniGrid-DoorKey-8x8-v0")
    input_dim = 7 * 7 * 3
    action_dim = env.action_space.n
    encoded_dim = 64

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "rudder_lstm_model.pth")
    encoder_path = os.path.join(current_dir, "rudder_lstm_encoder.pth")
    return_predictor_path = os.path.join(current_dir, "rudder_lstm_return_predictor.pth")

    encoder = RUDDERStateEncoder(input_dim=input_dim, encoded_dim=encoded_dim).to(DEVICE)
    model = RUDDER_DQN(input_size=encoded_dim, action_size=action_dim).to(DEVICE)
    target_model = RUDDER_DQN(input_size=encoded_dim, action_size=action_dim).to(DEVICE)
    return_predictor = LSTMReturnPredictor(
        input_dim=encoded_dim + action_dim,
        hidden_dim=128,
        num_layers=1
    ).to(DEVICE)

    encoder.apply(init_weights)
    model.apply(init_weights)
    target_model.load_state_dict(model.state_dict())
    return_predictor.apply(init_weights)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=1e-4)
    model_optimizer = optim.Adam(model.parameters(), lr=1e-4)
    return_optimizer = optim.Adam(return_predictor.parameters(), lr=1e-4)

    rewards, dqn_losses, return_losses = train_dqn(
        env=env,
        model=model,
        target_model=target_model,
        encoder=encoder,
        return_predictor=return_predictor,
        encoder_optimizer=encoder_optimizer,
        model_optimizer=model_optimizer,
        return_optimizer=return_optimizer,
        num_episodes=args.episodes,
        epsilon_min=args.epsilon_min,
        rudder_warmup_episodes=args.warmup
    )

    final_window = 50
    mean_final_reward = np.mean(rewards[-final_window:])
    success_threshold = 0.8
    episodes_to_threshold = next((i+1 for i, r in enumerate(rewards) if r >= success_threshold), len(rewards))
    dqn_loss_variance = float(np.var(dqn_losses)) if dqn_losses else 0.0
    return_loss_variance = float(np.var(return_losses)) if return_losses else 0.0
    reward_std = float(np.std(rewards))
    auc_reward = float(np.trapezoid(rewards, dx=1))

    print("===== METRICS =====")
    print(f"Mean reward in final {final_window} episodes: {mean_final_reward:.2f}")
    print(f"Episodes to first success (>{success_threshold}): {episodes_to_threshold}")
    print(f"DQN loss variance: {dqn_loss_variance:.6f}")
    print(f"Return predictor loss variance: {return_loss_variance:.6f}")
    print(f"Reward standard deviation: {reward_std:.2f}")
    print(f"AUC: {auc_reward:.2f}")
    print("===================")

    torch.save(model.state_dict(), model_path)
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(return_predictor.state_dict(), return_predictor_path)

    df = pd.DataFrame({"episode": range(1, len(rewards) + 1), "score": rewards})
    excel_path = os.path.join(current_dir, "episode_rewards_rudder_lstm.xlsx")
    df.to_excel(excel_path, index=False)

    print(f"Training finished.")
    print(f"DQN model saved to {model_path}")
    print(f"Encoder saved to {encoder_path}")
    print(f"Return predictor saved to {return_predictor_path}")
    print(f"Episode rewards saved to {excel_path}")

if __name__ == "__main__":
    main()