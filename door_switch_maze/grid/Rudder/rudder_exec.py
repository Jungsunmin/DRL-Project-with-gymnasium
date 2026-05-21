# ================================================================
# RUDDER DQN on MiniGrid-DoorKey-8x8-v0
# Closer-to-paper RUDDER implementation:
#   (I)   safe exploration,
#   (II)  lessons replay buffer for LSTM return predictor training,
#   (III) LSTM return prediction + contribution analysis based reward redistribution.
#
# Important note:
# This implementation uses a custom LSTM cell return predictor and
# gate-level epsilon-LRP-style contribution analysis. It stores the
# i/f/g/o/c/h gate flow during the recurrent forward pass, propagates
# final return relevance through the output layer and LSTM gates, and then
# enforces return-equivalence by correcting redistributed rewards to sum to
# the original discounted episode return.
# ================================================================

import argparse
import os
import random
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import minigrid  # noqa: F401  # ensures MiniGrid env registration
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ================================================================
# Device Selection
# ================================================================
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
print("==========================================")
print(f"📌 Using Device: {DEVICE}")
print("==========================================")

# ================================================================
# Reproducibility
# ================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ================================================================
# Preprocessing for MiniGrid
# ================================================================
def preprocess_obs(obs):
    img = obs["image"]  # Shape: (7, 7, 3), partial observation by default
    flat = img.flatten().astype(np.float32) / 255.0
    return flat

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
        if m.bias is not None:
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


def soft_update(target, source, tau=0.005):
    for tp, p in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

# ================================================================
# Data Structures
# ================================================================
@dataclass
class EpisodeLesson:
    trajectory: list
    episode_return: float
    prediction_error: float


class LessonsReplayBuffer:
    """
    RUDDER paper idea:
    Episodes with unexpected delayed rewards are stored in a lessons replay buffer.
    Sampling is prioritized by LSTM prediction error.
    """
    def __init__(self, capacity=2000, alpha=0.6, eps=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def add(self, trajectory, episode_return, prediction_error):
        lesson = EpisodeLesson(
            trajectory=trajectory,
            episode_return=float(episode_return),
            prediction_error=float(abs(prediction_error))
        )
        if len(self.buffer) >= self.capacity:
            min_idx = int(np.argmin([item.prediction_error for item in self.buffer]))
            self.buffer[min_idx] = lesson
        else:
            self.buffer.append(lesson)

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return []
        priorities = np.array(
            [(item.prediction_error + self.eps) ** self.alpha for item in self.buffer],
            dtype=np.float64
        )
        probs = priorities / priorities.sum()
        indices = np.random.choice(
            len(self.buffer),
            size=min(batch_size, len(self.buffer)),
            replace=len(self.buffer) < batch_size,
            p=probs
        )
        return [self.buffer[int(i)] for i in indices]

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
    Custom single-layer LSTM return predictor.

    This replaces torch.nn.LSTM so the i/f/g/o/c/h gate flow is explicitly
    available for gate-level LRP-style relevance propagation.

    Input at each time step is [encoded_state, one_hot_action].
    Output is predicted final return g_t for each prefix t.
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=1):
        super().__init__()
        if num_layers != 1:
            raise ValueError("This gate-level LRP implementation supports num_layers=1 only.")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.x2gates = nn.Linear(input_dim, 4 * hidden_dim)
        self.h2gates = nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, return_cache=False):
        batch_size, seq_len, _ = x.shape
        h_t = torch.zeros(batch_size, self.hidden_dim, dtype=x.dtype, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, dtype=x.dtype, device=x.device)

        outputs = []
        cache = {
            "x": [],
            "h_prev": [],
            "c_prev": [],
            "i": [],
            "f": [],
            "g": [],
            "o": [],
            "c": [],
            "h": []
        }

        for t in range(seq_len):
            x_t = x[:, t, :]
            h_prev = h_t
            c_prev = c_t

            gates = self.x2gates(x_t) + self.h2gates(h_prev)
            i_t, f_t, g_t, o_t = gates.chunk(4, dim=1)
            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)
            g_t = torch.tanh(g_t)
            o_t = torch.sigmoid(o_t)

            c_t = f_t * c_prev + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            outputs.append(h_t.unsqueeze(1))

            if return_cache:
                cache["x"].append(x_t)
                cache["h_prev"].append(h_prev)
                cache["c_prev"].append(c_prev)
                cache["i"].append(i_t)
                cache["f"].append(f_t)
                cache["g"].append(g_t)
                cache["o"].append(o_t)
                cache["c"].append(c_t)
                cache["h"].append(h_t)

        out = torch.cat(outputs, dim=1)
        pred_returns = self.fc(out).squeeze(-1)

        if return_cache:
            for key in cache:
                cache[key] = torch.stack(cache[key], dim=1)
            return pred_returns, cache
        return pred_returns

# ================================================================
# Action Selection: Safe Exploration
# ================================================================
def safe_epsilon_greedy(model, state, epsilon, action_space, safe_margin=0.10):
    """
    RUDDER paper idea:
    safe exploration avoids actions associated with clearly low Q-values.
    During epsilon exploration, this samples only among actions whose Q-value is
    within safe_margin of the best Q-value. If no safe candidate exists, it
    falls back to all actions.
    """
    with torch.no_grad():
        q_values = model(state).squeeze(0)

    if random.random() >= epsilon:
        return int(torch.argmax(q_values).item())

    max_q = torch.max(q_values)
    safe_mask = q_values >= (max_q - safe_margin)
    safe_actions = torch.nonzero(safe_mask, as_tuple=False).flatten().detach().cpu().numpy().tolist()

    if len(safe_actions) == 0:
        return random.randrange(action_space)
    return int(random.choice(safe_actions))

# ================================================================
# RUDDER Sequence Construction
# ================================================================
def build_return_predictor_sequence(encoder, states, actions, action_dim, detach_encoder=True):
    seq_items = []
    context = torch.no_grad() if detach_encoder else torch.enable_grad()
    with context:
        for state_tensor, action in zip(states, actions):
            encoded_state = encoder(state_tensor.to(DEVICE)).squeeze(0)
            action_vec = one_hot_action(action, action_dim)
            seq_items.append(torch.cat([encoded_state, action_vec], dim=0))
    return torch.stack(seq_items, dim=0).unsqueeze(0).to(DEVICE)


def get_episode_return_prediction(return_predictor, encoder, trajectory, action_dim):
    if len(trajectory) == 0:
        return 0.0
    states = [item[0] for item in trajectory]
    actions = [item[1] for item in trajectory]
    x_seq = build_return_predictor_sequence(encoder, states, actions, action_dim, detach_encoder=True)
    with torch.no_grad():
        pred = return_predictor(x_seq)
        return float(pred[0, -1].item())

# ================================================================
# RUDDER Return Predictor Training with Lessons Replay Buffer
# ================================================================
def train_return_predictor_on_lessons(return_predictor, return_optimizer, encoder,
                                      lessons_buffer, action_dim,
                                      batch_size=8, gamma=0.99):
    if len(lessons_buffer) == 0:
        return 0.0

    lessons = lessons_buffer.sample(batch_size)
    losses = []

    for lesson in lessons:
        trajectory = lesson.trajectory
        states = [item[0] for item in trajectory]
        actions = [item[1] for item in trajectory]
        rewards = [item[2] for item in trajectory]

        if len(states) == 0:
            continue

        episode_return = compute_discounted_return(rewards, gamma=gamma)
        x_seq = build_return_predictor_sequence(encoder, states, actions, action_dim, detach_encoder=True)

        # RUDDER trains the LSTM to predict the sequence-wide return.
        # Each prefix is supervised by the same final return.
        target = torch.full(
            (1, x_seq.size(1)),
            float(episode_return),
            dtype=torch.float32,
            device=DEVICE
        )
        pred = return_predictor(x_seq)
        losses.append(F.mse_loss(pred, target))

    if len(losses) == 0:
        return 0.0

    loss = torch.stack(losses).mean()
    return_optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(return_predictor.parameters(), max_norm=1.0)
    return_optimizer.step()

    return float(loss.item())

# ================================================================
# Contribution Analysis: full gate-level epsilon-LRP over custom LSTM
# ================================================================
def _signed_stabilizer(z, eps):
    return z + eps * torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))


def _lrp_linear_to_inputs(relevance_out, inputs, weight, eps=1e-6):
    """
    epsilon-LRP for y_j = sum_i x_i * w_ji.
    Returns relevance for each input vector in `inputs`.
    Bias is intentionally ignored in redistribution to preserve relevance on
    actual input sources.
    """
    z_parts = [torch.matmul(inp, weight.t()) for inp in inputs]
    z_total = torch.stack(z_parts, dim=0).sum(dim=0)
    z_total = _signed_stabilizer(z_total, eps)
    message = relevance_out / z_total
    relevances = [inp * torch.matmul(message, weight) for inp in inputs]
    return relevances


def _split_multiplicative_relevance(left, right, relevance, left_share=0.5):
    """
    Conservative practical rule for a product y = left * right.
    LSTM gates are control variables, so relevance is split between gate and
    content paths instead of assigning everything to only one operand.
    """
    left_strength = torch.abs(left)
    right_strength = torch.abs(right)
    denom = left_strength + right_strength + 1e-12
    adaptive_left = left_strength / denom
    left_ratio = 0.5 * adaptive_left + 0.5 * left_share
    right_ratio = 1.0 - left_ratio
    return relevance * left_ratio, relevance * right_ratio


def epsilon_lrp_lstm_contributions(return_predictor, x_seq, eps=1e-6):
    """
    Full gate-level epsilon-LRP-style contribution analysis for the custom LSTM.

    Relevance path:
        final return prediction
        -> final FC layer
        -> h_t at every prefix output
        -> h_t = o_t * tanh(c_t)
        -> c_t = f_t * c_{t-1} + i_t * g_t
        -> gates i/f/g/o
        -> x_t and h_{t-1}

    The returned vector has one scalar relevance per trajectory time step.
    """
    with torch.no_grad():
        pred_returns, cache = return_predictor(x_seq, return_cache=True)
        seq_len = x_seq.size(1)
        hidden_dim = return_predictor.hidden_dim

        # Output-layer LRP: distribute all final prediction relevance across
        # all prefix hidden outputs h_t that were scored by the shared FC head.
        hidden = cache["h"].squeeze(0)  # (T, H)
        fc_weight = return_predictor.fc.weight.squeeze(0)  # (H,)
        final_relevance = pred_returns[0, -1]

        z_t = (hidden * fc_weight.unsqueeze(0)).sum(dim=1)
        z_total = _signed_stabilizer(z_t.sum(), eps)
        r_h_direct = (z_t / z_total * final_relevance).unsqueeze(1) * (
            (hidden * fc_weight.unsqueeze(0)) / _signed_stabilizer(z_t.unsqueeze(1), eps)
        )
        r_h_direct = torch.nan_to_num(r_h_direct, nan=0.0, posinf=0.0, neginf=0.0)

        r_h = torch.zeros(seq_len, hidden_dim, dtype=x_seq.dtype, device=x_seq.device)
        r_c = torch.zeros(seq_len, hidden_dim, dtype=x_seq.dtype, device=x_seq.device)
        r_x = torch.zeros(seq_len, x_seq.size(-1), dtype=x_seq.dtype, device=x_seq.device)

        r_h += r_h_direct

        x_weight = return_predictor.x2gates.weight  # (4H, input_dim)
        h_weight = return_predictor.h2gates.weight  # (4H, H)

        # Backward through time.
        for t in reversed(range(seq_len)):
            x_t = cache["x"][:, t, :]
            h_prev = cache["h_prev"][:, t, :]
            c_prev = cache["c_prev"][:, t, :]
            i_t = cache["i"][:, t, :]
            f_t = cache["f"][:, t, :]
            g_t = cache["g"][:, t, :]
            o_t = cache["o"][:, t, :]
            c_t = cache["c"][:, t, :]

            # h_t = o_t * tanh(c_t)
            tanh_c = torch.tanh(c_t)
            r_o, r_tanh_c = _split_multiplicative_relevance(o_t, tanh_c, r_h[t].unsqueeze(0), left_share=0.35)
            r_c[t] += r_tanh_c.squeeze(0)

            # c_t = f_t * c_{t-1} + i_t * g_t
            forget_part = f_t * c_prev
            input_part = i_t * g_t
            c_parts = torch.stack([forget_part, input_part], dim=0)
            c_denom = _signed_stabilizer(c_parts.sum(dim=0), eps)
            c_message = r_c[t].unsqueeze(0) / c_denom
            r_forget_part = forget_part * c_message
            r_input_part = input_part * c_message

            r_f, r_c_prev = _split_multiplicative_relevance(f_t, c_prev, r_forget_part, left_share=0.35)
            r_i, r_g = _split_multiplicative_relevance(i_t, g_t, r_input_part, left_share=0.35)

            if t > 0:
                r_c[t - 1] += r_c_prev.squeeze(0)

            # Relevance at gate activations. Map it approximately to preactivation
            # gates and propagate through x2gates + h2gates using epsilon-LRP.
            r_gates = torch.cat([r_i, r_f, r_g, r_o], dim=1)
            rx_part, rh_prev_part = _lrp_linear_to_inputs(
                relevance_out=r_gates,
                inputs=[x_t, h_prev],
                weight=torch.cat([x_weight, h_weight], dim=1),
                eps=eps
            )

            r_x[t] += rx_part.squeeze(0)
            if t > 0:
                r_h[t - 1] += rh_prev_part.squeeze(0)

        step_relevance = r_x.sum(dim=1)
        step_relevance = torch.nan_to_num(step_relevance, nan=0.0, posinf=0.0, neginf=0.0)

    return step_relevance


def compute_rudder_redistributed_rewards(return_predictor, encoder, trajectory,
                                         action_dim, gamma=0.99,
                                         warmup=False,
                                         lrp_eps=1e-6,
                                         clip_min=-1.0, clip_max=1.0):
    original_rewards = [float(item[2]) for item in trajectory]

    if warmup or len(trajectory) == 0:
        return original_rewards

    states = [item[0] for item in trajectory]
    actions = [item[1] for item in trajectory]
    rewards = [item[2] for item in trajectory]
    episode_return = compute_discounted_return(rewards, gamma=gamma)

    x_seq = build_return_predictor_sequence(encoder, states, actions, action_dim, detach_encoder=True)

    # LRP contribution analysis over the final return prediction.
    step_contrib = epsilon_lrp_lstm_contributions(
        return_predictor=return_predictor,
        x_seq=x_seq,
        eps=lrp_eps
    )

    redistributed = step_contrib.detach().clone()

    # Return-equivalence correction:
    # Sum of redistributed rewards should equal the original episode return.
    residual = float(episode_return) - float(redistributed.sum().item())
    redistributed[-1] += residual

    redistributed = torch.clamp(redistributed, clip_min, clip_max)

    # Clipping can break return-equivalence, so apply a second small correction.
    residual_after_clip = float(episode_return) - float(redistributed.sum().item())
    redistributed[-1] += residual_after_clip

    return redistributed.detach().cpu().numpy().astype(np.float32).tolist()

# ================================================================
# Training Loop
# ================================================================
def train_dqn(env, model, target_model, encoder, target_encoder, return_predictor,
              encoder_optimizer, model_optimizer, return_optimizer,
              num_episodes=1000, env_gamma=0.99, dqn_gamma=0.0,
              epsilon_start=1.0, epsilon_min=0.05,
              batch_size=128, replay_start_size=5000,
              rudder_warmup_episodes=100,
              lessons_capacity=2000,
              lessons_batch_size=8,
              return_train_updates=4,
              lrp_eps=1e-6,
              safe_margin=0.10):
    replay_buffer = deque(maxlen=50000)
    lessons_buffer = LessonsReplayBuffer(capacity=lessons_capacity)
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
            with torch.no_grad():
                encoded_state = encoder(state_tensor)
            action = safe_epsilon_greedy(
                model=model,
                state=encoded_state,
                epsilon=epsilon,
                action_space=env.action_space.n,
                safe_margin=safe_margin
            )

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
                with torch.no_grad():
                    ns_enc_online = encoder(next_states)
                    ns_enc_target = target_encoder(next_states)
                    next_q_online = model(ns_enc_online)
                    next_actions = torch.argmax(next_q_online, dim=1)
                    next_q_target = target_model(ns_enc_target)
                    next_values = next_q_target.gather(1, next_actions.view(-1, 1)).squeeze(1)
                    targets = rewards_b + dqn_gamma * next_values * (1 - dones)

                q_vals = model(s_enc).gather(1, actions.view(-1, 1)).squeeze(1)
                dqn_loss = F.mse_loss(q_vals, targets)

                model_optimizer.zero_grad()
                encoder_optimizer.zero_grad()
                dqn_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=0.3)
                model_optimizer.step()
                encoder_optimizer.step()
                dqn_losses.append(float(dqn_loss.item()))

                soft_update(target_model, model, tau=0.005)
                soft_update(target_encoder, encoder, tau=0.005)

        episode_return = compute_discounted_return(
            [item[2] for item in episode_trajectory],
            gamma=env_gamma
        )
        pred_before = get_episode_return_prediction(
            return_predictor=return_predictor,
            encoder=encoder,
            trajectory=episode_trajectory,
            action_dim=env.action_space.n
        )
        prediction_error = episode_return - pred_before

        # RUDDER lessons replay buffer: store episodes unexpected for the LSTM.
        lessons_buffer.add(
            trajectory=episode_trajectory,
            episode_return=episode_return,
            prediction_error=prediction_error
        )

        # Train LSTM on prioritized lessons, not only on the latest trajectory.
        update_losses = []
        for _ in range(return_train_updates):
            loss_val = train_return_predictor_on_lessons(
                return_predictor=return_predictor,
                return_optimizer=return_optimizer,
                encoder=encoder,
                lessons_buffer=lessons_buffer,
                action_dim=env.action_space.n,
                batch_size=lessons_batch_size,
                gamma=env_gamma
            )
            update_losses.append(loss_val)
        return_loss = float(np.mean(update_losses)) if update_losses else 0.0
        return_losses.append(return_loss)

        use_warmup = episode < rudder_warmup_episodes
        redistributed_rewards = compute_rudder_redistributed_rewards(
            return_predictor=return_predictor,
            encoder=encoder,
            trajectory=episode_trajectory,
            action_dim=env.action_space.n,
            gamma=env_gamma,
            warmup=use_warmup,
            lrp_eps=lrp_eps,
            clip_min=-1.0,
            clip_max=1.0
        )

        for transition, redistributed_reward in zip(episode_trajectory, redistributed_rewards):
            state, action, original_reward, next_state, done_flag = transition
            replay_buffer.append((state, action, float(redistributed_reward), next_state, done_flag))

        all_rewards.append(total_reward)
        epsilon = max(
            epsilon_min,
            epsilon_start - (epsilon_start - epsilon_min) * (episode / max(1, num_episodes))
        )

        if episode % 10 == 0:
            mean_reward = np.mean(all_rewards[-50:]) if len(all_rewards) >= 50 else np.mean(all_rewards)
            mean_dqn_loss = np.mean(dqn_losses[-100:]) if dqn_losses else 0.0
            mean_return_loss = np.mean(return_losses[-50:]) if return_losses else 0.0
            mode = "ENV" if use_warmup else "RUDDER-LRP"
            print(
                f"[Ep {episode:03d}] Reward: {total_reward:.2f} | Mean(50): {mean_reward:.2f} | "
                f"Eps: {epsilon:.3f} | DQN Loss: {mean_dqn_loss:.4f} | "
                f"Return Loss: {mean_return_loss:.4f} | Lessons: {len(lessons_buffer)} | "
                f"PredErr: {prediction_error:.4f} | RewardMode: {mode}"
            )

        if (episode + 1) % 50 == 0:
            df = pd.DataFrame({"episode": range(1, len(all_rewards) + 1), "score": all_rewards})
            df.to_excel(os.path.join(current_dir, "episode_rewards_rudder_lstm.xlsx"), index=False)

            torch.save(model.state_dict(), os.path.join(current_dir, "rudder_lstm_model.pth"))
            torch.save(encoder.state_dict(), os.path.join(current_dir, "rudder_lstm_encoder.pth"))
            torch.save(target_encoder.state_dict(), os.path.join(current_dir, "rudder_lstm_target_encoder.pth"))
            torch.save(return_predictor.state_dict(), os.path.join(current_dir, "rudder_lstm_return_predictor.pth"))

            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(all_rewards, alpha=0.3, label="Raw Environment Reward")
            if len(all_rewards) >= 10:
                smooth_scores = pd.Series(all_rewards).rolling(window=10).mean()
                plt.plot(smooth_scores, linewidth=2, label="Smoothed Reward (MA 10)")
            plt.title("RUDDER Training Scores")
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
    parser.add_argument("--env_gamma", type=float, default=0.99)
    parser.add_argument("--dqn_gamma", type=float, default=0.0,
                        help="RUDDER direct Q estimation commonly uses near-zero future reward. Use 0.99 for ordinary TD-style DQN.")
    parser.add_argument("--lessons_capacity", type=int, default=2000)
    parser.add_argument("--lessons_batch_size", type=int, default=8)
    parser.add_argument("--return_train_updates", type=int, default=4)
    parser.add_argument("--lrp_eps", type=float, default=1e-6)
    parser.add_argument("--safe_margin", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    env = gym.make("MiniGrid-DoorKey-8x8-v0")
    input_dim = 7 * 7 * 3
    action_dim = env.action_space.n
    encoded_dim = 64

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "rudder_lstm_model.pth")
    encoder_path = os.path.join(current_dir, "rudder_lstm_encoder.pth")
    target_encoder_path = os.path.join(current_dir, "rudder_lstm_target_encoder.pth")
    return_predictor_path = os.path.join(current_dir, "rudder_lstm_return_predictor.pth")

    encoder = RUDDERStateEncoder(input_dim=input_dim, encoded_dim=encoded_dim).to(DEVICE)
    target_encoder = RUDDERStateEncoder(input_dim=input_dim, encoded_dim=encoded_dim).to(DEVICE)
    model = RUDDER_DQN(input_size=encoded_dim, action_size=action_dim).to(DEVICE)
    target_model = RUDDER_DQN(input_size=encoded_dim, action_size=action_dim).to(DEVICE)
    return_predictor = LSTMReturnPredictor(
        input_dim=encoded_dim + action_dim,
        hidden_dim=128,
        num_layers=1
    ).to(DEVICE)

    encoder.apply(init_weights)
    target_encoder.load_state_dict(encoder.state_dict())
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
        target_encoder=target_encoder,
        return_predictor=return_predictor,
        encoder_optimizer=encoder_optimizer,
        model_optimizer=model_optimizer,
        return_optimizer=return_optimizer,
        num_episodes=args.episodes,
        env_gamma=args.env_gamma,
        dqn_gamma=args.dqn_gamma,
        epsilon_min=args.epsilon_min,
        rudder_warmup_episodes=args.warmup,
        lessons_capacity=args.lessons_capacity,
        lessons_batch_size=args.lessons_batch_size,
        return_train_updates=args.return_train_updates,
        lrp_eps=args.lrp_eps,
        safe_margin=args.safe_margin
    )

    final_window = 50
    mean_final_reward = np.mean(rewards[-final_window:]) if rewards else 0.0
    success_threshold = 0.8
    episodes_to_threshold = next((i + 1 for i, r in enumerate(rewards) if r >= success_threshold), len(rewards))
    dqn_loss_variance = float(np.var(dqn_losses)) if dqn_losses else 0.0
    return_loss_variance = float(np.var(return_losses)) if return_losses else 0.0
    reward_std = float(np.std(rewards)) if rewards else 0.0
    auc_reward = float(np.trapezoid(rewards, dx=1)) if len(rewards) > 1 else 0.0

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
    torch.save(target_encoder.state_dict(), target_encoder_path)
    torch.save(return_predictor.state_dict(), return_predictor_path)

    df = pd.DataFrame({"episode": range(1, len(rewards) + 1), "score": rewards})
    excel_path = os.path.join(current_dir, "episode_rewards_rudder_lstm.xlsx")
    df.to_excel(excel_path, index=False)

    print("Training finished.")
    print(f"DQN model saved to {model_path}")
    print(f"Encoder saved to {encoder_path}")
    print(f"Target encoder saved to {target_encoder_path}")
    print(f"Return predictor saved to {return_predictor_path}")
    print(f"Episode rewards saved to {excel_path}")


if __name__ == "__main__":
    main()